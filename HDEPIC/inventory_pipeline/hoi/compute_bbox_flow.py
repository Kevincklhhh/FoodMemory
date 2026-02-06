#!/usr/bin/env python3
"""
Compute per-frame optical flow within HOI object bounding boxes using FlowFormer.

Reads an event from flow_test.json, loads HOI detections (hands23) for the
corresponding video/time range, runs FlowFormer on consecutive frame pairs,
and computes flow statistics within each detected object bounding box.
"""

from __future__ import annotations

import json
import os
import sys
import numpy as np
import torch
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJ_ROOT = Path(__file__).resolve().parent.parent
AMEGO_DIR = PROJ_ROOT / "models" / "AMEGO"
FLOWFORMER_DIR = AMEGO_DIR / "submodules" / "flowformer"
FLOWFORMER_CORE = FLOWFORMER_DIR / "core"
FLOWFORMER_WEIGHTS = FLOWFORMER_DIR / "models" / "sintel.pth"

EVENT_PATH = PROJ_ROOT / "outputs" / "03_gating" / "flow_test.json"
HOI_RESULTS_PATH = PROJ_ROOT / "outputs" / "02_inventory" / "P03" / "hands23_detection" / "P03_hands23_results.json"
FRAMES_ROOT = PROJ_ROOT / "outputs" / "02_inventory" / "P03" / "hands23_detection"
OUTPUT_PATH = PROJ_ROOT / "outputs" / "03_gating" / "flow_test_bbox_flow.json"

POST_EVENT_BUFFER = 3.0  # seconds after end_timestamp to include

# ---------------------------------------------------------------------------
# FlowFormer imports (add paths before importing)
# ---------------------------------------------------------------------------
sys.path.insert(0, str(FLOWFORMER_CORE))
sys.path.insert(0, str(AMEGO_DIR))

from submodules.flowformer.configs.sintel import get_cfg
from submodules.flowformer.core.FlowFormer import build_flowformer
from submodules.flowformer.core.utils.utils import InputPadder


def load_event(path: Path) -> dict:
    """Load event definition and extract the first segment's info."""
    with open(path) as f:
        event = json.load(f)
    seg = event["segments"][0]
    return {
        "narration_id": event["narration_id"],
        "food_name": event["food_name"],
        "video_id": seg["video_id"],
        "start": seg["start_timestamp"],
        "end": seg["end_timestamp"],
    }


def load_hoi_frames(hoi_path: Path, video_id: str, t_start: float, t_end: float) -> list[dict]:
    """Load HOI detection frames for the given video within [t_start, t_end]."""
    with open(hoi_path) as f:
        hoi_data = json.load(f)

    # Find the matching video entry
    video_entry = None
    for v in hoi_data["videos"]:
        if v["video_id"] == video_id:
            video_entry = v
            break
    if video_entry is None:
        raise ValueError(f"Video {video_id} not found in HOI results")

    # Filter frames within the time range
    frames = []
    for frame in video_entry["frames"]:
        ts = frame["timestamp"]
        if t_start <= ts <= t_end:
            frames.append(frame)
    frames.sort(key=lambda f: f["timestamp"])
    return frames


def build_model() -> torch.nn.Module:
    """Build and load FlowFormer model."""
    cfg = get_cfg()
    model = torch.nn.DataParallel(build_flowformer(cfg))
    print(f"Loading FlowFormer weights from {FLOWFORMER_WEIGHTS}")
    model.load_state_dict(torch.load(str(FLOWFORMER_WEIGHTS), map_location="cpu"))
    model.cuda()
    model.eval()
    return model


def load_image(path: str, max_dim: int = 704) -> torch.Tensor:
    """Load image as [1, 3, H, W] float tensor, resized to fit within max_dim."""
    from PIL import Image
    img = Image.open(path).convert("RGB")
    # Resize to reduce GPU memory (1408 -> 704 is 2x downscale)
    w, h = img.size
    scale = min(max_dim / w, max_dim / h)
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.BILINEAR)
    arr = np.array(img).astype(np.uint8)
    return torch.from_numpy(arr).permute(2, 0, 1).float().unsqueeze(0), scale


def compute_bbox_flow_stats(flow: torch.Tensor, bbox: list[float]) -> dict:
    """
    Compute flow statistics within a bounding box.

    Args:
        flow: [2, H, W] optical flow tensor (dx, dy)
        bbox: [x1, y1, x2, y2] bounding box coordinates
    Returns:
        dict with flow statistics
    """
    _, H, W = flow.shape
    x1 = max(0, int(round(bbox[0])))
    y1 = max(0, int(round(bbox[1])))
    x2 = min(W, int(round(bbox[2])))
    y2 = min(H, int(round(bbox[3])))

    if x2 <= x1 or y2 <= y1:
        return None

    roi = flow[:, y1:y2, x1:x2]  # [2, h, w]
    dx = roi[0]  # [h, w]
    dy = roi[1]  # [h, w]
    magnitude = torch.sqrt(dx ** 2 + dy ** 2)

    return {
        "bbox_flow_mean_magnitude": float(magnitude.mean()),
        "bbox_flow_median_magnitude": float(magnitude.median()),
        "bbox_flow_max_magnitude": float(magnitude.max()),
        "bbox_flow_mean_dx": float(dx.mean()),
        "bbox_flow_mean_dy": float(dy.mean()),
        "bbox_num_pixels": int(magnitude.numel()),
    }


def main():
    # 1. Load event
    event_info = load_event(EVENT_PATH)
    analysis_end = event_info["end"] + POST_EVENT_BUFFER
    event_info["analysis_end"] = analysis_end
    print(f"Event: {event_info['food_name']} in {event_info['video_id']} "
          f"[{event_info['start']:.2f}s - {event_info['end']:.2f}s] "
          f"(analysis extends to {analysis_end:.2f}s)")

    # 2. Load HOI detections in time range (extended by buffer)
    hoi_frames = load_hoi_frames(HOI_RESULTS_PATH, event_info["video_id"],
                                 event_info["start"], analysis_end)
    print(f"Found {len(hoi_frames)} HOI frames in range")

    if len(hoi_frames) < 2:
        print("Need at least 2 frames to compute flow. Exiting.")
        return

    # 3. Build FlowFormer model
    model = build_model()

    # 4. Process consecutive frame pairs
    frame_pairs = []
    for i in range(len(hoi_frames) - 1):
        f1 = hoi_frames[i]
        f2 = hoi_frames[i + 1]

        f1_path = str(FRAMES_ROOT / f1["frame_path"])
        f2_path = str(FRAMES_ROOT / f2["frame_path"])

        if not os.path.exists(f1_path) or not os.path.exists(f2_path):
            print(f"  Skipping: missing frame file(s)")
            continue

        # Load images (resized to fit GPU memory)
        img1, scale = load_image(f1_path)
        img2, _ = load_image(f2_path)
        img1 = img1.cuda()
        img2 = img2.cuda()

        # Pad for FlowFormer
        padder = InputPadder(img1.shape)
        img1_p, img2_p = padder.pad(img1, img2)

        # Run FlowFormer
        with torch.no_grad():
            flow_pred, _ = model(img1_p, img2_p)
        flow = padder.unpad(flow_pred[0]).cpu()  # [2, H, W]

        # Compute flow stats for each detection in frame 1
        for det in f1.get("detections", []):
            obj_bbox = det.get("obj_bbox")
            if obj_bbox is None:
                continue

            # Scale bbox to match resized image
            scaled_bbox = [c * scale for c in obj_bbox]
            stats = compute_bbox_flow_stats(flow, scaled_bbox)
            if stats is None:
                continue

            # Scale flow magnitudes back to original pixel space
            stats["bbox_flow_mean_magnitude"] /= scale
            stats["bbox_flow_median_magnitude"] /= scale
            stats["bbox_flow_max_magnitude"] /= scale
            stats["bbox_flow_mean_dx"] /= scale
            stats["bbox_flow_mean_dy"] /= scale

            pair_result = {
                "frame1": os.path.basename(f1["frame_path"]),
                "frame2": os.path.basename(f2["frame_path"]),
                "timestamp1": f1["timestamp"],
                "timestamp2": f2["timestamp"],
                "obj_bbox": [round(c, 2) for c in obj_bbox],
                "contact_state": det.get("contact_state", "unknown"),
                "hand_side": det.get("hand_side", "unknown"),
                **stats,
            }
            frame_pairs.append(pair_result)

        print(f"  Processed t={f1['timestamp']:.1f}s -> t={f2['timestamp']:.1f}s "
              f"({len(f1.get('detections', []))} detections)")

    # 5. Save results
    output = {
        "event": event_info,
        "frame_pairs": frame_pairs,
    }
    os.makedirs(OUTPUT_PATH.parent, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved {len(frame_pairs)} frame-pair results to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

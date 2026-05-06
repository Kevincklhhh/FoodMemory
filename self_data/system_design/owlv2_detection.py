#!/usr/bin/env python3
"""
owlv2_detection.py - Run OWLv2 zero-shot food detection on all session frames.

Uses OWLv2 (google/owlv2-base-patch16-ensemble) to detect food items in all
extracted frames for a session. Text queries come from ledger.json visual_class names.

Loads clip_offset_s from the hands23 results to compute session_timestamp_s
for each detection.

Usage:
    python system_design/owlv2_detection.py --participant kailai --session 20260310-195710
    python system_design/owlv2_detection.py --participant kailai --session 20260310-195710 --threshold 0.15

Prerequisites:
    - Frames extracted by 01_extract_and_detect_hands.py
    - ledger.json with food item visual_class names
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
from PIL import Image
from tqdm import tqdm
from transformers import Owlv2ForObjectDetection, Owlv2Processor

from utils import hands23_dir, interact_dir, load_food_items

OWL_MODEL_ID = "google/owlv2-base-patch16-ensemble"


def _ts_from_filename(name: str) -> float:
    """Extract clip timestamp from 'frame_XXXXXXXX_tN.NNs.jpg'."""
    parts = name.rsplit("_t", 1)
    if len(parts) == 2:
        return float(parts[1].rstrip("s.jpg"))
    return 0.0


def get_frame_paths_with_offsets(
    participant: str,
    session: str,
) -> Dict[str, Dict]:
    """
    Return {clip_stem: {"clip_offset_s": float, "frames": [Path, ...]}}
    by reading the hands23 results for clip offsets, then globbing frames on disk.
    """
    det_dir = hands23_dir(participant, session)
    hands23_file = det_dir / f"{participant}_{session}_hands23_results.json"

    # Build offset map from hands23 results
    offset_map: Dict[str, float] = {}
    if hands23_file.exists():
        with open(hands23_file) as f:
            data = json.load(f)
        for video in data.get("videos", []):
            offset_map[video["video_id"]] = video.get("clip_offset_s", 0.0)

    result = {}
    for clip_dir in sorted(det_dir.iterdir()):
        if not clip_dir.is_dir():
            continue
        frames_dir = clip_dir / "frames"
        if not frames_dir.exists():
            continue
        frames = sorted(frames_dir.glob("frame_*.jpg"))
        if not frames:
            continue
        clip_stem = clip_dir.name
        result[clip_stem] = {
            "clip_offset_s": offset_map.get(clip_stem, 0.0),
            "frames": frames,
        }
    return result


def run_owlv2_detection(
    participant: str,
    session: str,
    threshold: float = 0.1,
    device: str = "cuda:0",
    batch_size: int = 8,
) -> None:
    food_items = load_food_items(participant)
    print(f"\nParticipant: {participant} | Session: {session}")
    print(f"Food items: {len(food_items)}")
    for i, name in enumerate(food_items, 1):
        print(f"  {i:2d}. {name}")

    text_queries = [f"a photo of {name}" for name in food_items]

    print(f"\nLoading OWLv2 on {device}...")
    processor = Owlv2Processor.from_pretrained(OWL_MODEL_ID)
    model = Owlv2ForObjectDetection.from_pretrained(OWL_MODEL_ID).to(device)
    model.eval()

    clip_data = get_frame_paths_with_offsets(participant, session)
    if not clip_data:
        print(f"No frames found in {hands23_dir(participant, session)}")
        return

    total_frames = sum(len(v["frames"]) for v in clip_data.values())
    print(f"\nProcessing {total_frames} frames across {len(clip_data)} clips")

    # Prepare output dir
    out_dir = interact_dir(participant, session)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_videos = {}

    for clip_stem, info in clip_data.items():
        frames = info["frames"]
        clip_offset = info["clip_offset_s"]
        print(f"\n--- {clip_stem} ({len(frames)} frames, offset {clip_offset:.1f}s) ---")

        clip_detections = []

        for batch_start in tqdm(range(0, len(frames), batch_size), desc=clip_stem):
            batch_paths = frames[batch_start:batch_start + batch_size]
            batch_images = [Image.open(p).convert("RGB") for p in batch_paths]

            inputs = processor(
                text=[text_queries] * len(batch_images),
                images=batch_images,
                return_tensors="pt",
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)

            target_sizes = torch.tensor(
                [img.size[::-1] for img in batch_images], device=device
            )
            results_batch = processor.post_process_object_detection(
                outputs, threshold=threshold, target_sizes=target_sizes
            )

            for img_idx, frame_path in enumerate(batch_paths):
                clip_ts = _ts_from_filename(frame_path.name)
                session_ts = round(clip_offset + clip_ts, 3)

                res = results_batch[img_idx]
                boxes = res["boxes"].cpu().tolist()
                scores = res["scores"].cpu().tolist()
                label_ids = res["labels"].cpu().tolist()

                frame_dets = [
                    {
                        "label": food_items[lid],
                        "score": round(score, 4),
                        "box": [round(c, 1) for c in box],
                    }
                    for box, score, lid in zip(boxes, scores, label_ids)
                ]

                if frame_dets:
                    clip_detections.append({
                        "frame": frame_path.name,
                        "clip_timestamp_s": round(clip_ts, 3),
                        "session_timestamp_s": session_ts,
                        "detections": frame_dets,
                    })

        all_videos[clip_stem] = {
            "clip_offset_s": clip_offset,
            "num_frames": len(frames),
            "num_frames_with_detections": len(clip_detections),
            "frames": clip_detections,
        }

        det_counts: Dict[str, int] = {}
        for fd in clip_detections:
            for d in fd["detections"]:
                det_counts[d["label"]] = det_counts.get(d["label"], 0) + 1
        if det_counts:
            total_d = sum(det_counts.values())
            print(f"  {total_d} detections across {len(det_counts)} item types")
            for label, cnt in sorted(det_counts.items(), key=lambda x: -x[1])[:8]:
                print(f"    {label}: {cnt}")

    output = {
        "participant": participant,
        "session": session,
        "model": OWL_MODEL_ID,
        "threshold": threshold,
        "food_items": food_items,
        "num_labels": len(food_items),
        "videos": all_videos,
    }

    out_file = out_dir / f"{participant}_{session}_owlv2_detections.json"
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)

    total_det_frames = sum(v["num_frames_with_detections"] for v in all_videos.values())
    print(f"\nOverall: {total_det_frames}/{total_frames} frames with detections")
    print(f"Saved: {out_file}")


def main():
    parser = argparse.ArgumentParser(description="OWLv2 zero-shot food detection on session frames")
    parser.add_argument("--participant", required=True, help="Participant ID (e.g., kailai)")
    parser.add_argument("--session", required=True, help="Session ID (e.g., 20260310-195710)")
    parser.add_argument("--threshold", type=float, default=0.1, help="Detection threshold (default: 0.1)")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device (default: cuda:0)")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size (default: 8)")

    args = parser.parse_args()
    run_owlv2_detection(
        participant=args.participant,
        session=args.session,
        threshold=args.threshold,
        device=args.device,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()

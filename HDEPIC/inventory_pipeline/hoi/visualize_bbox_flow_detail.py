#!/usr/bin/env python3
"""
Visualize per-bbox optical flow using FlowFormer.

For each consecutive frame pair in the event time range, runs FlowFormer,
crops the flow to each detected object bbox, and produces a figure with:
  - Row per frame-pair/detection
  - Columns: frame1 crop | flow color map | flow arrows on frame1 crop | frame2 crop

Also produces a color wheel legend for the flow encoding.
"""

from __future__ import annotations

import json
import os
import sys
import numpy as np
import torch
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJ_ROOT = Path(__file__).resolve().parent.parent
AMEGO_DIR = PROJ_ROOT / "models" / "AMEGO"
FLOWFORMER_DIR = AMEGO_DIR / "submodules" / "flowformer"
FLOWFORMER_CORE = FLOWFORMER_DIR / "core"
FLOWFORMER_WEIGHTS = FLOWFORMER_DIR / "models" / "sintel.pth"

EVENT_PATH = PROJ_ROOT / "outputs" / "03_gating" / "flow_test.json"
HOI_RESULTS_PATH = (PROJ_ROOT / "outputs" / "02_inventory" / "P03" /
                    "hands23_detection" / "P03_hands23_results.json")
FRAMES_ROOT = PROJ_ROOT / "outputs" / "02_inventory" / "P03" / "hands23_detection"
OUTPUT_PATH = PROJ_ROOT / "outputs" / "03_gating" / "flow_test_bbox_flow_detail.png"

# ---------------------------------------------------------------------------
# FlowFormer imports
# ---------------------------------------------------------------------------
sys.path.insert(0, str(FLOWFORMER_CORE))
sys.path.insert(0, str(AMEGO_DIR))

from submodules.flowformer.configs.sintel import get_cfg
from submodules.flowformer.core.FlowFormer import build_flowformer
from submodules.flowformer.core.utils.utils import InputPadder
from submodules.flowformer.core.utils.flow_viz import flow_to_image

HAND_COLORS = {"left_hand": "#2196F3", "right_hand": "#F44336"}


def load_event(path):
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


def load_hoi_frames(hoi_path, video_id, t_start, t_end):
    with open(hoi_path) as f:
        hoi_data = json.load(f)
    for v in hoi_data["videos"]:
        if v["video_id"] == video_id:
            frames = [f for f in v["frames"] if t_start <= f["timestamp"] <= t_end]
            frames.sort(key=lambda f: f["timestamp"])
            return frames
    raise ValueError(f"Video {video_id} not found")


def build_model():
    cfg = get_cfg()
    model = torch.nn.DataParallel(build_flowformer(cfg))
    print(f"Loading FlowFormer weights from {FLOWFORMER_WEIGHTS}")
    model.load_state_dict(torch.load(str(FLOWFORMER_WEIGHTS), map_location="cpu"))
    model.cuda()
    model.eval()
    return model


def load_image_tensor(path, max_dim=704):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = min(max_dim / w, max_dim / h)
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.BILINEAR)
    arr = np.array(img).astype(np.uint8)
    tensor = torch.from_numpy(arr).permute(2, 0, 1).float().unsqueeze(0)
    return tensor, scale, img


def crop_bbox(arr, bbox):
    """Crop a [H, W, ...] or [C, H, W] array to integer bbox [x1,y1,x2,y2]."""
    if arr.ndim == 3 and arr.shape[0] in (2, 3):
        # [C, H, W]
        _, H, W = arr.shape
        x1 = max(0, int(round(bbox[0])))
        y1 = max(0, int(round(bbox[1])))
        x2 = min(W, int(round(bbox[2])))
        y2 = min(H, int(round(bbox[3])))
        return arr[:, y1:y2, x1:x2], (x1, y1, x2, y2)
    else:
        # [H, W, C] or [H, W]
        H = arr.shape[0]
        W = arr.shape[1]
        x1 = max(0, int(round(bbox[0])))
        y1 = max(0, int(round(bbox[1])))
        x2 = min(W, int(round(bbox[2])))
        y2 = min(H, int(round(bbox[3])))
        return arr[y1:y2, x1:x2], (x1, y1, x2, y2)


def draw_quiver(ax, flow_crop, step=12):
    """Overlay sparse flow arrows on an axes."""
    h, w = flow_crop.shape[1], flow_crop.shape[2]
    ys = np.arange(0, h, step)
    xs = np.arange(0, w, step)
    xx, yy = np.meshgrid(xs, ys)

    u = flow_crop[0][yy, xx]
    v = flow_crop[1][yy, xx]

    mag = np.sqrt(u**2 + v**2)
    # Normalize arrow lengths for visibility but keep direction
    max_mag = mag.max() + 1e-5
    scale_factor = step * 1.5 / max_mag

    ax.quiver(xx, yy, u * scale_factor, v * scale_factor,
              mag, cmap="hot", angles="xy", scale_units="xy", scale=1,
              width=0.003 * max(w, h), headwidth=4, headlength=5,
              minshaft=1.5)


def draw_colorwheel(ax):
    """Draw a color wheel legend for the flow encoding."""
    n = 256
    x = np.linspace(-1, 1, n)
    y = np.linspace(-1, 1, n)
    xx, yy = np.meshgrid(x, y)
    r = np.sqrt(xx**2 + yy**2)
    mask = r <= 1.0

    # Build flow array [H, W, 2]
    flow = np.stack([xx, yy], axis=-1)
    flow[~mask] = 0
    img = flow_to_image(flow, max_flow=1.0)
    img[~mask] = 255

    ax.imshow(img)
    ax.set_xlim(0, n)
    ax.set_ylim(n, 0)
    # Add direction labels
    c = n // 2
    offset = c + 10
    ax.text(n - 5, c, "→", ha="right", va="center", fontsize=10, fontweight="bold")
    ax.text(5, c, "←", ha="left", va="center", fontsize=10, fontweight="bold")
    ax.text(c, 5, "↑", ha="center", va="top", fontsize=10, fontweight="bold")
    ax.text(c, n - 5, "↓", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_title("Flow direction\ncolor key", fontsize=8)
    ax.axis("off")


def main():
    event_info = load_event(EVENT_PATH)
    print(f"Event: {event_info['food_name']} [{event_info['start']:.1f}s – {event_info['end']:.1f}s]")

    hoi_frames = load_hoi_frames(HOI_RESULTS_PATH, event_info["video_id"],
                                 event_info["start"], event_info["end"])
    print(f"Found {len(hoi_frames)} frames")

    model = build_model()

    # Collect all rows: (frame1_crop, flow_color_crop, flow_arrow_data, frame2_crop, label, hand)
    rows = []
    # Also track a global max flow for consistent color normalization
    global_max_flow = 0

    all_flow_crops = []  # store raw flow crops for second pass

    for i in range(len(hoi_frames) - 1):
        f1, f2 = hoi_frames[i], hoi_frames[i + 1]
        f1_path = str(FRAMES_ROOT / f1["frame_path"])
        f2_path = str(FRAMES_ROOT / f2["frame_path"])

        if not os.path.exists(f1_path) or not os.path.exists(f2_path):
            continue

        img1_t, scale, img1_pil = load_image_tensor(f1_path)
        img2_t, _, img2_pil = load_image_tensor(f2_path)

        padder = InputPadder(img1_t.shape)
        img1_p, img2_p = padder.pad(img1_t.cuda(), img2_t.cuda())

        with torch.no_grad():
            flow_pred, _ = model(img1_p, img2_p)
        flow = padder.unpad(flow_pred[0]).cpu().numpy()  # [2, H, W]

        img1_np = np.array(img1_pil)
        img2_np = np.array(img2_pil)

        for det in f1.get("detections", []):
            obj_bbox = det.get("obj_bbox")
            if obj_bbox is None:
                continue
            scaled_bbox = [c * scale for c in obj_bbox]

            flow_crop, (bx1, by1, bx2, by2) = crop_bbox(flow, scaled_bbox)
            if flow_crop.shape[1] < 4 or flow_crop.shape[2] < 4:
                continue

            img1_crop = img1_np[by1:by2, bx1:bx2]
            img2_crop = img2_np[by1:by2, bx1:bx2]

            mag = np.sqrt(flow_crop[0]**2 + flow_crop[1]**2)
            local_max = float(mag.max())
            global_max_flow = max(global_max_flow, local_max)

            hand = det.get("hand_side", "unknown")
            contact = det.get("contact_state", "unknown")
            t1 = f1["timestamp"]
            t2 = f2["timestamp"]
            mean_mag = float(mag.mean())
            label = (f"t={t1:.0f}→{t2:.0f}s  |  {hand.replace('_', ' ')}  |  "
                     f"{contact.replace('_', ' ')}  |  mean={mean_mag:.1f}px  max={local_max:.1f}px")

            all_flow_crops.append(flow_crop)
            rows.append({
                "img1_crop": img1_crop,
                "img2_crop": img2_crop,
                "flow_crop": flow_crop,
                "label": label,
                "hand": hand,
                "mean_mag": mean_mag,
                "max_mag": local_max,
            })

        print(f"  t={f1['timestamp']:.0f}s → t={f2['timestamp']:.0f}s done")

    if not rows:
        print("No rows to visualize.")
        return

    print(f"\n{len(rows)} bbox detections, global max flow = {global_max_flow:.1f}px")
    print("Rendering figure...")

    # --- Build the figure ---
    n_rows = len(rows)
    img_cols = 5  # frame1 | flow local | flow global | arrows | frame2

    fig_w = 24
    row_h = 3.5
    fig_h = row_h * n_rows + 2.0

    fig, axes = plt.subplots(
        n_rows, img_cols, figsize=(fig_w, fig_h),
        gridspec_kw={"hspace": 0.45, "wspace": 0.06})
    if n_rows == 1:
        axes = axes.reshape(1, img_cols)

    # Suptitle
    fig.suptitle(
        f"Optical Flow Detail — {event_info['food_name']}\n"
        f"Video {event_info['video_id']}  "
        f"[{event_info['start']:.1f}s – {event_info['end']:.1f}s]   "
        f"(global max flow: {global_max_flow:.1f}px)",
        fontsize=14, fontweight="bold", y=1.0)

    col_titles = ["Frame 1 (bbox crop)", "Flow (local norm)",
                  "Flow (global norm)", "Flow arrows on F1",
                  "Frame 2 (bbox crop)"]

    for row_idx, row in enumerate(rows):
        flow_crop = row["flow_crop"]
        img1_crop = row["img1_crop"]
        img2_crop = row["img2_crop"]
        hand_color = HAND_COLORS.get(row["hand"], "gray")

        flow_hw2 = np.transpose(flow_crop, (1, 2, 0))
        flow_color_local = flow_to_image(flow_hw2)
        flow_color_global = flow_to_image(flow_hw2, max_flow=global_max_flow / scale)

        img_data = [img1_crop, flow_color_local, flow_color_global, None, img2_crop]
        for j, img in enumerate(img_data):
            ax = axes[row_idx, j]
            if j == 3:
                ax.imshow(img1_crop)
                step = max(8, min(flow_crop.shape[1], flow_crop.shape[2]) // 15)
                draw_quiver(ax, flow_crop, step=step)
            else:
                ax.imshow(img)

            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor(hand_color)
                spine.set_linewidth(2.5)

        # Row label below the row, spanning all columns
        axes[row_idx, 2].set_xlabel(
            row["label"],
            fontsize=8, fontweight="bold", color=hand_color,
            labelpad=4)

        # Column headers on first data row only
        if row_idx == 0:
            for j in range(img_cols):
                axes[0, j].set_title(
                    col_titles[j],
                    fontsize=10, fontweight="bold", pad=6)

    # Color wheel inset
    cw_size = 0.055
    inset_ax = fig.add_axes([0.94, 0.92, cw_size, cw_size * fig_w / fig_h])
    draw_colorwheel(inset_ax)

    os.makedirs(OUTPUT_PATH.parent, exist_ok=True)
    fig.savefig(str(OUTPUT_PATH), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Visualize optical flow within HOI object bounding boxes.

Produces a multi-panel figure:
  - Top: flow magnitude timeline per hand (left/right), bars = mean, triangles = max
  - Middle: per-frame thumbnails with bbox overlay and mean flow arrow
  - Bottom: mean flow dx/dy components over time
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parent.parent
FLOW_JSON = PROJ_ROOT / "outputs" / "03_gating" / "flow_test_bbox_flow.json"
FRAMES_ROOT = PROJ_ROOT / "outputs" / "02_inventory" / "P03" / "hands23_detection"
OUT_DIR = PROJ_ROOT / "outputs" / "03_gating"

HAND_COLORS = {"left_hand": "#2196F3", "right_hand": "#F44336"}
BAR_OFFSETS = {"left_hand": -0.18, "right_hand": 0.18}


def load_data():
    with open(FLOW_JSON) as f:
        return json.load(f)


def _add_event_end_line(ax, event_end):
    """Draw a dashed vertical line at the event end timestamp."""
    ax.axvline(event_end, color="#4CAF50", linewidth=1.5, linestyle="--",
               alpha=0.8, zorder=10, label=f"event end ({event_end:.1f}s)")


def make_magnitude_plot(data, ax):
    """Bar chart of mean flow magnitude per frame pair, split by hand."""
    pairs = data["frame_pairs"]
    event = data["event"]
    labels_seen = set()

    for p in pairs:
        hand = p["hand_side"]
        t = (p["timestamp1"] + p["timestamp2"]) / 2.0 + BAR_OFFSETS.get(hand, 0)
        color = HAND_COLORS.get(hand, "gray")
        label = hand.replace("_", " ") if hand not in labels_seen else None
        labels_seen.add(hand)

        ax.bar(t, p["bbox_flow_mean_magnitude"], width=0.32,
               color=color, alpha=0.75, label=label,
               edgecolor="white", linewidth=0.5)
        ax.plot(t, p["bbox_flow_max_magnitude"], marker="v", color=color,
                markersize=5, markeredgecolor="black", markeredgewidth=0.5,
                zorder=5)

    ax.plot([], [], marker="v", color="gray", linestyle="none",
            markersize=5, label="max magnitude")

    _add_event_end_line(ax, event["end"])

    analysis_end = event.get("analysis_end", event["end"])
    ax.set_ylabel("Flow magnitude (px)")
    ax.set_title(
        f"Optical Flow in Object BBox — {event['food_name']}\n"
        f"Video {event['video_id']}  "
        f"[{event['start']:.1f}s – {event['end']:.1f}s]  "
        f"(analysis window to {analysis_end:.1f}s)",
        fontsize=11, fontweight="bold")
    ax.legend(loc="upper left", fontsize=8)

    t_min = event["start"] - 0.5
    t_max = analysis_end + 0.5
    ax.set_xlim(t_min, t_max)
    ax.grid(axis="y", alpha=0.3)


def make_direction_plot(data, ax):
    """Horizontal bar chart showing mean dx and dy per frame pair."""
    pairs = data["frame_pairs"]
    event = data["event"]
    labels_seen = set()

    for p in pairs:
        hand = p["hand_side"]
        t = (p["timestamp1"] + p["timestamp2"]) / 2.0 + BAR_OFFSETS.get(hand, 0)
        color = HAND_COLORS.get(hand, "gray")
        label = hand.replace("_", " ") if hand not in labels_seen else None
        labels_seen.add(hand)

        ax.bar(t, p["bbox_flow_mean_dx"], width=0.32,
               color=color, alpha=0.6, label=label,
               edgecolor="white", linewidth=0.5)
        ax.bar(t, p["bbox_flow_mean_dy"], width=0.16,
               color=color, alpha=0.95, hatch="//",
               edgecolor="white", linewidth=0.5)

    ax.bar([], [], width=0, color="gray", alpha=0.6, label="mean dx")
    ax.bar([], [], width=0, color="gray", alpha=0.95, hatch="//", label="mean dy")

    _add_event_end_line(ax, event["end"])

    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel("Flow component (px)")
    ax.set_xlabel("Time (s)")
    ax.legend(loc="upper left", fontsize=8, ncol=2)

    analysis_end = event.get("analysis_end", event["end"])
    ax.set_xlim(event["start"] - 0.5, analysis_end + 0.5)
    ax.grid(axis="y", alpha=0.3)


def make_frame_strip(data, axes, event_end):
    """Show frame thumbnails in a row with bbox overlays and flow arrows."""
    pairs = data["frame_pairs"]
    video_id = data["event"]["video_id"]

    # Group by frame1 timestamp
    by_ts = {}
    for p in pairs:
        t = p["timestamp1"]
        by_ts.setdefault(t, []).append(p)
    timestamps = sorted(by_ts.keys())

    for idx, ts in enumerate(timestamps):
        ax = axes[idx]
        frame_pairs_at_t = by_ts[ts]
        frame_name = frame_pairs_at_t[0]["frame1"]
        frame_path = FRAMES_ROOT / video_id / "frames" / frame_name

        if not frame_path.exists():
            ax.text(0.5, 0.5, "missing", ha="center", va="center", transform=ax.transAxes)
            ax.axis("off")
            continue

        img = Image.open(frame_path)
        orig_w, orig_h = img.size
        ax.imshow(img)

        for p in frame_pairs_at_t:
            x1, y1, x2, y2 = p["obj_bbox"]
            color = HAND_COLORS.get(p["hand_side"], "yellow")

            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                     linewidth=2.5, edgecolor=color,
                                     facecolor=color, alpha=0.15)
            ax.add_patch(rect)
            rect_border = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                            linewidth=2.5, edgecolor=color,
                                            facecolor="none")
            ax.add_patch(rect_border)

            # Flow arrow from bbox center
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            dx, dy = p["bbox_flow_mean_dx"], p["bbox_flow_mean_dy"]
            ax.annotate("", xy=(cx + dx, cy + dy), xytext=(cx, cy),
                        arrowprops=dict(arrowstyle="-|>", color="lime",
                                        lw=2.5, mutation_scale=15))
            # Magnitude label
            mag = p["bbox_flow_mean_magnitude"]
            ax.text(cx, y1 - 15, f"{mag:.0f}px", color=color,
                    fontsize=7, ha="center", va="bottom",
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.7, ec="none"))

        ax.set_xlim(0, orig_w)
        ax.set_ylim(orig_h, 0)
        ax.set_xticks([])
        ax.set_yticks([])

        # Mark frames after event end
        is_post_event = ts > event_end
        title_color = "#4CAF50" if is_post_event else "black"
        suffix = " *" if is_post_event else ""
        ax.set_title(f"t={ts:.0f}s{suffix}", fontsize=8, pad=2, color=title_color,
                     fontweight="bold" if is_post_event else "normal")

        if is_post_event:
            for spine in ax.spines.values():
                spine.set_edgecolor("#4CAF50")
                spine.set_linewidth(2.5)


def main():
    data = load_data()
    event = data["event"]

    # Count unique timestamps for frame strip
    timestamps = sorted(set(p["timestamp1"] for p in data["frame_pairs"]))
    n_frames = len(timestamps)

    fig = plt.figure(figsize=(max(18, n_frames * 1.6), 16))
    gs = fig.add_gridspec(3, n_frames,
                          height_ratios=[2.5, 2.5, 2],
                          hspace=0.3, wspace=0.08)

    # Top row: magnitude bar chart (span all columns)
    ax_mag = fig.add_subplot(gs[0, :])
    make_magnitude_plot(data, ax_mag)

    # Middle row: individual frame thumbnails
    frame_axes = [fig.add_subplot(gs[1, i]) for i in range(n_frames)]
    make_frame_strip(data, frame_axes, event["end"])

    # Bottom row: direction chart (span all columns)
    ax_dir = fig.add_subplot(gs[2, :])
    make_direction_plot(data, ax_dir)

    out_path = OUT_DIR / "flow_test_bbox_flow_viz.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved visualization to {out_path}")


if __name__ == "__main__":
    main()

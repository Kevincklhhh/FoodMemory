#!/usr/bin/env python3
"""
07_visualize_tracklets.py - Visualize HOI tracklets by drawing bounding boxes on frames

Usage:
    python 07_visualize_tracklets.py --tracklets tracklets.json --frames_dir rgb_frames/ --output_dir viz/
    python 07_visualize_tracklets.py --tracklets tracklets.json --video_dir outputs/03_inventory_results/P01-xxx/

    # Use with exported tracklets from 07_hoi_samples.py
    python 07_visualize_tracklets.py \
        --tracklets outputs/03_inventory_results/P01-xxx/HOI_AMEGO/P01-xxx_tracklets_120s-180s.json \
        --video_dir outputs/03_inventory_results/P01-xxx/
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import colorsys


def get_cluster_color(cluster_id: int, num_clusters: int = 80) -> Tuple[int, int, int]:
    """Generate a distinct color for each cluster ID using HSV color space."""
    hue = (cluster_id * 0.618033988749895) % 1.0  # Golden ratio for good distribution
    saturation = 0.7 + (cluster_id % 3) * 0.1  # Vary saturation slightly
    value = 0.9
    r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
    return (int(r * 255), int(g * 255), int(b * 255))


def load_tracklets(tracklets_path: Path) -> Dict[str, Any]:
    """Load tracklets JSON file."""
    with open(tracklets_path, 'r') as f:
        return json.load(f)


def build_frame_index(tracklets: List[Dict[str, Any]], original_fps: float) -> Dict[int, List[Dict]]:
    """
    Build an index mapping frame numbers to active tracklets and their bboxes.

    Returns:
        Dict mapping frame_num -> list of {tracklet_info, bbox}
    """
    frame_index = {}

    for tracklet in tracklets:
        track_id = tracklet["track_id"]
        cluster = tracklet["cluster"]
        hand_side = tracklet["hand_side"]

        # Get the raw tracklet data with per-frame bboxes
        # The analyzed tracklet only has avg_bbox, we need frame-level data
        start_frame = tracklet["start_frame"]
        end_frame = tracklet["end_frame"]

        # Store tracklet info for each frame in its range
        for frame_num in range(start_frame, end_frame + 1):
            if frame_num not in frame_index:
                frame_index[frame_num] = []
            frame_index[frame_num].append({
                "track_id": track_id,
                "cluster": cluster,
                "hand_side": hand_side,
                "bbox": tracklet["avg_bbox"]  # Use avg_bbox as approximation
            })

    return frame_index


def build_frame_index_from_raw(
    raw_tracklets: List[Dict[str, Any]]
) -> Dict[int, List[Dict]]:
    """
    Build frame index from raw HOI JSON data (with per-frame bboxes).

    Args:
        raw_tracklets: Raw tracklet data from HOI_AMEGO JSON

    Returns:
        Dict mapping frame_num -> list of {track_id, cluster, hand_side, bbox}
    """
    frame_index = {}

    for tracklet in raw_tracklets:
        track_id = tracklet["track_id"]
        cluster = tracklet.get("cluster", -1)
        sides = tracklet.get("side", [])
        frames = tracklet["num_frame"]
        bboxes = tracklet["obj_bbox"]

        for i, frame_num in enumerate(frames):
            if frame_num not in frame_index:
                frame_index[frame_num] = []

            bbox = bboxes[i]  # [x, y, w, h]
            side = sides[i] if i < len(sides) else "unknown"

            frame_index[frame_num].append({
                "track_id": track_id,
                "cluster": cluster,
                "hand_side": side if isinstance(side, str) else side[0],
                "bbox": bbox
            })

    return frame_index


def draw_bbox_on_frame(
    frame_path: Path,
    detections: List[Dict],
    output_path: Path,
    show_labels: bool = True,
    line_width: int = 2
) -> None:
    """Draw bounding boxes on a frame and save it."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        raise ImportError("PIL/Pillow is required. Install with: pip install Pillow")

    img = Image.open(frame_path)
    draw = ImageDraw.Draw(img)

    # Try to load a font, fall back to default
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except:
        font = ImageFont.load_default()

    for det in detections:
        bbox = det["bbox"]
        cluster = det["cluster"]
        track_id = det["track_id"]
        hand_side = det["hand_side"]

        # Handle both dict format (avg_bbox) and list format (raw bbox)
        if isinstance(bbox, dict):
            x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
        else:
            x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]

        color = get_cluster_color(cluster)

        # Draw rectangle
        draw.rectangle(
            [(x, y), (x + w, y + h)],
            outline=color,
            width=line_width
        )

        if show_labels:
            # Draw label background
            label = f"C{cluster} T{track_id}"
            hand_label = "L" if "left" in str(hand_side).lower() else "R"
            full_label = f"{label} ({hand_label})"

            # Get text size
            text_bbox = draw.textbbox((0, 0), full_label, font=font)
            text_w = text_bbox[2] - text_bbox[0]
            text_h = text_bbox[3] - text_bbox[1]

            # Draw label background
            label_y = max(0, y - text_h - 4)
            draw.rectangle(
                [(x, label_y), (x + text_w + 4, label_y + text_h + 4)],
                fill=color
            )

            # Draw text
            draw.text((x + 2, label_y + 2), full_label, fill=(255, 255, 255), font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)


def visualize_tracklets(
    tracklets_json: Path,
    frames_dir: Path,
    output_dir: Path,
    output_fps: float = 5.0,
    original_fps: float = 30.0,
    raw_hoi_json: Optional[Path] = None,
    show_labels: bool = True,
    line_width: int = 2
) -> Path:
    """
    Visualize tracklets by drawing bounding boxes on frames.

    Args:
        tracklets_json: Path to exported tracklets JSON from 07_hoi_samples.py
        frames_dir: Directory containing extracted frames (frame_*.jpg)
        output_dir: Output directory for visualized frames
        output_fps: Output frame rate (default: 5 FPS)
        original_fps: Original video frame rate (default: 30 FPS)
        raw_hoi_json: Optional path to raw HOI JSON for per-frame bboxes
        show_labels: Whether to show cluster/track labels
        line_width: Bounding box line width

    Returns:
        Path to output directory
    """
    print(f"Loading tracklets from: {tracklets_json}")
    data = load_tracklets(tracklets_json)

    video_id = data.get("video_id", "unknown")
    fps = data.get("fps", original_fps)
    query = data.get("query", {})
    tracklets = data.get("tracklets", [])

    start_seconds = query.get("start_seconds")
    end_seconds = query.get("end_seconds")

    print(f"Video ID: {video_id}")
    print(f"Original FPS: {fps}, Output FPS: {output_fps}")
    print(f"Tracklets to visualize: {len(tracklets)}")

    # Build frame index
    if raw_hoi_json and raw_hoi_json.exists():
        print(f"Loading raw HOI data for precise bboxes: {raw_hoi_json}")
        with open(raw_hoi_json, 'r') as f:
            raw_data = json.load(f)

        # Filter raw tracklets to only those in our exported set
        track_ids = {t["track_id"] for t in tracklets}
        filtered_raw = [t for t in raw_data if t["track_id"] in track_ids]
        frame_index = build_frame_index_from_raw(filtered_raw)
        print(f"Built frame index from {len(filtered_raw)} raw tracklets")
    else:
        # Use analyzed tracklets (avg_bbox only)
        frame_index = build_frame_index(tracklets, fps)
        print("Using averaged bounding boxes (raw HOI JSON not provided)")

    # Determine frame range
    if start_seconds is not None:
        start_frame = int(start_seconds * fps)
    else:
        start_frame = min(frame_index.keys()) if frame_index else 1

    if end_seconds is not None:
        end_frame = int(end_seconds * fps)
    else:
        end_frame = max(frame_index.keys()) if frame_index else 1

    print(f"Frame range: {start_frame} - {end_frame}")

    # Calculate frame sampling interval
    frame_interval = int(fps / output_fps)

    # Process frames
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_processed = 0
    frames_with_detections = 0

    for frame_num in range(start_frame, end_frame + 1, frame_interval):
        # Find frame file (1-indexed)
        frame_path = frames_dir / f"frame_{frame_num:010d}.jpg"

        if not frame_path.exists():
            continue

        # Get detections for this frame (check nearby frames too due to sampling)
        detections = []
        for check_frame in range(frame_num - frame_interval // 2, frame_num + frame_interval // 2 + 1):
            if check_frame in frame_index:
                detections.extend(frame_index[check_frame])

        # Remove duplicates (same track_id)
        seen_tracks = set()
        unique_detections = []
        for det in detections:
            if det["track_id"] not in seen_tracks:
                seen_tracks.add(det["track_id"])
                unique_detections.append(det)

        # Generate output filename with timestamp
        timestamp = frame_num / fps
        output_filename = f"frame_{frame_num:010d}_t{timestamp:.2f}s.jpg"
        output_path = output_dir / output_filename

        if unique_detections:
            draw_bbox_on_frame(
                frame_path, unique_detections, output_path,
                show_labels=show_labels, line_width=line_width
            )
            frames_with_detections += 1
        else:
            # Copy frame without annotations if no detections
            from PIL import Image
            img = Image.open(frame_path)
            img.save(output_path)

        frames_processed += 1

    print(f"\nProcessed {frames_processed} frames ({frames_with_detections} with detections)")
    print(f"Output saved to: {output_dir}")

    # Generate a summary/legend image
    generate_legend(tracklets, output_dir / "legend.png")

    return output_dir


def generate_legend(tracklets: List[Dict], output_path: Path) -> None:
    """Generate a legend showing cluster colors."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return

    # Get unique clusters
    clusters = sorted(set(t["cluster"] for t in tracklets))

    if not clusters:
        return

    # Create legend image
    row_height = 25
    width = 300
    height = row_height * (len(clusters) + 1) + 20

    img = Image.new('RGB', (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except:
        font = ImageFont.load_default()

    # Title
    draw.text((10, 10), "Cluster Legend", fill=(0, 0, 0), font=font)

    # Draw each cluster
    y = 40
    for cluster in clusters:
        color = get_cluster_color(cluster)

        # Color box
        draw.rectangle([(10, y), (30, y + 18)], fill=color, outline=(0, 0, 0))

        # Count tracklets in this cluster
        count = sum(1 for t in tracklets if t["cluster"] == cluster)

        # Label
        draw.text((40, y), f"Cluster {cluster} ({count} tracklets)", fill=(0, 0, 0), font=font)

        y += row_height

    img.save(output_path)
    print(f"Legend saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize HOI tracklets by drawing bounding boxes on frames"
    )

    parser.add_argument("--tracklets", type=Path, required=True,
                        help="Path to tracklets JSON (from 07_hoi_samples.py)")
    parser.add_argument("--frames_dir", type=Path,
                        help="Directory containing extracted frames")
    parser.add_argument("--video_dir", type=Path,
                        help="Video directory (contains rgb_frames/)")
    parser.add_argument("--output_dir", type=Path,
                        help="Output directory for visualized frames")
    parser.add_argument("--raw_hoi", type=Path,
                        help="Path to raw HOI JSON for per-frame bboxes")

    parser.add_argument("--output_fps", type=float, default=5.0,
                        help="Output frame rate (default: 5)")
    parser.add_argument("--original_fps", type=float, default=30.0,
                        help="Original video FPS (default: 30)")
    parser.add_argument("--no_labels", action="store_true",
                        help="Don't show cluster/track labels")
    parser.add_argument("--line_width", type=int, default=2,
                        help="Bounding box line width (default: 2)")

    args = parser.parse_args()

    # Resolve frames directory
    if args.frames_dir:
        frames_dir = args.frames_dir
    elif args.video_dir:
        frames_dir = args.video_dir / "rgb_frames"
    else:
        # Try to infer from tracklets JSON path
        tracklets_data = load_tracklets(args.tracklets)
        video_id = tracklets_data.get("video_id")
        if video_id and args.tracklets.parent.parent.name == video_id:
            frames_dir = args.tracklets.parent.parent / "rgb_frames"
        else:
            parser.error("Must specify --frames_dir or --video_dir")

    if not frames_dir.exists():
        print(f"ERROR: Frames directory not found: {frames_dir}")
        return 1

    # Resolve output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        # Default: create viz/ directory next to tracklets file
        output_dir = args.tracklets.parent / "viz" / args.tracklets.stem

    # Resolve raw HOI JSON
    raw_hoi = args.raw_hoi
    if raw_hoi is None and args.video_dir:
        # Try to find raw HOI JSON
        tracklets_data = load_tracklets(args.tracklets)
        video_id = tracklets_data.get("video_id")
        if video_id:
            candidate = args.video_dir / "HOI_AMEGO" / f"{video_id}.json"
            if candidate.exists():
                raw_hoi = candidate

    # Run visualization
    visualize_tracklets(
        tracklets_json=args.tracklets,
        frames_dir=frames_dir,
        output_dir=output_dir,
        output_fps=args.output_fps,
        original_fps=args.original_fps,
        raw_hoi_json=raw_hoi,
        show_labels=not args.no_labels,
        line_width=args.line_width
    )

    return 0


if __name__ == "__main__":
    exit(main())

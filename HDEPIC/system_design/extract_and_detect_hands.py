#!/usr/bin/env python3
"""
extract_and_detect_hands.py - Extract frames from full videos and run hands23 detector

Based on timeline_annotated.json, identifies videos containing food segments,
extracts frames from FULL videos at 1 fps, then runs hands23 detector on all frames.

Usage:
    # Activate hands23 environment first:
    conda activate hands23

    # Test on one video
    python extract_and_detect_hands.py --participant P03 --test-video P03-20240216-084005

    # Process all videos for a participant
    python extract_and_detect_hands.py --participant P03

    # Frame extraction only (no detection)
    python extract_and_detect_hands.py --participant P03 --skip-detection

Prerequisites:
    - timeline_annotated.json from 06_timeline_aggregation.py
    - hands23 model weights at models/hands23_detector/model_weights/model_hands23.pth
    - conda environment 'hands23' with detectron2 installed
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
from datetime import datetime

import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont

# Add hands23 detector to path
_SCRIPT_DIR = Path(__file__).parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
_HANDS23_DIR = _PROJECT_ROOT / "models" / "hands23_detector"
sys.path.insert(0, str(_HANDS23_DIR))

# Paths
VIDEO_BASE_DIR = _PROJECT_ROOT / "data" / "HD-EPIC" / "Videos"
OUTPUT_BASE_DIR = _PROJECT_ROOT / "outputs" / "02_inventory"
HANDS23_CONFIG = _HANDS23_DIR / "faster_rcnn_X_101_32x8d_FPN_3x_Hands23.yaml"
HANDS23_WEIGHTS = _HANDS23_DIR / "model_weights" / "model_hands23.pth"
FONT_PATH = _HANDS23_DIR / "utils" / "times_b.ttf"

# Colors for visualization (RGB)
HAND_COLORS = {
    "left_hand": (0, 90, 181),    # Blue
    "right_hand": (220, 50, 32),   # Red
}
FIRST_OBJ_COLOR = (255, 194, 10)   # Yellow/Orange
SECOND_OBJ_COLOR = (0, 159, 115)   # Green


def get_videos_from_timeline(timeline_data: Dict) -> Set[str]:
    """
    Extract unique video IDs from timeline_annotated.json.

    Returns:
        Set of video IDs that contain food segments
    """
    video_ids = set()

    for item in timeline_data.get('items', []):
        # Add videos from video_range
        for vid in item.get('video_range', []):
            video_ids.add(vid)

        # Add videos from segments
        for segment in item.get('dispensal_segments', []):
            vid = segment.get('video_id')
            if vid:
                video_ids.add(vid)

    return video_ids


def extract_frames_from_video(
    video_path: Path,
    output_dir: Path,
    fps: float = 1.0,
    max_frames: int = 0
) -> List[Path]:
    """
    Extract frames from entire video at specified fps.

    Args:
        video_path: Path to video file
        output_dir: Directory to save frames
        fps: Frames per second to extract
        max_frames: Maximum frames to extract (0 = no limit)

    Returns:
        List of extracted frame paths
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  ERROR: Could not open video {video_path}")
        return []

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / video_fps if video_fps > 0 else 0

    print(f"  Video: {video_fps:.1f} fps, {total_frames} frames, {video_duration:.1f}s duration")

    # Calculate frame extraction interval
    frame_interval = int(video_fps / fps) if fps > 0 else 1
    frame_interval = max(1, frame_interval)

    extracted_frames = []
    frame_idx = 0
    extracted_count = 0

    pbar = tqdm(total=total_frames // frame_interval, desc="  Extracting", leave=False)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            timestamp = frame_idx / video_fps
            frame_filename = f"frame_{frame_idx:08d}_t{timestamp:.2f}s.jpg"
            frame_path = output_dir / frame_filename

            cv2.imwrite(str(frame_path), frame)
            extracted_frames.append(frame_path)
            extracted_count += 1
            pbar.update(1)

            if max_frames > 0 and extracted_count >= max_frames:
                break

        frame_idx += 1

    pbar.close()
    cap.release()

    return extracted_frames


def load_hands23_predictor(
    config_file: Path = HANDS23_CONFIG,
    model_weights: Path = HANDS23_WEIGHTS,
    hand_thresh: float = 0.7,
    first_obj_thresh: float = 0.5,
    second_obj_thresh: float = 0.3,
    hand_rela: float = 0.3,
    obj_rela: float = 0.7
):
    """Load hands23 detector predictor."""
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor
    from hodetector.data import register_ho_pascal_voc, hoMapper
    from hodetector.modeling import roi_heads

    cfg = get_cfg()
    cfg.merge_from_file(str(config_file))
    cfg.MODEL.WEIGHTS = str(model_weights)

    cfg.HAND = hand_thresh
    cfg.FIRSTOBJ = first_obj_thresh
    cfg.SECONDOBJ = second_obj_thresh
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = min(hand_thresh, first_obj_thresh, second_obj_thresh)

    cfg.HAND_RELA = hand_rela
    cfg.OBJ_RELA = obj_rela

    cfg.freeze()

    return DefaultPredictor(cfg)


def parse_contact(contact: int) -> str:
    return {0: "no_contact", 1: "other_person_contact", 2: "self_contact",
            3: "object_contact", 4: "obj_to_obj_contact"}.get(contact, f"unknown_{contact}")


def parse_grasp(grasp: int) -> str:
    return {0: "NP-Palm", 1: "NP-Fin", 2: "Pow-Pris", 3: "Pre-Pris",
            4: "Pow-Circ", 5: "Pre-Circ", 6: "Later", 7: "Other"}.get(grasp, f"unknown_{grasp}")


def parse_touch(touch: int) -> str:
    return {0: "tool_touched", 1: "tool_held", 2: "tool_used",
            3: "container_touched", 4: "container_held",
            5: "neither_touched", 6: "neither_held"}.get(touch, f"unknown_{touch}")


def run_hands23_detection(predictor, image: np.ndarray) -> List[Dict[str, Any]]:
    """
    Run hands23 detector on an image.

    Returns:
        List of hand detections with bboxes and attributes
    """
    outputs = predictor(image)

    pred_boxes = outputs["instances"].get("pred_boxes").tensor.cpu().numpy()
    pred_dz = outputs["instances"].get("pred_dz").cpu().numpy()
    pred_classes = outputs["instances"].get("pred_classes").cpu().numpy()
    pred_scores = outputs["instances"].get("scores").cpu().numpy()
    pred_masks = outputs["instances"].get("pred_masks").cpu().numpy()

    interaction = pred_dz[:, 4]
    hand_side = pred_dz[:, 5]
    grasp = pred_dz[:, 6]
    touch_type = pred_dz[:, 7]
    contact_state = pred_dz[:, 8]

    hands = []

    for i in range(len(pred_classes)):
        if pred_classes[i] == 0:  # Hand class
            hand_info = {
                "hand_bbox": pred_boxes[i].tolist(),
                "hand_mask": pred_masks[i],
                "hand_side": "right_hand" if hand_side[i] == 1 else "left_hand",
                "contact_state": parse_contact(int(contact_state[i])),
                "grasp": parse_grasp(int(grasp[i])),
                "hand_score": float(pred_scores[i]),
                "obj_bbox": None,
                "obj_mask": None,
                "obj_touch": None,
                "obj_score": None,
                "second_obj_bbox": None,
                "second_obj_mask": None,
                "second_obj_score": None
            }

            if interaction[i] >= 0:
                obj_id = int(interaction[i])
                hand_info["obj_bbox"] = pred_boxes[obj_id].tolist()
                hand_info["obj_mask"] = pred_masks[obj_id]
                hand_info["obj_touch"] = parse_touch(int(touch_type[obj_id]))
                hand_info["obj_score"] = float(pred_scores[obj_id])

                if interaction[obj_id] >= 0:
                    second_obj_id = int(interaction[obj_id])
                    hand_info["second_obj_bbox"] = pred_boxes[second_obj_id].tolist()
                    hand_info["second_obj_mask"] = pred_masks[second_obj_id]
                    hand_info["second_obj_score"] = float(pred_scores[second_obj_id])

            hands.append(hand_info)

    return hands


def draw_bbox(draw: ImageDraw, bbox: List[float], color: tuple, width: int = 3):
    """Draw a bounding box."""
    draw.rectangle(bbox, outline=color, width=width)


def draw_text_with_bg(draw: ImageDraw, pos: tuple, text: str, font: ImageFont,
                      text_color: tuple, bg_color: tuple):
    """Draw text with background."""
    text_bbox = draw.textbbox(pos, text, font=font)
    padding = 4
    draw.rectangle([text_bbox[0] - padding, text_bbox[1] - padding,
                    text_bbox[2] + padding, text_bbox[3] + padding],
                   fill=bg_color)
    draw.text(pos, text, font=font, fill=text_color)


def visualize_detections(
    image: np.ndarray,
    detections: List[Dict[str, Any]],
    output_path: Path,
    font_path: Path = FONT_PATH
) -> Image.Image:
    """
    Visualize hand detections with bounding boxes and labels.

    Args:
        image: Original BGR image
        detections: List of hand detection dicts
        output_path: Path to save visualization
        font_path: Path to font file

    Returns:
        PIL Image with visualizations
    """
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    im = Image.fromarray(image_rgb).convert("RGBA")
    draw = ImageDraw.Draw(im)

    # Load font
    try:
        font = ImageFont.truetype(str(font_path), size=20)
        small_font = ImageFont.truetype(str(font_path), size=16)
    except:
        font = ImageFont.load_default()
        small_font = font

    for hand in detections:
        hand_bbox = hand["hand_bbox"]
        hand_side = hand["hand_side"]
        contact = hand["contact_state"]
        grasp = hand["grasp"]
        hand_color = HAND_COLORS.get(hand_side, (128, 128, 128))

        # Draw hand bbox
        draw_bbox(draw, hand_bbox, hand_color, width=3)

        # Draw hand label
        side_label = "L" if hand_side == "left_hand" else "R"
        if contact == "no_contact":
            label = f"{side_label}: {contact}"
        else:
            label = f"{side_label}: {contact}, {grasp}"

        label_pos = (hand_bbox[0], max(0, hand_bbox[1] - 25))
        draw_text_with_bg(draw, label_pos, label, small_font, (0, 0, 0), (255, 255, 255, 200))

        # Draw first object
        if hand["obj_bbox"] is not None:
            obj_bbox = hand["obj_bbox"]
            draw_bbox(draw, obj_bbox, FIRST_OBJ_COLOR, width=3)

            # Draw line from hand to object
            hand_center = ((hand_bbox[0] + hand_bbox[2]) / 2, (hand_bbox[1] + hand_bbox[3]) / 2)
            obj_center = ((obj_bbox[0] + obj_bbox[2]) / 2, (obj_bbox[1] + obj_bbox[3]) / 2)
            draw.line([hand_center, obj_center], fill=hand_color, width=2)

            # Object label
            obj_label = hand["obj_touch"] or "object"
            obj_label_pos = (obj_bbox[0], max(0, obj_bbox[1] - 25))
            draw_text_with_bg(draw, obj_label_pos, obj_label, small_font, (0, 0, 0), (255, 255, 255, 200))

            # Draw second object
            if hand["second_obj_bbox"] is not None:
                second_bbox = hand["second_obj_bbox"]
                draw_bbox(draw, second_bbox, SECOND_OBJ_COLOR, width=3)

                # Line from first to second object
                second_center = ((second_bbox[0] + second_bbox[2]) / 2, (second_bbox[1] + second_bbox[3]) / 2)
                draw.line([obj_center, second_center], fill=FIRST_OBJ_COLOR, width=2)

    # Convert back to RGB for saving
    im_rgb = im.convert("RGB")
    im_rgb.save(output_path)

    return im


def create_legend(output_path: Path, font_path: Path = FONT_PATH):
    """Create a legend image explaining the visualization colors."""
    width, height = 400, 250
    im = Image.new('RGB', (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(im)

    try:
        font = ImageFont.truetype(str(font_path), size=18)
        title_font = ImageFont.truetype(str(font_path), size=22)
    except:
        font = ImageFont.load_default()
        title_font = font

    # Title
    draw.text((20, 15), "Hands23 Detection Legend", font=title_font, fill=(0, 0, 0))

    y = 60
    items = [
        (HAND_COLORS["left_hand"], "Left Hand"),
        (HAND_COLORS["right_hand"], "Right Hand"),
        (FIRST_OBJ_COLOR, "First Object (held/touched)"),
        (SECOND_OBJ_COLOR, "Second Object"),
    ]

    for color, label in items:
        # Color box
        draw.rectangle([20, y, 50, y + 25], fill=color, outline=(0, 0, 0))
        # Label
        draw.text((65, y + 3), label, font=font, fill=(0, 0, 0))
        y += 40

    im.save(output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Extract frames from full videos and run hands23 detector"
    )
    parser.add_argument('--participant', required=True, help='Participant ID (e.g., P03)')
    parser.add_argument('--output-dir', type=Path, default=OUTPUT_BASE_DIR)
    parser.add_argument('--test-video', type=str, default=None, help='Test on a specific video ID')
    parser.add_argument('--fps', type=float, default=1.0, help='Frame extraction rate (default: 1 fps)')
    parser.add_argument('--max-frames', type=int, default=0, help='Max frames per video (0 = no limit)')
    parser.add_argument('--skip-detection', action='store_true', help='Only extract frames, skip detection')
    parser.add_argument('--no-visualization', action='store_true', help='Skip visualization output')
    parser.add_argument('--verbose', '-v', action='store_true')

    args = parser.parse_args()
    participant = args.participant
    participant_dir = args.output_dir / participant

    # Load timeline annotated data
    timeline_file = participant_dir / f"{participant}_timeline_annotated.json"
    if not timeline_file.exists():
        print(f"ERROR: {timeline_file.name} not found")
        print("       Run 06_timeline_aggregation.py first")
        return 1

    with open(timeline_file, 'r') as f:
        timeline_data = json.load(f)

    # Get videos to process
    all_video_ids = get_videos_from_timeline(timeline_data)
    print(f"Found {len(all_video_ids)} videos in timeline")

    if args.test_video:
        if args.test_video in all_video_ids:
            video_ids = [args.test_video]
            print(f"TEST MODE: Processing only video {args.test_video}")
        else:
            print(f"ERROR: Video {args.test_video} not found in timeline")
            print(f"Available videos: {sorted(all_video_ids)}")
            return 1
    else:
        video_ids = sorted(all_video_ids)

    # Create output directories
    output_root = participant_dir / "hands23_detection"
    output_root.mkdir(parents=True, exist_ok=True)

    # Load hands23 predictor
    predictor = None
    if not args.skip_detection:
        print("\nLoading hands23 detector...")
        if not HANDS23_WEIGHTS.exists():
            print(f"ERROR: Model weights not found at {HANDS23_WEIGHTS}")
            return 1
        try:
            predictor = load_hands23_predictor()
            print("Hands23 detector loaded successfully")
        except Exception as e:
            print(f"ERROR: Failed to load hands23 detector: {e}")
            print("Make sure you're in the 'hands23' conda environment:")
            print("  conda activate hands23")
            return 1

    # Create legend
    if not args.no_visualization:
        legend_path = output_root / "legend.png"
        create_legend(legend_path)
        print(f"Created legend: {legend_path}")

    # Process each video
    print(f"\n{'='*70}")
    print(f"PROCESSING {len(video_ids)} VIDEO(S) AT {args.fps} FPS")
    print(f"{'='*70}")

    all_results = {
        "participant": participant,
        "fps": args.fps,
        "timestamp": datetime.now().isoformat(),
        "videos": []
    }

    total_frames = 0
    total_hands = 0

    for vid_idx, video_id in enumerate(video_ids):
        print(f"\n[{vid_idx+1}/{len(video_ids)}] Video: {video_id}")

        video_path = VIDEO_BASE_DIR / participant / f"{video_id}.mp4"
        if not video_path.exists():
            print(f"  SKIP: Video file not found")
            continue

        # Create video-specific output directories
        video_output_dir = output_root / video_id
        frames_dir = video_output_dir / "frames"
        vis_dir = video_output_dir / "visualizations"

        frames_dir.mkdir(parents=True, exist_ok=True)
        if not args.no_visualization:
            vis_dir.mkdir(parents=True, exist_ok=True)

        # Extract frames
        print(f"  Extracting frames at {args.fps} fps...")
        frame_paths = extract_frames_from_video(
            video_path=video_path,
            output_dir=frames_dir,
            fps=args.fps,
            max_frames=args.max_frames
        )
        print(f"  Extracted {len(frame_paths)} frames")
        total_frames += len(frame_paths)

        video_result = {
            "video_id": video_id,
            "num_frames": len(frame_paths),
            "frames": []
        }

        # Run detection on each frame
        if predictor is not None and frame_paths:
            print(f"  Running hands23 detection...")
            hands_in_video = 0

            for frame_path in tqdm(frame_paths, desc="  Detecting", leave=False):
                image = cv2.imread(str(frame_path))
                if image is None:
                    continue

                detections = run_hands23_detection(predictor, image)
                num_hands = len(detections)
                hands_in_video += num_hands

                # Extract timestamp from filename
                timestamp = float(frame_path.stem.split('_t')[1].rstrip('s'))

                # Prepare serializable detection data (remove masks)
                serializable_detections = []
                for det in detections:
                    det_copy = {k: v for k, v in det.items()
                                if not k.endswith('_mask')}
                    serializable_detections.append(det_copy)

                frame_result = {
                    "frame_path": str(frame_path.relative_to(output_root)),
                    "timestamp": timestamp,
                    "num_hands": num_hands,
                    "detections": serializable_detections
                }

                # Visualize
                if not args.no_visualization and detections:
                    vis_path = vis_dir / f"vis_{frame_path.name}"
                    visualize_detections(image, detections, vis_path)
                    frame_result["visualization_path"] = str(vis_path.relative_to(output_root))

                video_result["frames"].append(frame_result)

            print(f"  Detected {hands_in_video} hands total")
            total_hands += hands_in_video
            video_result["total_hands"] = hands_in_video

        all_results["videos"].append(video_result)

    # Save results
    print(f"\n{'='*70}")
    print(f"SAVING RESULTS")
    print(f"{'='*70}")

    results_file = output_root / f"{participant}_hands23_results.json"
    all_results["total_frames"] = total_frames
    all_results["total_hands"] = total_hands

    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2)

    print(f"  Results: {results_file}")
    print(f"  Total frames: {total_frames}")
    print(f"  Total hands detected: {total_hands}")

    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"Videos processed: {len(video_ids)}")
    print(f"Total frames extracted: {total_frames}")
    if predictor is not None:
        print(f"Total hands detected: {total_hands}")
        print(f"Avg hands per frame: {total_hands / max(1, total_frames):.2f}")
    print(f"\nOutput directory: {output_root}")


if __name__ == '__main__':
    main()

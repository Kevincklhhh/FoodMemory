#!/usr/bin/env python3
"""
Visualize hand-object interaction detections using official Hands23 vis_utils.

This script generates annotated images using the official visualization utilities
from the Hands23 detector, including segmentation masks and proper color coding.
"""

import os
import sys
import cv2
import json
import argparse
import shutil
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Add hands23_detector to path
hands_detector_path = os.path.join(os.path.dirname(__file__), 'hands23_detector')
sys.path.insert(0, hands_detector_path)

from hands23_detector.utils.vis_utils import vis_per_image


# Mapping from integer codes to string labels
CONTACT_STATE_MAP = {
    0: "no_contact",
    1: "other_person_contact",
    2: "self_contact",
    3: "object_contact",
    4: "obj_to_obj_contact"
}

GRASP_MAP = {
    0: "NP-Palm",
    1: "NP-Fin",
    2: "Pow-Pris",
    3: "Pre-Pris",
    4: "Pow-Circ",
    5: "Pre-Circ",
    6: "Later",
    7: "Other"
}

TOUCH_MAP = {
    0: "tool_,_touched",
    1: "tool_,_held",
    2: "tool_,_used",
    3: "container_,_touched",
    4: "container_,_held",
    5: "neither_,_touched",
    6: "neither_,_held"
}


def convert_detection_to_pred_format(detection):
    """
    Convert detection JSON format to format expected by vis_utils.

    Args:
        detection: Detection dict from JSON

    Returns:
        list: Predictions in vis_utils format
    """
    preds = []

    for hand in detection["hands"]:
        pred = {
            "hand_bbox": hand["hand_bbox"],
            "hand_side": hand["hand_side"],
            "contact_state": CONTACT_STATE_MAP.get(hand["contact_state"], "no_contact"),
            "grasp": GRASP_MAP.get(hand["grasp"], "Other"),
            "obj_bbox": hand["obj_bbox"],
            "obj_touch": TOUCH_MAP.get(hand["obj_touch"], "None") if hand["obj_touch"] is not None else "None",
            "second_obj_bbox": hand["second_obj_bbox"]
        }
        preds.append(pred)

    return preds


def visualize_frame(frame_data, frames_dir, masks_dir, output_dir, font_path, use_simple=False):
    """
    Visualize detections for a single frame using official vis_utils.

    Args:
        frame_data: Detection data for frame
        frames_dir: Directory containing frame images
        masks_dir: Directory containing segmentation masks
        output_dir: Directory to save visualization
        font_path: Path to font file for labels
        use_simple: Use simple visualization without labels

    Returns:
        bool: Success status
    """
    filename = frame_data["filename"]
    detection = frame_data["detection"]

    # Load image
    image_path = os.path.join(frames_dir, filename)
    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: Could not read {image_path}")
        return False

    # Check if there are any hands detected
    if detection["num_hands"] == 0:
        # No detections - just copy the original frame
        output_path = os.path.join(output_dir, filename)
        shutil.copy2(image_path, output_path)
        return True

    # Convert detection to format expected by vis_utils
    preds = convert_detection_to_pred_format(detection)

    # Visualize using official utils
    vis_image = vis_per_image(
        im=image,
        preds=preds,
        filename=filename,
        masks_dir=masks_dir,
        font_path=font_path,
        use_simple=use_simple
    )

    # Convert back to BGR and save
    vis_image = vis_image.convert("RGB")
    vis_array = np.array(vis_image)
    vis_array = cv2.cvtColor(vis_array, cv2.COLOR_RGB2BGR)

    output_path = os.path.join(output_dir, filename)
    cv2.imwrite(output_path, vis_array)

    return True


def visualize_all_detections(detection_file, frames_dir, masks_dir, output_dir,
                             font_path=None, use_simple=False):
    """
    Generate visualizations for all frames using official vis_utils.

    Args:
        detection_file: Path to detection results JSON
        frames_dir: Directory containing frame images
        masks_dir: Directory containing segmentation masks
        output_dir: Directory to save visualizations
        font_path: Path to font file (defaults to hands23_detector/utils/times_b.ttf)
        use_simple: Use simple visualization without labels

    Returns:
        dict: Statistics about visualization generation
    """
    # Default font path
    if font_path is None:
        font_path = os.path.join(
            os.path.dirname(__file__),
            'hands23_detector/utils/times_b.ttf'
        )

    # Check if font exists
    if not os.path.exists(font_path):
        print(f"Warning: Font file not found at {font_path}")
        print("Visualization may fail or use default font")

    # Load detections
    print(f"Loading detections from {detection_file}")
    with open(detection_file, 'r') as f:
        data = json.load(f)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving visualizations to {output_dir}")

    # Process all frames
    frames_to_visualize = data["frames"]
    print(f"Visualizing all {len(frames_to_visualize)} frames")

    # Generate visualizations
    success_count = 0
    frames_with_detection = 0
    frames_without_detection = 0

    for frame_data in tqdm(frames_to_visualize, desc="Generating visualizations"):
        if visualize_frame(frame_data, frames_dir, masks_dir, output_dir, font_path, use_simple):
            success_count += 1
            if frame_data["detection"]["num_hands"] > 0:
                frames_with_detection += 1
            else:
                frames_without_detection += 1

    # Statistics
    stats = {
        "total_frames": len(frames_to_visualize),
        "successful": success_count,
        "failed": len(frames_to_visualize) - success_count,
        "frames_with_detection": frames_with_detection,
        "frames_without_detection": frames_without_detection
    }

    print(f"\nVisualization complete!")
    print(f"Successfully generated: {success_count}/{len(frames_to_visualize)}")
    print(f"  - Frames with detections: {frames_with_detection}")
    print(f"  - Frames without detections: {frames_without_detection}")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Visualize hand-object interactions using official Hands23 vis_utils"
    )
    parser.add_argument(
        "--detection_file",
        required=True,
        help="Path to detection results JSON file"
    )
    parser.add_argument(
        "--frames_dir",
        required=True,
        help="Directory containing extracted frames"
    )
    parser.add_argument(
        "--masks_dir",
        required=True,
        help="Directory containing segmentation masks"
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory to save visualizations"
    )
    parser.add_argument(
        "--font_path",
        default=None,
        help="Path to font file for labels (default: hands23_detector/utils/times_b.ttf)"
    )
    parser.add_argument(
        "--use_simple",
        action="store_true",
        help="Use simple visualization without labels"
    )

    args = parser.parse_args()

    visualize_all_detections(
        detection_file=args.detection_file,
        frames_dir=args.frames_dir,
        masks_dir=args.masks_dir,
        output_dir=args.output_dir,
        font_path=args.font_path,
        use_simple=args.use_simple
    )


if __name__ == "__main__":
    main()

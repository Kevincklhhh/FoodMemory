#!/usr/bin/env python3
"""
Visualize hand-object interaction detections on frames.

This script generates annotated images showing detected hands, objects,
and their interactions with bounding boxes and labels.
"""

import os
import cv2
import json
import argparse
from pathlib import Path
from tqdm import tqdm


# Color scheme for visualization
COLORS = {
    "hand": (0, 255, 0),        # Green for hands
    "obj": (255, 0, 0),          # Blue for first object
    "second_obj": (0, 165, 255), # Orange for second object
    "text": (255, 255, 255),     # White for text
    "bg": (0, 0, 0)              # Black for text background
}

# Grasp type labels
GRASP_LABELS = {
    0: "NP-Palm",
    1: "NP-Fin",
    2: "Pow-Pris",
    3: "Pre-Pris",
    4: "Pow-Circ",
    5: "Pre-Circ",
    6: "Later",
    7: "Other"
}

# Touch type labels
TOUCH_LABELS = {
    0: "Tool-Touch",
    1: "Tool-Held",
    2: "Tool-Used",
    3: "Cont-Touch",
    4: "Cont-Held",
    5: "Neither-Touch",
    6: "Neither-Held"
}

# Contact state labels
CONTACT_STATE_LABELS = {
    0: "No-Contact",
    1: "Self-Contact",
    2: "Obj-Contact",
    3: "Person-Contact",
    4: "Obj-to-Obj"
}


def draw_bbox(image, bbox, color, thickness=2):
    """Draw bounding box on image."""
    x1, y1, x2, y2 = [int(coord) for coord in bbox]
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    return (x1, y1)


def draw_label(image, text, position, color, bg_color):
    """Draw text label with background."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1

    # Get text size
    (text_width, text_height), baseline = cv2.getTextSize(
        text, font, font_scale, thickness
    )

    x, y = position
    # Draw background rectangle
    cv2.rectangle(
        image,
        (x, y - text_height - baseline - 2),
        (x + text_width + 2, y + 2),
        bg_color,
        -1
    )

    # Draw text
    cv2.putText(
        image,
        text,
        (x + 1, y),
        font,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA
    )

    return y + text_height + baseline + 5


def visualize_frame(frame_data, frames_dir, output_dir):
    """
    Visualize detections for a single frame.

    Args:
        frame_data: Detection data for frame
        frames_dir: Directory containing frame images
        output_dir: Directory to save visualization

    Returns:
        bool: Success status
    """
    filename = frame_data["filename"]
    detection = frame_data["detection"]
    timestamp = frame_data["timestamp"]

    # Load image
    image_path = os.path.join(frames_dir, filename)
    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: Could not read {image_path}")
        return False

    # Create output with alpha channel for better visualization
    h, w = image.shape[:2]

    # Draw timestamp
    timestamp_text = f"Time: {timestamp:.2f}s" if timestamp is not None else "Time: N/A"
    draw_label(image, timestamp_text, (10, 30), COLORS["text"], COLORS["bg"])

    # Draw interaction status
    status_text = f"Interactions: {detection['num_hands']} hands"
    if detection["has_interaction"]:
        status_text += " (with objects)"
    draw_label(image, status_text, (10, 60), COLORS["text"], COLORS["bg"])

    # Draw each hand and its interactions
    for hand in detection["hands"]:
        # Draw hand bounding box
        hand_pos = draw_bbox(image, hand["hand_bbox"], COLORS["hand"], 3)

        # Hand label
        hand_label = f"{hand['hand_side']}"
        contact_state = CONTACT_STATE_LABELS.get(hand["contact_state"], "Unknown")
        grasp = GRASP_LABELS.get(hand["grasp"], "Unknown")

        label_y = hand_pos[1]
        label_y = draw_label(
            image,
            hand_label,
            (hand_pos[0], label_y),
            COLORS["hand"],
            COLORS["bg"]
        )

        label_y = draw_label(
            image,
            f"Contact: {contact_state}",
            (hand_pos[0], label_y),
            COLORS["hand"],
            COLORS["bg"]
        )

        label_y = draw_label(
            image,
            f"Grasp: {grasp}",
            (hand_pos[0], label_y),
            COLORS["hand"],
            COLORS["bg"]
        )

        # Draw first object if there's interaction
        if hand["has_interaction"] and hand["obj_bbox"] is not None:
            obj_pos = draw_bbox(image, hand["obj_bbox"], COLORS["obj"], 2)

            # Object label
            touch_type = TOUCH_LABELS.get(hand["obj_touch"], "Unknown")
            obj_label = f"Object (touch: {touch_type})"

            draw_label(
                image,
                obj_label,
                (obj_pos[0], obj_pos[1]),
                COLORS["obj"],
                COLORS["bg"]
            )

            # Draw connection line from hand to object
            hand_center = (
                int((hand["hand_bbox"][0] + hand["hand_bbox"][2]) / 2),
                int((hand["hand_bbox"][1] + hand["hand_bbox"][3]) / 2)
            )
            obj_center = (
                int((hand["obj_bbox"][0] + hand["obj_bbox"][2]) / 2),
                int((hand["obj_bbox"][1] + hand["obj_bbox"][3]) / 2)
            )
            cv2.line(image, hand_center, obj_center, COLORS["obj"], 1, cv2.LINE_AA)

            # Draw second object if present
            if hand["second_obj_bbox"] is not None:
                second_obj_pos = draw_bbox(
                    image,
                    hand["second_obj_bbox"],
                    COLORS["second_obj"],
                    2
                )

                draw_label(
                    image,
                    "Second Object",
                    (second_obj_pos[0], second_obj_pos[1]),
                    COLORS["second_obj"],
                    COLORS["bg"]
                )

                # Draw connection line from object to second object
                second_obj_center = (
                    int((hand["second_obj_bbox"][0] + hand["second_obj_bbox"][2]) / 2),
                    int((hand["second_obj_bbox"][1] + hand["second_obj_bbox"][3]) / 2)
                )
                cv2.line(
                    image,
                    obj_center,
                    second_obj_center,
                    COLORS["second_obj"],
                    1,
                    cv2.LINE_AA
                )

    # Save visualization
    output_path = os.path.join(output_dir, filename)
    cv2.imwrite(output_path, image)

    return True


def visualize_all_detections(detection_file, frames_dir, output_dir,
                             only_interactions=False):
    """
    Generate visualizations for all detected frames.

    Args:
        detection_file: Path to detection results JSON
        frames_dir: Directory containing frame images
        output_dir: Directory to save visualizations
        only_interactions: If True, only visualize frames with interactions

    Returns:
        dict: Statistics about visualization generation
    """
    # Load detections
    print(f"Loading detections from {detection_file}")
    with open(detection_file, 'r') as f:
        data = json.load(f)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving visualizations to {output_dir}")

    # Filter frames if needed
    frames_to_visualize = data["frames"]
    if only_interactions:
        frames_to_visualize = [
            f for f in frames_to_visualize
            if f["detection"]["has_interaction"]
        ]
        print(f"Visualizing {len(frames_to_visualize)} frames with interactions")
    else:
        print(f"Visualizing all {len(frames_to_visualize)} frames")

    # Generate visualizations
    success_count = 0
    for frame_data in tqdm(frames_to_visualize, desc="Generating visualizations"):
        if visualize_frame(frame_data, frames_dir, output_dir):
            success_count += 1

    # Statistics
    stats = {
        "total_frames": len(frames_to_visualize),
        "successful": success_count,
        "failed": len(frames_to_visualize) - success_count
    }

    print(f"\nVisualization complete!")
    print(f"Successfully generated: {success_count}/{len(frames_to_visualize)}")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Visualize hand-object interaction detections"
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
        "--output_dir",
        required=True,
        help="Directory to save visualizations"
    )
    parser.add_argument(
        "--only_interactions",
        action="store_true",
        help="Only visualize frames with hand-object interactions"
    )

    args = parser.parse_args()

    visualize_all_detections(
        detection_file=args.detection_file,
        frames_dir=args.frames_dir,
        output_dir=args.output_dir,
        only_interactions=args.only_interactions
    )


if __name__ == "__main__":
    main()

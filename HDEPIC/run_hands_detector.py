#!/usr/bin/env python3
"""
Run Hands23 detector on extracted video frames.

This script processes frames extracted from videos and runs the hand-object
interaction detector on them, outputting structured predictions.
"""

import os
import sys
import torch
import cv2
import argparse
import json
import glob
from pathlib import Path
from tqdm import tqdm

# Add hands23_detector to path
hands_detector_path = os.path.join(os.path.dirname(__file__), 'hands23_detector')
sys.path.insert(0, hands_detector_path)

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from hodetector.data import register_ho_pascal_voc, hoMapper
from hodetector.modeling import roi_heads


def parse_timestamp_from_filename(filename):
    """
    Extract timestamp from frame filename.
    Expected format: video_id_frame_XXXXX_ts_XX.XX.jpg

    Args:
        filename: Frame filename

    Returns:
        float: Timestamp in seconds
    """
    try:
        # Split by '_ts_' to get timestamp part
        parts = filename.split('_ts_')
        if len(parts) < 2:
            return None
        # Get timestamp (remove .jpg extension)
        ts_str = parts[1].replace('.jpg', '')
        return float(ts_str)
    except Exception as e:
        print(f"Warning: Could not parse timestamp from {filename}: {e}")
        return None


class HandsDetector:
    """Wrapper for Hands23 detector"""

    def __init__(self, config_file, model_weights, hand_thresh=0.7,
                 first_obj_thresh=0.5, second_obj_thresh=0.3,
                 hand_rela=0.3, obj_rela=0.7):
        """Initialize detector with configuration"""

        cfg = get_cfg()
        cfg.merge_from_file(config_file)
        cfg.MODEL.WEIGHTS = model_weights

        # Set thresholds
        cfg.HAND = hand_thresh
        cfg.FIRSTOBJ = first_obj_thresh
        cfg.SECONDOBJ = second_obj_thresh
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = min(hand_thresh, first_obj_thresh, second_obj_thresh)
        cfg.HAND_RELA = hand_rela
        cfg.OBJ_RELA = obj_rela

        cfg.freeze()

        self.predictor = DefaultPredictor(cfg)

    def detect(self, image):
        """
        Run detection on image.

        Args:
            image: numpy array (BGR format)

        Returns:
            dict: Detection results
        """
        outputs = self.predictor(image)
        return self._parse_outputs(outputs)

    def _parse_outputs(self, outputs):
        """Parse detector outputs into structured format"""

        pred_boxes = outputs["instances"].get("pred_boxes").tensor.to("cpu").detach().numpy()
        pred_dz = outputs["instances"].get("pred_dz").to("cpu").detach().numpy()
        pred_classes = outputs["instances"].get("pred_classes").to("cpu").detach().numpy()
        pred_scores = outputs["instances"].get("scores").to("cpu").detach().numpy()

        interaction = torch.tensor(pred_dz[:, 4])
        hand_side = torch.tensor(pred_dz[:, 5])
        grasp = torch.tensor(pred_dz[:, 6])
        touch_type = torch.tensor(pred_dz[:, 7])
        contact_state = torch.tensor(pred_dz[:, 8])

        grasp_scores = torch.tensor(pred_dz[:, 10:18])
        touch_scores = torch.tensor(pred_dz[:, 18:25])

        hands = []
        hand_count = 0

        for i in range(len(pred_classes)):
            if pred_classes[i] == 0:  # Hand class
                hand_info = {
                    "hand_id": hand_count,
                    "hand_bbox": pred_boxes[i].tolist(),
                    "hand_side": "right_hand" if hand_side[i].item() == 1 else "left_hand",
                    "contact_state": int(contact_state[i].item()),
                    "grasp": int(grasp[i].item()),
                    "hand_score": float(pred_scores[i]),
                    "has_interaction": False,
                    "obj_bbox": None,
                    "obj_touch": None,
                    "obj_score": None,
                    "second_obj_bbox": None,
                    "second_obj_score": None
                }

                # Check for first object interaction
                if interaction[i] >= 0:
                    obj_id = int(interaction[i])
                    hand_info["has_interaction"] = True
                    hand_info["obj_bbox"] = pred_boxes[obj_id].tolist()
                    hand_info["obj_touch"] = int(touch_type[obj_id].item())
                    hand_info["obj_score"] = float(pred_scores[obj_id])

                    # Check for second object
                    if interaction[obj_id] >= 0:
                        second_obj_id = int(interaction[obj_id])
                        hand_info["second_obj_bbox"] = pred_boxes[second_obj_id].tolist()
                        hand_info["second_obj_score"] = float(pred_scores[second_obj_id])

                hands.append(hand_info)
                hand_count += 1

        return {
            "num_hands": hand_count,
            "hands": hands,
            "has_interaction": any(h["has_interaction"] for h in hands)
        }


def process_frames(frames_dir, output_file, config_file, model_weights,
                   hand_thresh=0.7, first_obj_thresh=0.5, second_obj_thresh=0.3,
                   hand_rela=0.3, obj_rela=0.7):
    """
    Process all frames in directory with hands detector.

    Args:
        frames_dir: Directory containing extracted frames
        output_file: Path to save detection results (JSON)
        config_file: Detectron2 config file
        model_weights: Path to model weights
        hand_thresh, first_obj_thresh, second_obj_thresh: Detection thresholds
        hand_rela, obj_rela: Interaction thresholds

    Returns:
        dict: Detection results for all frames
    """

    print("Initializing Hands23 detector...")
    detector = HandsDetector(
        config_file=config_file,
        model_weights=model_weights,
        hand_thresh=hand_thresh,
        first_obj_thresh=first_obj_thresh,
        second_obj_thresh=second_obj_thresh,
        hand_rela=hand_rela,
        obj_rela=obj_rela
    )

    # Load frame metadata
    metadata_path = os.path.join(frames_dir, "frame_metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        print(f"Loaded metadata for {len(metadata['frames'])} frames")
    else:
        print("Warning: No metadata file found, will process all images")
        metadata = None

    # Get list of frames
    frame_files = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
    print(f"Found {len(frame_files)} frames to process")

    results = {
        "frames_dir": frames_dir,
        "config": {
            "hand_thresh": hand_thresh,
            "first_obj_thresh": first_obj_thresh,
            "second_obj_thresh": second_obj_thresh,
            "hand_rela": hand_rela,
            "obj_rela": obj_rela
        },
        "frames": []
    }

    # Process each frame
    for frame_path in tqdm(frame_files, desc="Processing frames"):
        filename = os.path.basename(frame_path)

        # Read image
        image = cv2.imread(frame_path)
        if image is None:
            print(f"Warning: Could not read {frame_path}")
            continue

        # Parse timestamp from filename
        timestamp = parse_timestamp_from_filename(filename)

        # Run detection
        detection = detector.detect(image)

        # Store result
        frame_result = {
            "filename": filename,
            "timestamp": timestamp,
            "detection": detection
        }

        results["frames"].append(frame_result)

    # Save results
    print(f"\nSaving results to {output_file}")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary
    total_frames = len(results["frames"])
    frames_with_hands = sum(1 for f in results["frames"] if f["detection"]["num_hands"] > 0)
    frames_with_interaction = sum(1 for f in results["frames"] if f["detection"]["has_interaction"])

    print(f"\n{'='*50}")
    print(f"Detection Summary:")
    print(f"{'='*50}")
    print(f"Total frames processed: {total_frames}")
    print(f"Frames with hands detected: {frames_with_hands} ({frames_with_hands/total_frames*100:.1f}%)")
    print(f"Frames with hand-object interaction: {frames_with_interaction} ({frames_with_interaction/total_frames*100:.1f}%)")
    print(f"{'='*50}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run Hands23 detector on extracted video frames"
    )
    parser.add_argument(
        "--frames_dir",
        required=True,
        help="Directory containing extracted frames"
    )
    parser.add_argument(
        "--output_file",
        required=True,
        help="Path to save detection results (JSON)"
    )
    parser.add_argument(
        "--config_file",
        default="hands23_detector/faster_rcnn_X_101_32x8d_FPN_3x_Hands23.yaml",
        help="Detectron2 config file"
    )
    parser.add_argument(
        "--model_weights",
        default="hands23_detector/model_weights/model_hands23.pth",
        help="Path to model weights"
    )
    parser.add_argument(
        "--hand_thresh",
        type=float,
        default=0.7,
        help="Hand detection threshold"
    )
    parser.add_argument(
        "--first_obj_thresh",
        type=float,
        default=0.5,
        help="First object detection threshold"
    )
    parser.add_argument(
        "--second_obj_thresh",
        type=float,
        default=0.3,
        help="Second object detection threshold"
    )
    parser.add_argument(
        "--hand_rela",
        type=float,
        default=0.3,
        help="Hand-object interaction threshold"
    )
    parser.add_argument(
        "--obj_rela",
        type=float,
        default=0.7,
        help="Object-object interaction threshold"
    )

    args = parser.parse_args()

    process_frames(
        frames_dir=args.frames_dir,
        output_file=args.output_file,
        config_file=args.config_file,
        model_weights=args.model_weights,
        hand_thresh=args.hand_thresh,
        first_obj_thresh=args.first_obj_thresh,
        second_obj_thresh=args.second_obj_thresh,
        hand_rela=args.hand_rela,
        obj_rela=args.obj_rela
    )


if __name__ == "__main__":
    main()

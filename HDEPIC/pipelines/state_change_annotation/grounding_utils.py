#!/usr/bin/env python3
"""
Grounding utilities for food state change annotation.

Supports multiple grounding methods:
1. GroundingSAM - Text-prompted object grounding
2. Hands23 - Hand-object interaction detection
"""

import sys
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import cv2
import numpy as np

# Add hands23_detector to path
HANDS23_PATH = Path(__file__).parent.parent.parent / "models" / "hands23_detector"
sys.path.insert(0, str(HANDS23_PATH))

# Add detectron2 from hands23 directory (source code)
DETECTRON2_PATH = HANDS23_PATH / "detectron2"
if DETECTRON2_PATH.exists():
    sys.path.insert(0, str(DETECTRON2_PATH))


def run_hands23_detector(image_path: str, config_file: str, model_weights: str) -> Dict:
    """Run Hands23 detector on an image

    The Hands23 detector identifies hand-object interactions in egocentric videos.
    It detects hands and objects that are being directly manipulated.

    Interaction Levels Explained:
    - "first": Primary object directly interacted with by hand
              (e.g., hand grasping a knife, hand holding an onion)
    - "second": Secondary object interacted with by the first object
               (e.g., onion being cut by the knife held in hand)

    This hierarchical detection helps identify:
    - Direct hand-food contact (first level)
    - Tool-mediated food interactions (second level)

    Class Mapping:
    - Class 0: Hand
    - Class 1: First-level object (directly touched by hand)
    - Class 2: Second-level object (interacted with via first object)

    Args:
        image_path: Path to input image
        config_file: Path to detector config
        model_weights: Path to model weights

    Returns:
        Dictionary containing:
        - hands: List of hand detections with scores/bboxes
        - objects: List of interacted object detections with interaction_level
        - masks: Dictionary mapping object IDs to binary mask arrays
        - image_shape: (height, width) of input image
    """
    try:
        from detectron2.config import get_cfg
        from detectron2.engine import DefaultPredictor
        import torch

        # Import hands23 custom modules to register them
        import hodetector.modeling.roi_heads  # This registers custom ROI heads

        # Setup config
        cfg = get_cfg()
        cfg.merge_from_file(config_file)
        cfg.MODEL.WEIGHTS = model_weights

        # Set Hands23-specific thresholds (required by the model)
        # Using lower thresholds to capture more potential food objects
        cfg.HAND = 0.5  # Hand detection threshold (lowered from 0.7)
        cfg.FIRSTOBJ = 0.3  # First object threshold (lowered from 0.5)
        cfg.SECONDOBJ = 0.2  # Second object threshold (lowered from 0.3)
        cfg.HAND_RELA = 0.3  # Hand relationship threshold
        cfg.OBJ_RELA = 0.5  # Object relationship threshold (lowered from 0.7)

        # Set the minimum threshold for ROI heads
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = min(cfg.HAND, cfg.FIRSTOBJ, cfg.SECONDOBJ)
        cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

        # Create predictor
        predictor = DefaultPredictor(cfg)

        # Load and run on image
        image = cv2.imread(image_path)
        outputs = predictor(image)

        # Extract predictions
        instances = outputs["instances"].to("cpu")

        # Parse detections
        hands = []
        objects = []
        masks_dict = {}

        if len(instances) > 0:
            boxes = instances.pred_boxes.tensor.numpy()
            scores = instances.scores.numpy()
            classes = instances.pred_classes.numpy()

            # Check if masks are available
            has_masks = instances.has("pred_masks")
            if has_masks:
                masks = instances.pred_masks.numpy()

            for i in range(len(instances)):
                det = {
                    'bbox': boxes[i].tolist(),
                    'score': float(scores[i]),
                    'class_id': int(classes[i])
                }

                # Store mask if available
                if has_masks:
                    mask = masks[i]
                    masks_dict[i] = mask
                    det['mask_id'] = i

                # Class 0 = hand, Class 1 = first object, Class 2 = second object (based on actual model outputs)
                # The demo.py uses different logic - it processes all as potential hands/objects based on interaction links
                if classes[i] == 0:
                    det['type'] = 'hand'
                    hands.append(det)
                elif classes[i] in [1, 2]:
                    det['type'] = 'object'
                    det['interaction_level'] = 'first' if classes[i] == 1 else 'second'
                    objects.append(det)

        return {
            'hands': hands,
            'objects': objects,
            'masks': masks_dict,
            'image_shape': image.shape[:2]
        }

    except Exception as e:
        print(f"Error running Hands23 detector: {e}")
        return {'hands': [], 'objects': [], 'masks': {}, 'image_shape': None}


def save_mask_image(mask: np.ndarray, output_path: str, alpha: float = 0.5):
    """Save a binary mask as a semi-transparent image

    Args:
        mask: Binary mask array (H, W)
        output_path: Where to save the mask visualization
        alpha: Transparency level
    """
    # Create RGBA image
    h, w = mask.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)

    # Set mask region to white with alpha
    rgba[mask > 0] = [255, 255, 255, int(255 * alpha)]

    cv2.imwrite(output_path, rgba)


def try_groundingsam(image_path: str, text_prompt: str) -> Optional[Dict]:
    """Attempt to run GroundingSAM if available

    Args:
        image_path: Path to input image
        text_prompt: Text description of object to ground (e.g., "lemon")

    Returns:
        Detection dictionary or None if not available
    """
    try:
        # Try to import GroundingSAM
        # This is a placeholder - actual implementation depends on how GroundingSAM is installed
        import importlib.util

        # Check if groundingdino is available
        groundingdino_spec = importlib.util.find_spec("groundingdino")
        sam_spec = importlib.util.find_spec("segment_anything")

        if groundingdino_spec is None or sam_spec is None:
            return None

        # If available, implement GroundingSAM inference here
        # For now, return None to indicate it's not implemented yet
        return None

    except Exception as e:
        return None


def run_grounding_on_frame(
    frame_path: str,
    food_noun: str,
    output_dir: Path,
    hands23_config: Optional[str] = None,
    hands23_weights: Optional[str] = None
) -> List[Dict]:
    """Run all available grounding methods on a frame

    Args:
        frame_path: Path to frame image
        food_noun: Target food noun to ground
        output_dir: Directory to save generated masks
        hands23_config: Path to Hands23 config file
        hands23_weights: Path to Hands23 model weights

    Returns:
        List of grounding results with masks
    """
    results = []
    frame_name = Path(frame_path).stem
    output_dir.mkdir(parents=True, exist_ok=True)

    # Method 1: Try GroundingSAM
    groundingsam_result = try_groundingsam(frame_path, food_noun)
    if groundingsam_result:
        # Save mask and add to results
        mask_path = output_dir / f"{frame_name}_groundingsam_{food_noun}.png"
        save_mask_image(groundingsam_result['mask'], str(mask_path))

        results.append({
            'method': 'groundingsam',
            'food_noun': food_noun,
            'frame_path': frame_path,
            'mask_path': str(mask_path),
            'confidence': groundingsam_result.get('score', 0.0),
            'bbox': groundingsam_result.get('bbox', None)
        })

    # Method 2: Hands23 detector
    if hands23_config and hands23_weights:
        hands23_result = run_hands23_detector(frame_path, hands23_config, hands23_weights)

        # Save masks for detected objects
        for obj_idx, obj in enumerate(hands23_result['objects']):
            if 'mask_id' in obj and obj['mask_id'] in hands23_result['masks']:
                mask = hands23_result['masks'][obj['mask_id']]
                mask_path = output_dir / f"{frame_name}_hands23_obj{obj_idx}.png"
                save_mask_image(mask, str(mask_path))

                results.append({
                    'method': 'hands23',
                    'food_noun': food_noun,  # Note: Hands23 doesn't know the specific food, just detects interacted objects
                    'frame_path': frame_path,
                    'mask_path': str(mask_path),
                    'confidence': obj['score'],
                    'bbox': obj['bbox'],
                    'interaction_level': obj.get('interaction_level', 'unknown')
                })

    return results


def run_grounding_on_block(
    task: Dict,
    output_base_dir: Path,
    hands23_config: Optional[str] = None,
    hands23_weights: Optional[str] = None
) -> Dict:
    """Run grounding on all frames in an annotation task block

    Args:
        task: Annotation task dictionary
        output_base_dir: Base output directory
        hands23_config: Path to Hands23 config
        hands23_weights: Path to Hands23 weights

    Returns:
        Updated auto_grounding_data dictionary
    """
    grounding_data = {}

    # Create mask output directory for this block
    task_id = task['task_id']
    mask_dir = output_base_dir / "assets" / task['video_id'] / task_id.split('_')[-1] / "masks"

    # Process each food noun
    for food_noun in task['target_food_nouns']:
        grounding_data[food_noun] = {
            'before_frames': [],
            'after_frames': []
        }

        # Process before frames
        for frame_info in task['assets']['before_frames']:
            frame_path = output_base_dir / frame_info['path']
            if frame_path.exists():
                results = run_grounding_on_frame(
                    str(frame_path),
                    food_noun,
                    mask_dir,
                    hands23_config,
                    hands23_weights
                )

                for result in results:
                    # Make mask path relative to output_base_dir
                    rel_mask_path = Path(result['mask_path']).relative_to(output_base_dir)
                    result['mask_path'] = str(rel_mask_path)
                    result['frame_info'] = frame_info
                    grounding_data[food_noun]['before_frames'].append(result)

        # Process after frames
        for frame_info in task['assets']['after_frames']:
            frame_path = output_base_dir / frame_info['path']
            if frame_path.exists():
                results = run_grounding_on_frame(
                    str(frame_path),
                    food_noun,
                    mask_dir,
                    hands23_config,
                    hands23_weights
                )

                for result in results:
                    rel_mask_path = Path(result['mask_path']).relative_to(output_base_dir)
                    result['mask_path'] = str(rel_mask_path)
                    result['frame_info'] = frame_info
                    grounding_data[food_noun]['after_frames'].append(result)

    return grounding_data

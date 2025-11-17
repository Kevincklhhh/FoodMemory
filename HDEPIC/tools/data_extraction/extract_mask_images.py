#!/usr/bin/env python3
"""
Extract bounding box images from HD-EPIC videos based on p01_objects_list.json

Output files are named with unique identifiers that can be traced back to the JSON:
  {video_id}_{object_id}_{mask_id}_{frame}.jpg
"""

import json
import cv2
import numpy as np
from pathlib import Path
import argparse
import random


def parse_food_ids_file(file_path):
    """
    Parse food_objectIDs.txt file.

    Format: object_name | object_id | video_id separated by #

    Returns:
        List of dicts with 'object_name', 'object_id', 'video_id'
    """
    with open(file_path, 'r') as f:
        content = f.read()

    # Split by # to get individual entries
    entries = content.split('#')

    food_objects = []
    for entry in entries:
        entry = entry.strip()
        if not entry:
            continue

        # Split by | to get fields
        parts = [p.strip() for p in entry.split('|')]
        if len(parts) >= 3:
            food_objects.append({
                'object_name': parts[0],
                'object_id': parts[1],
                'video_id': parts[2]
            })

    return food_objects


def extract_mask_images(
    objects_json='p01_objects_list.json',
    video_dir='HD-EPIC/Videos/P01',
    output_dir='extracted_masks',
    num_samples=10,
    seed=42,
    food_ids_file=None,
    verbose=True
):
    """
    Extract bounding box images from videos based on object annotations.

    Args:
        objects_json: Path to p01_objects_list.json
        video_dir: Directory containing P01 videos
        output_dir: Output directory for extracted images
        num_samples: Number of masks to extract (ignored if food_ids_file is provided)
        seed: Random seed for reproducibility
        food_ids_file: Path to food_objectIDs.txt file (if provided, ignores num_samples)
        verbose: Print detailed progress
    """
    # Set random seed
    random.seed(seed)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load objects data
    if verbose:
        print(f"Loading {objects_json}...")

    with open(objects_json, 'r') as f:
        objects_data = json.load(f)

    if verbose:
        print(f"  Total mask instances: {len(objects_data)}")

    # Select masks based on input method
    selected_masks = []

    if food_ids_file:
        # Use food_objectIDs.txt file
        if verbose:
            print(f"\nLoading food objects from {food_ids_file}...")

        food_objects = parse_food_ids_file(food_ids_file)

        if verbose:
            print(f"  Found {len(food_objects)} food objects")
            print(f"\nMatching objects in JSON...")

        # Create lookup dict for faster search
        lookup = {}
        for item in objects_data:
            key = (item['object_id'], item['video_id'])
            if key not in lookup:
                lookup[key] = []
            lookup[key].append(item)

        # Find matching masks for each food object
        matched_objects = 0
        total_masks = 0
        for food_obj in food_objects:
            key = (food_obj['object_id'], food_obj['video_id'])
            if key in lookup:
                # Take ALL mask instances for this object
                masks = lookup[key]
                selected_masks.extend(masks)
                matched_objects += 1
                total_masks += len(masks)
                if verbose:
                    print(f"  {matched_objects}. {masks[0]['object_name']} "
                          f"(video: {masks[0]['video_id']}, {len(masks)} masks)")
            else:
                if verbose:
                    print(f"  ✗ Not found: {food_obj['object_name']} "
                          f"(ID: {food_obj['object_id'][:8]}..., video: {food_obj['video_id']})")

        if verbose:
            print(f"\nMatched {matched_objects}/{len(food_objects)} food objects")
            print(f"Total masks to extract: {total_masks}")

    else:
        # Original behavior: select diverse samples randomly
        from collections import defaultdict
        objects_by_name = defaultdict(list)
        for item in objects_data:
            objects_by_name[item['object_name']].append(item)

        # Select diverse samples
        diverse_types = [
            'glass', 'bowl', 'plate', 'biscuit', 'bread',
            'knife', 'bottle', 'cup', 'fork', 'spoon', 'tomato',
            'onion', 'cheese', 'egg', 'pan', 'pot', 'carrot',
            'pepper', 'oil', 'butter', 'salt', 'milk'
        ]

        if verbose:
            print(f"\nSelecting {num_samples} diverse masks...")

        for obj_type in diverse_types:
            if len(selected_masks) >= num_samples:
                break

            # Find objects containing this type
            matching = []
            for name, items in objects_by_name.items():
                if obj_type in name.lower():
                    matching.extend(items)

            if matching:
                selected = random.choice(matching)
                selected_masks.append(selected)
                if verbose:
                    print(f"  {len(selected_masks)}. {selected['object_name']} "
                          f"(video: {selected['video_id']}, frame: {selected['frame_number']})")

    # Extract images
    if verbose:
        print(f"\n{'='*80}")
        print("EXTRACTING MASK IMAGES")
        print('='*80)

    extracted_info = []
    success_count = 0

    for i, mask in enumerate(selected_masks):
        video_id = mask['video_id']
        object_id = mask['object_id']
        mask_id = mask['mask_id']
        frame_num = mask['frame_number']
        object_name = mask['object_name']

        # Construct unique filename
        # Format: {video_id}_{object_id}_{mask_id}_{frame}.jpg
        base_filename = f"{video_id}_{object_id[:8]}_{mask_id[:8]}_{frame_num:06d}"
        roi_filename = f"{base_filename}_roi.jpg"
        full_filename = f"{base_filename}_full.jpg"

        roi_path = output_path / roi_filename
        full_path = output_path / full_filename

        if verbose:
            print(f"\n[{i+1}/{len(selected_masks)}] {object_name}")
            print(f"  Video: {video_id}")
            print(f"  Object ID: {object_id}")
            print(f"  Mask ID: {mask_id}")
            print(f"  Frame: {frame_num}")

        # Video path
        video_path = Path(video_dir) / f"{video_id}.mp4"

        if not video_path.exists():
            if verbose:
                print(f"  ✗ Video not found: {video_path}")
            continue

        # Open video
        cap = cv2.VideoCapture(str(video_path))

        # Check frame number validity
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_num >= total_frames:
            if verbose:
                print(f"  ✗ Frame {frame_num} exceeds video length ({total_frames})")
            cap.release()
            continue

        # Seek to frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()

        if not ret:
            if verbose:
                print(f"  ✗ Failed to read frame")
            cap.release()
            continue

        # Get bounding box
        bbox = [mask['bbox_x1'], mask['bbox_y1'], mask['bbox_x2'], mask['bbox_y2']]

        if any(b is None for b in bbox):
            if verbose:
                print(f"  ✗ Invalid bbox (contains None)")
            cap.release()
            continue

        x1, y1, x2, y2 = map(int, bbox)

        # Ensure bbox is within frame bounds
        h, w = frame.shape[:2]
        x1 = max(0, min(x1, w-1))
        y1 = max(0, min(y1, h-1))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))

        if x2 <= x1 or y2 <= y1:
            if verbose:
                print(f"  ✗ Invalid bbox dimensions")
            cap.release()
            continue

        # Extract ROI
        roi = frame[y1:y2, x1:x2].copy()

        # Create annotated full frame
        frame_annotated = frame.copy()
        cv2.rectangle(frame_annotated, (x1, y1), (x2, y2), (0, 255, 0), 3)

        # Add label with background
        label = object_name
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        label_size = cv2.getTextSize(label, font, font_scale, thickness)[0]

        # Label background
        cv2.rectangle(frame_annotated,
                     (x1, y1-label_size[1]-10),
                     (x1+label_size[0]+10, y1),
                     (0, 255, 0), -1)

        # Label text
        cv2.putText(frame_annotated, label, (x1+5, y1-5),
                   font, font_scale, (0, 0, 0), thickness)

        # Save images
        cv2.imwrite(str(roi_path), roi)
        cv2.imwrite(str(full_path), frame_annotated)

        # Store metadata
        extracted_info.append({
            'index': i + 1,
            'object_name': object_name,
            'video_id': video_id,
            'object_id': object_id,
            'mask_id': mask_id,
            'track_id': mask['track_id'],
            'frame_number': frame_num,
            'time_start': mask['time_start'],
            'time_end': mask['time_end'],
            'bbox': bbox,
            'fixture': mask['fixture'],
            'position_3d': [mask['position_x'], mask['position_y'], mask['position_z']],
            'roi_filename': roi_filename,
            'full_filename': full_filename,
            'roi_size': [roi.shape[1], roi.shape[0]]  # width, height
        })

        success_count += 1

        if verbose:
            print(f"  ✓ Extracted ROI: {roi.shape[1]}x{roi.shape[0]} pixels")
            print(f"    → {roi_filename}")
            print(f"    → {full_filename}")

        cap.release()

    # Save metadata
    metadata_path = output_path / 'extracted_masks_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(extracted_info, f, indent=2)

    if verbose:
        print(f"\n{'='*80}")
        print(f"EXTRACTION COMPLETE")
        print('='*80)
        print(f"Successfully extracted: {success_count}/{len(selected_masks)} masks")
        print(f"Output directory: {output_path.resolve()}")
        print(f"Metadata saved: {metadata_path}")
        print(f"\nFilename format: {{video_id}}_{{object_id[:8]}}_{{mask_id[:8]}}_{{frame:06d}}.jpg")
        print(f"  - video_id: Identifies the source video")
        print(f"  - object_id: Persistent object identifier (first 8 chars)")
        print(f"  - mask_id: Specific mask instance identifier (first 8 chars)")
        print(f"  - frame: Frame number (6 digits, zero-padded)")

    return extracted_info


def main():
    parser = argparse.ArgumentParser(
        description='Extract mask bounding box images from HD-EPIC videos'
    )
    parser.add_argument('--json', '-j', default='p01_objects_list.json',
                       help='Path to p01_objects_list.json (default: p01_objects_list.json)')
    parser.add_argument('--video-dir', '-v', default='HD-EPIC/Videos/P01',
                       help='Directory containing P01 videos (default: HD-EPIC/Videos/P01)')
    parser.add_argument('--output', '-o', default='extracted_masks',
                       help='Output directory (default: extracted_masks)')
    parser.add_argument('--num-samples', '-n', type=int, default=10,
                       help='Number of masks to extract (default: 10, ignored if --food-ids is used)')
    parser.add_argument('--food-ids', '-f',
                       help='Path to food_objectIDs.txt file (overrides --num-samples)')
    parser.add_argument('--seed', '-s', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Suppress verbose output')

    args = parser.parse_args()

    extract_mask_images(
        objects_json=args.json,
        video_dir=args.video_dir,
        output_dir=args.output,
        num_samples=args.num_samples,
        seed=args.seed,
        food_ids_file=args.food_ids,
        verbose=not args.quiet
    )


if __name__ == '__main__':
    main()

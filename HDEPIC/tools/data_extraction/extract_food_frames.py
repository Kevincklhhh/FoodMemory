#!/usr/bin/env python3
"""
Extract frames from HD-EPIC videos for all food items in food_objectIDs.txt

This script extracts full frames (without bounding boxes) for all food items
listed in food_objectIDs.txt by matching them with p01_objects_list.json.

Output files are named: {video_id}_{object_id}_{mask_id}_{frame}.jpg
"""

import json
import cv2
from pathlib import Path
import argparse
from collections import defaultdict


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


def extract_food_frames(
    objects_json='p01_objects_list.json',
    video_dir='HD-EPIC/Videos/P01',
    output_dir='food_extracted_frames',
    food_ids_file='food_objectIDs.txt',
    verbose=True
):
    """
    Extract frames from videos for all food items without drawing bounding boxes.

    Args:
        objects_json: Path to p01_objects_list.json
        video_dir: Directory containing P01 videos
        output_dir: Output directory for extracted frames
        food_ids_file: Path to food_objectIDs.txt file
        verbose: Print detailed progress
    """
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

    # Load food objects
    if verbose:
        print(f"\nLoading food objects from {food_ids_file}...")

    food_objects = parse_food_ids_file(food_ids_file)

    if verbose:
        print(f"  Found {len(food_objects)} food object entries")

    # Get unique food items
    unique_foods = defaultdict(int)
    for food_obj in food_objects:
        unique_foods[food_obj['object_name']] += 1

    if verbose:
        print(f"  Unique food items: {len(unique_foods)}")
        print(f"\nMatching objects in JSON...")

    # Create lookup dict for faster search
    lookup = {}
    for item in objects_data:
        key = (item['object_id'], item['video_id'])
        if key not in lookup:
            lookup[key] = []
        lookup[key].append(item)

    # Find matching masks for each food object
    selected_masks = []
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
        print(f"Total frames to extract: {total_masks}")

    # Extract frames
    if verbose:
        print(f"\n{'='*80}")
        print("EXTRACTING FOOD FRAMES")
        print('='*80)

    extracted_info = []
    success_count = 0

    # Group masks by video to optimize video reading
    masks_by_video = defaultdict(list)
    for mask in selected_masks:
        masks_by_video[mask['video_id']].append(mask)

    if verbose:
        print(f"\nProcessing {len(masks_by_video)} videos...")

    for video_idx, (video_id, masks) in enumerate(masks_by_video.items()):
        if verbose:
            print(f"\n[Video {video_idx+1}/{len(masks_by_video)}] {video_id}")
            print(f"  Extracting {len(masks)} frames...")

        # Video path
        video_path = Path(video_dir) / f"{video_id}.mp4"

        if not video_path.exists():
            if verbose:
                print(f"  ✗ Video not found: {video_path}")
            continue

        # Open video once
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Sort masks by frame number for efficient sequential reading
        masks_sorted = sorted(masks, key=lambda x: x['frame_number'])

        current_frame = -1
        frame_cache = None

        for mask in masks_sorted:
            object_id = mask['object_id']
            mask_id = mask['mask_id']
            frame_num = mask['frame_number']
            object_name = mask['object_name']

            # Construct filename
            filename = f"{video_id}_{object_id[:8]}_{mask_id[:8]}_{frame_num:06d}.jpg"
            output_file = output_path / filename

            # Check if frame is already extracted
            if output_file.exists():
                success_count += 1
                continue

            # Check frame validity
            if frame_num >= total_frames:
                if verbose:
                    print(f"  ✗ Frame {frame_num} exceeds video length ({total_frames})")
                continue

            # Read frame only if different from current
            if current_frame != frame_num:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame_cache = cap.read()

                if not ret:
                    if verbose:
                        print(f"  ✗ Failed to read frame {frame_num}")
                    continue

                current_frame = frame_num

            # Save frame
            cv2.imwrite(str(output_file), frame_cache)

            # Store metadata
            extracted_info.append({
                'object_name': object_name,
                'video_id': video_id,
                'object_id': object_id,
                'mask_id': mask_id,
                'track_id': mask['track_id'],
                'frame_number': frame_num,
                'time_start': mask['time_start'],
                'time_end': mask['time_end'],
                'bbox': [mask['bbox_x1'], mask['bbox_y1'], mask['bbox_x2'], mask['bbox_y2']],
                'fixture': mask['fixture'],
                'position_3d': [mask['position_x'], mask['position_y'], mask['position_z']],
                'filename': filename,
                'frame_size': [frame_cache.shape[1], frame_cache.shape[0]]  # width, height
            })

            success_count += 1

        cap.release()

        if verbose:
            print(f"  ✓ Extracted {len(masks)} frames")

    # Save metadata
    metadata_path = output_path / 'food_frames_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(extracted_info, f, indent=2)

    # Save unique food list
    unique_foods_list = [{'name': name, 'count': count}
                        for name, count in sorted(unique_foods.items())]
    unique_foods_path = output_path / 'unique_food_items.json'
    with open(unique_foods_path, 'w') as f:
        json.dump(unique_foods_list, f, indent=2)

    if verbose:
        print(f"\n{'='*80}")
        print(f"EXTRACTION COMPLETE")
        print('='*80)
        print(f"Successfully extracted: {success_count}/{total_masks} frames")
        print(f"Output directory: {output_path.resolve()}")
        print(f"Metadata saved: {metadata_path}")
        print(f"Unique foods list: {unique_foods_path}")
        print(f"\nFilename format: {{video_id}}_{{object_id[:8]}}_{{mask_id[:8]}}_{{frame:06d}}.jpg")

    return extracted_info


def main():
    parser = argparse.ArgumentParser(
        description='Extract frames for all food items from HD-EPIC videos'
    )
    parser.add_argument('--json', '-j', default='p01_objects_list.json',
                       help='Path to p01_objects_list.json (default: p01_objects_list.json)')
    parser.add_argument('--video-dir', '-v', default='HD-EPIC/Videos/P01',
                       help='Directory containing P01 videos (default: HD-EPIC/Videos/P01)')
    parser.add_argument('--output', '-o', default='food_extracted_frames',
                       help='Output directory (default: food_extracted_frames)')
    parser.add_argument('--food-ids', '-f', default='food_objectIDs.txt',
                       help='Path to food_objectIDs.txt file (default: food_objectIDs.txt)')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Suppress verbose output')

    args = parser.parse_args()

    extract_food_frames(
        objects_json=args.json,
        video_dir=args.video_dir,
        output_dir=args.output,
        food_ids_file=args.food_ids,
        verbose=not args.quiet
    )


if __name__ == '__main__':
    main()

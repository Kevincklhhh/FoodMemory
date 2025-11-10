#!/usr/bin/env python3
"""
Step 2: Create visual food segmentation masks from extracted food items.

This script:
1. Reads the visor_food_items.json from Step 1
2. Loads corresponding frame images
3. Creates segmentation masks for each food occurrence
4. Saves visualizations with colored overlays

Input: visor_food_items.json (from Step 1)
Output: Segmented food images in food_segments/ directory
"""

import json
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict


def load_food_items(json_path: str):
    """Load food items from Step 1 output."""
    with open(json_path, 'r') as f:
        return json.load(f)


def create_food_segmentation_mask(image_path: Path, segments: list, output_path: Path = None):
    """Create segmentation mask for food items in a frame.

    Args:
        image_path: Path to the frame image
        segments: List of polygon segments
        output_path: Optional path to save the masked image

    Returns:
        mask: Binary mask of food items
        masked_image: Original image with food items highlighted
    """
    # Load image
    img = cv2.imread(str(image_path))
    if img is None:
        return None, None

    h, w = img.shape[:2]

    # Create mask
    mask = np.zeros((h, w), dtype=np.uint8)

    # Draw each segment's polygons
    for segment_group in segments:
        # Convert points to numpy array
        points = np.array(segment_group, dtype=np.int32)
        # Fill polygon
        cv2.fillPoly(mask, [points], 255)

    # Create masked image (overlay)
    masked_image = img.copy()
    # Create colored overlay for food regions
    colored_mask = np.zeros_like(img)
    colored_mask[mask > 0] = [0, 255, 0]  # Green for food items

    # Blend with original image
    masked_image = cv2.addWeighted(img, 0.7, colored_mask, 0.3, 0)

    # Save if output path provided
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), masked_image)

    return mask, masked_image


def process_video_food_items(
    video_id: str,
    video_data: dict,
    frames_base_dir: Path,
    output_dir: Path,
    split: str = 'train'
):
    """Process all food items for a single video."""
    participant_id = video_data['participant_id']
    food_occurrences = video_data['food_occurrences']

    if not food_occurrences:
        return 0

    # Group by frame
    by_frame = defaultdict(list)
    for food in food_occurrences:
        by_frame[food['frame_name']].append(food)

    # Determine frame directory
    frames_dir = frames_base_dir / split / participant_id
    video_output_dir = output_dir / video_id

    extracted_count = 0

    for frame_name, foods in by_frame.items():
        frame_path = frames_dir / frame_name

        if not frame_path.exists():
            continue

        # Collect all segments for this frame
        all_segments = []
        for food in foods:
            all_segments.extend(food['segments'])

        # Create output filename
        output_filename = frame_name.replace('.jpg', '_food_mask.jpg')
        output_path = video_output_dir / output_filename

        # Create masked image with all food items
        mask, masked_image = create_food_segmentation_mask(
            frame_path,
            all_segments,
            output_path
        )

        if mask is not None:
            extracted_count += 1

    return extracted_count


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Create food segmentation visualizations (Step 2)"
    )
    parser.add_argument(
        '--input',
        default='visor_food_items.json',
        help='Input JSON from Step 1'
    )
    parser.add_argument(
        '--frames-dir',
        default='GroundTruth-SparseAnnotations/rgb_frames',
        help='Base directory for RGB frames'
    )
    parser.add_argument(
        '--output-dir',
        default='food_segments',
        help='Output directory for segmented images'
    )
    parser.add_argument(
        '--videos',
        nargs='*',
        help='Specific video IDs to process (default: all)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        help='Limit number of videos to process'
    )

    args = parser.parse_args()

    print("=" * 80)
    print("STEP 2: Create Food Segmentation Visualizations")
    print("=" * 80)

    # Load food items
    print(f"\nLoading food items from {args.input}...")
    all_videos = load_food_items(args.input)
    print(f"✓ Loaded {len(all_videos)} videos")

    # Filter videos if specified
    if args.videos:
        all_videos = {k: v for k, v in all_videos.items() if k in args.videos}
        print(f"  Processing {len(all_videos)} specified videos")

    if args.limit:
        all_videos = dict(list(all_videos.items())[:args.limit])
        print(f"  Limited to {len(all_videos)} videos")

    frames_base = Path(args.frames_dir)
    output_base = Path(args.output_dir)

    print(f"\nProcessing videos...")
    print("=" * 80)

    total_extracted = 0
    processed_videos = 0

    for video_id, video_data in all_videos.items():
        if not video_data['food_occurrences']:
            continue

        # Determine split (train or val based on existing structure)
        split = 'train'  # Default to train, could be enhanced to detect automatically

        count = process_video_food_items(
            video_id,
            video_data,
            frames_base,
            output_base,
            split
        )

        if count > 0:
            processed_videos += 1
            total_extracted += count
            print(f"  {video_id}: {count} frames with food segments")

    print("\n" + "=" * 80)
    print(f"✓ Processed {processed_videos} videos")
    print(f"✓ Created {total_extracted} segmented images")
    print(f"✓ Output saved to {output_base}")
    print("=" * 80)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Step 4: Unzip all RGB frames and build food image index.

This script:
1. Unzips all RGB frame archives in GroundTruth-SparseAnnotations/rgb_frames/
2. Builds a comprehensive index of all food images with:
   - Image file paths
   - Mask annotations (polygon coordinates)
   - Frame numbers
   - Video metadata
3. Creates a searchable index for food inventory lookup

Note: VISOR has train/val splits for segmentation tasks, but we ignore this distinction
since it's not relevant for our food inventory retrieval use case. We simply search across
all splits to find frames for each video.

Output: food_image_index.json - Complete mapping of food items to images and annotations
"""

import json
import zipfile
from pathlib import Path
from typing import Dict, List
from collections import defaultdict
import os


def unzip_frame_archives(base_dir: Path, splits: List[str] = ['train', 'val']) -> Dict[str, int]:
    """Unzip all RGB frame archives.

    Returns:
        Dictionary with unzip statistics per split
    """
    stats = {}

    for split in splits:
        frames_dir = base_dir / split
        if not frames_dir.exists():
            print(f"Warning: {frames_dir} does not exist, skipping...")
            continue

        # Find all zip files
        zip_files = list(frames_dir.rglob('*.zip'))
        print(f"\nProcessing {split} split: found {len(zip_files)} zip files")

        unzipped_count = 0
        skipped_count = 0

        for zip_path in zip_files:
            # Determine extraction directory (same as zip file location)
            extract_dir = zip_path.parent

            # Check if already unzipped (look for extracted frames)
            video_id = zip_path.stem
            # Check if we have frames already
            existing_frames = list(extract_dir.glob(f"{video_id}_frame_*.jpg"))

            if existing_frames:
                skipped_count += 1
                if skipped_count % 10 == 1:  # Print occasionally
                    print(f"  Skipping {video_id} (already unzipped)")
                continue

            # Unzip
            print(f"  Unzipping {video_id}...")
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)
                unzipped_count += 1
            except Exception as e:
                print(f"    Error unzipping {zip_path}: {e}")

        stats[split] = {
            'total_zips': len(zip_files),
            'unzipped': unzipped_count,
            'skipped': skipped_count
        }

        print(f"  {split}: Unzipped {unzipped_count}, Skipped {skipped_count}")

    return stats


def build_food_image_index(
    food_items_json: Path,
    frames_base_dir: Path,
    splits: List[str] = ['train', 'val']
) -> Dict:
    """Build comprehensive food image index.

    Returns:
        Dictionary with multiple index structures for different lookup patterns
    """
    # Load food items
    print("\nLoading food items...")
    with open(food_items_json, 'r') as f:
        all_videos = json.load(f)
    print(f"✓ Loaded {len(all_videos)} videos")

    # Initialize index structures
    food_index = {
        'by_food_class': defaultdict(list),  # food_class -> list of image entries
        'by_video': {},                       # video_id -> list of food images
        'by_frame': {},                       # video_id -> frame_num -> list of foods
        'metadata': {
            'total_videos': 0,
            'total_food_images': 0,
            'total_food_occurrences': 0,
            'food_classes': {}
        }
    }

    print("\nBuilding food image index...")

    total_images_indexed = 0
    missing_images = 0

    for video_id, video_data in all_videos.items():
        participant_id = video_data['participant_id']
        food_occurrences = video_data['food_occurrences']

        if not food_occurrences:
            continue

        # Find frames directory (ignore train/val split - not relevant for our use case)
        # Just search for the video frames across all split directories
        image_path = None
        for split in splits:
            potential_path = frames_base_dir / split / participant_id
            if potential_path.exists():
                # Check if this split has frames for this video
                sample_frames = list(potential_path.glob(f"{video_id}_frame_*.jpg"))
                if sample_frames:
                    image_path = potential_path
                    break

        if not image_path:
            print(f"  Warning: Could not find frames for {video_id}")
            continue

        # Group occurrences by frame
        by_frame = defaultdict(list)
        for occurrence in food_occurrences:
            frame_num = occurrence['frame_number']
            by_frame[frame_num].append(occurrence)

        # Index each frame with food
        video_food_images = []

        for frame_num, foods_in_frame in by_frame.items():
            # Construct frame path
            frame_name = foods_in_frame[0]['frame_name']
            frame_path = image_path / frame_name

            # Check if image exists
            if not frame_path.exists():
                missing_images += 1
                continue

            # Create image entry
            image_entry = {
                'video_id': video_id,
                'participant_id': participant_id,
                'frame_number': frame_num,
                'frame_name': frame_name,
                'image_path': str(frame_path.relative_to(frames_base_dir)),
                'foods': []
            }

            # Add each food in this frame
            for food in foods_in_frame:
                food_entry = {
                    'class_id': food['class_id'],
                    'noun_key': food['noun_key'],
                    'object_name': food['object_name'],
                    'object_id': food['object_id'],
                    'segments': food['segments'],  # Polygon coordinates
                    'exhaustive': food['exhaustive']
                }
                image_entry['foods'].append(food_entry)

                # Index by food class
                food_index['by_food_class'][food['noun_key']].append({
                    'video_id': video_id,
                    'participant_id': participant_id,
                    'frame_number': frame_num,
                    'frame_name': frame_name,
                    'image_path': str(frame_path.relative_to(frames_base_dir)),
                    'class_id': food['class_id'],
                    'object_name': food['object_name'],
                    'object_id': food['object_id'],
                    'segments': food['segments']
                })

            video_food_images.append(image_entry)
            total_images_indexed += 1

        # Add to video index
        if video_food_images:
            food_index['by_video'][video_id] = video_food_images

            # Add to frame index
            food_index['by_frame'][video_id] = {
                img['frame_number']: img for img in video_food_images
            }

    # Calculate metadata
    food_index['metadata']['total_videos'] = len(food_index['by_video'])
    food_index['metadata']['total_food_images'] = total_images_indexed
    food_index['metadata']['total_food_occurrences'] = sum(
        len(video_data['food_occurrences'])
        for video_data in all_videos.values()
    )

    # Food class statistics
    for food_class, images in food_index['by_food_class'].items():
        food_index['metadata']['food_classes'][food_class] = {
            'total_images': len(images),
            'videos': len(set(img['video_id'] for img in images))
        }

    print(f"✓ Indexed {total_images_indexed} food images")
    if missing_images > 0:
        print(f"  Warning: {missing_images} frame images not found")

    return food_index


def create_food_inventory_lookup(food_index: Dict, output_path: Path):
    """Create simplified lookup table for food inventory queries.

    This creates a more compact index optimized for:
    - Finding all images of a specific food item
    - Getting sample images for food inventory
    """
    lookup = {
        'food_items': {},
        'metadata': food_index['metadata']
    }

    for food_class, images in food_index['by_food_class'].items():
        # Sort by video and frame for consistent ordering
        sorted_images = sorted(images, key=lambda x: (x['video_id'], x['frame_number']))

        lookup['food_items'][food_class] = {
            'total_occurrences': len(images),
            'total_videos': len(set(img['video_id'] for img in images)),
            'sample_images': sorted_images[:10],  # First 10 for quick preview
            'all_images': [
                {
                    'path': img['image_path'],
                    'video': img['video_id'],
                    'frame': img['frame_number'],
                    'object_id': img['object_id']
                }
                for img in sorted_images
            ]
        }

    with open(output_path, 'w') as f:
        json.dump(lookup, f, indent=2)

    print(f"✓ Saved food inventory lookup to {output_path}")


def print_statistics(food_index: Dict):
    """Print index statistics."""
    print("\n" + "=" * 80)
    print("FOOD IMAGE INDEX STATISTICS")
    print("=" * 80)

    metadata = food_index['metadata']

    print(f"\nTotal videos with food: {metadata['total_videos']}")
    print(f"Total food images indexed: {metadata['total_food_images']}")
    print(f"Total food occurrences: {metadata['total_food_occurrences']}")
    print(f"Unique food classes: {len(food_index['by_food_class'])}")

    print("\nTop 15 food classes by image count:")
    sorted_foods = sorted(
        metadata['food_classes'].items(),
        key=lambda x: x[1]['total_images'],
        reverse=True
    )
    for food_class, stats in sorted_foods[:15]:
        print(f"  {food_class:<20} {stats['total_images']:5d} images in {stats['videos']:3d} videos")

    print("\nTop 10 videos by food image count:")
    video_counts = [
        (video_id, len(images))
        for video_id, images in food_index['by_video'].items()
    ]
    for video_id, count in sorted(video_counts, key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {video_id:<15} {count:4d} food images")


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Unzip frames and build food image index (Step 4)"
    )
    parser.add_argument(
        '--frames-dir',
        default='GroundTruth-SparseAnnotations/rgb_frames',
        help='Base directory for RGB frames (default: GroundTruth-SparseAnnotations/rgb_frames)'
    )
    parser.add_argument(
        '--food-items',
        default='visor_food_items.json',
        help='Food items JSON from Step 1 (default: visor_food_items.json)'
    )
    parser.add_argument(
        '--output',
        default='food_image_index.json',
        help='Output index file (default: food_image_index.json)'
    )
    parser.add_argument(
        '--lookup-output',
        default='food_inventory_lookup.json',
        help='Output lookup file for inventory (default: food_inventory_lookup.json)'
    )
    parser.add_argument(
        '--skip-unzip',
        action='store_true',
        help='Skip unzipping step (if already done)'
    )
    parser.add_argument(
        '--splits',
        nargs='+',
        default=['train', 'val'],
        help='VISOR splits to process (default: train val)'
    )

    args = parser.parse_args()

    print("=" * 80)
    print("STEP 4: Build Food Image Index")
    print("=" * 80)

    frames_base = Path(args.frames_dir)

    # Step 1: Unzip all frames
    if not args.skip_unzip:
        print("\n" + "=" * 80)
        print("PHASE 1: Unzipping RGB Frame Archives")
        print("=" * 80)
        unzip_stats = unzip_frame_archives(frames_base, args.splits)
        print("\nUnzip complete!")
        for split, stats in unzip_stats.items():
            print(f"  {split}: {stats['unzipped']} unzipped, {stats['skipped']} skipped")
    else:
        print("\nSkipping unzip phase (--skip-unzip specified)")

    # Step 2: Build food image index
    print("\n" + "=" * 80)
    print("PHASE 2: Building Food Image Index")
    print("=" * 80)

    food_index = build_food_image_index(
        Path(args.food_items),
        frames_base,
        args.splits
    )

    # Step 3: Save index
    print("\nSaving index...")
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(food_index, f, indent=2)
    print(f"✓ Saved complete index to {output_path}")

    # Step 4: Create food inventory lookup
    print("\nCreating food inventory lookup...")
    lookup_path = Path(args.lookup_output)
    create_food_inventory_lookup(food_index, lookup_path)

    # Print statistics
    print_statistics(food_index)

    print("\n" + "=" * 80)
    print("✓ Index building complete!")
    print("=" * 80)
    print(f"\nOutput files:")
    print(f"  1. {output_path} - Complete food image index")
    print(f"  2. {lookup_path} - Optimized for food inventory lookup")


if __name__ == '__main__':
    main()

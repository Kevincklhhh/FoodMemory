#!/usr/bin/env python3
"""
Query food inventory index to find images of specific food items.

This script demonstrates how to use the food_inventory_lookup.json index
to find images for a synthetic household food inventory.

Usage examples:
  # Find all images of onion (first per video by default)
  python3 query_food_inventory.py --food onion

  # Get ALL frames with onion from ALL videos
  python3 query_food_inventory.py --food onion --all-occurrences

  # Find images for multiple food items
  python3 query_food_inventory.py --food onion cheese tomato

  # Get sample images (first N)
  python3 query_food_inventory.py --food onion --limit 5

  # Get all yoghurt frames from all videos and export to CSV
  python3 query_food_inventory.py --food yoghurt --all-occurrences --export yoghurt_all.csv

  # Export to CSV
  python3 query_food_inventory.py --food onion cheese --export inventory.csv
"""

import json
import csv
from pathlib import Path
from typing import List, Dict
import shutil


def load_food_index(index_path: str = 'food_inventory_lookup.json') -> Dict:
    """Load food inventory lookup index."""
    with open(index_path, 'r') as f:
        return json.load(f)


def query_food_items(index: Dict, food_items: List[str], limit: int = None, first_per_video: bool = True) -> Dict[str, List]:
    """Query the index for specific food items.

    Args:
        index: Food inventory lookup dictionary
        food_items: List of food item names to query
        limit: Optional limit on number of images per food item
        first_per_video: If True, return only first appearance per video (default: True)
                        If False, return ALL occurrences from ALL videos

    Returns:
        Dictionary mapping food item to list of image entries
    """
    results = {}

    for food_item in food_items:
        food_item = food_item.lower()

        if food_item not in index['food_items']:
            print(f"Warning: '{food_item}' not found in index")
            available = sorted(index['food_items'].keys())
            # Find similar items
            similar = [f for f in available if food_item in f or f in food_item]
            if similar:
                print(f"  Did you mean: {', '.join(similar[:5])}")
            results[food_item] = []
            continue

        food_data = index['food_items'][food_item]
        all_images = food_data['all_images']

        # If first_per_video, get only first appearance in each video
        if first_per_video:
            seen_videos = set()
            filtered_images = []
            for img in all_images:
                if img['video'] not in seen_videos:
                    filtered_images.append(img)
                    seen_videos.add(img['video'])
            images = filtered_images
        else:
            # Return all occurrences from all videos
            images = all_images

        if limit:
            images = images[:limit]

        results[food_item] = {
            'images': images,
            'total_occurrences': food_data['total_occurrences'],
            'total_videos': food_data['total_videos'],
            'unique_videos_returned': len(set(img['video'] for img in images))
        }

    return results


def print_query_results(results: Dict):
    """Print query results in a readable format."""
    print("\n" + "=" * 80)
    print("FOOD INVENTORY QUERY RESULTS")
    print("=" * 80)

    for food_item, data in results.items():
        if not data:
            print(f"\n{food_item.upper()}: NOT FOUND")
            continue

        images = data['images']
        print(f"\n{food_item.upper()}")
        print(f"  Total occurrences in dataset: {data['total_occurrences']}")
        print(f"  Total videos with this food: {data['total_videos']}")
        print(f"  Unique videos returned: {data['unique_videos_returned']}")
        print(f"  Images returned: {len(images)}")
        print(f"\n  Sample images:")
        for i, img in enumerate(images[:5], 1):
            print(f"    {i}. Video {img['video']}, Frame {img['frame']} -> {img['path']}")
        if len(images) > 5:
            print(f"    ... and {len(images) - 5} more")


def export_to_csv(results: Dict, output_path: str, frames_base: str = 'GroundTruth-SparseAnnotations/rgb_frames'):
    """Export query results to CSV.

    Args:
        results: Query results dictionary
        output_path: Output CSV file path
        frames_base: Base path for frames directory
    """
    with open(output_path, 'w', newline='') as f:
        fieldnames = [
            'food_item', 'image_path', 'full_path', 'video_id',
            'frame_number', 'object_id'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for food_item, data in results.items():
            if not data:
                continue

            for img in data['images']:
                writer.writerow({
                    'food_item': food_item,
                    'image_path': img['path'],
                    'full_path': f"{frames_base}/{img['path']}",
                    'video_id': img['video'],
                    'frame_number': img['frame'],
                    'object_id': img['object_id']
                })

    print(f"\n✓ Exported results to {output_path}")


def copy_food_images(results: Dict, output_dir: str, frames_base: str = 'GroundTruth-SparseAnnotations/rgb_frames'):
    """Copy food images to a separate directory organized by food item.

    Args:
        results: Query results dictionary
        output_dir: Output directory for organized images
        frames_base: Base path for frames directory
    """
    output_path = Path(output_dir)
    frames_base_path = Path(frames_base)

    for food_item, data in results.items():
        if not data:
            continue

        food_dir = output_path / food_item
        food_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nCopying images for {food_item}...")
        copied = 0

        for img in data['images']:
            src = frames_base_path / img['path']
            if not src.exists():
                continue

            # Create filename with video and frame info
            dst_name = f"{img['video']}_frame_{img['frame']:010d}.jpg"
            dst = food_dir / dst_name

            shutil.copy2(src, dst)
            copied += 1

        print(f"  Copied {copied} images to {food_dir}")


def list_available_foods(index: Dict, pattern: str = None):
    """List all available food items in the index.

    Args:
        index: Food inventory lookup dictionary
        pattern: Optional pattern to filter food items
    """
    foods = sorted(index['food_items'].keys())

    if pattern:
        foods = [f for f in foods if pattern.lower() in f.lower()]

    print("\n" + "=" * 80)
    print(f"AVAILABLE FOOD ITEMS ({len(foods)} items)")
    print("=" * 80)

    # Group by first letter
    current_letter = None
    for food in foods:
        letter = food[0].upper()
        if letter != current_letter:
            current_letter = letter
            print(f"\n{letter}:")

        food_data = index['food_items'][food]
        print(f"  {food:<25} ({food_data['total_occurrences']:4d} images, {food_data['total_videos']:2d} videos)")


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Query food inventory index for specific food items"
    )
    parser.add_argument(
        '--index',
        default='food_inventory_lookup.json',
        help='Food inventory lookup file (default: food_inventory_lookup.json)'
    )
    parser.add_argument(
        '--food',
        nargs='+',
        help='Food items to query (e.g., onion cheese tomato)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        help='Limit number of images per food item'
    )
    parser.add_argument(
        '--all-occurrences',
        action='store_true',
        help='Return ALL frames with the queried food from ALL videos (default: only first frame per video)'
    )
    parser.add_argument(
        '--export',
        help='Export results to CSV file'
    )
    parser.add_argument(
        '--copy-images',
        help='Copy images to directory organized by food item'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List all available food items'
    )
    parser.add_argument(
        '--search',
        help='Search for food items matching pattern'
    )
    parser.add_argument(
        '--frames-base',
        default='GroundTruth-SparseAnnotations/rgb_frames',
        help='Base path for frames directory'
    )

    args = parser.parse_args()

    # Load index
    print(f"Loading food inventory index from {args.index}...")
    index = load_food_index(args.index)
    print(f"✓ Loaded index with {len(index['food_items'])} food classes")

    # List mode
    if args.list or args.search:
        list_available_foods(index, args.search)
        return

    # Query mode
    if not args.food:
        parser.error("Please specify --food items to query or use --list to see available foods")

    if args.all_occurrences:
        print(f"\nQuerying for: {', '.join(args.food)} (all occurrences from all videos)")
    else:
        print(f"\nQuerying for: {', '.join(args.food)} (first per video)")

    results = query_food_items(
        index,
        args.food,
        args.limit,
        first_per_video=not args.all_occurrences
    )

    # Print results
    print_query_results(results)

    # Export if requested
    if args.export:
        export_to_csv(results, args.export, args.frames_base)

    # Copy images if requested
    if args.copy_images:
        copy_food_images(results, args.copy_images, args.frames_base)
        print(f"\n✓ Images organized in {args.copy_images}")


if __name__ == '__main__':
    main()

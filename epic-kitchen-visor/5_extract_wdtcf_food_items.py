#!/usr/bin/env python3
"""
Step 5: Extract food items from WDTCF (Where Did The Container First appear) dataset.

WDTCF is a temporal localization task that tracks where objects (including food)
were first stored in the kitchen.

This script:
1. Loads WDTCF ground truth annotations
2. Filters for food items using epic_food_nouns_detailed.json
3. Organizes food items by video with queryable instances
4. Extracts frame information for query and evidence

Output: wdtcf_food_items.json - Food items per video with instance details
"""

import json
from pathlib import Path
from typing import Dict, List
from collections import defaultdict


def load_food_nouns(json_path: str = 'epic_food_nouns_detailed.json') -> set:
    """Load food noun names from epic_food_nouns_detailed.json.

    Returns:
        Set of food noun names (lowercase)
    """
    with open(json_path, 'r') as f:
        food_data = json.load(f)

    food_nouns = {item['noun_name'].lower() for item in food_data}
    return food_nouns


def parse_wdtcf_key(key: str) -> tuple:
    """Parse WDTCF key into video_id and object_name.

    Args:
        key: WDTCF key in format '{video_id}_{object_name}'

    Returns:
        Tuple of (video_id, object_name)
    """
    # Split on underscore, but video_id contains underscores
    # Format: P{participant}_{video_num}_{object_name}
    parts = key.split('_')

    # Video ID is first two parts: P{participant}_{video_num}
    video_id = f"{parts[0]}_{parts[1]}"

    # Object name is remaining parts joined
    object_name = '_'.join(parts[2:])

    return video_id, object_name


def parse_frame_info(frame_name: str) -> dict:
    """Parse frame filename to extract video and frame number.

    Args:
        frame_name: Frame filename like 'P01_01_frame_0000096842.jpg'

    Returns:
        Dictionary with video_id, frame_number, frame_name
    """
    # Format: {video_id}_frame_{frame_number:010d}.jpg
    parts = frame_name.replace('.jpg', '').split('_')

    video_id = f"{parts[0]}_{parts[1]}"
    frame_number = int(parts[3])

    return {
        'video_id': video_id,
        'frame_number': frame_number,
        'frame_name': frame_name
    }


def extract_wdtcf_food_items(
    wdtcf_path: str = 'WDTCF_GT.json',
    food_nouns_path: str = 'epic_food_nouns_detailed.json'
) -> Dict:
    """Extract food items from WDTCF dataset.

    Returns:
        Dictionary organized by video with food instances
    """
    # Load food nouns
    print(f"Loading food nouns from {food_nouns_path}...")
    food_nouns = load_food_nouns(food_nouns_path)
    print(f"✓ Loaded {len(food_nouns)} food classes")

    # Load WDTCF data
    print(f"\nLoading WDTCF data from {wdtcf_path}...")
    with open(wdtcf_path, 'r') as f:
        wdtcf_data = json.load(f)
    print(f"✓ Loaded {len(wdtcf_data)} WDTCF entries")

    # Extract food items organized by video
    food_by_video = defaultdict(list)
    food_classes_found = set()
    total_food_instances = 0
    non_food_items = set()

    for key, entry in wdtcf_data.items():
        video_id, object_name = parse_wdtcf_key(key)

        # Check if this is a food item
        object_name_lower = object_name.lower().replace(':', '_')

        # Handle compound names (e.g., "ring:onion" -> "onion")
        # Check both full name and after colon/underscore
        is_food = False
        matched_food_name = None

        # Try exact match first
        if object_name_lower in food_nouns:
            is_food = True
            matched_food_name = object_name_lower
        else:
            # Try splitting on : or _ and checking parts
            for separator in [':', '_']:
                if separator in object_name_lower:
                    parts = object_name_lower.split(separator)
                    for part in parts:
                        if part in food_nouns:
                            is_food = True
                            matched_food_name = part
                            break
                if is_food:
                    break

        if not is_food:
            non_food_items.add(object_name)
            continue

        # Parse query and evidence frame info
        query_info = parse_frame_info(entry['query'])
        evidence_info = parse_frame_info(entry['evidence'])

        # Create food instance entry
        food_instance = {
            'instance_id': key,  # Original WDTCF key for queryability
            'object_name': object_name,
            'food_class': matched_food_name,
            'query_frame': query_info,
            'evidence_frame': evidence_info,
            'storage_locations': entry['answer']
        }

        food_by_video[video_id].append(food_instance)
        food_classes_found.add(matched_food_name)
        total_food_instances += 1

    # Build final structure
    result = {
        'metadata': {
            'total_videos': len(food_by_video),
            'total_food_instances': total_food_instances,
            'unique_food_classes': len(food_classes_found),
            'food_classes': sorted(list(food_classes_found)),
            'non_food_items': sorted(list(non_food_items))
        },
        'videos': {}
    }

    # Organize by video
    for video_id, food_instances in sorted(food_by_video.items()):
        # Get unique food classes in this video
        unique_foods = set(inst['food_class'] for inst in food_instances)

        result['videos'][video_id] = {
            'video_id': video_id,
            'total_food_instances': len(food_instances),
            'unique_food_classes': len(unique_foods),
            'food_classes': sorted(list(unique_foods)),
            'food_instances': sorted(food_instances, key=lambda x: x['query_frame']['frame_number'])
        }

    return result


def print_statistics(result: Dict):
    """Print extraction statistics."""
    print("\n" + "=" * 80)
    print("WDTCF FOOD EXTRACTION STATISTICS")
    print("=" * 80)

    metadata = result['metadata']

    print(f"\nTotal videos with food: {metadata['total_videos']}")
    print(f"Total food instances: {metadata['total_food_instances']}")
    print(f"Unique food classes: {metadata['unique_food_classes']}")

    print(f"\nFood classes found ({len(metadata['food_classes'])}):")
    for i, food_class in enumerate(metadata['food_classes'], 1):
        # Count instances
        count = sum(
            sum(1 for inst in video['food_instances'] if inst['food_class'] == food_class)
            for video in result['videos'].values()
        )
        print(f"  {i:3d}. {food_class:<20} ({count} instances)")

    print(f"\nNon-food items filtered out ({len(metadata['non_food_items'])}):")
    for item in metadata['non_food_items'][:20]:
        print(f"  - {item}")
    if len(metadata['non_food_items']) > 20:
        print(f"  ... and {len(metadata['non_food_items']) - 20} more")

    print("\nTop 10 videos by food instance count:")
    video_counts = [
        (video_id, data['total_food_instances'])
        for video_id, data in result['videos'].items()
    ]
    for video_id, count in sorted(video_counts, key=lambda x: x[1], reverse=True)[:10]:
        foods = ', '.join(result['videos'][video_id]['food_classes'][:5])
        if len(result['videos'][video_id]['food_classes']) > 5:
            foods += '...'
        print(f"  {video_id:<15} {count:2d} instances: {foods}")


def create_simple_food_list(result: Dict, output_path: str):
    """Create simple text file with food per video."""
    with open(output_path, 'w') as f:
        f.write("WDTCF FOOD ITEMS BY VIDEO (Simple List)\n")
        f.write("=" * 80 + "\n\n")

        for video_id, video_data in sorted(result['videos'].items()):
            foods = ', '.join(sorted(video_data['food_classes']))
            f.write(f"{video_id}: {foods}\n")

    print(f"✓ Saved simple list to {output_path}")


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract food items from WDTCF dataset (Step 5)"
    )
    parser.add_argument(
        '--wdtcf',
        default='WDTCF_GT.json',
        help='WDTCF ground truth file (default: WDTCF_GT.json)'
    )
    parser.add_argument(
        '--food-nouns',
        default='epic_food_nouns_detailed.json',
        help='Food nouns JSON (default: epic_food_nouns_detailed.json)'
    )
    parser.add_argument(
        '--output',
        default='wdtcf_food_items.json',
        help='Output JSON file (default: wdtcf_food_items.json)'
    )
    parser.add_argument(
        '--simple-output',
        default='wdtcf_food_per_video_simple.txt',
        help='Simple text output (default: wdtcf_food_per_video_simple.txt)'
    )

    args = parser.parse_args()

    print("=" * 80)
    print("STEP 5: Extract Food Items from WDTCF Dataset")
    print("=" * 80)

    # Extract food items
    result = extract_wdtcf_food_items(args.wdtcf, args.food_nouns)

    # Print statistics
    print_statistics(result)

    # Save output
    print(f"\nSaving results to {args.output}...")
    with open(args.output, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"✓ Saved to {args.output}")

    # Create simple list
    create_simple_food_list(result, args.simple_output)

    print("\n" + "=" * 80)
    print("✓ Extraction complete!")
    print("=" * 80)
    print(f"\nOutput files:")
    print(f"  1. {args.output} - Complete food instances with query/evidence frames")
    print(f"  2. {args.simple_output} - Simple list of food per video")


if __name__ == '__main__':
    main()

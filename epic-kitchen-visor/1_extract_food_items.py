#!/usr/bin/env python3
"""
Step 1: Extract food items from VISOR annotations using class_id mapping.

This script:
1. Loads EPIC-100 noun classes and food classifications
2. Processes VISOR annotations to find food items
3. Outputs a single JSON file with all food occurrences and their frame locations

Output: visor_food_items.json - Contains all food items with frame-level annotations
"""

import json
import csv
from pathlib import Path
from typing import Dict, List, Set
from collections import defaultdict
import re


def load_epic_noun_classes(csv_path: str) -> Dict[int, Dict]:
    """Load EPIC-100 noun classes mapping."""
    noun_classes = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            class_id = int(row['id'])
            noun_classes[class_id] = {
                'key': row['key'],
                'category': row['category']
            }
    return noun_classes


def load_food_class_ids(json_path: str) -> Dict[int, str]:
    """Load food class IDs from epic_food_nouns_detailed.json.

    Returns:
        dict: Mapping of class_id (int) to noun_name (str)
    """
    with open(json_path, 'r') as f:
        food_data = json.load(f)

    food_class_map = {}
    for item in food_data:
        class_id = item['class_id']
        noun_name = item['noun_name']
        food_class_map[class_id] = noun_name

    return food_class_map


def extract_frame_number(frame_name: str) -> int:
    """Extract frame number from frame filename."""
    match = re.search(r'frame_(\d+)', frame_name)
    if match:
        return int(match.group(1))
    return 0


def analyze_visor_video(
    json_file: Path,
    noun_classes: Dict,
    food_class_ids: Dict[int, str]
) -> Dict:
    """Analyze a single VISOR annotation file for food items.

    Returns:
        Dictionary with video info and all food item occurrences with frame details
    """
    with open(json_file, 'r') as f:
        data = json.load(f)

    video_id = json_file.stem
    participant_id = video_id.split('_')[0]

    # Store all food occurrences with their frame and segmentation info
    food_occurrences = []

    # Sort frames chronologically
    frames = sorted(
        data['video_annotations'],
        key=lambda x: extract_frame_number(x['image']['name'])
    )

    for frame_data in frames:
        frame_name = frame_data['image']['name']
        frame_num = extract_frame_number(frame_name)

        # Check each object in this frame
        for obj in frame_data['annotations']:
            class_id = obj.get('class_id')

            if class_id is None or class_id not in food_class_ids:
                continue

            # This is a food item!
            food_occurrences.append({
                'frame_name': frame_name,
                'frame_number': frame_num,
                'class_id': class_id,
                'noun_key': food_class_ids[class_id],
                'object_name': obj['name'],
                'object_id': obj['id'],
                'segments': obj.get('segments', []),
                'exhaustive': obj.get('exhaustive', 'n')
            })

    return {
        'video_id': video_id,
        'participant_id': participant_id,
        'total_frames_annotated': len(frames),
        'food_occurrences': food_occurrences
    }


def process_all_visor_annotations(
    visor_dir: Path,
    noun_classes: Dict,
    food_class_ids: Dict[int, str],
    splits: List[str] = ['train', 'val']
) -> Dict[str, Dict]:
    """Process all VISOR annotation files across splits."""
    all_videos = {}

    for split in splits:
        split_dir = visor_dir / 'annotations' / split

        if not split_dir.exists():
            print(f"Warning: {split_dir} does not exist, skipping...")
            continue

        json_files = sorted(split_dir.glob('*.json'))
        print(f"\nProcessing {split} split: {len(json_files)} videos")
        print("=" * 80)

        for json_file in json_files:
            video_id = json_file.stem
            print(f"  Analyzing {video_id}...", end='')

            result = analyze_visor_video(json_file, noun_classes, food_class_ids)
            all_videos[video_id] = result

            print(f" Found {len(result['food_occurrences'])} food occurrences")

    return all_videos


def print_summary(results: Dict[str, Dict]):
    """Print summary statistics."""
    print("\n" + "=" * 80)
    print("EXTRACTION SUMMARY")
    print("=" * 80)

    total_videos = len(results)
    videos_with_food = sum(1 for v in results.values() if v['food_occurrences'])
    total_food_occurrences = sum(len(v['food_occurrences']) for v in results.values())

    # Count unique food classes
    unique_classes = set()
    food_counts = defaultdict(int)
    for video_data in results.values():
        for food in video_data['food_occurrences']:
            unique_classes.add(food['noun_key'])
            food_counts[food['noun_key']] += 1

    print(f"\nTotal videos analyzed: {total_videos}")
    print(f"Videos with food items: {videos_with_food}")
    print(f"Total food occurrences: {total_food_occurrences}")
    print(f"Unique food classes: {len(unique_classes)}")

    print("\nTop 10 most common food classes:")
    for noun_key, count in sorted(food_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {noun_key:<30} {count:4d} occurrences")


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract food items from VISOR annotations (Step 1)"
    )
    parser.add_argument(
        '--visor-dir',
        default='GroundTruth-SparseAnnotations',
        help='Path to VISOR annotations directory'
    )
    parser.add_argument(
        '--noun-classes',
        default='EPIC_100_noun_classes_v2.csv',
        help='Path to EPIC_100_noun_classes_v2.csv'
    )
    parser.add_argument(
        '--food-json',
        default='epic_food_nouns_detailed.json',
        help='Path to epic_food_nouns_detailed.json'
    )
    parser.add_argument(
        '--output',
        default='visor_food_items.json',
        help='Output JSON file'
    )
    parser.add_argument(
        '--splits',
        nargs='+',
        default=['train', 'val'],
        help='VISOR splits to process'
    )

    args = parser.parse_args()

    print("=" * 80)
    print("STEP 1: Extract Food Items from VISOR Annotations")
    print("=" * 80)

    print("\nLoading EPIC-100 noun classes...")
    noun_classes = load_epic_noun_classes(args.noun_classes)
    print(f"✓ Loaded {len(noun_classes)} noun classes")

    print("\nLoading food class IDs...")
    food_class_ids = load_food_class_ids(args.food_json)
    print(f"✓ Loaded {len(food_class_ids)} food classes")

    print("\nProcessing VISOR annotations...")
    results = process_all_visor_annotations(
        Path(args.visor_dir),
        noun_classes,
        food_class_ids,
        args.splits
    )

    print("\nSaving results...")
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Saved to {output_path}")

    print_summary(results)

    print("\n✓ Done! Next step: Run 2_create_food_segments.py")


if __name__ == '__main__':
    main()

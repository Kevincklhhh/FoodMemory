#!/usr/bin/env python3
"""
Step 2: Extract food items from P01 narrations using class_id mapping.

This script:
1. Loads HD-EPIC noun classes and food classifications
2. Processes P01 narrations to find food items
3. Outputs a single JSON file with all food occurrences

Input: participant_P01_narrations.csv and hdepic_food_nouns_detailed.json
Output: hdepic_p01_food_items.json - Contains all food items with narration details
"""

import json
import csv
from pathlib import Path
from typing import Dict, List
from collections import defaultdict
import ast


def load_food_class_ids(json_path: str) -> Dict[int, str]:
    """Load food class IDs from hdepic_food_nouns_detailed.json.

    Returns:
        dict: Mapping of class_id (int) to noun_key (str)
    """
    with open(json_path, 'r') as f:
        food_data = json.load(f)

    food_class_map = {}
    for item in food_data:
        class_id = item['class_id']
        noun_key = item['noun_key']
        food_class_map[class_id] = noun_key

    return food_class_map


def extract_food_from_narrations(
    narrations_csv: Path,
    food_class_ids: Dict[int, str]
) -> Dict[str, List]:
    """Extract food items from P01 narrations.

    Args:
        narrations_csv: Path to participant_P01_narrations.csv
        food_class_ids: Mapping of food class_id to noun_key

    Returns:
        Dictionary mapping video_id to list of food occurrences
    """
    video_food_items = defaultdict(list)

    with open(narrations_csv, 'r') as f:
        reader = csv.DictReader(f)

        for row in reader:
            video_id = row['video_id']
            participant_id = row['participant_id']
            narration_id = row['unique_narration_id']
            narration = row['narration']
            start_time = float(row['start_timestamp'])
            end_time = float(row['end_timestamp'])
            narration_time = float(row['narration_timestamp'])

            # Parse noun_classes - it's a list like "[3, 3]"
            try:
                noun_classes = ast.literal_eval(row['noun_classes'])
            except:
                noun_classes = []

            # Parse nouns - it's a list like "['upper cupboard', 'handle of cupboard']"
            try:
                nouns = ast.literal_eval(row['nouns'])
            except:
                nouns = []

            # Check which noun classes are food
            for idx, noun_class_id in enumerate(noun_classes):
                if noun_class_id in food_class_ids:
                    # This is a food item!
                    noun_text = nouns[idx] if idx < len(nouns) else f"noun_class_{noun_class_id}"

                    food_occurrence = {
                        'narration_id': narration_id,
                        'narration': narration,
                        'start_timestamp': start_time,
                        'end_timestamp': end_time,
                        'narration_timestamp': narration_time,
                        'class_id': noun_class_id,
                        'noun_key': food_class_ids[noun_class_id],
                        'noun_text': noun_text,
                        'hands': ast.literal_eval(row['hands']) if row['hands'] else []
                    }

                    video_food_items[video_id].append(food_occurrence)

    return dict(video_food_items)


def print_summary(results: Dict[str, List]):
    """Print summary statistics."""
    print("\n" + "=" * 80)
    print("EXTRACTION SUMMARY")
    print("=" * 80)

    total_videos = len(results)
    total_food_occurrences = sum(len(items) for items in results.values())

    # Count unique food classes
    unique_classes = set()
    food_counts = defaultdict(int)

    for video_id, food_items in results.items():
        for food in food_items:
            unique_classes.add(food['noun_key'])
            food_counts[food['noun_key']] += 1

    print(f"\nTotal P01 videos with food: {total_videos}")
    print(f"Total food occurrences: {total_food_occurrences}")
    print(f"Unique food classes: {len(unique_classes)}")

    print("\nTop 10 most common food items:")
    for noun_key, count in sorted(food_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {noun_key:<30} {count:4d} occurrences")

    print("\nVideos with food:")
    for video_id in sorted(results.keys()):
        print(f"  {video_id:<25} {len(results[video_id]):4d} food occurrences")


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract food items from P01 narrations (Step 2)"
    )
    parser.add_argument(
        '--narrations',
        default='participant_P01_narrations.csv',
        help='Path to P01 narrations CSV'
    )
    parser.add_argument(
        '--food-json',
        default='hdepic_food_nouns_detailed.json',
        help='Path to hdepic_food_nouns_detailed.json (from Step 1)'
    )
    parser.add_argument(
        '--output',
        default='hdepic_p01_food_items.json',
        help='Output JSON file'
    )

    args = parser.parse_args()

    print("=" * 80)
    print("STEP 2: Extract Food Items from P01 Narrations")
    print("=" * 80)

    print("\nLoading food class IDs...")
    food_class_ids = load_food_class_ids(args.food_json)
    print(f"✓ Loaded {len(food_class_ids)} food classes")

    print("\nProcessing P01 narrations...")
    results = extract_food_from_narrations(
        Path(args.narrations),
        food_class_ids
    )

    print("\nSaving results...")
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Saved to {output_path}")

    print_summary(results)

    print("\n✓ Done! Next step: Run 3_analyze_hdepic_food_per_video.py")


if __name__ == '__main__':
    main()

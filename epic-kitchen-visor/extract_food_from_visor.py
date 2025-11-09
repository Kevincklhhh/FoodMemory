#!/usr/bin/env python3
"""
Extract food items from VISOR annotations with their first appearance timestamps.

For each video in VISOR:
- Identifies all food items present
- Records the first frame where each food item appears
- Calculates timestamp ranges for food interactions

Output: Comprehensive mapping of video -> food items with timestamps
"""

import json
import csv
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict
import re


def load_food_nouns(food_file: str) -> Set[str]:
    """Load the list of food nouns from the classification results."""
    food_nouns = set()
    with open(food_file, 'r') as f:
        for line in f:
            noun = line.strip()
            if noun:
                food_nouns.add(noun.lower())
    return food_nouns


def extract_frame_number(frame_name: str) -> int:
    """
    Extract frame number from frame filename.
    Example: "P01_01_frame_0000000140.jpg" -> 140
    """
    match = re.search(r'frame_(\d+)', frame_name)
    if match:
        return int(match.group(1))
    return 0


def frame_to_timestamp(frame_num: int, fps: float = 60.0) -> float:
    """Convert frame number to timestamp in seconds."""
    return frame_num / fps


def is_food_item(object_name: str, food_nouns: Set[str]) -> bool:
    """
    Check if an object name matches a food noun.
    Handles variations and compound names.
    """
    object_lower = object_name.lower()

    # Direct match
    if object_lower in food_nouns:
        return True

    # Check if any food noun is in the object name
    for food in food_nouns:
        if food in object_lower:
            return True

    # Handle colon-separated names (e.g., "bean:green")
    if ':' in object_lower:
        parts = object_lower.split(':')
        for part in parts:
            if part in food_nouns:
                return True

    return False


def analyze_visor_video(json_file: Path, food_nouns: Set[str]) -> Dict:
    """
    Analyze a single VISOR annotation file to extract food items.

    Returns:
        Dictionary with video info and food items with their first appearances
    """
    with open(json_file, 'r') as f:
        data = json.load(f)

    video_id = json_file.stem  # e.g., "P01_01"
    participant_id = video_id.split('_')[0]

    # Track first appearance of each food item
    food_first_appearance = {}
    food_frame_ranges = defaultdict(list)

    # Sort frames by frame number to ensure chronological order
    frames = sorted(data['video_annotations'], key=lambda x: extract_frame_number(x['image']['name']))

    for frame_data in frames:
        frame_name = frame_data['image']['name']
        frame_num = extract_frame_number(frame_name)
        timestamp = frame_to_timestamp(frame_num)

        # Check each object in this frame
        for obj in frame_data['annotations']:
            obj_name = obj['name']

            # Check if this is a food item
            if is_food_item(obj_name, food_nouns):
                # Normalize name for grouping
                obj_key = obj_name.lower()

                # Record first appearance
                if obj_key not in food_first_appearance:
                    food_first_appearance[obj_key] = {
                        'object_name': obj_name,
                        'first_frame': frame_num,
                        'first_timestamp': timestamp,
                        'frame_name': frame_name,
                        'class_id': obj.get('class_id', None)
                    }

                # Track all frames where this food appears
                food_frame_ranges[obj_key].append(frame_num)

    # Calculate appearance ranges for each food item
    food_items = []
    for food_key, first_info in food_first_appearance.items():
        frames = sorted(food_frame_ranges[food_key])

        food_items.append({
            'food_name': first_info['object_name'],
            'first_frame': first_info['first_frame'],
            'first_timestamp': first_info['first_timestamp'],
            'last_frame': frames[-1],
            'last_timestamp': frame_to_timestamp(frames[-1]),
            'total_frames': len(frames),
            'class_id': first_info['class_id']
        })

    return {
        'video_id': video_id,
        'participant_id': participant_id,
        'total_frames_annotated': len(frames),
        'food_items': sorted(food_items, key=lambda x: x['first_frame'])
    }


def process_all_visor_annotations(
    visor_dir: Path,
    food_nouns: Set[str],
    splits: List[str] = ['train', 'val']
) -> Dict[str, Dict]:
    """
    Process all VISOR annotation files across train and val splits.

    Returns:
        Dictionary mapping video_id to video analysis results
    """
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

            result = analyze_visor_video(json_file, food_nouns)
            all_videos[video_id] = result

            print(f" Found {len(result['food_items'])} food items")

    return all_videos


def save_results(
    results: Dict[str, Dict],
    output_dir: Path,
    prefix: str = 'visor_food_mapping'
):
    """Save results in multiple formats for easy analysis."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save detailed JSON
    json_file = output_dir / f'{prefix}.json'
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Saved detailed JSON to {json_file}")

    # 2. Save as CSV (one row per food item per video)
    csv_file = output_dir / f'{prefix}_detailed.csv'
    with open(csv_file, 'w', newline='') as f:
        fieldnames = [
            'video_id', 'participant_id', 'food_name', 'class_id',
            'first_frame', 'first_timestamp', 'last_frame', 'last_timestamp',
            'total_frames', 'duration_seconds'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for video_id, video_data in sorted(results.items()):
            for food in video_data['food_items']:
                duration = food['last_timestamp'] - food['first_timestamp']
                writer.writerow({
                    'video_id': video_id,
                    'participant_id': video_data['participant_id'],
                    'food_name': food['food_name'],
                    'class_id': food['class_id'],
                    'first_frame': food['first_frame'],
                    'first_timestamp': f"{food['first_timestamp']:.2f}",
                    'last_frame': food['last_frame'],
                    'last_timestamp': f"{food['last_timestamp']:.2f}",
                    'total_frames': food['total_frames'],
                    'duration_seconds': f"{duration:.2f}"
                })
    print(f"✓ Saved detailed CSV to {csv_file}")

    # 3. Save summary by video
    summary_file = output_dir / f'{prefix}_by_video.txt'
    with open(summary_file, 'w') as f:
        f.write("VISOR FOOD ITEMS BY VIDEO\n")
        f.write("=" * 80 + "\n\n")

        # Group by participant
        by_participant = defaultdict(list)
        for video_id, video_data in results.items():
            by_participant[video_data['participant_id']].append((video_id, video_data))

        for participant in sorted(by_participant.keys()):
            f.write(f"\n{participant}\n")
            f.write("-" * 80 + "\n")

            for video_id, video_data in sorted(by_participant[participant]):
                f.write(f"\n  {video_id} ({len(video_data['food_items'])} food items)\n")

                if video_data['food_items']:
                    for food in video_data['food_items']:
                        duration = food['last_timestamp'] - food['first_timestamp']
                        f.write(f"    • {food['food_name']:<30} ")
                        f.write(f"First: {food['first_timestamp']:7.2f}s ")
                        f.write(f"Last: {food['last_timestamp']:7.2f}s ")
                        f.write(f"({food['total_frames']} frames, {duration:.1f}s duration)\n")
                else:
                    f.write("    (No food items detected)\n")

    print(f"✓ Saved video summary to {summary_file}")

    # 4. Save food item statistics
    stats_file = output_dir / f'{prefix}_statistics.txt'
    with open(stats_file, 'w') as f:
        f.write("VISOR FOOD ITEM STATISTICS\n")
        f.write("=" * 80 + "\n\n")

        # Count occurrences of each food item
        food_counts = defaultdict(int)
        food_videos = defaultdict(set)

        for video_id, video_data in results.items():
            for food in video_data['food_items']:
                food_name = food['food_name'].lower()
                food_counts[food_name] += 1
                food_videos[food_name].add(video_id)

        f.write(f"Total videos analyzed: {len(results)}\n")
        f.write(f"Videos with food items: {sum(1 for v in results.values() if v['food_items'])}\n")
        f.write(f"Unique food items found: {len(food_counts)}\n\n")

        f.write("Most common food items:\n")
        f.write("-" * 80 + "\n")
        for food, count in sorted(food_counts.items(), key=lambda x: x[1], reverse=True)[:30]:
            video_count = len(food_videos[food])
            f.write(f"  {food:<30} {count:3d} appearances across {video_count:3d} videos\n")

    print(f"✓ Saved statistics to {stats_file}")


def print_summary(results: Dict[str, Dict]):
    """Print summary statistics."""
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    total_videos = len(results)
    videos_with_food = sum(1 for v in results.values() if v['food_items'])
    total_food_occurrences = sum(len(v['food_items']) for v in results.values())

    # Collect unique food items
    unique_foods = set()
    for video_data in results.values():
        for food in video_data['food_items']:
            unique_foods.add(food['food_name'].lower())

    print(f"\nTotal videos analyzed: {total_videos}")
    print(f"Videos with food items: {videos_with_food}")
    print(f"Videos without food: {total_videos - videos_with_food}")
    print(f"Total food item occurrences: {total_food_occurrences}")
    print(f"Unique food items found: {len(unique_foods)}")
    print(f"Average food items per video: {total_food_occurrences/total_videos:.1f}")

    # Show most common foods
    food_counts = defaultdict(int)
    for video_data in results.values():
        for food in video_data['food_items']:
            food_counts[food['food_name'].lower()] += 1

    print("\nTop 10 most common food items:")
    for food, count in sorted(food_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {food:<30} {count:3d} occurrences")


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract food items from VISOR annotations"
    )
    parser.add_argument(
        '--visor-dir',
        default='/home/kailaic/NeuroTrace/kitchen/epic-kitchen-visor/GroundTruth-SparseAnnotations',
        help='Path to VISOR GroundTruth-SparseAnnotations directory'
    )
    parser.add_argument(
        '--food-nouns',
        default='/home/kailaic/NeuroTrace/kitchen/epic-kitchen-visor/epic_food_nouns_names.txt',
        help='Path to food nouns list'
    )
    parser.add_argument(
        '--output-dir',
        default='/home/kailaic/NeuroTrace/kitchen/epic-kitchen-visor',
        help='Output directory for results'
    )
    parser.add_argument(
        '--splits',
        nargs='+',
        default=['train', 'val'],
        help='VISOR splits to process (train, val)'
    )

    args = parser.parse_args()

    print("Loading food nouns...")
    food_nouns = load_food_nouns(args.food_nouns)
    print(f"Loaded {len(food_nouns)} food nouns")

    print("\nProcessing VISOR annotations...")
    results = process_all_visor_annotations(
        Path(args.visor_dir),
        food_nouns,
        args.splits
    )

    print("\nSaving results...")
    save_results(results, Path(args.output_dir))

    print_summary(results)

    print("\n✓ Done!")


if __name__ == '__main__':
    main()

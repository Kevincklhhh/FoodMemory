#!/usr/bin/env python3
"""
Extract food items from VISOR annotations using proper class_id mapping.

CORRECT APPROACH:
1. VISOR annotation.class_id -> EPIC_100_noun_classes_v2.csv
2. Get canonical noun key for that class_id
3. Check if noun key is in our food nouns list

This is more accurate than string matching on object names.
"""

import json
import csv
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict
import re


def load_epic_noun_classes(csv_path: str) -> Dict[int, Dict]:
    """
    Load EPIC-100 noun classes mapping.

    Returns:
        Dictionary mapping class_id -> {key, instances, category}
    """
    noun_classes = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            class_id = int(row['id'])
            noun_classes[class_id] = {
                'key': row['key'],
                'instances': eval(row['instances']),
                'category': row['category']
            }
    return noun_classes


def load_food_nouns(food_file: str) -> Set[str]:
    """Load the list of food noun keys from classification results."""
    food_nouns = set()
    with open(food_file, 'r') as f:
        for line in f:
            noun = line.strip()
            if noun:
                food_nouns.add(noun.lower())
    return food_nouns


def extract_frame_number(frame_name: str) -> int:
    """Extract frame number from frame filename."""
    match = re.search(r'frame_(\d+)', frame_name)
    if match:
        return int(match.group(1))
    return 0


def frame_to_timestamp(frame_num: int, fps: float = 60.0) -> float:
    """Convert frame number to timestamp in seconds."""
    return frame_num / fps


def is_food_class(class_id: int, noun_classes: Dict, food_nouns: Set[str]) -> bool:
    """
    Check if a class_id corresponds to a food item.

    Uses proper EPIC-100 taxonomy mapping:
    class_id -> noun_classes[class_id]['key'] -> check if in food_nouns
    """
    if class_id not in noun_classes:
        return False

    noun_key = noun_classes[class_id]['key'].lower()
    return noun_key in food_nouns


def analyze_visor_video(
    json_file: Path,
    noun_classes: Dict,
    food_nouns: Set[str]
) -> Dict:
    """
    Analyze a single VISOR annotation file using class_id mapping.

    Returns:
        Dictionary with video info and food items with their first appearances
    """
    with open(json_file, 'r') as f:
        data = json.load(f)

    video_id = json_file.stem
    participant_id = video_id.split('_')[0]

    # Track first appearance of each food item by class_id
    food_first_appearance = {}
    food_frame_ranges = defaultdict(list)

    # Sort frames chronologically
    frames = sorted(
        data['video_annotations'],
        key=lambda x: extract_frame_number(x['image']['name'])
    )

    for frame_data in frames:
        frame_name = frame_data['image']['name']
        frame_num = extract_frame_number(frame_name)
        timestamp = frame_to_timestamp(frame_num)

        # Check each object in this frame
        for obj in frame_data['annotations']:
            class_id = obj.get('class_id')

            if class_id is None:
                continue

            # Check if this class_id is food using proper mapping
            if is_food_class(class_id, noun_classes, food_nouns):
                # Use class_id as key to group identical items
                obj_key = class_id

                # Get canonical noun key
                noun_key = noun_classes[class_id]['key']

                # Record first appearance
                if obj_key not in food_first_appearance:
                    food_first_appearance[obj_key] = {
                        'class_id': class_id,
                        'noun_key': noun_key,
                        'object_name': obj['name'],  # Keep original name for reference
                        'category': noun_classes[class_id]['category'],
                        'first_frame': frame_num,
                        'first_timestamp': timestamp,
                        'frame_name': frame_name
                    }

                # Track all frames where this food appears
                food_frame_ranges[obj_key].append(frame_num)

    # Calculate appearance ranges for each food item
    food_items = []
    for class_id, first_info in food_first_appearance.items():
        frames = sorted(food_frame_ranges[class_id])

        food_items.append({
            'class_id': first_info['class_id'],
            'noun_key': first_info['noun_key'],
            'object_name': first_info['object_name'],
            'category': first_info['category'],
            'first_frame': first_info['first_frame'],
            'first_timestamp': first_info['first_timestamp'],
            'last_frame': frames[-1],
            'last_timestamp': frame_to_timestamp(frames[-1]),
            'total_frames': len(frames)
        })

    return {
        'video_id': video_id,
        'participant_id': participant_id,
        'total_frames_annotated': len(frames),
        'food_items': sorted(food_items, key=lambda x: x['first_frame'])
    }


def process_all_visor_annotations(
    visor_dir: Path,
    noun_classes: Dict,
    food_nouns: Set[str],
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

            result = analyze_visor_video(json_file, noun_classes, food_nouns)
            all_videos[video_id] = result

            print(f" Found {len(result['food_items'])} food items")

    return all_videos


def save_results(
    results: Dict[str, Dict],
    output_dir: Path,
    prefix: str = 'visor_food_mapping_v2'
):
    """Save results in multiple formats."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save detailed JSON
    json_file = output_dir / f'{prefix}.json'
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Saved detailed JSON to {json_file}")

    # 2. Save as CSV
    csv_file = output_dir / f'{prefix}_detailed.csv'
    with open(csv_file, 'w', newline='') as f:
        fieldnames = [
            'video_id', 'participant_id', 'class_id', 'noun_key',
            'object_name', 'category',
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
                    'class_id': food['class_id'],
                    'noun_key': food['noun_key'],
                    'object_name': food['object_name'],
                    'category': food['category'],
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
        f.write("VISOR FOOD ITEMS BY VIDEO (using class_id mapping)\n")
        f.write("=" * 80 + "\n\n")

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
                        f.write(f"    • [{food['class_id']:3d}] {food['noun_key']:<20} ")
                        f.write(f"(original: {food['object_name']:<30}) ")
                        f.write(f"{food['first_timestamp']:7.2f}s - {food['last_timestamp']:7.2f}s ")
                        f.write(f"({food['total_frames']} frames)\n")
                else:
                    f.write("    (No food items detected)\n")

    print(f"✓ Saved video summary to {summary_file}")

    # 4. Save statistics
    stats_file = output_dir / f'{prefix}_statistics.txt'
    with open(stats_file, 'w') as f:
        f.write("VISOR FOOD ITEM STATISTICS (class_id based)\n")
        f.write("=" * 80 + "\n\n")

        # Count by noun_key (canonical)
        food_counts = defaultdict(int)
        food_videos = defaultdict(set)

        for video_id, video_data in results.items():
            for food in video_data['food_items']:
                noun_key = food['noun_key']
                food_counts[noun_key] += 1
                food_videos[noun_key].add(video_id)

        f.write(f"Total videos analyzed: {len(results)}\n")
        f.write(f"Videos with food items: {sum(1 for v in results.values() if v['food_items'])}\n")
        f.write(f"Unique food classes found: {len(food_counts)}\n\n")

        f.write("Most common food items (by EPIC-100 noun class):\n")
        f.write("-" * 80 + "\n")
        for noun_key, count in sorted(food_counts.items(), key=lambda x: x[1], reverse=True)[:30]:
            video_count = len(food_videos[noun_key])
            f.write(f"  {noun_key:<30} {count:3d} appearances across {video_count:3d} videos\n")

    print(f"✓ Saved statistics to {stats_file}")


def print_summary(results: Dict[str, Dict]):
    """Print summary statistics."""
    print("\n" + "=" * 80)
    print("SUMMARY (Using Correct class_id Mapping)")
    print("=" * 80)

    total_videos = len(results)
    videos_with_food = sum(1 for v in results.values() if v['food_items'])
    total_food_occurrences = sum(len(v['food_items']) for v in results.values())

    unique_classes = set()
    for video_data in results.values():
        for food in video_data['food_items']:
            unique_classes.add(food['noun_key'])

    print(f"\nTotal videos analyzed: {total_videos}")
    print(f"Videos with food items: {videos_with_food}")
    print(f"Total food item occurrences: {total_food_occurrences}")
    print(f"Unique food noun classes: {len(unique_classes)}")
    print(f"Average food items per video: {total_food_occurrences/total_videos:.1f}")

    # Show most common by canonical noun key
    food_counts = defaultdict(int)
    for video_data in results.values():
        for food in video_data['food_items']:
            food_counts[food['noun_key']] += 1

    print("\nTop 10 most common food classes (by canonical noun):")
    for noun_key, count in sorted(food_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {noun_key:<30} {count:3d} occurrences")


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract food items from VISOR using class_id mapping"
    )
    parser.add_argument(
        '--visor-dir',
        default='/home/kailaic/NeuroTrace/kitchen/epic-kitchen-visor/GroundTruth-SparseAnnotations',
        help='Path to VISOR annotations directory'
    )
    parser.add_argument(
        '--noun-classes',
        default='/home/kailaic/NeuroTrace/kitchen/epic-kitchen-visor/EPIC_100_noun_classes_v2.csv',
        help='Path to EPIC_100_noun_classes_v2.csv'
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
        help='VISOR splits to process'
    )

    args = parser.parse_args()

    print("Loading EPIC-100 noun classes...")
    noun_classes = load_epic_noun_classes(args.noun_classes)
    print(f"Loaded {len(noun_classes)} noun classes")

    print("\nLoading food nouns...")
    food_nouns = load_food_nouns(args.food_nouns)
    print(f"Loaded {len(food_nouns)} food nouns")

    print("\nProcessing VISOR annotations with class_id mapping...")
    results = process_all_visor_annotations(
        Path(args.visor_dir),
        noun_classes,
        food_nouns,
        args.splits
    )

    print("\nSaving results...")
    save_results(results, Path(args.output_dir))

    print_summary(results)

    print("\n✓ Done!")


if __name__ == '__main__':
    main()

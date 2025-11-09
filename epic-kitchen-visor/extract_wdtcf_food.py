#!/usr/bin/env python3
"""
Extract food annotations from WDTCF_GT.json (Where Did This Come From ground truth).

WDTCF format:
- Key: "{video_id}_{object_name}" (e.g., "P01_01_celery")
- Value: {"query": frame_path, "evidence": frame_path, "answer": [location]}

Output: Food items with their video IDs and source locations.
"""

import json
from pathlib import Path
from typing import Dict, Set
from collections import defaultdict
import csv
import re


def load_food_nouns(food_file: str) -> Set[str]:
    """Load the list of food noun keys."""
    food_nouns = set()
    with open(food_file, 'r') as f:
        for line in f:
            noun = line.strip()
            if noun:
                food_nouns.add(noun.lower())
    return food_nouns


def extract_frame_number(frame_path: str) -> int:
    """Extract frame number from frame path."""
    match = re.search(r'frame_(\d+)', frame_path)
    if match:
        return int(match.group(1))
    return 0


def frame_to_timestamp(frame_num: int, fps: float = 60.0) -> float:
    """Convert frame number to timestamp in seconds."""
    return frame_num / fps


def is_food_object(object_name: str, food_nouns: Set[str]) -> bool:
    """
    Check if an object name is a food item.

    Handles variations:
    - Direct match: "milk" -> food
    - Compound names: "bean:green" -> food (if "bean" is food)
    """
    object_lower = object_name.lower()

    # Direct match
    if object_lower in food_nouns:
        return True

    # Handle colon-separated compound names
    if ':' in object_lower:
        parts = object_lower.split(':')
        for part in parts:
            if part in food_nouns:
                return True

    return False


def analyze_wdtcf(wdtcf_file: str, food_nouns: Set[str]) -> Dict:
    """
    Analyze WDTCF_GT.json and extract food annotations.

    Returns:
        Dictionary with:
        - videos: set of video IDs with WDTCF annotations
        - food_annotations: list of food items with metadata
        - by_video: food items grouped by video
    """
    with open(wdtcf_file, 'r') as f:
        data = json.load(f)

    videos = set()
    food_annotations = []
    by_video = defaultdict(list)

    for annotation_key, annotation_data in data.items():
        # Parse key: "P01_01_celery" -> video_id="P01_01", object_name="celery"
        parts = annotation_key.split('_')
        if len(parts) < 3:
            continue

        # video_id is typically "P{participant}_{video_num}"
        video_id = f"{parts[0]}_{parts[1]}"
        object_name = '_'.join(parts[2:])  # Rest is object name

        videos.add(video_id)

        # Check if this object is food
        if is_food_object(object_name, food_nouns):
            query_frame = annotation_data['query']
            evidence_frame = annotation_data['evidence']
            answer_locations = annotation_data['answer']

            # Extract frame numbers for timestamps
            query_frame_num = extract_frame_number(query_frame)
            evidence_frame_num = extract_frame_number(evidence_frame)

            food_item = {
                'annotation_key': annotation_key,
                'video_id': video_id,
                'participant_id': parts[0],
                'object_name': object_name,
                'query_frame': query_frame,
                'query_frame_num': query_frame_num,
                'query_timestamp': frame_to_timestamp(query_frame_num),
                'evidence_frame': evidence_frame,
                'evidence_frame_num': evidence_frame_num,
                'evidence_timestamp': frame_to_timestamp(evidence_frame_num),
                'source_locations': answer_locations
            }

            food_annotations.append(food_item)
            by_video[video_id].append(food_item)

    return {
        'total_annotations': len(data),
        'videos': sorted(videos),
        'food_annotations': food_annotations,
        'by_video': dict(by_video)
    }


def save_results(results: Dict, output_dir: Path):
    """Save WDTCF food extraction results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save detailed JSON
    json_file = output_dir / 'wdtcf_food_annotations.json'
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Saved detailed JSON to {json_file}")

    # 2. Save as CSV
    csv_file = output_dir / 'wdtcf_food_annotations.csv'
    with open(csv_file, 'w', newline='') as f:
        fieldnames = [
            'annotation_key', 'video_id', 'participant_id', 'object_name',
            'query_frame_num', 'query_timestamp',
            'evidence_frame_num', 'evidence_timestamp',
            'source_locations'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for food in results['food_annotations']:
            writer.writerow({
                'annotation_key': food['annotation_key'],
                'video_id': food['video_id'],
                'participant_id': food['participant_id'],
                'object_name': food['object_name'],
                'query_frame_num': food['query_frame_num'],
                'query_timestamp': f"{food['query_timestamp']:.2f}",
                'evidence_frame_num': food['evidence_frame_num'],
                'evidence_timestamp': f"{food['evidence_timestamp']:.2f}",
                'source_locations': ', '.join(food['source_locations'])
            })
    print(f"✓ Saved CSV to {csv_file}")

    # 3. Save summary by video
    summary_file = output_dir / 'wdtcf_food_by_video.txt'
    with open(summary_file, 'w') as f:
        f.write("WDTCF FOOD ANNOTATIONS BY VIDEO\n")
        f.write("=" * 80 + "\n\n")

        by_participant = defaultdict(list)
        for video_id, foods in results['by_video'].items():
            participant = video_id.split('_')[0]
            by_participant[participant].append((video_id, foods))

        for participant in sorted(by_participant.keys()):
            f.write(f"\n{participant}\n")
            f.write("-" * 80 + "\n")

            for video_id, foods in sorted(by_participant[participant]):
                f.write(f"\n  {video_id} ({len(foods)} food items)\n")

                for food in sorted(foods, key=lambda x: x['object_name']):
                    f.write(f"    • {food['object_name']:<20} ")
                    f.write(f"Query: {food['query_timestamp']:7.2f}s ")
                    f.write(f"Evidence: {food['evidence_timestamp']:7.2f}s ")
                    f.write(f"From: {', '.join(food['source_locations'])}\n")

    print(f"✓ Saved video summary to {summary_file}")

    # 4. Save statistics
    stats_file = output_dir / 'wdtcf_food_statistics.txt'
    with open(stats_file, 'w') as f:
        f.write("WDTCF FOOD ANNOTATION STATISTICS\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Total WDTCF annotations: {results['total_annotations']}\n")
        f.write(f"Total videos with WDTCF: {len(results['videos'])}\n")
        f.write(f"Food annotations: {len(results['food_annotations'])}\n")
        f.write(f"Videos with food annotations: {len(results['by_video'])}\n\n")

        # Count by food type
        food_counts = defaultdict(int)
        food_videos = defaultdict(set)
        food_locations = defaultdict(lambda: defaultdict(int))

        for food in results['food_annotations']:
            name = food['object_name']
            food_counts[name] += 1
            food_videos[name].add(food['video_id'])
            for loc in food['source_locations']:
                food_locations[name][loc] += 1

        f.write("Most common food items:\n")
        f.write("-" * 80 + "\n")
        for food, count in sorted(food_counts.items(), key=lambda x: x[1], reverse=True):
            video_count = len(food_videos[food])
            f.write(f"  {food:<30} {count:2d} annotations across {video_count:2d} videos\n")

        f.write("\n\nMost common source locations:\n")
        f.write("-" * 80 + "\n")
        location_counts = defaultdict(int)
        for food in results['food_annotations']:
            for loc in food['source_locations']:
                location_counts[loc] += 1

        for loc, count in sorted(location_counts.items(), key=lambda x: x[1], reverse=True):
            f.write(f"  {loc:<30} {count:3d} occurrences\n")

    print(f"✓ Saved statistics to {stats_file}")


def print_summary(results: Dict):
    """Print summary statistics."""
    print("\n" + "=" * 80)
    print("WDTCF FOOD ANNOTATION SUMMARY")
    print("=" * 80)

    print(f"\nTotal WDTCF annotations: {results['total_annotations']}")
    print(f"Total videos with WDTCF: {len(results['videos'])}")
    print(f"Food annotations: {len(results['food_annotations'])}")
    print(f"Videos with food: {len(results['by_video'])}")

    # Count source locations
    location_counts = defaultdict(int)
    for food in results['food_annotations']:
        for loc in food['source_locations']:
            location_counts[loc] += 1

    print(f"\nSource locations:")
    for loc, count in sorted(location_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {loc:<20} {count:3d} occurrences")

    # Top food items
    food_counts = defaultdict(int)
    for food in results['food_annotations']:
        food_counts[food['object_name']] += 1

    print(f"\nTop 10 food items:")
    for food, count in sorted(food_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {food:<30} {count:2d} annotations")


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract food annotations from WDTCF_GT.json"
    )
    parser.add_argument(
        '--wdtcf-file',
        default='/home/kailaic/NeuroTrace/kitchen/epic-kitchen-visor/WDTCF_GT.json',
        help='Path to WDTCF_GT.json'
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

    args = parser.parse_args()

    print("Loading food nouns...")
    food_nouns = load_food_nouns(args.food_nouns)
    print(f"Loaded {len(food_nouns)} food nouns")

    print("\nAnalyzing WDTCF annotations...")
    results = analyze_wdtcf(args.wdtcf_file, food_nouns)

    print("\nSaving results...")
    save_results(results, Path(args.output_dir))

    print_summary(results)

    print("\n✓ Done!")


if __name__ == '__main__':
    main()

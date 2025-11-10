#!/usr/bin/env python3
"""
Step 3: Analyze extracted food items and generate per-video food lists.

This script:
1. Reads visor_food_items.json from Step 1
2. Analyzes food occurrences per video
3. Generates comprehensive reports:
   - JSON: Detailed food items per video with frame ranges
   - CSV: Tabular format for easy analysis
   - TXT: Human-readable summary

Input: visor_food_items.json (from Step 1)
Output: Multiple formats for food analysis per video
"""

import json
import csv
from pathlib import Path
from collections import defaultdict
from typing import Dict, List


def load_food_items(json_path: str) -> Dict:
    """Load food items from Step 1 output."""
    with open(json_path, 'r') as f:
        return json.load(f)


def analyze_video_food_items(video_data: Dict) -> Dict:
    """Analyze food items for a single video.

    Returns:
        Dictionary with food summary per video including:
        - Unique food items (by class_id)
        - Frame ranges for each food item
        - Total occurrences per food
        - First and last appearance
    """
    video_id = video_data['video_id']
    participant_id = video_data['participant_id']
    food_occurrences = video_data['food_occurrences']

    # Group by class_id to get unique food items
    food_by_class = defaultdict(list)

    for occurrence in food_occurrences:
        class_id = occurrence['class_id']
        food_by_class[class_id].append(occurrence)

    # Analyze each unique food item
    food_items = []
    for class_id, occurrences in food_by_class.items():
        # Sort by frame number
        sorted_occurrences = sorted(occurrences, key=lambda x: x['frame_number'])

        first_occ = sorted_occurrences[0]
        last_occ = sorted_occurrences[-1]

        # Get all frame numbers where this food appears
        frame_numbers = [occ['frame_number'] for occ in sorted_occurrences]

        # Calculate frame ranges (consecutive frames grouped)
        frame_ranges = []
        if frame_numbers:
            range_start = frame_numbers[0]
            range_end = frame_numbers[0]

            for frame_num in frame_numbers[1:]:
                if frame_num == range_end + 1:
                    # Consecutive frame
                    range_end = frame_num
                else:
                    # Gap detected, save current range
                    frame_ranges.append([range_start, range_end])
                    range_start = frame_num
                    range_end = frame_num

            # Add final range
            frame_ranges.append([range_start, range_end])

        food_items.append({
            'class_id': class_id,
            'noun_key': first_occ['noun_key'],
            'object_names': list(set(occ['object_name'] for occ in occurrences)),
            'total_occurrences': len(occurrences),
            'first_frame': first_occ['frame_number'],
            'last_frame': last_occ['frame_number'],
            'frame_count': len(frame_numbers),
            'frame_ranges': frame_ranges,
            'frame_numbers': frame_numbers
        })

    # Sort by first appearance
    food_items.sort(key=lambda x: x['first_frame'])

    return {
        'video_id': video_id,
        'participant_id': participant_id,
        'total_frames_annotated': video_data['total_frames_annotated'],
        'total_food_occurrences': len(food_occurrences),
        'unique_food_items': len(food_items),
        'food_items': food_items
    }


def generate_summary_json(results: Dict, output_path: Path):
    """Generate JSON summary with food items per video."""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Saved JSON summary to {output_path}")


def generate_summary_csv(results: Dict, output_path: Path):
    """Generate CSV with one row per food item per video."""
    with open(output_path, 'w', newline='') as f:
        fieldnames = [
            'video_id', 'participant_id', 'class_id', 'noun_key',
            'object_names', 'total_occurrences', 'frame_count',
            'first_frame', 'last_frame', 'frame_ranges'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for video_id, video_analysis in sorted(results.items()):
            for food in video_analysis['food_items']:
                writer.writerow({
                    'video_id': video_id,
                    'participant_id': video_analysis['participant_id'],
                    'class_id': food['class_id'],
                    'noun_key': food['noun_key'],
                    'object_names': ', '.join(food['object_names']),
                    'total_occurrences': food['total_occurrences'],
                    'frame_count': food['frame_count'],
                    'first_frame': food['first_frame'],
                    'last_frame': food['last_frame'],
                    'frame_ranges': str(food['frame_ranges'])
                })

    print(f"✓ Saved CSV summary to {output_path}")


def generate_summary_txt(results: Dict, output_path: Path):
    """Generate human-readable text summary."""
    with open(output_path, 'w') as f:
        f.write("VISOR FOOD ITEMS PER VIDEO\n")
        f.write("=" * 80 + "\n\n")

        # Group by participant
        by_participant = defaultdict(list)
        for video_id, video_analysis in results.items():
            by_participant[video_analysis['participant_id']].append((video_id, video_analysis))

        for participant in sorted(by_participant.keys()):
            f.write(f"\n{participant}\n")
            f.write("-" * 80 + "\n")

            for video_id, video_analysis in sorted(by_participant[participant]):
                f.write(f"\n  {video_id}\n")
                f.write(f"    Total frames: {video_analysis['total_frames_annotated']}\n")
                f.write(f"    Food occurrences: {video_analysis['total_food_occurrences']}\n")
                f.write(f"    Unique food items: {video_analysis['unique_food_items']}\n\n")

                if video_analysis['food_items']:
                    f.write("    Food items:\n")
                    for food in video_analysis['food_items']:
                        # Format frame ranges nicely
                        range_strs = []
                        for start, end in food['frame_ranges']:
                            if start == end:
                                range_strs.append(f"{start}")
                            else:
                                range_strs.append(f"{start}-{end}")
                        ranges_display = ', '.join(range_strs)

                        f.write(f"      • {food['noun_key']:<20} ")
                        f.write(f"[class_id: {food['class_id']:3d}]  ")
                        f.write(f"{food['total_occurrences']:4d} occurrences  ")
                        f.write(f"({food['frame_count']} frames)\n")
                        f.write(f"        Variants: {', '.join(food['object_names'])}\n")
                        f.write(f"        Frames: {ranges_display}\n")
                else:
                    f.write("    (No food items detected)\n")

                f.write("\n")

    print(f"✓ Saved text summary to {output_path}")


def generate_video_list_only(results: Dict, output_path: Path):
    """Generate simple list of food items per video (concise format)."""
    with open(output_path, 'w') as f:
        f.write("FOOD ITEMS BY VIDEO (Simple List)\n")
        f.write("=" * 80 + "\n\n")

        for video_id, video_analysis in sorted(results.items()):
            food_list = [food['noun_key'] for food in video_analysis['food_items']]
            f.write(f"{video_id}: {', '.join(food_list)}\n")

    print(f"✓ Saved simple list to {output_path}")


def print_statistics(results: Dict):
    """Print summary statistics."""
    print("\n" + "=" * 80)
    print("ANALYSIS STATISTICS")
    print("=" * 80)

    total_videos = len(results)
    videos_with_food = sum(1 for v in results.values() if v['food_items'])
    total_unique_foods = sum(v['unique_food_items'] for v in results.values())
    total_occurrences = sum(v['total_food_occurrences'] for v in results.values())

    # Global food counts
    global_food_counts = defaultdict(int)
    global_video_counts = defaultdict(set)

    for video_id, video_analysis in results.items():
        for food in video_analysis['food_items']:
            global_food_counts[food['noun_key']] += food['total_occurrences']
            global_video_counts[food['noun_key']].add(video_id)

    print(f"\nTotal videos: {total_videos}")
    print(f"Videos with food: {videos_with_food}")
    print(f"Total food occurrences: {total_occurrences}")
    print(f"Total unique food items across all videos: {total_unique_foods}")
    print(f"Unique food classes globally: {len(global_food_counts)}")
    print(f"Average food items per video: {total_unique_foods/total_videos:.1f}")

    print("\nTop 15 most common food items:")
    for noun_key, count in sorted(global_food_counts.items(), key=lambda x: x[1], reverse=True)[:15]:
        video_count = len(global_video_counts[noun_key])
        print(f"  {noun_key:<20} {count:5d} occurrences in {video_count:3d} videos")

    print("\nVideos with most unique food items:")
    top_videos = sorted(results.items(), key=lambda x: x[1]['unique_food_items'], reverse=True)[:10]
    for video_id, video_analysis in top_videos:
        print(f"  {video_id:<15} {video_analysis['unique_food_items']:3d} unique food items")


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze food items per video (Step 3)"
    )
    parser.add_argument(
        '--input',
        default='visor_food_items.json',
        help='Input JSON from Step 1 (default: visor_food_items.json)'
    )
    parser.add_argument(
        '--output-prefix',
        default='food_per_video',
        help='Output file prefix (default: food_per_video)'
    )

    args = parser.parse_args()

    print("=" * 80)
    print("STEP 3: Analyze Food Items Per Video")
    print("=" * 80)

    # Load food items
    print(f"\nLoading food items from {args.input}...")
    all_videos = load_food_items(args.input)
    print(f"✓ Loaded {len(all_videos)} videos")

    # Analyze each video
    print("\nAnalyzing food items per video...")
    results = {}
    for video_id, video_data in all_videos.items():
        results[video_id] = analyze_video_food_items(video_data)
    print(f"✓ Analyzed {len(results)} videos")

    # Generate outputs
    print("\nGenerating output files...")

    # JSON format (detailed)
    json_output = Path(f"{args.output_prefix}.json")
    generate_summary_json(results, json_output)

    # CSV format (tabular)
    csv_output = Path(f"{args.output_prefix}.csv")
    generate_summary_csv(results, csv_output)

    # Text format (human-readable)
    txt_output = Path(f"{args.output_prefix}.txt")
    generate_summary_txt(results, txt_output)

    # Simple list
    simple_output = Path(f"{args.output_prefix}_simple.txt")
    generate_video_list_only(results, simple_output)

    # Print statistics
    print_statistics(results)

    print("\n" + "=" * 80)
    print("✓ Analysis complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()

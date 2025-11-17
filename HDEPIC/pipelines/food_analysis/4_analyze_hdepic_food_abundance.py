#!/usr/bin/env python3
"""
Step 4: Analyze food abundance across P01 videos.

This script:
1. Analyzes which food items appear in which videos
2. Calculates statistics about food distribution
3. Generates reports about food availability

Input: hdepic_food_per_video.json (from Step 3)
Output: Food abundance analysis
"""

import json
from collections import defaultdict
from typing import Dict, List


def analyze_food_abundance(food_per_video_path: str = 'hdepic_food_per_video.json') -> Dict:
    """Analyze food abundance across videos.

    Returns:
        Dictionary with food class statistics
    """
    # Load food per video data
    print(f"Loading food per video data from {food_per_video_path}...")
    with open(food_per_video_path, 'r') as f:
        videos = json.load(f)
    print(f"✓ Loaded {len(videos)} videos")

    # Build food abundance statistics
    food_stats = defaultdict(lambda: {
        'videos': set(),
        'video_occurrence_count': defaultdict(int),
        'total_occurrences': 0
    })

    for video_id, video_data in videos.items():
        # Get all food classes in this video
        for food_item in video_data['food_items']:
            food_class = food_item['noun_key']

            # Add to statistics
            food_stats[food_class]['videos'].add(video_id)
            food_stats[food_class]['video_occurrence_count'][video_id] = food_item['total_occurrences']
            food_stats[food_class]['total_occurrences'] += food_item['total_occurrences']

    # Convert to serializable format and calculate metrics
    result = {}

    for food_class, stats in sorted(food_stats.items()):
        videos_count = len(stats['videos'])
        total_occurrences = stats['total_occurrences']

        # Calculate average occurrences per video
        avg_per_video = total_occurrences / videos_count if videos_count > 0 else 0

        # Find max occurrences in a single video
        max_in_video = max(stats['video_occurrence_count'].values()) if stats['video_occurrence_count'] else 0

        result[food_class] = {
            'total_videos': videos_count,
            'total_occurrences': total_occurrences,
            'avg_occurrences_per_video': round(avg_per_video, 2),
            'max_occurrences_in_single_video': max_in_video,
            'videos': sorted(list(stats['videos'])),
            'video_occurrence_distribution': dict(stats['video_occurrence_count'])
        }

    return result


def print_abundance_table(food_abundance: Dict):
    """Print food abundance statistics as a table."""
    print("\n" + "=" * 100)
    print("FOOD ABUNDANCE ANALYSIS: P01 Videos")
    print("=" * 100)
    print("\nGoal: Understand distribution of food items across P01 videos")
    print("=" * 100)

    # Sort by total occurrences
    sorted_foods = sorted(
        food_abundance.items(),
        key=lambda x: x[1]['total_occurrences'],
        reverse=True
    )

    print(f"\n{'Food Class':<25} {'Videos':<8} {'Total':<8} {'Avg/Video':<12} {'Max/Video':<10}")
    print("-" * 100)

    for food_class, stats in sorted_foods:
        print(
            f"{food_class:<25} "
            f"{stats['total_videos']:<8} "
            f"{stats['total_occurrences']:<8} "
            f"{stats['avg_occurrences_per_video']:<12.2f} "
            f"{stats['max_occurrences_in_single_video']:<10}"
        )

    # Summary statistics
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f"Total unique food classes: {len(food_abundance)}")
    print(f"Total food occurrences: {sum(s['total_occurrences'] for s in food_abundance.values())}")
    print(f"Total videos analyzed: {len(set(v for s in food_abundance.values() for v in s['videos']))}")


def generate_food_abundance_table(food_abundance: Dict, output_csv: str):
    """Generate CSV table of food abundance."""
    import csv

    with open(output_csv, 'w', newline='') as f:
        fieldnames = [
            'food_class', 'total_videos', 'total_occurrences',
            'avg_occurrences_per_video', 'max_occurrences_in_single_video', 'videos'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for food_class, stats in sorted(food_abundance.items(), key=lambda x: x[1]['total_occurrences'], reverse=True):
            writer.writerow({
                'food_class': food_class,
                'total_videos': stats['total_videos'],
                'total_occurrences': stats['total_occurrences'],
                'avg_occurrences_per_video': stats['avg_occurrences_per_video'],
                'max_occurrences_in_single_video': stats['max_occurrences_in_single_video'],
                'videos': ', '.join(stats['videos'])
            })

    print(f"✓ Saved abundance table to {output_csv}")


def print_detailed_breakdown(food_abundance: Dict):
    """Print detailed breakdown of top food items."""
    print("\n" + "=" * 100)
    print("DETAILED BREAKDOWN: Top 10 Food Items")
    print("=" * 100)

    # Sort by total occurrences
    sorted_foods = sorted(
        food_abundance.items(),
        key=lambda x: x[1]['total_occurrences'],
        reverse=True
    )[:10]

    for food_class, stats in sorted_foods:
        print(f"\n{food_class.upper()}")
        print(f"  Total occurrences: {stats['total_occurrences']}")
        print(f"  Appears in {stats['total_videos']} videos")
        print(f"  Average per video: {stats['avg_occurrences_per_video']:.2f}")
        print(f"  Videos: {', '.join(stats['videos'])}")

        # Show distribution
        print(f"  Distribution:")
        for video_id, count in sorted(stats['video_occurrence_distribution'].items(), key=lambda x: x[1], reverse=True):
            print(f"    {video_id}: {count} occurrences")


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze food abundance across P01 videos (Step 4)"
    )
    parser.add_argument(
        '--food-per-video',
        default='hdepic_food_per_video.json',
        help='Food per video JSON (from Step 3)'
    )
    parser.add_argument(
        '--output',
        default='hdepic_food_abundance_analysis.json',
        help='Output JSON file'
    )
    parser.add_argument(
        '--output-csv',
        default='hdepic_food_abundance_table.csv',
        help='Output CSV table'
    )

    args = parser.parse_args()

    print("=" * 100)
    print("STEP 4: Food Abundance Analysis")
    print("=" * 100)

    # Analyze
    food_abundance = analyze_food_abundance(args.food_per_video)

    # Print statistics
    print_abundance_table(food_abundance)
    print_detailed_breakdown(food_abundance)

    # Save results
    print(f"\n\nSaving results to {args.output}...")
    with open(args.output, 'w') as f:
        json.dump(food_abundance, f, indent=2)
    print(f"✓ Saved to {args.output}")

    # Generate CSV table
    generate_food_abundance_table(food_abundance, args.output_csv)

    print("\n" + "=" * 100)
    print("✓ Analysis complete!")
    print("=" * 100)


if __name__ == '__main__':
    main()

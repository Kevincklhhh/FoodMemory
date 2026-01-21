#!/usr/bin/env python3
"""
Find Non-Recipe Videos

Identifies videos for a participant that are NOT used in any recipe.
These videos can be safely skipped/deleted since they won't be processed
by the inventory pipeline.

This script works for ALL participants (doesn't require discovery files).

Usage:
    python find_non_recipe_videos.py --participant P05
    python find_non_recipe_videos.py --all
    python find_non_recipe_videos.py --all --save
"""

import argparse
import json
from pathlib import Path

from inventory_utils import (
    DEFAULT_OUTPUT_DIR,
    load_pickle_data,
    load_recipes,
)


def get_all_videos_for_participant(participant: str) -> set:
    """Get all video IDs for a participant from the narrations pickle."""
    df = load_pickle_data()
    video_ids = df[df['video_id'].str.startswith(f"{participant}-")]['video_id'].unique()
    return set(video_ids)


def get_recipe_videos_for_participant(participant: str) -> set:
    """Get all video IDs used in recipes for a participant."""
    recipes = load_recipes()
    video_ids = set()

    for recipe_id, recipe in recipes.items():
        if not recipe_id.startswith(f"{participant}_"):
            continue
        for capture in recipe.get('captures', []):
            video_ids.update(capture.get('videos', []))

    return video_ids


def get_video_sizes(video_ids: list, base_path: Path) -> dict:
    """Get file sizes for videos if they exist."""
    sizes = {}
    for vid in video_ids:
        participant = vid.split('-')[0]
        video_path = base_path / participant / f"{vid}.mp4"
        if video_path.exists():
            sizes[vid] = video_path.stat().st_size
        else:
            # Try uppercase extension
            video_path = base_path / participant / f"{vid}.MP4"
            if video_path.exists():
                sizes[vid] = video_path.stat().st_size
    return sizes


def analyze_participant(participant: str, video_base_path: Path = None, verbose: bool = False) -> dict:
    """Analyze a single participant's video coverage."""
    print(f"\n{'='*60}")
    print(f"PARTICIPANT: {participant}")
    print(f"{'='*60}")

    # Get all videos from narrations
    all_videos = get_all_videos_for_participant(participant)
    print(f"Total videos in narrations: {len(all_videos)}")

    # Get videos used in recipes
    recipe_videos = get_recipe_videos_for_participant(participant)
    print(f"Videos in recipes: {len(recipe_videos)}")

    # Find non-recipe videos
    non_recipe_videos = all_videos - recipe_videos

    print(f"Non-recipe videos (safe to skip/delete): {len(non_recipe_videos)}")

    # Get sizes if video path provided
    sizes = {}
    total_size = 0
    if video_base_path and video_base_path.exists():
        sizes = get_video_sizes(sorted(non_recipe_videos), video_base_path)
        total_size = sum(sizes.values())
        if total_size > 0:
            print(f"Total size: {total_size / 1024 / 1024:.0f} MB ({total_size / 1024 / 1024 / 1024:.1f} GB)")

    if verbose or len(non_recipe_videos) <= 15:
        print(f"\n  Non-recipe videos:")
        for vid in sorted(non_recipe_videos):
            size_str = ""
            if vid in sizes:
                size_str = f"  # {sizes[vid] / 1024 / 1024:.0f} MB"
            print(f"    {vid}{size_str}")

    return {
        'participant': participant,
        'total_videos': len(all_videos),
        'recipe_videos': len(recipe_videos),
        'non_recipe_total': len(non_recipe_videos),
        'non_recipe_size_bytes': total_size,
        'non_recipe_videos': sorted(non_recipe_videos),
        'recipe_videos_list': sorted(recipe_videos),
        'all_videos': sorted(all_videos),
        'sizes': {k: v for k, v in sorted(sizes.items())},
    }


def main():
    parser = argparse.ArgumentParser(
        description="Find videos not used in any recipe (safe to skip/delete)"
    )
    parser.add_argument(
        '--participant', '-p',
        help='Participant ID (e.g., P05). Use --all for all participants.'
    )
    parser.add_argument(
        '--all', '-a',
        action='store_true',
        help='Analyze all participants'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help='Output directory for saving results'
    )
    parser.add_argument(
        '--video-path',
        type=Path,
        default=Path('data/HD-EPIC/Videos'),
        help='Path to video files (for size calculation)'
    )
    parser.add_argument(
        '--save',
        action='store_true',
        help='Save results to JSON and TXT files'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show all non-recipe videos even if there are many'
    )

    args = parser.parse_args()

    if not args.participant and not args.all:
        parser.error("Must specify --participant or --all")

    results = []

    if args.all:
        # All participants P01-P09
        participants = [f"P0{i}" for i in range(1, 10)]
    else:
        participants = [args.participant]

    for participant in participants:
        result = analyze_participant(
            participant,
            args.video_path if args.video_path.exists() else None,
            args.verbose
        )
        results.append(result)

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Participant':<12} {'Total':<8} {'Recipe':<8} {'Non-Recipe':<12} {'Size':<10}")
    print("-" * 60)

    total_non_recipe = 0
    total_size = 0
    for r in results:
        size_str = "-"
        if r['non_recipe_size_bytes'] > 0:
            size_str = f"{r['non_recipe_size_bytes'] / 1024 / 1024:.0f} MB"
        print(f"{r['participant']:<12} {r['total_videos']:<8} {r['recipe_videos']:<8} {r['non_recipe_total']:<12} {size_str:<10}")
        total_non_recipe += r['non_recipe_total']
        total_size += r['non_recipe_size_bytes']

    print("-" * 60)
    size_str = f"{total_size / 1024 / 1024:.0f} MB" if total_size > 0 else "-"
    print(f"{'TOTAL':<12} {'':<8} {'':<8} {total_non_recipe:<12} {size_str:<10}")

    if args.save:
        # Save JSON report
        json_path = args.output_dir / "non_recipe_videos_report.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved JSON report to: {json_path}")

        # Save simple text list
        txt_path = args.output_dir / "non_recipe_videos.txt"
        with open(txt_path, 'w') as f:
            f.write("# Non-Recipe Videos (safe to skip/delete)\n")
            f.write("# These videos are not used in any recipe\n")
            f.write(f"# Total: {total_non_recipe} videos")
            if total_size > 0:
                f.write(f" (~{total_size / 1024 / 1024 / 1024:.1f} GB)")
            f.write("\n\n")

            for r in results:
                if r['non_recipe_videos']:
                    size_str = ""
                    if r['non_recipe_size_bytes'] > 0:
                        size_str = f", ~{r['non_recipe_size_bytes'] / 1024 / 1024:.0f} MB"
                    f.write(f"# {r['participant']} ({len(r['non_recipe_videos'])} videos{size_str})\n")
                    for vid in r['non_recipe_videos']:
                        size_comment = ""
                        if vid in r['sizes']:
                            size_comment = f"    # {r['sizes'][vid] / 1024 / 1024:.0f} MB"
                        f.write(f"{vid}{size_comment}\n")
                    f.write("\n")

        print(f"Saved video list to: {txt_path}")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Find Irrelevant Videos

Identifies videos for a participant that are NOT referenced in any item's
video_range in the discovery_edit.json file.

These are videos that don't contain any discovered food items (likely
non-cooking videos like setup, breaks, etc.)

Usage:
    python find_irrelevant_videos.py --participant P01
    python find_irrelevant_videos.py --all
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict

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


def get_discovery_videos(discovery_path: Path) -> set:
    """Get all video IDs referenced in items' video_range from discovery_edit.json."""
    if not discovery_path.exists():
        return set()

    with open(discovery_path, 'r') as f:
        data = json.load(f)

    video_ids = set()
    for item in data.get('items', []):
        video_ids.update(item.get('video_range', []))

    return video_ids


def analyze_participant(participant: str, output_dir: Path, verbose: bool = False) -> dict:
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

    # Get videos from discovery
    discovery_path = output_dir / participant / f"{participant}_discovery_edit.json"
    if not discovery_path.exists():
        # Try without _edit suffix
        discovery_path = output_dir / participant / f"{participant}_discovery.json"

    discovery_videos = get_discovery_videos(discovery_path)
    print(f"Videos in discovery items: {len(discovery_videos)}")

    # Find irrelevant videos (in narrations but not in any item's video_range)
    irrelevant = all_videos - discovery_videos

    # Further categorize irrelevant videos
    irrelevant_but_in_recipes = irrelevant & recipe_videos
    irrelevant_and_not_in_recipes = irrelevant - recipe_videos

    # Videos in recipes but not discovered
    in_recipes_not_discovered = recipe_videos - discovery_videos

    print(f"\nIrrelevant videos (not in any item's video_range): {len(irrelevant)}")
    if irrelevant_but_in_recipes:
        print(f"  - In recipes but no items discovered: {len(irrelevant_but_in_recipes)}")
    if irrelevant_and_not_in_recipes:
        print(f"  - Not in any recipe: {len(irrelevant_and_not_in_recipes)}")

    if verbose or len(irrelevant) <= 10:
        if irrelevant_but_in_recipes:
            print(f"\n  Videos in recipes but no items discovered:")
            for vid in sorted(irrelevant_but_in_recipes):
                print(f"    - {vid}")

        if irrelevant_and_not_in_recipes:
            print(f"\n  Videos not in any recipe (likely non-cooking):")
            for vid in sorted(irrelevant_and_not_in_recipes):
                print(f"    - {vid}")

    return {
        'participant': participant,
        'total_videos': len(all_videos),
        'recipe_videos': len(recipe_videos),
        'discovery_videos': len(discovery_videos),
        'irrelevant_total': len(irrelevant),
        'irrelevant_but_in_recipes': sorted(irrelevant_but_in_recipes),
        'irrelevant_not_in_recipes': sorted(irrelevant_and_not_in_recipes),
        'all_videos': sorted(all_videos),
        'discovery_videos_list': sorted(discovery_videos),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Find videos not referenced in discovery items"
    )
    parser.add_argument(
        '--participant', '-p',
        help='Participant ID (e.g., P01). Use --all for all participants.'
    )
    parser.add_argument(
        '--all', '-a',
        action='store_true',
        help='Analyze all participants with discovery files'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help='Output directory containing discovery files'
    )
    parser.add_argument(
        '--save',
        action='store_true',
        help='Save results to JSON file'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show all irrelevant videos even if there are many'
    )

    args = parser.parse_args()

    if not args.participant and not args.all:
        parser.error("Must specify --participant or --all")

    results = []

    if args.all:
        # Find all participants with discovery files
        discovery_files = list(args.output_dir.glob("P*/*_discovery*.json"))
        participants = sorted(set(f.parent.name for f in discovery_files))
        print(f"Found discovery files for: {', '.join(participants)}")
    else:
        participants = [args.participant]

    for participant in participants:
        result = analyze_participant(participant, args.output_dir, args.verbose)
        results.append(result)

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Participant':<12} {'Total':<8} {'Recipe':<8} {'Discovered':<12} {'Irrelevant':<10}")
    print("-" * 60)
    for r in results:
        print(f"{r['participant']:<12} {r['total_videos']:<8} {r['recipe_videos']:<8} {r['discovery_videos']:<12} {r['irrelevant_total']:<10}")

    if args.save:
        output_path = args.output_dir / "irrelevant_videos_report.json"
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved report to: {output_path}")


if __name__ == '__main__':
    main()

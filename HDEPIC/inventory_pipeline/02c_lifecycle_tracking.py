#!/usr/bin/env python3
"""
Step 2: Lifecycle Tracking

Reads user-verified inventory from {participant}_discovery_edit.json and tracks
lifecycle events (RETRIEVAL, ACCESS, DISPENSING, RESTOCKING) for each included item.

Prerequisites:
    Run 02b_inventory_discovery.py first to generate _discovery_edit.json file.
    Review and edit that file to exclude non-food items (set "include": false).

Usage:
    # Process all items for a participant
    python 02c_lifecycle_tracking.py --participant P03

    # Test with first N items
    python 02c_lifecycle_tracking.py --participant P03 --limit 3

Inputs:
    {participant}/{participant}_discovery_edit.json : User-verified inventory (from Step 1)

Outputs:
    {participant}/{participant}_lifecycle.json : Lifecycle events for each item
"""

import argparse
from pathlib import Path
from collections import defaultdict

from inventory_utils import (
    GPTClient,
    DEFAULT_OUTPUT_DIR,
    load_raw_narrations_for_videos,
    run_lifecycle_for_item,
)
import json


def main():
    parser = argparse.ArgumentParser(
        description="Step 2: Lifecycle Tracking - Track lifecycle events for verified inventory"
    )
    parser.add_argument(
        '--participant',
        required=True,
        help='Participant ID to process (e.g., P03)'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help='Output directory for results'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit processing to first N items (for testing)'
    )
    parser.add_argument(
        '--model',
        default='gpt-5.2',
        choices=['gpt-4o', 'o4-mini', 'gpt-4.1-mini', 'gpt-5', 'gpt-5.2', 'o3'],
        help='Model to use (default: gpt-5.2)'
    )
    parser.add_argument(
        '--reasoning',
        action='store_true',
        help='Enable reasoning mode'
    )
    parser.add_argument(
        '--reasoning-effort',
        choices=['low', 'medium', 'high'],
        default='high',
        help='Reasoning effort level (default: high)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Print verbose output'
    )
    parser.add_argument(
        '--discovery-file',
        type=str,
        default=None,
        help='Custom discovery file name (default: {participant}_discovery_edit.json)'
    )

    args = parser.parse_args()
    participant = args.participant

    # Load discovery file (custom or default)
    if args.discovery_file:
        discovery_file = args.output_dir / participant / args.discovery_file
    else:
        discovery_file = args.output_dir / participant / f"{participant}_discovery_edit.json"
    if not discovery_file.exists():
        print(f"ERROR: Discovery file not found: {discovery_file}")
        print(f"       Run 02b_inventory_discovery.py --participant {participant} first")
        return

    print(f"Loading {discovery_file.name}...")
    with open(discovery_file, 'r') as f:
        discovery_data = json.load(f)

    # Filter items with include=True
    all_items = discovery_data.get('items', [])
    items = [item for item in all_items if item.get('include', True)]
    excluded_count = len(all_items) - len(items)

    print(f"  Total items: {len(all_items)}")
    print(f"  Included: {len(items)}, Excluded: {excluded_count}")

    if not items:
        print("ERROR: No items to process (all excluded or empty)")
        return

    # Apply limit if specified
    if args.limit:
        items = items[:args.limit]
        print(f"  Limited to first {args.limit} items for testing")

    # Group items by video_range to optimize narration loading
    # Convert video_range list to tuple for hashing
    range_to_items = defaultdict(list)
    for item in items:
        video_range = tuple(sorted(item.get('video_range', [])))
        range_to_items[video_range].append(item)

    print(f"\n  {len(items)} items across {len(range_to_items)} unique video ranges")

    # Initialize API
    print(f"\nInitializing {args.model} API" + (" with reasoning..." if args.reasoning else "..."))
    try:
        api = GPTClient(args.model, use_reasoning=args.reasoning)
        print("  API initialized successfully")
    except Exception as e:
        print(f"ERROR: Failed to initialize API: {e}")
        return

    print(f"\n{'='*70}")
    print(f"LIFECYCLE TRACKING: {participant}")
    print(f"{'='*70}")

    # Cache for narrations by video_range
    narrations_cache = {}

    # Process items, grouped by video_range
    all_results = {}
    item_idx = 0

    for video_range, range_items in sorted(range_to_items.items()):
        videos = list(video_range)

        # Load narrations for this range (with caching)
        if video_range not in narrations_cache:
            narrations = load_raw_narrations_for_videos(videos)
            if narrations is None:
                print(f"\n  WARNING: No narrations for {videos}")
                narrations = ""
            narrations_cache[video_range] = narrations

        narrations = narrations_cache[video_range]
        line_count = len(narrations.strip().split('\n')) if narrations else 0

        print(f"\n--- Video Range: {', '.join(videos)} ({line_count} lines) ---")

        # Process each item in this range
        for item in range_items:
            item_idx += 1
            narration_id = item.get('narration_id', 'unknown')
            food_name = item.get('food_name', 'unknown')

            print(f"\n  [{item_idx}/{len(items)}] {food_name}")
            print(f"       narration_id: {narration_id}")

            if not narrations:
                print("       SKIP: No narrations available")
                all_results[narration_id] = {
                    'food_name': food_name,
                    'video_range': videos,
                    'events': [],
                    'error': 'No narrations available'
                }
                continue

            # Run lifecycle tracking for this item
            print(f"       Calling GPT for lifecycle events...", end=" ", flush=True)
            try:
                events = run_lifecycle_for_item(
                    api, food_name, narrations, args.verbose,
                    args.reasoning, args.reasoning_effort
                )

                if events:
                    print(f"found {len(events)} events")
                    if args.verbose:
                        for evt in events:
                            print(f"         - {evt.get('stage')}: {evt.get('narration_id')}")
                else:
                    print("no events found")
                    events = []

                all_results[narration_id] = {
                    'food_name': food_name,
                    'video_range': videos,
                    'source_action': item.get('source_action'),
                    'ingredient_matches': item.get('ingredient_matches', []),
                    'events': events
                }

            except Exception as e:
                print(f"ERROR: {e}")
                all_results[narration_id] = {
                    'food_name': food_name,
                    'video_range': videos,
                    'events': [],
                    'error': str(e)
                }

    # Save results
    print(f"\n{'='*70}")
    print(f"SAVING RESULTS")
    print(f"{'='*70}")

    output_file = args.output_dir / participant / f"{participant}_lifecycle.json"
    output_data = {
        'participant': participant,
        'total_items': len(items),
        'items_with_events': sum(1 for r in all_results.values() if r.get('events')),
        'total_events': sum(len(r.get('events', [])) for r in all_results.values()),
        'items': all_results
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)

    print(f"  Output: {output_file}")
    print(f"  Items processed: {len(items)}")
    print(f"  Items with events: {output_data['items_with_events']}")
    print(f"  Total events: {output_data['total_events']}")

    # Summary table
    print(f"\n{'='*70}")
    print(f"LIFECYCLE SUMMARY: {participant}")
    print(f"{'='*70}")
    print(f"{'Narration ID':<28} {'Food Name':<25} {'Events':<8}")
    print("-" * 65)

    for narration_id, result in all_results.items():
        narr = narration_id[-27:]
        food = (result.get('food_name') or '')[:24]
        event_count = len(result.get('events', []))
        print(f"{narr:<28} {food:<25} {event_count:<8}")

    print("-" * 65)
    print(f"Total: {len(items)} items, {output_data['total_events']} events")

    print(f"\n{'='*70}")
    print(f"COMPLETE")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()

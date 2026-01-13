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
import csv
from pathlib import Path
from collections import defaultdict

from inventory_utils import (
    GPTClient,
    DEFAULT_OUTPUT_DIR,
    load_raw_narrations_for_videos,
    load_pickle_data,
    run_lifecycle_for_item,
)
import json

# Path to recipe timestamps CSVs
RECIPE_TIMESTAMPS_DIR = Path(__file__).parent.parent / "data" / "hd-epic-annotations" / "high-level" / "activities"


def load_recipe_timestamps(participant: str) -> list:
    """Load recipe_timestamps.csv for a participant."""
    csv_path = RECIPE_TIMESTAMPS_DIR / f"{participant}_recipe_timestamps.csv"
    if not csv_path.exists():
        print(f"  WARNING: Recipe timestamps not found: {csv_path}")
        return []

    timestamps = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            timestamps.append(row)
    return timestamps


def get_narration_timestamp(narrations_df, narration_id: str) -> float:
    """Get start_timestamp for a narration_id."""
    row = narrations_df[narrations_df['unique_narration_id'] == narration_id]
    if row.empty:
        return None
    return float(row.iloc[0]['start_timestamp'])


def get_active_recipe(recipe_timestamps: list, video_id: str, timestamp: float, participant: str) -> str:
    """Find active recipe at a given timestamp using recipe_timestamps.csv."""
    for row in recipe_timestamps:
        if row['video_id'] != video_id or not row['recipe_id']:
            continue
        start = float(row['start_time'])
        end = float('inf') if row['end_time'] == 'end' else float(row['end_time'])
        if start <= timestamp <= end:
            # Convert R06 -> P03_R06
            return f"{participant}_{row['recipe_id']}"
    return None


def assign_ingredient_to_dispensal(
    event: dict,
    ingredient_matches: list,
    recipe_timestamps: list,
    narrations_df,
    participant: str
) -> dict:
    """
    Assign a recipe ingredient to a DISPENSING event.

    Uses recipe_timestamps.csv to find which recipe is active at the dispensal time,
    then finds the matching ingredient from ingredient_matches.

    Returns the event with added fields:
        - assigned_recipe_id: e.g., "P03_R06"
        - assigned_ingredient_id: e.g., "P03_R06_I02"
        - assigned_amount: e.g., 28
        - assigned_amount_unit: e.g., "ml"
    """
    if event.get('stage') != 'DISPENSING':
        return event

    narration_id = event.get('narration_id')
    if not narration_id:
        return event

    # Extract video_id from narration_id (e.g., "P03-20240216-185832-1228" -> "P03-20240216-185832")
    parts = narration_id.rsplit('-', 1)
    if len(parts) != 2:
        return event
    video_id = parts[0]

    # Get narration timestamp
    timestamp = get_narration_timestamp(narrations_df, narration_id)
    if timestamp is None:
        event['assignment_error'] = 'narration_not_found'
        return event

    event['timestamp'] = timestamp

    # Find active recipe at this timestamp
    active_recipe = get_active_recipe(recipe_timestamps, video_id, timestamp, participant)
    if not active_recipe:
        event['assignment_error'] = 'no_active_recipe'
        return event

    event['assigned_recipe_id'] = active_recipe

    # Find matching ingredient from ingredient_matches
    for match in ingredient_matches:
        # Handle both "P03_R06" and "P03_R06_C0" formats
        match_recipe = match.get('recipe_id', '')
        if match_recipe.startswith(active_recipe):
            event['assigned_ingredient_id'] = match.get('matched_ingredient_id') or match.get('ingredient_id')
            event['assigned_ingredient_name'] = match.get('matched_ingredient')
            event['assigned_amount'] = match.get('amount')
            event['assigned_amount_unit'] = match.get('amount_unit')
            return event

    # No matching ingredient found in ingredient_matches for active recipe
    event['assignment_error'] = f'no_ingredient_match_for_{active_recipe}'
    return event


def update_existing_lifecycle(output_dir: Path, participant: str):
    """
    Update existing lifecycle.json with ingredient assignments without re-running GPT.

    This is useful when you want to add ingredient assignments to existing lifecycle
    data without re-running the expensive GPT calls.
    """
    lifecycle_file = output_dir / participant / f"{participant}_lifecycle.json"
    if not lifecycle_file.exists():
        print(f"ERROR: Lifecycle file not found: {lifecycle_file}")
        return

    print(f"{'='*70}")
    print(f"UPDATE MODE: Assigning ingredients to existing lifecycle events")
    print(f"{'='*70}")

    # Load existing lifecycle data
    print(f"\nLoading {lifecycle_file.name}...")
    with open(lifecycle_file, 'r') as f:
        lifecycle_data = json.load(f)

    print(f"  Items: {lifecycle_data.get('total_items')}")
    print(f"  Events: {lifecycle_data.get('total_events')}")

    # Load recipe timestamps
    print(f"\nLoading recipe timestamps for {participant}...")
    recipe_timestamps = load_recipe_timestamps(participant)
    print(f"  Loaded {len(recipe_timestamps)} activity segments")

    # Load narrations dataframe
    print("Loading narrations data...")
    try:
        narrations_df = load_pickle_data()
        print(f"  Loaded {len(narrations_df)} narrations")
    except Exception as e:
        print(f"ERROR: Could not load narrations data: {e}")
        return

    # Process each item
    total_dispensals = 0
    assigned_dispensals = 0
    errors_by_type = {}

    for item_id, item_data in lifecycle_data['items'].items():
        ingredient_matches = item_data.get('ingredient_matches', [])
        events = item_data.get('events', [])

        for evt in events:
            if evt.get('stage') == 'DISPENSING':
                total_dispensals += 1

                # Clear any previous assignment
                for key in ['assigned_recipe_id', 'assigned_ingredient_id',
                           'assigned_ingredient_name', 'assigned_amount',
                           'assigned_amount_unit', 'assignment_error', 'timestamp']:
                    evt.pop(key, None)

                # Assign ingredient
                assign_ingredient_to_dispensal(
                    evt, ingredient_matches, recipe_timestamps,
                    narrations_df, participant
                )

                if evt.get('assigned_recipe_id'):
                    assigned_dispensals += 1
                elif evt.get('assignment_error'):
                    err = evt['assignment_error']
                    errors_by_type[err] = errors_by_type.get(err, 0) + 1

    # Update stats in output
    lifecycle_data['total_dispensals'] = total_dispensals
    lifecycle_data['assigned_dispensals'] = assigned_dispensals

    # Save updated file
    with open(lifecycle_file, 'w', encoding='utf-8') as f:
        json.dump(lifecycle_data, f, indent=2)

    print(f"\n{'='*70}")
    print(f"RESULTS")
    print(f"{'='*70}")
    print(f"  Total dispensals: {total_dispensals}")
    print(f"  Assigned to recipe ingredients: {assigned_dispensals}")
    print(f"  Unassigned: {total_dispensals - assigned_dispensals}")

    if errors_by_type:
        print(f"\n  Unassigned breakdown:")
        for err, count in sorted(errors_by_type.items(), key=lambda x: -x[1]):
            print(f"    {err}: {count}")

    print(f"\n  Updated: {lifecycle_file}")


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
    parser.add_argument(
        '--update-only',
        action='store_true',
        help='Only update existing lifecycle.json with ingredient assignments (no GPT calls)'
    )

    args = parser.parse_args()
    participant = args.participant

    # Handle --update-only mode
    if args.update_only:
        update_existing_lifecycle(args.output_dir, participant)
        return

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

    # Load recipe timestamps for ingredient assignment
    print(f"\nLoading recipe timestamps for {participant}...")
    recipe_timestamps = load_recipe_timestamps(participant)
    print(f"  Loaded {len(recipe_timestamps)} activity segments")

    # Load narrations dataframe for timestamp lookups
    print("Loading narrations data...")
    try:
        narrations_df = load_pickle_data()
        print(f"  Loaded {len(narrations_df)} narrations")
    except Exception as e:
        print(f"  WARNING: Could not load narrations data: {e}")
        narrations_df = None

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

                    # Assign recipe ingredients to DISPENSING events
                    ingredient_matches = item.get('ingredient_matches', [])
                    if narrations_df is not None and recipe_timestamps:
                        dispensing_count = 0
                        assigned_count = 0
                        for evt in events:
                            if evt.get('stage') == 'DISPENSING':
                                dispensing_count += 1
                                assign_ingredient_to_dispensal(
                                    evt, ingredient_matches, recipe_timestamps,
                                    narrations_df, participant
                                )
                                if evt.get('assigned_recipe_id'):
                                    assigned_count += 1
                        if dispensing_count > 0:
                            print(f"       Assigned ingredients: {assigned_count}/{dispensing_count} dispensals")

                    if args.verbose:
                        for evt in events:
                            stage = evt.get('stage')
                            narr = evt.get('narration_id')
                            if stage == 'DISPENSING' and evt.get('assigned_recipe_id'):
                                print(f"         - {stage}: {narr} -> {evt.get('assigned_recipe_id')} ({evt.get('assigned_amount')} {evt.get('assigned_amount_unit')})")
                            else:
                                print(f"         - {stage}: {narr}")
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

    # Compute stats
    total_events = 0
    total_dispensals = 0
    assigned_dispensals = 0
    for r in all_results.values():
        for evt in r.get('events', []):
            total_events += 1
            if evt.get('stage') == 'DISPENSING':
                total_dispensals += 1
                if evt.get('assigned_recipe_id'):
                    assigned_dispensals += 1

    output_file = args.output_dir / participant / f"{participant}_lifecycle.json"
    output_data = {
        'participant': participant,
        'total_items': len(items),
        'items_with_events': sum(1 for r in all_results.values() if r.get('events')),
        'total_events': total_events,
        'total_dispensals': total_dispensals,
        'assigned_dispensals': assigned_dispensals,
        'items': all_results
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)

    print(f"  Output: {output_file}")
    print(f"  Items processed: {len(items)}")
    print(f"  Items with events: {output_data['items_with_events']}")
    print(f"  Total events: {total_events}")
    print(f"  Dispensals: {total_dispensals} total, {assigned_dispensals} assigned to recipe ingredients")

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

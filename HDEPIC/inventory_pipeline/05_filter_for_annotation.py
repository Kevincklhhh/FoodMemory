#!/usr/bin/env python3
"""
Step 5: Filter Items with Known/Easy Quantities

Extracts items where quantity can be determined:
- Items with difficulty = LOW (countable discrete units)
- Items with matched ingredient that has "amount" field (from recipe)

These items have known or easily measurable quantities.

Prerequisites:
    Run 02c_lifecycle_tracking.py and 04_dispensal_classification.py first

Usage:
    python 05_filter_for_annotation.py --participant P01

Inputs:
    {participant}_lifecycle.json
    {participant}_discovery_edit.json
    {participant}_dispensal_classified.json

Outputs:
    {participant}_known_quantities.json
    {participant}_known_quantities.txt (human-readable)

Output contains for each item:
    - food_name
    - video_range
    - difficulty
    - matched_ingredient_weight (if any)
    - event_narration_ids (all lifecycle events)
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict

from inventory_utils import DEFAULT_OUTPUT_DIR


def main():
    parser = argparse.ArgumentParser(
        description="Step 5: Extract items with known/easy quantities"
    )
    parser.add_argument(
        '--participant',
        required=True,
        help='Participant ID to process (e.g., P01)'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help='Output directory for results'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Print verbose output'
    )

    args = parser.parse_args()
    participant = args.participant
    participant_dir = args.output_dir / participant

    # Load required files
    lifecycle_file = participant_dir / f"{participant}_lifecycle.json"
    discovery_file = participant_dir / f"{participant}_discovery_edit.json"
    classified_file = participant_dir / f"{participant}_dispensal_classified.json"

    # Check files exist
    missing = []
    for f in [lifecycle_file, discovery_file, classified_file]:
        if not f.exists():
            missing.append(f.name)

    if missing:
        print(f"ERROR: Missing files: {', '.join(missing)}")
        print(f"       Run the prerequisite steps first.")
        return

    print(f"Loading input files...")

    with open(lifecycle_file, 'r') as f:
        lifecycle_data = json.load(f)
    print(f"  {lifecycle_file.name}: {len(lifecycle_data.get('items', {}))} items")

    with open(discovery_file, 'r') as f:
        discovery_data = json.load(f)
    print(f"  {discovery_file.name}: {len(discovery_data.get('items', []))} items")

    with open(classified_file, 'r') as f:
        classified_data = json.load(f)
    print(f"  {classified_file.name}: {len(classified_data.get('items', {}))} items")

    # Build lookup: narration_id -> discovery item
    discovery_by_id = {
        item['narration_id']: item
        for item in discovery_data.get('items', [])
    }

    # Build lookup: narration_id -> difficulty
    difficulty_by_id = {
        narr_id: item.get('difficulty', 'UNKNOWN')
        for narr_id, item in classified_data.get('items', {}).items()
    }

    # Process items from lifecycle
    lifecycle_items = lifecycle_data.get('items', {})

    selected_items = []
    filtered_out = 0
    selected_low = 0
    selected_has_amount = 0
    no_dispensing = 0

    print(f"\n{'='*70}")
    print(f"EXTRACTING ITEMS WITH KNOWN QUANTITIES: {participant}")
    print(f"{'='*70}")

    for narr_id, lifecycle_item in lifecycle_items.items():
        food_name = lifecycle_item.get('food_name', 'unknown')
        events = lifecycle_item.get('events', [])
        video_range = lifecycle_item.get('video_range', [])

        # Check if has DISPENSING events
        has_dispensing = any(e.get('stage') == 'DISPENSING' for e in events)
        if not has_dispensing:
            no_dispensing += 1
            if args.verbose:
                print(f"  SKIP (no dispensing): {food_name}")
            continue

        # Get difficulty
        difficulty = difficulty_by_id.get(narr_id, 'UNKNOWN')

        # Get discovery item for ingredient matches
        discovery_item = discovery_by_id.get(narr_id, {})
        ingredient_matches = discovery_item.get('ingredient_matches', [])

        # Check if has matched ingredient with amount
        matched_weight = None
        has_amount = False
        for match in ingredient_matches:
            if match.get('matched_ingredient') and match.get('amount'):
                has_amount = True
                matched_weight = {
                    'ingredient': match.get('matched_ingredient'),
                    'amount': match.get('amount'),
                    'unit': match.get('amount_unit'),
                    'recipe': match.get('recipe_id')
                }
                break

        # Selection criteria: LOW difficulty OR has amount
        is_low = (difficulty == 'LOW')

        if not is_low and not has_amount:
            filtered_out += 1
            if args.verbose:
                print(f"  FILTER OUT: {food_name} (difficulty={difficulty}, no amount)")
            continue

        # Track selection reason
        if is_low:
            selected_low += 1
        if has_amount:
            selected_has_amount += 1

        # Collect all event narration IDs
        event_narration_ids = [e.get('narration_id') for e in events if e.get('narration_id')]

        # This item has known/easy quantity
        item_data = {
            'narration_id': narr_id,
            'food_name': food_name,
            'video_range': video_range,
            'difficulty': difficulty,
            'matched_ingredient_weight': matched_weight,
            'selection_reason': [],
            'num_events': len(events),
            'num_dispensing': sum(1 for e in events if e.get('stage') == 'DISPENSING'),
            'event_narration_ids': event_narration_ids,
            'events': events
        }

        if is_low:
            item_data['selection_reason'].append('LOW_DIFFICULTY')
        if has_amount:
            item_data['selection_reason'].append('HAS_AMOUNT')

        selected_items.append(item_data)

    # Summary
    print(f"\n  Total items in lifecycle: {len(lifecycle_items)}")
    print(f"  No dispensing (skipped): {no_dispensing}")
    print(f"  Filtered out (MID/HIGH, no amount): {filtered_out}")
    print(f"  Selected items: {len(selected_items)}")
    print(f"    - LOW difficulty: {selected_low}")
    print(f"    - Has amount: {selected_has_amount}")

    # Count by difficulty
    difficulty_counts = defaultdict(int)
    for item in selected_items:
        difficulty_counts[item['difficulty']] += 1

    print(f"\n  By difficulty: LOW={difficulty_counts.get('LOW', 0)}, MID={difficulty_counts.get('MID', 0)}, HIGH={difficulty_counts.get('HIGH', 0)}")

    # Save results
    print(f"\n{'='*70}")
    print(f"SAVING RESULTS")
    print(f"{'='*70}")

    output_json = participant_dir / f"{participant}_known_quantities.json"
    output_data = {
        'participant': participant,
        'total_lifecycle_items': len(lifecycle_items),
        'no_dispensing': no_dispensing,
        'filtered_out': filtered_out,
        'selected_count': len(selected_items),
        'selected_low_difficulty': selected_low,
        'selected_has_amount': selected_has_amount,
        'difficulty_breakdown': dict(difficulty_counts),
        'items': selected_items
    }

    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)

    print(f"  JSON: {output_json.name}")

    # Write human-readable text output
    output_txt = participant_dir / f"{participant}_known_quantities.txt"
    with open(output_txt, 'w', encoding='utf-8') as f:
        f.write(f"# Known Quantities: {participant}\n")
        f.write(f"# Total selected: {len(selected_items)}\n")
        f.write(f"# Selection: LOW difficulty ({selected_low}) OR has amount ({selected_has_amount})\n")
        f.write(f"#\n")
        f.write(f"# Filtered out: {filtered_out} items (MID/HIGH difficulty without known amount)\n")
        f.write(f"\n")

        # Group by selection reason
        f.write(f"{'='*70}\n")
        f.write(f"LOW DIFFICULTY ITEMS (countable/discrete)\n")
        f.write(f"{'='*70}\n\n")

        low_items = [i for i in selected_items if 'LOW_DIFFICULTY' in i['selection_reason']]
        for item in low_items:
            f.write(f"{item['food_name']}\n")
            f.write(f"  narration_id: {item['narration_id']}\n")
            f.write(f"  video_range: {', '.join(item['video_range'])}\n")
            f.write(f"  events: {item['num_events']} total, {item['num_dispensing']} dispensing\n")
            if item['matched_ingredient_weight']:
                w = item['matched_ingredient_weight']
                f.write(f"  amount: {w['amount']} {w['unit'] or ''} ({w['ingredient']}) from {w['recipe']}\n")
            f.write(f"  event_ids:\n")
            for evt in item['events']:
                stage = evt.get('stage', 'UNKNOWN')
                evt_id = evt.get('narration_id', 'N/A')
                action = evt.get('action', '')[:50]
                f.write(f"    [{stage:10}] {evt_id}: {action}...\n")
            f.write(f"\n")

        f.write(f"{'='*70}\n")
        f.write(f"ITEMS WITH KNOWN AMOUNT (from recipe)\n")
        f.write(f"{'='*70}\n\n")

        # Items with amount but NOT low difficulty
        amount_items = [i for i in selected_items if 'HAS_AMOUNT' in i['selection_reason'] and 'LOW_DIFFICULTY' not in i['selection_reason']]
        for item in amount_items:
            w = item['matched_ingredient_weight']
            f.write(f"{item['food_name']}\n")
            f.write(f"  narration_id: {item['narration_id']}\n")
            f.write(f"  video_range: {', '.join(item['video_range'])}\n")
            f.write(f"  difficulty: {item['difficulty']}\n")
            f.write(f"  amount: {w['amount']} {w['unit'] or ''} ({w['ingredient']}) from {w['recipe']}\n")
            f.write(f"  events: {item['num_events']} total, {item['num_dispensing']} dispensing\n")
            f.write(f"  event_ids:\n")
            for evt in item['events']:
                stage = evt.get('stage', 'UNKNOWN')
                evt_id = evt.get('narration_id', 'N/A')
                action = evt.get('action', '')[:50]
                f.write(f"    [{stage:10}] {evt_id}: {action}...\n")
            f.write(f"\n")

    print(f"  TXT: {output_txt.name}")

    # Print summary table
    print(f"\n{'='*70}")
    print(f"KNOWN QUANTITIES: {participant}")
    print(f"{'='*70}")
    print(f"{'Food Name':<30} {'Difficulty':<6} {'Amount':<20} {'Events':<6}")
    print("-" * 65)

    for item in selected_items:
        food = (item.get('food_name') or '')[:29]
        diff = item.get('difficulty', '?')[:5]
        w = item.get('matched_ingredient_weight')
        amount_str = f"{w['amount']} {w['unit'] or ''}" if w else "-"
        amount_str = amount_str[:19]
        num_events = item.get('num_events', 0)
        print(f"{food:<30} {diff:<6} {amount_str:<20} {num_events:<6}")

    print("-" * 65)
    print(f"Total: {len(selected_items)} items with known/easy quantities")

    print(f"\n{'='*70}")
    print(f"COMPLETE")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Step 1b: Intersection Deduplication

Finds items in overlapping video ranges and uses GPT to identify same physical
foods that were discovered with different names/narration_ids. Merges them into
canonical items with extended video_ranges.

Prerequisites:
    Run 02b_inventory_discovery.py first to generate _discovery_edit.json

Usage:
    python 02b2_intersection_dedup.py --participant P01

Inputs:
    {participant}/{participant}_discovery_edit.json

Outputs:
    {participant}/{participant}_discovery_deduped.json  : Ready for 02c
    {participant}/{participant}_dedup_report.json       : Deduplication details

Next Step:
    python 02c_lifecycle_tracking.py --participant P01 --discovery-file P01_discovery_deduped.json
"""

import argparse
import json
import re
from pathlib import Path
from collections import defaultdict

from inventory_utils import GPTClient, DEFAULT_OUTPUT_DIR


def find_intersection_groups(items_list):
    """Find groups of items with partially overlapping video_ranges."""

    # Group items by video_range
    range_to_items = defaultdict(list)
    for item in items_list:
        vr = tuple(sorted(item.get('video_range', [])))
        range_to_items[vr].append(item)

    unique_ranges = list(range_to_items.keys())

    # Find pairs of ranges that partially overlap
    intersecting_pairs = []
    for i, r1 in enumerate(unique_ranges):
        s1 = set(r1)
        for j, r2 in enumerate(unique_ranges):
            if i >= j:
                continue
            s2 = set(r2)
            if s1 & s2 and s1 != s2:  # Share videos but not identical
                intersecting_pairs.append((r1, r2))

    if not intersecting_pairs:
        return []

    # Union-Find for connected components
    parent = {}

    def find(x):
        if parent.setdefault(x, x) != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        parent[find(x)] = find(y)

    # Get all ranges involved in intersections
    intersecting_ranges = set()
    for r1, r2 in intersecting_pairs:
        intersecting_ranges.add(r1)
        intersecting_ranges.add(r2)
        union(r1, r2)

    # Group ranges by component
    components = defaultdict(list)
    for r in intersecting_ranges:
        components[find(r)].append(r)

    # Build groups with items
    groups = []
    for group_idx, (comp_id, ranges) in enumerate(sorted(components.items(), key=lambda x: min(x[1])), 1):
        # Compute union of all ranges in this component
        union_range = set()
        for r in ranges:
            union_range.update(r)

        # Collect all items in this component
        group_items = []
        for r in ranges:
            group_items.extend(range_to_items[r])
        group_items.sort(key=lambda x: x['narration_id'])

        groups.append({
            'group_id': group_idx,
            'num_items': len(group_items),
            'num_ranges': len(ranges),
            'union_range': sorted(union_range),
            'items': group_items
        })

    return groups


def call_gpt_dedup(api, items):
    """Call GPT to identify same physical foods in a group."""

    item_list = '\n'.join([f"- {item['narration_id']}: {item['food_name']}" for item in items])

    prompt = f'''You are analyzing food items discovered across multiple cooking videos.
Some items may refer to the SAME physical food item but were discovered at different times with different names.

Your task: Group items that are the SAME physical food item.

Rules:
- Only group items that are clearly the SAME physical item (same container/package)
- Different instances of the same food TYPE are NOT the same item (e.g., two different onions are separate)
- Consider: same food, similar description, could be the same physical item used across cooking sessions
- Be conservative: only group if highly confident they are the same physical item

Items:
{item_list}

Output as JSON:
{{
  "same_food_groups": [
    {{
      "canonical_name": "best name for this item",
      "canonical_id": "narration_id of first/primary occurrence",
      "all_ids": ["narration_id1", "narration_id2", ...],
      "reason": "brief explanation why these are the same physical item"
    }}
  ],
  "unique_items": ["narration_id1", ...]
}}

Only include groups where multiple items refer to the SAME physical item.
List all remaining items (not grouped) in unique_items.
'''

    messages = [{'role': 'user', 'content': prompt}]
    response = api.chat_completion(messages)
    content = response.choices[0].message.content

    # Parse JSON from response
    json_match = re.search(r'\{[\s\S]*\}', content)
    if json_match:
        return json.loads(json_match.group())
    return None


def apply_deduplication(discovery_data, dedup_results):
    """Apply deduplication results to discovery data."""

    # Build narration_id -> item mapping
    id_to_item = {item['narration_id']: item for item in discovery_data['items']}

    merge_count = 0

    for group_result in dedup_results:
        for same_food in group_result.get('same_food_groups', []):
            canonical_id = same_food['canonical_id']
            all_ids = same_food['all_ids']

            canonical_item = id_to_item.get(canonical_id)
            if not canonical_item:
                continue

            # Collect video_ranges and ingredient_matches from all items
            merged_range = set(canonical_item.get('video_range', []))
            merged_matches = list(canonical_item.get('ingredient_matches', []))

            for other_id in all_ids:
                if other_id == canonical_id:
                    continue
                other_item = id_to_item.get(other_id)
                if other_item:
                    # Merge video_range
                    merged_range.update(other_item.get('video_range', []))
                    # Merge ingredient_matches (avoid duplicates)
                    for match in other_item.get('ingredient_matches', []):
                        if match not in merged_matches:
                            merged_matches.append(match)
                    # Mark as alias
                    other_item['canonical_id'] = canonical_id
                    other_item['include'] = False
                    merge_count += 1

            # Update canonical item
            canonical_item['video_range'] = sorted(merged_range)
            canonical_item['ingredient_matches'] = merged_matches
            canonical_item['aliases'] = [id for id in all_ids if id != canonical_id]

    return merge_count


def main():
    parser = argparse.ArgumentParser(
        description="Step 1b: Intersection Deduplication - Merge same foods across overlapping video ranges"
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
        '--model',
        default='gpt-5.2',
        choices=['gpt-4o', 'o4-mini', 'gpt-4.1-mini', 'gpt-5', 'gpt-5.2'],
        help='Model to use (default: gpt-5.2)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Find intersection groups but do not call GPT'
    )

    args = parser.parse_args()
    participant = args.participant

    # Load discovery file
    discovery_file = args.output_dir / participant / f"{participant}_discovery_edit.json"
    if not discovery_file.exists():
        print(f"ERROR: Discovery file not found: {discovery_file}")
        print(f"       Run 02b_inventory_discovery.py --participant {participant} first")
        return

    print(f"Loading {discovery_file.name}...")
    with open(discovery_file, 'r') as f:
        discovery_data = json.load(f)

    items_list = discovery_data.get('items', [])
    print(f"  Total items: {len(items_list)}")

    # Find intersection groups
    print(f"\n{'='*70}")
    print(f"FINDING INTERSECTION GROUPS")
    print(f"{'='*70}")

    groups = find_intersection_groups(items_list)

    if not groups:
        print("\n  No intersection groups found (all ranges are disjoint or identical)")
        print("  No deduplication needed.")
        return

    total_items_in_groups = sum(g['num_items'] for g in groups)
    print(f"\n  Found {len(groups)} intersection groups with {total_items_in_groups} items")

    for group in groups:
        print(f"\n  GROUP {group['group_id']}: {group['num_items']} items, {group['num_ranges']} ranges")
        print(f"    Union: {len(group['union_range'])} videos")

    if args.dry_run:
        print("\n  [DRY RUN] Skipping GPT deduplication")
        return

    # Initialize API
    print(f"\n{'='*70}")
    print(f"GPT DEDUPLICATION")
    print(f"{'='*70}")

    print(f"\nInitializing {args.model} API...")
    api = GPTClient(args.model)
    print("  API initialized successfully")

    # Process each group
    dedup_results = []
    total_merged = 0

    for group in groups:
        print(f"\n--- GROUP {group['group_id']}: {group['num_items']} items ---")
        print(f"  Calling GPT to identify same foods...")

        result = call_gpt_dedup(api, group['items'])

        if result:
            same_food_groups = result.get('same_food_groups', [])
            merged_count = sum(len(g['all_ids']) - 1 for g in same_food_groups)
            total_merged += merged_count

            print(f"  Found {len(same_food_groups)} same-food groups ({merged_count} items to merge)")

            for sf in same_food_groups:
                print(f"    - {sf['canonical_name']}: {len(sf['all_ids'])} items")

            dedup_results.append({
                'group_id': group['group_id'],
                'same_food_groups': same_food_groups,
                'unique_items': result.get('unique_items', [])
            })
        else:
            print(f"  ERROR: Failed to get GPT response")
            dedup_results.append({
                'group_id': group['group_id'],
                'same_food_groups': [],
                'unique_items': [item['narration_id'] for item in group['items']]
            })

    # Apply deduplication
    print(f"\n{'='*70}")
    print(f"APPLYING DEDUPLICATION")
    print(f"{'='*70}")

    merge_count = apply_deduplication(discovery_data, dedup_results)

    # Count results
    included = [i for i in discovery_data['items'] if i.get('include', True)]
    excluded = [i for i in discovery_data['items'] if not i.get('include', True)]

    print(f"\n  Merged {merge_count} alias items into canonical items")
    print(f"  Items included: {len(included)}")
    print(f"  Items excluded (aliases): {len(excluded)}")

    # Save outputs
    print(f"\n{'='*70}")
    print(f"SAVING RESULTS")
    print(f"{'='*70}")

    output_dir = args.output_dir / participant

    # Save deduped discovery file
    deduped_file = output_dir / f"{participant}_discovery_deduped.json"
    with open(deduped_file, 'w', encoding='utf-8') as f:
        json.dump(discovery_data, f, indent=2)
    print(f"\n  Deduped discovery: {deduped_file.name}")

    # Save dedup report
    report_file = output_dir / f"{participant}_dedup_report.json"
    report = {
        'participant': participant,
        'total_items_before': len(items_list),
        'total_items_after': len(included),
        'items_merged': merge_count,
        'intersection_groups': len(groups),
        'dedup_results': dedup_results
    }
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    print(f"  Dedup report: {report_file.name}")

    # Print summary
    print(f"\n{'='*70}")
    print(f"DEDUPLICATION SUMMARY: {participant}")
    print(f"{'='*70}")
    print(f"  Intersection groups: {len(groups)}")
    print(f"  Same-food groups found: {sum(len(r['same_food_groups']) for r in dedup_results)}")
    print(f"  Items before: {len(items_list)}")
    print(f"  Items after: {len(included)} (+{len(excluded)} aliases)")

    print(f"\n  NEXT STEP:")
    print(f"  python 02c_lifecycle_tracking.py --participant {participant} --discovery-file {participant}_discovery_deduped.json")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Step 1: Inventory Discovery

Discovers ALL inventory items (root food entities) for a participant from video
narrations. Outputs a single editable file for user verification.

Usage:
    python 02b_inventory_discovery.py --participant P03

Outputs:
    {participant}/{participant}_discovery_edit.json : Editable inventory list
        - Set "include": false to exclude items from lifecycle tracking
        - Each item shows all recipe ingredient matches for reference

Next Step:
    After reviewing and editing the _discovery_edit.json file, run:
    python 02c_lifecycle_tracking.py --participant P03
"""

import argparse
import re
from pathlib import Path
from collections import defaultdict

from inventory_utils import (
    GPTClient,
    DEFAULT_OUTPUT_DIR,
    load_recipes,
    load_raw_narrations_for_videos,
    run_inventory_discovery,
    run_ingredient_mapping,
    list_recipes,
    build_optimized_discovery_plan,
    filter_inventory_by_videos,
)
import json
import copy


def find_intersection_groups(items_list):
    """Find groups of items with partially overlapping video_ranges.

    Returns groups of items whose video_ranges share some videos but are not identical.
    Uses Union-Find to find connected components of overlapping ranges.
    """
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


def apply_deduplication(items_by_narration, dedup_results):
    """Apply deduplication results to items dictionary.

    Modifies items in-place:
    - Canonical items get extended video_ranges and aliases list
    - Alias items get include=False and canonical_id pointer

    Returns merge count.
    """
    merge_count = 0

    for group_result in dedup_results:
        for same_food in group_result.get('same_food_groups', []):
            canonical_id = same_food['canonical_id']
            all_ids = same_food['all_ids']

            canonical_item = items_by_narration.get(canonical_id)
            if not canonical_item:
                continue

            # Ensure video_range is a set for merging
            if isinstance(canonical_item.get('video_range'), list):
                canonical_item['video_range'] = set(canonical_item['video_range'])

            # Collect video_ranges and ingredient_matches from all items
            merged_matches = list(canonical_item.get('ingredient_matches', []))

            for other_id in all_ids:
                if other_id == canonical_id:
                    continue
                other_item = items_by_narration.get(other_id)
                if other_item:
                    # Merge video_range
                    other_range = other_item.get('video_range', [])
                    if isinstance(other_range, list):
                        other_range = set(other_range)
                    canonical_item['video_range'].update(other_range)

                    # Merge ingredient_matches (avoid duplicates)
                    for match in other_item.get('ingredient_matches', []):
                        if match not in merged_matches:
                            merged_matches.append(match)

                    # Mark as alias
                    other_item['canonical_id'] = canonical_id
                    other_item['include'] = False
                    merge_count += 1

            # Update canonical item
            canonical_item['ingredient_matches'] = merged_matches
            canonical_item['aliases'] = [id for id in all_ids if id != canonical_id]

    return merge_count


def main():
    parser = argparse.ArgumentParser(
        description="Step 1: Inventory Discovery - Discover all food items for a participant"
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
        '--list-recipes',
        action='store_true',
        help='List available recipes and exit'
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

    args = parser.parse_args()
    participant = args.participant

    # Load recipes
    print("Loading recipes...")
    try:
        recipes = load_recipes()
        print(f"  Loaded {len(recipes)} recipes")
    except Exception as e:
        print(f"ERROR: Failed to load recipes: {e}")
        return

    # List recipes if requested
    if args.list_recipes:
        list_recipes(recipes, args.output_dir)
        return

    # Get recipes for this participant
    recipe_ids = [r for r in recipes.keys() if r.startswith(f"{participant}_")]
    if not recipe_ids:
        print(f"ERROR: No recipes found for {participant}")
        return

    # Initialize API
    print(f"\nInitializing {args.model} API" + (" with reasoning..." if args.reasoning else "..."))
    try:
        api = GPTClient(args.model, use_reasoning=args.reasoning)
        print("  API initialized successfully")
    except Exception as e:
        print(f"ERROR: Failed to initialize API: {e}")
        return

    # Build optimized discovery plan
    plan = build_optimized_discovery_plan(recipes, recipe_ids)
    stats = plan['stats']
    discovery_ranges = plan['discovery_ranges']
    range_to_captures = plan['range_to_captures']
    subset_map = plan['subset_map']

    print(f"\n{'='*70}")
    print(f"INVENTORY DISCOVERY: {participant}")
    print(f"{'='*70}")
    print(f"Recipes: {len(recipe_ids)}, Captures: {stats['total_captures']}")
    print(f"Discovery calls: {stats['discovery_calls']} (saved {stats['total_savings']})")

    # =========================================================================
    # PHASE 1: Run discovery for each unique video range
    # =========================================================================
    print(f"\n{'='*70}")
    print(f"PHASE 1: DISCOVERY ({len(discovery_ranges)} calls)")
    print(f"{'='*70}")

    # Cache: video_range -> list of discovered items
    discovery_cache = {}

    for range_idx, video_range in enumerate(sorted(discovery_ranges), 1):
        videos = list(video_range)

        print(f"\n[{range_idx}/{len(discovery_ranges)}] Videos: {', '.join(videos)}")

        # Load narrations
        narrations = load_raw_narrations_for_videos(videos)
        if narrations is None:
            print("  ERROR: No narrations found")
            discovery_cache[video_range] = []
            continue

        line_count = len(narrations.strip().split('\n'))
        print(f"  Narrations: {line_count}")

        # Run inventory discovery
        inventory = run_inventory_discovery(
            api, narrations, args.verbose,
            use_reasoning=args.reasoning,
            reasoning_effort=args.reasoning_effort
        )

        if not inventory:
            print("  WARNING: No items found")
            inventory = []
        else:
            print(f"  Found: {len(inventory)} items")

        discovery_cache[video_range] = inventory

    # =========================================================================
    # PHASE 2: Collect all items and run ingredient mapping per recipe
    # =========================================================================
    print(f"\n{'='*70}")
    print(f"PHASE 2: INGREDIENT MAPPING")
    print(f"{'='*70}")

    # Master dict: narration_id -> item info with all matches
    items_by_narration = {}

    all_ranges = sorted(range_to_captures.keys())
    for video_range in all_ranges:
        captures_list = range_to_captures[video_range]
        videos = list(video_range)

        # Get inventory for this range
        if video_range in discovery_cache:
            base_inventory = discovery_cache[video_range]
        elif video_range in subset_map:
            superset_range = subset_map[video_range]
            superset_inventory = discovery_cache.get(superset_range, [])
            base_inventory = filter_inventory_by_videos(superset_inventory, videos)
            print(f"\n  [SUBSET] {videos[0]}... -> {len(base_inventory)} items from superset")
        else:
            base_inventory = []

        # Process each recipe-capture for ingredient mapping
        for cap in captures_list:
            recipe_id = cap['recipe_id']
            recipe_name = cap['recipe_name']
            suffix = cap['suffix']
            ingredients = cap['ingredients']

            print(f"\n  {recipe_id}{suffix}: {recipe_name}")

            # Deep copy for this recipe's mapping
            inventory = copy.deepcopy(base_inventory)

            # Run ingredient mapping if we have items and ingredients
            mappings = {}
            if inventory and ingredients:
                print(f"    Mapping {len(inventory)} items to {len(ingredients)} ingredients...")
                mapping_result = run_ingredient_mapping(
                    api, inventory, ingredients, recipe_name, args.verbose,
                    use_reasoning=args.reasoning,
                    reasoning_effort=args.reasoning_effort
                )
                if mapping_result:
                    mappings = {m['inventory_item']: m for m in mapping_result}
                    matched = sum(1 for m in mapping_result if m.get('matched_ingredient'))
                    print(f"    Matched: {matched}/{len(inventory)}")

            # Add items to master dict, with this recipe's mapping
            for item in inventory:
                narration_id = item.get('narration_id')
                if not narration_id:
                    continue

                # Initialize item if new
                if narration_id not in items_by_narration:
                    items_by_narration[narration_id] = {
                        'narration_id': narration_id,
                        'food_name': item.get('food_name'),
                        'source_action': item.get('source_action'),
                        'video_range': set(),  # Will collect all relevant videos
                        'ingredient_matches': [],
                        'include': True
                    }

                # Add videos from this recipe's range to the item's video_range
                items_by_narration[narration_id]['video_range'].update(videos)

                # Add this recipe's ingredient match (only if there's a match)
                food_name = item.get('food_name', '')
                if food_name in mappings:
                    m = mappings[food_name]
                    matched_ingredient = m.get('matched_ingredient')
                    if matched_ingredient:  # Only add if there's an actual match
                        match_info = {
                            'recipe_id': recipe_id + suffix,
                            'recipe_name': recipe_name,
                            'matched_ingredient': matched_ingredient,
                            'amount': m.get('amount'),
                            'amount_unit': m.get('amount_unit')
                        }
                        items_by_narration[narration_id]['ingredient_matches'].append(match_info)

    # =========================================================================
    # PHASE 3: Generate output
    # =========================================================================
    print(f"\n{'='*70}")
    print(f"PHASE 3: OUTPUT")
    print(f"{'='*70}")

    # Find intersection videos (videos in multiple discovery calls)
    video_to_discovery_ranges = defaultdict(list)
    for video_range in discovery_ranges:
        for video_id in video_range:
            video_to_discovery_ranges[video_id].append(list(video_range))

    intersection_videos = {
        video_id: ranges
        for video_id, ranges in video_to_discovery_ranges.items()
        if len(ranges) > 1
    }

    if intersection_videos:
        print(f"\n  Intersection videos ({len(intersection_videos)} videos in multiple discovery calls):")
        for video_id, ranges in sorted(intersection_videos.items()):
            print(f"    {video_id}: in {len(ranges)} calls")

    # Convert to sorted list (keep video_range as set for now - dedup will modify)
    items_list = list(sorted(items_by_narration.values(), key=lambda x: x['narration_id']))

    # =========================================================================
    # PHASE 3a: INTERSECTION DEDUPLICATION
    # =========================================================================
    print(f"\n{'='*70}")
    print(f"PHASE 3a: INTERSECTION DEDUPLICATION")
    print(f"{'='*70}")

    # Find intersection groups
    groups = find_intersection_groups([
        {**item, 'video_range': sorted(item['video_range'])}
        for item in items_list
    ])

    dedup_results = []
    total_same_food_groups = 0

    if not groups:
        print("\n  No intersection groups found (all ranges are disjoint or identical)")
        print("  No deduplication needed.")
    else:
        total_items_in_groups = sum(g['num_items'] for g in groups)
        print(f"\n  Found {len(groups)} intersection groups with {total_items_in_groups} items")

        for group in groups:
            print(f"\n  GROUP {group['group_id']}: {group['num_items']} items, {group['num_ranges']} ranges")
            print(f"    Union: {len(group['union_range'])} videos")

        # Process each group with GPT
        for group in groups:
            print(f"\n--- GROUP {group['group_id']}: {group['num_items']} items ---")
            print(f"  Calling GPT to identify same foods...", end=" ", flush=True)

            result = call_gpt_dedup(api, group['items'])

            if result:
                same_food_groups = result.get('same_food_groups', [])
                merged_count = sum(len(g['all_ids']) - 1 for g in same_food_groups)
                total_same_food_groups += len(same_food_groups)

                print(f"found {len(same_food_groups)} same-food groups ({merged_count} items to merge)")

                for sf in same_food_groups:
                    print(f"    - {sf['canonical_name']}: {len(sf['all_ids'])} items")

                dedup_results.append({
                    'group_id': group['group_id'],
                    'same_food_groups': same_food_groups,
                    'unique_items': result.get('unique_items', [])
                })
            else:
                print("ERROR: Failed to get GPT response")
                dedup_results.append({
                    'group_id': group['group_id'],
                    'same_food_groups': [],
                    'unique_items': [item['narration_id'] for item in group['items']]
                })

        # Apply deduplication
        if dedup_results:
            merge_count = apply_deduplication(items_by_narration, dedup_results)
            print(f"\n  Applied deduplication: {merge_count} items merged as aliases")

    # Now convert video_range sets to sorted lists
    for item in items_list:
        if isinstance(item['video_range'], set):
            item['video_range'] = sorted(item['video_range'])

    # Count stats
    included_items = [i for i in items_list if i.get('include', True)]
    excluded_items = [i for i in items_list if not i.get('include', True)]
    total_items = len(items_list)
    items_with_match = sum(
        1 for item in items_list
        if any(m.get('matched_ingredient') for m in item['ingredient_matches'])
    )

    # Prepare output
    output_data = {
        'participant': participant,
        'total_items': total_items,
        'included_items': len(included_items),
        'excluded_items': len(excluded_items),
        'items_with_ingredient_match': items_with_match,
        'discovery_stats': stats,
        'dedup_stats': {
            'intersection_groups': len(groups) if groups else 0,
            'same_food_groups': total_same_food_groups,
            'items_merged_as_aliases': len(excluded_items)
        },
        'recipes_used': recipe_ids,
        'intersection_videos': {
            video_id: {
                'discovery_ranges': ranges,
                'num_calls': len(ranges)
            }
            for video_id, ranges in sorted(intersection_videos.items())
        },
        '_instructions': "Set 'include': false to exclude items from lifecycle tracking. Items with canonical_id are aliases of other items.",
        'items': items_list
    }

    # Save
    output_dir = args.output_dir / participant
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{participant}_discovery_edit.json"

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n  Output: {output_path}")
    print(f"  Total items: {total_items}")
    print(f"  Included: {len(included_items)}, Excluded (aliases): {len(excluded_items)}")
    print(f"  Items with ingredient match: {items_with_match}")

    # Print summary table
    print(f"\n{'='*70}")
    print(f"DISCOVERY SUMMARY: {participant}")
    print(f"{'='*70}")
    print(f"{'Narration ID':<28} {'Food Name':<25} {'Matches':<5} {'Best Match':<20}")
    print("-" * 80)
    for item in items_list:
        narr = item['narration_id'][-27:]
        food = (item.get('food_name') or '')[:24]
        matches = item['ingredient_matches']
        match_count = sum(1 for m in matches if m.get('matched_ingredient'))
        best_match = next((m.get('matched_ingredient') for m in matches if m.get('matched_ingredient')), '-')
        if best_match and len(best_match) > 19:
            best_match = best_match[:19]
        print(f"{narr:<28} {food:<25} {match_count:<5} {best_match:<20}")
    print("-" * 80)
    print(f"Total: {len(included_items)} items (+{len(excluded_items)} aliases)")

    # Final summary
    print(f"\n{'='*70}")
    print(f"COMPLETE")
    print(f"{'='*70}")
    print(f"  Discovery calls: {stats['discovery_calls']}")
    print(f"  Deduplication: {total_same_food_groups} same-food groups found")
    print(f"  Items: {len(included_items)} included, {len(excluded_items)} aliases")

    # =========================================================================
    # PHASE 3b: Generate intersection groups file
    # =========================================================================
    generate_intersection_groups(items_list, output_dir, participant)

    print(f"\n  NEXT STEPS:")
    print(f"  1. (Optional) Review {output_path.name} and set 'include': false for items to exclude")
    print(f"  2. Run: python 02c_lifecycle_tracking.py --participant {participant}")


def generate_intersection_groups(items_list, output_dir, participant):
    """Generate intersection groups file showing items in disjoint connected components."""
    # Group items by video_range
    range_to_items = defaultdict(list)
    for item in items_list:
        vr = tuple(sorted(item.get('video_range', [])))
        range_to_items[vr].append(item)

    unique_ranges = list(range_to_items.keys())

    # Find pairs of ranges that partially overlap (share videos but not identical)
    intersecting_pairs = []
    for i, r1 in enumerate(unique_ranges):
        s1 = set(r1)
        for j, r2 in enumerate(unique_ranges):
            if i >= j:
                continue
            s2 = set(r2)
            if s1 & s2 and s1 != s2:  # Share videos but not identical
                intersecting_pairs.append((r1, r2, s1 & s2))

    if not intersecting_pairs:
        print(f"\n  No intersection groups found (all ranges are disjoint or identical)")
        return

    # Find connected components using Union-Find
    parent = {}

    def find(x):
        if parent.setdefault(x, x) != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        parent[find(x)] = find(y)

    # Get all ranges involved in intersections
    intersecting_ranges = set()
    for r1, r2, _ in intersecting_pairs:
        intersecting_ranges.add(r1)
        intersecting_ranges.add(r2)
        union(r1, r2)

    # Group ranges by component
    components = defaultdict(list)
    for r in intersecting_ranges:
        components[find(r)].append(r)

    # Build output
    output_lines = []
    output_lines.append(f'{participant} Intersection Groups')
    output_lines.append('=' * 70)
    output_lines.append('')
    output_lines.append(f'Items whose video_range partially overlaps with other items.')
    output_lines.append(f'Items in the same group share videos transitively.')
    output_lines.append('')

    total_items_in_groups = 0
    group_data = []

    for group_idx, (comp_id, ranges) in enumerate(sorted(components.items(), key=lambda x: min(x[1])), 1):
        # Compute union of all ranges in this component
        union_range = set()
        for r in ranges:
            union_range.update(r)
        union_sorted = sorted(union_range)

        # Collect all items in this component
        group_items = []
        for r in ranges:
            group_items.extend(range_to_items[r])
        group_items.sort(key=lambda x: x['narration_id'])

        total_items_in_groups += len(group_items)

        # Short video name helper
        def short_vid(v):
            return v.split('-')[-1]  # e.g., '121517' from 'P01-20240203-121517'

        output_lines.append(f'GROUP {group_idx}: {len(group_items)} items, {len(ranges)} ranges')
        output_lines.append('-' * 70)
        output_lines.append(f'Union range ({len(union_sorted)} videos):')
        output_lines.append(f'  {" → ".join(short_vid(v) for v in union_sorted)}')
        output_lines.append('')
        output_lines.append('Items:')
        for item in group_items:
            vr_short = [short_vid(v) for v in item['video_range']]
            output_lines.append(f'  {item["narration_id"]}: {item["food_name"]}')
            output_lines.append(f'      range: [{", ".join(vr_short)}]')
        output_lines.append('')

        # Store for JSON output
        group_data.append({
            'group_id': group_idx,
            'num_items': len(group_items),
            'num_ranges': len(ranges),
            'union_range': union_sorted,
            'original_ranges': [list(r) for r in sorted(ranges)],
            'items': [
                {
                    'narration_id': item['narration_id'],
                    'food_name': item['food_name'],
                    'video_range': item['video_range']
                }
                for item in group_items
            ]
        })

    output_lines.append('=' * 70)
    output_lines.append(f'SUMMARY: {len(components)} groups, {total_items_in_groups} items')
    output_lines.append('')

    # Save text file
    txt_path = output_dir / f"{participant}_intersection_groups.txt"
    with open(txt_path, 'w') as f:
        f.write('\n'.join(output_lines))

    # Save JSON file
    json_path = output_dir / f"{participant}_intersection_groups.json"
    json_data = {
        'participant': participant,
        'total_groups': len(components),
        'total_items_in_groups': total_items_in_groups,
        'groups': group_data
    }
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)

    print(f"\n  Intersection groups: {txt_path.name}")
    print(f"    {len(components)} disjoint groups, {total_items_in_groups} items")


if __name__ == '__main__':
    main()

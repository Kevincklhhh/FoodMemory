#!/usr/bin/env python3
"""
Step 4: Dispensal Difficulty Classification

Classifies each inventory item by difficulty of measuring quantity dispensed.
Reads lifecycle tracking output and rates each item based on ALL its lifecycle events.

Prerequisites:
    Run 02c_lifecycle_tracking.py first to generate _lifecycle.json

Usage:
    python 04_dispensal_classification.py --participant P01
    python 04_dispensal_classification.py --participant P01 --limit 5  # Test mode

Inputs:
    {participant}/{participant}_lifecycle.json

Outputs:
    {participant}/{participant}_dispensal_classified.json

Difficulty Levels:
    LOW:  Discrete/countable - quantity obvious (1 egg, 2 carrots)
    MID:  Geometric portions - estimable from visible shape/size
    HIGH: Continuous/variable - hard to measure without instruments
"""

import argparse
import json
import re
from pathlib import Path

from inventory_utils import GPTClient, DEFAULT_OUTPUT_DIR


ITEM_DIFFICULTY_PROMPT = """You are a Quantity Estimation Difficulty Classifier.

Your task is to evaluate how DIFFICULT it would be for a computer vision system to measure
the QUANTITY of an ingredient being used during cooking, based on all observed lifecycle events.

## DIFFICULTY LEVELS:

### LOW - Discrete/Countable Units
The quantity is inherently countable or comes in distinct units.
- **Characteristics:**
  - Whole items that can be counted (1 egg, 2 carrots, 3 biscuits)
  - Pre-portioned units (1 stock cube, 1 coffee capsule)
  - Items removed intact from packaging
- **Examples:**
  - Taking eggs from carton → LOW (count visible)
  - Using a stock cube → LOW (discrete unit)
  - Picking oranges from bag → LOW (countable)

### MID - Geometric/Estimable Portions
The quantity can be estimated from visible geometric properties.
- **Characteristics:**
  - Cut portions where dimensions are visible (slice thickness, chunk size)
  - Scoops with defined shape (spoonful as reference)
  - Portions with clear visual boundaries
- **Examples:**
  - Cutting butter slices → MID (estimate from slice geometry)
  - Grating cheese → MID (estimate from pile size)
  - Spooning yogurt → MID (spoon is reference)

### HIGH - Continuous/Variable Flow
The quantity is difficult to measure without instruments.
- **Characteristics:**
  - Poured liquids with variable flow rate
  - Shaken/sprinkled seasonings with unpredictable dispersion
  - Squeezed contents with variable pressure
  - Loose materials with variable density
- **Examples:**
  - Pouring milk/oil → HIGH (flow rate varies)
  - Shaking salt/spices → HIGH (unpredictable)
  - Scooping flour → HIGH (loose powder)

## INPUT FORMAT:
You will receive:
1. Item name (the ingredient)
2. All lifecycle events for this item (RETRIEVAL, ACCESS, DISPENSING, RESTOCKING)

## OUTPUT FORMAT:
Return JSON:
{
  "difficulty": "LOW" | "MID" | "HIGH",
  "reasoning": "Brief explanation based on dispensing method observed (1-2 sentences)"
}

## IMPORTANT:
1. Focus on how the item is DISPENSED (the DISPENSING events are most important)
2. Consider ALL events for context (how it's retrieved, opened, used)
3. Rate based on: "Can a computer vision system estimate the quantity used from video?"
"""


def classify_item(api, food_name, events, reasoning_effort="medium"):
    """Classify difficulty for a single item based on all its events."""

    # Format events for GPT
    events_text = []
    for evt in events:
        stage = evt.get('stage', 'UNKNOWN')
        action = evt.get('action', 'N/A')
        narr_id = evt.get('narration_id', '')
        events_text.append(f"  [{stage}] {action}")

    input_text = f"""Item: {food_name}

Lifecycle Events:
{chr(10).join(events_text)}

Based on how this item is used (especially dispensing), classify the difficulty of measuring the quantity."""

    messages = [
        {"role": "system", "content": ITEM_DIFFICULTY_PROMPT},
        {"role": "user", "content": input_text}
    ]

    try:
        response = api.chat_completion(messages)
        response_text = response.choices[0].message.content

        # Parse JSON from response
        json_match = re.search(r'\{[^{}]*\}', response_text, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group(0))
            return result.get("difficulty", "UNKNOWN"), result.get("reasoning", "")

        return "UNKNOWN", f"Failed to parse: {response_text[:100]}"

    except Exception as e:
        return "ERROR", str(e)


def main():
    parser = argparse.ArgumentParser(
        description="Step 4: Classify dispensal difficulty per inventory item"
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
        choices=['gpt-4o', 'o4-mini', 'gpt-4.1-mini', 'gpt-5', 'gpt-5.2', 'o3'],
        help='Model to use (default: gpt-5.2)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit to first N items (for testing)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Print verbose output'
    )

    args = parser.parse_args()
    participant = args.participant

    # Load lifecycle file
    lifecycle_file = args.output_dir / participant / f"{participant}_lifecycle.json"
    if not lifecycle_file.exists():
        print(f"ERROR: Lifecycle file not found: {lifecycle_file}")
        print(f"       Run 02c_lifecycle_tracking.py --participant {participant} first")
        return

    print(f"Loading {lifecycle_file.name}...")
    with open(lifecycle_file, 'r') as f:
        lifecycle_data = json.load(f)

    items = lifecycle_data.get('items', {})
    print(f"  Total items: {len(items)}")

    # Filter items with DISPENSING events
    items_with_dispensing = {}
    for narr_id, item in items.items():
        events = item.get('events', [])
        has_dispensing = any(e.get('stage') == 'DISPENSING' for e in events)
        if has_dispensing:
            items_with_dispensing[narr_id] = item

    print(f"  Items with DISPENSING: {len(items_with_dispensing)}")

    if not items_with_dispensing:
        print("ERROR: No items with DISPENSING events found")
        return

    # Apply limit if specified
    item_list = list(items_with_dispensing.items())
    if args.limit:
        item_list = item_list[:args.limit]
        print(f"  Limited to first {args.limit} items for testing")

    # Initialize API
    print(f"\nInitializing {args.model} API...")
    try:
        api = GPTClient(args.model)
        print("  API initialized successfully")
    except Exception as e:
        print(f"ERROR: Failed to initialize API: {e}")
        return

    print(f"\n{'='*70}")
    print(f"DISPENSAL CLASSIFICATION: {participant}")
    print(f"{'='*70}")

    # Process each item
    results = {}
    difficulty_counts = {"LOW": 0, "MID": 0, "HIGH": 0, "UNKNOWN": 0, "ERROR": 0}

    for idx, (narr_id, item) in enumerate(item_list, 1):
        food_name = item.get('food_name', 'unknown')
        events = item.get('events', [])
        dispensing_count = sum(1 for e in events if e.get('stage') == 'DISPENSING')

        print(f"\n[{idx}/{len(item_list)}] {food_name}")
        print(f"       {len(events)} events ({dispensing_count} dispensing)")
        print(f"       Classifying...", end=" ", flush=True)

        difficulty, reasoning = classify_item(api, food_name, events)

        print(f"{difficulty}")
        if args.verbose and reasoning:
            print(f"       Reasoning: {reasoning}")

        results[narr_id] = {
            'food_name': food_name,
            'num_events': len(events),
            'num_dispensing': dispensing_count,
            'difficulty': difficulty,
            'reasoning': reasoning,
            'events': events
        }

        if difficulty in difficulty_counts:
            difficulty_counts[difficulty] += 1
        else:
            difficulty_counts["UNKNOWN"] += 1

    # Save results
    print(f"\n{'='*70}")
    print(f"SAVING RESULTS")
    print(f"{'='*70}")

    output_file = args.output_dir / participant / f"{participant}_dispensal_classified.json"
    output_data = {
        'participant': participant,
        'total_items_with_dispensing': len(items_with_dispensing),
        'items_classified': len(results),
        'summary': difficulty_counts,
        'items': results
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)

    print(f"  Output: {output_file}")

    # Print summary table
    print(f"\n{'='*70}")
    print(f"CLASSIFICATION SUMMARY: {participant}")
    print(f"{'='*70}")
    print(f"{'Food Name':<35} {'Events':<8} {'Dispense':<8} {'Difficulty':<10}")
    print("-" * 65)

    for narr_id, result in results.items():
        food = (result.get('food_name') or '')[:34]
        num_events = result.get('num_events', 0)
        num_disp = result.get('num_dispensing', 0)
        diff = result.get('difficulty', 'UNKNOWN')
        print(f"{food:<35} {num_events:<8} {num_disp:<8} {diff:<10}")

    print("-" * 65)
    print(f"Total: {len(results)} items")
    print(f"  LOW: {difficulty_counts['LOW']}, MID: {difficulty_counts['MID']}, HIGH: {difficulty_counts['HIGH']}")

    print(f"\n{'='*70}")
    print(f"COMPLETE")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()

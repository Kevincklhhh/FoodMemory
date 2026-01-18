#!/usr/bin/env python3
"""
Step 6: Timeline Aggregation - Consolidate Dispensing Events

Uses GPT 5.2 with high reasoning to analyze narration lines and determine:
1. Continuous time range for all dispensing actions (start to end)
2. Total count of items if food is countable (e.g., eggs, slices of bread)

Prerequisites:
    Run 05_filter_for_annotation.py first to generate known_quantities.json

Usage:
    # Test on first 3 items
    python 06_timeline_aggregation.py --participant P03 --test 3

    # Process all items
    python 06_timeline_aggregation.py --participant P03

Inputs:
    {participant}_known_quantities.json
    HD_EPIC_Narrations.pkl (for full narration text)

Outputs:
    {participant}_timeline_aggregated.json
"""

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional

from inventory_utils import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PICKLE_PATH,
    GPTClient,
    extract_json_from_response,
)


DISPENSAL_AGGREGATION_PROMPT = """You are a "Video Timeline Aggregator" for cooking events.
Your task is to consolidate fragmented narration lines into time ranges that capture dispensing actions for a specific ingredient.

INPUT:
1. TARGET INGREDIENT: "{target_ingredient}"
2. NARRATION LOG: A chronological list of user actions with timestamps.

YOUR GOAL:
Identify dispensing segments for the Target Ingredient. Each segment has:
- **Start:** The timestamp of the FIRST dispensing-related action in that segment.
- **End:** The timestamp of the LAST dispensing-related action in that segment.
- **Count:** If the item is countable, the number of units dispensed in that segment.

GUIDELINES:
- **Merge Fragmented Actions:** The log often breaks a single action into multiple lines (e.g., "Pick egg 1", "Pick egg 2", "Pick egg 3"). Treat these as ONE continuous segment.
- **Split into Multiple Segments:** If dispensing actions are separated by MORE THAN 30 SECONDS, output them as SEPARATE segments. For example:
  - Actions at 100s, 105s, 110s -> ONE segment [100s - 110s]
  - Actions at 100s, 105s, then 200s -> TWO segments [100s - 105s] and [200s - 200s]
- **Time Range:** Each segment's range [start, end] should cover all continuous actions within 30 seconds of each other.
- **Count Logic:** Look for quantifiers in the text:
  - "one egg" -> +1
  - "two more eggs" -> +2
  - "another egg" -> +1
  - "a scoop" -> +1
  - "a slice" -> +1
  - If the item is uncountable (e.g. pouring milk with no unit mentioned), return null for count.

INPUT LOG:
{narration_log}

OUTPUT FORMAT (JSON):
{{
  "food_name": "{target_ingredient}",
  "dispensal_segments": [
    {{
      "start_timestamp": <float>,
      "end_timestamp": <float>,
      "count": <int or null>,
      "count_unit": <string or null>
    }}
  ],
  "total_count": <int or null - sum of all segment counts>,
  "count_unit": <string describing what was counted, e.g., "eggs", "slices", "scoops", or null if uncountable>,
  "reasoning": "Explain how you grouped the actions into segments and calculated counts."
}}
"""


def load_narrations_as_dataframe():
    """Load narrations pickle file and return the DataFrame."""
    if not DEFAULT_PICKLE_PATH.exists():
        raise FileNotFoundError(f"Pickle file not found: {DEFAULT_PICKLE_PATH}")
    with open(DEFAULT_PICKLE_PATH, 'rb') as f:
        return pickle.load(f)


def get_narrations_for_videos(df, video_ids: List[str]) -> Dict[str, Dict]:
    """
    Get narration data for specific videos.

    Returns dict mapping narration_id to {start, end, text}.
    """
    narrations = {}
    for video_id in video_ids:
        video_df = df[df['video_id'] == video_id].copy()
        for _, row in video_df.iterrows():
            narrations[row['unique_narration_id']] = {
                'start': row['start_timestamp'],
                'end': row['end_timestamp'],
                'text': row['narration'].strip(),
                'video_id': video_id
            }
    return narrations


def format_narration_log(events: List[Dict], all_narrations: Dict[str, Dict]) -> str:
    """
    Format the narration log for GPT input.

    Includes DISPENSING events with their full narration text and timestamps.
    """
    lines = []
    for evt in events:
        if evt.get('stage') != 'DISPENSING':
            continue

        narr_id = evt.get('narration_id', '')
        narr_data = all_narrations.get(narr_id, {})

        start = narr_data.get('start') or evt.get('timestamp', 0)
        end = narr_data.get('end', start)
        text = narr_data.get('text') or evt.get('action', '')

        lines.append(f"[{start:.2f}s - {end:.2f}s] {narr_id}: {text}")

    return "\n".join(lines)


def run_timeline_aggregation(
    api: GPTClient,
    food_name: str,
    narration_log: str,
    reasoning_effort: str = "high",
    verbose: bool = False
) -> Optional[Dict]:
    """
    Call GPT to aggregate timeline for a single food item.
    Returns segments (may be multiple if actions are >30s apart).
    """
    prompt = DISPENSAL_AGGREGATION_PROMPT.format(
        target_ingredient=food_name,
        narration_log=narration_log
    )

    try:
        response_text, usage = api.responses_create(
            instructions=prompt,
            input_text=f"Analyze the dispensing events for: {food_name}",
            reasoning_effort=reasoning_effort
        )

        if verbose:
            print(f"    Tokens: {usage.input_tokens} in, {usage.output_tokens} out")
            if hasattr(usage, 'output_tokens_details') and usage.output_tokens_details:
                print(f"    Reasoning tokens: {usage.output_tokens_details.reasoning_tokens}")

        if verbose:
            print(f"    Raw response: {response_text[:500]}...")

        # Parse JSON from response - try object first, then array
        result = None
        import re

        # Try to find JSON in markdown code block (object or array)
        code_block_match = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', response_text)
        if code_block_match:
            try:
                result = json.loads(code_block_match.group(1))
            except json.JSONDecodeError:
                pass

        # If not found, try raw JSON object
        if result is None:
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if json_match:
                try:
                    result = json.loads(json_match.group(0))
                except json.JSONDecodeError:
                    pass

        # Fallback to extract_json_from_response for arrays
        if result is None:
            result = extract_json_from_response(response_text)
            if isinstance(result, list) and len(result) > 0:
                result = result[0]

        # Handle backward compatibility: convert old format to new format
        if result and 'dispensal_segment' in result and 'dispensal_segments' not in result:
            old_seg = result['dispensal_segment']
            result['dispensal_segments'] = [{
                'start_timestamp': old_seg.get('start_timestamp'),
                'end_timestamp': old_seg.get('end_timestamp'),
                'count': result.get('total_count'),
                'count_unit': result.get('count_unit')
            }]

        if verbose and result:
            print(f"    Segments returned: {len(result.get('dispensal_segments', []))}")

        return result

    except Exception as e:
        print(f"    ERROR: API call failed: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Step 6: Aggregate dispensing timelines using GPT 5.2"
    )
    parser.add_argument(
        '--participant',
        required=True,
        help='Participant ID to process (e.g., P01, P03)'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help='Output directory for results'
    )
    parser.add_argument(
        '--test',
        type=int,
        default=0,
        help='Test mode: only process first N items (0 = all)'
    )
    parser.add_argument(
        '--model',
        default='gpt-5.2',
        help='Model to use (default: gpt-5.2)'
    )
    parser.add_argument(
        '--reasoning-effort',
        default='high',
        choices=['low', 'medium', 'high'],
        help='Reasoning effort level (default: high)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Print verbose output'
    )

    args = parser.parse_args()
    participant = args.participant
    participant_dir = args.output_dir / participant

    # Load known quantities
    known_quantities_file = participant_dir / f"{participant}_known_quantities.json"
    if not known_quantities_file.exists():
        print(f"ERROR: {known_quantities_file.name} not found")
        print("       Run 05_filter_for_annotation.py first")
        return

    with open(known_quantities_file, 'r') as f:
        known_data = json.load(f)

    items = known_data.get('items', [])
    print(f"Loaded {len(items)} items from {known_quantities_file.name}")

    # Apply test limit
    if args.test > 0:
        items = items[:args.test]
        print(f"TEST MODE: Processing only first {len(items)} items")

    # Load narrations
    print(f"Loading narrations from pickle...")
    df = load_narrations_as_dataframe()

    # Collect all video IDs needed
    all_video_ids = set()
    for item in items:
        all_video_ids.update(item.get('video_range', []))

    print(f"Loading narrations for {len(all_video_ids)} videos...")
    all_narrations = get_narrations_for_videos(df, list(all_video_ids))
    print(f"Loaded {len(all_narrations)} narration entries")

    # Initialize GPT client
    print(f"\nInitializing {args.model} with reasoning_effort={args.reasoning_effort}...")
    api = GPTClient(model=args.model, use_reasoning=True)

    # Process items
    print(f"\n{'='*70}")
    print(f"PROCESSING ITEMS")
    print(f"{'='*70}")

    results = []
    for i, item in enumerate(items):
        food_name = item.get('food_name', 'unknown')
        narr_id = item.get('narration_id', '')
        events = item.get('events', [])
        difficulty = item.get('difficulty', 'UNKNOWN')

        # Count dispensing events
        dispensing_events = [e for e in events if e.get('stage') == 'DISPENSING']

        print(f"\n[{i+1}/{len(items)}] {food_name}")
        print(f"  Narration ID: {narr_id}")
        print(f"  Difficulty: {difficulty}")
        print(f"  Dispensing events: {len(dispensing_events)}")

        if not dispensing_events:
            print(f"  SKIP: No dispensing events")
            continue

        # Format narration log
        narration_log = format_narration_log(events, all_narrations)

        if args.verbose:
            print(f"\n  --- Narration Log ---")
            for line in narration_log.split('\n'):
                print(f"  {line}")
            print(f"  ---")

        # Call GPT
        print(f"  Calling {args.model}...", end=" ", flush=True)
        result = run_timeline_aggregation(
            api,
            food_name,
            narration_log,
            reasoning_effort=args.reasoning_effort,
            verbose=args.verbose
        )

        if result:
            segments = result.get('dispensal_segments', [])
            total_count = result.get('total_count')
            count_unit = result.get('count_unit')

            # Display segments
            count_str = f"{total_count} {count_unit}" if total_count is not None else "uncountable"
            if len(segments) == 1:
                seg = segments[0]
                print(f"[{seg.get('start_timestamp', 0):.2f}s - {seg.get('end_timestamp', 0):.2f}s] count={count_str}")
            else:
                print(f"{len(segments)} segments, total count={count_str}")
                for j, seg in enumerate(segments):
                    seg_count = seg.get('count')
                    seg_unit = seg.get('count_unit', '')
                    seg_count_str = f"{seg_count} {seg_unit}" if seg_count is not None else "-"
                    print(f"    [{j+1}] {seg.get('start_timestamp', 0):.2f}s - {seg.get('end_timestamp', 0):.2f}s (count={seg_count_str})")

            if args.verbose and result.get('reasoning'):
                print(f"  Reasoning: {result.get('reasoning')[:200]}...")

            # Store result
            results.append({
                'narration_id': narr_id,
                'food_name': food_name,
                'difficulty': difficulty,
                'video_range': item.get('video_range', []),
                'dispensal_segments': segments,
                'total_count': total_count,
                'count_unit': count_unit,
                'reasoning': result.get('reasoning'),
                'num_dispensing_events': len(dispensing_events),
                'num_segments': len(segments),
                'matched_ingredient_weight': item.get('matched_ingredient_weight')
            })
        else:
            print("FAILED")
            results.append({
                'narration_id': narr_id,
                'food_name': food_name,
                'difficulty': difficulty,
                'error': 'API call failed'
            })

    # Save results
    print(f"\n{'='*70}")
    print(f"SAVING RESULTS")
    print(f"{'='*70}")

    output_file = participant_dir / f"{participant}_timeline_aggregated.json"
    output_data = {
        'participant': participant,
        'model': args.model,
        'reasoning_effort': args.reasoning_effort,
        'total_items': len(items),
        'processed_items': len(results),
        'items': results
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)

    print(f"  Saved: {output_file.name}")

    # Summary
    successful = sum(1 for r in results if 'dispensal_segments' in r)
    with_count = sum(1 for r in results if r.get('total_count') is not None)
    multi_segment = sum(1 for r in results if r.get('num_segments', 0) > 1)
    total_segments = sum(r.get('num_segments', 0) for r in results)

    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"  Total items: {len(items)}")
    print(f"  Successfully processed: {successful}")
    print(f"  With count (countable): {with_count}")
    print(f"  Uncountable: {successful - with_count}")
    print(f"  Items with multiple segments: {multi_segment}")
    print(f"  Total segments: {total_segments}")

    # Print results table
    print(f"\n{'Food Name':<30} {'Segments':<10} {'Time Range(s)':<25} {'Count':<15}")
    print("-" * 80)
    for r in results:
        if 'dispensal_segments' not in r:
            continue
        food = (r.get('food_name') or '')[:29]
        segments = r.get('dispensal_segments', [])
        num_segs = len(segments)
        count = r.get('total_count')
        unit = r.get('count_unit', '')
        count_str = f"{count} {unit}" if count is not None else "-"

        if num_segs == 1:
            seg = segments[0]
            time_range = f"{seg.get('start_timestamp', 0):.1f}s - {seg.get('end_timestamp', 0):.1f}s"
            print(f"{food:<30} {num_segs:<10} {time_range:<25} {count_str:<15}")
        else:
            # First row with food name
            seg = segments[0]
            time_range = f"{seg.get('start_timestamp', 0):.1f}s - {seg.get('end_timestamp', 0):.1f}s"
            print(f"{food:<30} {num_segs:<10} {time_range:<25} {count_str:<15}")
            # Additional segments
            for seg in segments[1:]:
                time_range = f"{seg.get('start_timestamp', 0):.1f}s - {seg.get('end_timestamp', 0):.1f}s"
                print(f"{'':<30} {'':<10} {time_range:<25} {'':<15}")


if __name__ == '__main__':
    main()
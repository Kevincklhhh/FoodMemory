#!/usr/bin/env python3
"""
init_failure_cases.py - Initialize a v2 failure_cases file from wrong_predictions

Creates a lightweight reference-based index for failure cases that:
1. References segment_ids from timeline_annotated (no data duplication)
2. Stores only case metadata (include, priority, notes, tags)
3. Merges with timeline/vlm data at runtime in the visualizer

Usage:
    # Initialize from wrong predictions (top N by error)
    python init_failure_cases.py --tag qwen_low --top 20

    # Filter by minimum absolute error
    python init_failure_cases.py --tag qwen_low --min-error 3

    # Filter by participant
    python init_failure_cases.py --tag qwen_low --participant P03

    # Custom output name
    python init_failure_cases.py --tag qwen_low --top 10 --name curated

Outputs:
    outputs/02_inventory/failure_cases_{name}.json (v2 schema)
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from inventory_utils import DEFAULT_OUTPUT_DIR, generate_segment_id


def load_wrong_predictions(output_dir: Path, tag: str) -> dict:
    """Load wrong_predictions_{tag}.json file."""
    wrong_file = output_dir / "prediction_analysis" / f"wrong_predictions_{tag}.json"
    if not wrong_file.exists():
        raise FileNotFoundError(f"Wrong predictions file not found: {wrong_file}")

    with open(wrong_file, 'r') as f:
        return json.load(f)


def load_timeline_segment_ids(output_dir: Path, participants: List[str]) -> Dict[str, Dict]:
    """
    Load timeline_annotated files and build a lookup of segment_ids.

    Returns:
        Dict mapping (participant, narration_id, video_id, start_ts, end_ts) -> segment_id
    """
    segment_id_lookup = {}

    for participant in participants:
        timeline_file = output_dir / participant / f"{participant}_timeline_annotated.json"
        if not timeline_file.exists():
            print(f"  WARNING: {timeline_file.name} not found")
            continue

        with open(timeline_file, 'r') as f:
            timeline_data = json.load(f)

        for item in timeline_data.get('items', []):
            narration_id = item.get('narration_id', '')
            for seg in item.get('dispensal_segments', []):
                video_id = seg.get('video_id', '')
                start_ts = seg.get('start_timestamp', 0)
                end_ts = seg.get('end_timestamp', 0)

                # Get existing segment_id or generate one
                seg_id = seg.get('segment_id')
                if not seg_id:
                    seg_id = generate_segment_id(narration_id, video_id, start_ts, end_ts)

                key = (participant, narration_id, video_id, round(start_ts, 2), round(end_ts, 2))
                segment_id_lookup[key] = seg_id

    return segment_id_lookup


def filter_items(
    items: List[dict],
    top: int = 0,
    min_error: int = 0,
    participant: Optional[str] = None,
    difficulty: Optional[str] = None,
) -> List[dict]:
    """
    Filter items based on criteria.

    Args:
        items: List of items from wrong_predictions
        top: Keep only top N items (by total_error)
        min_error: Keep items with |total_error| >= min_error
        participant: Filter to specific participant
        difficulty: Filter by difficulty level (LOW/MID/HIGH)

    Returns:
        Filtered list of items
    """
    filtered = items[:]

    # Filter by participant
    if participant:
        filtered = [
            item for item in filtered
            if item.get('participant') == participant
        ]

    # Filter by difficulty
    if difficulty:
        filtered = [
            item for item in filtered
            if item.get('difficulty', '').upper() == difficulty.upper()
        ]

    # Filter by minimum error
    if min_error > 0:
        filtered = [
            item for item in filtered
            if abs(item.get('total_error', 0) or 0) >= min_error
        ]

    # Sort by absolute error (descending)
    filtered.sort(key=lambda x: abs(x.get('total_error', 0) or 0), reverse=True)

    # Take top N
    if top > 0:
        filtered = filtered[:top]

    return filtered


def create_failure_cases_v2(
    wrong_predictions: dict,
    items: List[dict],
    name: str,
    segment_id_lookup: Dict,
) -> dict:
    """
    Create a v2 failure_cases JSON structure from filtered items.

    v2 format: flat list of cases with references only, no data duplication.
    """
    tag = wrong_predictions.get('tag', 'unknown')

    cases = []
    case_counter = 0

    for item in items:
        participant = item.get('participant', 'unknown')
        narration_id = item.get('narration_id', '')

        # Process each wrong segment
        for seg in item.get('wrong_segments', []):
            video_id = seg.get('video_id', '')
            start_ts = seg.get('start_timestamp', 0)
            end_ts = seg.get('end_timestamp', 0)

            # Look up segment_id from timeline data
            key = (participant, narration_id, video_id, round(start_ts, 2), round(end_ts, 2))
            seg_id = segment_id_lookup.get(key)

            if not seg_id:
                # Generate segment_id if not found in timeline
                seg_id = generate_segment_id(narration_id, video_id, start_ts, end_ts)
                print(f"  WARNING: Generated segment_id for {narration_id} (not in timeline)")

            case_counter += 1
            cases.append({
                'case_id': f"FC{case_counter:03d}",
                'participant': participant,
                'narration_id': narration_id,
                'segment_id': seg_id,
                'include': True,
                'priority': 0,
                'notes': '',
                'tags': [],
            })

    output = {
        'name': name,
        'schema_version': 2,
        'vlm_tag': tag,
        'created_at': datetime.now().isoformat(),
        'created_from': f"wrong_predictions_{tag}.json",
        'total_cases': len(cases),
        'original_stats': {
            'total_wrong_items': wrong_predictions.get('total_wrong_items', 0),
            'total_wrong_segments': wrong_predictions.get('total_wrong_segments', 0),
        },
        'cases': cases,
    }

    return output


def main():
    parser = argparse.ArgumentParser(
        description="Initialize a v2 failure_cases file from wrong_predictions"
    )
    parser.add_argument(
        '--tag',
        required=True,
        help='Tag of wrong_predictions file to use (e.g., "qwen", "qwen_low")'
    )
    parser.add_argument(
        '--name',
        default=None,
        help='Name for the failure_cases file (default: same as tag)'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help='Output directory (default: outputs/02_inventory)'
    )
    parser.add_argument(
        '--top',
        type=int,
        default=0,
        help='Keep only top N items by error magnitude (0 = all)'
    )
    parser.add_argument(
        '--min-error',
        type=int,
        default=0,
        help='Keep items with |error| >= this value'
    )
    parser.add_argument(
        '--participant',
        help='Filter to specific participant (e.g., P03)'
    )
    parser.add_argument(
        '--difficulty',
        choices=['LOW', 'MID', 'HIGH'],
        help='Filter by difficulty level'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )

    args = parser.parse_args()

    # Use tag as name if not specified
    name = args.name or args.tag

    print(f"Loading wrong_predictions_{args.tag}.json...")
    wrong_predictions = load_wrong_predictions(args.output_dir, args.tag)

    items = wrong_predictions.get('items', [])
    print(f"Found {len(items)} wrong prediction items")

    # Filter items
    filtered = filter_items(
        items,
        top=args.top,
        min_error=args.min_error,
        participant=args.participant,
        difficulty=args.difficulty,
    )

    print(f"\nFiltered to {len(filtered)} items:")
    if args.top > 0:
        print(f"  - Top {args.top} by error")
    if args.min_error > 0:
        print(f"  - Minimum error: {args.min_error}")
    if args.participant:
        print(f"  - Participant: {args.participant}")
    if args.difficulty:
        print(f"  - Difficulty: {args.difficulty}")

    if not filtered:
        print("\nNo items match the filter criteria.")
        return 1

    # Collect unique participants
    participants = list(set(item.get('participant', '') for item in filtered))
    print(f"\nParticipants: {', '.join(participants)}")

    # Load timeline data to get segment_ids
    print(f"\nLoading timeline data for segment_ids...")
    segment_id_lookup = load_timeline_segment_ids(args.output_dir, participants)
    print(f"Loaded {len(segment_id_lookup)} segments")

    # Create v2 failure cases structure
    failure_cases = create_failure_cases_v2(wrong_predictions, filtered, name, segment_id_lookup)

    # Save output
    failure_cases_dir = args.output_dir / "failure_cases"
    failure_cases_dir.mkdir(parents=True, exist_ok=True)
    output_file = failure_cases_dir / f"failure_cases_{name}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(failure_cases, f, indent=2)

    print(f"\nSaved: {output_file}")
    print(f"Schema version: 2 (reference-based)")
    print(f"Total cases: {failure_cases['total_cases']}")

    # Calculate approximate file size savings
    old_size_estimate = len(filtered) * 2000  # ~2KB per item in v1
    new_size = output_file.stat().st_size
    print(f"File size: {new_size / 1024:.1f} KB (estimated {100 - 100*new_size/old_size_estimate:.0f}% smaller than v1)")

    if args.verbose:
        print(f"\n{'='*60}")
        print("FAILURE CASES (v2 references)")
        print(f"{'='*60}")
        for case in failure_cases['cases'][:10]:  # Show first 10
            print(f"  [{case['case_id']}] {case['participant']} {case['segment_id']}")
        if len(failure_cases['cases']) > 10:
            print(f"  ... and {len(failure_cases['cases']) - 10} more")

    print(f"\n{'='*60}")
    print("NEXT STEPS")
    print(f"{'='*60}")
    print(f"1. Edit {output_file.name} to:")
    print(f"   - Set 'include': false for cases to skip")
    print(f"   - Add 'notes' for observations")
    print(f"   - Add 'tags' for categorization (e.g., ['occlusion', 'small-objects'])")
    print(f"   - Adjust 'priority' values (lower = higher priority)")
    print(f"\n2. Run VLM on failure cases:")
    print(f"   python 07_vlm_QA.py --failure-cases failure_cases_{name}.json --tag {name}_v2")
    print(f"\n3. View in visualizer:")
    print(f"   cd food-inventory-visualizer && npm start")
    print(f"   # Select 'Failure Cases' view")

    return 0


if __name__ == '__main__':
    exit(main())

#!/usr/bin/env python3
"""
Export Lifecycle Edits Tool

Exports inventory items and ALL their lifecycle events to editable files.
User can filter items/events, then use 04_dispensal_classification.py to classify dispensal actions.

Usage:
    # Export all recipes
    python 03_export_lifecycle_edits.py

    # Export specific recipe
    python 03_export_lifecycle_edits.py --recipe-id P01_R03

    # Exclude water items
    python 03_export_lifecycle_edits.py --exclude-water

Output structure:
    outputs/lifecycle_edits/
    └── lifecycle_P01_R03.txt

Output format (YAML-like, editable):
    # Recipe: P01_R03 - Cacio e Pepe
    # Videos: P01-20240202-161354, P01-20240202-161948

    butter:
      - narration_id: P01-20240202-161354-80
        stage: RETRIEVAL
        action: User retrieves butter from fridge
      - narration_id: P01-20240202-161354-81
        stage: DISPENSING
        method: cut_portion
        action: User slices the butter using a knife

    # To exclude an item: delete the entire block or add "# SKIP" before item name
    # To exclude an event: delete the line or prefix with "#"
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional

_SCRIPT_DIR = Path(__file__).parent
_PROJECT_ROOT = _SCRIPT_DIR.parent

DEFAULT_INPUT_DIR = _PROJECT_ROOT / "outputs" / "inventory_lifecycle"
DEFAULT_OUTPUT_DIR = _PROJECT_ROOT / "outputs" / "lifecycle_edits"


def load_lifecycle_data(lifecycle_file: Path) -> Optional[Dict]:
    """Load lifecycle data from JSON file."""
    if not lifecycle_file.exists():
        return None

    with open(lifecycle_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_all_events(events_by_item: Dict) -> Dict[str, List[Dict]]:
    """Extract all lifecycle events from lifecycle data."""
    all_events_by_item = {}

    for item_name, events in events_by_item.items():
        item_events = []
        for event in events:
            stage = event.get('stage', 'UNKNOWN')
            event_data = {
                'narration_id': event.get('narration_id', 'UNKNOWN'),
                'stage': stage,
                'action': event.get('action', '')
            }
            # Include method only for DISPENSING stage
            if stage == 'DISPENSING':
                event_data['method'] = event.get('method', 'unknown')

            item_events.append(event_data)

        if item_events:
            all_events_by_item[item_name] = item_events

    return all_events_by_item


def write_lifecycle_file(
    output_file: Path,
    recipe_id: str,
    recipe_name: str,
    videos: List[str],
    capture_index: Optional[int],
    events_by_item: Dict[str, List[Dict]],
    exclude_water: bool = False
):
    """Write lifecycle actions to editable text file."""

    # Count events by stage
    stage_counts = {'RETRIEVAL': 0, 'ACCESS': 0, 'DISPENSING': 0, 'RESTOCKING': 0, 'OTHER': 0}
    for events in events_by_item.values():
        for event in events:
            stage = event.get('stage', 'OTHER')
            if stage in stage_counts:
                stage_counts[stage] += 1
            else:
                stage_counts['OTHER'] += 1

    with open(output_file, 'w', encoding='utf-8') as f:
        # Header
        f.write(f"# Recipe: {recipe_id} - {recipe_name}\n")
        if capture_index is not None:
            f.write(f"# Capture: {capture_index}\n")
        f.write(f"# Videos: {', '.join(videos)}\n")
        f.write(f"#\n")
        f.write(f"# INSTRUCTIONS:\n")
        f.write(f"#   - To exclude an item: delete the block or add '# SKIP' before item name\n")
        f.write(f"#   - To exclude an event: delete the line or prefix with '#'\n")
        f.write(f"#   - Run 04_dispensal_classification.py to classify DISPENSING events\n")
        f.write(f"#   - Run 05_get_timestamps.py to extract video clips\n")
        f.write(f"#\n")
        f.write(f"# Total items: {len(events_by_item)}\n")
        total_events = sum(len(events) for events in events_by_item.values())
        f.write(f"# Total events: {total_events}\n")
        f.write(f"#   RETRIEVAL: {stage_counts['RETRIEVAL']}\n")
        f.write(f"#   ACCESS: {stage_counts['ACCESS']}\n")
        f.write(f"#   DISPENSING: {stage_counts['DISPENSING']}\n")
        f.write(f"#   RESTOCKING: {stage_counts['RESTOCKING']}\n")
        f.write(f"\n")

        # Sort items alphabetically
        for item_name in sorted(events_by_item.keys()):
            # Skip water if requested
            if exclude_water and 'water' in item_name.lower():
                continue

            events = events_by_item[item_name]

            f.write(f"{item_name}:\n")
            for event in events:
                f.write(f"  - narration_id: {event['narration_id']}\n")
                f.write(f"    stage: {event['stage']}\n")
                if 'method' in event:
                    f.write(f"    method: {event['method']}\n")
                f.write(f"    action: {event['action']}\n")
            f.write(f"\n")


def export_recipe(
    input_dir: Path,
    output_dir: Path,
    recipe_id: str,
    exclude_water: bool = False
) -> bool:
    """Export lifecycle actions for a single recipe/capture."""

    lifecycle_file = input_dir / f"lifecycle_{recipe_id}.json"

    if not lifecycle_file.exists():
        print(f"  WARNING: Lifecycle file not found: {lifecycle_file}")
        return False

    data = load_lifecycle_data(lifecycle_file)
    if not data:
        return False

    recipe_name = data.get('recipe_name', 'Unknown')
    videos = data.get('videos', [])
    capture_index = data.get('capture_index')
    events_by_item = data.get('events_by_item', {})

    # Extract all events
    all_events_by_item = extract_all_events(events_by_item)

    if not all_events_by_item:
        print(f"  WARNING: No events found for {recipe_id}")
        return False

    # Write output file
    output_file = output_dir / f"lifecycle_{recipe_id}.txt"
    write_lifecycle_file(
        output_file, recipe_id, recipe_name, videos,
        capture_index, all_events_by_item, exclude_water
    )

    item_count = len(all_events_by_item)
    event_count = sum(len(events) for events in all_events_by_item.values())

    # Count dispensing events
    dispensing_count = sum(
        1 for events in all_events_by_item.values()
        for e in events if e.get('stage') == 'DISPENSING'
    )

    print(f"  {recipe_id}: {item_count} items, {event_count} events ({dispensing_count} dispensing) -> {output_file.name}")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Export lifecycle actions to editable files"
    )
    parser.add_argument(
        '--recipe-id',
        default=None,
        help='Specific recipe ID to export (e.g., P01_R03 or P01_R01_C0)'
    )
    parser.add_argument(
        '--input-dir',
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help='Directory containing lifecycle JSON files'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help='Output directory for lifecycle action files'
    )
    parser.add_argument(
        '--exclude-water',
        action='store_true',
        help='Exclude water-related items from export'
    )
    parser.add_argument(
        '--participant',
        default=None,
        help='Filter by participant (e.g., P01)'
    )

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Input: {args.input_dir}")
    print(f"Output: {args.output_dir}")
    if args.exclude_water:
        print("Excluding water items")
    print()

    if args.recipe_id:
        # Export single recipe
        success = export_recipe(
            args.input_dir, args.output_dir,
            args.recipe_id, args.exclude_water
        )
        if not success:
            print("Export failed")
            return
    else:
        # Export all recipes
        lifecycle_files = sorted(args.input_dir.glob("lifecycle_*.json"))

        if not lifecycle_files:
            print("ERROR: No lifecycle files found")
            return

        print(f"Found {len(lifecycle_files)} lifecycle files\n")

        success_count = 0
        for lf in lifecycle_files:
            recipe_id = lf.stem.replace("lifecycle_", "")

            # Filter by participant if specified
            if args.participant and not recipe_id.startswith(args.participant):
                continue

            if export_recipe(args.input_dir, args.output_dir, recipe_id, args.exclude_water):
                success_count += 1

        print(f"\nExported {success_count} files to {args.output_dir}")


if __name__ == '__main__':
    main()

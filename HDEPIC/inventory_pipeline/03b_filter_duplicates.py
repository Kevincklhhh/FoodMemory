#!/usr/bin/env python3
"""
Filter Duplicate Events Tool

Automatically comments out events (by narration_id) that have already appeared in previous lifecycle files.
Useful when you want unique events across recipes (e.g., for data collection).

Usage:
    # Preview what would be filtered (dry run)
    python 03b_filter_duplicates.py --dry-run

    # Apply filtering to all files
    python 03b_filter_duplicates.py

    # Filter specific participant
    python 03b_filter_duplicates.py --participant P01

    # Reset: remove all comment markers
    python 03b_filter_duplicates.py --reset

Processing order: Files are processed alphabetically (P01_R01_C0, P01_R01_C1, ..., P01_R03, ...).
Events in earlier files are kept; duplicates in later files are commented out.
"""

import re
import argparse
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict

_SCRIPT_DIR = Path(__file__).parent
_PROJECT_ROOT = _SCRIPT_DIR.parent

DEFAULT_LIFECYCLE_DIR = _PROJECT_ROOT / "outputs" / "lifecycle_edits"


def parse_lifecycle_file(filepath: Path) -> Tuple[List[str], List[Dict]]:
    """
    Parse lifecycle file and return lines and event info.

    Returns:
        lines: All lines in the file
        events: List of dicts with {narration_id, item_name, line_start, line_end, is_commented}
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    events = []
    current_item = None
    i = 0

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Item header: no leading whitespace, ends with ':', not a comment
        if stripped and not line.startswith(' ') and not line.startswith('#') and stripped.endswith(':'):
            current_item = stripped[:-1]
            i += 1
            continue

        # Check for SKIP marker followed by item
        if stripped.startswith('# SKIP'):
            # Next line should be item header
            if i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if next_line.endswith(':'):
                    current_item = next_line[:-1]
                    i += 2
                    continue

        # Event start: "  - narration_id: ..."
        if line.startswith('  - narration_id:') or line.startswith('  # - narration_id:'):
            is_commented = line.strip().startswith('#')
            event_start = i

            # Extract narration_id
            if is_commented:
                narr_match = re.search(r'#\s*-\s*narration_id:\s*(\S+)', line)
            else:
                narr_match = re.search(r'-\s*narration_id:\s*(\S+)', line)

            if narr_match:
                narration_id = narr_match.group(1)

                # Find event end (next event or next item or end)
                event_end = i + 1
                while event_end < len(lines):
                    next_line = lines[event_end]
                    # Next event
                    if next_line.strip().startswith('- narration_id:') or next_line.strip().startswith('# - narration_id:'):
                        break
                    # Next item (no indent, ends with :)
                    if next_line.strip() and not next_line.startswith(' ') and not next_line.startswith('#'):
                        break
                    # SKIP marker
                    if '# SKIP' in next_line:
                        break
                    event_end += 1

                events.append({
                    'narration_id': narration_id,
                    'item_name': current_item,
                    'line_start': event_start,
                    'line_end': event_end,
                    'is_commented': is_commented
                })

        i += 1

    return lines, events


def comment_event(lines: List[str], event: Dict) -> List[str]:
    """Comment out an event by adding # to each line."""
    for i in range(event['line_start'], event['line_end']):
        line = lines[i]
        # Only comment if not already commented and has content
        if line.strip() and not line.strip().startswith('#'):
            # Preserve indentation
            indent = len(line) - len(line.lstrip())
            lines[i] = line[:indent] + '# ' + line[indent:]
    return lines


def uncomment_event(lines: List[str], event: Dict) -> List[str]:
    """Remove comment markers from an event."""
    for i in range(event['line_start'], event['line_end']):
        line = lines[i]
        # Remove # marker if present
        if line.strip().startswith('#'):
            # Find the # and remove it (plus optional space)
            idx = line.index('#')
            if idx + 1 < len(line) and line[idx + 1] == ' ':
                lines[i] = line[:idx] + line[idx + 2:]
            else:
                lines[i] = line[:idx] + line[idx + 1:]
    return lines


def process_files(
    lifecycle_dir: Path,
    participant: str = None,
    dry_run: bool = False,
    reset: bool = False,
    verbose: bool = False
) -> Dict[str, int]:
    """
    Process all lifecycle files and filter duplicate events by narration_id.

    Returns:
        Dict mapping recipe_id -> count of filtered events
    """
    # Find all lifecycle files
    files = sorted(lifecycle_dir.glob("lifecycle_*.txt"))

    if participant:
        files = [f for f in files if f.stem.replace("lifecycle_", "").startswith(participant)]

    if not files:
        print("No lifecycle files found")
        return {}

    print(f"Processing {len(files)} files...")
    if dry_run:
        print("DRY RUN - no files will be modified\n")
    elif reset:
        print("RESET MODE - uncommenting all events\n")
    else:
        print()

    # Track seen narration_ids across all files
    seen_narration_ids: Set[str] = set()
    # Track what was filtered per file
    filtered_counts: Dict[str, int] = {}
    # Track first occurrence
    first_occurrence: Dict[str, str] = {}

    for filepath in files:
        recipe_id = filepath.stem.replace("lifecycle_", "")
        lines, events = parse_lifecycle_file(filepath)

        if reset:
            # Uncomment all events
            modified = False
            for event in events:
                if event['is_commented']:
                    lines = uncomment_event(lines, event)
                    modified = True

            if modified and not dry_run:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.writelines(lines)
                print(f"  {recipe_id}: Reset comments")
            else:
                print(f"  {recipe_id}: No changes needed")
            continue

        modified = False
        filtered_count = 0
        kept_count = 0

        for event in events:
            narr_id = event['narration_id']

            # Skip already commented events
            if event['is_commented']:
                if verbose:
                    print(f"  {recipe_id}: {narr_id} ({event['item_name']}) - already commented")
                continue

            # Check if this is a duplicate
            if narr_id in seen_narration_ids:
                filtered_count += 1
                if verbose:
                    print(f"  {recipe_id}: {narr_id} ({event['item_name']}) <- duplicate (first in {first_occurrence[narr_id]})")

                if not dry_run:
                    lines = comment_event(lines, event)
                    modified = True
            else:
                # First occurrence - remember it
                seen_narration_ids.add(narr_id)
                first_occurrence[narr_id] = recipe_id
                kept_count += 1
                if verbose:
                    print(f"  {recipe_id}: {narr_id} ({event['item_name']}) - kept")

        # Write back if modified
        if modified and not dry_run:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.writelines(lines)

        filtered_counts[recipe_id] = filtered_count

        # Summary for this file
        if filtered_count > 0:
            print(f"  {recipe_id}: {filtered_count} duplicates filtered, {kept_count} kept")
        else:
            print(f"  {recipe_id}: {kept_count} events (no duplicates)")

    return filtered_counts


def main():
    parser = argparse.ArgumentParser(
        description="Filter duplicate events (by narration_id) across lifecycle files"
    )
    parser.add_argument(
        '--lifecycle-dir',
        type=Path,
        default=DEFAULT_LIFECYCLE_DIR,
        help='Directory containing lifecycle edit files'
    )
    parser.add_argument(
        '--participant',
        default=None,
        help='Filter by participant (e.g., P01)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview changes without modifying files'
    )
    parser.add_argument(
        '--reset',
        action='store_true',
        help='Uncomment all events (undo filtering)'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Show details for each event'
    )

    args = parser.parse_args()

    print(f"Lifecycle directory: {args.lifecycle_dir}")

    filtered_counts = process_files(
        args.lifecycle_dir,
        args.participant,
        args.dry_run,
        args.reset,
        args.verbose
    )

    if filtered_counts and not args.reset:
        print(f"\n{'='*60}")
        total_filtered = sum(filtered_counts.values())
        print(f"Total duplicate events {'would be ' if args.dry_run else ''}filtered: {total_filtered}")

        if args.dry_run:
            print("\nRun without --dry-run to apply changes")


if __name__ == '__main__':
    main()

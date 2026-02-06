#!/usr/bin/env python3
"""
migrate_add_segment_ids.py - Add segment_ids to existing timeline_annotated files

This migration script adds segment_id to each dispensal_segment in existing
timeline_annotated.json files. The segment_id is a stable hash-based identifier
that allows referencing segments from failure_cases without data duplication.

Usage:
    # Dry run (preview changes)
    python migrate_add_segment_ids.py --dry-run

    # Apply changes
    python migrate_add_segment_ids.py

    # Specific participant
    python migrate_add_segment_ids.py --participant P03
"""

import argparse
import json
from pathlib import Path

from inventory_utils import DEFAULT_OUTPUT_DIR, generate_segment_id, add_segment_ids_to_item


def migrate_timeline_file(timeline_file: Path, dry_run: bool = False) -> dict:
    """
    Add segment_ids to a timeline_annotated file.

    Returns:
        Dict with migration stats
    """
    with open(timeline_file, 'r') as f:
        data = json.load(f)

    items = data.get('items', [])
    stats = {
        'total_items': len(items),
        'total_segments': 0,
        'segments_with_id': 0,
        'segments_added_id': 0,
    }

    for item in items:
        narration_id = item.get('narration_id', '')
        segments = item.get('dispensal_segments', [])

        for seg in segments:
            stats['total_segments'] += 1

            if seg.get('segment_id'):
                stats['segments_with_id'] += 1
                continue

            # Generate segment_id
            video_id = seg.get('video_id', '')
            start_ts = seg.get('start_timestamp', 0)
            end_ts = seg.get('end_timestamp', 0)

            seg_id = generate_segment_id(narration_id, video_id, start_ts, end_ts)
            seg['segment_id'] = seg_id
            stats['segments_added_id'] += 1

    if not dry_run and stats['segments_added_id'] > 0:
        with open(timeline_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Add segment_ids to existing timeline_annotated files"
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help='Output directory (default: outputs/02_inventory)'
    )
    parser.add_argument(
        '--participant',
        help='Migrate specific participant only (e.g., P03)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview changes without writing to files'
    )

    args = parser.parse_args()

    print(f"{'='*60}")
    print("MIGRATION: Add segment_ids to timeline_annotated files")
    print(f"{'='*60}")

    if args.dry_run:
        print("DRY RUN MODE - no files will be modified\n")

    # Find timeline_annotated files
    if args.participant:
        pattern = f"{args.participant}/{args.participant}_timeline_annotated.json"
    else:
        pattern = "*/*_timeline_annotated.json"

    timeline_files = list(args.output_dir.glob(pattern))

    if not timeline_files:
        print(f"No timeline_annotated files found matching {pattern}")
        return 1

    print(f"Found {len(timeline_files)} timeline_annotated files\n")

    total_stats = {
        'files_processed': 0,
        'files_modified': 0,
        'total_segments': 0,
        'segments_with_id': 0,
        'segments_added_id': 0,
    }

    for timeline_file in sorted(timeline_files):
        participant = timeline_file.parent.name
        print(f"Processing {participant}...", end=" ")

        stats = migrate_timeline_file(timeline_file, dry_run=args.dry_run)

        total_stats['files_processed'] += 1
        total_stats['total_segments'] += stats['total_segments']
        total_stats['segments_with_id'] += stats['segments_with_id']
        total_stats['segments_added_id'] += stats['segments_added_id']

        if stats['segments_added_id'] > 0:
            total_stats['files_modified'] += 1
            action = "would add" if args.dry_run else "added"
            print(f"{action} {stats['segments_added_id']} segment_ids")
        else:
            print(f"already has all segment_ids")

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Files processed: {total_stats['files_processed']}")
    print(f"Files modified: {total_stats['files_modified']}")
    print(f"Total segments: {total_stats['total_segments']}")
    print(f"Segments already with ID: {total_stats['segments_with_id']}")
    print(f"Segments added ID: {total_stats['segments_added_id']}")

    if args.dry_run and total_stats['segments_added_id'] > 0:
        print(f"\nRun without --dry-run to apply changes")

    return 0


if __name__ == '__main__':
    exit(main())

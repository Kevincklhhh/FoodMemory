#!/usr/bin/env python3
"""
migrate_add_segment_tags.py - Add tags/notes fields to dispensal_segments
and mark "difficult" segments where all 3 models get the count wrong.

Usage:
    # Dry run (preview changes)
    python migrate_add_segment_tags.py --dry-run

    # Apply changes
    python migrate_add_segment_tags.py

    # Specific participant
    python migrate_add_segment_tags.py --participant P03
"""

import argparse
import json
from pathlib import Path

from inventory_utils import DEFAULT_OUTPUT_DIR

BASE = Path(__file__).resolve().parent.parent / "outputs" / "02_inventory"

EVAL_REPORTS = {
    "qwen": BASE / "vlm_qa_hybrid_no_transfer_qwen_count_eval_report.json",
    "gpt5_v2": BASE / "vlm_qa_hybrid_no_transfer_gpt5_v2_count_eval_report.json",
    "gemini3": BASE / "vlm_qa_hybrid_gemini3_batch_low_count_eval_report.json",
}


def load_all_wrong_segments(eval_reports: dict) -> set:
    """Identify segments where all 3 models get the count wrong.

    Returns a set of (participant, narration_id, segment_idx) tuples.
    """
    models = list(eval_reports.keys())
    eval_data = {}

    for name, path in eval_reports.items():
        if not path.exists():
            print(f"  WARNING: eval report not found: {path}")
            continue
        with open(path) as f:
            report = json.load(f)
        segs = {}
        for detail in report["details"]:
            participant = detail["participant"]
            for item in detail["items"]:
                narration_id = item["narration_id"]
                for seg in item["segments"]:
                    key = (participant, narration_id, seg["segment_idx"])
                    segs[key] = {
                        "is_correct": seg["is_correct"],
                        "skipped": seg.get("skipped", False),
                    }
        eval_data[name] = segs
        print(f"  Loaded {len(segs)} segments from {name}")

    if len(eval_data) < len(eval_reports):
        print("  WARNING: not all eval reports found, difficult tagging may be incomplete")

    available_models = list(eval_data.keys())
    if not available_models:
        return set()

    # Collect all keys present in any report
    all_keys = set()
    for segs in eval_data.values():
        all_keys.update(segs.keys())

    all_wrong = set()
    for key in all_keys:
        wrong_count = 0
        evaluated_count = 0
        for m in available_models:
            seg = eval_data[m].get(key)
            if seg and not seg.get("skipped", False):
                evaluated_count += 1
                if not seg["is_correct"]:
                    wrong_count += 1
        # All evaluated models must be wrong, and at least all models must have evaluated
        if evaluated_count == len(available_models) and wrong_count == evaluated_count:
            all_wrong.add(key)

    print(f"  Found {len(all_wrong)} all-wrong segments across {len(all_keys)} total")
    return all_wrong


def build_segment_id_to_key(timeline_data: dict, participant: str) -> dict:
    """Build a mapping from segment_id -> (participant, narration_id, segment_idx)."""
    lookup = {}
    for item in timeline_data.get("items", []):
        narration_id = item.get("narration_id", "")
        for idx, seg in enumerate(item.get("dispensal_segments", [])):
            seg_id = seg.get("segment_id")
            if seg_id:
                lookup[seg_id] = (participant, narration_id, idx)
    return lookup


def migrate_timeline_file(timeline_file: Path, all_wrong: set, dry_run: bool = False) -> dict:
    """Add tags/notes fields and mark difficult segments."""
    with open(timeline_file, 'r') as f:
        data = json.load(f)

    participant = timeline_file.parent.name
    items = data.get('items', [])

    stats = {
        'total_segments': 0,
        'fields_added': 0,
        'difficult_tagged': 0,
        'already_tagged': 0,
    }

    for item in items:
        narration_id = item.get('narration_id', '')
        segments = item.get('dispensal_segments', [])

        for idx, seg in enumerate(segments):
            stats['total_segments'] += 1

            # Add tags field if missing
            if 'tags' not in seg:
                seg['tags'] = []
                stats['fields_added'] += 1

            # Add notes field if missing
            if 'notes' not in seg:
                seg['notes'] = ''
                stats['fields_added'] += 1

            # Check if this segment is all-wrong
            key = (participant, narration_id, idx)
            if key in all_wrong:
                if 'difficult' not in seg['tags']:
                    seg['tags'].append('difficult')
                    stats['difficult_tagged'] += 1
                else:
                    stats['already_tagged'] += 1

    if not dry_run and (stats['fields_added'] > 0 or stats['difficult_tagged'] > 0):
        with open(timeline_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Add tags/notes to dispensal_segments and mark difficult segments"
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
    print("MIGRATION: Add tags/notes + mark difficult segments")
    print(f"{'='*60}")

    if args.dry_run:
        print("DRY RUN MODE - no files will be modified\n")

    # Load eval reports to identify all-wrong segments
    print("Loading eval reports...")
    all_wrong = load_all_wrong_segments(EVAL_REPORTS)

    # Find timeline_annotated files
    if args.participant:
        pattern = f"{args.participant}/{args.participant}_timeline_annotated.json"
    else:
        pattern = "*/*_timeline_annotated.json"

    timeline_files = list(args.output_dir.glob(pattern))

    if not timeline_files:
        print(f"No timeline_annotated files found matching {pattern}")
        return 1

    print(f"\nFound {len(timeline_files)} timeline_annotated files\n")

    total_stats = {
        'files_processed': 0,
        'files_modified': 0,
        'total_segments': 0,
        'fields_added': 0,
        'difficult_tagged': 0,
        'already_tagged': 0,
    }

    for timeline_file in sorted(timeline_files):
        participant = timeline_file.parent.name
        print(f"Processing {participant}...", end=" ")

        stats = migrate_timeline_file(timeline_file, all_wrong, dry_run=args.dry_run)

        total_stats['files_processed'] += 1
        total_stats['total_segments'] += stats['total_segments']
        total_stats['fields_added'] += stats['fields_added']
        total_stats['difficult_tagged'] += stats['difficult_tagged']
        total_stats['already_tagged'] += stats['already_tagged']

        modified = stats['fields_added'] > 0 or stats['difficult_tagged'] > 0
        if modified:
            total_stats['files_modified'] += 1
            action = "would" if args.dry_run else "did"
            parts = []
            if stats['fields_added'] > 0:
                parts.append(f"add fields to {stats['fields_added']} segs")
            if stats['difficult_tagged'] > 0:
                parts.append(f"tag {stats['difficult_tagged']} difficult")
            print(f"{action} {', '.join(parts)}")
        else:
            extra = f" ({stats['already_tagged']} already difficult)" if stats['already_tagged'] else ""
            print(f"no changes needed{extra}")

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Files processed: {total_stats['files_processed']}")
    print(f"Files modified: {total_stats['files_modified']}")
    print(f"Total segments: {total_stats['total_segments']}")
    print(f"Fields added: {total_stats['fields_added']}")
    print(f"Difficult tagged: {total_stats['difficult_tagged']}")
    print(f"Already tagged: {total_stats['already_tagged']}")

    if args.dry_run and (total_stats['fields_added'] > 0 or total_stats['difficult_tagged'] > 0):
        print(f"\nRun without --dry-run to apply changes")

    return 0


if __name__ == '__main__':
    exit(main())

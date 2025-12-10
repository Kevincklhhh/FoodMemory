#!/usr/bin/env python3
"""
Gemini Export Tool

Export narrations to TXT format for manual Gemini web processing.

Task 1: Filter - Export single video narrations for food classification
    python tools/gemini_export.py filter --video-id P01-20240202-110250

Task 2: Inventory - Export filtered food narrations for inventory discovery
    python tools/gemini_export.py inventory --video-id P01-20240202-110250
    python tools/gemini_export.py inventory --start-video P01-20240202-110250 --end-video P01-20240203-093333

Output format:
    narration_id | narration_text
"""

import csv
import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Set

# Default paths
DEFAULT_CSV_PATH = Path("/home/kailaic/NeuroTrace/kitchen/HDEPIC/participant_P01_narrations.csv")
DEFAULT_OUTPUT_DIR = Path("/home/kailaic/NeuroTrace/kitchen/HDEPIC/outputs/gemini_input")


def load_narrations_for_video(csv_path: Path, video_id: str) -> List[Dict]:
    """Load all narrations for a specific video."""
    narrations = []

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['video_id'] == video_id:
                narrations.append({
                    'unique_narration_id': row['unique_narration_id'],
                    'narration': row['narration'].strip(),
                    'start_timestamp': float(row['start_timestamp']),
                })

    # Sort by timestamp
    narrations.sort(key=lambda x: x['start_timestamp'])
    return narrations


def get_all_video_ids(csv_path: Path) -> List[str]:
    """Get list of all unique video IDs in the CSV."""
    video_ids = set()

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            video_ids.add(row['video_id'])

    return sorted(video_ids)


def export_for_filter(video_id: str, csv_path: Path, output_dir: Path) -> Path:
    """
    Export single video narrations for food filtering.

    Output format: narration_id | narration_text
    Also creates an empty response file for user to paste Gemini output.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load narrations
    narrations = load_narrations_for_video(csv_path, video_id)

    if not narrations:
        print(f"ERROR: No narrations found for video {video_id}")
        return None

    # Write input TXT
    output_file = output_dir / f"filter_{video_id}.txt"

    with open(output_file, 'w', encoding='utf-8') as f:
        for narr in narrations:
            line = f"{narr['unique_narration_id']} | {narr['narration']}\n"
            f.write(line)

    # Create empty response file for user to paste Gemini output
    response_file = output_dir / f"filter_{video_id}_response.txt"
    response_file.touch()

    return output_file


def load_filter_response(response_file: Path) -> Optional[Set[str]]:
    """Load narration IDs from Task 1 filter response (JSON array)."""
    if not response_file.exists():
        return None

    with open(response_file, 'r', encoding='utf-8') as f:
        content = f.read().strip()

    if not content:
        return None

    try:
        narration_ids = json.loads(content)
        if isinstance(narration_ids, list):
            return set(narration_ids)
    except json.JSONDecodeError:
        print(f"WARNING: Could not parse JSON from {response_file}")
        return None

    return None


def export_for_inventory(
    video_ids: List[str],
    csv_path: Path,
    input_dir: Path,
    output_dir: Path
) -> Optional[Path]:
    """
    Export filtered food narrations for inventory discovery.

    Uses Task 1 response files to get filtered narration IDs,
    then looks up full narration text from CSV.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    all_narrations = []
    videos_processed = []
    videos_skipped = []

    for video_id in video_ids:
        # Load filter response for this video
        response_file = input_dir / f"filter_{video_id}_response.txt"
        filtered_ids = load_filter_response(response_file)

        if filtered_ids is None:
            videos_skipped.append(video_id)
            continue

        # Load narrations from CSV
        narrations = load_narrations_for_video(csv_path, video_id)

        # Filter to only food-related narrations
        for narr in narrations:
            if narr['unique_narration_id'] in filtered_ids:
                narr['video_id'] = video_id
                all_narrations.append(narr)

        videos_processed.append(video_id)

    if not all_narrations:
        print("ERROR: No filtered narrations found")
        return None

    # Determine output filename
    if len(video_ids) == 1:
        output_file = output_dir / f"inventory_{video_ids[0]}.txt"
    else:
        output_file = output_dir / f"inventory_{video_ids[0]}_to_{video_ids[-1]}.txt"

    # Write to TXT with video headers
    with open(output_file, 'w', encoding='utf-8') as f:
        current_video = None
        for narr in all_narrations:
            if narr['video_id'] != current_video:
                if current_video is not None:
                    f.write("\n")
                current_video = narr['video_id']
                f.write(f"=== VIDEO: {current_video} ===\n")

            line = f"{narr['unique_narration_id']} | {narr['narration']}\n"
            f.write(line)

    return output_file, videos_processed, videos_skipped, len(all_narrations)


def main():
    parser = argparse.ArgumentParser(
        description="Export narrations to TXT for Gemini web processing"
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Filter command
    filter_parser = subparsers.add_parser(
        'filter',
        help='Export single video for food filtering'
    )
    filter_parser.add_argument(
        '--video-id',
        required=True,
        help='Video ID to export (e.g., P01-20240202-110250)'
    )
    filter_parser.add_argument(
        '--csv',
        type=Path,
        default=DEFAULT_CSV_PATH,
        help='Path to narrations CSV'
    )
    filter_parser.add_argument(
        '--output-dir',
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help='Output directory for TXT files'
    )

    # Inventory command
    inventory_parser = subparsers.add_parser(
        'inventory',
        help='Export filtered food narrations for inventory discovery'
    )
    inventory_parser.add_argument(
        '--video-id',
        default=None,
        help='Single video ID to export'
    )
    inventory_parser.add_argument(
        '--start-video',
        default=None,
        help='Start video ID for range export'
    )
    inventory_parser.add_argument(
        '--end-video',
        default=None,
        help='End video ID for range export'
    )
    inventory_parser.add_argument(
        '--csv',
        type=Path,
        default=DEFAULT_CSV_PATH,
        help='Path to narrations CSV'
    )
    inventory_parser.add_argument(
        '--input-dir',
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help='Directory containing filter response files'
    )
    inventory_parser.add_argument(
        '--output-dir',
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help='Output directory for TXT files'
    )

    # List command (helper)
    list_parser = subparsers.add_parser(
        'list',
        help='List all available video IDs'
    )
    list_parser.add_argument(
        '--csv',
        type=Path,
        default=DEFAULT_CSV_PATH,
        help='Path to narrations CSV'
    )

    args = parser.parse_args()

    if args.command == 'filter':
        print(f"Exporting narrations for video: {args.video_id}")
        print(f"CSV: {args.csv}")
        print(f"Output: {args.output_dir}")

        output_file = export_for_filter(args.video_id, args.csv, args.output_dir)

        if output_file:
            # Count lines
            with open(output_file, 'r') as f:
                line_count = sum(1 for _ in f)

            response_file = output_file.parent / f"filter_{args.video_id}_response.txt"

            print(f"\nExported {line_count} narrations to:")
            print(f"  {output_file}")
            print(f"\nResponse file (paste Gemini output here):")
            print(f"  {response_file}")
            print(f"\nPaste contents into Gemini web interface for food classification.")

    elif args.command == 'inventory':
        # Determine video IDs to process
        all_video_ids = get_all_video_ids(args.csv)

        if args.video_id:
            # Single video
            video_ids = [args.video_id]
        elif args.start_video and args.end_video:
            # Video range
            try:
                start_idx = all_video_ids.index(args.start_video)
                end_idx = all_video_ids.index(args.end_video)
                video_ids = all_video_ids[start_idx:end_idx + 1]
            except ValueError as e:
                print(f"ERROR: Video not found - {e}")
                return
        else:
            print("ERROR: Specify --video-id OR --start-video and --end-video")
            return

        print(f"Exporting filtered narrations for inventory discovery")
        print(f"Videos: {len(video_ids)} ({video_ids[0]} to {video_ids[-1]})")
        print(f"Input (filter responses): {args.input_dir}")
        print(f"Output: {args.output_dir}")

        result = export_for_inventory(video_ids, args.csv, args.input_dir, args.output_dir)

        if result:
            output_file, processed, skipped, total_narrations = result

            print(f"\nProcessed {len(processed)} videos, skipped {len(skipped)}")
            if skipped:
                print(f"  Skipped (no filter response): {', '.join(skipped)}")

            print(f"\nExported {total_narrations} filtered narrations to:")
            print(f"  {output_file}")
            print(f"\nPaste contents into Gemini web interface for inventory discovery.")

    elif args.command == 'list':
        video_ids = get_all_video_ids(args.csv)
        print(f"Found {len(video_ids)} videos:\n")
        for vid in video_ids:
            print(f"  {vid}")

    else:
        parser.print_help()


if __name__ == '__main__':
    main()

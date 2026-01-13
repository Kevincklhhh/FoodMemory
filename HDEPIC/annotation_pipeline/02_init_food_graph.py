#!/usr/bin/env python3
"""
Initialize Food Graph Directory

Creates a new food graph directory with:
1. Filtered narrations from 01_filter/filter_*.txt - already filtered by Gemini
2. Empty inventory.json template - user fills from Gemini output
3. Placeholder for state_change.json (populated by 04_parse_responses.py)

Reads from:
    outputs/01_filter/filter_{video_id}.txt         # All narrations
    outputs/01_filter/filter_{video_id}_response.txt  # Gemini filter response

Workflow:
1. Run this script to create directory and filtered narration file
2. User copies narrations to Gemini, gets inventory items back
3. User pastes Gemini output into inventory.json
4. Run 04_parse_responses.py to populate state_change.json
5. Run 05_build_graph.py to build the graph

Usage:
    python 02_init_food_graph.py --start-video P01-20240202-161354 --end-video P01-20240202-161948
    python 02_init_food_graph.py --video-id P01-20240202-110250  # Single video
"""

import json
import argparse
import re
from pathlib import Path
from typing import List, Dict, Set, Tuple

# Default paths (relative to this script's location in annotation_pipeline/)
_SCRIPT_DIR = Path(__file__).parent
_PROJECT_ROOT = _SCRIPT_DIR.parent

DEFAULT_GEMINI_INPUT_DIR = _PROJECT_ROOT / "outputs" / "01_filter"
DEFAULT_OUTPUT_DIR = _PROJECT_ROOT / "outputs" / "03_graphs"


def load_filter_files(gemini_input_dir: Path, video_id: str) -> Tuple[List[str], Set[int]]:
    """
    Load filter input and response files for a video.

    Returns:
        Tuple of (all_lines, relevant_line_numbers)
    """
    input_file = gemini_input_dir / f"filter_{video_id}.txt"
    response_file = gemini_input_dir / f"filter_{video_id}_response.txt"

    if not input_file.exists():
        print(f"  WARNING: No filter file found: {input_file.name}")
        return [], set()

    # Read all narration lines
    with open(input_file, 'r', encoding='utf-8') as f:
        all_lines = f.read().strip().split('\n')

    # Parse response to get relevant line numbers
    relevant_lines = set()
    if response_file.exists():
        with open(response_file, 'r', encoding='utf-8') as f:
            response_text = f.read().strip()

        if response_text:
            try:
                # Parse JSON response
                response_data = json.loads(response_text)

                # Extract relevant_lines from all segments
                if isinstance(response_data, list):
                    for segment in response_data:
                        if isinstance(segment, dict) and 'relevant_lines' in segment:
                            relevant_lines.update(segment['relevant_lines'])
            except json.JSONDecodeError:
                print(f"  WARNING: Could not parse response: {response_file.name}")
    else:
        print(f"  WARNING: No response file found: {response_file.name}")

    return all_lines, relevant_lines


def get_videos_in_range(gemini_input_dir: Path, start_video: str, end_video: str) -> List[str]:
    """Get list of video IDs in range that have filter files."""
    # Find all filter files
    filter_files = list(gemini_input_dir.glob("filter_P*.txt"))
    video_ids = []

    for f in filter_files:
        # Extract video ID from filename like "filter_P01-20240202-110250.txt"
        match = re.match(r'filter_(P\d+-\d+-\d+)\.txt', f.name)
        if match and not f.name.endswith('_response.txt'):
            video_ids.append(match.group(1))

    video_ids.sort()

    # Filter to range
    try:
        start_idx = video_ids.index(start_video)
        end_idx = video_ids.index(end_video)
        return video_ids[start_idx:end_idx + 1]
    except ValueError as e:
        print(f"ERROR: Video not found: {e}")
        print(f"Available videos: {video_ids[:10]}...")
        return []


def format_filtered_narrations(
    gemini_input_dir: Path,
    video_ids: List[str],
    include_all: bool = False
) -> Tuple[str, int, int]:
    """
    Format filtered narrations for all videos.

    Returns:
        Tuple of (formatted_text, total_lines, filtered_lines)
    """
    lines = []
    total_count = 0
    filtered_count = 0

    for video_id in video_ids:
        all_lines, relevant_lines = load_filter_files(gemini_input_dir, video_id)

        if not all_lines:
            continue

        # Add video header
        if lines:
            lines.append("")  # Blank line between videos
        lines.append(f"=== VIDEO: {video_id} ===")

        total_count += len(all_lines)

        for i, line in enumerate(all_lines, 1):
            if include_all or i in relevant_lines:
                lines.append(line)
                filtered_count += 1

    return "\n".join(lines), total_count, filtered_count


def create_inventory_template() -> List[Dict]:
    """Create empty inventory template with example structure."""
    return [
        {
            "narration ID": "EXAMPLE-001",
            "food_name": "example_food",
            "source_action": "Example: Pick up the food from storage"
        }
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Initialize food graph directory with narrations and empty inventory"
    )
    parser.add_argument(
        '--video-id',
        default=None,
        help='Single video ID (shorthand for --start-video X --end-video X)'
    )
    parser.add_argument(
        '--start-video',
        default=None,
        help='Start video ID for range (e.g., P01-20240202-161354)'
    )
    parser.add_argument(
        '--end-video',
        default=None,
        help='End video ID for range (e.g., P01-20240202-161948)'
    )
    parser.add_argument(
        '--gemini-input-dir',
        type=Path,
        default=DEFAULT_GEMINI_INPUT_DIR,
        help='Directory containing filter_*.txt files'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help='Base output directory for food graphs'
    )
    parser.add_argument(
        '--all-narrations',
        action='store_true',
        help='Include all narrations, not just Gemini-filtered ones'
    )

    args = parser.parse_args()

    # Handle --video-id as shorthand
    if args.video_id:
        args.start_video = args.video_id
        args.end_video = args.video_id

    if not args.start_video or not args.end_video:
        print("ERROR: Must specify --video-id OR both --start-video and --end-video")
        parser.print_help()
        return

    # Create range name
    if args.start_video == args.end_video:
        range_name = args.start_video
    else:
        range_name = f"{args.start_video}_to_{args.end_video}"

    print("=" * 60)
    print("INITIALIZE FOOD GRAPH DIRECTORY")
    print("=" * 60)
    print(f"Video range: {args.start_video} -> {args.end_video}")
    print(f"Range name:  {range_name}")
    print(f"Input:       {args.gemini_input_dir}")
    print(f"Output:      {args.output_dir / range_name}")

    # Get videos in range
    print(f"\n[1/3] Finding videos in range...")
    video_ids = get_videos_in_range(args.gemini_input_dir, args.start_video, args.end_video)
    if not video_ids:
        print("ERROR: No videos found in specified range")
        return
    print(f"      Found {len(video_ids)} videos: {video_ids}")

    # Format filtered narrations
    print(f"\n[2/3] Loading filtered narrations...")
    narrations_text, total_count, filtered_count = format_filtered_narrations(
        args.gemini_input_dir,
        video_ids,
        include_all=args.all_narrations
    )
    print(f"      Total narrations: {total_count}")
    print(f"      Filtered (food-related): {filtered_count}")

    if not narrations_text:
        print("ERROR: No narrations found")
        return

    # Create output directory
    work_dir = args.output_dir / range_name
    work_dir.mkdir(parents=True, exist_ok=True)

    # Write narrations file
    print(f"\n[3/3] Writing output files...")

    narrations_file = work_dir / "narrations_for_gemini.txt"
    with open(narrations_file, 'w', encoding='utf-8') as f:
        f.write(narrations_text)
    print(f"      Created: {narrations_file.name} ({filtered_count} narrations)")

    # Write empty inventory template
    inventory_file = work_dir / "inventory.json"
    if not inventory_file.exists():
        with open(inventory_file, 'w', encoding='utf-8') as f:
            json.dump(create_inventory_template(), f, indent=2)
        print(f"      Created: {inventory_file.name} (template - fill with Gemini output)")
    else:
        print(f"      Skipped: {inventory_file.name} (already exists)")

    # Create placeholder state_change.json info
    state_change_file = work_dir / "state_change.json"
    if not state_change_file.exists():
        print(f"      Note: {state_change_file.name} will be created by 04_parse_responses.py")

    # Summary
    print(f"\n{'=' * 60}")
    print("COMPLETE")
    print(f"{'=' * 60}")
    print(f"Directory: {work_dir}")
    print(f"\nNext steps:")
    print(f"  1. Open {narrations_file.name} and copy to Gemini")
    print(f"  2. Ask Gemini to identify food arrivals (first appearance of each food)")
    print(f"  3. Paste Gemini's JSON output into {inventory_file.name}")
    print(f"  4. Run: python 04_parse_responses.py --start-video {args.start_video} --end-video {args.end_video}")
    print(f"  5. Run: python 05_build_graph.py --dir {work_dir}")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Initialize Food Graph Directory

Creates a new food graph directory with:
1. Filtered narrations (food-related) for the video range - for Gemini prompt
2. Empty inventory.json template - user fills from Gemini output
3. Placeholder for state_change.json (populated by parse_state_inference.py)

Workflow:
1. Run this script to create directory and narration file
2. User copies narrations to Gemini, gets inventory items back
3. User pastes Gemini output into inventory.json
4. Run parse_state_inference.py to populate state_change.json
5. Run 06_food_graph_from_gemini.py to build the graph

Usage:
    python 00_init_food_graph.py --start-video P01-20240202-161354 --end-video P01-20240202-161948
    python 00_init_food_graph.py --video-id P01-20240202-110250  # Single video
"""

import json
import csv
import argparse
from pathlib import Path
from typing import List, Dict, Optional

# Default paths (relative to this script's location in gemini_pipelines/)
_SCRIPT_DIR = Path(__file__).parent
_PROJECT_ROOT = _SCRIPT_DIR.parent

DEFAULT_CSV_PATH = _PROJECT_ROOT / "P01" / "participant_P01_narrations.csv"
DEFAULT_OUTPUT_DIR = _PROJECT_ROOT / "gemini_outputs" / "food_graph"

# Keywords that suggest food-related narrations
FOOD_KEYWORDS = [
    # Food items
    'pasta', 'cheese', 'butter', 'salt', 'pepper', 'water', 'oil', 'sauce',
    'egg', 'bread', 'meat', 'vegetable', 'fruit', 'milk', 'cream', 'flour',
    'sugar', 'rice', 'noodle', 'chicken', 'beef', 'pork', 'fish', 'tomato',
    'onion', 'garlic', 'potato', 'carrot', 'lettuce', 'spinach', 'mushroom',
    'coffee', 'tea', 'juice', 'wine', 'beer', 'soup', 'broth', 'stock',
    # Containers/tools related to food
    'pan', 'pot', 'bowl', 'plate', 'cup', 'glass', 'mug', 'jar', 'bottle',
    'kettle', 'grater', 'spatula', 'ladle', 'fork', 'knife', 'spoon',
    'strainer', 'colander', 'fridge', 'oven', 'stove', 'hob', 'scale',
    # Actions
    'pour', 'stir', 'mix', 'cook', 'boil', 'fry', 'bake', 'grate', 'slice',
    'cut', 'chop', 'dice', 'peel', 'season', 'taste', 'eat', 'drink',
    'fill', 'empty', 'heat', 'melt', 'dissolve',
]


def load_narrations(csv_path: Path) -> List[Dict]:
    """Load narrations from CSV file."""
    narrations = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            narrations.append(row)
    return narrations


def get_video_ids_from_csv(narrations: List[Dict]) -> List[str]:
    """Get unique video IDs from narrations, sorted chronologically."""
    video_ids = sorted(set(n['video_id'] for n in narrations))
    return video_ids


def filter_narrations_by_video_range(
    narrations: List[Dict],
    start_video: str,
    end_video: str
) -> List[Dict]:
    """Filter narrations to specified video range."""
    # Get all video IDs and find range
    all_videos = get_video_ids_from_csv(narrations)

    try:
        start_idx = all_videos.index(start_video)
        end_idx = all_videos.index(end_video)
    except ValueError as e:
        print(f"ERROR: Video not found: {e}")
        print(f"Available videos: {all_videos[:10]}...")
        return []

    videos_in_range = set(all_videos[start_idx:end_idx + 1])
    return [n for n in narrations if n['video_id'] in videos_in_range]


def is_food_related(narration_text: str) -> bool:
    """Check if narration is food-related based on keywords."""
    text_lower = narration_text.lower()
    return any(keyword in text_lower for keyword in FOOD_KEYWORDS)


def filter_food_narrations(narrations: List[Dict]) -> List[Dict]:
    """Filter to only food-related narrations."""
    return [n for n in narrations if is_food_related(n.get('narration', ''))]


def format_narrations_for_prompt(narrations: List[Dict]) -> str:
    """Format narrations as text for Gemini prompt."""
    lines = []
    current_video = None

    for n in narrations:
        video_id = n['video_id']
        narration_id = n['narration_id']
        narration_text = n['narration']

        # Add video header when video changes
        if video_id != current_video:
            if current_video is not None:
                lines.append("")  # Blank line between videos
            lines.append(f"=== VIDEO: {video_id} ===")
            current_video = video_id

        lines.append(f"{narration_id} | {narration_text}")

    return "\n".join(lines)


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
        '--csv-path',
        type=Path,
        default=DEFAULT_CSV_PATH,
        help='Path to narrations CSV'
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
        help='Include all narrations, not just food-related ones'
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
    print(f"CSV:         {args.csv_path}")
    print(f"Output:      {args.output_dir / range_name}")

    # Load narrations
    print(f"\n[1/4] Loading narrations from CSV...")
    narrations = load_narrations(args.csv_path)
    print(f"      Total narrations: {len(narrations)}")

    # Filter to video range
    print(f"\n[2/4] Filtering to video range...")
    range_narrations = filter_narrations_by_video_range(
        narrations, args.start_video, args.end_video
    )
    print(f"      Narrations in range: {len(range_narrations)}")

    if not range_narrations:
        print("ERROR: No narrations found in specified range")
        return

    # Filter to food-related (optional)
    if args.all_narrations:
        filtered_narrations = range_narrations
        print(f"\n[3/4] Using all narrations (--all-narrations flag)")
    else:
        print(f"\n[3/4] Filtering to food-related narrations...")
        filtered_narrations = filter_food_narrations(range_narrations)
        print(f"      Food-related narrations: {len(filtered_narrations)}")

    # Create output directory
    work_dir = args.output_dir / range_name
    work_dir.mkdir(parents=True, exist_ok=True)

    # Write narrations file
    print(f"\n[4/4] Writing output files...")

    narrations_file = work_dir / "narrations_for_gemini.txt"
    narrations_text = format_narrations_for_prompt(filtered_narrations)
    with open(narrations_file, 'w', encoding='utf-8') as f:
        f.write(narrations_text)
    print(f"      Created: {narrations_file.name}")

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
        print(f"      Note: {state_change_file.name} will be created by parse_state_inference.py")

    # Summary
    print(f"\n{'=' * 60}")
    print("COMPLETE")
    print(f"{'=' * 60}")
    print(f"Directory: {work_dir}")
    print(f"\nNext steps:")
    print(f"  1. Open {narrations_file.name} and copy to Gemini")
    print(f"  2. Ask Gemini to identify food arrivals (first appearance of each food)")
    print(f"  3. Paste Gemini's JSON output into {inventory_file.name}")
    print(f"  4. Run: python parse_state_inference.py --start-video {args.start_video} --end-video {args.end_video}")
    print(f"  5. Run: python 06_food_graph_from_gemini.py --dir {work_dir}")


if __name__ == '__main__':
    main()

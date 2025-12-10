#!/usr/bin/env python3
"""
Parse State Inference Responses

Parses response.txt files from state_inference chunks and combines them into
a single state_change.json file with global timestamps.

Usage:
    python parse_state_inference.py --start-video P01-20240202-161354 --end-video P01-20240202-161948
    python parse_state_inference.py --video-id P01-20240202-110250  # Single video

Input:
    gemini_outputs/state_inference/{video_id}/
    ├── chunk_000/
    │   └── response.txt   # Gemini output (JSON array of events)
    ├── chunk_001/
    │   └── response.txt
    └── manifest.json      # Contains global_start/global_end for timestamp conversion

Output:
    gemini_outputs/food_graph/{start_video}_to_{end_video}/
    └── state_change.json  # Combined events with global timestamps
"""

import json
import argparse
import re
from pathlib import Path
from typing import List, Dict, Optional

# Default paths (relative to this script's location in gemini_pipelines/)
_SCRIPT_DIR = Path(__file__).parent
_PROJECT_ROOT = _SCRIPT_DIR.parent

DEFAULT_STATE_INFERENCE_DIR = _PROJECT_ROOT / "gemini_outputs" / "state_inference"
DEFAULT_OUTPUT_DIR = _PROJECT_ROOT / "gemini_outputs" / "food_graph"


def get_video_ids_in_range(
    state_inference_dir: Path,
    start_video: str,
    end_video: str
) -> List[str]:
    """Get list of video IDs in the specified range that have state_inference data."""
    # Get all video directories that have manifest.json
    all_videos = []
    for video_dir in state_inference_dir.iterdir():
        if video_dir.is_dir() and (video_dir / "manifest.json").exists():
            all_videos.append(video_dir.name)

    # Sort chronologically (video IDs are formatted as P01-YYYYMMDD-HHMMSS)
    all_videos.sort()

    # Filter to range
    try:
        start_idx = all_videos.index(start_video)
        end_idx = all_videos.index(end_video)
        return all_videos[start_idx:end_idx + 1]
    except ValueError as e:
        print(f"ERROR: Video not found in state_inference directory: {e}")
        print(f"Available videos: {all_videos}")
        return []


def extract_json_from_response(response_text: str) -> Optional[List[Dict]]:
    """
    Extract JSON array from response text.

    Handles cases where response might have:
    - Pure JSON
    - JSON wrapped in markdown code blocks (```json ... ```)
    - Extra text before/after JSON
    """
    if not response_text.strip():
        return None

    # Try to find JSON in markdown code block first
    code_block_match = re.search(r'```(?:json)?\s*(\[[\s\S]*?\])\s*```', response_text)
    if code_block_match:
        try:
            return json.loads(code_block_match.group(1))
        except json.JSONDecodeError:
            pass

    # Try to find raw JSON array
    json_match = re.search(r'\[[\s\S]*\]', response_text)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass

    # Try parsing the whole thing as JSON
    try:
        result = json.loads(response_text)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass

    return None


def parse_video_responses(
    video_id: str,
    video_order: int,
    state_inference_dir: Path
) -> List[Dict]:
    """
    Parse all response.txt files for a single video.

    Args:
        video_id: Video identifier
        video_order: Order of this video in the processing sequence (for sorting)
        state_inference_dir: Directory containing state_inference outputs

    Returns list of events with global timestamps and video_order for sorting.
    """
    video_dir = state_inference_dir / video_id
    manifest_file = video_dir / "manifest.json"

    if not manifest_file.exists():
        print(f"  WARNING: No manifest.json found for {video_id}")
        return []

    # Load manifest
    with open(manifest_file, 'r') as f:
        manifest = json.load(f)

    events = []

    for chunk_info in manifest.get('chunks', []):
        chunk_idx = chunk_info['chunk_idx']
        global_start = chunk_info['global_start']

        # Determine response file path
        chunk_dir = video_dir / f"chunk_{chunk_idx:03d}"
        response_file = chunk_dir / "response.txt"

        if not response_file.exists():
            print(f"  WARNING: No response.txt found for {video_id}/chunk_{chunk_idx:03d}")
            continue

        # Read and parse response
        with open(response_file, 'r', encoding='utf-8') as f:
            response_text = f.read()

        if not response_text.strip():
            print(f"  WARNING: Empty response.txt for {video_id}/chunk_{chunk_idx:03d}")
            continue

        chunk_events = extract_json_from_response(response_text)

        if chunk_events is None:
            print(f"  ERROR: Could not parse JSON from {video_id}/chunk_{chunk_idx:03d}/response.txt")
            continue

        print(f"    Chunk {chunk_idx}: {len(chunk_events)} events")

        # Convert local timestamps to global timestamps
        for event in chunk_events:
            # Add global_start to local timestamps
            if 'timestamp_start' in event:
                event['timestamp_start'] = event['timestamp_start'] + global_start
            if 'timestamp_end' in event:
                event['timestamp_end'] = event['timestamp_end'] + global_start

            # Add video_order for sorting (will be removed before output)
            event['_video_order'] = video_order
            event['_video_id'] = video_id

            events.append(event)

    return events


def parse_state_inference(
    start_video: str,
    end_video: str,
    state_inference_dir: Path,
    output_dir: Path
) -> Optional[Path]:
    """
    Parse state inference responses for a range of videos.

    Returns path to output state_change.json file.
    """
    # Get videos in range
    video_ids = get_video_ids_in_range(state_inference_dir, start_video, end_video)

    if not video_ids:
        print("ERROR: No videos found in specified range")
        return None

    print(f"\nProcessing {len(video_ids)} videos:")
    for vid in video_ids:
        print(f"  - {vid}")

    # Parse all videos
    all_events = []

    for video_order, video_id in enumerate(video_ids):
        print(f"\nParsing {video_id}...")
        video_events = parse_video_responses(video_id, video_order, state_inference_dir)
        all_events.extend(video_events)
        print(f"  Total events from {video_id}: {len(video_events)}")

    if not all_events:
        print("\nERROR: No events parsed from any video")
        return None

    # Sort events by video order first, then by timestamp within each video
    all_events.sort(key=lambda e: (e.get('_video_order', 0), e.get('timestamp_start', 0)))

    # Re-number event IDs sequentially and remove internal sorting fields
    for i, event in enumerate(all_events, 1):
        event['event_id'] = i
        # Remove internal fields used for sorting
        event.pop('_video_order', None)
        event.pop('_video_id', None)

    # Create output directory
    if start_video == end_video:
        range_name = start_video
    else:
        range_name = f"{start_video}_to_{end_video}"

    output_range_dir = output_dir / range_name
    output_range_dir.mkdir(parents=True, exist_ok=True)

    # Save state_change.json
    output_file = output_range_dir / "state_change.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_events, f, indent=2)

    return output_file


def main():
    parser = argparse.ArgumentParser(
        description="Parse state inference responses into state_change.json"
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
        '--state-inference-dir',
        type=Path,
        default=DEFAULT_STATE_INFERENCE_DIR,
        help='Directory containing state_inference outputs'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help='Output directory for state_change.json'
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

    print("=" * 60)
    print("PARSE STATE INFERENCE RESPONSES")
    print("=" * 60)
    print(f"Video range: {args.start_video} -> {args.end_video}")
    print(f"Input:  {args.state_inference_dir}")
    print(f"Output: {args.output_dir}")

    output_file = parse_state_inference(
        start_video=args.start_video,
        end_video=args.end_video,
        state_inference_dir=args.state_inference_dir,
        output_dir=args.output_dir
    )

    if output_file:
        # Count events
        with open(output_file, 'r') as f:
            events = json.load(f)

        print(f"\n{'=' * 60}")
        print("COMPLETE")
        print(f"{'=' * 60}")
        print(f"Output: {output_file}")
        print(f"Total events: {len(events)}")

        # Show first few events as preview
        if events:
            print(f"\nFirst 3 events preview:")
            for event in events[:3]:
                print(f"  {event['event_id']}: {event.get('primary_action', 'N/A')} "
                      f"({event.get('timestamp_start', 0):.1f}s - {event.get('timestamp_end', 0):.1f}s)")


if __name__ == '__main__':
    main()

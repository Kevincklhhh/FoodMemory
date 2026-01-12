#!/usr/bin/env python3
"""
Extract Lifecycle Video Clips Tool

Reads lifecycle edit files and extracts video clips for ALL inventory lifecycle actions.
Skips commented-out events (filtered duplicates).
Each clip is named to trace back to the inventory transaction.

Usage:
    # Process all lifecycle files
    python 05_get_timestamps.py

    # Process specific recipe
    python 05_get_timestamps.py --recipe-id P01_R03

    # Custom padding (default: 5 seconds before and after)
    python 05_get_timestamps.py --pad-before 3 --pad-after 3

    # Skip video extraction, only generate index
    python 05_get_timestamps.py --index-only

    # Only extract DISPENSING stage clips
    python 05_get_timestamps.py --stage DISPENSING

Output structure:
    outputs/lifecycle_clips/
    ├── P01_R03/
    │   ├── index.json           # Full metadata for all clips
    │   ├── summary.txt          # Human-readable summary
    │   ├── black_pepper_jar_P01-20240202-161354-118_RETRIEVAL.mp4
    │   ├── black_pepper_jar_P01-20240202-161948-62_DISPENSING.mp4
    │   └── ...
    └── P01_R08/
        └── ...

Index JSON contains:
    - clip_filename
    - narration_id, video_id
    - original_start, original_end (narration timestamps)
    - clip_start, clip_end (with padding applied)
    - item_name, stage, method (for DISPENSING)
    - action, narration
"""

import csv
import json
import re
import subprocess
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

_SCRIPT_DIR = Path(__file__).parent
_PROJECT_ROOT = _SCRIPT_DIR.parent

DEFAULT_LIFECYCLE_DIR = _PROJECT_ROOT / "outputs" / "lifecycle_edits"
DEFAULT_OUTPUT_DIR = _PROJECT_ROOT / "outputs" / "lifecycle_clips"
DEFAULT_CSV_PATH = _PROJECT_ROOT / "P01" / "participant_P01_narrations.csv"
DEFAULT_VIDEO_DIR = _PROJECT_ROOT / "P01"


def load_narrations_csv(csv_path: Path) -> Dict[str, Dict]:
    """Load narrations CSV and create lookup by narration_id."""
    narrations = {}

    if not csv_path.exists():
        print(f"WARNING: CSV file not found: {csv_path}")
        return narrations

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            narr_id = row.get('unique_narration_id', '')
            if narr_id:
                narrations[narr_id] = {
                    'video_id': row.get('video_id', ''),
                    'start_timestamp': float(row.get('start_timestamp', 0)),
                    'end_timestamp': float(row.get('end_timestamp', 0)),
                    'narration': row.get('narration', '').strip()
                }

    return narrations


def parse_lifecycle_file(filepath: Path) -> Tuple[Dict, List[Dict]]:
    """
    Parse lifecycle edit file and extract metadata and events.
    Skips commented-out events.

    Returns:
        metadata: {recipe_id, recipe_name, videos, capture_index}
        events: List of {narration_id, item_name, stage, method, action}
    """
    metadata = {
        'recipe_id': filepath.stem.replace('lifecycle_', ''),
        'recipe_name': '',
        'videos': [],
        'capture_index': None
    }
    events = []

    if not filepath.exists():
        return metadata, events

    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Parse header comments
    for line in lines:
        if line.startswith('# Recipe:'):
            match = re.match(r'# Recipe: (\S+) - (.+)', line)
            if match:
                metadata['recipe_id'] = match.group(1)
                metadata['recipe_name'] = match.group(2).strip()
        elif line.startswith('# Capture:'):
            match = re.match(r'# Capture: (\d+)', line)
            if match:
                metadata['capture_index'] = int(match.group(1))
        elif line.startswith('# Videos:'):
            match = re.match(r'# Videos: (.+)', line)
            if match:
                metadata['videos'] = [v.strip() for v in match.group(1).split(',')]

    # Parse items and events
    current_item = None
    i = 0

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Skip empty lines and header comments
        if not stripped:
            i += 1
            continue

        # Item header: no leading whitespace, ends with ':', not a comment
        if not line.startswith(' ') and not line.startswith('#') and stripped.endswith(':'):
            current_item = stripped[:-1]
            i += 1
            continue

        # Skip SKIP markers and commented items
        if stripped.startswith('# SKIP') or (not line.startswith(' ') and stripped.startswith('#')):
            i += 1
            continue

        # Event start: "  - narration_id: ..." (not commented)
        if line.startswith('  - narration_id:') and current_item:
            # Parse event fields
            event = {'item_name': current_item}

            # narration_id
            narr_match = re.search(r'-\s*narration_id:\s*(\S+)', line)
            if narr_match:
                event['narration_id'] = narr_match.group(1)

            # Read subsequent fields until next event or item
            j = i + 1
            while j < len(lines):
                next_line = lines[j]
                next_stripped = next_line.strip()

                # Stop at next event, next item, or comment
                if next_stripped.startswith('- narration_id:') or next_stripped.startswith('# -'):
                    break
                if next_stripped and not next_line.startswith(' '):
                    break

                # Parse field
                if next_stripped.startswith('stage:'):
                    event['stage'] = next_stripped.replace('stage:', '').strip()
                elif next_stripped.startswith('method:'):
                    event['method'] = next_stripped.replace('method:', '').strip()
                elif next_stripped.startswith('action:'):
                    event['action'] = next_stripped.replace('action:', '').strip()

                j += 1

            # Only add if we have required fields
            if 'narration_id' in event and 'stage' in event:
                events.append(event)

            i = j
            continue

        # Skip commented events
        if line.startswith('  # - narration_id:'):
            # Skip until next event or item
            j = i + 1
            while j < len(lines):
                next_line = lines[j]
                if next_line.strip().startswith('- narration_id:') or next_line.strip().startswith('# - narration_id:'):
                    break
                if next_line.strip() and not next_line.startswith(' ') and not next_line.startswith('#'):
                    break
                j += 1
            i = j
            continue

        i += 1

    return metadata, events


def sanitize_filename(name: str) -> str:
    """Sanitize string for use in filename."""
    sanitized = re.sub(r'[^\w\-]', '_', name)
    sanitized = re.sub(r'_+', '_', sanitized)
    sanitized = sanitized.strip('_')
    return sanitized[:50]


def get_video_duration(video_path: Path) -> float:
    """Get video duration in seconds using ffprobe."""
    try:
        result = subprocess.run(
            [
                'ffprobe', '-v', 'error',
                '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                str(video_path)
            ],
            capture_output=True, text=True, check=True
        )
        return float(result.stdout.strip())
    except (subprocess.CalledProcessError, ValueError):
        return 0.0


def extract_clip(
    video_path: Path,
    output_path: Path,
    start_time: float,
    end_time: float,
    video_duration: float = 0.0
) -> bool:
    """Extract video clip using ffmpeg."""
    start_time = max(0, start_time)
    if video_duration > 0:
        end_time = min(end_time, video_duration)

    duration = end_time - start_time
    if duration <= 0:
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        subprocess.run(
            [
                'ffmpeg', '-y',
                '-ss', str(start_time),
                '-i', str(video_path),
                '-t', str(duration),
                '-c:v', 'libx264',
                '-c:a', 'aac',
                '-preset', 'fast',
                '-crf', '23',
                str(output_path)
            ],
            capture_output=True, check=True
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"    ERROR: {e.stderr.decode()[:200] if e.stderr else str(e)}")
        return False


def process_recipe(
    lifecycle_file: Path,
    narrations: Dict[str, Dict],
    output_dir: Path,
    video_dir: Path,
    pad_before: float = 5.0,
    pad_after: float = 5.0,
    index_only: bool = False,
    verbose: bool = False,
    stage_filter: str = None
) -> Dict:
    """Process a single lifecycle file and extract video clips."""

    metadata, events = parse_lifecycle_file(lifecycle_file)

    if not events:
        print(f"  WARNING: No events in {lifecycle_file.name}")
        return {}

    recipe_id = metadata['recipe_id']
    recipe_name = metadata['recipe_name']

    # Filter by stage if specified
    if stage_filter:
        events = [e for e in events if e.get('stage') == stage_filter]
        if not events:
            print(f"  WARNING: No {stage_filter} events in {lifecycle_file.name}")
            return {}

    # Output directory for this recipe
    recipe_output_dir = output_dir / recipe_id
    recipe_output_dir.mkdir(parents=True, exist_ok=True)

    # Track video durations for clamping
    video_durations = {}

    # Count events by stage
    stage_counts = defaultdict(int)
    for e in events:
        stage_counts[e.get('stage', 'UNKNOWN')] += 1

    # Build index
    index = {
        'recipe_id': recipe_id,
        'recipe_name': recipe_name,
        'videos': metadata.get('videos', []),
        'capture_index': metadata.get('capture_index'),
        'stage_counts': dict(stage_counts),
        'pad_before': pad_before,
        'pad_after': pad_after,
        'clips': []
    }

    extracted_count = 0
    skipped_count = 0

    for i, event in enumerate(events):
        narr_id = event.get('narration_id', '')

        # Get timestamp info from narrations CSV
        narr_data = narrations.get(narr_id, {})
        video_id = narr_data.get('video_id', '')
        original_start = narr_data.get('start_timestamp', 0)
        original_end = narr_data.get('end_timestamp', 0)

        if not video_id:
            # Try to extract from narration_id (format: P01-20240202-161948-62)
            parts = narr_id.rsplit('-', 1)
            if len(parts) == 2:
                video_id = parts[0]

        if not video_id or original_start == 0:
            if verbose:
                print(f"    [{i+1}/{len(events)}] SKIP {narr_id}: missing timestamp")
            skipped_count += 1
            continue

        # Calculate clip times with padding
        clip_start = max(0, original_start - pad_before)
        clip_end = original_end + pad_after

        # Get video duration for clamping
        if video_id not in video_durations:
            video_path = video_dir / f"{video_id}.mp4"
            if video_path.exists():
                video_durations[video_id] = get_video_duration(video_path)
            else:
                video_durations[video_id] = 0

        video_duration = video_durations.get(video_id, 0)
        if video_duration > 0:
            clip_end = min(clip_end, video_duration)

        # Build filename: {item}_{narration_id}_{STAGE}.mp4
        stage = event.get('stage', 'UNKNOWN')
        item_name = event.get('item_name', 'unknown')
        item_sanitized = sanitize_filename(item_name)

        clip_filename = f"{item_sanitized}_{narr_id}_{stage}.mp4"
        clip_path = recipe_output_dir / clip_filename

        # Build clip metadata
        clip_meta = {
            'clip_filename': clip_filename,
            'narration_id': narr_id,
            'video_id': video_id,
            'original_start': original_start,
            'original_end': original_end,
            'clip_start': clip_start,
            'clip_end': clip_end,
            'duration': clip_end - clip_start,
            'item_name': item_name,
            'stage': stage,
            'method': event.get('method', ''),
            'action': event.get('action', ''),
            'narration': narr_data.get('narration', '')
        }

        index['clips'].append(clip_meta)

        # Extract video clip
        if not index_only:
            video_path = video_dir / f"{video_id}.mp4"
            if video_path.exists():
                success = extract_clip(
                    video_path, clip_path,
                    clip_start, clip_end,
                    video_duration
                )
                if success:
                    extracted_count += 1
                    if verbose:
                        print(f"    [{i+1}/{len(events)}] {stage} {item_sanitized} -> {clip_filename}")
                else:
                    skipped_count += 1
            else:
                if verbose:
                    print(f"    [{i+1}/{len(events)}] SKIP: video not found {video_path}")
                skipped_count += 1
        else:
            extracted_count += 1

    # Write index file
    index_path = recipe_output_dir / "index.json"
    with open(index_path, 'w', encoding='utf-8') as f:
        json.dump(index, f, indent=2)

    # Write human-readable summary
    summary_path = recipe_output_dir / "summary.txt"
    write_summary(summary_path, index)

    return {
        'recipe_id': recipe_id,
        'total': len(events),
        'extracted': extracted_count,
        'skipped': skipped_count,
        'stage_counts': dict(stage_counts),
        'output_dir': str(recipe_output_dir)
    }


def write_summary(output_path: Path, index: Dict):
    """Write human-readable summary file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"# Recipe: {index['recipe_id']} - {index['recipe_name']}\n")
        f.write(f"# Videos: {', '.join(index.get('videos', []))}\n")
        if index.get('capture_index') is not None:
            f.write(f"# Capture: {index['capture_index']}\n")
        f.write(f"# Padding: -{index['pad_before']}s / +{index['pad_after']}s\n")

        stage_counts = index.get('stage_counts', {})
        if stage_counts:
            f.write(f"# Stages: ")
            f.write(", ".join(f"{k}={v}" for k, v in sorted(stage_counts.items())))
            f.write("\n")
        f.write(f"# Total clips: {len(index.get('clips', []))}\n")
        f.write("\n")

        # Group by stage
        clips_by_stage = defaultdict(list)
        for clip in index.get('clips', []):
            stage = clip.get('stage', 'UNKNOWN')
            clips_by_stage[stage].append(clip)

        for stage in ['RETRIEVAL', 'ACCESS', 'DISPENSING', 'RESTOCKING', 'UNKNOWN']:
            clips = clips_by_stage.get(stage, [])
            if not clips:
                continue

            f.write(f"=== {stage} ({len(clips)} clips) ===\n\n")
            for clip in clips:
                method_str = f"[{clip['method']:12}]" if clip.get('method') else "[            ]"
                f.write(f"{method_str} {clip['item_name']}\n")
                f.write(f"  File: {clip['clip_filename']}\n")
                f.write(f"  Time: {clip['original_start']:.2f} - {clip['original_end']:.2f} (clip: {clip['clip_start']:.2f} - {clip['clip_end']:.2f})\n")
                action = clip.get('action', '')
                f.write(f"  Action: {action[:100]}...\n" if len(action) > 100 else f"  Action: {action}\n")
                f.write("\n")


def main():
    parser = argparse.ArgumentParser(
        description="Extract video clips for all inventory lifecycle actions"
    )
    parser.add_argument(
        '--recipe-id',
        default=None,
        help='Specific recipe ID to process (e.g., P01_R03)'
    )
    parser.add_argument(
        '--lifecycle-dir',
        type=Path,
        default=DEFAULT_LIFECYCLE_DIR,
        help='Directory containing lifecycle edit files'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help='Output directory for video clips'
    )
    parser.add_argument(
        '--video-dir',
        type=Path,
        default=DEFAULT_VIDEO_DIR,
        help='Directory containing source video files'
    )
    parser.add_argument(
        '--csv',
        type=Path,
        default=DEFAULT_CSV_PATH,
        help='Path to narrations CSV'
    )
    parser.add_argument(
        '--pad-before',
        type=float,
        default=5.0,
        help='Seconds to pad before event start (default: 5)'
    )
    parser.add_argument(
        '--pad-after',
        type=float,
        default=5.0,
        help='Seconds to pad after event end (default: 5)'
    )
    parser.add_argument(
        '--stage',
        choices=['RETRIEVAL', 'ACCESS', 'DISPENSING', 'RESTOCKING'],
        default=None,
        help='Only extract clips for specific stage'
    )
    parser.add_argument(
        '--index-only',
        action='store_true',
        help='Only generate index files, skip video extraction'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Verbose output'
    )
    parser.add_argument(
        '--participant',
        default=None,
        help='Filter by participant (e.g., P01)'
    )

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Lifecycle files: {args.lifecycle_dir}")
    print(f"Video directory: {args.video_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Padding: -{args.pad_before}s / +{args.pad_after}s")
    if args.stage:
        print(f"Stage filter: {args.stage}")
    if args.index_only:
        print("Mode: INDEX ONLY (no video extraction)")
    print()

    # Load narrations
    print("Loading narrations CSV...")
    narrations = load_narrations_csv(args.csv)
    print(f"  Loaded {len(narrations)} narrations\n")

    if args.recipe_id:
        # Process single recipe
        lifecycle_file = args.lifecycle_dir / f"lifecycle_{args.recipe_id}.txt"
        if not lifecycle_file.exists():
            print(f"ERROR: Lifecycle file not found: {lifecycle_file}")
            return

        print(f"Processing {args.recipe_id}...")
        result = process_recipe(
            lifecycle_file, narrations, args.output_dir, args.video_dir,
            args.pad_before, args.pad_after, args.index_only, args.verbose,
            args.stage
        )
        if result:
            print(f"  Extracted: {result['extracted']}/{result['total']} clips")
            print(f"  Stages: {result['stage_counts']}")
            print(f"  Output: {result['output_dir']}")
    else:
        # Process all lifecycle files
        lifecycle_files = sorted(args.lifecycle_dir.glob("lifecycle_*.txt"))

        if not lifecycle_files:
            print("ERROR: No lifecycle files found")
            return

        print(f"Found {len(lifecycle_files)} lifecycle files\n")

        total_extracted = 0
        total_events = 0
        total_stage_counts = defaultdict(int)

        for lf in lifecycle_files:
            recipe_id = lf.stem.replace("lifecycle_", "")

            # Filter by participant if specified
            if args.participant and not recipe_id.startswith(args.participant):
                continue

            print(f"  {recipe_id}: ", end='', flush=True)
            result = process_recipe(
                lf, narrations, args.output_dir, args.video_dir,
                args.pad_before, args.pad_after, args.index_only, args.verbose,
                args.stage
            )

            if result:
                total_extracted += result['extracted']
                total_events += result['total']
                for stage, count in result.get('stage_counts', {}).items():
                    total_stage_counts[stage] += count
                print(f"{result['extracted']}/{result['total']} clips")
            else:
                print("SKIPPED")

        print(f"\n{'='*60}")
        print(f"COMPLETE: {total_extracted}/{total_events} clips extracted")
        print(f"Stages: {dict(total_stage_counts)}")
        print(f"Output: {args.output_dir}")
        print(f"{'='*60}")


if __name__ == '__main__':
    main()

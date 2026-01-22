#!/usr/bin/env python3
"""
Rename video files based on their recording timestamp metadata.
Extracts creation_time from video metadata and renames to yyyymmdd-hhmmss format.
"""

import subprocess
import json
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo


def get_creation_time(video_path):
    """Extract creation_time from video metadata using ffprobe."""
    try:
        cmd = [
            'ffprobe',
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_format',
            '-show_streams',
            video_path
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        metadata = json.loads(result.stdout)

        # Try to find creation_time in format tags
        if 'format' in metadata and 'tags' in metadata['format']:
            tags = metadata['format']['tags']
            # Prioritize Apple's creationdate (more accurate for iOS/Meta devices)
            if 'com.apple.quicktime.creationdate' in tags:
                return tags['com.apple.quicktime.creationdate']
            if 'creation_time' in tags:
                return tags['creation_time']

        # Try to find creation_time in stream tags
        if 'streams' in metadata:
            for stream in metadata['streams']:
                if 'tags' in stream and 'creation_time' in stream['tags']:
                    return stream['tags']['creation_time']

        return None
    except Exception as e:
        print(f"Error reading metadata from {video_path}: {e}")
        return None


def parse_timestamp(creation_time_str):
    """Parse creation_time string to datetime object."""
    try:
        # Handle format: "2025-10-22T02:18:13.000000Z" or "2025-10-21T22:18:13-0400"
        # Replace 'Z' with '+00:00' for UTC timestamps
        if creation_time_str.endswith('Z'):
            creation_time_str = creation_time_str.replace('Z', '+00:00')

        # Handle timezone offset without colon (e.g., "-0400" -> "-04:00")
        # ISO format expects colon in timezone offset
        import re
        # Match timezone offset at end: +/-HHMM
        creation_time_str = re.sub(r'([+-]\d{2})(\d{2})$', r'\1:\2', creation_time_str)

        dt = datetime.fromisoformat(creation_time_str)
        return dt
    except Exception as e:
        print(f"Error parsing timestamp '{creation_time_str}': {e}")
        return None


def rename_video_files(directory='.', dry_run=True, timezone='UTC'):
    """
    Rename all video files in directory based on their recording timestamp.

    Args:
        directory: Path to directory containing video files
        dry_run: If True, only print what would be renamed without actually renaming
        timezone: Target timezone for filename (e.g., 'America/New_York', 'America/Detroit', 'UTC')
    """
    video_extensions = {'.mov', '.MOV', '.mp4', '.MP4', '.avi', '.AVI', '.mkv', '.MKV'}

    dir_path = Path(directory)
    if not dir_path.exists():
        print(f"Error: Directory {directory} does not exist")
        return

    video_files = [f for f in dir_path.iterdir() if f.suffix in video_extensions and f.is_file()]

    if not video_files:
        print(f"No video files found in {directory}")
        return

    print(f"Found {len(video_files)} video file(s)")
    print(f"Mode: {'DRY RUN (no changes will be made)' if dry_run else 'RENAMING FILES'}")
    print(f"Timezone: {timezone}\n")

    renamed_count = 0
    skipped_count = 0

    for video_file in sorted(video_files):
        print(f"Processing: {video_file.name}")

        # Get creation time
        creation_time = get_creation_time(str(video_file))
        if not creation_time:
            print(f"  ⚠ Skipping: Could not extract creation_time metadata\n")
            skipped_count += 1
            continue

        # Parse timestamp
        dt = parse_timestamp(creation_time)
        if not dt:
            print(f"  ⚠ Skipping: Could not parse timestamp\n")
            skipped_count += 1
            continue

        # Convert to target timezone
        try:
            target_tz = ZoneInfo(timezone)
            dt_local = dt.astimezone(target_tz)
        except Exception as e:
            print(f"  ⚠ Error converting timezone: {e}")
            dt_local = dt

        # Format new filename: yyyymmdd-hhmmss
        new_name = dt_local.strftime('%Y%m%d-%H%M%S') + video_file.suffix
        new_path = video_file.parent / new_name

        # Check if file already exists
        if new_path.exists() and new_path != video_file:
            # Add suffix to avoid collision
            counter = 1
            while new_path.exists():
                new_name = dt.strftime('%Y%m%d-%H%M%S') + f"_{counter}" + video_file.suffix
                new_path = video_file.parent / new_name
                counter += 1
            print(f"  ⚠ File exists, using: {new_name}")

        print(f"  Creation time (UTC): {creation_time}")
        print(f"  Local time ({timezone}): {dt_local.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  New name: {new_name}")

        if not dry_run:
            try:
                video_file.rename(new_path)
                print(f"  ✓ Renamed successfully")
                renamed_count += 1
            except Exception as e:
                print(f"  ✗ Error renaming: {e}")
                skipped_count += 1
        else:
            print(f"  → Would rename to: {new_name}")
            renamed_count += 1

        print()

    print("=" * 60)
    if dry_run:
        print(f"Summary (DRY RUN): {renamed_count} file(s) would be renamed, {skipped_count} skipped")
        print("\nTo actually rename files, run with --execute flag")
    else:
        print(f"Summary: {renamed_count} file(s) renamed, {skipped_count} skipped")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Rename video files based on recording timestamp metadata',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run (preview changes)
  python rename_by_timestamp.py

  # Actually rename files
  python rename_by_timestamp.py --execute

  # Rename files in specific directory
  python rename_by_timestamp.py --execute --dir /path/to/videos
        """
    )

    parser.add_argument(
        '--execute',
        action='store_true',
        help='Actually rename files (default is dry run)'
    )

    parser.add_argument(
        '--dir',
        default='.',
        help='Directory containing video files (default: current directory)'
    )

    parser.add_argument(
        '--timezone',
        default='America/Detroit',
        help='Timezone for filename (default: America/Detroit). Use UTC for no conversion.'
    )

    args = parser.parse_args()

    rename_video_files(directory=args.dir, dry_run=not args.execute, timezone=args.timezone)
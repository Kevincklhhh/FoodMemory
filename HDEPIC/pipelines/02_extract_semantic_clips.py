#!/usr/bin/env python3
"""
Extract Video Clips for Semantic Groups

This script extracts video clips corresponding to each semantic group
from the narration_grouping/*.json files.

Features:
- Variable-length clips based on semantic group boundaries
- Duration padding: clips shorter than min_duration are padded equally on both sides
- Records both original and padded timestamps in manifest

Usage:
    python 02_extract_semantic_clips.py --video-id P01-20240202-110250
    python 02_extract_semantic_clips.py --all-videos
    python 02_extract_semantic_clips.py --video-id P01-20240202-110250 --dry-run
"""

import json
import argparse
import subprocess
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict

from semantic_utils import (
    SemanticGroup,
    find_videos_with_groupings,
    load_and_enrich_video_groups,
)


# Default paths (relative to this script's location in pipelines/)
_SCRIPT_DIR = Path(__file__).parent
_PROJECT_ROOT = _SCRIPT_DIR.parent

DEFAULT_VIDEO_DIR = _PROJECT_ROOT / "P01"
DEFAULT_GROUPING_DIR = _PROJECT_ROOT / "narration_grouping"
DEFAULT_CSV_PATH = _PROJECT_ROOT / "P01" / "participant_P01_narrations.csv"
DEFAULT_OUTPUT_DIR = _PROJECT_ROOT / "outputs" / "food_clips"


@dataclass
class ClipInfo:
    """Information about an extracted clip."""
    group_id: int
    query: str
    original_start: float
    original_end: float
    original_duration: float
    clip_start: float
    clip_end: float
    clip_duration: float
    padded: bool
    padding_amount: float
    clip_path: str
    extracted: bool = True
    error: Optional[str] = None


def get_video_path(video_dir: Path, video_id: str) -> Optional[Path]:
    """Get path to source video file."""
    for ext in ['.mp4', '.MP4', '.mkv', '.avi']:
        video_path = video_dir / f"{video_id}{ext}"
        if video_path.exists():
            return video_path
    return None


def get_video_duration(video_path: Path) -> Optional[float]:
    """Get video duration using ffprobe."""
    cmd = [
        'ffprobe',
        '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        str(video_path)
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            return float(result.stdout.strip())
    except:
        pass
    return None


def calculate_padded_times(
    original_start: float,
    original_end: float,
    min_duration: float,
    video_duration: Optional[float] = None
) -> Tuple[float, float, float, bool]:
    """
    Calculate padded start/end times to ensure minimum clip duration.

    Args:
        original_start: Original start time in seconds
        original_end: Original end time in seconds
        min_duration: Minimum clip duration in seconds
        video_duration: Total video duration (to avoid exceeding)

    Returns:
        Tuple of (clip_start, clip_end, padding_amount, was_padded)
    """
    original_duration = original_end - original_start

    if original_duration >= min_duration:
        return original_start, original_end, 0.0, False

    # Calculate padding needed on each side
    padding_needed = min_duration - original_duration
    padding_per_side = padding_needed / 2

    # Apply padding
    clip_start = original_start - padding_per_side
    clip_end = original_end + padding_per_side

    # Ensure start is not negative
    if clip_start < 0:
        # Shift the window forward
        clip_end += abs(clip_start)
        clip_start = 0

    # Ensure end doesn't exceed video duration
    if video_duration is not None and clip_end > video_duration:
        # Shift the window backward
        overflow = clip_end - video_duration
        clip_start = max(0, clip_start - overflow)
        clip_end = video_duration

    return clip_start, clip_end, padding_per_side, True


def extract_clip(
    video_path: Path,
    output_path: Path,
    start_time: float,
    end_time: float,
    fps: int = 2
) -> bool:
    """
    Extract a video clip using ffmpeg.

    Args:
        video_path: Path to source video
        output_path: Path to save clip
        start_time: Start time in seconds
        end_time: End time in seconds
        fps: Output frame rate (lower = smaller file)

    Returns:
        True if successful
    """
    duration = end_time - start_time

    cmd = [
        'ffmpeg',
        '-y',  # Overwrite
        '-ss', str(start_time),  # Seek to start
        '-i', str(video_path),
        '-t', str(duration),  # Duration
        '-r', str(fps),  # Frame rate
        '-c:v', 'libx264',  # Video codec
        '-preset', 'fast',
        '-crf', '28',  # Quality (higher = smaller file)
        '-an',  # No audio
        '-movflags', '+faststart',
        str(output_path)
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False


def process_video(
    video_id: str,
    grouping_dir: Path,
    csv_path: Path,
    video_dir: Path,
    output_dir: Path,
    min_duration: float = 10.0,
    fps: int = 2,
    dry_run: bool = False
) -> Dict:
    """
    Extract clips for all semantic groups in a video.

    Args:
        video_id: Video ID to process
        grouping_dir: Path to narration_grouping directory
        csv_path: Path to narrations CSV
        video_dir: Path to source videos
        output_dir: Output directory for clips
        min_duration: Minimum clip duration (pad if shorter)
        fps: Output frame rate
        dry_run: If True, don't actually extract clips

    Returns:
        Dict with manifest information
    """
    print(f"\n[{video_id}] Processing...")

    # Load semantic groups
    groups = load_and_enrich_video_groups(grouping_dir, csv_path, video_id)
    if groups is None:
        print(f"  SKIP: No semantic groupings found")
        return {"video_id": video_id, "error": "no_groupings", "clips": []}

    print(f"  Semantic groups: {len(groups)}")

    # Get source video
    video_path = get_video_path(video_dir, video_id)
    if video_path is None:
        print(f"  ERROR: Video not found in {video_dir}")
        return {"video_id": video_id, "error": "video_not_found", "clips": []}

    print(f"  Source: {video_path}")

    # Get video duration for bounds checking
    video_duration = get_video_duration(video_path)
    if video_duration:
        print(f"  Video duration: {video_duration:.1f}s")

    # Create output directory
    clips_dir = output_dir / video_id
    if not dry_run:
        clips_dir.mkdir(parents=True, exist_ok=True)

    # Process each group
    clips_info: List[ClipInfo] = []
    success = 0
    failed = 0
    padded_count = 0

    for group in groups:
        # Calculate padded times
        clip_start, clip_end, padding, was_padded = calculate_padded_times(
            group.start_time,
            group.end_time,
            min_duration,
            video_duration
        )

        if was_padded:
            padded_count += 1

        clip_name = f"group_{group.group_id:03d}.mp4"
        clip_path = clips_dir / clip_name

        # Create clip info
        clip_info = ClipInfo(
            group_id=group.group_id,
            query=group.query,
            original_start=group.start_time,
            original_end=group.end_time,
            original_duration=group.duration,
            clip_start=round(clip_start, 2),
            clip_end=round(clip_end, 2),
            clip_duration=round(clip_end - clip_start, 2),
            padded=was_padded,
            padding_amount=round(padding, 2),
            clip_path=clip_name,
        )

        # Log
        pad_indicator = f"[PAD +{padding:.1f}s]" if was_padded else ""
        print(f"  G{group.group_id:02d}: {group.start_time:.1f}s-{group.end_time:.1f}s "
              f"-> {clip_start:.1f}s-{clip_end:.1f}s ({clip_end-clip_start:.1f}s) "
              f"{pad_indicator}", end=" ")

        if dry_run:
            print("[DRY-RUN]")
            clip_info.extracted = False
            clip_info.error = "dry_run"
        else:
            # Extract clip
            if extract_clip(video_path, clip_path, clip_start, clip_end, fps):
                success += 1
                print("OK")
            else:
                failed += 1
                clip_info.extracted = False
                clip_info.error = "extraction_failed"
                print("FAIL")

        clips_info.append(clip_info)

    # Summary
    print(f"\n  Summary: {success} extracted, {failed} failed, {padded_count} padded")

    # Build manifest
    manifest = {
        "video_id": video_id,
        "source_video": str(video_path),
        "total_groups": len(groups),
        "clips_extracted": success,
        "clips_failed": failed,
        "clips_padded": padded_count,
        "fps": fps,
        "min_duration": min_duration,
        "clips": [asdict(c) for c in clips_info]
    }

    # Save manifest
    if not dry_run:
        manifest_file = clips_dir / "manifest.json"
        with open(manifest_file, 'w') as f:
            json.dump(manifest, f, indent=2)
        print(f"  Manifest saved: {manifest_file}")

    return manifest


def main():
    parser = argparse.ArgumentParser(
        description="Extract video clips for semantic groups",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process single video
    python 02_extract_semantic_clips.py --video-id P01-20240202-110250

    # Process all videos with semantic groupings
    python 02_extract_semantic_clips.py --all-videos

    # Dry run (show what would be extracted)
    python 02_extract_semantic_clips.py --video-id P01-20240202-110250 --dry-run

    # Custom minimum duration
    python 02_extract_semantic_clips.py --video-id P01-20240202-110250 --min-duration 15
        """
    )

    parser.add_argument(
        '--video-id',
        type=str,
        default=None,
        help='Single video ID to process'
    )
    parser.add_argument(
        '--video-ids',
        nargs='+',
        default=None,
        help='Multiple video IDs to process'
    )
    parser.add_argument(
        '--all-videos',
        action='store_true',
        help='Process all videos with semantic groupings'
    )
    parser.add_argument(
        '--grouping-dir',
        type=Path,
        default=DEFAULT_GROUPING_DIR,
        help='Directory containing semantic grouping JSON files'
    )
    parser.add_argument(
        '--csv',
        type=Path,
        default=DEFAULT_CSV_PATH,
        help='Path to narrations CSV file'
    )
    parser.add_argument(
        '--video-dir',
        type=Path,
        default=DEFAULT_VIDEO_DIR,
        help='Directory containing source videos'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help='Output directory for clips'
    )
    parser.add_argument(
        '--fps',
        type=int,
        default=2,
        help='Output frame rate for clips (default: 2)'
    )
    parser.add_argument(
        '--min-duration',
        type=float,
        default=10.0,
        help='Minimum clip duration in seconds (default: 10.0)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be extracted without running ffmpeg'
    )

    args = parser.parse_args()

    print("=" * 70)
    print("SEMANTIC GROUP CLIP EXTRACTION")
    print("=" * 70)
    print(f"Grouping dir: {args.grouping_dir}")
    print(f"Video dir:    {args.video_dir}")
    print(f"Output dir:   {args.output_dir}")
    print(f"FPS:          {args.fps}")
    print(f"Min duration: {args.min_duration}s")
    if args.dry_run:
        print("MODE:         DRY-RUN (no clips will be extracted)")

    # Determine videos to process
    if args.video_id:
        video_ids = [args.video_id]
    elif args.video_ids:
        video_ids = args.video_ids
    elif args.all_videos:
        video_ids = sorted(find_videos_with_groupings(args.grouping_dir))
    else:
        print("\nERROR: Specify --video-id, --video-ids, or --all-videos")
        return

    print(f"\nVideos to process: {len(video_ids)}")

    # Create output directory
    if not args.dry_run:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    # Process each video
    results = []
    for video_id in video_ids:
        result = process_video(
            video_id=video_id,
            grouping_dir=args.grouping_dir,
            csv_path=args.csv,
            video_dir=args.video_dir,
            output_dir=args.output_dir,
            min_duration=args.min_duration,
            fps=args.fps,
            dry_run=args.dry_run
        )
        results.append(result)

    # Summary
    print("\n" + "=" * 70)
    print("EXTRACTION COMPLETE")
    print("=" * 70)

    total_groups = sum(r.get('total_groups', 0) for r in results)
    total_extracted = sum(r.get('clips_extracted', 0) for r in results)
    total_padded = sum(r.get('clips_padded', 0) for r in results)
    total_failed = sum(r.get('clips_failed', 0) for r in results)

    print(f"Videos processed: {len(results)}")
    print(f"Total groups:     {total_groups}")
    print(f"Clips extracted:  {total_extracted}")
    print(f"Clips padded:     {total_padded} ({100*total_padded/total_groups:.1f}% of groups)" if total_groups > 0 else "")
    print(f"Clips failed:     {total_failed}")
    print(f"Output:           {args.output_dir}")


if __name__ == '__main__':
    main()

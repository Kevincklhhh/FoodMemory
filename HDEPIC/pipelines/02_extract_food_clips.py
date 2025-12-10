#!/usr/bin/env python3
"""
Extract Video Clips for Food Blocks

This script extracts video clips corresponding to each food block
for use with VLM processing.

Usage:
    python 02_extract_food_clips.py --video-id P01-20240202-110250
    python 02_extract_food_clips.py --all-videos
"""

import json
import argparse
import subprocess
from pathlib import Path
from typing import List, Dict


# Default paths (relative to this script's location in pipelines/)
_SCRIPT_DIR = Path(__file__).parent
_PROJECT_ROOT = _SCRIPT_DIR.parent

VIDEO_DIR = _PROJECT_ROOT / "P01"


def get_video_path(video_id: str) -> Path:
    """Get path to source video file"""
    # Try different possible extensions
    for ext in ['.mp4', '.MP4', '.mkv', '.avi']:
        video_path = VIDEO_DIR / f"{video_id}{ext}"
        if video_path.exists():
            return video_path
    return VIDEO_DIR / f"{video_id}.mp4"


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
            timeout=60
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"    Timeout extracting clip")
        return False
    except Exception as e:
        print(f"    Error: {e}")
        return False


def load_food_blocks(blocks_dir: Path, video_id: str) -> List[Dict]:
    """Load food blocks for a video"""
    blocks_file = blocks_dir / f"{video_id}_food_blocks.json"
    if not blocks_file.exists():
        return []

    with open(blocks_file, 'r') as f:
        data = json.load(f)

    blocks = data.get('blocks', [])
    # Filter to food blocks only
    food_blocks = [b for b in blocks if b.get('has_food_action', False)]
    return sorted(food_blocks, key=lambda x: x['block_id'])


def process_video(
    video_id: str,
    blocks_dir: Path,
    output_dir: Path,
    fps: int = 2
) -> Dict:
    """
    Extract clips for all food blocks in a video.

    Returns:
        Dict with clip_paths mapping block_id -> clip_path
    """
    print(f"\n[{video_id}] Processing...")

    # Get source video
    video_path = get_video_path(video_id)
    if not video_path.exists():
        print(f"  ERROR: Video not found: {video_path}")
        return {"video_id": video_id, "error": "video_not_found", "clips": {}}

    print(f"  Source: {video_path}")

    # Load food blocks
    food_blocks = load_food_blocks(blocks_dir, video_id)
    if not food_blocks:
        print(f"  No food blocks found")
        return {"video_id": video_id, "error": "no_food_blocks", "clips": {}}

    print(f"  Food blocks: {len(food_blocks)}")

    # Create output directory
    clips_dir = output_dir / video_id
    clips_dir.mkdir(parents=True, exist_ok=True)

    # Extract clips
    clips = {}
    success = 0
    failed = 0

    for block in food_blocks:
        block_id = block['block_id']
        start_time = block['block_start_time']
        end_time = block['block_end_time']

        clip_name = f"block_{block_id:03d}.mp4"
        clip_path = clips_dir / clip_name

        print(f"  Block {block_id}: {start_time:.1f}s - {end_time:.1f}s -> {clip_name}", end=" ")

        if extract_clip(video_path, clip_path, start_time, end_time, fps):
            clips[block_id] = str(clip_path)
            success += 1
            print("✓")
        else:
            failed += 1
            print("✗")

    print(f"  Done: {success} clips extracted, {failed} failed")

    # Save manifest
    manifest = {
        "video_id": video_id,
        "source_video": str(video_path),
        "total_blocks": len(food_blocks),
        "clips_extracted": success,
        "fps": fps,
        "clips": clips
    }

    manifest_file = clips_dir / "manifest.json"
    with open(manifest_file, 'w') as f:
        json.dump(manifest, f, indent=2)

    return manifest


def main():
    parser = argparse.ArgumentParser(description="Extract video clips for food blocks")
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
        help='Process all videos with food blocks'
    )
    parser.add_argument(
        '--blocks-dir',
        type=Path,
        default=Path('../outputs/food_classification'),
        help='Directory containing food_blocks JSON files'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('../outputs/food_clips'),
        help='Output directory for clips'
    )
    parser.add_argument(
        '--fps',
        type=int,
        default=2,
        help='Output frame rate for clips (default: 2)'
    )

    args = parser.parse_args()

    print("=" * 70)
    print("FOOD BLOCK CLIP EXTRACTION")
    print("=" * 70)
    print(f"Blocks dir: {args.blocks_dir}")
    print(f"Output dir: {args.output_dir}")
    print(f"FPS: {args.fps}")

    # Determine videos to process
    if args.video_id:
        video_ids = [args.video_id]
    elif args.video_ids:
        video_ids = args.video_ids
    elif args.all_videos:
        # Find all videos with food blocks
        block_files = sorted(args.blocks_dir.glob("*_food_blocks.json"))
        video_ids = [f.stem.replace("_food_blocks", "") for f in block_files]
    else:
        print("\nERROR: Specify --video-id, --video-ids, or --all-videos")
        return

    print(f"\nVideos to process: {len(video_ids)}")

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Process each video
    results = []
    for video_id in video_ids:
        result = process_video(
            video_id,
            args.blocks_dir,
            args.output_dir,
            args.fps
        )
        results.append(result)

    # Summary
    print("\n" + "=" * 70)
    print("EXTRACTION COMPLETE")
    print("=" * 70)

    total_clips = sum(r.get('clips_extracted', 0) for r in results)
    print(f"Total videos: {len(results)}")
    print(f"Total clips: {total_clips}")
    print(f"Output: {args.output_dir}")


if __name__ == '__main__':
    main()

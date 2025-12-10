#!/usr/bin/env python3
"""
Gemini State Inference Tool

Prepare video chunks and prompts for manual Gemini web processing to infer state changes.

NEW: Uses semantic grouping from filter response instead of fixed-duration chunks.
Groups are combined using a sliding window algorithm (max 600s span).

Usage:
    python gemini_state_inference.py --video-id P01-20240202-110250
    python gemini_state_inference.py --video-id P01-20240202-110250 --max-duration 600

Output:
    gemini_outputs/state_inference/{video_id}/
    ├── chunk_000/
    │   ├── video.mp4      # 1 fps encoded video segment
    │   ├── prompt.txt     # System + user prompt for Gemini
    │   └── response.txt   # Empty file for user to paste Gemini output
    ├── chunk_001/
    │   └── ...
    └── manifest.json      # Chunk metadata for reconstruction

Workflow:
    1. Run this script to prepare chunks
    2. For each chunk:
       a. Copy SYSTEM PROMPT from prompt.txt to Gemini
       b. Upload video.mp4 to Gemini
       c. Copy USER PROMPT from prompt.txt to Gemini
       d. Paste Gemini's response into response.txt
"""

import csv
import json
import argparse
import subprocess
from pathlib import Path
from typing import List, Dict, Optional

# Default paths (relative to this script's location in gemini_pipelines/)
_SCRIPT_DIR = Path(__file__).parent
_PROJECT_ROOT = _SCRIPT_DIR.parent

DEFAULT_CSV_PATH = _PROJECT_ROOT / "P01" / "participant_P01_narrations.csv"
DEFAULT_VIDEO_DIR = _PROJECT_ROOT / "P01"
DEFAULT_FILTER_DIR = _PROJECT_ROOT / "gemini_outputs" / "gemini_input"
DEFAULT_OUTPUT_DIR = _PROJECT_ROOT / "gemini_outputs" / "state_inference"
PROMPT_FILE = _SCRIPT_DIR / "gemini_state_inference.md"

# Max duration constraint for sliding window algorithm (in seconds)
DEFAULT_MAX_DURATION = 600.0


def get_video_duration(video_path: Path) -> float:
    """Get video duration in seconds using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return float(result.stdout.strip())


def extract_video_chunk(
    video_path: Path,
    start_time: float,
    end_time: float,
    output_path: Path,
    fps: int = 1
) -> bool:
    """Extract video segment at specified FPS using ffmpeg."""
    duration = end_time - start_time

    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start_time),
        "-i", str(video_path),
        "-t", str(duration),
        "-vf", f"fps={fps}",
        "-c:v", "libx264",
        "-crf", "28",
        "-preset", "fast",
        "-an",  # No audio
        "-movflags", "+faststart",
        str(output_path)
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    return result.returncode == 0


def load_narrations_for_video(csv_path: Path, video_id: str) -> Dict[str, Dict]:
    """Load all narrations for a specific video, indexed by narration ID."""
    narrations = {}

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['video_id'] == video_id:
                narr_id = row['unique_narration_id']
                narrations[narr_id] = {
                    'unique_narration_id': narr_id,
                    'narration': row['narration'].strip(),
                    'start_timestamp': float(row['start_timestamp']),
                    'end_timestamp': float(row['end_timestamp']),
                }

    return narrations


def parse_line_number_to_narration_id(video_id: str, line_number: int) -> str:
    """Convert line number to narration ID format."""
    return f"{video_id}-{line_number}"


def load_semantic_groups(filter_dir: Path, video_id: str) -> Optional[List[Dict]]:
    """
    Load semantic groups from filter response.

    Expected format:
    [
        {
            "label": "Prepare Coffee and Unpack Oranges",
            "relevant_lines": [3, 6, 7, 8, ...]
        },
        ...
    ]
    """
    response_file = filter_dir / f"filter_{video_id}_response.txt"

    if not response_file.exists():
        return None

    with open(response_file, 'r', encoding='utf-8') as f:
        content = f.read().strip()

    if not content:
        return None

    try:
        groups = json.loads(content)
        if isinstance(groups, list) and len(groups) > 0:
            # Validate structure
            if all('label' in g and 'relevant_lines' in g for g in groups):
                return groups
        return None
    except json.JSONDecodeError:
        return None


def resolve_semantic_blocks(
    groups: List[Dict],
    narrations: Dict[str, Dict],
    video_id: str
) -> List[Dict]:
    """
    Convert semantic groups to blocks with resolved timestamps.

    Returns list of blocks, each with:
    - label: semantic label
    - narration_ids: list of narration IDs
    - narrations: list of narration dicts with timestamps
    - start_time: earliest start_timestamp
    - end_time: latest end_timestamp
    """
    blocks = []

    for group in groups:
        label = group['label']
        line_numbers = group['relevant_lines']

        # Resolve narration IDs and look up timestamps
        block_narrations = []
        for line_num in line_numbers:
            narr_id = parse_line_number_to_narration_id(video_id, line_num)
            if narr_id in narrations:
                block_narrations.append(narrations[narr_id])
            else:
                print(f"  WARNING: Narration ID {narr_id} not found in CSV")

        if not block_narrations:
            print(f"  WARNING: No valid narrations for block '{label}'")
            continue

        # Sort by start_timestamp
        block_narrations.sort(key=lambda x: x['start_timestamp'])

        # Calculate block time span
        start_time = min(n['start_timestamp'] for n in block_narrations)
        end_time = max(n['end_timestamp'] for n in block_narrations)

        blocks.append({
            'label': label,
            'narration_ids': [n['unique_narration_id'] for n in block_narrations],
            'narrations': block_narrations,
            'start_time': start_time,
            'end_time': end_time,
        })

    # Sort blocks by start time
    blocks.sort(key=lambda x: x['start_time'])
    return blocks


def apply_sliding_window(
    blocks: List[Dict],
    max_duration: float
) -> List[Dict]:
    """
    Apply sliding window algorithm to combine semantic blocks.

    Algorithm:
    1. Sort blocks by start time (already done)
    2. Initialize window with first block
    3. For each subsequent block:
       - Calculate projected span: (next_block_end - current_window_start)
       - If projected_span <= max_duration: add to current window
       - Otherwise: close current window, start new window

    Returns list of chunks, each containing multiple semantic blocks.
    """
    if not blocks:
        return []

    chunks = []
    current_chunk = {
        'blocks': [blocks[0]],
        'labels': [blocks[0]['label']],
        'start_time': blocks[0]['start_time'],
        'end_time': blocks[0]['end_time'],
        'narrations': list(blocks[0]['narrations']),
    }

    for block in blocks[1:]:
        projected_span = block['end_time'] - current_chunk['start_time']

        if projected_span <= max_duration:
            # Add block to current chunk
            current_chunk['blocks'].append(block)
            current_chunk['labels'].append(block['label'])
            current_chunk['end_time'] = max(current_chunk['end_time'], block['end_time'])
            current_chunk['narrations'].extend(block['narrations'])
        else:
            # Close current chunk, start new one
            chunks.append(current_chunk)
            current_chunk = {
                'blocks': [block],
                'labels': [block['label']],
                'start_time': block['start_time'],
                'end_time': block['end_time'],
                'narrations': list(block['narrations']),
            }

    # Don't forget the last chunk
    chunks.append(current_chunk)

    # Sort narrations within each chunk by timestamp
    for chunk in chunks:
        chunk['narrations'].sort(key=lambda x: x['start_timestamp'])

    return chunks


def build_chunk_prompt(chunk: Dict) -> str:
    """Build prompt with normalized timestamps for a chunk."""
    chunk_start = chunk['start_time']

    lines = [
        f"NARRATION LOG (local video timestamp):",
        f"Semantic Groups: {', '.join(chunk['labels'])}",
        ""
    ]

    for narr in chunk['narrations']:
        local_ts = narr['start_timestamp'] - chunk_start
        narr_id = narr['unique_narration_id']
        # Format: [local_ts] (narration_id) narration_text
        lines.append(f"[{local_ts:.1f}s] ({narr_id}) {narr['narration']}")

    lines.extend([
        "",
        "Please analyze each narration against the video and return JSON as specified in the system prompt."
    ])

    return "\n".join(lines)


def process_video(
    video_id: str,
    max_duration: float,
    fps: int,
    csv_path: Path,
    video_dir: Path,
    filter_dir: Path,
    output_dir: Path,
    verbose: bool = False
):
    """Process a video: create chunks based on semantic groups and prompts."""
    # Setup paths
    video_path = video_dir / f"{video_id}.mp4"
    output_video_dir = output_dir / video_id

    if not video_path.exists():
        print(f"ERROR: Video not found: {video_path}")
        return

    # Get video duration
    print(f"Getting video duration...")
    video_duration = get_video_duration(video_path)
    print(f"  Duration: {video_duration:.1f}s")

    # Load all narrations
    print(f"Loading narrations...")
    all_narrations = load_narrations_for_video(csv_path, video_id)
    print(f"  Total narrations: {len(all_narrations)}")

    # Load semantic groups from filter response
    print(f"Loading semantic groups...")
    semantic_groups = load_semantic_groups(filter_dir, video_id)
    if not semantic_groups:
        print(f"ERROR: No semantic groups found in filter response for {video_id}")
        print(f"  Expected file: {filter_dir / f'filter_{video_id}_response.txt'}")
        return

    print(f"  Found {len(semantic_groups)} semantic groups")

    # Resolve semantic blocks with timestamps
    print(f"Resolving semantic blocks...")
    semantic_blocks = resolve_semantic_blocks(semantic_groups, all_narrations, video_id)
    print(f"  Resolved {len(semantic_blocks)} blocks with valid narrations")

    if verbose:
        for i, block in enumerate(semantic_blocks):
            print(f"    Block {i}: '{block['label']}' ({block['start_time']:.1f}s - {block['end_time']:.1f}s)")

    # Apply sliding window algorithm
    print(f"\nApplying sliding window (max_duration={max_duration}s)...")
    chunks = apply_sliding_window(semantic_blocks, max_duration)
    print(f"  Generated {len(chunks)} chunks")

    # Create output directory
    output_video_dir.mkdir(parents=True, exist_ok=True)

    # Load system prompt
    system_prompt = ""
    if PROMPT_FILE.exists():
        with open(PROMPT_FILE, 'r') as f:
            system_prompt = f.read()

    # Process each chunk
    manifest = {
        'video_id': video_id,
        'video_duration': video_duration,
        'max_duration': max_duration,
        'fps': fps,
        'total_semantic_groups': len(semantic_groups),
        'total_chunks': len(chunks),
        'chunks': []
    }

    for chunk_idx, chunk in enumerate(chunks):
        span = chunk['end_time'] - chunk['start_time']
        print(f"\n  Chunk {chunk_idx}: {chunk['start_time']:.1f}s - {chunk['end_time']:.1f}s (span: {span:.1f}s)")
        print(f"    Groups: {chunk['labels']}")
        print(f"    Narrations: {len(chunk['narrations'])}")

        # Create chunk directory
        chunk_dir = output_video_dir / f"chunk_{chunk_idx:03d}"
        chunk_dir.mkdir(parents=True, exist_ok=True)

        # Extract video chunk
        video_output = chunk_dir / "video.mp4"
        print(f"    Extracting video at {fps} fps...")
        success = extract_video_chunk(
            video_path,
            chunk['start_time'],
            chunk['end_time'],
            video_output,
            fps
        )
        if not success:
            print(f"    ERROR: Failed to extract video")

        # Build and save prompt
        prompt = build_chunk_prompt(chunk)
        prompt_file = chunk_dir / "prompt.txt"
        with open(prompt_file, 'w') as f:
            # Include system prompt at top for reference
            f.write("=== SYSTEM PROMPT ===\n\n")
            f.write(system_prompt)
            f.write("\n\n=== USER PROMPT ===\n\n")
            f.write(prompt)

        # Create empty response file for user to paste Gemini output
        response_file = chunk_dir / "response.txt"
        response_file.touch()

        # Add to manifest
        manifest['chunks'].append({
            'chunk_idx': chunk_idx,
            'global_start': chunk['start_time'],
            'global_end': chunk['end_time'],
            'span': span,
            'semantic_labels': chunk['labels'],
            'narration_count': len(chunk['narrations']),
            'narration_ids': [n['unique_narration_id'] for n in chunk['narrations']],
            'video_path': str(video_output.relative_to(output_video_dir)),
            'prompt_path': str(prompt_file.relative_to(output_video_dir)),
            'response_path': str(response_file.relative_to(output_video_dir))
        })

        if verbose:
            print(f"    Saved: {video_output.name}, {prompt_file.name}, {response_file.name}")

    # Save manifest
    manifest_file = output_video_dir / "manifest.json"
    with open(manifest_file, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"\n{'='*60}")
    print(f"COMPLETE: {video_id}")
    print(f"{'='*60}")
    print(f"Output: {output_video_dir}")
    print(f"Semantic groups: {len(semantic_groups)}")
    print(f"Chunks: {len(chunks)}")
    print(f"Manifest: {manifest_file}")
    print(f"\nFor each chunk:")
    print(f"  1. Open chunk_XXX/prompt.txt")
    print(f"  2. Copy SYSTEM PROMPT to Gemini")
    print(f"  3. Upload chunk_XXX/video.mp4")
    print(f"  4. Copy USER PROMPT to Gemini")
    print(f"  5. Paste Gemini response into chunk_XXX/response.txt")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare video chunks and prompts for Gemini state inference (semantic grouping)"
    )
    parser.add_argument(
        '--video-id',
        required=True,
        help='Video ID to process (e.g., P01-20240202-110250)'
    )
    parser.add_argument(
        '--max-duration',
        type=float,
        default=DEFAULT_MAX_DURATION,
        help=f'Maximum chunk duration in seconds (default: {DEFAULT_MAX_DURATION})'
    )
    parser.add_argument(
        '--fps',
        type=int,
        default=1,
        help='Output video FPS (default: 1)'
    )
    parser.add_argument(
        '--csv',
        type=Path,
        default=DEFAULT_CSV_PATH,
        help='Path to narrations CSV'
    )
    parser.add_argument(
        '--video-dir',
        type=Path,
        default=DEFAULT_VIDEO_DIR,
        help='Directory containing video files'
    )
    parser.add_argument(
        '--filter-dir',
        type=Path,
        default=DEFAULT_FILTER_DIR,
        help='Directory containing filter response files'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help='Output directory'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print verbose output'
    )

    args = parser.parse_args()

    print("="*60)
    print("GEMINI STATE INFERENCE PREPARATION")
    print("(Semantic Grouping Mode)")
    print("="*60)
    print(f"Video:          {args.video_id}")
    print(f"Max duration:   {args.max_duration}s")
    print(f"FPS:            {args.fps}")
    print(f"Output:         {args.output_dir}")

    process_video(
        video_id=args.video_id,
        max_duration=args.max_duration,
        fps=args.fps,
        csv_path=args.csv,
        video_dir=args.video_dir,
        filter_dir=args.filter_dir,
        output_dir=args.output_dir,
        verbose=args.verbose
    )


if __name__ == '__main__':
    main()

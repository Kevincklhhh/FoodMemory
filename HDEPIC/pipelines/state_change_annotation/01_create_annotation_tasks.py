#!/usr/bin/env python3
"""
Food State Change Annotation Benchmark Creator

This script processes HD-EPIC food narrations to create annotation tasks
for labeling food state changes (e.g., preparation, consumption states).

Pipeline:
1. Load food narrations from JSON
2. Merge consecutive narrations into interaction blocks
3. Extract frames and video clips for each block
4. Run grounding models (GroundingSAM + Hands23) to pre-populate masks
5. Package everything into annotation task JSON objects
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Set, Tuple
from collections import defaultdict
import cv2
import subprocess


class InteractionBlock:
    """Represents a merged group of consecutive food narrations"""

    def __init__(self, block_id: int):
        self.block_id = block_id
        self.narrations = []
        self.start_time = None
        self.end_time = None
        self.target_food_nouns = set()

    def add_narration(self, narration: Dict):
        """Add a narration to this block"""
        self.narrations.append(narration)

        # Update time boundaries
        if self.start_time is None or narration['start_timestamp'] < self.start_time:
            self.start_time = narration['start_timestamp']
        if self.end_time is None or narration['end_timestamp'] > self.end_time:
            self.end_time = narration['end_timestamp']

        # Collect target food nouns
        for food_item in narration['food_items']:
            self.target_food_nouns.add(food_item['noun_key'])

    def to_dict(self) -> Dict:
        """Convert block to dictionary representation"""
        return {
            'block_id': self.block_id,
            'block_start_time': self.start_time,
            'block_end_time': self.end_time,
            'narrations': [n['narration'].strip() for n in self.narrations],
            'narration_details': self.narrations,
            'target_food_nouns': sorted(list(self.target_food_nouns))
        }


def load_food_narrations(json_path: str) -> Tuple[str, List[Dict]]:
    """Load food narrations from JSON file

    Returns:
        (video_id, list of narrations)
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Extract video_id and narrations
    video_id = list(data.keys())[0]
    narrations = data[video_id]

    return video_id, narrations


def merge_into_blocks(narrations: List[Dict], max_block_duration: float = 30.0) -> List[InteractionBlock]:
    """Merge consecutive narrations into interaction blocks with maximum duration constraint

    Strategy: Aggregate narrations sequentially until adding the next narration
    would cause the block duration to exceed max_block_duration.

    Block duration = (block.end_time - block.start_time)

    Args:
        narrations: List of food narration objects
        max_block_duration: Maximum block duration in seconds (default: 30.0)

    Returns:
        List of InteractionBlock objects
    """
    if not narrations:
        return []

    # Sort by start time
    sorted_narrations = sorted(narrations, key=lambda x: x['start_timestamp'])

    blocks = []
    current_block = InteractionBlock(block_id=0)
    current_block.add_narration(sorted_narrations[0])

    for narration in sorted_narrations[1:]:
        # Calculate what the block duration would be if we add this narration
        potential_start = current_block.start_time
        potential_end = max(current_block.end_time, narration['end_timestamp'])
        potential_duration = potential_end - potential_start

        if potential_duration <= max_block_duration:
            # Adding this narration keeps us under the limit
            current_block.add_narration(narration)
        else:
            # Adding this would exceed limit - start new block
            blocks.append(current_block)
            current_block = InteractionBlock(block_id=len(blocks))
            current_block.add_narration(narration)

    # Add final block
    blocks.append(current_block)

    return blocks


def extract_frame_from_video(video_path: str, timestamp: float, output_path: str) -> bool:
    """Extract a single frame from video at specified timestamp

    Args:
        video_path: Path to video file
        timestamp: Time in seconds
        output_path: Where to save the extracted frame

    Returns:
        True if successful, False otherwise
    """
    try:
        # Use opencv to extract frame
        cap = cv2.VideoCapture(video_path)

        # Set position to timestamp
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)

        # Read frame
        ret, frame = cap.read()
        cap.release()

        if ret:
            cv2.imwrite(output_path, frame)
            return True
        else:
            print(f"Warning: Could not read frame at {timestamp}s")
            return False

    except Exception as e:
        print(f"Error extracting frame: {e}")
        return False


def extract_clip_from_video(video_path: str, start_time: float, end_time: float,
                            output_path: str) -> bool:
    """Extract a video clip segment

    Clip Selection Strategy:
    - Start: block.start_time - frame_offset (1.0s default)
    - End: block.end_time + frame_offset (1.0s default)
    - This captures temporal context before/after the interaction block
    - Ensures annotators see the full state change sequence

    Args:
        video_path: Path to source video
        start_time: Start time in seconds (already includes offset)
        end_time: End time in seconds (already includes offset)
        output_path: Where to save the clip

    Returns:
        True if successful, False otherwise
    """
    try:
        duration = end_time - start_time

        # Use ffmpeg to extract clip
        cmd = [
            'ffmpeg', '-y',  # Overwrite output file
            '-ss', str(start_time),  # Start time
            '-i', video_path,  # Input file
            '-t', str(duration),  # Duration
            '-c:v', 'libx264',  # Video codec
            '-crf', '23',  # Quality
            '-preset', 'fast',  # Encoding speed
            output_path
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        return result.returncode == 0

    except Exception as e:
        print(f"Error extracting clip: {e}")
        return False


def create_annotation_task(
    block: InteractionBlock,
    video_id: str,
    video_path: str,
    output_dir: Path,
    frame_offset: float = 1.0
) -> Dict:
    """Create an annotation task object for a single interaction block

    Frame Extraction Strategy:
    - Only extracts frames for EACH NARRATION within the block
    - Before frame: narration.start_time - frame_offset
    - After frame: narration.end_time + frame_offset
    - No "boundary" frames for the entire block (removed to reduce redundancy)
    - This ensures each narration's state change is captured individually

    Clip Extraction:
    - One video clip per block
    - Spans: (block.start_time - offset) to (block.end_time + offset)
    - Provides full temporal context for all narrations in the block

    Args:
        block: InteractionBlock to process
        video_id: Video identifier
        video_path: Path to source video file
        output_dir: Base output directory for assets
        frame_offset: Seconds to offset before/after timestamps (default: 1.0s)

    Returns:
        Annotation task dictionary
    """
    task_id = f"{video_id}_block_{block.block_id:03d}"

    # Create asset directory for this block
    asset_dir = output_dir / "assets" / video_id / f"block_{block.block_id:03d}"
    asset_dir.mkdir(parents=True, exist_ok=True)

    # Define timestamps
    before_timestamp = max(0, block.start_time - frame_offset)
    after_timestamp = block.end_time + frame_offset

    # Storage for extracted assets
    assets = {
        'clip_path': None,
        'before_frames': [],
        'after_frames': []
    }

    # Extract video clip for the entire block (with temporal context)
    clip_path = asset_dir / f"{task_id}_clip.mp4"
    if extract_clip_from_video(video_path, before_timestamp, after_timestamp, str(clip_path)):
        assets['clip_path'] = str(clip_path.relative_to(output_dir))

    # Extract frames for each narration in the block
    # (boundary frames removed - each narration captures its own state change)
    for i, narration in enumerate(block.narrations):
        narr_before_ts = max(0, narration['start_timestamp'] - frame_offset)
        narr_after_ts = narration['end_timestamp'] + frame_offset

        # Before frame for this narration
        narr_before_frame = asset_dir / f"{task_id}_narr{i:02d}_before.jpg"
        if extract_frame_from_video(video_path, narr_before_ts, str(narr_before_frame)):
            assets['before_frames'].append({
                'path': str(narr_before_frame.relative_to(output_dir)),
                'timestamp': narr_before_ts,
                'type': 'narration',
                'narration_index': i,
                'narration_id': narration['narration_id']
            })

        # After frame for this narration
        narr_after_frame = asset_dir / f"{task_id}_narr{i:02d}_after.jpg"
        if extract_frame_from_video(video_path, narr_after_ts, str(narr_after_frame)):
            assets['after_frames'].append({
                'path': str(narr_after_frame.relative_to(output_dir)),
                'timestamp': narr_after_ts,
                'type': 'narration',
                'narration_index': i,
                'narration_id': narration['narration_id']
            })

    # Initialize empty ground truth state table
    # Note: VLM will populate instances with vlm_state_after, annotators will populate annotator_state_after
    ground_truth_state_table = {}
    for noun in block.target_food_nouns:
        ground_truth_state_table[noun] = {
            'instances': {}  # Will store instance_id -> state mappings
        }

    # Create task object
    task = {
        'task_id': task_id,
        'video_id': video_id,
        'block_start_time': block.start_time,
        'block_end_time': block.end_time,
        'narrations_in_block': [n['narration'].strip() for n in block.narrations],
        'narration_details': block.narrations,
        'target_food_nouns': sorted(list(block.target_food_nouns)),
        'assets': assets,
        'auto_grounding_data': {},  # To be filled by grounding models
        'ground_truth_state_table': ground_truth_state_table
    }

    return task


def print_block_summary(blocks: List[InteractionBlock]):
    """Print summary statistics about merged blocks"""
    print("\n" + "=" * 80)
    print("INTERACTION BLOCKS SUMMARY")
    print("=" * 80)
    print(f"Total blocks created: {len(blocks)}")
    print(f"Average narrations per block: {sum(len(b.narrations) for b in blocks) / len(blocks):.1f}")

    print("\nBlock Details:")
    for block in blocks:
        duration = block.end_time - block.start_time
        print(f"\nBlock {block.block_id}:")
        print(f"  Time: {block.start_time:.2f}s - {block.end_time:.2f}s (duration: {duration:.2f}s)")
        print(f"  Narrations: {len(block.narrations)}")
        print(f"  Food nouns: {', '.join(sorted(block.target_food_nouns))}")


def main():
    """Main pipeline execution"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Create food state change annotation tasks from HD-EPIC data"
    )
    parser.add_argument(
        '--input',
        default='../../outputs/food_analysis/per_video_extractions/P01-20240203-121517_food_items.json',
        help='Path to food items JSON file'
    )
    parser.add_argument(
        '--video-path',
        default='../../data/HD-EPIC/Videos/P01/P01-20240203-121517.mp4',
        help='Path to source video file'
    )
    parser.add_argument(
        '--output-dir',
        default='../../outputs/state_change_annotation',
        help='Output directory for annotation tasks'
    )
    parser.add_argument(
        '--max-block-duration',
        type=float,
        default=30.0,
        help='Maximum block duration (seconds) for VLM context window'
    )
    parser.add_argument(
        '--frame-offset',
        type=float,
        default=1.0,
        help='Offset (seconds) for before/after timestamps'
    )

    args = parser.parse_args()

    # Setup paths
    input_path = Path(args.input)
    video_path = Path(args.video_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("FOOD STATE CHANGE ANNOTATION TASK CREATOR")
    print("=" * 80)
    print(f"Input: {input_path}")
    print(f"Video: {video_path}")
    print(f"Output: {output_dir}")
    print(f"Max block duration: {args.max_block_duration}s")
    print(f"Frame offset: {args.frame_offset}s")

    # Step 1: Load food narrations
    print("\n[Step 1] Loading food narrations...")
    video_id, narrations = load_food_narrations(str(input_path))
    print(f"✓ Loaded {len(narrations)} food narrations for video {video_id}")

    # Step 2: Merge into interaction blocks
    print("\n[Step 2] Merging into interaction blocks...")
    blocks = merge_into_blocks(narrations, args.max_block_duration)
    print(f"✓ Created {len(blocks)} interaction blocks")
    print_block_summary(blocks)

    # Step 3: Create annotation tasks
    print("\n[Step 3] Creating annotation tasks...")
    annotation_tasks = []

    for block in blocks:
        print(f"\nProcessing block {block.block_id}...")
        task = create_annotation_task(
            block,
            video_id,
            str(video_path),
            output_dir,
            args.frame_offset
        )
        annotation_tasks.append(task)
        print(f"  ✓ Extracted {len(task['assets']['before_frames'])} before frames")
        print(f"  ✓ Extracted {len(task['assets']['after_frames'])} after frames")
        if task['assets']['clip_path']:
            print(f"  ✓ Extracted video clip")

    # Step 4: Save output
    output_file = output_dir / f"{video_id}_annotation_tasks.json"
    print(f"\n[Step 4] Saving annotation tasks to {output_file}...")

    with open(output_file, 'w') as f:
        json.dump(annotation_tasks, f, indent=2)

    print(f"✓ Saved {len(annotation_tasks)} annotation tasks")

    # Print summary statistics
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE - SUMMARY")
    print("=" * 80)
    print(f"Total annotation tasks: {len(annotation_tasks)}")
    print(f"Total frames extracted: {sum(len(t['assets']['before_frames']) + len(t['assets']['after_frames']) for t in annotation_tasks)}")
    print(f"Total clips extracted: {sum(1 for t in annotation_tasks if t['assets']['clip_path'])}")
    print(f"Unique food nouns: {len(set(noun for task in annotation_tasks for noun in task['target_food_nouns']))}")
    print(f"\nOutput saved to: {output_file}")
    print("=" * 80)


if __name__ == '__main__':
    main()

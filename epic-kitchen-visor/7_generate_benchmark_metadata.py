#!/usr/bin/env python3
"""
Generate benchmark metadata for food instance retrieval.

This script:
1. Scans frames in retrieve_benchmarks/{food_class}/
2. Groups frames by instance ID based on unique settings (participant, session_type)
3. User can optionally provide custom instance ID mapping
4. Outputs JSON metadata with instance IDs assigned
5. A separate script can later assign query/evidence/distractor roles

Usage:
  # Auto-assign instance IDs based on settings
  python3 7_generate_benchmark_metadata.py --food yoghurt

  # Provide custom instance ID mapping
  python3 7_generate_benchmark_metadata.py --food yoghurt \
      --mapping instance_mapping.json
"""

import json
import pickle
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import re


def parse_video_id(video_id: str) -> Tuple[str, str, int]:
    """Parse video ID into participant, session type, and video number.

    Args:
        video_id: Video ID like 'P02_03' or 'P02_112'

    Returns:
        Tuple of (participant_id, session_type, video_num)
        session_type is 'original' (< 100) or 'new_collection' (>= 100)
    """
    parts = video_id.split('_')
    participant_id = parts[0]
    video_num = int(parts[1])

    session_type = 'new_collection' if video_num >= 100 else 'original'

    return participant_id, session_type, video_num


def get_setting_id(video_id: str) -> str:
    """Get unique setting identifier for a video.

    Setting = (participant_id, session_type)
    This represents a unique physical context where food instances are distinct.

    Args:
        video_id: Video ID like 'P02_03'

    Returns:
        Setting ID like 'P02_original' or 'P04_new_collection'
    """
    participant_id, session_type, _ = parse_video_id(video_id)
    return f"{participant_id}_{session_type}"


def load_epic100_data(train_pkl: Path, val_pkl: Path) -> Dict:
    """Load EPIC-100 train and validation pickle files."""
    print("Loading EPIC-100 annotations...")

    with open(train_pkl, 'rb') as f:
        train_df = pickle.load(f)

    with open(val_pkl, 'rb') as f:
        val_df = pickle.load(f)

    # Combine and index by video_id and frame range
    epic100_index = {}

    for df, split in [(train_df, 'train'), (val_df, 'val')]:
        for _, row in df.iterrows():
            video_id = row['video_id']

            if video_id not in epic100_index:
                epic100_index[video_id] = []

            epic100_index[video_id].append({
                'narration_id': row.name if hasattr(row, 'name') else f"{video_id}_{len(epic100_index[video_id])}",
                'narration_text': row['narration'],
                'start_frame': int(row['start_frame']),
                'stop_frame': int(row['stop_frame']),
                'start_timestamp': row['start_timestamp'],
                'stop_timestamp': row['stop_timestamp'],
                'verb': row['verb'],
                'verb_class': int(row['verb_class']),
                'noun': row['noun'],
                'noun_class': int(row['noun_class']),
                'split': split
            })

    print(f"✓ Loaded {len(epic100_index)} videos from EPIC-100")
    return epic100_index


def find_epic100_context(video_id: str, frame_number: int, epic100_index: Dict) -> Optional[Dict]:
    """Find the narration context for a specific frame."""
    if video_id not in epic100_index:
        return None

    # Find narration that contains this frame
    for narration in epic100_index[video_id]:
        if narration['start_frame'] <= frame_number <= narration['stop_frame']:
            return narration

    # If no exact match, find closest narration
    closest = min(
        epic100_index[video_id],
        key=lambda n: min(abs(n['start_frame'] - frame_number), abs(n['stop_frame'] - frame_number))
    )
    return closest


def load_food_image_index(index_path: Path) -> Dict:
    """Load food image index."""
    print("Loading food image index...")
    with open(index_path, 'r') as f:
        index = json.load(f)
    print(f"✓ Loaded index with {len(index['by_food_class'])} food classes")
    return index


def parse_frame_filename(filename: str) -> tuple:
    """Parse frame filename to extract video_id and frame_number.

    Example: P02_12_frame_0000004438.jpg -> ('P02_12', 4438)
    """
    match = re.match(r'(P\d+_\d+)_frame_(\d+)\.jpg', filename)
    if match:
        video_id = match.group(1)
        frame_number = int(match.group(2))
        return video_id, frame_number
    return None, None


def scan_food_frames(food_dir: Path, food_class: str) -> List[Dict]:
    """Scan frames in retrieve_benchmarks/{food_class}/ directory."""
    frames = []

    if not food_dir.exists():
        print(f"Error: {food_dir} does not exist")
        return frames

    for frame_file in sorted(food_dir.glob('*.jpg')):
        video_id, frame_number = parse_frame_filename(frame_file.name)

        if video_id and frame_number:
            frames.append({
                'filename': frame_file.name,
                'video_id': video_id,
                'frame_number': frame_number,
                'participant_id': video_id.split('_')[0]
            })

    print(f"✓ Found {len(frames)} {food_class} frames")
    return frames


def load_instance_mapping(mapping_path: Path) -> Dict[str, str]:
    """Load custom instance ID mapping from JSON file.

    Expected format:
    {
        "P02_12_frame_0000004926.jpg": "instance_001",
        "P04_101_frame_0000011919.jpg": "instance_002",
        ...
    }
    """
    print(f"Loading custom instance mapping from {mapping_path}...")
    with open(mapping_path, 'r') as f:
        mapping = json.load(f)
    print(f"✓ Loaded mapping for {len(mapping)} frames")
    return mapping


def auto_assign_instance_ids(frames: List[Dict]) -> Dict[str, str]:
    """Auto-assign instance IDs based on unique settings.

    Frames from same setting (participant + session_type) get same instance ID.

    Args:
        frames: List of frame info dicts

    Returns:
        Dictionary mapping filename to instance_id
    """
    print("Auto-assigning instance IDs based on settings (participant + session_type)...")

    # Group frames by setting
    setting_groups = defaultdict(list)
    for frame in frames:
        setting_id = get_setting_id(frame['video_id'])
        setting_groups[setting_id].append(frame['filename'])

    # Assign instance IDs
    filename_to_instance = {}
    for instance_id, filenames in sorted(setting_groups.items()):
        for filename in filenames:
            filename_to_instance[filename] = instance_id

    print(f"✓ Created {len(setting_groups)} instance IDs for {len(filename_to_instance)} frames")

    # Print summary
    for instance_id, filenames in sorted(setting_groups.items()):
        print(f"  {instance_id}: {len(filenames)} frames")

    return filename_to_instance


def enrich_frame_metadata(
    frames: List[Dict],
    food_class: str,
    food_index: Dict,
    epic100_index: Dict,
    instance_mapping: Dict[str, str]
) -> List[Dict]:
    """Enrich frame metadata with EPIC-100, food index, and instance ID."""
    print("Enriching frame metadata with instance IDs...")

    enriched = []
    food_class_data = food_index['by_food_class'].get(food_class, [])

    for frame_info in frames:
        video_id = frame_info['video_id']
        frame_number = frame_info['frame_number']
        filename = frame_info['filename']

        # Find matching entry in food index
        food_entry = next(
            (entry for entry in food_class_data
             if entry['video_id'] == video_id and entry['frame_number'] == frame_number),
            None
        )

        if not food_entry:
            print(f"  Warning: No food index entry for {filename}")
            continue

        # Find EPIC-100 context
        epic100_context = find_epic100_context(video_id, frame_number, epic100_index)

        # Get instance ID from mapping
        instance_id = instance_mapping.get(filename)
        if not instance_id:
            print(f"  Warning: No instance ID for {filename}")
            continue

        # Build enriched metadata
        metadata = {
            'frame_id': f"{video_id}_frame_{frame_number:010d}",
            'filename': filename,
            'video_id': video_id,
            'participant_id': frame_info['participant_id'],
            'frame_number': frame_number,
            'instance_id': instance_id,
            'image_path': food_entry['image_path'],
            'visor_object_id': food_entry['object_id'],
            'class_id': food_entry['class_id'],
            'object_name': food_entry['object_name'],
            'epic100_context': epic100_context,
            'segments': food_entry.get('segments')
        }

        enriched.append(metadata)

    print(f"✓ Enriched {len(enriched)} frames with metadata")
    return enriched


def generate_benchmark_metadata(
    food_class: str,
    frames: List[Dict],
    template_path: Path
) -> Dict:
    """Generate benchmark metadata JSON organized by instance ID."""
    print("Generating benchmark metadata...")

    # Load template
    with open(template_path, 'r') as f:
        metadata = json.load(f)

    # Update benchmark info
    metadata['benchmark_info']['food_class_focus'] = food_class
    metadata['benchmark_info']['generated_date'] = "2025-11-12"

    # Group frames by instance ID
    instances = defaultdict(list)
    for frame in frames:
        instances[frame['instance_id']].append(frame)

    # Store all frames organized by instance
    metadata['instances'] = {}
    for instance_id, instance_frames in sorted(instances.items()):
        metadata['instances'][instance_id] = {
            'instance_id': instance_id,
            'frame_count': len(instance_frames),
            'frames': sorted(instance_frames, key=lambda x: (x['video_id'], x['frame_number']))
        }

    # Update statistics
    metadata['statistics'] = {
        'total_frames': len(frames),
        'total_instances': len(instances),
        'food_class': food_class,
        'frames_per_instance': {iid: len(iframes) for iid, iframes in instances.items()},
        'videos_covered': sorted(set(f['video_id'] for f in frames))
    }

    # Clear test_cases (to be assigned by separate script)
    metadata['test_cases'] = []

    print(f"✓ Generated metadata for {len(frames)} frames in {len(instances)} instances")
    return metadata


def main():
    parser = argparse.ArgumentParser(
        description="Generate benchmark metadata with instance IDs for food instance retrieval"
    )
    parser.add_argument(
        '--food',
        required=True,
        help='Food class to generate metadata for (e.g., yoghurt)'
    )
    parser.add_argument(
        '--mapping',
        help='Optional JSON file with custom instance ID mapping (filename -> instance_id)'
    )
    parser.add_argument(
        '--benchmarks-dir',
        default='retrieve_benchmarks',
        help='Directory containing benchmark subdirectories (default: retrieve_benchmarks)'
    )
    parser.add_argument(
        '--epic100-train',
        default='EPIC_100_train.pkl',
        help='Path to EPIC-100 train pickle file'
    )
    parser.add_argument(
        '--epic100-val',
        default='EPIC_100_validation.pkl',
        help='Path to EPIC-100 validation pickle file'
    )
    parser.add_argument(
        '--food-index',
        default='food_image_index.json',
        help='Path to food image index JSON'
    )
    parser.add_argument(
        '--template',
        default='benchmark_metadata_template.json',
        help='Path to benchmark metadata template'
    )
    parser.add_argument(
        '--output',
        help='Output JSON file (default: {food}_benchmark_instances.json)'
    )

    args = parser.parse_args()

    # Set default output path
    if not args.output:
        args.output = f"retrieve_benchmarks/{args.food}_benchmark_instances.json"

    print("=" * 80)
    print(f"GENERATING BENCHMARK METADATA FOR: {args.food}")
    print("=" * 80)
    print()

    # Load data sources
    epic100_index = load_epic100_data(Path(args.epic100_train), Path(args.epic100_val))
    food_index = load_food_image_index(Path(args.food_index))

    # Scan frames
    food_dir = Path(args.benchmarks_dir) / args.food
    frames = scan_food_frames(food_dir, args.food)

    if not frames:
        print(f"Error: No frames found in {food_dir}")
        return

    # Get instance ID mapping
    if args.mapping:
        instance_mapping = load_instance_mapping(Path(args.mapping))
    else:
        print("\nNo custom mapping provided. Auto-assigning instance IDs based on settings...")
        instance_mapping = auto_assign_instance_ids(frames)

    # Enrich metadata with instance IDs
    enriched_frames = enrich_frame_metadata(
        frames,
        args.food,
        food_index,
        epic100_index,
        instance_mapping
    )

    # Generate metadata
    metadata = generate_benchmark_metadata(
        args.food,
        enriched_frames,
        Path(args.template)
    )

    # Save output
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print()
    print("=" * 80)
    print(f"✓ Saved benchmark metadata to: {output_path}")
    print("=" * 80)
    print()
    print("BENCHMARK SUMMARY:")
    print(f"  Total frames: {len(enriched_frames)}")
    print(f"  Total instances: {metadata['statistics']['total_instances']}")
    print(f"  Frames per instance:")
    for instance_id, count in sorted(metadata['statistics']['frames_per_instance'].items()):
        print(f"    {instance_id}: {count} frames")
    print()
    print("NEXT STEP:")
    print("  Use a separate script to randomly assign query/evidence/distractor roles")
    print("=" * 80)


if __name__ == '__main__':
    main()

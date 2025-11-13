#!/usr/bin/env python3
"""
List all objects from P01 videos with their tracking details.

This script extracts and displays:
- Object names and IDs
- Track time segments
- Associated mask IDs
- Mask frame numbers
- Bounding boxes
- Fixture names
"""

import json
import csv
from pathlib import Path


def load_annotations():
    """Load mask_info.json and assoc_info.json"""
    mask_path = Path('hd-epic-annotations/scene-and-object-movements/mask_info.json')
    assoc_path = Path('hd-epic-annotations/scene-and-object-movements/assoc_info.json')

    with open(mask_path, 'r') as f:
        mask_info = json.load(f)

    with open(assoc_path, 'r') as f:
        assoc_info = json.load(f)

    return mask_info, assoc_info


def extract_p01_data(mask_info, assoc_info):
    """Extract all P01 video data"""
    p01_data = []

    # Filter P01 videos
    p01_videos = sorted([vid for vid in assoc_info.keys() if vid.startswith('P01')])

    print(f"Found {len(p01_videos)} P01 videos\n")

    for video_id in p01_videos:
        video_objects = assoc_info[video_id]
        video_masks = mask_info.get(video_id, {})

        for object_id, object_data in video_objects.items():
            object_name = object_data['name']

            for track_idx, track in enumerate(object_data['tracks']):
                track_id = track['track_id']
                time_start, time_end = track['time_segment']
                mask_ids = track['masks']

                # Get mask details for each mask in this track
                for mask_id in mask_ids:
                    if mask_id in video_masks:
                        mask = video_masks[mask_id]

                        # Handle potential None values
                        bbox = mask.get('bbox', [None, None, None, None])
                        if bbox is None:
                            bbox = [None, None, None, None]

                        position = mask.get('3d_location', [None, None, None])
                        if position is None:
                            position = [None, None, None]

                        p01_data.append({
                            'video_id': video_id,
                            'object_id': object_id,
                            'object_name': object_name,
                            'track_id': track_id,
                            'track_index': track_idx,
                            'time_start': time_start,
                            'time_end': time_end,
                            'mask_id': mask_id,
                            'frame_number': mask.get('frame_number'),
                            'bbox_x1': bbox[0],
                            'bbox_y1': bbox[1],
                            'bbox_x2': bbox[2],
                            'bbox_y2': bbox[3],
                            'position_x': position[0],
                            'position_y': position[1],
                            'position_z': position[2],
                            'fixture': mask.get('fixture', 'N/A')
                        })

    return p01_data


def print_summary(data):
    """Print summary statistics"""
    videos = set(item['video_id'] for item in data)
    objects = set((item['video_id'], item['object_id']) for item in data)
    tracks = set((item['video_id'], item['track_id']) for item in data)

    print("=" * 80)
    print("P01 ANNOTATION SUMMARY")
    print("=" * 80)
    print(f"Videos: {len(videos)}")
    print(f"Unique objects: {len(objects)}")
    print(f"Total tracks: {len(tracks)}")
    print(f"Total mask instances: {len(data)}")
    print("=" * 80)
    print()


def print_detailed_list(data, max_items=50):
    """Print detailed list of items"""
    print("DETAILED OBJECT LIST (First {} items)".format(max_items))
    print("=" * 80)

    for i, item in enumerate(data[:max_items]):
        print(f"\n[{i+1}] Video: {item['video_id']}")
        print(f"    Object: {item['object_name']} (ID: {item['object_id'][:16]}...)")
        print(f"    Track {item['track_index']}: {item['time_start']:.2f}s - {item['time_end']:.2f}s")
        print(f"    Mask ID: {item['mask_id']}")
        print(f"    Frame: {item['frame_number']}")

        # Handle None values for bbox
        if all(item[k] is not None for k in ['bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2']):
            print(f"    BBox: [{item['bbox_x1']:.1f}, {item['bbox_y1']:.1f}, "
                  f"{item['bbox_x2']:.1f}, {item['bbox_y2']:.1f}]")
        else:
            print(f"    BBox: N/A")

        # Handle None values for 3D position
        if all(item[k] is not None for k in ['position_x', 'position_y', 'position_z']):
            print(f"    3D Position: [{item['position_x']:.3f}, {item['position_y']:.3f}, "
                  f"{item['position_z']:.3f}]")
        else:
            print(f"    3D Position: N/A")

        print(f"    Fixture: {item['fixture']}")

    if len(data) > max_items:
        print(f"\n... and {len(data) - max_items} more items")


def save_to_csv(data, output_file='p01_objects_list.csv'):
    """Save data to CSV file"""
    if not data:
        print("No data to save")
        return

    fieldnames = list(data[0].keys())

    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)

    print(f"\nSaved {len(data)} entries to {output_file}")


def save_to_json(data, output_file='p01_objects_list.json'):
    """Save data to JSON file"""
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Saved {len(data)} entries to {output_file}")


def print_per_video_summary(data):
    """Print summary per video"""
    from collections import defaultdict

    video_stats = defaultdict(lambda: {
        'objects': set(),
        'tracks': set(),
        'masks': 0
    })

    for item in data:
        vid = item['video_id']
        video_stats[vid]['objects'].add(item['object_id'])
        video_stats[vid]['tracks'].add(item['track_id'])
        video_stats[vid]['masks'] += 1

    print("\n" + "=" * 80)
    print("PER-VIDEO SUMMARY")
    print("=" * 80)
    print(f"{'Video ID':<30} {'Objects':<10} {'Tracks':<10} {'Masks':<10}")
    print("-" * 80)

    for video_id in sorted(video_stats.keys()):
        stats = video_stats[video_id]
        print(f"{video_id:<30} {len(stats['objects']):<10} "
              f"{len(stats['tracks']):<10} {stats['masks']:<10}")

    print("=" * 80)


def main():
    """Main function"""
    print("Loading annotations...")
    mask_info, assoc_info = load_annotations()

    print("Extracting P01 data...")
    p01_data = extract_p01_data(mask_info, assoc_info)

    # Print summaries
    print_summary(p01_data)
    print_per_video_summary(p01_data)
    print_detailed_list(p01_data, max_items=20)

    # Save to files
    print("\n" + "=" * 80)
    print("SAVING TO FILES")
    print("=" * 80)
    save_to_csv(p01_data)
    save_to_json(p01_data)

    print("\nDone!")


if __name__ == '__main__':
    main()

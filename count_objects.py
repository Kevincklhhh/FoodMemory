#!/usr/bin/env python3
"""
Count and analyze objects from p01_objects_list.json

This script provides statistics about:
- Total number of unique objects
- Object name distribution
- Object counts per video
- Most/least common objects
"""

import json
from collections import defaultdict, Counter
import argparse


def count_objects(json_file='p01_objects_list.json', verbose=True):
    """
    Count and analyze objects from the JSON file.

    Args:
        json_file: Path to p01_objects_list.json
        verbose: Print detailed statistics

    Returns:
        Dictionary containing analysis results
    """
    # Load data
    if verbose:
        print(f"Loading {json_file}...")

    with open(json_file, 'r') as f:
        data = json.load(f)

    if verbose:
        print(f"Loaded {len(data)} mask instances\n")

    # Collect statistics
    unique_objects = set()  # (video_id, object_id) tuples
    object_names = []
    objects_by_video = defaultdict(set)  # video_id -> set of object_ids
    objects_by_name = defaultdict(set)  # object_name -> set of (video_id, object_id)
    mask_counts = Counter()  # object_name -> count

    for item in data:
        video_id = item['video_id']
        object_id = item['object_id']
        object_name = item['object_name']

        unique_objects.add((video_id, object_id))
        objects_by_video[video_id].add(object_id)
        objects_by_name[object_name].add((video_id, object_id))
        object_names.append(object_name)
        mask_counts[object_name] += 1

    # Calculate statistics
    unique_object_names = sorted(set(object_names))
    name_counts = Counter(object_names)
    videos = sorted(objects_by_video.keys())

    # Print results
    if verbose:
        print("=" * 80)
        print("OBJECT STATISTICS")
        print("=" * 80)
        print(f"\nTotal mask instances: {len(data)}")
        print(f"Unique objects (across all videos): {len(unique_objects)}")
        print(f"Unique object names: {len(unique_object_names)}")
        print(f"Videos: {len(videos)}")
        print(f"Average objects per video: {len(unique_objects) / len(videos):.1f}")
        print(f"Average mask instances per object: {len(data) / len(unique_objects):.1f}")

        print(f"\n{'=' * 80}")
        print("ALL UNIQUE OBJECT NAMES (Alphabetically)")
        print('=' * 80)

        for i, name in enumerate(unique_object_names, 1):
            count = name_counts[name]
            unique_count = len(objects_by_name[name])
            print(f"{i:3d}. {name:<50} (instances: {count}, unique objects: {unique_count})")

        print(f"\n{'=' * 80}")
        print("TOP 20 MOST COMMON OBJECTS (by mask instances)")
        print('=' * 80)

        for i, (name, count) in enumerate(mask_counts.most_common(20), 1):
            unique_count = len(objects_by_name[name])
            print(f"{i:2d}. {name:<45} {count:4d} instances, {unique_count:3d} unique objects")

        print(f"\n{'=' * 80}")
        print("TOP 20 LEAST COMMON OBJECTS (by mask instances)")
        print('=' * 80)

        least_common = sorted(mask_counts.items(), key=lambda x: (x[1], x[0]))[:20]
        for i, (name, count) in enumerate(least_common, 1):
            unique_count = len(objects_by_name[name])
            print(f"{i:2d}. {name:<45} {count:4d} instances, {unique_count:3d} unique objects")

        print(f"\n{'=' * 80}")
        print("OBJECTS PER VIDEO")
        print('=' * 80)

        for video_id in videos:
            obj_count = len(objects_by_video[video_id])
            # Count mask instances in this video
            mask_count = sum(1 for item in data if item['video_id'] == video_id)
            print(f"{video_id}: {obj_count:3d} objects, {mask_count:4d} mask instances")

        print(f"\n{'=' * 80}")
        print("OBJECT NAME CATEGORIES")
        print('=' * 80)

        # Categorize by common keywords
        categories = {
            'Food items': ['bread', 'cheese', 'egg', 'tomato', 'onion', 'carrot',
                          'pepper', 'salad', 'meat', 'chicken', 'fish', 'rice',
                          'pasta', 'fruit', 'vegetable', 'biscuit', 'cake'],
            'Containers': ['bowl', 'plate', 'cup', 'glass', 'jar', 'container',
                          'box', 'tin', 'packet', 'bag', 'bottle'],
            'Utensils': ['knife', 'fork', 'spoon', 'spatula', 'whisk', 'peeler',
                        'grater', 'tongs'],
            'Cookware': ['pan', 'pot', 'lid', 'tray'],
            'Appliances': ['kettle', 'toaster', 'microwave', 'oven', 'fridge'],
            'Furniture': ['cupboard', 'drawer', 'door', 'shelf', 'counter'],
            'Liquids/Condiments': ['water', 'milk', 'oil', 'sauce', 'vinegar',
                                   'juice', 'liquid'],
        }

        categorized = defaultdict(list)
        uncategorized = []

        for name in unique_object_names:
            found = False
            for category, keywords in categories.items():
                if any(keyword in name.lower() for keyword in keywords):
                    categorized[category].append(name)
                    found = True
                    break
            if not found:
                uncategorized.append(name)

        for category in sorted(categorized.keys()):
            print(f"\n{category} ({len(categorized[category])} types):")
            for name in sorted(categorized[category])[:10]:  # Show first 10
                print(f"  - {name}")
            if len(categorized[category]) > 10:
                print(f"  ... and {len(categorized[category]) - 10} more")

        if uncategorized:
            print(f"\nUncategorized ({len(uncategorized)} types):")
            for name in sorted(uncategorized)[:10]:
                print(f"  - {name}")
            if len(uncategorized) > 10:
                print(f"  ... and {len(uncategorized) - 10} more")

    # Return results
    return {
        'total_mask_instances': len(data),
        'unique_objects': len(unique_objects),
        'unique_object_names': len(unique_object_names),
        'object_names_list': unique_object_names,
        'name_counts': dict(name_counts),
        'videos': len(videos),
        'objects_by_video': {k: len(v) for k, v in objects_by_video.items()},
        'objects_by_name': {k: len(v) for k, v in objects_by_name.items()},
    }


def save_object_names(object_names, output_file='object_names_list.txt'):
    """Save all unique object names to a text file."""
    with open(output_file, 'w') as f:
        for name in object_names:
            f.write(f"{name}\n")
    print(f"\nSaved {len(object_names)} object names to: {output_file}")


def save_object_name_id_mapping(data, output_file='object_name_id_mapping.txt'):
    """
    Save object name to object ID mapping to a text file.

    Format: object_name | object_id | video_id
    """
    # Collect all unique (object_name, object_id, video_id) tuples
    mappings = []
    seen = set()

    for item in data:
        key = (item['object_name'], item['object_id'], item['video_id'])
        if key not in seen:
            mappings.append({
                'object_name': item['object_name'],
                'object_id': item['object_id'],
                'video_id': item['video_id']
            })
            seen.add(key)

    # Sort by object name, then video_id
    mappings.sort(key=lambda x: (x['object_name'], x['video_id']))

    # Write to file
    with open(output_file, 'w') as f:
        # Write header
        f.write("# Object Name to Object ID Mapping\n")
        f.write("# Format: object_name | object_id | video_id\n")
        f.write(f"# Total unique mappings: {len(mappings)}\n")
        f.write("#" + "="*77 + "\n\n")

        for mapping in mappings:
            f.write(f"{mapping['object_name']} | {mapping['object_id']} | {mapping['video_id']}\n")

    print(f"\nSaved {len(mappings)} object name/ID mappings to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Count and analyze objects from p01_objects_list.json'
    )
    parser.add_argument('--json', '-j', default='p01_objects_list.json',
                       help='Path to JSON file (default: p01_objects_list.json)')
    parser.add_argument('--output', '-o',
                       help='Save object names list to file')
    parser.add_argument('--mapping', '-m', action='store_true',
                       help='Save object name/ID mapping to file (object_name_id_mapping.txt)')
    parser.add_argument('--mapping-file', default='object_name_id_mapping.txt',
                       help='Output file for mapping (default: object_name_id_mapping.txt)')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Only print object names')

    args = parser.parse_args()

    # Load raw data for mapping
    with open(args.json, 'r') as f:
        raw_data = json.load(f)

    # Count objects
    results = count_objects(args.json, verbose=not args.quiet)

    # Save object names if requested
    if args.output:
        save_object_names(results['object_names_list'], args.output)

    # Save object name/ID mapping if requested
    if args.mapping:
        save_object_name_id_mapping(raw_data, args.mapping_file)

    # If quiet mode, just print names
    if args.quiet:
        print("\nAll unique object names:")
        for name in results['object_names_list']:
            print(name)


if __name__ == '__main__':
    main()

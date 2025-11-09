#!/usr/bin/env python3
"""
Analyze VISOR annotation coverage compared to EPIC-KITCHENS-100 dataset.

Compares:
1. VISOR sparse annotations (segmentation masks)
2. VISOR WDTCF annotations (object provenance)
3. EPIC-100 full dataset (action annotations)

Output: Per-participant coverage statistics
"""

import json
import csv
from pathlib import Path
from typing import Dict, Set, List
from collections import defaultdict
import re


def load_epic100_videos(epic_dir: Path) -> Dict:
    """
    Load video information from EPIC-100 dataset.

    Returns dict with:
    - all_videos: set of all video IDs
    - by_participant: dict mapping participant -> list of videos
    - by_split: dict mapping split -> list of videos
    """
    train_file = epic_dir / 'EPIC_100_train.csv'
    val_file = epic_dir / 'EPIC_100_validation.csv'

    all_videos = set()
    by_participant = defaultdict(set)
    by_split = defaultdict(set)

    # Load train
    with open(train_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            video_id = row['video_id']
            participant_id = row['participant_id']

            all_videos.add(video_id)
            by_participant[participant_id].add(video_id)
            by_split['train'].add(video_id)

    # Load validation
    with open(val_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            video_id = row['video_id']
            participant_id = row['participant_id']

            all_videos.add(video_id)
            by_participant[participant_id].add(video_id)
            by_split['validation'].add(video_id)

    return {
        'all_videos': all_videos,
        'by_participant': dict(by_participant),
        'by_split': dict(by_split)
    }


def load_visor_sparse_annotations(visor_dir: Path) -> Dict:
    """
    Load VISOR sparse annotation coverage.

    Returns dict with:
    - videos: set of video IDs with sparse annotations
    - by_participant: dict mapping participant -> list of videos
    """
    annotation_dir = visor_dir / 'GroundTruth-SparseAnnotations' / 'annotations'

    videos = set()
    by_participant = defaultdict(set)

    # Check both train and validation splits
    for split in ['train', 'val']:
        split_dir = annotation_dir / split
        if not split_dir.exists():
            continue

        # Each JSON file is named {video_id}.json
        for json_file in split_dir.glob('*.json'):
            video_id = json_file.stem
            participant_id = video_id.split('_')[0]

            videos.add(video_id)
            by_participant[participant_id].add(video_id)

    return {
        'videos': videos,
        'by_participant': dict(by_participant)
    }


def load_visor_wdtcf_annotations(visor_dir: Path) -> Dict:
    """
    Load VISOR WDTCF annotation coverage.

    Returns dict with:
    - videos: set of video IDs with WDTCF annotations
    - by_participant: dict mapping participant -> list of videos
    """
    wdtcf_file = visor_dir / 'WDTCF_GT.json'

    with open(wdtcf_file, 'r') as f:
        data = json.load(f)

    videos = set()
    by_participant = defaultdict(set)

    for annotation_key in data.keys():
        # Parse key: "P01_01_celery" -> video_id="P01_01"
        parts = annotation_key.split('_')
        if len(parts) < 3:
            continue

        video_id = f"{parts[0]}_{parts[1]}"
        participant_id = parts[0]

        videos.add(video_id)
        by_participant[participant_id].add(video_id)

    return {
        'videos': videos,
        'by_participant': dict(by_participant)
    }


def calculate_coverage(epic_data: Dict, visor_sparse: Dict, visor_wdtcf: Dict) -> Dict:
    """
    Calculate coverage statistics.

    Returns comprehensive coverage analysis per participant.
    """
    all_participants = sorted(set(
        list(epic_data['by_participant'].keys()) +
        list(visor_sparse['by_participant'].keys()) +
        list(visor_wdtcf['by_participant'].keys())
    ))

    participant_stats = {}

    for participant in all_participants:
        epic_videos = epic_data['by_participant'].get(participant, set())
        sparse_videos = visor_sparse['by_participant'].get(participant, set())
        wdtcf_videos = visor_wdtcf['by_participant'].get(participant, set())

        # Combined VISOR coverage
        visor_any = sparse_videos | wdtcf_videos

        # Calculate percentages
        total_videos = len(epic_videos)
        sparse_coverage = (len(sparse_videos) / total_videos * 100) if total_videos > 0 else 0
        wdtcf_coverage = (len(wdtcf_videos) / total_videos * 100) if total_videos > 0 else 0
        any_coverage = (len(visor_any) / total_videos * 100) if total_videos > 0 else 0

        participant_stats[participant] = {
            'total_videos': total_videos,
            'sparse_videos': len(sparse_videos),
            'wdtcf_videos': len(wdtcf_videos),
            'any_visor': len(visor_any),
            'sparse_coverage_pct': sparse_coverage,
            'wdtcf_coverage_pct': wdtcf_coverage,
            'any_visor_coverage_pct': any_coverage,
            'sparse_only': len(sparse_videos - wdtcf_videos),
            'wdtcf_only': len(wdtcf_videos - sparse_videos),
            'both': len(sparse_videos & wdtcf_videos)
        }

    # Overall statistics
    total_epic = len(epic_data['all_videos'])
    total_sparse = len(visor_sparse['videos'])
    total_wdtcf = len(visor_wdtcf['videos'])
    total_any = len(visor_sparse['videos'] | visor_wdtcf['videos'])

    overall_stats = {
        'total_epic_videos': total_epic,
        'total_sparse_videos': total_sparse,
        'total_wdtcf_videos': total_wdtcf,
        'total_any_visor': total_any,
        'sparse_coverage_pct': (total_sparse / total_epic * 100),
        'wdtcf_coverage_pct': (total_wdtcf / total_epic * 100),
        'any_visor_coverage_pct': (total_any / total_epic * 100),
        'sparse_only': len(visor_sparse['videos'] - visor_wdtcf['videos']),
        'wdtcf_only': len(visor_wdtcf['videos'] - visor_sparse['videos']),
        'both': len(visor_sparse['videos'] & visor_wdtcf['videos'])
    }

    # Split coverage
    split_stats = {}
    for split_name, split_videos in epic_data['by_split'].items():
        sparse_in_split = visor_sparse['videos'] & split_videos
        wdtcf_in_split = visor_wdtcf['videos'] & split_videos
        any_in_split = sparse_in_split | wdtcf_in_split

        split_stats[split_name] = {
            'total_videos': len(split_videos),
            'sparse_videos': len(sparse_in_split),
            'wdtcf_videos': len(wdtcf_in_split),
            'any_visor': len(any_in_split),
            'sparse_coverage_pct': (len(sparse_in_split) / len(split_videos) * 100),
            'wdtcf_coverage_pct': (len(wdtcf_in_split) / len(split_videos) * 100),
            'any_visor_coverage_pct': (len(any_in_split) / len(split_videos) * 100)
        }

    return {
        'overall': overall_stats,
        'by_split': split_stats,
        'by_participant': participant_stats
    }


def save_results(coverage: Dict, output_dir: Path):
    """Save coverage analysis results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save JSON
    json_file = output_dir / 'visor_coverage_analysis.json'
    with open(json_file, 'w') as f:
        json.dump(coverage, f, indent=2)
    print(f"\n✓ Saved JSON to {json_file}")

    # 2. Save per-participant CSV
    csv_file = output_dir / 'visor_coverage_by_participant.csv'
    with open(csv_file, 'w', newline='') as f:
        fieldnames = [
            'participant_id', 'total_videos',
            'sparse_videos', 'sparse_coverage_pct',
            'wdtcf_videos', 'wdtcf_coverage_pct',
            'any_visor', 'any_visor_coverage_pct',
            'sparse_only', 'wdtcf_only', 'both'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for participant, stats in sorted(coverage['by_participant'].items()):
            writer.writerow({
                'participant_id': participant,
                'total_videos': stats['total_videos'],
                'sparse_videos': stats['sparse_videos'],
                'sparse_coverage_pct': f"{stats['sparse_coverage_pct']:.1f}",
                'wdtcf_videos': stats['wdtcf_videos'],
                'wdtcf_coverage_pct': f"{stats['wdtcf_coverage_pct']:.1f}",
                'any_visor': stats['any_visor'],
                'any_visor_coverage_pct': f"{stats['any_visor_coverage_pct']:.1f}",
                'sparse_only': stats['sparse_only'],
                'wdtcf_only': stats['wdtcf_only'],
                'both': stats['both']
            })
    print(f"✓ Saved participant CSV to {csv_file}")

    # 3. Save detailed text report
    report_file = output_dir / 'visor_coverage_report.txt'
    with open(report_file, 'w') as f:
        f.write("VISOR ANNOTATION COVERAGE ANALYSIS\n")
        f.write("=" * 100 + "\n\n")

        # Overall statistics
        overall = coverage['overall']
        f.write("OVERALL STATISTICS\n")
        f.write("-" * 100 + "\n")
        f.write(f"Total EPIC-KITCHENS-100 videos: {overall['total_epic_videos']}\n")
        f.write(f"Videos with VISOR sparse annotations: {overall['total_sparse_videos']} ({overall['sparse_coverage_pct']:.1f}%)\n")
        f.write(f"Videos with VISOR WDTCF annotations: {overall['total_wdtcf_videos']} ({overall['wdtcf_coverage_pct']:.1f}%)\n")
        f.write(f"Videos with ANY VISOR annotation: {overall['total_any_visor']} ({overall['any_visor_coverage_pct']:.1f}%)\n\n")
        f.write(f"Sparse only: {overall['sparse_only']}\n")
        f.write(f"WDTCF only: {overall['wdtcf_only']}\n")
        f.write(f"Both sparse and WDTCF: {overall['both']}\n\n")

        # Split statistics
        f.write("\nCOVERAGE BY SPLIT\n")
        f.write("-" * 100 + "\n")
        for split_name, split_stats in coverage['by_split'].items():
            f.write(f"\n{split_name.upper()}:\n")
            f.write(f"  Total videos: {split_stats['total_videos']}\n")
            f.write(f"  Sparse annotations: {split_stats['sparse_videos']} ({split_stats['sparse_coverage_pct']:.1f}%)\n")
            f.write(f"  WDTCF annotations: {split_stats['wdtcf_videos']} ({split_stats['wdtcf_coverage_pct']:.1f}%)\n")
            f.write(f"  Any VISOR: {split_stats['any_visor']} ({split_stats['any_visor_coverage_pct']:.1f}%)\n")

        # Per-participant statistics
        f.write("\n\nCOVERAGE BY PARTICIPANT\n")
        f.write("-" * 100 + "\n")
        f.write(f"{'Participant':<12} {'Total':>6} {'Sparse':>7} {'%':>6} {'WDTCF':>7} {'%':>6} {'Any':>6} {'%':>6} {'Details':<30}\n")
        f.write("-" * 100 + "\n")

        for participant, stats in sorted(coverage['by_participant'].items()):
            details = f"S:{stats['sparse_only']} W:{stats['wdtcf_only']} Both:{stats['both']}"
            f.write(f"{participant:<12} {stats['total_videos']:>6} "
                   f"{stats['sparse_videos']:>7} {stats['sparse_coverage_pct']:>5.1f}% "
                   f"{stats['wdtcf_videos']:>7} {stats['wdtcf_coverage_pct']:>5.1f}% "
                   f"{stats['any_visor']:>6} {stats['any_visor_coverage_pct']:>5.1f}% "
                   f"{details:<30}\n")

        # Summary
        f.write("\n\nSUMMARY\n")
        f.write("-" * 100 + "\n")

        participant_count = len(coverage['by_participant'])
        participants_with_sparse = sum(1 for p, s in coverage['by_participant'].items() if s['sparse_videos'] > 0)
        participants_with_wdtcf = sum(1 for p, s in coverage['by_participant'].items() if s['wdtcf_videos'] > 0)

        f.write(f"Total participants in EPIC-100: {participant_count}\n")
        f.write(f"Participants with sparse annotations: {participants_with_sparse}\n")
        f.write(f"Participants with WDTCF annotations: {participants_with_wdtcf}\n\n")

        # Participants with best coverage
        f.write("TOP 10 PARTICIPANTS BY COVERAGE:\n")
        sorted_participants = sorted(
            coverage['by_participant'].items(),
            key=lambda x: x[1]['any_visor_coverage_pct'],
            reverse=True
        )[:10]

        for participant, stats in sorted_participants:
            f.write(f"  {participant}: {stats['any_visor']}/{stats['total_videos']} videos ({stats['any_visor_coverage_pct']:.1f}%)\n")

    print(f"✓ Saved detailed report to {report_file}")


def print_summary(coverage: Dict):
    """Print summary to console."""
    overall = coverage['overall']

    print("\n" + "=" * 100)
    print("VISOR ANNOTATION COVERAGE SUMMARY")
    print("=" * 100)

    print(f"\nTotal EPIC-KITCHENS-100 videos: {overall['total_epic_videos']}")
    print(f"Videos with VISOR sparse annotations: {overall['total_sparse_videos']} ({overall['sparse_coverage_pct']:.1f}%)")
    print(f"Videos with VISOR WDTCF annotations: {overall['total_wdtcf_videos']} ({overall['wdtcf_coverage_pct']:.1f}%)")
    print(f"Videos with ANY VISOR annotation: {overall['total_any_visor']} ({overall['any_visor_coverage_pct']:.1f}%)")

    print(f"\nAnnotation overlap:")
    print(f"  Sparse only: {overall['sparse_only']}")
    print(f"  WDTCF only: {overall['wdtcf_only']}")
    print(f"  Both: {overall['both']}")

    print("\nTop 5 participants by coverage:")
    sorted_participants = sorted(
        coverage['by_participant'].items(),
        key=lambda x: x[1]['any_visor_coverage_pct'],
        reverse=True
    )[:5]

    for participant, stats in sorted_participants:
        print(f"  {participant}: {stats['any_visor']}/{stats['total_videos']} videos ({stats['any_visor_coverage_pct']:.1f}%)")


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze VISOR annotation coverage vs EPIC-KITCHENS-100"
    )
    parser.add_argument(
        '--epic-dir',
        default='/home/kailaic/NeuroTrace/kitchen/epic-kitchens-100-annotations',
        help='Path to EPIC-KITCHENS-100 annotations directory'
    )
    parser.add_argument(
        '--visor-dir',
        default='/home/kailaic/NeuroTrace/kitchen/epic-kitchen-visor',
        help='Path to VISOR directory'
    )
    parser.add_argument(
        '--output-dir',
        default='/home/kailaic/NeuroTrace/kitchen/epic-kitchen-visor',
        help='Output directory for results'
    )

    args = parser.parse_args()

    print("Loading EPIC-KITCHENS-100 video information...")
    epic_data = load_epic100_videos(Path(args.epic_dir))
    print(f"Found {len(epic_data['all_videos'])} total videos")

    print("\nLoading VISOR sparse annotations...")
    visor_sparse = load_visor_sparse_annotations(Path(args.visor_dir))
    print(f"Found {len(visor_sparse['videos'])} videos with sparse annotations")

    print("\nLoading VISOR WDTCF annotations...")
    visor_wdtcf = load_visor_wdtcf_annotations(Path(args.visor_dir))
    print(f"Found {len(visor_wdtcf['videos'])} videos with WDTCF annotations")

    print("\nCalculating coverage statistics...")
    coverage = calculate_coverage(epic_data, visor_sparse, visor_wdtcf)

    print("\nSaving results...")
    save_results(coverage, Path(args.output_dir))

    print_summary(coverage)

    print("\n✓ Done!")


if __name__ == '__main__':
    main()

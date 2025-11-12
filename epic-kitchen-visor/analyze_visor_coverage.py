#!/usr/bin/env python3
"""
Analyze VISOR annotation coverage over EPIC-KITCHENS-100 per participant.
Compares video coverage between VISOR and EPIC-100 datasets.
"""

import pandas as pd
import json
from pathlib import Path
from collections import defaultdict

# Paths
EPIC100_TRAIN = Path("/home/kailaic/NeuroTrace/kitchen/epic-kitchens-100-annotations/EPIC_100_train.csv")
EPIC100_VAL = Path("/home/kailaic/NeuroTrace/kitchen/epic-kitchens-100-annotations/EPIC_100_validation.csv")
VISOR_TRAIN = Path("/home/kailaic/NeuroTrace/kitchen/epic-kitchen-visor/GroundTruth-SparseAnnotations/annotations/train")
VISOR_VAL = Path("/home/kailaic/NeuroTrace/kitchen/epic-kitchen-visor/GroundTruth-SparseAnnotations/annotations/val")

def get_epic100_videos():
    """Get all unique videos per participant from EPIC-100."""
    train_df = pd.read_csv(EPIC100_TRAIN)
    val_df = pd.read_csv(EPIC100_VAL)

    # Combine train and validation
    all_df = pd.concat([train_df, val_df])

    # Get unique video IDs per participant
    participant_videos = defaultdict(set)
    for _, row in all_df.iterrows():
        participant_id = row['participant_id']
        video_id = row['video_id']
        participant_videos[participant_id].add(video_id)

    return dict(participant_videos)

def get_visor_videos():
    """Get all videos with VISOR annotations per participant."""
    participant_videos = defaultdict(set)

    # Process train annotations
    for json_file in VISOR_TRAIN.glob("*.json"):
        video_id = json_file.stem  # e.g., "P01_01"
        participant_id = video_id.split('_')[0]  # e.g., "P01"
        participant_videos[participant_id].add(video_id)

    # Process val annotations
    for json_file in VISOR_VAL.glob("*.json"):
        video_id = json_file.stem
        participant_id = video_id.split('_')[0]
        participant_videos[participant_id].add(video_id)

    return dict(participant_videos)

def calculate_coverage_stats(epic100_videos, visor_videos):
    """Calculate coverage statistics per participant."""
    all_participants = sorted(set(list(epic100_videos.keys()) + list(visor_videos.keys())))

    stats = []
    for participant in all_participants:
        epic_vids = epic100_videos.get(participant, set())
        visor_vids = visor_videos.get(participant, set())

        total_videos = len(epic_vids)
        annotated_videos = len(visor_vids)
        coverage_pct = (annotated_videos / total_videos * 100) if total_videos > 0 else 0

        # Check for videos in VISOR but not in EPIC-100 (shouldn't happen)
        extra_visor = visor_vids - epic_vids

        stats.append({
            'participant_id': participant,
            'epic100_videos': total_videos,
            'visor_videos': annotated_videos,
            'coverage_pct': coverage_pct,
            'extra_visor_videos': len(extra_visor)
        })

    return stats

def generate_report(stats):
    """Generate a detailed coverage report."""
    df = pd.DataFrame(stats)

    # Overall statistics
    total_epic_videos = df['epic100_videos'].sum()
    total_visor_videos = df['visor_videos'].sum()
    overall_coverage = (total_visor_videos / total_epic_videos * 100) if total_epic_videos > 0 else 0

    # Save detailed table
    output_csv = Path("/home/kailaic/NeuroTrace/kitchen/epic-kitchen-visor/visor_coverage_per_participant.csv")
    df.to_csv(output_csv, index=False)
    print(f"Saved detailed coverage table to: {output_csv}")

    # Generate text report
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("VISOR ANNOTATION COVERAGE OVER EPIC-KITCHENS-100")
    report_lines.append("=" * 80)
    report_lines.append("")

    report_lines.append("OVERALL STATISTICS:")
    report_lines.append(f"  Total EPIC-100 videos: {total_epic_videos}")
    report_lines.append(f"  Total VISOR annotated videos: {total_visor_videos}")
    report_lines.append(f"  Overall coverage: {overall_coverage:.2f}%")
    report_lines.append(f"  Number of participants in EPIC-100: {len(df[df['epic100_videos'] > 0])}")
    report_lines.append(f"  Number of participants with VISOR annotations: {len(df[df['visor_videos'] > 0])}")
    report_lines.append("")

    report_lines.append("COVERAGE DISTRIBUTION:")
    report_lines.append(f"  Participants with 100% coverage: {len(df[df['coverage_pct'] == 100])}")
    report_lines.append(f"  Participants with 50-99% coverage: {len(df[(df['coverage_pct'] >= 50) & (df['coverage_pct'] < 100)])}")
    report_lines.append(f"  Participants with 1-49% coverage: {len(df[(df['coverage_pct'] > 0) & (df['coverage_pct'] < 50)])}")
    report_lines.append(f"  Participants with 0% coverage: {len(df[df['coverage_pct'] == 0])}")
    report_lines.append("")

    report_lines.append("PER-PARTICIPANT COVERAGE:")
    report_lines.append(f"{'Participant':<15} {'EPIC-100':<12} {'VISOR':<12} {'Coverage':<12}")
    report_lines.append("-" * 55)

    for _, row in df.iterrows():
        report_lines.append(
            f"{row['participant_id']:<15} "
            f"{row['epic100_videos']:<12} "
            f"{row['visor_videos']:<12} "
            f"{row['coverage_pct']:>6.2f}%"
        )

    report_lines.append("")
    report_lines.append("=" * 80)

    report_text = "\n".join(report_lines)

    # Save text report
    report_file = Path("/home/kailaic/NeuroTrace/kitchen/epic-kitchen-visor/visor_coverage_report.txt")
    report_file.write_text(report_text)
    print(f"Saved coverage report to: {report_file}")

    # Print to console
    print("\n" + report_text)

    return df

def main():
    print("Loading EPIC-KITCHENS-100 videos...")
    epic100_videos = get_epic100_videos()

    print("Loading VISOR annotated videos...")
    visor_videos = get_visor_videos()

    print("Calculating coverage statistics...")
    stats = calculate_coverage_stats(epic100_videos, visor_videos)

    print("Generating coverage report...")
    df = generate_report(stats)

    return df

if __name__ == "__main__":
    main()
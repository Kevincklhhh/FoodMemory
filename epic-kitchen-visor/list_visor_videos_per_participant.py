#!/usr/bin/env python3
"""
List specific VISOR-annotated videos per participant.
Shows which videos from EPIC-KITCHENS-100 have VISOR annotations.
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

def get_visor_videos_with_split():
    """Get all videos with VISOR annotations per participant, including split info."""
    participant_videos = defaultdict(lambda: {'train': set(), 'val': set()})

    # Process train annotations
    for json_file in VISOR_TRAIN.glob("*.json"):
        video_id = json_file.stem  # e.g., "P01_01"
        participant_id = video_id.split('_')[0]  # e.g., "P01"
        participant_videos[participant_id]['train'].add(video_id)

    # Process val annotations
    for json_file in VISOR_VAL.glob("*.json"):
        video_id = json_file.stem
        participant_id = video_id.split('_')[0]
        participant_videos[participant_id]['val'].add(video_id)

    return dict(participant_videos)

def generate_video_list_report(epic100_videos, visor_videos):
    """Generate detailed report listing all videos per participant."""
    all_participants = sorted(set(list(epic100_videos.keys()) + list(visor_videos.keys())))

    report_lines = []
    report_lines.append("=" * 100)
    report_lines.append("VISOR ANNOTATED VIDEOS PER PARTICIPANT")
    report_lines.append("=" * 100)
    report_lines.append("")

    # Also prepare CSV data
    csv_rows = []

    for participant in all_participants:
        epic_vids = epic100_videos.get(participant, set())
        visor_info = visor_videos.get(participant, {'train': set(), 'val': set()})
        visor_train = visor_info['train']
        visor_val = visor_info['val']
        all_visor_vids = visor_train | visor_val

        total_videos = len(epic_vids)
        annotated_videos = len(all_visor_vids)
        coverage_pct = (annotated_videos / total_videos * 100) if total_videos > 0 else 0

        # Missing videos
        missing_vids = epic_vids - all_visor_vids

        report_lines.append(f"{participant} - Coverage: {annotated_videos}/{total_videos} ({coverage_pct:.1f}%)")
        report_lines.append("-" * 100)

        if all_visor_vids:
            report_lines.append(f"  VISOR Train Videos ({len(visor_train)}):")
            if visor_train:
                for vid in sorted(visor_train):
                    report_lines.append(f"    - {vid}")
                    csv_rows.append({
                        'participant_id': participant,
                        'video_id': vid,
                        'visor_split': 'train',
                        'has_visor': True
                    })
            else:
                report_lines.append(f"    (none)")

            report_lines.append(f"  VISOR Val Videos ({len(visor_val)}):")
            if visor_val:
                for vid in sorted(visor_val):
                    report_lines.append(f"    - {vid}")
                    csv_rows.append({
                        'participant_id': participant,
                        'video_id': vid,
                        'visor_split': 'val',
                        'has_visor': True
                    })
            else:
                report_lines.append(f"    (none)")
        else:
            report_lines.append(f"  No VISOR annotations")

        if missing_vids:
            report_lines.append(f"  Missing VISOR Annotations ({len(missing_vids)}):")
            # Show first 10 missing videos to avoid cluttering
            missing_list = sorted(missing_vids)
            for vid in missing_list[:10]:
                report_lines.append(f"    - {vid}")
                csv_rows.append({
                    'participant_id': participant,
                    'video_id': vid,
                    'visor_split': None,
                    'has_visor': False
                })
            if len(missing_list) > 10:
                report_lines.append(f"    ... and {len(missing_list) - 10} more")
                # Add remaining to CSV
                for vid in missing_list[10:]:
                    csv_rows.append({
                        'participant_id': participant,
                        'video_id': vid,
                        'visor_split': None,
                        'has_visor': False
                    })

        report_lines.append("")

    report_lines.append("=" * 100)

    report_text = "\n".join(report_lines)

    # Save text report
    report_file = Path("/home/kailaic/NeuroTrace/kitchen/epic-kitchen-visor/visor_video_list_per_participant.txt")
    report_file.write_text(report_text)
    print(f"Saved video list report to: {report_file}")

    # Save CSV
    df = pd.DataFrame(csv_rows)
    csv_file = Path("/home/kailaic/NeuroTrace/kitchen/epic-kitchen-visor/visor_video_list_per_participant.csv")
    df.to_csv(csv_file, index=False)
    print(f"Saved video list CSV to: {csv_file}")

    # Print to console
    print("\n" + report_text)

    return df

def generate_visor_only_list(visor_videos):
    """Generate a simple list of just the VISOR-annotated videos."""
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("ALL VISOR-ANNOTATED VIDEOS (GROUPED BY PARTICIPANT)")
    report_lines.append("=" * 80)
    report_lines.append("")

    all_participants = sorted(visor_videos.keys())

    for participant in all_participants:
        visor_info = visor_videos[participant]
        visor_train = visor_info['train']
        visor_val = visor_info['val']
        all_visor_vids = visor_train | visor_val

        if all_visor_vids:
            report_lines.append(f"{participant} ({len(all_visor_vids)} videos):")
            report_lines.append(f"  Train: {', '.join(sorted(visor_train)) if visor_train else '(none)'}")
            report_lines.append(f"  Val: {', '.join(sorted(visor_val)) if visor_val else '(none)'}")
            report_lines.append("")

    report_lines.append("=" * 80)

    report_text = "\n".join(report_lines)

    # Save simple list
    report_file = Path("/home/kailaic/NeuroTrace/kitchen/epic-kitchen-visor/visor_annotated_videos_only.txt")
    report_file.write_text(report_text)
    print(f"Saved VISOR-only video list to: {report_file}")

    print("\n" + report_text)

def main():
    print("Loading EPIC-KITCHENS-100 videos...")
    epic100_videos = get_epic100_videos()

    print("Loading VISOR annotated videos...")
    visor_videos = get_visor_videos_with_split()

    print("\nGenerating detailed video list report...")
    df = generate_video_list_report(epic100_videos, visor_videos)

    print("\nGenerating VISOR-only video list...")
    generate_visor_only_list(visor_videos)

    return df

if __name__ == "__main__":
    main()

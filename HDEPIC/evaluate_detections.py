#!/usr/bin/env python3
"""
Evaluate hand-object interaction detections against narration annotations.

This script compares the frames where hand-object interactions were detected
with the timestamp ranges in the narration CSV file.

Evaluation metrics:
1. Coverage: Each narration time range should have at least one detection
2. Precision: Detections outside narration time ranges are flagged
"""

import os
import argparse
import json
import pandas as pd
from pathlib import Path


def load_detections(detection_file):
    """
    Load detection results from JSON file.

    Args:
        detection_file: Path to detection results JSON

    Returns:
        list: Detection results with timestamps
    """
    with open(detection_file, 'r') as f:
        data = json.load(f)

    # Extract frames with hand-object interactions
    interactions = []
    for frame in data["frames"]:
        if frame["detection"]["has_interaction"]:
            interactions.append({
                "filename": frame["filename"],
                "timestamp": frame["timestamp"],
                "num_hands": frame["detection"]["num_hands"],
                "hands": frame["detection"]["hands"]
            })

    return interactions


def load_narrations(narration_csv, video_id):
    """
    Load narration annotations for specific video.

    Args:
        narration_csv: Path to narration CSV file
        video_id: Video ID to filter by

    Returns:
        pd.DataFrame: Narration annotations for the video
    """
    df = pd.read_csv(narration_csv)

    # Filter by video_id
    df_video = df[df['video_id'] == video_id].copy()

    print(f"Loaded {len(df_video)} narrations for video {video_id}")

    return df_video


def evaluate_coverage(interactions, narrations):
    """
    Evaluate if each narration time range has at least one detection.

    Args:
        interactions: List of detection results with timestamps
        narrations: DataFrame with narration annotations

    Returns:
        dict: Coverage evaluation results
    """
    results = []

    for idx, narration in narrations.iterrows():
        start_time = narration['start_timestamp']
        end_time = narration['end_timestamp']
        narration_id = narration['unique_narration_id']
        narration_text = narration['narration']
        hands_involved = narration['hands']

        # Find detections within this time range
        matching_detections = [
            d for d in interactions
            if d['timestamp'] is not None and start_time <= d['timestamp'] <= end_time
        ]

        has_coverage = len(matching_detections) > 0

        result = {
            "narration_id": narration_id,
            "start_time": start_time,
            "end_time": end_time,
            "duration": end_time - start_time,
            "narration": narration_text,
            "hands_involved": hands_involved,
            "has_coverage": has_coverage,
            "num_detections": len(matching_detections),
            "detection_timestamps": [d['timestamp'] for d in matching_detections]
        }

        results.append(result)

    return results


def evaluate_precision(interactions, narrations):
    """
    Find detections that fall outside any narration time range.

    Args:
        interactions: List of detection results with timestamps
        narrations: DataFrame with narration annotations

    Returns:
        list: Detections outside narration ranges
    """
    outside_detections = []

    for detection in interactions:
        timestamp = detection['timestamp']
        if timestamp is None:
            continue

        # Check if timestamp falls within any narration range
        in_any_range = False
        for idx, narration in narrations.iterrows():
            if narration['start_timestamp'] <= timestamp <= narration['end_timestamp']:
                in_any_range = True
                break

        if not in_any_range:
            outside_detections.append(detection)

    return outside_detections


def generate_report(coverage_results, outside_detections, narrations, output_file):
    """
    Generate evaluation report.

    Args:
        coverage_results: Coverage evaluation results
        outside_detections: Detections outside narration ranges
        narrations: DataFrame with narration annotations
        output_file: Path to save report
    """
    report = []

    report.append("=" * 80)
    report.append("HAND-OBJECT INTERACTION DETECTION EVALUATION REPORT")
    report.append("=" * 80)
    report.append("")

    # Overall statistics
    total_narrations = len(coverage_results)
    covered_narrations = sum(1 for r in coverage_results if r["has_coverage"])
    uncovered_narrations = total_narrations - covered_narrations

    report.append("OVERALL STATISTICS")
    report.append("-" * 80)
    report.append(f"Total narrations: {total_narrations}")
    report.append(f"Narrations with detections: {covered_narrations} ({covered_narrations/total_narrations*100:.1f}%)")
    report.append(f"Narrations without detections: {uncovered_narrations} ({uncovered_narrations/total_narrations*100:.1f}%)")
    report.append(f"Detections outside narration ranges: {len(outside_detections)}")
    report.append("")

    # Narration time coverage
    total_narration_time = sum(r["duration"] for r in coverage_results)
    video_duration = narrations['end_timestamp'].max()
    coverage_ratio = total_narration_time / video_duration if video_duration > 0 else 0

    report.append(f"Total video duration: {video_duration:.2f} seconds")
    report.append(f"Total narration time: {total_narration_time:.2f} seconds")
    report.append(f"Narration coverage: {coverage_ratio*100:.1f}% of video")
    report.append("")

    # Coverage details
    report.append("=" * 80)
    report.append("COVERAGE DETAILS (Narrations with/without detections)")
    report.append("=" * 80)
    report.append("")

    # Uncovered narrations (missing detections)
    if uncovered_narrations > 0:
        report.append(f"UNCOVERED NARRATIONS ({uncovered_narrations}):")
        report.append("-" * 80)
        for result in coverage_results:
            if not result["has_coverage"]:
                report.append(f"ID: {result['narration_id']}")
                report.append(f"Time: {result['start_time']:.2f}s - {result['end_time']:.2f}s (duration: {result['duration']:.2f}s)")
                report.append(f"Hands: {result['hands_involved']}")
                report.append(f"Narration: {result['narration']}")
                report.append(f"Status: NO DETECTION FOUND")
                report.append("")
    else:
        report.append("All narrations have at least one detection! ✓")
        report.append("")

    # Covered narrations summary
    report.append(f"COVERED NARRATIONS ({covered_narrations}):")
    report.append("-" * 80)
    for result in coverage_results:
        if result["has_coverage"]:
            report.append(f"ID: {result['narration_id']}")
            report.append(f"Time: {result['start_time']:.2f}s - {result['end_time']:.2f}s")
            report.append(f"Detections: {result['num_detections']} at timestamps {[f'{t:.2f}s' for t in result['detection_timestamps']]}")
            report.append("")

    # Outside detections
    report.append("=" * 80)
    report.append(f"DETECTIONS OUTSIDE NARRATION RANGES ({len(outside_detections)})")
    report.append("=" * 80)
    report.append("")

    if len(outside_detections) > 0:
        report.append("These detections may indicate:")
        report.append("1. False positives from the detector")
        report.append("2. Interactions not captured in narrations")
        report.append("3. Timing misalignment between video and narrations")
        report.append("")

        # Group by time for easier reading
        for detection in outside_detections:
            report.append(f"Timestamp: {detection['timestamp']:.2f}s")
            report.append(f"Hands detected: {detection['num_hands']}")
            report.append(f"Filename: {detection['filename']}")
            report.append("")
    else:
        report.append("No detections outside narration ranges! ✓")
        report.append("")

    # Summary statistics
    report.append("=" * 80)
    report.append("SUMMARY")
    report.append("=" * 80)
    coverage_pct = (covered_narrations / total_narrations * 100) if total_narrations > 0 else 0
    report.append(f"Coverage Rate: {coverage_pct:.1f}% ({covered_narrations}/{total_narrations} narrations)")
    report.append(f"False Positive Rate: {len(outside_detections)} detections outside narration ranges")

    if uncovered_narrations == 0 and len(outside_detections) == 0:
        report.append("\n✓ PERFECT MATCH: All narrations covered, no extra detections!")
    elif uncovered_narrations == 0:
        report.append(f"\n⚠ All narrations covered, but {len(outside_detections)} extra detections found")
    else:
        report.append(f"\n⚠ {uncovered_narrations} narrations missing detections")

    report.append("")

    # Write report
    report_text = "\n".join(report)
    print(report_text)

    with open(output_file, 'w') as f:
        f.write(report_text)

    print(f"\nReport saved to: {output_file}")

    # Also save JSON format
    json_output = output_file.replace('.txt', '.json')
    json_data = {
        "statistics": {
            "total_narrations": total_narrations,
            "covered_narrations": covered_narrations,
            "uncovered_narrations": uncovered_narrations,
            "coverage_percentage": coverage_pct,
            "outside_detections": len(outside_detections),
            "video_duration": video_duration,
            "total_narration_time": total_narration_time
        },
        "coverage_results": coverage_results,
        "outside_detections": outside_detections
    }

    with open(json_output, 'w') as f:
        json.dump(json_data, f, indent=2)

    print(f"JSON report saved to: {json_output}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate hand-object interaction detections against narrations"
    )
    parser.add_argument(
        "--detection_file",
        required=True,
        help="Path to detection results JSON file"
    )
    parser.add_argument(
        "--narration_csv",
        required=True,
        help="Path to narration CSV file"
    )
    parser.add_argument(
        "--video_id",
        required=True,
        help="Video ID to evaluate"
    )
    parser.add_argument(
        "--output_file",
        required=True,
        help="Path to save evaluation report (TXT)"
    )

    args = parser.parse_args()

    print("Loading data...")
    interactions = load_detections(args.detection_file)
    print(f"Loaded {len(interactions)} frames with hand-object interactions")

    narrations = load_narrations(args.narration_csv, args.video_id)

    print("\nEvaluating coverage...")
    coverage_results = evaluate_coverage(interactions, narrations)

    print("Evaluating precision...")
    outside_detections = evaluate_precision(interactions, narrations)

    print("\nGenerating report...")
    generate_report(coverage_results, outside_detections, narrations, args.output_file)


if __name__ == "__main__":
    main()

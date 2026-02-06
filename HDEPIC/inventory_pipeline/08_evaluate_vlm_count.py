#!/usr/bin/env python3
"""
Evaluator for VLM QA results - Count accuracy evaluation.

For items with LOW difficulty, evaluates count accuracy where each segment
has exactly one corresponding prediction (1:1 mapping).

Metrics:
- Mean Accuracy: fraction of segments with correct count prediction
- Mean Absolute Error (MAE): average |predicted - ground_truth| across segments

Usage:
    # Evaluate all participants
    python evaluate_vlm_count.py

    # Evaluate specific model
    python evaluate_vlm_count.py --model qwen

    # Custom output directory
    python evaluate_vlm_count.py --output-dir outputs/02_inventory
"""

import json
import argparse
from pathlib import Path
from datetime import datetime

from inventory_utils import DEFAULT_OUTPUT_DIR


def load_timeline_gt(timeline_path: Path) -> dict:
    """
    Load ground truth from timeline_annotated.json.

    Returns two lookups:
      item_lookup:    narration_id -> {food_name, difficulty, total_count, count_unit}
      segment_lookup: (narration_id, segment_idx) -> {count, count_unit}
    """
    item_lookup = {}
    segment_lookup = {}

    if not timeline_path.exists():
        return item_lookup, segment_lookup

    with open(timeline_path, 'r') as f:
        data = json.load(f)

    for item in data.get('items', []):
        narr_id = item.get('narration_id', '')
        item_lookup[narr_id] = {
            'food_name': item.get('food_name'),
            'difficulty': item.get('difficulty'),
            'total_count': item.get('total_count'),
            'count_unit': item.get('count_unit'),
        }
        for idx, seg in enumerate(item.get('dispensal_segments', [])):
            segment_lookup[(narr_id, idx)] = {
                'count': seg.get('count'),
                'count_unit': seg.get('count_unit'),
            }

    return item_lookup, segment_lookup


def evaluate_participant(results_path: Path, timeline_path: Path = None) -> dict:
    """
    Evaluate VLM QA results for a single participant.

    GT is loaded from timeline_annotated when available, falling back
    to fields embedded in VLM results for backward compatibility.

    Returns dict with participant metrics and item details.
    """
    with open(results_path, 'r') as f:
        data = json.load(f)

    # Load GT from timeline_annotated (preferred source)
    tl_item_lookup, tl_seg_lookup = {}, {}
    if timeline_path:
        tl_item_lookup, tl_seg_lookup = load_timeline_gt(timeline_path)

    items = data.get('items', [])

    # Collect segment-level evaluations for LOW difficulty items
    all_segments = []
    item_summaries = []

    for item in items:
        narr_id = item.get('narration_id', '')

        # Get difficulty from timeline, fall back to VLM data
        tl_item = tl_item_lookup.get(narr_id, {})
        difficulty = (tl_item.get('difficulty') or item.get('difficulty', '')).upper()
        food_name = tl_item.get('food_name') or item.get('food_name')

        if difficulty != 'LOW':
            continue

        segments = item.get('segments', [])
        item_correct = 0
        item_total = 0
        item_abs_errors = []
        seg_details = []

        for seg in segments:
            seg_idx = seg.get('segment_idx', 0)

            # Get GT from timeline, fall back to VLM segment data
            tl_seg = tl_seg_lookup.get((narr_id, seg_idx), {})
            gt = tl_seg.get('count') if tl_seg.get('count') is not None else seg.get('ground_truth_count')
            gt_unit = tl_seg.get('count_unit') or seg.get('ground_truth_unit')

            pred = seg.get('predicted_count')

            # Skip if either is null
            if gt is None or pred is None:
                seg_details.append({
                    'segment_idx': seg_idx,
                    'ground_truth': gt,
                    'predicted': pred,
                    'is_correct': None,
                    'absolute_error': None,
                    'skipped': True
                })
                continue

            is_correct = (pred == gt)
            abs_error = abs(pred - gt)

            seg_details.append({
                'segment_idx': seg_idx,
                'ground_truth': gt,
                'predicted': pred,
                'unit': gt_unit,
                'is_correct': is_correct,
                'absolute_error': abs_error,
                'skipped': False
            })

            all_segments.append({
                'is_correct': is_correct,
                'absolute_error': abs_error
            })

            item_total += 1
            if is_correct:
                item_correct += 1
            item_abs_errors.append(abs_error)

        if item_total > 0 or seg_details:
            item_summaries.append({
                'food_name': food_name,
                'narration_id': narr_id,
                'num_segments': len(segments),
                'num_evaluated': item_total,
                'num_correct': item_correct,
                'accuracy': item_correct / item_total if item_total > 0 else None,
                'mean_abs_error': sum(item_abs_errors) / len(item_abs_errors) if item_abs_errors else None,
                'segments': seg_details
            })

    # Aggregate metrics
    n_total = len(all_segments)
    n_correct = sum(1 for s in all_segments if s['is_correct'])
    total_abs_error = sum(s['absolute_error'] for s in all_segments)

    return {
        'participant': data.get('participant'),
        'model': data.get('model'),
        'source_file': str(results_path),
        'n_segments': n_total,
        'n_correct': n_correct,
        'mean_accuracy': n_correct / n_total if n_total > 0 else None,
        'mean_absolute_error': total_abs_error / n_total if n_total > 0 else None,
        'items': item_summaries
    }


def find_vlm_qa_results(output_dir: Path, tag: str = None) -> list:
    """
    Find all VLM QA result files in output directory.

    Returns list of (results_path, timeline_path) tuples.
    timeline_path is the timeline_annotated.json in the same participant dir (or None).

    Args:
        output_dir: Directory containing participant subdirectories
        tag: Filter by tag (e.g., 'qwen', 'baseline'). If None, find all.
    """
    pattern = f"*_vlm_qa_{tag}_results.json" if tag else "*_vlm_qa_*_results.json"
    results = []

    for participant_dir in sorted(output_dir.iterdir()):
        if not participant_dir.is_dir():
            continue
        # Find timeline_annotated for this participant
        p_name = participant_dir.name
        timeline_path = participant_dir / f"{p_name}_timeline_annotated.json"
        if not timeline_path.exists():
            timeline_path = None

        for f in participant_dir.glob(pattern):
            # Skip evaluation files
            if '_count_eval' in f.name:
                continue
            results.append((f, timeline_path))

    return results


def evaluate_all(output_dir: Path, tag: str = None, model: str = None, report_path: Path = None) -> dict:
    """
    Evaluate all participants and generate consolidated report.

    Args:
        output_dir: Directory containing participant subdirectories
        tag: Filter by tag in filename (e.g., 'qwen_v1'). Preferred over model.
        model: Deprecated, use tag instead. Kept for backwards compatibility.
        report_path: Output path for report JSON
    """
    # Support both tag and model for backwards compatibility
    filter_tag = tag or model
    results_files = find_vlm_qa_results(output_dir, filter_tag)

    if not results_files:
        print(f"No VLM QA results found in {output_dir}" +
              (f" with tag '{filter_tag}'" if filter_tag else ""))
        return None

    print(f"Found {len(results_files)} result files")

    # Evaluate each participant
    participant_results = []
    all_segments = []

    for f, timeline_path in results_files:
        tl_info = f" (timeline: {timeline_path.name})" if timeline_path else " (no timeline)"
        print(f"  Evaluating {f.name}...{tl_info}")
        result = evaluate_participant(f, timeline_path=timeline_path)
        participant_results.append(result)

        # Collect all segments for aggregate metrics
        for item in result['items']:
            for seg in item['segments']:
                if not seg.get('skipped'):
                    all_segments.append({
                        'is_correct': seg['is_correct'],
                        'absolute_error': seg['absolute_error']
                    })

    # Aggregate metrics across all participants
    n_total = len(all_segments)
    n_correct = sum(1 for s in all_segments if s['is_correct'])
    total_abs_error = sum(s['absolute_error'] for s in all_segments)

    report = {
        'generated_at': datetime.now().isoformat(),
        'filter': 'LOW difficulty only',
        'tag': filter_tag or 'all',
        'aggregate': {
            'n_participants': len(participant_results),
            'n_segments': n_total,
            'n_correct': n_correct,
            'mean_accuracy': n_correct / n_total if n_total > 0 else None,
            'mean_absolute_error': total_abs_error / n_total if n_total > 0 else None
        },
        'per_participant': [
            {
                'participant': r['participant'],
                'model': r['model'],
                'n_segments': r['n_segments'],
                'n_correct': r['n_correct'],
                'mean_accuracy': r['mean_accuracy'],
                'mean_absolute_error': r['mean_absolute_error']
            }
            for r in participant_results
        ],
        'details': participant_results
    }

    # Save report
    if report_path is None:
        tag_suffix = f"_{filter_tag}" if filter_tag else ""
        eval_reports_dir = output_dir / "eval_reports"
        eval_reports_dir.mkdir(parents=True, exist_ok=True)
        report_path = eval_reports_dir / f"vlm_qa{tag_suffix}_count_eval_report.json"

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\nSaved: {report_path}")
    return report


def print_report(report: dict):
    """Print human-readable summary of the report."""
    agg = report['aggregate']

    print("\n" + "=" * 70)
    print("VLM QA Count Evaluation Report - LOW Difficulty Items")
    print("=" * 70)
    print(f"Tag: {report.get('tag', report.get('model', 'unknown'))}")
    print(f"Participants: {agg['n_participants']}")
    print("-" * 70)

    # Per-participant summary table
    print(f"\n{'Participant':<12} {'Segments':>10} {'Correct':>10} {'Accuracy':>12} {'MAE':>10}")
    print("-" * 70)

    for p in report['per_participant']:
        acc = f"{p['mean_accuracy']:.1%}" if p['mean_accuracy'] is not None else "N/A"
        mae = f"{p['mean_absolute_error']:.2f}" if p['mean_absolute_error'] is not None else "N/A"
        print(f"{p['participant']:<12} {p['n_segments']:>10} {p['n_correct']:>10} {acc:>12} {mae:>10}")

    print("-" * 70)
    acc = f"{agg['mean_accuracy']:.1%}" if agg['mean_accuracy'] is not None else "N/A"
    mae = f"{agg['mean_absolute_error']:.2f}" if agg['mean_absolute_error'] is not None else "N/A"
    print(f"{'TOTAL':<12} {agg['n_segments']:>10} {agg['n_correct']:>10} {acc:>12} {mae:>10}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate VLM QA count predictions for LOW difficulty items across all participants"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory containing participant results"
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Filter by tag in filename (e.g., 'qwen_v1', 'baseline')"
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output path for report JSON"
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="JSON output only, no summary table"
    )

    args = parser.parse_args()

    if not args.output_dir.exists():
        print(f"Error: Directory not found: {args.output_dir}")
        return 1

    report = evaluate_all(args.output_dir, tag=args.tag, report_path=args.output)

    if report and not args.quiet:
        print_report(report)

    return 0


if __name__ == "__main__":
    exit(main())

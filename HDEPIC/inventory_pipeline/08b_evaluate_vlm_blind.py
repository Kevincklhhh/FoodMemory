#!/usr/bin/env python3
"""
Evaluator for VLM QA Blind Mode results.

Extends count evaluation with blind mode specific metrics:
- Item detection accuracy (detected_item_name vs food_name)
- Category mismatch rate (continuous prediction for discrete GT)

Also analyzes correlation between these factors and count accuracy.

Usage:
    python evaluate_vlm_blind.py --tag qwen_blind_v2
    python evaluate_vlm_blind.py --tag qwen_blind_v2 --all-difficulty
"""

import json
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict

from inventory_utils import DEFAULT_OUTPUT_DIR


def evaluate_segment(seg: dict, food_name: str) -> dict:
    """
    Evaluate a single segment for all mismatch types.

    Returns dict with evaluation results.
    """
    gt = seg.get('ground_truth_count')
    pred = seg.get('predicted_count')
    gt_unit = seg.get('ground_truth_unit', '')
    pred_unit = seg.get('predicted_unit', '')
    detected_item = seg.get('detected_item_name')
    item_match = seg.get('item_match')
    item_similarity = seg.get('item_similarity')
    quantity_category = seg.get('quantity_category', '')
    match_status = seg.get('match', '')

    result = {
        'segment_id': seg.get('segment_id'),
        'segment_idx': seg.get('segment_idx'),
        'ground_truth': gt,
        'predicted': pred,
        'gt_unit': gt_unit,
        'pred_unit': pred_unit,
        'detected_item': detected_item,
        'expected_item': food_name,
        'match_status': match_status,
    }

    # Count accuracy
    if gt is not None and pred is not None:
        result['count_correct'] = (pred == gt)
        result['count_close'] = abs(pred - gt) <= 1
        result['absolute_error'] = abs(pred - gt)
    else:
        result['count_correct'] = None
        result['count_close'] = None
        result['absolute_error'] = None

    # Item match (blind mode)
    if item_match is not None:
        result['item_match'] = item_match
        result['item_similarity'] = item_similarity
    elif detected_item:
        # Fallback: exact string match
        result['item_match'] = detected_item.lower().strip() == food_name.lower().strip()
        result['item_similarity'] = 1.0 if result['item_match'] else 0.0
    else:
        result['item_match'] = None
        result['item_similarity'] = None

    # Category mismatch: VLM says continuous but GT is discrete
    result['category_mismatch'] = (
        quantity_category == 'continuous' and gt is not None
    )

    # Prediction missing (VLM returned null count for discrete GT)
    result['prediction_missing'] = (gt is not None and pred is None)

    return result


def evaluate_participant(results_path: Path, all_difficulty: bool = False) -> dict:
    """
    Evaluate VLM QA blind mode results for a single participant.
    """
    with open(results_path, 'r') as f:
        data = json.load(f)

    items = data.get('items', [])

    # Collect segment-level evaluations
    all_segments = []
    item_summaries = []

    for item in items:
        difficulty = item.get('difficulty', '').upper()
        if not all_difficulty and difficulty != 'LOW':
            continue

        food_name = item.get('food_name', '')
        segments = item.get('segments', [])
        seg_evals = []

        for seg in segments:
            seg_eval = evaluate_segment(seg, food_name)
            seg_eval['difficulty'] = difficulty
            seg_evals.append(seg_eval)
            all_segments.append(seg_eval)

        # Item summary
        countable_segs = [s for s in seg_evals if s['count_correct'] is not None]
        item_summaries.append({
            'food_name': food_name,
            'narration_id': item.get('narration_id'),
            'difficulty': difficulty,
            'num_segments': len(segments),
            'num_countable': len(countable_segs),
            'count_correct': sum(1 for s in countable_segs if s['count_correct']),
            'item_match': sum(1 for s in seg_evals if s['item_match'] is True),
            'item_mismatch': sum(1 for s in seg_evals if s['item_match'] is False),
            'category_mismatch': sum(1 for s in seg_evals if s['category_mismatch']),
            'segments': seg_evals,
        })

    return {
        'participant': data.get('participant'),
        'model': data.get('model'),
        'tag': data.get('tag'),
        'blind_mode': data.get('blind_mode', False),
        'source_file': str(results_path),
        'all_segments': all_segments,
        'items': item_summaries,
    }


def compute_aggregate_metrics(all_segments: list) -> dict:
    """Compute aggregate metrics from all segments."""

    # Filter to countable segments
    countable = [s for s in all_segments if s['count_correct'] is not None]

    # Basic count metrics
    n_total = len(countable)
    n_correct = sum(1 for s in countable if s['count_correct'])
    n_close = sum(1 for s in countable if s['count_close'])
    total_abs_error = sum(s['absolute_error'] for s in countable)

    # Blind mode metrics - all segments
    n_with_item_info = sum(1 for s in all_segments if s['item_match'] is not None)
    n_item_match = sum(1 for s in all_segments if s['item_match'] is True)
    n_item_mismatch = sum(1 for s in all_segments if s['item_match'] is False)

    n_category_mismatch = sum(1 for s in all_segments if s['category_mismatch'])

    n_prediction_missing = sum(1 for s in all_segments if s['prediction_missing'])

    # Correlation analysis: count accuracy conditioned on mismatches
    item_match_correct = sum(1 for s in countable if s['item_match'] is True and s['count_correct'])
    item_match_total = sum(1 for s in countable if s['item_match'] is True)

    item_mismatch_correct = sum(1 for s in countable if s['item_match'] is False and s['count_correct'])
    item_mismatch_total = sum(1 for s in countable if s['item_match'] is False)

    return {
        # Count accuracy
        'n_segments_total': len(all_segments),
        'n_segments_countable': n_total,
        'n_count_correct': n_correct,
        'n_count_close': n_close,
        'count_accuracy': n_correct / n_total if n_total > 0 else None,
        'count_close_rate': n_close / n_total if n_total > 0 else None,
        'mean_absolute_error': total_abs_error / n_total if n_total > 0 else None,

        # Item detection (blind mode)
        'n_with_item_detection': n_with_item_info,
        'n_item_match': n_item_match,
        'n_item_mismatch': n_item_mismatch,
        'item_match_rate': n_item_match / n_with_item_info if n_with_item_info > 0 else None,
        'item_mismatch_rate': n_item_mismatch / n_with_item_info if n_with_item_info > 0 else None,

        # Category mismatch
        'n_category_mismatch': n_category_mismatch,
        'category_mismatch_rate': n_category_mismatch / len(all_segments) if len(all_segments) > 0 else None,

        # Prediction missing
        'n_prediction_missing': n_prediction_missing,
        'prediction_missing_rate': n_prediction_missing / len(all_segments) if len(all_segments) > 0 else None,

        # Correlation: count accuracy given item match/mismatch
        'count_accuracy_given_item_match': item_match_correct / item_match_total if item_match_total > 0 else None,
        'count_accuracy_given_item_mismatch': item_mismatch_correct / item_mismatch_total if item_mismatch_total > 0 else None,
        'n_item_match_countable': item_match_total,
        'n_item_mismatch_countable': item_mismatch_total,
    }


def find_vlm_qa_results(output_dir: Path, tag: str = None) -> list:
    """Find all VLM QA result files in output directory."""
    pattern = f"*_vlm_qa_{tag}_results.json" if tag else "*_vlm_qa_*_results.json"
    results = []

    for participant_dir in sorted(output_dir.iterdir()):
        if not participant_dir.is_dir() or not participant_dir.name.startswith('P'):
            continue
        for f in participant_dir.glob(pattern):
            if '_count_eval' in f.name or '_blind_eval' in f.name:
                continue
            results.append(f)

    return results


def evaluate_all(output_dir: Path, tag: str, all_difficulty: bool = False, report_path: Path = None) -> dict:
    """Evaluate all participants and generate consolidated report."""

    results_files = find_vlm_qa_results(output_dir, tag)

    if not results_files:
        print(f"No VLM QA results found with tag '{tag}'")
        return None

    print(f"Found {len(results_files)} result files")

    # Evaluate each participant
    participant_results = []
    all_segments = []

    for f in results_files:
        print(f"  Evaluating {f.name}...")
        result = evaluate_participant(f, all_difficulty)
        participant_results.append(result)
        all_segments.extend(result['all_segments'])

    # Aggregate metrics
    aggregate = compute_aggregate_metrics(all_segments)

    # Per-participant summary
    per_participant = []
    for r in participant_results:
        p_metrics = compute_aggregate_metrics(r['all_segments'])
        per_participant.append({
            'participant': r['participant'],
            'model': r['model'],
            'tag': r['tag'],
            'blind_mode': r['blind_mode'],
            **{k: v for k, v in p_metrics.items()}
        })

    report = {
        'generated_at': datetime.now().isoformat(),
        'tag': tag,
        'filter': 'All difficulties' if all_difficulty else 'LOW difficulty only',
        'aggregate': aggregate,
        'per_participant': per_participant,
    }

    # Save report
    if report_path is None:
        difficulty_suffix = '_all' if all_difficulty else ''
        eval_reports_dir = output_dir / "eval_reports"
        eval_reports_dir.mkdir(parents=True, exist_ok=True)
        report_path = eval_reports_dir / f"vlm_qa_{tag}{difficulty_suffix}_blind_eval_report.json"

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\nSaved: {report_path}")
    return report


def print_report(report: dict):
    """Print human-readable summary of the report."""
    agg = report['aggregate']

    print("\n" + "=" * 80)
    print("VLM QA Blind Mode Evaluation Report")
    print("=" * 80)
    print(f"Tag: {report.get('tag')}")
    print(f"Filter: {report.get('filter')}")
    print(f"Participants: {len(report['per_participant'])}")

    # Overall metrics
    print("\n" + "-" * 80)
    print("AGGREGATE METRICS")
    print("-" * 80)

    print(f"\n{'Metric':<40} {'Value':>15} {'Count':>15}")
    print("-" * 70)

    # Count accuracy
    acc = f"{agg['count_accuracy']:.1%}" if agg['count_accuracy'] is not None else "N/A"
    print(f"{'Count Accuracy (exact)':<40} {acc:>15} {agg['n_count_correct']:>10}/{agg['n_segments_countable']}")

    close = f"{agg['count_close_rate']:.1%}" if agg['count_close_rate'] is not None else "N/A"
    print(f"{'Count Close Rate (±1)':<40} {close:>15} {agg['n_count_close']:>10}/{agg['n_segments_countable']}")

    mae = f"{agg['mean_absolute_error']:.2f}" if agg['mean_absolute_error'] is not None else "N/A"
    print(f"{'Mean Absolute Error':<40} {mae:>15}")

    # Blind mode metrics
    print("\n--- Blind Mode Metrics ---")

    item_rate = f"{agg['item_match_rate']:.1%}" if agg['item_match_rate'] is not None else "N/A"
    print(f"{'Item Detection Match Rate':<40} {item_rate:>15} {agg['n_item_match']:>10}/{agg['n_with_item_detection']}")

    item_mis = f"{agg['item_mismatch_rate']:.1%}" if agg['item_mismatch_rate'] is not None else "N/A"
    print(f"{'Item Detection Mismatch Rate':<40} {item_mis:>15} {agg['n_item_mismatch']:>10}/{agg['n_with_item_detection']}")

    cat_mis = f"{agg['category_mismatch_rate']:.1%}" if agg['category_mismatch_rate'] is not None else "N/A"
    print(f"{'Category Mismatch Rate (cont. vs disc.)':<40} {cat_mis:>15} {agg['n_category_mismatch']:>10}/{agg['n_segments_total']}")

    pred_mis = f"{agg['prediction_missing_rate']:.1%}" if agg['prediction_missing_rate'] is not None else "N/A"
    print(f"{'Prediction Missing Rate':<40} {pred_mis:>15} {agg['n_prediction_missing']:>10}/{agg['n_segments_total']}")

    # Correlation analysis
    print("\n--- Correlation: Count Accuracy by Condition ---")

    acc_item_match = f"{agg['count_accuracy_given_item_match']:.1%}" if agg['count_accuracy_given_item_match'] is not None else "N/A"
    print(f"{'Count Accuracy | Item Match':<40} {acc_item_match:>15} (n={agg['n_item_match_countable']})")

    acc_item_mis = f"{agg['count_accuracy_given_item_mismatch']:.1%}" if agg['count_accuracy_given_item_mismatch'] is not None else "N/A"
    print(f"{'Count Accuracy | Item Mismatch':<40} {acc_item_mis:>15} (n={agg['n_item_mismatch_countable']})")

    # Per-participant table
    print("\n" + "-" * 80)
    print("PER-PARTICIPANT SUMMARY")
    print("-" * 80)
    print(f"\n{'Participant':<8} {'Segs':>6} {'Count':>8} {'Item':>8} {'Cat':>6} {'Miss':>6}")
    print(f"{'':8} {'':>6} {'Acc':>8} {'Match':>8} {'Mis':>6} {'Pred':>6}")
    print("-" * 70)

    for p in report['per_participant']:
        acc = f"{p['count_accuracy']:.0%}" if p['count_accuracy'] is not None else "N/A"
        item = f"{p['item_match_rate']:.0%}" if p['item_match_rate'] is not None else "N/A"
        cat = f"{p['n_category_mismatch']}"
        miss = f"{p['n_prediction_missing']}"
        print(f"{p['participant']:<8} {p['n_segments_total']:>6} {acc:>8} {item:>8} {cat:>6} {miss:>6}")

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate VLM QA blind mode results with mismatch analysis"
    )
    parser.add_argument(
        "--tag",
        type=str,
        required=True,
        help="Tag to filter result files (e.g., 'qwen_blind_v2')"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory containing participant results"
    )
    parser.add_argument(
        "--all-difficulty",
        action="store_true",
        help="Include all difficulty levels (default: LOW only)"
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

    report = evaluate_all(
        args.output_dir,
        tag=args.tag,
        all_difficulty=args.all_difficulty,
        report_path=args.output
    )

    if report and not args.quiet:
        print_report(report)

    return 0


if __name__ == "__main__":
    exit(main())

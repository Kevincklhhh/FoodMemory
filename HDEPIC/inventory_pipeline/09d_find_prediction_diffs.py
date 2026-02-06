#!/usr/bin/env python3
"""
Find segments where predictions differ between two VLM result tags.

Generates a failure-cases-compatible JSON (schema_version 2) containing
segments where tag_a and tag_b disagree on predicted_count.

Usage:
    # Find where hybrid_qwen differs from qwen_promptv1
    python find_prediction_diffs.py --tag-a qwen_promptv1 --tag-b hybrid_qwen

    # Only show segments where one is correct and the other is wrong
    python find_prediction_diffs.py --tag-a qwen_promptv1 --tag-b hybrid_qwen --flip-only

    # Filter to specific participant
    python find_prediction_diffs.py --tag-a qwen_promptv1 --tag-b hybrid_qwen --participant P03
"""

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

from inventory_utils import DEFAULT_OUTPUT_DIR


def load_segments(output_dir: Path, tag: str, participant: str = None):
    """Load all LOW-difficulty segments from VLM result files for a given tag.

    Returns dict: segment_id -> {participant, narration_id, food_name, predicted_count,
                                  ground_truth_count, match, ...}
    """
    segments = {}

    if participant:
        participants = [participant]
    else:
        participants = sorted(
            p.name for p in output_dir.iterdir()
            if p.is_dir() and p.name.startswith('P')
        )

    for pid in participants:
        result_file = output_dir / pid / f"{pid}_vlm_qa_{tag}_results.json"
        if not result_file.exists():
            continue

        data = json.load(open(result_file))
        for item in data.get('items', []):
            if item.get('difficulty', '').upper() != 'LOW':
                continue
            for seg in item.get('segments', []):
                sid = seg.get('segment_id')
                if not sid:
                    continue
                segments[sid] = {
                    'participant': pid,
                    'narration_id': item.get('narration_id'),
                    'food_name': item.get('food_name'),
                    'difficulty': item.get('difficulty'),
                    'segment_id': sid,
                    'predicted_count': seg.get('predicted_count'),
                    'ground_truth_count': seg.get('ground_truth_count'),
                    'match': seg.get('match'),
                }

    return segments


def main():
    parser = argparse.ArgumentParser(
        description="Find segments with differing predictions between two VLM tags"
    )
    parser.add_argument('--tag-a', required=True, help='First VLM result tag (baseline)')
    parser.add_argument('--tag-b', required=True, help='Second VLM result tag (comparison)')
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--participant', help='Filter to single participant')
    parser.add_argument('--flip-only', action='store_true',
                        help='Only include segments where correctness flipped (one exact, other wrong)')
    parser.add_argument('--output', type=Path, help='Output JSON path (default: auto-generated)')
    args = parser.parse_args()

    segs_a = load_segments(args.output_dir, args.tag_a, args.participant)
    segs_b = load_segments(args.output_dir, args.tag_b, args.participant)

    print(f"Tag A ({args.tag_a}): {len(segs_a)} LOW segments")
    print(f"Tag B ({args.tag_b}): {len(segs_b)} LOW segments")

    common_ids = sorted(set(segs_a.keys()) & set(segs_b.keys()))
    print(f"Common segments: {len(common_ids)}")

    # Find diffs
    diffs = []
    stats = Counter()

    for sid in common_ids:
        a = segs_a[sid]
        b = segs_b[sid]

        pred_a = a['predicted_count']
        pred_b = b['predicted_count']
        gt = a['ground_truth_count']
        match_a = a['match']
        match_b = b['match']

        # Skip if both predictions are identical
        if pred_a == pred_b:
            stats['same'] += 1
            continue

        # Skip if both are None
        if pred_a is None and pred_b is None:
            stats['both_null'] += 1
            continue

        # Categorize the diff
        a_correct = match_a == 'exact'
        b_correct = match_b == 'exact'

        if a_correct and not b_correct:
            category = 'a_correct_b_wrong'
        elif b_correct and not a_correct:
            category = 'b_correct_a_wrong'
        elif a_correct and b_correct:
            category = 'both_correct_diff_pred'  # shouldn't happen for exact
        else:
            category = 'both_wrong_diff_pred'

        stats[category] += 1

        if args.flip_only and category not in ('a_correct_b_wrong', 'b_correct_a_wrong'):
            continue

        diffs.append({
            'segment_id': sid,
            'participant': a['participant'],
            'narration_id': a['narration_id'],
            'food_name': a['food_name'],
            'ground_truth': gt,
            'pred_a': pred_a,
            'match_a': match_a,
            'pred_b': pred_b,
            'match_b': match_b,
            'category': category,
        })

    # Sort: flips first (b_correct_a_wrong, then a_correct_b_wrong), then by participant
    cat_order = {'b_correct_a_wrong': 0, 'a_correct_b_wrong': 1, 'both_wrong_diff_pred': 2, 'both_correct_diff_pred': 3}
    diffs.sort(key=lambda d: (cat_order.get(d['category'], 9), d['participant'], d['food_name']))

    # Print summary
    print(f"\n{'='*70}")
    print(f"DIFF SUMMARY: {args.tag_a} vs {args.tag_b}")
    print(f"{'='*70}")
    print(f"  Same prediction:        {stats['same']:>5}")
    print(f"  A correct, B wrong:     {stats['a_correct_b_wrong']:>5}")
    print(f"  B correct, A wrong:     {stats['b_correct_a_wrong']:>5}")
    print(f"  Both wrong, diff pred:  {stats['both_wrong_diff_pred']:>5}")
    total_diff = len(diffs) if not args.flip_only else stats['a_correct_b_wrong'] + stats['b_correct_a_wrong']
    print(f"  Total diffs:            {sum(v for k, v in stats.items() if k != 'same' and k != 'both_null'):>5}")

    # Print table
    if diffs:
        print(f"\n{'PID':<6} {'Food':<30} {'GT':>4} {'PredA':>6} {'MatchA':<8} {'PredB':>6} {'MatchB':<8} {'Category'}")
        print('-' * 95)
        for d in diffs:
            print(f"{d['participant']:<6} {d['food_name'][:29]:<30} "
                  f"{str(d['ground_truth']):>4} "
                  f"{str(d['pred_a']):>6} {(d['match_a'] or '-'):<8} "
                  f"{str(d['pred_b']):>6} {(d['match_b'] or '-'):<8} "
                  f"{d['category']}")

    # Generate failure-cases-compatible output
    output_path = args.output
    if not output_path:
        suffix = 'flips' if args.flip_only else 'diffs'
        if args.flip_only:
            # Flip analysis goes to prediction_analysis/
            subdir = args.output_dir / "prediction_analysis"
            output_path = subdir / f"prediction_flips_{args.tag_a}_vs_{args.tag_b}.json"
        else:
            # Diff-based failure cases go to failure_cases/
            subdir = args.output_dir / "failure_cases"
            output_path = subdir / f"failure_cases_diffs_{args.tag_a}_vs_{args.tag_b}.json"
        subdir.mkdir(parents=True, exist_ok=True)

    cases = []
    for i, d in enumerate(diffs, 1):
        cases.append({
            'case_id': f"DIFF{i:03d}",
            'participant': d['participant'],
            'narration_id': d['narration_id'],
            'segment_id': d['segment_id'],
            'include': True,
            'priority': 1 if d['category'] in ('b_correct_a_wrong', 'a_correct_b_wrong') else 0,
            'notes': f"GT={d['ground_truth']} A({args.tag_a})={d['pred_a']}({d['match_a']}) B({args.tag_b})={d['pred_b']}({d['match_b']})",
            'tags': [d['category']],
        })

    output_data = {
        'name': f"diffs_{args.tag_a}_vs_{args.tag_b}",
        'schema_version': 2,
        'vlm_tag': args.tag_b,
        'created_at': datetime.now().isoformat(),
        'created_from': None,
        'description': f"Segments where {args.tag_a} and {args.tag_b} disagree on predicted_count",
        'tag_a': args.tag_a,
        'tag_b': args.tag_b,
        'total_cases': len(cases),
        'stats': dict(stats),
        'cases': cases,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nSaved {len(cases)} cases to {output_path}")
    return 0


if __name__ == '__main__':
    exit(main())

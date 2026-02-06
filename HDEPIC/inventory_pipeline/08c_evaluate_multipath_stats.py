#!/usr/bin/env python3
"""
Multipath evidence analysis - statistics on path usage, validity, and accuracy.

Analyzes VLM multipath results to show:
- How often each path (source/destination/transfer) is VALID vs INVALID
- Which path is selected by the synthesis step
- Accuracy (exact/close/wrong) broken down by selected path
- Confidence distribution per path

Usage:
    python evaluate_multipath_stats.py
    python evaluate_multipath_stats.py --tag qwen_multipath
    python evaluate_multipath_stats.py --participant P03
"""

import json
import argparse
from collections import Counter, defaultdict
from pathlib import Path

from inventory_utils import DEFAULT_OUTPUT_DIR


def collect_segments(output_dir: Path, tag: str, participant: str = None):
    """Collect all multipath segments from result files."""
    segments = []

    if participant:
        participants = [participant]
    else:
        participants = sorted(
            p.name for p in output_dir.iterdir()
            if p.is_dir() and p.name.startswith('P')
        )

    for pid in participants:
        pattern = f"{pid}_vlm_qa_{tag}_results.json"
        result_file = output_dir / pid / pattern
        if not result_file.exists():
            continue

        data = json.load(open(result_file))
        if data.get('prompt_mode') != 'multipath':
            continue

        for item in data.get('items', []):
            difficulty = item.get('difficulty', '').upper()
            for seg in item.get('segments', []):
                if not seg.get('paths') or not seg.get('final_synthesis'):
                    continue
                segments.append({
                    'participant': pid,
                    'food_name': item.get('food_name'),
                    'difficulty': difficulty,
                    'segment_id': seg.get('segment_id'),
                    'paths': seg['paths'],
                    'synthesis': seg['final_synthesis'],
                    'predicted_count': seg.get('predicted_count'),
                    'ground_truth_count': seg.get('ground_truth_count'),
                    'match': seg.get('match'),
                })

    return segments


def print_report(segments, title="ALL"):
    """Print statistics for a set of segments."""
    n = len(segments)
    if n == 0:
        print(f"  No segments found.\n")
        return

    # --- Path validity ---
    path_names = ['source', 'destination', 'transfer']
    validity = {p: Counter() for p in path_names}
    confidence = {p: Counter() for p in path_names}

    for seg in segments:
        for pname in path_names:
            path = seg['paths'].get(pname, {})
            status = path.get('status', 'UNKNOWN')
            is_valid = status == 'VALID'
            validity[pname]['VALID' if is_valid else status] += 1
            if is_valid:
                confidence[pname][path.get('confidence', 'unknown')] += 1

    print(f"  Path Validity ({n} segments):")
    print(f"    {'Path':<14} {'VALID':>6} {'INVALID':>8} {'% Valid':>8}")
    print(f"    {'-'*38}")
    for pname in path_names:
        v = validity[pname]['VALID']
        inv = n - v
        pct = 100 * v / n
        print(f"    {pname:<14} {v:>6} {inv:>8} {pct:>7.1f}%")

    # --- Path validity detail ---
    print(f"\n  Invalid Reason Breakdown:")
    for pname in path_names:
        reasons = {k: v for k, v in validity[pname].items() if k != 'VALID'}
        if reasons:
            parts = ', '.join(f"{k}={v}" for k, v in sorted(reasons.items(), key=lambda x: -x[1]))
            print(f"    {pname:<14} {parts}")

    # --- Confidence for VALID paths ---
    print(f"\n  Confidence (VALID paths only):")
    print(f"    {'Path':<14} {'high':>6} {'medium':>8} {'low':>6}")
    print(f"    {'-'*36}")
    for pname in path_names:
        c = confidence[pname]
        print(f"    {pname:<14} {c.get('high',0):>6} {c.get('medium',0):>8} {c.get('low',0):>6}")

    # --- Selected path ---
    selected = Counter()
    for seg in segments:
        best = seg['synthesis'].get('best_path_selected', 'unknown')
        selected[best] += 1

    print(f"\n  Selected Path (synthesis best_path_selected):")
    for pname in path_names + sorted(set(selected.keys()) - set(path_names)):
        if selected[pname]:
            pct = 100 * selected[pname] / n
            print(f"    {pname:<14} {selected[pname]:>6}  ({pct:.1f}%)")

    # --- Accuracy by selected path ---
    match_by_path = defaultdict(Counter)
    for seg in segments:
        best = seg['synthesis'].get('best_path_selected', 'unknown')
        m = seg.get('match')
        if m:
            match_by_path[best][m] += 1
        elif seg.get('ground_truth_count') is None:
            match_by_path[best]['no_gt'] += 1
        else:
            match_by_path[best]['no_pred'] += 1

    print(f"\n  Accuracy by Selected Path:")
    print(f"    {'Path':<14} {'exact':>6} {'close':>6} {'wrong':>6} {'no_gt':>6} {'total':>6} {'acc%':>7}")
    print(f"    {'-'*54}")
    for pname in path_names + sorted(set(match_by_path.keys()) - set(path_names)):
        mc = match_by_path.get(pname, Counter())
        if not mc:
            continue
        exact = mc.get('exact', 0)
        close = mc.get('close', 0)
        wrong = mc.get('wrong', 0)
        no_gt = mc.get('no_gt', 0)
        total = sum(mc.values())
        evaluated = exact + close + wrong
        acc = 100 * exact / evaluated if evaluated > 0 else 0
        print(f"    {pname:<14} {exact:>6} {close:>6} {wrong:>6} {no_gt:>6} {total:>6} {acc:>6.1f}%")

    # --- Overall accuracy ---
    match_overall = Counter()
    for seg in segments:
        m = seg.get('match')
        if m:
            match_overall[m] += 1
    evaluated = sum(match_overall.values())
    if evaluated > 0:
        exact = match_overall.get('exact', 0)
        close = match_overall.get('close', 0)
        wrong = match_overall.get('wrong', 0)
        print(f"\n  Overall ({evaluated} evaluated):")
        print(f"    exact={exact} ({100*exact/evaluated:.1f}%)  "
              f"close={close} ({100*close/evaluated:.1f}%)  "
              f"wrong={wrong} ({100*wrong/evaluated:.1f}%)")

    # --- Number of valid paths per segment ---
    valid_counts = Counter()
    for seg in segments:
        nv = sum(
            1 for pname in path_names
            if seg['paths'].get(pname, {}).get('status') == 'VALID'
        )
        valid_counts[nv] += 1

    print(f"\n  Valid Paths per Segment:")
    for nv in sorted(valid_counts.keys()):
        print(f"    {nv} valid: {valid_counts[nv]:>6}  ({100*valid_counts[nv]/n:.1f}%)")

    print()


def main():
    parser = argparse.ArgumentParser(description="Multipath evidence statistics")
    parser.add_argument('--tag', default='qwen_multipath', help='VLM result tag')
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--participant', help='Single participant (e.g. P03)')
    parser.add_argument('--low-only', action='store_true', help='Only LOW difficulty')
    args = parser.parse_args()

    segments = collect_segments(args.output_dir, args.tag, args.participant)
    if args.low_only:
        segments = [s for s in segments if s['difficulty'] == 'LOW']

    print(f"=" * 60)
    print(f"Multipath Statistics: tag={args.tag}")
    print(f"{'LOW only' if args.low_only else 'All difficulties'}")
    print(f"=" * 60)

    # Overall
    print(f"\n[ALL] ({len(segments)} segments)")
    print_report(segments)

    # Per difficulty
    difficulties = sorted(set(s['difficulty'] for s in segments))
    if len(difficulties) > 1:
        for diff in difficulties:
            subset = [s for s in segments if s['difficulty'] == diff]
            print(f"[{diff}] ({len(subset)} segments)")
            print_report(subset)

    # Per participant
    participants = sorted(set(s['participant'] for s in segments))
    if len(participants) > 1:
        print(f"{'='*60}")
        print("Per-Participant Summary")
        print(f"{'='*60}")
        print(f"  {'PID':<6} {'Segs':>5} {'Exact':>6} {'Close':>6} {'Wrong':>6} {'Acc%':>6}  {'Src%':>5} {'Dst%':>5} {'Xfr%':>5}")
        print(f"  {'-'*62}")
        for pid in participants:
            subset = [s for s in segments if s['participant'] == pid]
            ns = len(subset)
            mc = Counter(s.get('match') for s in subset if s.get('match'))
            exact = mc.get('exact', 0)
            close = mc.get('close', 0)
            wrong = mc.get('wrong', 0)
            evald = exact + close + wrong
            acc = 100 * exact / evald if evald > 0 else 0

            sel = Counter(s['synthesis'].get('best_path_selected') for s in subset)
            src_pct = 100 * sel.get('source', 0) / ns
            dst_pct = 100 * sel.get('destination', 0) / ns
            xfr_pct = 100 * sel.get('transfer', 0) / ns
            print(f"  {pid:<6} {ns:>5} {exact:>6} {close:>6} {wrong:>6} {acc:>5.1f}%  {src_pct:>4.0f}% {dst_pct:>4.0f}% {xfr_pct:>4.0f}%")

    return 0


if __name__ == '__main__':
    exit(main())

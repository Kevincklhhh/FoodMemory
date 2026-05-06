#!/usr/bin/env python3
"""
04_evaluate_adatad_segments.py — Evaluate AdaTAD temporal proposals against
actions.json ground-truth using mAP @ tIoU.

AdaTAD detections are per-clip; this script converts them to session-relative
timestamps, applies score thresholding and NMS, then computes:
  - Class-agnostic mAP@tIoU (all GT = "activity", all dets = "activity")
  - Per-tIoU recall (what fraction of GT segments are covered?)
  - AR@AN (average recall vs average number of proposals)

Usage:
    python 04_evaluate_adatad_segments.py --participant kailai \
        --score-threshold 0.3 --nms-theta 0.1
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    get_session_clips,
    get_sessions,
    load_actions,
    load_ledger,
    outputs_dir,
    instance_id_to_visual_class,
)


# ── helpers ──────────────────────────────────────────────────────────────────


def temporal_iou(pred, gt):
    """Compute temporal IoU between two segments [start, end]."""
    inter_start = max(pred[0], gt[0])
    inter_end = min(pred[1], gt[1])
    inter = max(0.0, inter_end - inter_start)
    union = (pred[1] - pred[0]) + (gt[1] - gt[0]) - inter
    return inter / union if union > 0 else 0.0


def nms_temporal(detections, theta):
    """Soft-NMS style 1D NMS on temporal detections.

    detections: list of {segment: [s, e], score: float, ...}
    theta: IoU threshold — suppress lower-scored dets overlapping above theta.
    Returns filtered list.
    """
    if not detections:
        return []
    dets = sorted(detections, key=lambda d: d["score"], reverse=True)
    keep = []
    for d in dets:
        suppress = False
        for k in keep:
            if temporal_iou(d["segment"], k["segment"]) >= theta:
                suppress = True
                break
        if not suppress:
            keep.append(d)
    return keep


def compute_ap(precision, recall):
    """Compute AP using 101-point interpolation (COCO-style)."""
    # Append sentinel values
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    # Make precision monotonically decreasing
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    # 101-point interpolation
    t = np.linspace(0, 1, 101)
    ap = 0.0
    for thresh in t:
        p = mpre[mrec >= thresh]
        ap += p.max() if len(p) > 0 else 0.0
    return ap / 101.0


def compute_ap_at_tiou(predictions, ground_truths, tiou_threshold):
    """Compute class-agnostic AP at a single tIoU threshold.

    predictions: list of {segment: [s, e], score: float, session: str}
        sorted by score descending.
    ground_truths: list of {segment: [s, e], session: str}

    Returns: AP, per-prediction tp/fp arrays, recall values.
    """
    npos = len(ground_truths)
    if npos == 0:
        return 0.0, 0, 0

    # Index GT by session for efficiency
    gt_by_session = defaultdict(list)
    for i, gt in enumerate(ground_truths):
        gt_by_session[gt["session"]].append((i, gt))

    tp = np.zeros(len(predictions))
    fp = np.zeros(len(predictions))
    matched_gt = set()

    for i, pred in enumerate(predictions):
        best_iou = 0.0
        best_gt_idx = -1
        for gt_idx, gt in gt_by_session.get(pred["session"], []):
            iou = temporal_iou(pred["segment"], gt["segment"])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        # tiou_threshold=0 means "any intersection" (iou > 0)
        iou_ok = best_iou > 0 if tiou_threshold == 0 else best_iou >= tiou_threshold
        if iou_ok and best_gt_idx not in matched_gt:
            tp[i] = 1
            matched_gt.add(best_gt_idx)
        else:
            fp[i] = 1

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recall = tp_cum / npos
    precision = tp_cum / (tp_cum + fp_cum)

    ap = compute_ap(precision, recall)
    return ap, recall[-1] if len(recall) > 0 else 0.0, npos


# ── main ─────────────────────────────────────────────────────────────────────


def load_adatad_session(participant, session):
    """Load AdaTAD detections, convert clip-relative → session-relative timestamps."""
    det_path = outputs_dir(participant, session) / "adatad_detections.json"
    if not det_path.exists():
        return {"noun": [], "verb": []}

    raw = json.loads(det_path.read_text())
    clips = get_session_clips(participant, session)
    # Build clip_name → cumulative offset
    clip_offsets = {}
    offset = 0.0
    for fname, _, dur in clips:
        clip_name = fname.replace(".mp4", "")
        clip_offsets[clip_name] = offset
        offset += dur

    result = {}
    for task in ["noun", "verb"]:
        session_dets = []
        for clip_name, dets in raw.get(task, {}).items():
            off = clip_offsets.get(clip_name, 0.0)
            for d in dets:
                session_dets.append({
                    "segment": [d["segment"][0] + off, d["segment"][1] + off],
                    "label": d["label"],
                    "score": d["score"],
                })
        result[task] = session_dets
    return result


def main():
    parser = argparse.ArgumentParser(description="Evaluate AdaTAD segments vs actions.json GT")
    parser.add_argument("--participant", default="kailai")
    parser.add_argument("--score-threshold", type=float, default=0.3)
    parser.add_argument("--nms-theta", type=float, default=0.1,
                        help="NMS IoU threshold (0=no NMS, 1=keep all)")
    parser.add_argument("--task", default="noun", choices=["noun", "verb", "both"],
                        help="Which AdaTAD task to use as proposals")
    parser.add_argument("--tiou-thresholds", type=float, nargs="+",
                        default=[0.0, 0.1, 0.25, 0.5, 0.75],
                        help="tIoU thresholds for mAP computation (0.0 = any intersection)")
    parser.add_argument("--sessions", nargs="+", default=None,
                        help="Specific sessions (default: all with both detections and GT)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    ledger = load_ledger(args.participant)
    iid_to_vc = instance_id_to_visual_class(ledger)
    all_sessions = args.sessions or get_sessions(args.participant)

    # ── Collect all predictions and GT ───────────────────────────────────
    all_preds = []  # {segment, score, session, label}
    all_gt = []     # {segment, session, item, visual_class}
    session_stats = []

    tasks = ["noun", "verb"] if args.task == "both" else [args.task]

    for session in all_sessions:
        # Load GT
        ann_path = Path(f"participants/{args.participant}/annotations/{session}/actions.json")
        det_path = outputs_dir(args.participant, session) / "adatad_detections.json"
        if not ann_path.exists() or not det_path.exists():
            continue

        actions = load_actions(args.participant, session)
        gt_segments = []
        for a in actions:
            gt_segments.append({
                "segment": [float(a["start"]), float(a["end"])],
                "session": session,
                "item": a["item"],
                "visual_class": iid_to_vc.get(a["item"], a["item"]),
                "action": a.get("action", ""),
                "stage": a.get("stage", ""),
            })
        all_gt.extend(gt_segments)

        # Load AdaTAD
        adatad = load_adatad_session(args.participant, session)
        session_dets = []
        for task in tasks:
            session_dets.extend(adatad.get(task, []))

        # Filter by score
        session_dets = [d for d in session_dets if d["score"] >= args.score_threshold]

        # NMS
        if args.nms_theta < 1.0:
            session_dets = nms_temporal(session_dets, args.nms_theta)

        for d in session_dets:
            d["session"] = session
        all_preds.extend(session_dets)

        session_stats.append({
            "session": session,
            "n_gt": len(gt_segments),
            "n_preds_after_filter": len(session_dets),
        })

    # Sort predictions by score (descending) for AP computation
    all_preds.sort(key=lambda d: d["score"], reverse=True)

    print(f"{'='*70}")
    print(f"AdaTAD Segment Evaluation — mAP @ tIoU")
    print(f"{'='*70}")
    print(f"Participant:     {args.participant}")
    print(f"Task:            {args.task}")
    print(f"Score threshold: {args.score_threshold}")
    print(f"NMS theta:       {args.nms_theta}")
    print(f"Sessions:        {len(session_stats)}")
    print(f"GT segments:     {len(all_gt)}")
    print(f"Predictions:     {len(all_preds)}")
    print()

    # ── Per-session summary ──────────────────────────────────────────────
    if args.verbose:
        print("Per-session breakdown:")
        for s in session_stats:
            print(f"  {s['session']}: {s['n_gt']} GT, {s['n_preds_after_filter']} preds")
        print()

    # ── Class-agnostic mAP @ tIoU ────────────────────────────────────────
    print(f"{'tIoU':>8}  {'AP':>8}  {'Recall':>8}")
    print(f"{'-'*8}  {'-'*8}  {'-'*8}")
    aps = []
    for tiou in args.tiou_thresholds:
        ap, recall, npos = compute_ap_at_tiou(all_preds, all_gt, tiou)
        aps.append(ap)
        print(f"{tiou:8.2f}  {ap:8.4f}  {recall:8.4f}")

    mean_ap = np.mean(aps)
    print(f"{'mAP':>8}  {mean_ap:8.4f}")
    print()

    # ── Breakdown by GT stage ────────────────────────────────────────────
    stages = sorted(set(g["stage"] for g in all_gt if g["stage"]))
    if stages:
        print(f"{'Stage':<14} {'N':>4}", end="")
        for tiou in args.tiou_thresholds:
            label = "any" if tiou == 0 else f"{tiou}"
            print(f"  {'R@'+label:>8}", end="")
        print(f"  {'mean_dur':>8}")
        print("-" * (24 + 10 * len(args.tiou_thresholds) + 10))

        for stage in stages + ["(all)"]:
            gt_sub = [g for g in all_gt if g["stage"] == stage] if stage != "(all)" else all_gt
            if not gt_sub:
                continue
            mean_dur = np.mean([g["segment"][1] - g["segment"][0] for g in gt_sub])
            print(f"{stage:<14} {len(gt_sub):>4}", end="")
            for tiou in args.tiou_thresholds:
                _, recall, _ = compute_ap_at_tiou(all_preds, gt_sub, tiou)
                print(f"  {recall:8.1%}", end="")
            print(f"  {mean_dur:7.1f}s")
        print()

    # ── Recall @ fixed number of proposals ───────────────────────────────
    for tiou_label, tiou_val in [("any intersection", 0), ("tIoU≥0.5", 0.5)]:
        print(f"Recall @ N proposals (class-agnostic, {tiou_label}):")
        for max_n in [10, 25, 50, 100, 200, 500, len(all_preds)]:
            if max_n > len(all_preds):
                continue
            subset = all_preds[:max_n]
            matched = 0
            for gt in all_gt:
                for pred in subset:
                    if pred["session"] != gt["session"]:
                        continue
                    iou = temporal_iou(pred["segment"], gt["segment"])
                    hit = iou > 0 if tiou_val == 0 else iou >= tiou_val
                    if hit:
                        matched += 1
                        break
            r = matched / len(all_gt) if all_gt else 0
            label = f"all ({len(all_preds)})" if max_n == len(all_preds) else str(max_n)
            print(f"  N={label:>6}: recall={r:.4f} ({matched}/{len(all_gt)})")
        print()
    print()

    # ── Per-GT segment detail (verbose) ──────────────────────────────────
    if args.verbose:
        print("Per-GT segment matching (tIoU=0.5):")
        for gt in all_gt:
            best_iou = 0.0
            best_pred = None
            for pred in all_preds:
                if pred["session"] != gt["session"]:
                    continue
                iou = temporal_iou(pred["segment"], gt["segment"])
                if iou > best_iou:
                    best_iou = iou
                    best_pred = pred
            status = "HIT " if best_iou >= 0.5 else "MISS"
            gt_dur = gt["segment"][1] - gt["segment"][0]
            detail = f"best_iou={best_iou:.3f}"
            if best_pred:
                detail += f" label={best_pred['label']} score={best_pred['score']:.3f}"
            print(f"  {status} [{gt['session']}] {gt['visual_class']:40s} "
                  f"[{gt['segment'][0]:6.1f}-{gt['segment'][1]:6.1f}] ({gt_dur:4.1f}s) {detail}")

    # ── Label distribution of matched preds ──────────────────────────────
    print("AdaTAD noun labels that match GT segments (tIoU≥0.1):")
    label_match_counts = defaultdict(int)
    label_total_counts = defaultdict(int)
    for pred in all_preds:
        label_total_counts[pred["label"]] += 1
    for gt in all_gt:
        for pred in all_preds:
            if pred["session"] != gt["session"]:
                continue
            if temporal_iou(pred["segment"], gt["segment"]) >= 0.1:
                label_match_counts[pred["label"]] += 1
    for label, cnt in sorted(label_match_counts.items(), key=lambda x: -x[1])[:20]:
        total = label_total_counts[label]
        print(f"  {label:25s}: {cnt:3d} matches / {total:3d} total")


if __name__ == "__main__":
    main()

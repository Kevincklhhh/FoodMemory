#!/usr/bin/env python3
"""
evaluate_interaction_detect.py - Evaluate detection methods against actions.json GT.

Compares VLM, HOI+SigLIP, and OWLv2 detections against ground truth from
annotations/{session}/actions.json. Uses temporal distance + semantic name
similarity for matching.

Matching criteria:
  - VLM: detection midpoint within max_distance of GT midpoint + name match
  - HOI/OWLv2: any frame within [GT_start - max_distance, GT_end + max_distance] + name match

Name matching: SentenceTransformer cosine similarity >= name_threshold.

Usage:
    python system_design/evaluate_interaction_detect.py --participant kailai --session 20260310-195710 --vlm-tag qwen_block30s
    python system_design/evaluate_interaction_detect.py --participant kailai --session 20260310-195710 --vlm-tag qwen_block30s --max-distance 5.0

Prerequisites:
    - actions.json from annotations/{session}/
    - vlm_{tag}_results.json from vlm_detect.py (optional)
    - {P}_{session}_hoi_siglip_results.json from hoi_siglip_interact_detect.py (optional)
    - {P}_{session}_owlv2_detections.json from owlv2_detection.py (optional)
    - pip install sentence-transformers
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from utils import (
    interact_dir,
    instance_id_to_visual_class,
    load_actions,
    load_food_items,
    load_ledger,
    outputs_dir,
    participant_dir,
)

DEFAULT_MAX_DISTANCE = 10.0
DEFAULT_NAME_THRESHOLD = 0.7
SENTENCE_MODEL = "all-MiniLM-L6-v2"


# =============================================================================
# SEMANTIC NAME MATCHING
# =============================================================================

def build_name_matcher(food_items: List[str]):
    """Return a function (name_a, name_b) -> similarity float."""
    try:
        from sentence_transformers import SentenceTransformer, util
        model = SentenceTransformer(SENTENCE_MODEL)
        # Pre-encode known food items
        known_embs = model.encode(food_items, convert_to_tensor=True, show_progress_bar=False)
        known_lower = [n.lower() for n in food_items]

        def match(a: str, b: str) -> float:
            # Fast exact match
            if a.lower().strip() == b.lower().strip():
                return 1.0
            emb_a = model.encode(a, convert_to_tensor=True)
            emb_b = model.encode(b, convert_to_tensor=True)
            return float(util.cos_sim(emb_a, emb_b).item())

        print(f"  Loaded SentenceTransformer: {SENTENCE_MODEL}")
        return match
    except ImportError:
        print("  WARNING: sentence-transformers not installed, using exact matching")

        def match_exact(a: str, b: str) -> float:
            return 1.0 if a.lower().strip() == b.lower().strip() else 0.0

        return match_exact


# =============================================================================
# GT LOADING
# =============================================================================

def load_gt_segments(participant: str, session: str) -> List[Dict]:
    """
    Load GT segments from actions.json, mapping instance_id → visual_class.
    Returns list of {visual_class, action, start, end}.
    """
    actions = load_actions(participant, session)
    ledger = load_ledger(participant)
    id_to_class = instance_id_to_visual_class(ledger)

    segments = []
    for a in actions:
        instance_id = a.get("item", "")
        visual_class = id_to_class.get(instance_id, instance_id)
        segments.append({
            "visual_class": visual_class,
            "instance_id": instance_id,
            "action": a.get("action", ""),
            "start": float(a.get("start", 0)),
            "end": float(a.get("end", 0)),
        })
    return segments


# =============================================================================
# DETECTION LOADING
# =============================================================================

def load_vlm_detections(participant: str, session: str, tag: str) -> Optional[List[Dict]]:
    """Load VLM detections from vlm_{tag}_results.json."""
    f = outputs_dir(participant, session) / f"vlm_{tag}_results.json"
    if not f.exists():
        return None
    with open(f) as fh:
        data = json.load(fh)
    dets = []
    for det in data.get("detections", []):
        start = det.get("det_start_abs")
        end = det.get("det_end_abs")
        if start is None or end is None:
            continue
        dets.append({
            "food_name": det.get("detected_item_name", ""),
            "start": float(start),
            "end": float(end),
            "interaction_type": det.get("interaction_type", ""),
            "confidence": det.get("confidence", ""),
        })
    return dets


def load_hoi_detections(participant: str, session: str) -> Optional[List[Dict]]:
    """Load HOI+SigLIP detections as flat frame list."""
    f = interact_dir(participant, session) / f"{participant}_{session}_hoi_siglip_results.json"
    if not f.exists():
        return None
    with open(f) as fh:
        data = json.load(fh)
    return data.get("detections", [])


def load_owlv2_detections(participant: str, session: str) -> Optional[List[Dict]]:
    """Load OWLv2 detections as flat frame list with session_timestamp_s."""
    f = interact_dir(participant, session) / f"{participant}_{session}_owlv2_detections.json"
    if not f.exists():
        return None
    with open(f) as fh:
        data = json.load(fh)
    flat = []
    for clip_data in data.get("videos", {}).values():
        for frame in clip_data.get("frames", []):
            ts = frame.get("session_timestamp_s", frame.get("clip_timestamp_s", 0))
            for det in frame.get("detections", []):
                flat.append({
                    "session_timestamp_s": ts,
                    "food_name": det["label"],
                    "score": det["score"],
                })
    return flat


# =============================================================================
# MATCHING
# =============================================================================

def match_vlm_to_gt(
    gt: Dict,
    vlm_dets: List[Dict],
    name_match_fn,
    name_threshold: float,
    max_distance: float,
) -> Dict:
    """Check if any VLM detection matches this GT segment."""
    gt_mid = (gt["start"] + gt["end"]) / 2
    best_dist = float("inf")
    best_item = None
    wrong_items = []

    for det in vlm_dets:
        det_mid = (det["start"] + det["end"]) / 2
        dist = abs(det_mid - gt_mid)
        if dist > max_distance:
            continue
        sim = name_match_fn(det["food_name"], gt["visual_class"])
        if sim >= name_threshold:
            if dist < best_dist:
                best_dist = dist
                best_item = det["food_name"]
        else:
            wrong_items.append(det["food_name"])

    if best_item is not None:
        return {"detected": True, "distance_s": round(best_dist, 2), "matched_item": best_item}

    if wrong_items:
        unique_wrong = list(dict.fromkeys(wrong_items))[:5]
        return {"detected": False, "miss_reason": "wrong_item", "wrong_items": unique_wrong}
    return {"detected": False, "miss_reason": "no_detection"}


def match_frame_to_gt(
    gt: Dict,
    frame_dets: List[Dict],
    name_match_fn,
    name_threshold: float,
    max_distance: float,
    ts_key: str = "session_timestamp_s",
    name_key: str = "food_name",
) -> Dict:
    """Check if any frame detection matches this GT segment (temporal window + name)."""
    window_start = gt["start"] - max_distance
    window_end = gt["end"] + max_distance
    best_item = None
    best_ts = None
    wrong_items = []

    for det in frame_dets:
        ts = det.get(ts_key, 0)
        if ts < window_start or ts > window_end:
            continue
        sim = name_match_fn(det[name_key], gt["visual_class"])
        if sim >= name_threshold:
            if best_item is None:
                best_item = det[name_key]
                best_ts = ts
        else:
            wrong_items.append(det[name_key])

    if best_item is not None:
        return {"detected": True, "matched_item": best_item, "matched_at_s": round(best_ts, 2)}

    if wrong_items:
        unique_wrong = list(dict.fromkeys(wrong_items))[:5]
        return {"detected": False, "miss_reason": "wrong_item", "wrong_items": unique_wrong}
    return {"detected": False, "miss_reason": "no_detection"}


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_method(
    method_name: str,
    gt_segments: List[Dict],
    detections,
    match_fn,
    name_match_fn,
    name_threshold: float,
    max_distance: float,
) -> Dict:
    if detections is None:
        return {"available": False}

    results = []
    for gt in gt_segments:
        res = match_fn(gt, detections, name_match_fn, name_threshold, max_distance)
        res["visual_class"] = gt["visual_class"]
        res["action"] = gt["action"]
        res["start"] = gt["start"]
        res["end"] = gt["end"]
        results.append(res)

    n_gt = len(results)
    n_detected = sum(1 for r in results if r["detected"])
    detection_rate = n_detected / n_gt if n_gt > 0 else 0.0

    # Per-item detection rate
    by_item: Dict[str, List[bool]] = defaultdict(list)
    for r in results:
        by_item[r["visual_class"]].append(r["detected"])
    per_item = {
        item: {
            "detected": sum(v),
            "total": len(v),
            "rate": round(sum(v) / len(v), 3),
        }
        for item, v in sorted(by_item.items())
    }

    # Miss breakdown
    misses = [r for r in results if not r["detected"]]
    miss_no_det = sum(1 for r in misses if r.get("miss_reason") == "no_detection")
    miss_wrong = sum(1 for r in misses if r.get("miss_reason") == "wrong_item")

    return {
        "available": True,
        "n_gt": n_gt,
        "n_detected": n_detected,
        "detection_rate": round(detection_rate, 4),
        "miss_no_detection": miss_no_det,
        "miss_wrong_item": miss_wrong,
        "per_item": per_item,
        "per_segment": results,
    }


def compute_vlm_precision(vlm_dets: Optional[List[Dict]], gt_segments: List[Dict],
                           name_match_fn, name_threshold: float, max_distance: float) -> Optional[float]:
    """Fraction of VLM detections that match any GT segment."""
    if vlm_dets is None:
        return None
    if not vlm_dets:
        return 0.0
    matched = 0
    for det in vlm_dets:
        det_mid = (det["start"] + det["end"]) / 2
        for gt in gt_segments:
            gt_mid = (gt["start"] + gt["end"]) / 2
            if abs(det_mid - gt_mid) <= max_distance:
                sim = name_match_fn(det["food_name"], gt["visual_class"])
                if sim >= name_threshold:
                    matched += 1
                    break
    return round(matched / len(vlm_dets), 4)


# =============================================================================
# MAIN
# =============================================================================

def run_evaluation(
    participant: str,
    session: str,
    vlm_tag: str,
    max_distance: float = DEFAULT_MAX_DISTANCE,
    name_threshold: float = DEFAULT_NAME_THRESHOLD,
    verbose: bool = False,
) -> Optional[Dict]:
    print(f"\n{'='*70}")
    print(f"EVALUATION | participant={participant} session={session}")
    print(f"VLM tag: {vlm_tag} | max_distance={max_distance}s | name_thresh={name_threshold}")
    print(f"{'='*70}")

    gt_segments = load_gt_segments(participant, session)
    if not gt_segments:
        print("No GT segments found in actions.json")
        return None
    print(f"GT segments: {len(gt_segments)}")

    food_items = load_food_items(participant)
    name_match_fn = build_name_matcher(food_items)

    # Load all detection sources
    vlm_dets = load_vlm_detections(participant, session, vlm_tag)
    hoi_dets = load_hoi_detections(participant, session)
    owlv2_dets = load_owlv2_detections(participant, session)

    print(f"VLM detections: {len(vlm_dets) if vlm_dets is not None else 'N/A'}")
    print(f"HOI+SigLIP detections: {len(hoi_dets) if hoi_dets is not None else 'N/A'}")
    print(f"OWLv2 detections: {len(owlv2_dets) if owlv2_dets is not None else 'N/A'}")

    # Evaluate each method
    vlm_eval = evaluate_method(
        "vlm", gt_segments, vlm_dets,
        match_fn=match_vlm_to_gt,
        name_match_fn=name_match_fn,
        name_threshold=name_threshold,
        max_distance=max_distance,
    )

    hoi_eval = evaluate_method(
        "hoi_siglip", gt_segments, hoi_dets,
        match_fn=match_frame_to_gt,
        name_match_fn=name_match_fn,
        name_threshold=name_threshold,
        max_distance=max_distance,
    )

    owlv2_eval = evaluate_method(
        "owlv2", gt_segments, owlv2_dets,
        match_fn=match_frame_to_gt,
        name_match_fn=name_match_fn,
        name_threshold=name_threshold,
        max_distance=max_distance,
    )

    # VLM precision
    vlm_precision = compute_vlm_precision(
        vlm_dets, gt_segments, name_match_fn, name_threshold, max_distance
    )

    # Print summary
    print(f"\n{'='*50}")
    print(f"RESULTS")
    print(f"{'='*50}")
    for method, ev in [
        (f"VLM ({vlm_tag})", vlm_eval),
        ("HOI+SigLIP", hoi_eval),
        ("OWLv2", owlv2_eval),
    ]:
        if ev.get("available"):
            print(f"  {method}: {ev['n_detected']}/{ev['n_gt']} = {ev['detection_rate']:.1%}"
                  f"  (miss_no_det={ev['miss_no_detection']}, miss_wrong={ev['miss_wrong_item']})")
        else:
            print(f"  {method}: N/A (no detections file found)")
    if vlm_precision is not None:
        print(f"  VLM precision: {vlm_precision:.1%}")

    if verbose:
        for method, ev in [
            (f"VLM", vlm_eval),
            ("HOI+SigLIP", hoi_eval),
            ("OWLv2", owlv2_eval),
        ]:
            if not ev.get("available"):
                continue
            print(f"\n  {method} per-segment:")
            for r in ev["per_segment"]:
                icon = "✓" if r["detected"] else "✗"
                print(f"    {icon} {r['visual_class']:<25} {r['action']:<35} "
                      f"[{r['start']:.0f}s-{r['end']:.0f}s]"
                      + (f" miss:{r.get('miss_reason')}" if not r["detected"] else
                         f" match:{r.get('matched_item')}"))

    results = {
        "participant": participant,
        "session": session,
        "vlm_tag": vlm_tag,
        "max_distance_s": max_distance,
        "name_threshold": name_threshold,
        "n_gt_segments": len(gt_segments),
        "methods": {
            f"vlm_{vlm_tag}": vlm_eval,
            "hoi_siglip": hoi_eval,
            "owlv2": owlv2_eval,
        },
        "vlm_precision": vlm_precision,
    }

    out_dir = interact_dir(participant, session)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{participant}_{session}_eval_results_{vlm_tag}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved: {out_file}")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate detection methods against actions.json ground truth"
    )
    parser.add_argument("--participant", required=True, help="Participant ID (e.g., kailai)")
    parser.add_argument("--session", required=True, help="Session ID (e.g., 20260310-195710)")
    parser.add_argument("--vlm-tag", required=True,
                        help="VLM results tag (e.g., qwen_block30s)")
    parser.add_argument("--max-distance", type=float, default=DEFAULT_MAX_DISTANCE,
                        help=f"Max temporal distance for matching in seconds (default: {DEFAULT_MAX_DISTANCE})")
    parser.add_argument("--name-threshold", type=float, default=DEFAULT_NAME_THRESHOLD,
                        help=f"Semantic name similarity threshold (default: {DEFAULT_NAME_THRESHOLD})")
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()
    run_evaluation(
        participant=args.participant,
        session=args.session,
        vlm_tag=args.vlm_tag,
        max_distance=args.max_distance,
        name_threshold=args.name_threshold,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()

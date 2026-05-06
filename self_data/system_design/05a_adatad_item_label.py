#!/usr/bin/env python3
"""
05a_adatad_item_label.py — Label AdaTAD verb segments with inventory items using
existing HOI + DINOv2 matching results.

For each AdaTAD verb detection (score ≥ threshold, after NMS), finds overlapping
hands23 frames with object_contact, looks up their DINOv2 food-package similarity
scores, and aggregates (top-3 mean) to assign item labels per segment.

Most segments will have no match (noise); only segments with hand-object contact
AND a DINOv2 similarity above threshold survive.

Output feeds into an LLM as rough structured context — vocabulary precision is
not critical.

Usage:
    python system_design/05a_adatad_item_label.py --participant kailai
    python system_design/05a_adatad_item_label.py --participant kailai --session 20260310-195710
    python system_design/05a_adatad_item_label.py --participant kailai --all

Prerequisites:
    - AdaTAD detections: participants/{P}/outputs/{session}/adatad_detections.json
    - hands23 results:   participants/{P}/outputs/{session}/hands23_detection/*_hands23_results.json
    - DINO matches:      participants/{P}/outputs/{session}/hands23_detection/*_dino_matches.json
"""

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    get_session_clips,
    get_sessions,
    hands23_dir,
    load_ledger,
    outputs_dir,
    participant_dir,
)


# ── AdaTAD loading (from 09_evaluate) ──────────────────────────────────────

def nms_temporal(detections: List[dict], theta: float) -> List[dict]:
    """1D NMS: suppress lower-scored segments overlapping above theta."""
    if not detections:
        return []
    dets = sorted(detections, key=lambda d: d["score"], reverse=True)
    keep = []
    for d in dets:
        suppress = False
        for k in keep:
            s1, e1 = d["segment"]
            s2, e2 = k["segment"]
            inter = max(0.0, min(e1, e2) - max(s1, s2))
            union = (e1 - s1) + (e2 - s2) - inter
            if union > 0 and inter / union >= theta:
                suppress = True
                break
        if not suppress:
            keep.append(d)
    return keep


def load_adatad_verbs(
    participant: str, session: str,
    score_threshold: float = 0.3, nms_theta: float = 0.5,
) -> List[dict]:
    """Load verb-only AdaTAD detections, convert to session time, filter + NMS.

    Returns list of {segment: [start, end], label: str, score: float}.
    """
    det_path = outputs_dir(participant, session) / "adatad_detections.json"
    if not det_path.exists():
        return []

    raw = json.loads(det_path.read_text())
    clips = get_session_clips(participant, session)

    # Build clip_name → cumulative offset
    clip_offsets = {}
    offset = 0.0
    for fname, _, dur in clips:
        clip_offsets[fname.replace(".mp4", "")] = offset
        offset += dur

    # Convert verb detections to session time
    dets = []
    for clip_name, clip_dets in raw.get("verb", {}).items():
        off = clip_offsets.get(clip_name, 0.0)
        for d in clip_dets:
            if d["score"] >= score_threshold:
                dets.append({
                    "segment": [d["segment"][0] + off, d["segment"][1] + off],
                    "label": d["label"],
                    "score": d["score"],
                })

    # NMS
    if nms_theta < 1.0:
        dets = nms_temporal(dets, nms_theta)

    dets.sort(key=lambda d: d["segment"][0])
    return dets


# ── HOI + DINO loading ─────────────────────────────────────────────────────

def load_dino_matches(participant: str, session: str) -> List[dict]:
    """Load all DINO food-package matches for a session.

    Returns flat list of:
        {timestamp, contact_state, hand_side, obj_bbox, top_matches: [...]}
    sorted by timestamp.
    """
    det_dir = hands23_dir(participant, session)
    dino_files = list(det_dir.glob("*_dino_matches.json"))
    if not dino_files:
        return []

    data = json.loads(dino_files[0].read_text())
    flat = []
    for video in data.get("videos", []):
        for m in video.get("matches", []):
            flat.append(m)

    flat.sort(key=lambda m: m["timestamp"])
    return flat


def load_siglip_matches(participant: str, session: str) -> List[dict]:
    """Load all SigLIP text-to-image matches for a session.

    Returns flat list of:
        {timestamp, contact_state, hand_side, top_matches: [{food_name, similarity}]}
    sorted by timestamp.
    """
    det_dir = hands23_dir(participant, session)
    sig_files = list(det_dir.glob("*_siglip_matches.json"))
    if not sig_files:
        return []

    data = json.loads(sig_files[0].read_text())
    flat = []
    for video in data.get("videos", []):
        for m in video.get("matches", []):
            flat.append(m)

    flat.sort(key=lambda m: m["timestamp"])
    return flat


def load_hands23_frames(participant: str, session: str) -> List[dict]:
    """Load hands23 results — all frames with detections.

    Returns flat list of {session_timestamp_s, detections: [...]}.
    """
    det_dir = hands23_dir(participant, session)
    h23_files = list(det_dir.glob("*_hands23_results.json"))
    if not h23_files:
        return []

    data = json.loads(h23_files[0].read_text())
    flat = []
    for video in data.get("videos", []):
        for frame in video.get("frames", []):
            flat.append(frame)

    flat.sort(key=lambda f: f["session_timestamp_s"])
    return flat


# ── Segment → Item matching ────────────────────────────────────────────────

def _find_contact_crops(matches: List[dict], seg_start: float, seg_end: float) -> List[dict]:
    """Find matches within [seg_start-0.5, seg_end+0.5] with object_contact."""
    crops = []
    for m in matches:
        ts = m["timestamp"]
        if ts < seg_start - 0.5:
            continue
        if ts > seg_end + 0.5:
            break  # sorted
        if m.get("contact_state") != "object_contact":
            continue
        if not m.get("top_matches"):
            continue
        crops.append(m)
    return crops


def _aggregate_scores(
    crops: List[dict], key_field: str, score_field: str, top_k_pool: int,
) -> Dict[str, float]:
    """Collect per-item scores from crops and return top-K mean per item."""
    item_sims: Dict[str, List[float]] = defaultdict(list)
    for crop in crops:
        for match in crop["top_matches"]:
            item_sims[match[key_field]].append(match[score_field])

    scores = {}
    for name, sims in item_sims.items():
        sims_sorted = sorted(sims, reverse=True)
        scores[name] = sum(sims_sorted[:top_k_pool]) / len(sims_sorted[:top_k_pool])
    return scores


def match_segment_items(
    segment: List[float],
    dino_matches: List[dict],
    siglip_matches: List[dict],
    sim_threshold: float = 0.15,
    siglip_threshold: float = 0.15,
    top_k_pool: int = 3,
    multi_item_ratio: float = 0.85,
) -> dict:
    """For a temporal segment, aggregate DINO + SigLIP matches to identify items.

    DINOv2 (visual): image-to-image matching via instance_id.
    SigLIP (text): text-to-image zero-shot matching via food_name.

    Returns:
        {
            "n_contact_crops": int,
            "items": [{"instance_id", "visual_class", "score", "source"}],
            "all_scores": {instance_id: score},       # DINOv2
            "siglip_scores": {food_name: score},       # SigLIP
        }
    """
    seg_start, seg_end = segment

    # DINOv2
    dino_crops = _find_contact_crops(dino_matches, seg_start, seg_end)
    dino_scores = _aggregate_scores(dino_crops, "instance_id", "similarity", top_k_pool)

    # SigLIP
    sig_crops = _find_contact_crops(siglip_matches, seg_start, seg_end)
    sig_scores = _aggregate_scores(sig_crops, "food_name", "similarity", top_k_pool)

    n_contact = max(len(dino_crops), len(sig_crops))

    if not dino_scores and not sig_scores:
        return {"n_contact_crops": n_contact, "items": [], "all_scores": {}, "siglip_scores": {}}

    # Build items[] from DINOv2 (above sim_threshold)
    items = []
    if dino_scores:
        best_dino = max(dino_scores.values())
        if best_dino >= sim_threshold:
            cutoff = best_dino * multi_item_ratio
            for iid, score in sorted(dino_scores.items(), key=lambda x: -x[1]):
                if score >= sim_threshold and score >= cutoff:
                    items.append({
                        "instance_id": iid,
                        "score": round(score, 4),
                        "source": "visual",
                    })

    # Add SigLIP-only items (above siglip_threshold, not already matched by DINOv2)
    if sig_scores:
        # Map existing items' visual_class to avoid duplicates
        matched_names = set()
        for it in items:
            # instance_id like "soy_sauce_20260310" — we'll check against food_name later
            matched_names.add(it["instance_id"])

        best_sig = max(sig_scores.values())
        if best_sig >= siglip_threshold:
            cutoff = best_sig * multi_item_ratio
            for food_name, score in sorted(sig_scores.items(), key=lambda x: -x[1]):
                if score >= siglip_threshold and score >= cutoff:
                    # Check if any DINOv2 item already covers this food
                    already_covered = False
                    for it in items:
                        vc = it.get("visual_class", "")
                        if vc and vc.lower() == food_name.lower():
                            already_covered = True
                            break
                    if not already_covered:
                        items.append({
                            "food_name": food_name,
                            "score": round(score, 4),
                            "source": "text",
                        })

    return {
        "n_contact_crops": n_contact,
        "items": items,
        "all_scores": {k: round(v, 4) for k, v in
                       sorted(dino_scores.items(), key=lambda x: -x[1])},
        "siglip_scores": {k: round(v, 4) for k, v in
                          sorted(sig_scores.items(), key=lambda x: -x[1])},
    }


# ── Session processing ─────────────────────────────────────────────────────

def run_session(
    participant: str,
    session: str,
    score_threshold: float,
    nms_theta: float,
    sim_threshold: float,
    siglip_threshold: float,
    top_k_pool: int,
    multi_item_ratio: float,
) -> Optional[dict]:
    """Process one session: label AdaTAD verb segments with inventory items."""

    # Load AdaTAD verb segments
    verbs = load_adatad_verbs(participant, session, score_threshold, nms_theta)
    if not verbs:
        print(f"  {session}: no AdaTAD verb segments")
        return None

    # Load DINO matches
    dino_matches = load_dino_matches(participant, session)
    if not dino_matches:
        print(f"  {session}: no DINO matches (run 03_dino_food_matching.py first)")
        return None

    # Load SigLIP matches (optional — graceful fallback)
    siglip_matches = load_siglip_matches(participant, session)
    if not siglip_matches:
        print(f"  {session}: no SigLIP matches (text matching disabled)")

    # Load ledger for visual_class lookup
    ledger = load_ledger(participant)
    iid_to_vc = {iid: item["visual_class"] for iid, item in ledger["items"].items()}

    # Match each verb segment
    labeled = []
    n_matched = 0
    for seg in verbs:
        result = match_segment_items(
            seg["segment"], dino_matches, siglip_matches,
            sim_threshold=sim_threshold,
            siglip_threshold=siglip_threshold,
            top_k_pool=top_k_pool,
            multi_item_ratio=multi_item_ratio,
        )

        # Attach visual_class to DINOv2-sourced items
        for item in result["items"]:
            if "instance_id" in item:
                item["visual_class"] = iid_to_vc.get(item["instance_id"], item["instance_id"])

        entry = {
            "segment": [round(seg["segment"][0], 2), round(seg["segment"][1], 2)],
            "duration": round(seg["segment"][1] - seg["segment"][0], 2),
            "verb": seg["label"],
            "verb_score": round(seg["score"], 4),
            "n_contact_crops": result["n_contact_crops"],
            "items": result["items"],
        }

        if result["items"]:
            n_matched += 1

        entry["all_scores"] = result["all_scores"]
        entry["siglip_scores"] = result.get("siglip_scores", {})

        labeled.append(entry)

    # Stats
    n_with_contact = sum(1 for l in labeled if l["n_contact_crops"] > 0)
    print(f"  {session}: {len(verbs)} verb segments → "
          f"{n_with_contact} with contact → {n_matched} with item match")

    return {
        "session": session,
        "n_verb_segments": len(verbs),
        "n_with_contact": n_with_contact,
        "n_with_item_match": n_matched,
        "segments": labeled,
    }


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Label AdaTAD verb segments with inventory items via HOI+DINOv2+SigLIP"
    )
    parser.add_argument("--participant", default="kailai")
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--session", help="Single session")
    group.add_argument("--all", action="store_true", help="All sessions")
    parser.add_argument("--score-threshold", type=float, default=0.3,
                        help="AdaTAD verb score threshold (default: 0.3)")
    parser.add_argument("--nms-theta", type=float, default=0.5,
                        help="NMS IoU threshold (default: 0.5)")
    parser.add_argument("--sim-threshold", type=float, default=0.15,
                        help="DINO similarity threshold for item match (default: 0.15)")
    parser.add_argument("--siglip-threshold", type=float, default=0.15,
                        help="SigLIP similarity threshold for item match (default: 0.15)")
    parser.add_argument("--top-k-pool", type=int, default=3,
                        help="Top-K mean pooling across contact crops (default: 3)")
    parser.add_argument("--multi-item-ratio", type=float, default=0.85,
                        help="Include items within this ratio of top score (default: 0.85)")
    args = parser.parse_args()

    sessions = (
        [args.session] if args.session
        else get_sessions(args.participant) if args.all
        else get_sessions(args.participant)
    )

    print(f"{'='*70}")
    print(f"AdaTAD Verb → Item Labeling (HOI + DINOv2 + SigLIP)")
    print(f"{'='*70}")
    print(f"Participant:        {args.participant}")
    print(f"Score threshold:    {args.score_threshold}")
    print(f"NMS theta:          {args.nms_theta}")
    print(f"DINO sim threshold: {args.sim_threshold}")
    print(f"SigLIP threshold:   {args.siglip_threshold}")
    print(f"Top-K pool:         {args.top_k_pool}")
    print(f"Multi-item ratio:   {args.multi_item_ratio}")
    print(f"Sessions:           {len(sessions)}")
    print()

    all_results = []
    for session in sessions:
        result = run_session(
            participant=args.participant,
            session=session,
            score_threshold=args.score_threshold,
            nms_theta=args.nms_theta,
            sim_threshold=args.sim_threshold,
            siglip_threshold=args.siglip_threshold,
            top_k_pool=args.top_k_pool,
            multi_item_ratio=args.multi_item_ratio,
        )
        if result:
            all_results.append(result)

    if not all_results:
        print("\nNo results produced. Check prerequisites.")
        return

    # Aggregate stats
    total_segs = sum(r["n_verb_segments"] for r in all_results)
    total_contact = sum(r["n_with_contact"] for r in all_results)
    total_matched = sum(r["n_with_item_match"] for r in all_results)

    print(f"\n{'='*70}")
    print(f"Summary: {total_segs} verb segments → "
          f"{total_contact} with contact ({total_contact/total_segs*100:.0f}%) → "
          f"{total_matched} with item ({total_matched/total_segs*100:.0f}%)")
    print(f"{'='*70}")

    # Print matched segments
    print(f"\nMatched segments:")
    for r in all_results:
        for seg in r["segments"]:
            if not seg["items"]:
                continue
            items_str = ", ".join(
                f"{it.get('visual_class') or it.get('food_name', '?')} ({it['score']:.3f} {it.get('source', '?')})"
                for it in seg["items"]
            )
            print(f"  [{r['session']}] {seg['segment'][0]:6.1f}-{seg['segment'][1]:6.1f}s "
                  f"({seg['duration']:4.1f}s) {seg['verb']:15s} "
                  f"score={seg['verb_score']:.3f}  → {items_str}")

    # Save output
    output = {
        "participant": args.participant,
        "timestamp": datetime.now().isoformat(),
        "config": {
            "score_threshold": args.score_threshold,
            "nms_theta": args.nms_theta,
            "sim_threshold": args.sim_threshold,
            "siglip_threshold": args.siglip_threshold,
            "top_k_pool": args.top_k_pool,
            "multi_item_ratio": args.multi_item_ratio,
        },
        "stats": {
            "total_verb_segments": total_segs,
            "with_contact": total_contact,
            "with_item_match": total_matched,
        },
        "sessions": all_results,
    }

    out_dir = participant_dir(args.participant) / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "adatad_item_labels.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()

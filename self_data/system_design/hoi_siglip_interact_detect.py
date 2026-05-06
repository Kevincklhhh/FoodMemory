#!/usr/bin/env python3
"""
hoi_siglip_interact_detect.py - Combine HOI + SigLIP results into per-frame food detections.

Reads SigLIP food matches from 02_siglip_food_matching.py and extracts a flat
list of detections with session-relative timestamps. Each detection represents
a frame where a hand-object contact was observed and a food item was identified
above the similarity threshold.

Usage:
    python system_design/hoi_siglip_interact_detect.py --participant kailai --session 20260310-195710
    python system_design/hoi_siglip_interact_detect.py --participant kailai --session 20260310-195710 --threshold 0.15

Prerequisites:
    - {P}_{session}_siglip_matches.json from 02_siglip_food_matching.py
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

from utils import hands23_dir, interact_dir

DEFAULT_THRESHOLD = 0.15


def load_siglip_matches(participant: str, session: str) -> Optional[Dict]:
    f = hands23_dir(participant, session) / f"{participant}_{session}_siglip_matches.json"
    if not f.exists():
        print(f"ERROR: {f} not found. Run 02_siglip_food_matching.py first.")
        return None
    with open(f) as fh:
        return json.load(fh)


def extract_detections(siglip_data: Dict, threshold: float) -> List[Dict]:
    """
    Build a flat list of detections above threshold.

    For each frame, collects all food matches across all hand-object pairs.
    If multiple hands detected the same food in a frame, keeps the best similarity.
    Includes contact_state if any hand was in object contact.
    """
    detections = []

    for video in siglip_data.get("videos", []):
        # Group by session_timestamp_s
        ts_groups: Dict[float, List[Dict]] = {}
        for match in video.get("matches", []):
            ts = match.get("timestamp", 0)
            ts_groups.setdefault(ts, []).append(match)

        for ts in sorted(ts_groups.keys()):
            matches = ts_groups[ts]

            # Best similarity per food name across all hand-object pairs
            frame_foods: Dict[str, float] = {}
            frame_contact = None

            for m in matches:
                contact = m.get("contact_state", "")
                if contact in ("object_contact", "obj_to_obj_contact"):
                    frame_contact = contact
                for food_match in m.get("top_matches", []):
                    name = food_match["food_name"]
                    sim = food_match["similarity"]
                    if sim >= threshold:
                        if name not in frame_foods or sim > frame_foods[name]:
                            frame_foods[name] = sim

            for food_name, sim in frame_foods.items():
                detections.append({
                    "session_timestamp_s": ts,
                    "food_name": food_name,
                    "similarity": sim,
                    "contact_state": frame_contact,
                })

    # Sort by timestamp
    detections.sort(key=lambda d: d["session_timestamp_s"])
    return detections


def run_session(
    participant: str,
    session: str,
    threshold: float = DEFAULT_THRESHOLD,
    verbose: bool = False,
) -> Optional[Dict]:
    siglip_data = load_siglip_matches(participant, session)
    if siglip_data is None:
        return None

    print(f"\n{'='*70}")
    print(f"HOI+SigLIP | participant={participant} session={session}")
    print(f"Threshold: {threshold}")
    print(f"{'='*70}")

    detections = extract_detections(siglip_data, threshold)

    # Summary per food
    from collections import Counter
    food_counts = Counter(d["food_name"] for d in detections)
    print(f"Total detections: {len(detections)} across {len(food_counts)} food types")
    for name, cnt in food_counts.most_common(10):
        print(f"  {name}: {cnt}")

    if verbose:
        for det in detections:
            print(f"  t={det['session_timestamp_s']:.1f}s  {det['food_name']} "
                  f"(sim={det['similarity']:.3f}  {det['contact_state']})")

    results = {
        "participant": participant,
        "session": session,
        "method": "hoi_siglip",
        "threshold": threshold,
        "num_detections": len(detections),
        "detections": detections,
    }

    out_dir = interact_dir(participant, session)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{participant}_{session}_hoi_siglip_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved: {out_file}")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Combine HOI + SigLIP results into flat food detection list"
    )
    parser.add_argument("--participant", required=True, help="Participant ID (e.g., kailai)")
    parser.add_argument("--session", required=True, help="Session ID (e.g., 20260310-195710)")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help=f"SigLIP similarity threshold (default: {DEFAULT_THRESHOLD})")
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()
    run_session(
        participant=args.participant,
        session=args.session,
        threshold=args.threshold,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()

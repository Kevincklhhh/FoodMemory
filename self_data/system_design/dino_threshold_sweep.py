#!/usr/bin/env python3
"""DINO threshold calibration sweep against actions.json ground truth.

For every GT action in every annotated session:
  - Compute the PEAK per-frame DINO score for that iid within
    [start - pad, end + pad] (HOI-contact frames only).
  - Tally per (visual_class, iid) the distribution of peak scores.

For every (vc, iid), separately compute the "bleed" distribution: peak DINO
scores in HOI-contact frames OUTSIDE every GT action span for that iid.
That gives a precision picture: a tau that keeps 95% of in-action peaks
also keeps how many outside-action frames?

Outputs a per-vc table:
    vc, n_actions, recall@tau, bleed_frames@tau (per tau in 0.10..0.50)
plus a recommended per-iid tau (largest tau where in-action recall ≥ 0.90).

Usage:
    python system_design/dino_threshold_sweep.py --participant kailai
    python system_design/dino_threshold_sweep.py --participant kailai --pad 1.0
    python system_design/dino_threshold_sweep.py --participant kailai --recall-floor 0.95 --output dino_thresholds.json
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    get_sessions,
    hands23_dir,
    load_ledger,
    participant_dir,
)


def load_dino_per_iid_per_t(participant: str, session: str) -> dict[str, list[tuple[float, float]]]:
    """For each iid, return sorted list of (timestamp, max_dino_across_hands)."""
    path = hands23_dir(participant, session) / f"{participant}_{session}_dino_matches.json"
    if not path.exists():
        return {}
    d = json.loads(path.read_text())
    by_iid: dict[str, dict[float, float]] = defaultdict(dict)
    for v in d["videos"]:
        for m in v["matches"]:
            if m.get("contact_state") != "object_contact":
                continue
            t = round(m["timestamp"], 2)
            for tm in m.get("top_matches") or []:
                iid = tm["instance_id"]
                sim = float(tm["similarity"])
                cur = by_iid[iid].get(t, 0.0)
                if sim > cur:
                    by_iid[iid][t] = sim
    return {iid: sorted(d.items()) for iid, d in by_iid.items()}


def load_actions(participant: str, session: str) -> list[dict]:
    path = participant_dir(participant) / "annotations" / session / "actions.json"
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return []


def peak_in_range(
    points: list[tuple[float, float]],
    start: float,
    end: float,
) -> tuple[float, int]:
    """(peak_dino, n_frames_in_range) over [start, end]."""
    peak = 0.0
    n = 0
    for t, s in points:
        if start <= t <= end:
            n += 1
            if s > peak:
                peak = s
    return peak, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--participant", default="kailai")
    ap.add_argument("--pad", type=float, default=1.0,
                    help="Pad around each GT action (s) when computing in-action peak.")
    ap.add_argument("--taus", type=str, default="0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50",
                    help="Comma-separated thresholds to sweep.")
    ap.add_argument("--recall-floor", type=float, default=0.90,
                    help="Recommended tau is max tau with action-level recall ≥ this floor.")
    ap.add_argument("--output", type=Path, default=None,
                    help="Optional JSON path for per-iid recommended thresholds.")
    args = ap.parse_args()

    taus = [float(x) for x in args.taus.split(",")]

    ledger = load_ledger(args.participant)
    iid_to_vc: dict[str, str] = {iid: e["visual_class"] for iid, e in ledger["items"].items()}

    sessions = get_sessions(args.participant)

    # Per-iid stats accumulators.
    per_iid_in_action_peaks: dict[str, list[float]] = defaultdict(list)
    per_iid_bleed_peaks: dict[str, list[float]] = defaultdict(list)
    per_iid_n_actions: dict[str, int] = defaultdict(int)

    n_sessions_with_actions = 0
    for session in sessions:
        actions = load_actions(args.participant, session)
        if not actions:
            continue
        n_sessions_with_actions += 1
        dino_by_iid = load_dino_per_iid_per_t(args.participant, session)

        # Build per-iid action interval list.
        action_intervals: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for a in actions:
            iid = a.get("item")
            if iid is None or iid not in iid_to_vc:
                continue
            try:
                s = float(a["start"]) - args.pad
                e = float(a["end"]) + args.pad
            except (KeyError, TypeError, ValueError):
                continue
            if e <= s:
                continue
            action_intervals[iid].append((s, e))

        # In-action peak per (iid, action).
        for iid, intervals in action_intervals.items():
            points = dino_by_iid.get(iid, [])
            for s, e in intervals:
                peak, n = peak_in_range(points, s, e)
                if n > 0:  # only count actions where we had at least 1 HOI frame
                    per_iid_in_action_peaks[iid].append(peak)
                    per_iid_n_actions[iid] += 1
                else:
                    # No HOI frame in range — pipeline can't possibly recall.
                    # Still count as an action so recall isn't inflated.
                    per_iid_in_action_peaks[iid].append(0.0)
                    per_iid_n_actions[iid] += 1

        # Bleed peaks: per-iid HOI-contact frames OUTSIDE all GT spans.
        for iid, points in dino_by_iid.items():
            if iid not in iid_to_vc:
                continue
            ivs = action_intervals.get(iid, [])
            for t, sim in points:
                if any(s <= t <= e for s, e in ivs):
                    continue
                per_iid_bleed_peaks[iid].append(sim)

    print(f"Annotated sessions scanned: {n_sessions_with_actions}/{len(sessions)}")
    print()

    # Aggregate per-vc (sum across iids of same vc).
    vc_actions: dict[str, list[float]] = defaultdict(list)
    vc_bleed: dict[str, list[float]] = defaultdict(list)
    iid_to_n_actions: dict[str, int] = dict(per_iid_n_actions)

    for iid, peaks in per_iid_in_action_peaks.items():
        vc = iid_to_vc.get(iid, "?")
        vc_actions[vc].extend(peaks)
    for iid, peaks in per_iid_bleed_peaks.items():
        vc = iid_to_vc.get(iid, "?")
        vc_bleed[vc].extend(peaks)

    # ── Per-vc recall + bleed table ────────────────────────────────────────
    print(f"{'visual_class':38s} {'n_acts':>6s}  " + "  ".join(f"τ{t:.2f}" for t in taus))
    print(f"{'(action recall %)':38s}")
    print("-" * (46 + 8 * len(taus)))
    rec_by_vc: dict[str, list[float]] = {}
    for vc in sorted(vc_actions):
        peaks = vc_actions[vc]
        if not peaks:
            continue
        recalls = [sum(1 for p in peaks if p >= t) / len(peaks) * 100 for t in taus]
        rec_by_vc[vc] = recalls
        cells = "  ".join(f"{r:5.1f}" for r in recalls)
        print(f"{vc[:38]:38s} {len(peaks):>6d}  {cells}")

    print()
    print(f"{'visual_class':38s} {'n_bleed':>7s}  " + "  ".join(f"τ{t:.2f}" for t in taus))
    print(f"{'(bleed frames kept)':38s}")
    print("-" * (47 + 8 * len(taus)))
    for vc in sorted(vc_bleed):
        peaks = vc_bleed[vc]
        if not peaks:
            continue
        kept = [sum(1 for p in peaks if p >= t) for t in taus]
        cells = "  ".join(f"{k:5d}" for k in kept)
        print(f"{vc[:38]:38s} {len(peaks):>7d}  {cells}")

    # ── Recommendation ─────────────────────────────────────────────────────
    print()
    print(f"=== Recommended per-iid τ (max τ with recall ≥ {args.recall_floor:.0%}) ===")
    rec_per_iid: dict[str, dict] = {}
    for iid in sorted(per_iid_in_action_peaks):
        peaks = per_iid_in_action_peaks[iid]
        if not peaks:
            continue
        best_tau = 0.10
        for t in taus:
            r = sum(1 for p in peaks if p >= t) / len(peaks)
            if r >= args.recall_floor:
                best_tau = max(best_tau, t)
        recall_at_best = sum(1 for p in peaks if p >= best_tau) / len(peaks)
        bleed_kept = sum(1 for p in per_iid_bleed_peaks.get(iid, []) if p >= best_tau)
        bleed_total = len(per_iid_bleed_peaks.get(iid, []))
        rec_per_iid[iid] = {
            "tau": best_tau,
            "n_actions": len(peaks),
            "recall": round(recall_at_best, 3),
            "bleed_frames_kept": bleed_kept,
            "bleed_frames_total": bleed_total,
            "bleed_kept_pct": round(bleed_kept / max(bleed_total, 1) * 100, 1),
        }
        vc = iid_to_vc.get(iid, "?")
        print(f"  {iid:42s} ({vc[:24]:24s})  τ={best_tau:.2f}  "
              f"recall={recall_at_best:.0%} ({len(peaks)} acts)  "
              f"bleed_kept={bleed_kept}/{bleed_total} ({bleed_kept/max(bleed_total,1)*100:.0f}%)")

    if args.output:
        args.output.write_text(json.dumps({
            "participant": args.participant,
            "pad": args.pad,
            "recall_floor": args.recall_floor,
            "per_iid": rec_per_iid,
        }, indent=2) + "\n")
        print(f"\nSaved per-iid thresholds to {args.output}")


if __name__ == "__main__":
    main()

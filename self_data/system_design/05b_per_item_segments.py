#!/usr/bin/env python3
"""Per-item temporal segments from per-frame HOI + SigLIP/DINOv2 (no-TAD branch).

Lightweight alternative to the AdaTAD path (05a): derives item-keyed segments
directly from the per-frame SigLIP (visual_class) and DINO (instance_id) scores
produced by 02 and 03, without running a learned temporal action detector.

Pipeline per item (visual_class + instance_id):
  1. For each timestamp where a HOI object_contact crop exists, compute:
       score(t) = max over crops at t of max(siglip_score, dino_score * scale)
     (siglip matches by visual_class; dino matches by instance_id)
  2. active(t) = (siglip(t) >= tau_siglip) OR (dino(t) >= tau_dino)
  3. Close small gaps (<= gap_close seconds)
  4. Open short runs (< min_duration seconds)
  5. Emit contiguous runs as (start, end) segments, with peak/avg scores.

Output: outputs/{session}/per_item_segments.json  (consumed by
06_avp_round1_remaining_noTAD.py and 06_avp_round1_remaining_noplanner.py)

Usage:
  python system_design/05b_per_item_segments.py --participant kailai \
      --session 20260310-195710 --write \
      --tau-siglip 0.15 --tau-dino 0.15 \
      --gap-close 3 --min-duration 2
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    get_sessions,
    hands23_dir,
    load_actions,
    load_inventory,
    outputs_dir,
    participant_dir,
)


def load_hoi_timestamps(participant: str, session: str) -> list[float]:
    """Return sorted timestamps where hands23 reports object_contact."""
    path = hands23_dir(participant, session) / f"{participant}_{session}_hands23_results.json"
    d = json.loads(path.read_text())
    ts_with_contact = set()
    all_ts = set()
    for v in d["videos"]:
        for f in v["frames"]:
            t = round(f["session_timestamp_s"], 2)
            all_ts.add(t)
            for det in f["detections"]:
                if det.get("contact_state") == "object_contact":
                    ts_with_contact.add(t)
                    break
    return sorted(all_ts), sorted(ts_with_contact)


def load_siglip_scores(participant: str, session: str) -> dict[float, dict[str, float]]:
    """timestamp -> {visual_class: max_similarity}."""
    path = hands23_dir(participant, session) / f"{participant}_{session}_siglip_matches.json"
    d = json.loads(path.read_text())
    out: dict[float, dict[str, float]] = defaultdict(dict)
    for v in d["videos"]:
        for m in v["matches"]:
            t = round(m["timestamp"], 2)
            for tm in m.get("top_matches") or []:
                name = tm["food_name"]
                sim = tm["similarity"]
                out[t][name] = max(out[t].get(name, 0.0), sim)
    return dict(out)


def load_dino_scores(participant: str, session: str) -> dict[float, dict[str, float]]:
    """timestamp -> {instance_id: max_similarity}."""
    path = hands23_dir(participant, session) / f"{participant}_{session}_dino_matches.json"
    d = json.loads(path.read_text())
    out: dict[float, dict[str, float]] = defaultdict(dict)
    for v in d["videos"]:
        for m in v["matches"]:
            t = round(m["timestamp"], 2)
            for tm in m.get("top_matches") or []:
                iid = tm["instance_id"]
                sim = tm["similarity"]
                out[t][iid] = max(out[t].get(iid, 0.0), sim)
    return dict(out)


def build_item_signal(
    timeline: list[float],
    hoi_ts: set[float],
    siglip_by_t: dict[float, dict[str, float]],
    dino_by_t: dict[float, dict[str, float]],
    visual_class: str,
    instance_id: str,
    tau_siglip: float,
    tau_dino: float,
) -> list[tuple[float, bool, float, float]]:
    """Return [(t, active, siglip_score, dino_score)] along timeline."""
    out = []
    for t in timeline:
        s = siglip_by_t.get(t, {}).get(visual_class, 0.0)
        dv = dino_by_t.get(t, {}).get(instance_id, 0.0)
        active = (t in hoi_ts) and (s >= tau_siglip or dv >= tau_dino)
        out.append((t, active, s, dv))
    return out


def morphological_segments(
    signal: list[tuple[float, bool, float, float]],
    gap_close: float,
    min_duration: float,
) -> list[dict]:
    """Produce segments from boolean signal with close-then-open morphology."""
    # Step 1: extract raw active runs.
    runs: list[tuple[int, int]] = []  # [(start_idx, end_idx_inclusive)]
    i = 0
    n = len(signal)
    while i < n:
        if signal[i][1]:
            j = i
            while j + 1 < n and signal[j + 1][1]:
                j += 1
            runs.append((i, j))
            i = j + 1
        else:
            i += 1

    # Step 2: close small gaps between consecutive runs.
    if not runs:
        return []
    closed: list[tuple[int, int]] = [runs[0]]
    for s, e in runs[1:]:
        prev_s, prev_e = closed[-1]
        gap = signal[s][0] - signal[prev_e][0]
        if gap <= gap_close:
            closed[-1] = (prev_s, e)
        else:
            closed.append((s, e))

    # Step 3: drop short runs (open).
    segments: list[dict] = []
    for s, e in closed:
        duration = signal[e][0] - signal[s][0]
        if duration < min_duration:
            continue
        sig_scores = [signal[k][2] for k in range(s, e + 1)]
        dino_scores = [signal[k][3] for k in range(s, e + 1)]
        # Frames within the segment where the item was specifically detected
        hot_idxs = [k for k in range(s, e + 1) if signal[k][1]]
        segments.append({
            "start": signal[s][0],
            "end": signal[e][0],
            "duration": round(duration, 2),
            "n_frames": e - s + 1,
            "n_hot_frames": len(hot_idxs),
            "peak_siglip": round(max(sig_scores), 3),
            "peak_dino": round(max(dino_scores), 3),
            "avg_siglip_hot": round(
                sum(signal[k][2] for k in hot_idxs) / max(len(hot_idxs), 1), 3
            ),
            "avg_dino_hot": round(
                sum(signal[k][3] for k in hot_idxs) / max(len(hot_idxs), 1), 3
            ),
        })
    return segments


def load_owlv2_scene_points(participant: str, session: str) -> list[tuple[float, str]]:
    """Return [(timestamp, scene)] from scene_tags_owlv2.json, sorted by t.
    `scene` is one of {'storage', 'sink', 'stove', 'unknown'}. Missing file -> []."""
    path = outputs_dir(participant, session) / "scene_tags_owlv2.json"
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return []
    pts = [
        (float(v.get("timestamp", 0.0)), str(v.get("scene") or "unknown"))
        for v in (data.get("frames") or {}).values()
    ]
    pts.sort(key=lambda x: x[0])
    return pts


def aggregate_scene_anchor(
    owl_points: list[tuple[float, str]],
    start: float,
    end: float,
    pad: float = 0.5,
    min_frames: int = 2,
    min_ratio: float = 0.3,
) -> tuple[str | None, int, int]:
    """Pick the dominant non-unknown OWLv2 scene tag within [start-pad, end+pad].

    Returns (anchor, n_frames_in_range, n_unknown_in_range).
    Anchor is None if no non-unknown scene has at least `min_frames` frames
    AND at least `min_ratio` of the non-unknown frames."""
    window = [s for (t, s) in owl_points if start - pad <= t <= end + pad]
    n = len(window)
    counts = Counter(window)
    n_unknown = counts.get("unknown", 0)
    real = [(s, c) for s, c in counts.items() if s != "unknown"]
    if not real:
        return None, n, n_unknown
    real.sort(key=lambda x: -x[1])
    top_scene, top_ct = real[0]
    n_real = n - n_unknown
    if top_ct < min_frames and (n_real == 0 or top_ct / n_real < min_ratio):
        return None, n, n_unknown
    return top_scene, n, n_unknown


def load_gt_segments_by_item(actions: list[dict]) -> dict[str, list[tuple[float, float, str]]]:
    """instance_id -> [(start, end, stage)] from human annotations."""
    out: dict[str, list[tuple[float, float, str]]] = defaultdict(list)
    for a in actions:
        out[a["item"]].append((a["start"], a["end"], a.get("stage", "")))
    for iid in out:
        out[iid].sort()
    return dict(out)


def gt_range(gt: list[tuple[float, float, str]]) -> str:
    if not gt:
        return "(no GT)"
    stages = [s for _, _, s in gt if s]
    stage_summary = "→".join(dict.fromkeys(stages))  # preserve order, dedup
    return (
        f"GT {gt[0][0]:.1f}–{gt[-1][1]:.1f}s "
        f"({len(gt)} actions, {stage_summary})"
    )


def print_chronological_timeline(
    per_item_segments: dict[str, tuple[dict, list[dict]]],
):
    """Flatten per-item segments into one timeline, sorted by start, for planner view."""
    flat = []
    for _iid, (inv, segs) in per_item_segments.items():
        for s in segs:
            flat.append((s, inv))
    flat.sort(key=lambda x: x[0]["start"])

    print("\n--- PLANNER-STYLE CHRONOLOGICAL TIMELINE ---")
    print("(what the text-only planner would see instead of AdaTAD segments)\n")
    for seg, inv in flat:
        bits = []
        if seg["peak_siglip"] > 0:
            bits.append(f"{inv['visual_class']} {seg['peak_siglip']:.2f}(siglip)")
        if seg["peak_dino"] >= 0.15:
            tag = "visual+text" if seg["peak_siglip"] > 0 else "visual"
            bits.append(f"{inv['visual_class']} {seg['peak_dino']:.2f}({tag})")
        if not bits:
            bits.append(f"{inv['visual_class']} (weak)")
        print(
            f"[{seg['start']:6.1f}–{seg['end']:6.1f}s] "
            f"HOI-active  —  [{', '.join(bits)}]  "
            f"({seg['n_hot_frames']}/{seg['n_frames']} hot)"
        )


def build_session_segments(
    participant: str,
    session: str,
    tau_siglip: float,
    tau_dino: float,
    gap_close: float,
    min_duration: float,
    inventory_scope: str = "full",
) -> tuple[dict, dict[str, tuple[dict, list[dict]]]]:
    """Core builder. Returns (summary_stats, per_item_segments).

    per_item_segments: instance_id -> (inventory_entry, [segment dicts]).
    Keyed by instance_id (not visual_class) so two simultaneously-active
    purchase instances of the same product (e.g. two egg cartons) each
    get their own segment list.
    """
    inventory = load_inventory(participant, session, scope=inventory_scope)
    all_ts, hoi_ts = load_hoi_timestamps(participant, session)
    hoi_ts_set = set(hoi_ts)
    siglip_by_t = load_siglip_scores(participant, session)
    dino_by_t = load_dino_scores(participant, session)

    per_item: dict[str, tuple[dict, list[dict]]] = {}
    for inv in sorted(
        inventory, key=lambda x: (x["visual_class"].lower(), x["instance_id"])
    ):
        vc = inv["visual_class"]
        iid = inv["instance_id"]
        signal = build_item_signal(
            all_ts, hoi_ts_set, siglip_by_t, dino_by_t,
            vc, iid, tau_siglip, tau_dino,
        )
        segs = morphological_segments(signal, gap_close, min_duration)
        per_item[iid] = (inv, segs)

    summary = {
        "n_frames": len(all_ts),
        "n_hoi_frames": len(hoi_ts),
        "duration_s": (all_ts[-1] if all_ts else 0.0),
    }
    return summary, per_item


def write_segments_json(
    participant: str,
    session: str,
    tau_siglip: float,
    tau_dino: float,
    gap_close: float,
    min_duration: float,
    inventory_scope: str = "full",
) -> Path:
    """Write per_item_segments.json for a session. Returns output path."""
    summary, per_item = build_session_segments(
        participant, session, tau_siglip, tau_dino, gap_close, min_duration,
        inventory_scope=inventory_scope,
    )

    owl_points = load_owlv2_scene_points(participant, session)

    flat: list[dict] = []
    for _iid, (inv, segs) in per_item.items():
        for s in segs:
            anchor, n_owl, n_unk = aggregate_scene_anchor(
                owl_points, s["start"], s["end"]
            )
            flat.append({
                "instance_id": inv["instance_id"],
                "visual_class": inv["visual_class"],
                "start": s["start"],
                "end": s["end"],
                "duration": s["duration"],
                "n_frames": s["n_frames"],
                "n_hot_frames": s["n_hot_frames"],
                "peak_siglip": s["peak_siglip"],
                "peak_dino": s["peak_dino"],
                "avg_siglip_hot": s["avg_siglip_hot"],
                "avg_dino_hot": s["avg_dino_hot"],
                "scene_anchor": anchor,
                "scene_owl_frames": n_owl,
                "scene_unknown_frames": n_unk,
            })
    flat.sort(key=lambda x: x["start"])

    out_dir = outputs_dir(participant, session)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "per_item_segments.json"
    payload = {
        "participant": participant,
        "session": session,
        "params": {
            "tau_siglip": tau_siglip,
            "tau_dino": tau_dino,
            "gap_close": gap_close,
            "min_duration": min_duration,
            "inventory_scope": inventory_scope,
        },
        "summary": summary,
        "segments": flat,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    return out_path


def print_report(
    participant: str,
    session: str,
    tau_siglip: float,
    tau_dino: float,
    gap_close: float,
    min_duration: float,
    chrono: bool = False,
    inventory_scope: str = "full",
):
    inventory = load_inventory(participant, session, scope=inventory_scope)
    all_ts, hoi_ts = load_hoi_timestamps(participant, session)
    hoi_ts_set = set(hoi_ts)
    siglip_by_t = load_siglip_scores(participant, session)
    dino_by_t = load_dino_scores(participant, session)
    try:
        actions = load_actions(participant, session)
    except FileNotFoundError:
        actions = []
    gt_by_item = load_gt_segments_by_item(actions)

    print(f"=== Session {session} ({participant}) ===")
    print(f"Params: tau_siglip={tau_siglip}, tau_dino={tau_dino}, "
          f"gap_close={gap_close}s, min_duration={min_duration}s")
    print(f"Timeline: {len(all_ts)} frames, "
          f"{len(hoi_ts)} with object_contact "
          f"({len(hoi_ts) / max(len(all_ts), 1) * 100:.0f}%)")
    print(f"Duration: {all_ts[0]:.1f}–{all_ts[-1]:.1f}s")
    print()

    per_item: dict[str, tuple[dict, list[dict]]] = {}
    for inv in sorted(
        inventory, key=lambda x: (x["visual_class"].lower(), x["instance_id"])
    ):
        vc = inv["visual_class"]
        iid = inv["instance_id"]
        signal = build_item_signal(
            all_ts, hoi_ts_set, siglip_by_t, dino_by_t,
            vc, iid, tau_siglip, tau_dino,
        )
        hot_n = sum(1 for _, a, _, _ in signal if a)
        segs = morphological_segments(signal, gap_close, min_duration)
        per_item[iid] = (inv, segs)

        gt = gt_by_item.get(iid, [])
        print(f"{vc}  ({iid})")
        print(f"  raw active frames: {hot_n}  |  {gt_range(gt)}")
        if not segs:
            print(f"  NO SEGMENTS")
        else:
            for seg in segs:
                print(
                    f"  [{seg['start']:6.1f}–{seg['end']:6.1f}s] "
                    f"dur={seg['duration']:5.1f}s "
                    f"hot={seg['n_hot_frames']}/{seg['n_frames']}  "
                    f"siglip peak={seg['peak_siglip']:.2f} "
                    f"hot-avg={seg['avg_siglip_hot']:.2f}  "
                    f"dino peak={seg['peak_dino']:.2f} "
                    f"hot-avg={seg['avg_dino_hot']:.2f}"
                )
        if gt:
            print(f"  GT actions:")
            for s, e, stage in gt:
                print(f"    [{s:6.1f}–{e:6.1f}s] {stage}")
        print()

    if chrono:
        print_chronological_timeline(per_item)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--participant", default="kailai")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--session", help="Single session (default: first).")
    group.add_argument("--all", action="store_true",
                       help="Iterate all sessions for the participant.")
    parser.add_argument("--tau-siglip", type=float, default=0.15)
    parser.add_argument("--tau-dino", type=float, default=0.15)
    parser.add_argument("--gap-close", type=float, default=2.0)
    parser.add_argument("--min-duration", type=float, default=1.5)
    parser.add_argument("--chrono", action="store_true",
                        help="Also print chronological planner-style timeline.")
    parser.add_argument("--write", action="store_true",
                        help="Write per_item_segments.json to each session's outputs dir.")
    parser.add_argument("--inventory-scope", choices=["full", "session"], default="full",
                        help="Which items to track per-frame. "
                             "'full' = all items in stock at session time (default, "
                             "matches real home deployment); "
                             "'session' = GT-annotated subset only.")
    args = parser.parse_args()

    if args.all:
        sessions = get_sessions(args.participant)
    elif args.session:
        sessions = [args.session]
    else:
        sessions = [get_sessions(args.participant)[0]]

    for session in sessions:
        try:
            if args.write:
                out = write_segments_json(
                    args.participant, session,
                    args.tau_siglip, args.tau_dino,
                    args.gap_close, args.min_duration,
                    inventory_scope=args.inventory_scope,
                )
                with open(out) as f:
                    n = len(json.load(f)["segments"])
                print(f"[{session}] wrote {out} ({n} segments)")
            else:
                print_report(
                    args.participant, session,
                    args.tau_siglip, args.tau_dino,
                    args.gap_close, args.min_duration,
                    chrono=args.chrono,
                    inventory_scope=args.inventory_scope,
                )
        except FileNotFoundError as e:
            print(f"[{session}] SKIP: {e}")


if __name__ == "__main__":
    main()

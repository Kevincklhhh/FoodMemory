#!/usr/bin/env python3
"""AVP Round 1 (remaining-only, Iterative): multi-round batched plan-observe loop.

Rebased on 06_avp_round1_remaining_CandList_HOI_PerItem.py. Keeps:
  - CandList candidate-aware observer (per_instance + handling_status).
  - HOI per-hand detections (grasp, obj_touch, other-hand cues).
  - Per-item evidence blocks (one chronological section per visual_class).

Adds a batched plan-observe loop (not the one-window-per-round form
from earlier drafts — that was inefficient on large sessions):

  Round 0 (planner):
    Reads per-item HOI evidence + inventory. Initializes a
    `food_journeys` sketch for every active visual_class, marks
    evidence-free iids under `skipped_items`, and emits a BATCH of
    observation_windows covering ALL active visual_classes (one
    window per class is typical; a transparent item may get two).

  Round 0 observer pass:
    Every window in the batch runs as an independent observer call.
    Each observer returns per_instance + `window_observation` (free
    text: scene, stock-container vs derivative, dispensing action,
    fill estimate) + `needs_followup` (structured per-iid reasons a
    later window could help).

  Round 1+ (planner, persistent chat):
    Receives the aggregated observer batch JSON + a rolling ledger
    (resolved / unresolved / given_up). Refines food_journeys using
    the observer text as ground truth, flips status fields, then
    emits either `action.type = "stop"` (all items resolved or
    given_up) or a SMALLER batch of observation_windows for
    still-unresolved items with a viable alternative angle.

  Termination: planner emits stop; OR every tracked iid is
  resolved/given_up (short-circuit); OR --max-rounds hit; OR
  frame budget exhausted.

The planner's window-choice rule:
  - Opaque package → target the DISPENSING window (pour/crack/scoop).
  - Transparent package → target a STOCK-CONTAINER-VISIBLE window,
    ideally AFTER dispensing, preferring `storage` returns; beware
    late-journey derivative hits (bowl of cracked eggs, plate of
    grated cheese) that DINO cannot distinguish from the package.
  - Observer feedback drives replanning: followups like "container
    left view" or "cannot disambiguate siblings" map to a later
    storage window or a retrieval-window tie-breaker.

This variant is designed to trade off VLM frames against recall+CNPE:
compared to PerItem (which blasts journey + dense windows for every
active item up front), the iterative loop spends frames only where an
earlier round left an item unresolved.

Usage:
  python system_design/06_avp_round1_remaining_Iterative.py \
      --participant kailai --tag Iterative_v1
  # smoke test (session 181229, segment-compressed evidence, flicker-permissive):
  python system_design/06_avp_round1_remaining_Iterative.py \
      --participant kailai --session 20260318-181229 \
      --tag Iterative_smoke_v3 --max-rounds 3 --max-frames 30 \
      --evidence-mode segments
"""

import argparse
import importlib.util
import json
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))
from frame_sampling import (
    extract_segments_frames as _extract_segments_frames,
)


def _load_per_item_segments_module():
    """Import sibling `05b_per_item_segments.py` (name starts with digit)."""
    spec = importlib.util.spec_from_file_location(
        "per_item_segments_05b",
        Path(__file__).resolve().parent / "05b_per_item_segments.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_PER_ITEM_SEGMENTS = _load_per_item_segments_module()
from utils import (
    get_session_clips,
    get_sessions,
    hands23_dir,
    load_inventory,
    load_ledger,
    outputs_dir,
    participant_dir,
)

_KITCHEN_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(_KITCHEN_DIR / ".env")

MAX_IMAGES = 50

CACHE_DIR = Path(__file__).resolve().parent.parent / "cache" / "avp_remaining_Iterative"

QWEN_URL = "http://saltyfish.eecs.umich.edu:8000/v1/chat/completions"
QWEN_MODEL_DEFAULT = "Qwen/Qwen3-VL-30B-A3B-Instruct"

VLLM_ENDPOINTS = {
    "qwen": ("http://saltyfish.eecs.umich.edu:8000/v1/chat/completions", "Qwen/Qwen3-VL-30B-A3B-Instruct"),
    "gemma": ("http://saltyfish.eecs.umich.edu:8000/v1/chat/completions", "google/gemma-4-31B-it"),
}

OUTPUT_PREFIX = "avp_Iterative_remaining"

DEFAULT_MAX_ROUNDS = 3  # round 0 (full batch) + up to 2 replans; usually stops at 1–2
# No session-wide frame cap and no per-round window cap: each observer call
# is an independent API call, so the only hard limit is MAX_IMAGES per call
# (GPT's image-input limit). The planner is expected to emit one window per
# active visual_class, which is naturally bounded by the inventory size.


# ---------------------------------------------------------------------------
# Clients
# ---------------------------------------------------------------------------

def make_client(model: str = "gpt-5.4"):
    from openai import AzureOpenAI
    api_key = os.getenv("AZURE_OPENAI_API_KEY_2") or os.getenv("AZURE_OPENAI_API_KEY")
    endpoint = (
        os.getenv("AZURE_OPENAI_ENDPOINT_2")
        or os.getenv("AZURE_OPENAI_ENDPOINT")
        or ""
    ).strip()
    if not api_key or not endpoint:
        raise ValueError("Missing Azure OpenAI API credentials")
    return AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version="2025-03-01-preview",
    )


# ---------------------------------------------------------------------------
# Per-frame data loading (unchanged from PerItem)
# ---------------------------------------------------------------------------

def load_hoi_timestamps(participant: str, session: str) -> tuple[list[float], set[float]]:
    path = hands23_dir(participant, session) / f"{participant}_{session}_hands23_results.json"
    d = json.loads(path.read_text())
    all_ts: set[float] = set()
    hoi_ts: set[float] = set()
    for v in d["videos"]:
        for f in v["frames"]:
            t = round(f["session_timestamp_s"], 2)
            all_ts.add(t)
            for det in f["detections"]:
                if det.get("contact_state") == "object_contact":
                    hoi_ts.add(t)
                    break
    return sorted(all_ts), hoi_ts


def load_siglip_by_t(participant: str, session: str) -> dict[float, dict[str, dict[str, float]]]:
    path = hands23_dir(participant, session) / f"{participant}_{session}_siglip_matches.json"
    if not path.exists():
        return {}
    d = json.loads(path.read_text())
    out: dict[float, dict[str, dict[str, float]]] = {}
    for v in d["videos"]:
        for m in v["matches"]:
            if m.get("contact_state") != "object_contact":
                continue
            t = round(m["timestamp"], 2)
            hs = m.get("hand_side") or "unknown"
            per_hand = out.setdefault(t, {}).setdefault(hs, {})
            for tm in m.get("top_matches") or []:
                name = tm["food_name"]
                sim = float(tm["similarity"])
                per_hand[name] = max(per_hand.get(name, 0.0), sim)
    return out


def load_dino_by_t(participant: str, session: str) -> dict[float, dict[str, dict[str, float]]]:
    path = hands23_dir(participant, session) / f"{participant}_{session}_dino_matches.json"
    if not path.exists():
        return {}
    d = json.loads(path.read_text())
    out: dict[float, dict[str, dict[str, float]]] = {}
    for v in d["videos"]:
        for m in v["matches"]:
            if m.get("contact_state") != "object_contact":
                continue
            t = round(m["timestamp"], 2)
            hs = m.get("hand_side") or "unknown"
            per_hand = out.setdefault(t, {}).setdefault(hs, {})
            for tm in m.get("top_matches") or []:
                iid = tm["instance_id"]
                sim = float(tm["similarity"])
                per_hand[iid] = max(per_hand.get(iid, 0.0), sim)
    return out


def load_hoi_details_by_t(participant: str, session: str) -> dict[float, dict[str, dict]]:
    path = hands23_dir(participant, session) / f"{participant}_{session}_hands23_results.json"
    if not path.exists():
        return {}
    d = json.loads(path.read_text())
    out: dict[float, dict[str, dict]] = {}
    for v in d["videos"]:
        for f in v["frames"]:
            t = round(f["session_timestamp_s"], 2)
            for det in f["detections"]:
                if det.get("contact_state") != "object_contact":
                    continue
                hs = det.get("hand_side") or "unknown"
                out.setdefault(t, {})[hs] = {
                    "grasp": det.get("grasp"),
                    "obj_touch": det.get("obj_touch"),
                    "contact_state": det.get("contact_state"),
                }
    return out


def load_owlv2_scene_by_t(participant: str, session: str) -> dict[float, str]:
    path = outputs_dir(participant, session) / "scene_tags_owlv2.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}
    out: dict[float, str] = {}
    for v in (data.get("frames") or {}).values():
        t = round(float(v.get("timestamp", 0.0)), 2)
        out[t] = str(v.get("scene") or "unknown")
    return out


def load_transparency_profile(participant: str) -> dict:
    path = participant_dir(participant) / "confusable_profile.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}
    out: dict = {}
    for iid, entry in (data.get("items") or {}).items():
        if isinstance(entry, dict) and "is_transparent_package" in entry:
            out[iid] = bool(entry["is_transparent_package"])
    return out


# ---------------------------------------------------------------------------
# Per-item evidence formatter (unchanged from PerItem) — also returns the set
# of visual_classes and iids that have per-frame evidence, for the loop's
# "candidates worth tracking" logic.
# ---------------------------------------------------------------------------

_HAND_LABEL = {"left_hand": "L", "right_hand": "R"}
_HIDDEN_OBJ_TOUCH = {"neither_held", "neither_touched", None, ""}


def format_per_item_evidence(
    hoi_ts_sorted: list[float],
    siglip_by_t: dict[float, dict[str, dict[str, float]]],
    dino_by_t: dict[float, dict[str, dict[str, float]]],
    scene_by_t: dict[float, str],
    hoi_details_by_t: dict[float, dict[str, dict]],
    inventory: list[dict],
    min_score: float = 0.15,
    transparency_by_iid: dict | None = None,
) -> tuple[str, dict, set[str], set[str]]:
    """Same format as CandList_HOI_PerItem, returns extra sets for loop bookkeeping.

    Returns (text, stats, active_vcs, active_iids):
      - active_vcs: visual_classes with ≥1 hit above min_score
      - active_iids: iids whose class has ≥1 hit (candidate tracking pool)
    """
    transparency_by_iid = transparency_by_iid or {}

    inv_iids = {inv["instance_id"] for inv in inventory}
    inv_vcs = {inv["visual_class"] for inv in inventory}
    iid_to_vc: dict[str, str] = {inv["instance_id"]: inv["visual_class"] for inv in inventory}
    vc_to_iids: dict[str, list[str]] = defaultdict(list)
    inv_by_iid = {inv["instance_id"]: inv for inv in inventory}
    ordered_vcs: list[str] = []
    seen_vc: set[str] = set()
    for inv in inventory:
        vc = inv["visual_class"]
        vc_to_iids[vc].append(inv["instance_id"])
        if vc not in seen_vc:
            seen_vc.add(vc)
            ordered_vcs.append(vc)

    rows_by_vc: dict[str, list[dict]] = defaultdict(list)
    t_hand_primary: dict[float, dict[str, tuple[str, float]]] = {}
    t_hand_hoi: dict[float, dict[str, tuple[str | None, str | None]]] = {}

    for t in hoi_ts_sorted:
        dino_by_hand = dino_by_t.get(t) or {}
        sig_by_hand = siglip_by_t.get(t) or {}
        hands_detail = hoi_details_by_t.get(t) or {}
        scene = scene_by_t.get(t, "unknown") or "unknown"

        for hs, detail in hands_detail.items():
            g = detail.get("grasp")
            ot = detail.get("obj_touch")
            if ot in _HIDDEN_OBJ_TOUCH:
                ot = None
            t_hand_hoi.setdefault(t, {})[hs] = (g, ot)

        all_hands = set(dino_by_hand) | set(sig_by_hand) | set(hands_detail)
        for hs in all_hands:
            dino_vc: dict[str, float] = {}
            for iid, s in (dino_by_hand.get(hs) or {}).items():
                if iid not in inv_iids or s < min_score:
                    continue
                vc = iid_to_vc.get(iid)
                if not vc:
                    continue
                dino_vc[vc] = max(dino_vc.get(vc, 0.0), s)
            sig_vc: dict[str, float] = {
                vc: s for vc, s in (sig_by_hand.get(hs) or {}).items()
                if vc in inv_vcs and s >= min_score
            }
            if not dino_vc and not sig_vc:
                continue

            if dino_vc:
                primary_vc = max(dino_vc, key=lambda k: dino_vc[k])
                primary_score = dino_vc[primary_vc]
            else:
                primary_vc = max(sig_vc, key=lambda k: sig_vc[k])
                primary_score = sig_vc[primary_vc]
            t_hand_primary.setdefault(t, {})[hs] = (primary_vc, primary_score)

            detail = hands_detail.get(hs) or {}
            grasp = detail.get("grasp")
            obj_touch = detail.get("obj_touch")
            if obj_touch in _HIDDEN_OBJ_TOUCH:
                obj_touch = None

            for vc in set(dino_vc) | set(sig_vc):
                rows_by_vc[vc].append({
                    "t": t,
                    "hand": hs,
                    "scene": scene,
                    "grasp": grasp,
                    "touch": obj_touch,
                    "dino": dino_vc.get(vc),
                    "sig": sig_vc.get(vc),
                })

    lines: list[str] = []
    n_items_emitted = 0
    n_rows_emitted = 0
    active_vcs: set[str] = set()
    active_iids: set[str] = set()

    for vc in ordered_vcs:
        rows = rows_by_vc.get(vc) or []
        if not rows:
            continue
        n_items_emitted += 1
        active_vcs.add(vc)
        iids = vc_to_iids[vc]
        for iid in iids:
            active_iids.add(iid)

        n_transp = 0
        n_opaque = 0
        for iid in iids:
            if iid in transparency_by_iid:
                if transparency_by_iid[iid]:
                    n_transp += 1
                else:
                    n_opaque += 1
            else:
                vis = inv_by_iid[iid].get("visible_during_interaction", True)
                if vis:
                    n_transp += 1
                else:
                    n_opaque += 1
        tp_tag = "[transparent]" if n_transp >= n_opaque else "[opaque]"

        caps = {str(inv_by_iid[iid].get("package_amount", "?")) for iid in iids}
        pkg_str = next(iter(caps)) if len(caps) == 1 else "varies"
        unit = inv_by_iid[iids[0]].get("unit", "g")
        unit_label = "g" if unit == "g" else "ct"

        lines.append(
            f'### "{vc}" (iids: {", ".join(iids)}; {tp_tag}; pkg={pkg_str}, unit={unit_label})'
        )
        for r in sorted(rows, key=lambda x: x["t"]):
            label = _HAND_LABEL.get(r["hand"], "U")
            hdr_parts: list[str] = []
            if r["grasp"]:
                hdr_parts.append(r["grasp"])
            if r["touch"]:
                hdr_parts.append(r["touch"])
            hand_hdr = f"{label}[{' '.join(hdr_parts)}]" if hdr_parts else f"{label}[]"

            score_bits: list[str] = []
            if r["dino"] is not None:
                score_bits.append(f"dino={r['dino']:.2f}")
            if r["sig"] is not None:
                score_bits.append(f"sig={r['sig']:.2f}")
            scores_str = " ".join(score_bits)

            other_str = ""
            for oh, (ovc, osc) in (t_hand_primary.get(r["t"]) or {}).items():
                if oh == r["hand"] or ovc == vc:
                    continue
                olabel = _HAND_LABEL.get(oh, "U")
                ohoi = (t_hand_hoi.get(r["t"]) or {}).get(oh) or (None, None)
                ohoi_parts = [p for p in (ohoi[0], ohoi[1]) if p]
                ohoi_str = f"[{' '.join(ohoi_parts)}] " if ohoi_parts else ""
                other_str = f'   (other: {olabel}{ohoi_str}"{ovc}"={osc:.2f})'
                break
            if not other_str:
                for oh, (g, ot) in (t_hand_hoi.get(r["t"]) or {}).items():
                    if oh == r["hand"]:
                        continue
                    if (t_hand_primary.get(r["t"]) or {}).get(oh):
                        continue
                    parts = [p for p in (g, ot) if p]
                    if not parts:
                        continue
                    olabel = _HAND_LABEL.get(oh, "U")
                    other_str = f'   (other: {olabel}[{" ".join(parts)}] no_inv_match)'
                    break

            lines.append(
                f'  {r["t"]:7.1f}s  {r["scene"]:<8} {hand_hdr:<28} {scores_str}{other_str}'
            )
            n_rows_emitted += 1
        lines.append("")

    stats = {
        "n_hoi_frames_total": len(hoi_ts_sorted),
        "n_items_emitted": n_items_emitted,
        "n_rows_emitted": n_rows_emitted,
        "min_score": min_score,
    }
    return "\n".join(lines), stats, active_vcs, active_iids


# ---------------------------------------------------------------------------
# Segment-compressed evidence formatter (re-uses 05b_per_item_segments).
#
# Rationale: the per-frame evidence block above is ~15k tokens; for a
# multi-round loop that pays to re-send this on every planner call, it's
# wasteful. 05b emits HOI-gated item segments (start/end/peak_scores/scene),
# so one row per segment, typically 1–3 rows per item. Total block ~1–2k
# tokens.
#
# Flicker caveat: a strict tau + min_duration filter drops real usage where
# the item is briefly obstructed (on–off–on–off). We mitigate by:
#   (1) defaulting to permissive params (lower tau, larger gap_close, small
#       min_duration) so flickers merge into one segment.
#   (2) emitting a compact "flicker" note for items that have per-frame HOI
#       hits above min_score but still produced no segment, so the planner
#       can still initialize a food_journey.
# ---------------------------------------------------------------------------

def _scene_distribution_in_segment(
    start: float,
    end: float,
    owl_points: list[tuple[float, str]],
    pad: float = 0.5,
) -> tuple[str | None, str]:
    """Dominant non-unknown scene + compact distribution string."""
    from collections import Counter
    window = [s for (t, s) in owl_points if start - pad <= t <= end + pad]
    if not window:
        return None, ""
    c = Counter(window)
    real = {s: v for s, v in c.items() if s != "unknown"}
    top = max(real, key=lambda k: real[k]) if real else None
    # Compact dist: top-2 non-unknown
    sorted_scenes = sorted(real.items(), key=lambda x: -x[1])[:2]
    n_unk = c.get("unknown", 0)
    parts = [f"{s}:{n}" for s, n in sorted_scenes]
    if n_unk:
        parts.append(f"unk:{n_unk}")
    return top, ",".join(parts)


def format_per_item_segments_evidence(
    participant: str,
    session: str,
    inventory: list[dict],
    transparency_by_iid: dict | None,
    min_score: float,
    siglip_by_t: dict[float, dict[str, dict[str, float]]],
    dino_by_t: dict[float, dict[str, dict[str, float]]],
    hoi_sorted: list[float],
    seg_tau_siglip: float,
    seg_tau_dino: float,
    seg_gap_close: float,
    seg_min_duration: float,
    inventory_scope: str = "full",
    flicker_min_score: float = 0.15,
    flicker_min_hits: int = 2,
    flicker_peak_score: float = 0.25,
) -> tuple[str, dict, set[str], set[str]]:
    """Segment-compressed counterpart to `format_per_item_evidence`.

    Emits one block per visual_class with rows = (start, end, dur, hot/total,
    peak_siglip, peak_dino, scene_anchor). Falls back to a flicker note for
    items with per-frame hits ≥ min_score but no segment.

    Returns (text, stats, active_vcs, active_iids).
    """
    transparency_by_iid = transparency_by_iid or {}

    _, per_item = _PER_ITEM_SEGMENTS.build_session_segments(
        participant, session,
        tau_siglip=seg_tau_siglip,
        tau_dino=seg_tau_dino,
        gap_close=seg_gap_close,
        min_duration=seg_min_duration,
        inventory_scope=inventory_scope,
    )
    owl_points = _PER_ITEM_SEGMENTS.load_owlv2_scene_points(participant, session)

    inv_by_iid = {inv["instance_id"]: inv for inv in inventory}
    inv_iids = set(inv_by_iid)
    vc_to_iids: dict[str, list[str]] = defaultdict(list)
    ordered_vcs: list[str] = []
    seen_vc: set[str] = set()
    for inv in inventory:
        vc = inv["visual_class"]
        vc_to_iids[vc].append(inv["instance_id"])
        if vc not in seen_vc:
            seen_vc.add(vc)
            ordered_vcs.append(vc)

    # Flicker map: visual_class -> list[(t, peak_sig, peak_dino)] built from
    # per-hand data (not 05b's aggregated) so we can surface per-frame hits
    # even when no segment was emitted.
    iid_to_vc = {inv["instance_id"]: inv["visual_class"] for inv in inventory}
    flicker_floor = max(min_score, flicker_min_score)
    flicker_hits: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    for t in hoi_sorted:
        dino_by_hand = dino_by_t.get(t) or {}
        sig_by_hand = siglip_by_t.get(t) or {}
        best_sig: dict[str, float] = {}
        best_dino: dict[str, float] = {}
        for hs in set(dino_by_hand) | set(sig_by_hand):
            for iid, s in (dino_by_hand.get(hs) or {}).items():
                if iid not in inv_iids or s < flicker_floor:
                    continue
                vc = iid_to_vc.get(iid)
                if not vc:
                    continue
                best_dino[vc] = max(best_dino.get(vc, 0.0), s)
            for vc, s in (sig_by_hand.get(hs) or {}).items():
                if vc in vc_to_iids and s >= flicker_floor:
                    best_sig[vc] = max(best_sig.get(vc, 0.0), s)
        for vc in set(best_sig) | set(best_dino):
            flicker_hits[vc].append((t, best_sig.get(vc, 0.0), best_dino.get(vc, 0.0)))

    lines: list[str] = []
    n_items_emitted = 0
    n_rows_emitted = 0
    active_vcs: set[str] = set()
    active_iids: set[str] = set()

    for vc in ordered_vcs:
        iids = vc_to_iids[vc]
        # Collect 05b segments across all iids of this vc.
        vc_segs: list[tuple[str, dict]] = []
        for iid in iids:
            _, segs = per_item.get(iid, ({}, []))
            for s in segs:
                vc_segs.append((iid, s))
        flickers = flicker_hits.get(vc, [])
        flicker_peak = max(
            (max(s, d) for _, s, d in flickers), default=0.0
        )

        if not vc_segs and (
            len(flickers) < flicker_min_hits
            or flicker_peak < flicker_peak_score
        ):
            continue
        n_items_emitted += 1
        active_vcs.add(vc)
        for iid in iids:
            active_iids.add(iid)

        # Transparency / package header (same rule as per-frame formatter).
        n_transp = 0
        n_opaque = 0
        for iid in iids:
            if iid in transparency_by_iid:
                if transparency_by_iid[iid]:
                    n_transp += 1
                else:
                    n_opaque += 1
            else:
                vis = inv_by_iid[iid].get("visible_during_interaction", True)
                if vis:
                    n_transp += 1
                else:
                    n_opaque += 1
        tp_tag = "[transparent]" if n_transp >= n_opaque else "[opaque]"
        caps = {str(inv_by_iid[iid].get("package_amount", "?")) for iid in iids}
        pkg_str = next(iter(caps)) if len(caps) == 1 else "varies"
        unit = inv_by_iid[iids[0]].get("unit", "g")
        unit_label = "g" if unit == "g" else "ct"

        lines.append(
            f'### "{vc}" (iids: {", ".join(iids)}; {tp_tag}; pkg={pkg_str}, unit={unit_label})'
        )

        if vc_segs:
            vc_segs.sort(key=lambda x: x[1]["start"])
            for iid, s in vc_segs:
                _, scene_dist = _scene_distribution_in_segment(
                    s["start"], s["end"], owl_points
                )
                scene_str = scene_dist if scene_dist else "unknown"
                lines.append(
                    f"  [{s['start']:6.1f}-{s['end']:6.1f}s] "
                    f"dur={s['duration']:4.1f}s "
                    f"hot={s['n_hot_frames']}/{s['n_frames']} "
                    f"sig_peak={s['peak_siglip']:.2f} dino_peak={s['peak_dino']:.2f} "
                    f"scene={scene_str}  iid={iid}"
                )
                n_rows_emitted += 1
        else:
            # No coherent segment but per-frame flickers exist — surface a note.
            ts = sorted(t for t, _, _ in flickers)
            peak_sig = max((s for _, s, _ in flickers), default=0.0)
            peak_dino = max((d for _, _, d in flickers), default=0.0)
            span = f"{ts[0]:.1f}-{ts[-1]:.1f}s" if ts else "?"
            sample = ", ".join(f"{t:.1f}" for t in ts[:6])
            if len(ts) > 6:
                sample += ", ..."
            lines.append(
                f"  FLICKER ONLY: {len(ts)} HOI-contact hits over {span} "
                f"(sig_peak={peak_sig:.2f}, dino_peak={peak_dino:.2f}; "
                f"sample ts: {sample}) — no coherent segment formed under "
                f"tau_sig={seg_tau_siglip}, tau_dino={seg_tau_dino}, "
                f"gap_close={seg_gap_close}s, min_dur={seg_min_duration}s"
            )
            n_rows_emitted += 1
        lines.append("")

    stats = {
        "n_hoi_frames_total": len(hoi_sorted),
        "n_items_emitted": n_items_emitted,
        "n_rows_emitted": n_rows_emitted,
        "min_score": min_score,
        "seg_tau_siglip": seg_tau_siglip,
        "seg_tau_dino": seg_tau_dino,
        "seg_gap_close": seg_gap_close,
        "seg_min_duration": seg_min_duration,
    }
    return "\n".join(lines), stats, active_vcs, active_iids


def format_chronological_segments_evidence(
    participant: str,
    session: str,
    inventory: list[dict],
    transparency_by_iid: dict | None,
    min_score: float,
    siglip_by_t: dict[float, dict[str, dict[str, float]]],
    dino_by_t: dict[float, dict[str, dict[str, float]]],
    hoi_sorted: list[float],
    seg_tau_siglip: float,
    seg_tau_dino: float,
    seg_gap_close: float,
    seg_min_duration: float,
    inventory_scope: str = "full",
    flicker_min_score: float = 0.15,
    flicker_min_hits: int = 2,
    flicker_peak_score: float = 0.25,
) -> tuple[str, dict, set[str], set[str]]:
    """Chronological counterpart to `format_per_item_segments_evidence`.

    Emits a single timeline sorted by start-time. Each row is one segment
    (item × interval) — overlapping segments from different items appear as
    consecutive rows. Items without coherent segments are listed in a small
    FLICKER tail.
    """
    transparency_by_iid = transparency_by_iid or {}

    _, per_item = _PER_ITEM_SEGMENTS.build_session_segments(
        participant, session,
        tau_siglip=seg_tau_siglip,
        tau_dino=seg_tau_dino,
        gap_close=seg_gap_close,
        min_duration=seg_min_duration,
        inventory_scope=inventory_scope,
    )
    owl_points = _PER_ITEM_SEGMENTS.load_owlv2_scene_points(participant, session)

    inv_by_iid = {inv["instance_id"]: inv for inv in inventory}
    inv_iids = set(inv_by_iid)
    iid_to_vc = {iid: inv["visual_class"] for iid, inv in inv_by_iid.items()}
    vc_to_iids: dict[str, list[str]] = defaultdict(list)
    for inv in inventory:
        vc_to_iids[inv["visual_class"]].append(inv["instance_id"])

    # Collect every (start, segment, iid) and sort chronologically.
    flat: list[tuple[float, str, dict]] = []
    active_vcs: set[str] = set()
    active_iids: set[str] = set()
    for iid, (_inv, segs) in per_item.items():
        if iid not in inv_iids:
            continue
        for s in segs:
            flat.append((s["start"], iid, s))
    flat.sort(key=lambda x: (x[0], iid_to_vc.get(x[1], "")))

    rows: list[tuple[float, str]] = []
    n_rows_emitted = 0
    for _t0, iid, s in flat:
        active_vcs.add(iid_to_vc[iid])
        active_iids.add(iid)
        _, scene_dist = _scene_distribution_in_segment(s["start"], s["end"], owl_points)
        scene_str = scene_dist if scene_dist else "unknown"
        rows.append((
            s["start"],
            f"[{s['start']:6.1f}-{s['end']:6.1f}s] dur={s['duration']:4.1f}s "
            f"dino={s['peak_dino']:.2f} scene={scene_str}  \"{iid_to_vc[iid]}\""
        ))
        n_rows_emitted += 1

    # Flicker fallback: vcs with per-frame HOI hits >= flicker_floor still
    # surface a row, so the planner can emit a window over the flicker
    # timestamps. Mirrors the per-item-segments mode behaviour at
    # format_per_item_segments_evidence. flicker_floor = max(min_score,
    # flicker_min_score) so flickers can be held to a stricter bar than
    # segment hits without ever being looser.
    flicker_floor = max(min_score, flicker_min_score)
    flicker_hits: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    for t in hoi_sorted:
        dino_by_hand = dino_by_t.get(t) or {}
        sig_by_hand = siglip_by_t.get(t) or {}
        best_sig: dict[str, float] = {}
        best_dino: dict[str, float] = {}
        for hs in set(dino_by_hand) | set(sig_by_hand):
            for iid, s in (dino_by_hand.get(hs) or {}).items():
                if iid not in inv_iids or s < flicker_floor:
                    continue
                vc = iid_to_vc.get(iid)
                if not vc:
                    continue
                best_dino[vc] = max(best_dino.get(vc, 0.0), s)
            for vc, s in (sig_by_hand.get(hs) or {}).items():
                if vc in vc_to_iids and s >= flicker_floor:
                    best_sig[vc] = max(best_sig.get(vc, 0.0), s)
        for vc in set(best_sig) | set(best_dino):
            flicker_hits[vc].append((t, best_sig.get(vc, 0.0), best_dino.get(vc, 0.0)))

    # Build per-vc list of existing segment intervals so flicker hits that
    # fall INSIDE an existing segment can be skipped (no need to duplicate
    # what's already in the timeline). Hits OUTSIDE every existing segment
    # for the same vc still get rendered, even when other segments exist —
    # otherwise a vc with one weak early segment and a much stronger later
    # burst that 05b couldn't form into a segment would silently lose the
    # later burst.
    vc_segment_intervals: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for iid, (_inv, segs) in per_item.items():
        if iid not in inv_iids:
            continue
        vc = iid_to_vc[iid]
        for s in segs:
            vc_segment_intervals[vc].append((s["start"], s["end"]))

    n_flicker = 0
    n_flicker_clusters = 0
    for vc, hits in flicker_hits.items():
        if not hits:
            continue
        intervals = vc_segment_intervals.get(vc, [])
        # Drop hits that fall inside an existing segment for this vc.
        outside_hits = [
            h for h in hits
            if not any(s <= h[0] <= e for (s, e) in intervals)
        ]
        if not outside_hits:
            continue
        # Cluster hits using the same gap_close as 05b's segment builder so
        # each cluster of contiguous hits is its own row.
        sorted_hits = sorted(outside_hits, key=lambda x: x[0])
        clusters: list[list[tuple[float, float, float]]] = [[sorted_hits[0]]]
        for h in sorted_hits[1:]:
            if h[0] - clusters[-1][-1][0] <= seg_gap_close:
                clusters[-1].append(h)
            else:
                clusters.append([h])
        kept_any = False
        for cluster_hits in clusters:
            if len(cluster_hits) < flicker_min_hits:
                continue
            cluster_peak = max(max(h[1], h[2]) for h in cluster_hits)
            if cluster_peak < flicker_peak_score:
                continue
            ts_c = [h[0] for h in cluster_hits]
            peak_dino_c = max(h[2] for h in cluster_hits)
            start, end = ts_c[0], ts_c[-1]
            duration = max(end - start, 0.0)
            _, scene_dist = _scene_distribution_in_segment(start, end, owl_points)
            scene_str = scene_dist if scene_dist else "unknown"
            rows.append((
                start,
                f"[{start:6.1f}-{end:6.1f}s] dur={duration:4.1f}s "
                f"dino={peak_dino_c:.2f} scene={scene_str}  \"{vc}\""
            ))
            n_rows_emitted += 1
            n_flicker_clusters += 1
            kept_any = True
        if not kept_any:
            continue
        active_vcs.add(vc)
        for iid in vc_to_iids.get(vc, []):
            active_iids.add(iid)
        n_flicker += 1

    rows.sort(key=lambda x: x[0])
    lines = [r[1] for r in rows]

    stats = {
        "n_hoi_frames_total": len(hoi_sorted),
        "n_items_emitted": len(active_vcs),
        "n_rows_emitted": n_rows_emitted,
        "n_flicker_vcs": n_flicker,
        "n_flicker_clusters": n_flicker_clusters,
        "min_score": min_score,
        "seg_tau_siglip": seg_tau_siglip,
        "seg_tau_dino": seg_tau_dino,
        "seg_gap_close": seg_gap_close,
        "seg_min_duration": seg_min_duration,
    }
    return "\n".join(lines), stats, active_vcs, active_iids


def format_inventory_for_prompt(
    inventory: list[dict],
    transparency_by_iid: dict | None = None,
) -> str:
    transparency_by_iid = transparency_by_iid or {}
    lines = []
    for inv in inventory:
        unit_label = "grams" if inv["unit"] == "g" else "count"
        iid = inv["instance_id"]
        if iid in transparency_by_iid:
            tag = "[transparent]" if transparency_by_iid[iid] else "[opaque]"
        else:
            visible = inv.get("visible_during_interaction", True)
            tag = "[transparent]" if visible else "[opaque]"
        lines.append(
            f"- {iid}: \"{inv['visual_class']}\" "
            f"({unit_label}, package={inv['package_amount']}, {tag})"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Planner system prompt — persistent chat
# ---------------------------------------------------------------------------

PLANNER_SYSTEM_PROMPT = """\
You are an expert kitchen activity analyst running an interactive \
plan-observe loop for remaining-amount estimation on an egocentric \
cooking session.

## Your job (read this first)
Your goal is NOT to estimate a numeric remaining amount. Your goal is to \
VERIFY whether each item was ACTUALLY USED in this session, and — when \
it was used — to steer the observer to a window that lets IT read a \
remaining amount. The numeric `remaining` always comes from an observer \
report. You never invent one.

For every visual_class, you are deciding one of:
- **`used`** — observer evidence that the stock container was opened \
and/or a portion physically left it (a dispense, or a confirmed \
post-use derivative matching the product).
- **`not_used`** — observer confirms retrieval and return with no \
intervening dispense.
- **`unknown`** — still undecided.

The observer produces a `remaining` number via one of two read \
methods and reports which in `per_instance[].read_method`:
- **`fill_line`** — read remaining off the stock container itself. \
For a transparent package this is a direct read of the visible \
fill level. For an opaque package the fill is not directly visible, \
but the same window's dispensing action exposes secondary cues that \
let you infer the same number: container heft, tilt angle required \
to dispense, squeeze effort and side-wall deformation, flow regime \
(steady vs sputtery/gurgling), residual at the lip when inverted. \
Either path returns `remaining` directly. Optionally populate \
`starting_amount` and `dispensed_estimate` as a sanity-check \
decomposition (`remaining ≈ starting_amount - dispensed_estimate`); \
they are diagnostic, not required.
- **`derivative_volume`** — the stock container is not visible in \
this window (or the dispensing action it isn't readable for \
starting-amount cues), and only the dispensed/derivative product is \
visible. Observer commits only `dispensed_estimate` and emits \
`needs_starting_amount` so the planner stitches a separate \
stock-container window. Used when the stock container is hidden \
end-to-end in the chosen window.

The observer chooses whichever method the frames support; \
`given_up` only when neither method is feasible across every \
detection burst.

## Loop contract
- Round 0: read the inventory + per-item HOI evidence, initialize \
`food_journeys` (one per active visual_class, `usage_status="unknown"` \
for all of them), list evidence-free iids under `skipped_items`, and \
emit a BATCH of observation windows — typically one per active class. \
Observers run in parallel and report back.
- `skipped_items` is for inventory iids with ZERO detection rows \
in the evidence section (or only flicker-only / very-low-confidence \
hits the segment formatter explicitly flagged). If an iid's vc has \
ANY detection burst in the evidence — even fragmented, low-DINO, or \
overlapping with another vc — emit a food_journey for it and pick \
at least one observation window. The observer, not the planner, \
decides whether a burst is real or cross-talk. Use the cross-talk \
note from the evidence section to choose the right vc when bursts \
overlap, but do not skip the iid.
- Rounds 1..N: observer reports arrive. For each item you UPDATE \
`usage_status` based on observer text (the observer's prose is ground \
truth over raw DINO scores), then either mark `status=resolved` (with \
the observer reading copied into `resolved_value`), `status=given_up` \
(with a reason), or leave `unresolved` and emit a targeted follow-up \
window. Stop only when every active item is resolved or given_up.

## Evidence you're looking at (each round 0 user message)

The round-0 user message contains:
- **Session Inventory** — every iid with its visual_class, unit, package \
capacity, and `[opaque|transparent]` tag. Use this to look up \
`package_type` and `candidate_instance_ids` per visual_class.
- **Per-Item Detections** — a list of HOI-gated detection rows. The \
exact row layout (columns, sort order) is described in the parenthetical \
header of that section in this round's user message — read it. \
Common columns: `[start-end]`, `dur`, peak DINO similarity, OWLv2 \
`scene` distribution (`storage` / `sink` / `stove` / `unknown`), and \
the visual_class. Only detections with similarity ≥ {min_score} are \
shown; visual_classes with no qualifying detection are omitted entirely.

Cross-talk note: when several visual_classes claim overlapping \
time ranges, the highest DINO peak is usually the actual focus and \
the others are visual-similarity bleed. Use the inventory list \
(package_type, visual appearance) to sanity-check.

## Session timeline — describe activity once, reference per item

Walk the chronological evidence ONCE and emit a `session_timeline`: \
one entry per coherent activity block (a contiguous span where the \
user does ONE thing). Each entry carries:
  - `idx` — stable integer, referenced by `food_journeys.timeline_refs` \
and `observation_windows.timeline_refs`.
  - `time` — `[start, end]` covering the rows folded into this entry.
  - `scene` — `storage` / `sink` / `stove` / `counter` / `unknown` \
(use `→` for a short transition, e.g. `sink→stove`).
  - `user_action` — ≤12-word verb-led phrase describing what the \
wearer is DOING (not what's in their hand).
  - `items` — list of visual_classes meaningfully present in this \
moment (handled, staged, derivative-bowl-visible, retrieval, return). \
**Drop pure DINO-bleed cross-talk siblings** — pick the focus and \
omit the bleed. Cross-talk handling is one decision per timeline \
entry, not per item.
  - `dispensed` — subset of `items` the planner hypothesizes LEFT \
its original container at this moment. Usually empty (most entries \
are retrieval / staging / return / handling-without-dispense). \
Use it ONLY when evidence supports a positive dispensing hypothesis: \
sustained DINO peak in `sink` / `stove`, derivative appearing in \
cookware, prolonged precision-grasp on a container. The observer is \
the final arbiter — `dispensed` is a hypothesis the observer can \
confirm or contradict in `window_observation`.

Do NOT pre-label a detection as "derivative-only" before the observer \
has looked at it. A derivative bowl visible at a moment is still \
worth listing in `items` (and possibly `dispensed`) because that's \
exactly what proves usage; only the observer can distinguish stock \
container from derivative.

## Window-choice heuristic (per window)
Pick the window with the highest marginal info for verifying USAGE. \
Express the choice as a list of `timeline_refs` plus the matching \
`segments`.

1. **Opaque package.** The fill line itself isn't directly \
visible, but the dispensing window exposes the heft/tilt/squeeze/\
flow cues that read out remaining off the container — same \
`fill_line` read_method, different visual cues. Target a timeline \
entry where this vc appears in `dispensed`, and pick a window that \
covers the FULL dispense action (start to end of pour/scoop/\
squeeze) so the cues are readable. If round-0 observer cannot find \
dispensing in your chosen window, expect a \
`reason_tag=needs_dispensal_window` followup with per-segment \
observations; round-1 picks a different timeline entry based on \
that narrative. Do NOT pick a storage-return entry for opaque \
items — retrieval/return gestures don't expose fill state.

2. **Transparent package.** Target a stock-container-visible moment \
where the fill is readable — typically a LATE `storage` return \
entry, but a `counter`-staging or pre-cook entry also works. The \
observer's preferred path is `fill_line`. If only derivative is \
visible in this window, expect a \
`reason_tag=needs_starting_amount` followup carrying a \
`dispensed_estimate`; round-1 then picks an EARLY pre-dispense \
fill-line view (so `final = fill_observed - dispensed_estimate`) OR \
an alternate post-dispense fill-line view (so \
`final = fill_observed` directly). Pick the SINGLE highest-signal \
fill-line entry first; only emit alternates after observer feedback.

3. **Sibling ambiguity (multiple iids of the same visual_class).** \
Emit one window whose segments expose ALL siblings together (a frame \
where every candidate is visible) plus the handling/dispense frame. \
The observer disambiguates in prose.

4. **Observer-driven replanning (rounds 1+) — signal → action playbook.**
React to the observer signal, not your prior assumption. Match the
observer's `window_observation` / `needs_followup` text against the
left column and take the right-column action:

   | Observer signal                                                                  | Action this round                                                                                  |
   | -------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
   | `needs_followup.reason_tag = "needs_starting_amount"` (transparent + dispensed_estimate populated) | emit a follow-up window for that iid targeting a stock-container fill-line moment. Pick the next sub-range using this priority: (a) if the observer's `needs_followup.reason` names a specific moment, target that moment; (b) a late `storage`/`counter` return sub-range (observer returns `remaining` directly); (c) an early pre-dispense fill view (planner will compute `resolved_value = fill_observed - dispensed_estimate`); (d) a different timeline entry the vc appears in. Each follow-up segment must be non-overlapping with previously sampled segments for this vc. Resolve with `status="resolved"`, `resolved_value=null`, `resolved_dispensed = <observer's dispensed_estimate>`, `resolved_source_round = <that round>`, `given_up_reason: null` only when no unsampled sub-range remains. |
   | `needs_followup.reason_tag = "needs_dispensal_window"` (opaque, no dispensing caught) | re-read `per_segment_observations` and emit a follow-up window over the segment(s) the observer flagged as most likely dispensing. Drop segments the observer described as retrieval/return only. If no other detection burst remains, given_up — but ONLY if no prior observer round on this vc returned a `dispensed_estimate`; if any did, preserve it (resolve with `resolved_value=null, resolved_dispensed=<that estimate>`) instead of given_up. |
   | `needs_followup.reason_tag = "container_not_visible"`                            | emit a follow-up window targeting an unsampled sub-range — either a different timeline entry for that vc, or a different sub-range within an already-touched entry. Flip to `given_up` only when no unsampled sub-range remains. |
   | multiple sibling iids visible, observer cannot disambiguate                      | emit one window whose segments include a frame where ALL candidates are visible together + the handling/dispense frame. |
   | `handling_status=not_visible` for every candidate + no plausible alt window      | `given_up` with reason citing exhausted detection ranges.                                          |
   | `handling_status=false_detection` for an iid                                     | flip to `given_up` immediately. `given_up_reason = "false_detection"` + `false_detection_actual` if given. Do NOT emit further windows for this iid. |
   | `remaining` returned non-null AND no `needs_followup` for that iid               | resolve: copy that exact number to `resolved_value`, record `resolved_source_round`.               |

`given_up` requires every detection burst for this visual_class \
to have been either observed or discarded with an observer-backed \
reason. The `given_up_reason` must name the bursts checked and say \
why neither `fill_line` nor `derivative_volume` is feasible. Items \
with a non-null `dispensed_estimate` from any round are not eligible \
for given_up — resolve them with `resolved_value=null, \
resolved_dispensed=<that estimate>`.

5. **Batch sizing.**
   - Round 0: one window per active visual_class. A class may get a \
second window only when one window genuinely cannot answer both \
usage AND fill (multi-segment windows handle most cases). Do NOT \
emit multiple windows as a hedge.
   - Rounds 1+: one window per still-unresolved item that has a \
viable alternative angle. Do NOT emit windows for `resolved` or \
`given_up` items.

## Output schema — EVERY round (identical shape, round 0 + later rounds)

```json
{{
  "session_timeline": [
    {{
      "idx": <int>,
      "time": [<start>, <end>],
      "scene": "<storage|sink|stove|counter|unknown>",
      "user_action": "<≤12-word verb-led phrase>",
      "items": ["<vc>", ...],
      "dispensed": ["<vc>", ...]
    }}
  ],
  "food_journeys": [
    {{
      "visual_class": "<vc>",
      "candidate_instance_ids": ["<iid>", ...],
      "package_type": "opaque" | "transparent",
      "timeline_refs": [<idx>, ...],
      "usage_status": "unknown" | "used" | "not_used",
      "status": "unresolved" | "resolved" | "given_up",
      "resolved_value": <number or null>,
      "resolved_starting": <number or null>,
      "resolved_dispensed": <number or null>,
      "resolved_source_round": <int or null>,
      "given_up_reason": "<string or null>"
    }}
  ],
  "skipped_items": [
    {{"instance_id": "<iid>", "reasoning": "<why no coherent usage>"}}
  ],
  "action": {{
    "type": "observe" | "stop",
    "observation_windows": [
      {{
        "visual_class": "<vc>",
        "candidate_instance_ids": ["<iid>", ...],
        "segments": [[<start>, <end>], ...],
        "timeline_refs": [<idx>, ...],
        "why": "<one sentence — cite package_type + which timeline entries this window probes + observer feedback if any>",
        "confidence": "high" | "medium" | "low"
      }}
    ]
  }}
}}
```

Rules:
- `session_timeline` is emitted on round 0 and CARRIED FORWARD \
verbatim every round. Only amend it (in place, same `idx` values) \
when an observer report contradicts a `dispensed` claim — e.g. \
observer says container was opened but nothing left it, so remove \
the vc from that entry's `dispensed`. Never renumber `idx`.
- `food_journeys` is CARRIED FORWARD every round. Round 0 \
initializes with `timeline_refs` set to every timeline `idx` where \
this vc appears in `items`, `usage_status="unknown"` (UNLESS the vc \
already appears in some entry's `dispensed`, in which case round-0 \
`usage_status="used"`), `status="unresolved"`, `resolved_value=null`. \
Later rounds UPDATE `usage_status` from observer prose and FLIP \
`status` only when one of the conditions below holds:
   - `usage_status="used"` AND an observer report on this \
visual_class returned a non-null `remaining` (from \
`read_method="fill_line"`): set `status="resolved"`, \
`resolved_value = <that exact observer number>`, and \
`resolved_source_round` = the round whose observer returned it. \
You may not invent or adjust this number. If the observer \
optionally populated `starting_amount` and `dispensed_estimate` \
(opaque-package diagnostic decomposition), copy them to \
`resolved_starting` / `resolved_dispensed` for traceability, \
otherwise leave both null. Do NOT compute `resolved_value = \
package_capacity − dispensed_estimate`; package_capacity is the \
receipt amount, not the session-start fill.
   - Transparent iid: an earlier observer round returned a \
`dispensed_estimate` with `reason_tag="needs_starting_amount"`, AND \
a later observer round returned a `fill_line` read in the SAME \
session: set `status="resolved"`, `resolved_value` = (if the \
fill-line window is post-dispense) `<fill_observed>` directly, OR \
(if the fill-line window is pre-dispense) `<fill_observed - \
dispensed_estimate>`. Record both contributing rounds in \
`resolved_source_round` (use the later round's index as the primary \
value; cite the earlier round in `given_up_reason: null` and in \
prose only).
   - `usage_status="not_used"`: set `status="resolved"`, \
`resolved_value` = the package's full capacity from the inventory \
(only valid when the package was unopened at session start; if not, \
leave `resolved_value=null` and `resolved_dispensed=0`).
   - No viable observation window remains to decide `usage_status` OR \
to read a remaining amount after use: set `status="given_up"` with a \
specific `given_up_reason`.
- The `usage_status="used"` shortcut is allowed only when the vc \
appears in at least one timeline entry's `dispensed`. Otherwise \
default to `unknown` and let the observer flip it.
- Resolving ON AN OBSERVER REPORT THAT RAISED `needs_followup` for \
that iid is prohibited. Either emit a follow-up window, or given_up.
- Sub-range coverage tracking: treat each timeline entry as a \
continuous time range (`time: [start, end]`). Before emitting a \
follow-up window, identify which sub-ranges of that entry have not \
yet been sampled for this vc. Follow-up windows target unsampled \
sub-ranges, and may reuse a timeline entry as long as the new \
segment is non-overlapping with prior segments in that entry.
- `action.type = "stop"` ONLY when every item is `resolved` or \
`given_up`. Omit `observation_windows` when stopping (or leave it as \
an empty array).
- `skipped_items` is emitted on round 0 and on the final `stop`; \
intermediate `observe` rounds should repeat the round-0 list unchanged \
(or omit it to keep the output small — both are accepted).
- Use EXACT `visual_class` and `instance_id` strings from inventory.
- `candidate_instance_ids` MUST include EVERY inventory iid of that \
visual_class (even if length 1).
- `segments` is a LIST of [start_s, end_s] ranges concatenated into \
ONE observer call. Use this to give the observer multiple moments in \
context within a single call (retrieval + dispense + return for the \
same item is ONE window with three segments, not three windows).
- **Segment sizing.** For long bursts (>20s), spread 2–4 short \
(2–4s) segments across head/middle/tail rather than concentrating \
samples at one end. For short (<10s) moments, one 3–6s segment is \
fine.
- JSON only — no prose outside the fenced block.

Budget: hard cap {max_rounds} rounds. Each observer call is an \
independent API call with its own frame limit (up to {max_frames} \
frames total across all segments in that call). There is no cap on \
how many observation_windows you may emit — spend one per item that \
needs observation. Round 0 typically emits ~one window per active \
visual_class; later rounds emit fewer, only for items still \
unresolved.
"""


PLANNER_ROUND0_USER = """\
## Session Inventory
{inventory}

## Per-Item Detections ({evidence_format_note})
{evidence}

Walk the evidence chronologically and emit `session_timeline` first \
(one entry per coherent activity block, with `dispensed` set only \
when evidence supports a positive dispensing hypothesis). Then \
initialize `food_journeys` for every visual_class that appears in \
any timeline entry's `items` (set `timeline_refs` accordingly), and \
mark evidence-free iids under `skipped_items`. Finally, emit an \
`observation_windows` BATCH covering ALL active visual_classes — \
one window per class, per the system-prompt heuristic, each with \
its own `timeline_refs`. Use multi-segment windows when the observer \
needs more than one moment to answer. Each observer call handles up \
to {max_frames} frames across its segments.

Output JSON only, matching the schema in the system prompt.
"""


PLANNER_FOLLOWUP_USER = """\
## Observer reports — round {round_idx} ({n_windows} windows, {n_frames_round} frames)

```json
{observer_batch_json}
```

## Ledger delta
Resolved: {resolved_lines}
Unresolved: {unresolved_lines}
Given up: {given_up_lines}
Skipped (do NOT silently re-add to `food_journeys`): {skipped_lines}

Budget: {rounds_used}/{max_rounds} rounds, {frames_used} frames.

Carry `session_timeline` forward verbatim (amend a `dispensed` list \
in place only if an observer report contradicts it; never renumber \
`idx`). Apply the system-prompt playbook to each iid and emit the \
next plan. JSON only.
"""


# ---------------------------------------------------------------------------
# Planner call + parser
# ---------------------------------------------------------------------------

def _is_refusal(text: str) -> bool:
    low = text.strip().lower()
    return low.startswith("i'm sorry") or low.startswith("i cannot") or "cannot assist" in low


def _parse_planner_json(text: str) -> dict | None:
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        try:
            return json.loads(fence.group(1))
        except json.JSONDecodeError:
            pass
    obj_match = re.search(r"\{.*\"action\".*\}", text, re.DOTALL)
    if obj_match:
        try:
            return json.loads(obj_match.group())
        except json.JSONDecodeError:
            return None
    return None


def _validate_window(
    window: dict,
    inv_by_iid: dict,
    inv_by_vc: dict,
) -> tuple[dict | None, str]:
    """Normalize a planner-emitted observation_window.

    Accepts either shape:
      - Multi-segment: {"segments": [[s,e], [s,e], ...], ...}
      - Single continuous: {"start": s, "end": e, ...}  (legacy / shorthand)

    Returns a normalized dict with `segments: list[list[float]]` (always
    populated, even for single-segment windows).
    """
    if not isinstance(window, dict):
        return None, "window not a dict"

    segments_raw = window.get("segments")
    if isinstance(segments_raw, list) and segments_raw:
        segments: list[list[float]] = []
        for seg in segments_raw:
            if not isinstance(seg, (list, tuple)) or len(seg) < 2:
                continue
            try:
                s, e = float(seg[0]), float(seg[1])
            except (TypeError, ValueError):
                continue
            if e <= s:
                continue
            segments.append([s, e])
        if not segments:
            return None, f"all segments malformed ({segments_raw})"
    else:
        # Legacy / single-range shorthand.
        try:
            s = float(window["start"])
            e = float(window["end"])
        except (KeyError, TypeError, ValueError):
            return None, "no valid segments nor start/end"
        if e <= s:
            return None, f"end<=start ({s}, {e})"
        segments = [[s, e]]

    vc = window.get("visual_class")
    if not vc or not isinstance(vc, str):
        return None, "missing visual_class"
    cand_iids_raw = window.get("candidate_instance_ids") or []
    if not isinstance(cand_iids_raw, list) or not cand_iids_raw:
        return None, "candidate_instance_ids missing/empty"
    valid_iids = [str(c) for c in cand_iids_raw if str(c) in inv_by_iid]
    if not valid_iids:
        return None, f"no valid iids ({cand_iids_raw})"
    # Backfill vc-matched iids the planner may have dropped.
    for iid in inv_by_vc.get(vc, []):
        if iid not in valid_iids:
            valid_iids.append(iid)
    overall_start = min(s for s, _ in segments)
    overall_end = max(e for _, e in segments)
    return (
        {
            "segments": segments,
            "start": overall_start,
            "end": overall_end,
            "visual_class": vc,
            "candidate_instance_ids": valid_iids,
            "why": window.get("why", ""),
            "confidence": window.get("confidence", ""),
        },
        "",
    )


def call_planner(
    client,
    messages: list[dict],
    model: str,
    max_retries: int = 5,
) -> tuple[str, dict]:
    for attempt in range(max_retries):
        t0 = time.time()
        try:
            response = client.responses.create(
                model=model,
                input=messages,
                reasoning={"effort": "medium"},
            )
            response_text = response.output_text or ""
            usage = response.usage
            stats = {
                "inference_time_s": round(time.time() - t0, 2),
                "model": model,
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "attempt": attempt + 1,
            }
        except Exception as e:
            print(f"  Planner ERROR (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(5 * (attempt + 1))
                continue
            return "", {"error": str(e), "inference_time_s": round(time.time() - t0, 2)}

        if _is_refusal(response_text) or (
            response_text and "action" not in response_text
        ):
            print(f"  Planner refusal/truncated (attempt {attempt + 1}), retrying...")
            if attempt < max_retries - 1:
                time.sleep(5 * (attempt + 1))
                continue

        return response_text, stats

    return "", {"error": "max retries exceeded"}


# ---------------------------------------------------------------------------
# Observer — candidate-aware + window_observation + needs_followup
# ---------------------------------------------------------------------------

OBSERVER_PROMPT = """\
You are analyzing frames from an egocentric kitchen video recorded with smart glasses.

These {n_frames} frames are sampled from one or more short time \
segments inside a cooking session. When more than one segment is \
passed, they are NOT continuous footage — there may be large time \
gaps between consecutive frames (e.g. retrieval at 12s, dispensing \
at 45–50s, and return at 180s may all appear in one call). Each \
frame is labeled with its session timestamp.

Frame timestamps: {frame_timestamps}
Segments supplied: {segments_str}

## Target Item
You are looking for: "{visual_class}"
- Unit: {unit_label}
- Package capacity: {package_capacity}

## Tracked purchase instances (candidates)
The inventory tracks {n_candidates} purchase instance(s) of this product. \
Identical-looking packages of the same product are tracked as separate \
instances; your job is to decide which physical package is being handled \
and to report a remaining amount for the `handled` candidate only.

{candidate_table}

## Context from the reasoner (planner)
{segment_descriptions}

## Context from prior observer rounds (may be empty)
{prior_context}

## Task
1. Confirm whether "{visual_class}" is visible.
2. For every candidate iid, decide its `handling_status`:
   - `handled` — this specific instance is the physical package being \
retrieved, opened, dispensed from, or otherwise used. At most one \
`handled` per session unless two distinct packages are clearly \
handled separately.
   - `visible_untouched` — this candidate is visible but NOT handled \
(ONLY used for sibling-instance disambiguation). ALWAYS set \
`remaining: null` (ledger carry-forward is the answer).
   - `not_visible` — the target product is plausibly nearby (e.g. \
left frame at the wrong moment, or sampled times missed it) but the \
candidate package itself is not in any of these frames. Set \
`remaining: null`. A different window MIGHT resolve it.
   - `false_detection` — the frames are dominated by a DIFFERENT \
product from "{visual_class}", and the detection that put this iid \
on the candidate list was visual-similarity bleed. Set `remaining: \
null`, `read_method: null`. Optionally name the actually-visible \
product in `false_detection_actual`. The reasoner will give up on \
this iid immediately — do NOT also raise `needs_followup` for it.
3. Read what you can. The starting remaining at session start is \
not known a priori — `package_capacity` is the receipt amount and \
items may have been used in earlier sessions. Do not compute \
`remaining = package_capacity − dispensed`. Set `read_method`:
   - **`fill_line`** — read remaining directly off the stock \
container. Returns `remaining` directly. There are two routes that \
both fall under `fill_line`:
       - TRANSPARENT package: read the visible fill level / exposed \
opening / weighable container directly.
       - OPAQUE package: the fill line itself isn't visible, but \
the dispensing window exposes secondary cues that read out the \
same number — container heft and ease of lift, tilt angle required \
to dispense, squeeze effort and side-wall deformation (for soft \
bottles), pour stream regime (steady vs sputtery/gurgling — \
sputtery suggests near-empty), pour duration relative to amount \
delivered, residual at the lip or rim when the container is \
inverted, audible cues if the bottle is shaken near-empty. Use \
these to commit a single `remaining` number. Do NOT use retrieval \
or return segments for this — those gestures don't expose fill \
state; only the dispensing action does. \
       Optionally populate `starting_amount` and `dispensed_estimate` \
as a sanity-check decomposition; they are diagnostic only — the \
committed signal is `remaining`. PREFERRED when available.
   - **`derivative_volume`** — the stock container is not visible \
in this window (and for an opaque package the dispensing action \
isn't readable for fill cues either); only the derivative is. \
Estimate `dispensed_estimate` (in the inventory unit; state the \
conversion factor in `reasoning`; for `count` unit, count \
discretely), set `remaining: null`, and emit `needs_followup` with \
`reason_tag: "needs_starting_amount"`. The planner will request a \
follow-up window targeting a stock-container view (transparent: \
pre/post-dispense fill view; opaque: a dispensing window with \
readable heft/tilt/flow cues) so it can combine with your \
`dispensed_estimate`.
   - **`null`** — set `remaining: null` AND `read_method: null` when \
neither signal is usable. If the dispensal moment is missing for an \
OPAQUE item (you can see the bottle being handled but no dispensing \
action and no derivative in cookware), populate \
`per_segment_observations` with one entry per supplied segment and \
emit a `needs_followup` with `reason_tag: "needs_dispensal_window"`. \
The planner will use your per-segment narrative to re-hypothesize \
when dispensal occurred.

If a window the planner emitted for one read method actually \
supports the other, USE whichever the frames support — do not return \
`null` and ask for a different window.
4. Cite evidence frames.

## Honesty & followup

For each candidate iid you produce EXACTLY ONE of these outcomes — \
they are MUTUALLY EXCLUSIVE:

A1. **Committed remaining (fill_line).** Set `remaining` to a \
non-null value with `read_method: "fill_line"`. Works for either \
package type: transparent → direct visible-fill read; opaque → \
inferred from the dispensing window's heft/tilt/squeeze/flow cues. \
This is your committed remaining for that iid; do not also list \
this iid in `needs_followup`. For an opaque case you may optionally \
populate `starting_amount` and `dispensed_estimate` as a diagnostic \
decomposition; the committed signal is still `remaining`.

B1. **Followup: needs_starting_amount** (stock container not \
readable in this window, derivative-only). Set \
`handling_status: "handled"`, `read_method: "derivative_volume"`, \
`remaining: null`. Populate `dispensed_estimate` with the amount \
that left the container in this window. Add a `needs_followup` \
entry with `reason_tag: "needs_starting_amount"` and a free-text \
`reason` naming a candidate alternative window — for transparent: \
an early pre-dispense or later post-dispense fill view; for \
opaque: a dispensing-action window where heft/tilt/flow cues are \
readable.

B2. **Followup: needs_dispensal_window** (opaque items, no \
dispensing action observed — needed for the heft/tilt/squeeze \
cues that read remaining off an opaque container). \
Set `handling_status: "handled"` (if the bottle/package itself is \
in frame) or `not_visible` (if it is not). Set `remaining: null`, \
`read_method: null`, `starting_amount: null`, \
`dispensed_estimate: null`. Populate `per_segment_observations` \
with one entry per supplied segment describing scene, hand \
activity, and your hypothesis about whether dispensal could have \
occurred there. Add a `needs_followup` entry with \
`reason_tag: "needs_dispensal_window"` and a free-text `reason` \
naming the segment(s) most likely to contain dispensing.

B3. **Followup: container_not_visible** (default for \
`handling_status: not_visible` with no other useful signal). \
`remaining: null`, `read_method: null`. Populate `needs_followup` \
with `reason_tag: "container_not_visible"` and a free-text reason. \
Use this when no narrowing hypothesis is available — the planner \
will pick the next best window heuristically.

C. **A false-detection verdict.** Set \
`handling_status: "false_detection"`, `remaining: null`, \
`read_method: null`, do NOT add to `needs_followup`, and (optionally) \
fill `false_detection_actual`. Use this when the candidate product \
is not what's actually in the frames; no later window of this iid \
will help.

- Fill `window_observation` (1–2 sentences) covering: scene; \
stock-container state (visible & fill-readable / visible but fill \
not readable / only derivative visible / not visible at all); any \
dispensing action and qualitative amount dispensed; whether a later \
window is likely to resolve outstanding uncertainty.

Think step by step:
- How many distinct physical packages of this product are visible? If \
one, mark one candidate `handled` and the rest `not_visible` — do NOT \
double-count.
- For the handled package: when is its stock container last visible? \
What portion remains?
- Are portions already taken out? Note as used, not remaining.
- Is any observed content actually a derivative, not the package?

Output ONLY JSON:
```json
{{
  "window_observation": "<1-2 sentences — scene, stock vs derivative, dispensing action + amount, followup hint>",
  "per_instance": [
    {{
      "instance_id": "<iid exactly as listed>",
      "handling_status": "handled" | "visible_untouched" | "not_visible" | "false_detection",
      "remaining": <number or null>,
      "read_method": "fill_line" | "derivative_volume" | null,
      "starting_amount": <number or null>,
      "dispensed_estimate": <number or null>,
      "per_segment_observations": [
        {{"segment_idx": <int>, "time": [<start>, <end>], "observation": "<short — scene, hand action, target visibility, dispensing hypothesis>"}}
      ] | null,
      "false_detection_actual": "<actually-visible product, if handling_status=false_detection>" | null,
      "reasoning": "<≤40 words — cues used (visible fill / heft / tilt / squeeze / flow) when read_method=fill_line on an opaque package>",
      "evidence_frames": [<timestamp floats of key frames>]
    }}
  ],
  "needs_followup": [
    {{"instance_id": "<iid>", "reason_tag": "needs_starting_amount" | "needs_dispensal_window" | "container_not_visible", "reason": "<what a later window could resolve>"}}
  ]
}}
```"""


def extract_window_frames(
    segments: list[list[float]],
    video_durations: list[tuple[Path, float]],
    padding: float = 2.0,
    fps: float = 1.0,
    max_frames: int = MAX_IMAGES,
) -> tuple[list[str], list[float]]:
    """Extract frames for one observer call from a list of [start, end] segments.

    All segments from ONE observation_window entry are concatenated into a
    single frame set (sorted by timestamp), matching PerItem's
    journey_samples + dense_windows combination but expressed as one unified
    segments list. `max_frames` is the TOTAL budget across all segments.
    """
    if not segments:
        return [], []
    return _extract_segments_frames(
        [(float(s[0]), float(s[1])) for s in segments],
        video_durations,
        padding=padding,
        max_frames=max_frames,
        target_fps=fps,
    )


def _build_observer_prompt(
    n_frames: int,
    timestamps: list[float],
    candidates: list[dict],
    window: dict,
    prior_observations: list[str],
) -> str:
    visual_class = window.get("visual_class") or candidates[0]["visual_class"]
    unit_label = "grams" if candidates[0]["unit"] == "g" else "count"
    caps = {str(c["package_amount"]) for c in candidates}
    package_capacity = caps.pop() if len(caps) == 1 else (
        "varies by instance (see candidate table)"
    )

    candidate_lines = []
    for c in candidates:
        iid = c["instance_id"]
        m = re.search(r"_(\d{8})$", iid)
        purchase = m.group(1) if m else "unknown"
        cap = c.get("package_amount", "?")
        candidate_lines.append(
            f"- `{iid}` — purchased {purchase} — package {cap}"
        )
    candidate_table = "\n".join(candidate_lines)

    segs = window.get("segments") or [[window.get("start", 0.0), window.get("end", 0.0)]]
    segs_str = ", ".join(f"{s[0]:.1f}–{s[1]:.1f}s" for s in segs)

    descs: list[str] = []
    descs.append(
        f"Segments: {segs_str} "
        f"(confidence={window.get('confidence', '?')})"
    )
    if window.get("why"):
        descs.append(f"Reasoner note: {window['why']}")
    segment_descriptions = "\n".join(f"- {d}" for d in descs)

    if prior_observations:
        prior_context = "\n".join(f"- {o}" for o in prior_observations)
    else:
        prior_context = "(none; this is the first observer round)"

    frame_ts_str = ", ".join(f"{t:.1f}s" for t in timestamps)

    return OBSERVER_PROMPT.format(
        n_frames=n_frames,
        frame_timestamps=frame_ts_str,
        segments_str=segs_str,
        visual_class=visual_class,
        unit_label=unit_label,
        package_capacity=package_capacity,
        n_candidates=len(candidates),
        candidate_table=candidate_table,
        segment_descriptions=segment_descriptions,
        prior_context=prior_context,
    )


def run_observer(
    client,
    frames_b64: list[str],
    timestamps: list[float],
    candidates: list[dict],
    window: dict,
    prior_observations: list[str],
    model: str,
    prompt_save_path: Path | None = None,
) -> tuple[str, str, dict]:
    prompt = _build_observer_prompt(
        len(frames_b64), timestamps, candidates, window, prior_observations
    )
    if prompt_save_path is not None:
        prompt_save_path.parent.mkdir(parents=True, exist_ok=True)
        prompt_save_path.write_text(prompt)

    content: list[dict] = [{"type": "input_text", "text": prompt}]
    for i, (fb64, ts) in enumerate(zip(frames_b64, timestamps)):
        content.append({"type": "input_text", "text": f"[Frame {i+1}, t={ts:.1f}s]"})
        content.append({
            "type": "input_image",
            "image_url": f"data:image/jpeg;base64,{fb64}",
            "detail": "high",
        })

    max_retries = 5
    for attempt in range(max_retries):
        t0 = time.time()
        try:
            response = client.responses.create(
                model=model,
                input=[{"role": "user", "content": content}],
                reasoning={"effort": "medium"},
            )
            response_text = response.output_text or ""
            usage = response.usage
            stats = {
                "inference_time_s": round(time.time() - t0, 2),
                "model": model,
                "num_frames": len(frames_b64),
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "attempt": attempt + 1,
            }
        except Exception as e:
            err_str = str(e)
            transient = any(m in err_str for m in (
                "503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED",
                "500", "INTERNAL", "deadline", "timeout", "connection", "Connection",
            ))
            if "content_policy_violation" in err_str and attempt < max_retries - 1:
                print(f" content filter (attempt {attempt + 1}), retrying...", end="", flush=True)
                time.sleep(5 * (attempt + 1))
                continue
            if transient and attempt < max_retries - 1:
                wait = 4 * (2 ** attempt)
                print(f" transient API error (attempt {attempt + 1}), retrying in {wait}s...", end="", flush=True)
                time.sleep(wait)
                continue
            vc = window.get("visual_class") or (candidates[0]["visual_class"] if candidates else "?")
            print(f"  Observer ERROR ({vc}): {e}")
            return "", prompt, {"error": err_str, "inference_time_s": round(time.time() - t0, 2)}

        if _is_refusal(response_text) and attempt < max_retries - 1:
            print(f" refusal (attempt {attempt + 1}), retrying...", end="", flush=True)
            time.sleep(5)
            continue

        return response_text, prompt, stats

    return "", prompt, {"error": "max retries exceeded"}


def run_observer_qwen(
    frames_b64: list[str],
    timestamps: list[float],
    candidates: list[dict],
    window: dict,
    prior_observations: list[str],
    qwen_url: str = QWEN_URL,
    qwen_model: str = QWEN_MODEL_DEFAULT,
    prompt_save_path: Path | None = None,
) -> tuple[str, str, dict]:
    import requests

    prompt = _build_observer_prompt(
        len(frames_b64), timestamps, candidates, window, prior_observations
    )
    if prompt_save_path is not None:
        prompt_save_path.parent.mkdir(parents=True, exist_ok=True)
        prompt_save_path.write_text(prompt)

    content: list[dict] = [{"type": "text", "text": prompt}]
    for i, (fb64, ts) in enumerate(zip(frames_b64, timestamps), 1):
        content.append({"type": "text", "text": f"[Frame {i}, t={ts:.1f}s]"})
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{fb64}"},
        })

    messages = [{"role": "user", "content": content}]

    t0 = time.time()
    max_retries = 4
    for attempt in range(max_retries):
        try:
            resp = requests.post(qwen_url, json={
                "model": qwen_model,
                "messages": messages,
                "max_tokens": 4096,
                "temperature": 0.3,
            }, timeout=600)
            resp.raise_for_status()
            result = resp.json()
            response_text = result["choices"][0]["message"]["content"]
            usage = result.get("usage", {})
            stats = {
                "inference_time_s": round(time.time() - t0, 2),
                "model": qwen_model,
                "num_frames": len(frames_b64),
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
                "attempt": attempt + 1,
            }
            break
        except Exception as e:
            last_err = str(e)
            transient = any(m in last_err for m in (
                "503", "UNAVAILABLE", "429", "500", "502", "504",
                "deadline", "timeout", "connection", "Connection",
            ))
            if transient and attempt < max_retries - 1:
                wait = 4 * (2 ** attempt)
                print(f" Qwen transient error (attempt {attempt + 1}), retrying in {wait}s...", end="", flush=True)
                time.sleep(wait)
                continue
            print(f"  Qwen ERROR: {e}")
            return "", prompt, {"error": last_err, "inference_time_s": round(time.time() - t0, 2)}

    return response_text, prompt, stats


def parse_observer_response(response_text: str) -> tuple[str, list[dict], list[dict]]:
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response_text, re.DOTALL)
    text_to_parse = fence.group(1) if fence else response_text
    obj_match = re.search(r"\{.*\"per_instance\".*\}", text_to_parse, re.DOTALL)
    if not obj_match:
        return "", [], []
    try:
        parsed = json.loads(obj_match.group())
    except json.JSONDecodeError:
        return "", [], []
    if not isinstance(parsed, dict):
        return "", [], []

    window_observation = str(parsed.get("window_observation", ""))

    entries = parsed.get("per_instance") or []
    per_instance: list[dict] = []
    if isinstance(entries, list):
        for e in entries:
            if not isinstance(e, dict):
                continue
            iid = e.get("instance_id")
            if not iid:
                continue
            rem = e.get("remaining")
            try:
                rem_val = float(rem) if rem is not None else None
            except (TypeError, ValueError):
                rem_val = None
            read_method = e.get("read_method")
            if read_method not in ("fill_line", "derivative_volume", None):
                read_method = None
            handling_status = e.get("handling_status")
            if handling_status not in (
                "handled", "visible_untouched", "not_visible", "false_detection",
            ):
                handling_status = None
            fd_actual = e.get("false_detection_actual")
            if not isinstance(fd_actual, str):
                fd_actual = None
            disp = e.get("dispensed_estimate")
            try:
                disp_val = float(disp) if disp is not None else None
            except (TypeError, ValueError):
                disp_val = None
            start = e.get("starting_amount")
            try:
                start_val = float(start) if start is not None else None
            except (TypeError, ValueError):
                start_val = None
            psegs_raw = e.get("per_segment_observations")
            if isinstance(psegs_raw, list):
                psegs = [s for s in psegs_raw if isinstance(s, dict)]
            else:
                psegs = None
            per_instance.append({
                "instance_id": iid,
                "handling_status": handling_status,
                "remaining": rem_val,
                "read_method": read_method,
                "starting_amount": start_val,
                "dispensed_estimate": disp_val,
                "per_segment_observations": psegs,
                "false_detection_actual": fd_actual,
                "reasoning": e.get("reasoning", ""),
                "evidence_frames": e.get("evidence_frames", []),
            })

    # Drop followup requests for iids that have a committed signal:
    # either a non-null `remaining` (fill_line) OR an opaque
    # `derivative_volume` reading where dispensed_estimate is the
    # committed answer and no further followup is needed.
    iids_committed: set[str] = set()
    for e in per_instance:
        iid = e.get("instance_id")
        if not iid:
            continue
        if e.get("remaining") is not None:
            iids_committed.add(iid)
            continue
        # Opaque-derivative commitment: read_method=derivative_volume +
        # dispensed_estimate populated. We can't tell package_type
        # here, so we conservatively treat this as committed only when
        # the observer did not also raise a followup for it (handled
        # by the dedup below).
    followups_raw = parsed.get("needs_followup") or []
    needs_followup: list[dict] = []
    if isinstance(followups_raw, list):
        for f in followups_raw:
            if isinstance(f, dict) and f.get("instance_id"):
                if f["instance_id"] in iids_committed:
                    continue
                tag = f.get("reason_tag")
                if tag not in (
                    "needs_starting_amount",
                    "needs_dispensal_window",
                    "container_not_visible",
                    None,
                ):
                    tag = None
                needs_followup.append({
                    "instance_id": f["instance_id"],
                    "reason_tag": tag,
                    "reason": f.get("reason", ""),
                })
    return window_observation, per_instance, needs_followup


# ---------------------------------------------------------------------------
# SessionState — tracks resolution + given_up across rounds
# ---------------------------------------------------------------------------

class SessionState:
    """Tracks per-iid resolution for the loop.

    A candidate iid is:
      - resolved     — observer returned non-null remaining with
                       handling_status ∈ {handled, visible_untouched},
                       OR planner emitted food_journey with
                       status=resolved and resolved_value.
      - given_up     — planner emitted food_journey with status=given_up.
      - unresolved   — otherwise.

    The pool of candidates we track = active_iids (iids whose class has
    per-frame evidence). Items with no evidence live in skipped_items
    and are not part of this tracking.
    """

    def __init__(self, inventory: list[dict], active_iids: set[str]):
        self.inv_by_iid = {inv["instance_id"]: inv for inv in inventory}
        self.active_iids = set(active_iids)
        self.latest: dict[str, dict] = {}  # iid -> latest observer per_instance entry
        self.pending_followups: dict[str, list[str]] = defaultdict(list)
        self.resolved: dict[str, dict] = {}  # iid -> {status, remaining, source}
        self.given_up: dict[str, str] = {}  # iid -> reason
        # Cumulative across all rounds. Planner may move an iid OUT of this
        # bucket in a later round by emitting a food_journey for it.
        self.skipped: dict[str, dict] = {}  # iid -> full skipped_items entry
        self.observer_rounds: list[dict] = []
        self.total_frames_used: int = 0
        self.latest_planner_journeys: list[dict] = []

    def apply_observer_round(
        self,
        window: dict,
        per_instance: list[dict],
        needs_followup: list[dict],
        window_observation: str,
        n_frames: int,
    ) -> None:
        self.total_frames_used += n_frames
        followup_iids = {
            f.get("instance_id")
            for f in (needs_followup or [])
            if isinstance(f, dict) and f.get("instance_id")
        }
        for entry in per_instance:
            iid = entry["instance_id"]
            if iid not in self.inv_by_iid:
                continue
            self.latest[iid] = entry
            status = entry.get("handling_status")
            rem = entry.get("remaining")
            # Observer-declared false_detection short-circuits the loop:
            # the planner doesn't need a round to confirm cross-talk.
            if status == "false_detection" and iid not in self.resolved:
                actual = entry.get("false_detection_actual")
                reason = "observer reported false_detection"
                if actual:
                    reason += f" — actual product visible: {actual}"
                self.given_up[iid] = reason
                self.pending_followups.pop(iid, None)
                continue
            # Observer's local reading is only treated as authoritative when
            # the observer itself did NOT flag a followup. If it did, the
            # planner must decide between a follow-up window and given_up.
            if iid in followup_iids:
                continue
            if rem is not None and status in ("handled", "visible_untouched"):
                self.resolved[iid] = {
                    "status": status,
                    "remaining": rem,
                    "starting": entry.get("starting_amount"),
                    "dispensed": entry.get("dispensed_estimate"),
                    "source": "observer_fill_line",
                }
                self.pending_followups.pop(iid, None)
                self.given_up.pop(iid, None)

        for f in needs_followup:
            iid = f.get("instance_id")
            reason = f.get("reason", "")
            if iid and iid in self.inv_by_iid and iid not in self.resolved:
                self.pending_followups[iid].append(reason)

        self.observer_rounds.append({
            "window": window,
            "window_observation": window_observation,
            "per_instance": per_instance,
            "needs_followup": needs_followup,
            "n_frames": n_frames,
        })

    def apply_planner_skips(self, skipped_items: list[dict]) -> None:
        """Accumulate planner-emitted skipped_items across rounds.

        Once an iid is skipped with a reason, keep it in the bucket so the
        next round's user prompt shows "previously skipped: <reason>". If
        the planner later revives the iid in food_journeys, apply_planner_journeys
        will remove it from this bucket.
        """
        if not isinstance(skipped_items, list):
            return
        for s in skipped_items:
            if not isinstance(s, dict):
                continue
            iid = s.get("instance_id")
            if iid and iid in self.inv_by_iid:
                self.skipped[iid] = s

    def apply_planner_journeys(self, journeys: list[dict]) -> None:
        """Read planner's food_journeys to pick up resolved / given_up flips."""
        if not isinstance(journeys, list):
            return
        self.latest_planner_journeys = journeys
        for j in journeys:
            if not isinstance(j, dict):
                continue
            status = j.get("status")
            iids = j.get("candidate_instance_ids") or []
            if not isinstance(iids, list):
                continue
            for iid in iids:
                iid = str(iid)
                if iid not in self.inv_by_iid:
                    continue
                # Any appearance in food_journeys means the planner revived it
                # from a prior skip; drop the stale skip entry.
                self.skipped.pop(iid, None)
                if status == "resolved":
                    val = j.get("resolved_value")
                    try:
                        val_num = float(val) if val is not None else None
                    except (TypeError, ValueError):
                        val_num = None
                    disp = j.get("resolved_dispensed")
                    try:
                        disp_num = float(disp) if disp is not None else None
                    except (TypeError, ValueError):
                        disp_num = None
                    start = j.get("resolved_starting")
                    try:
                        start_num = float(start) if start is not None else None
                    except (TypeError, ValueError):
                        start_num = None
                    if (val_num is not None or disp_num is not None) and iid not in self.resolved:
                        self.resolved[iid] = {
                            "status": "handled",
                            "remaining": val_num,
                            "starting": start_num,
                            "dispensed": disp_num,
                            "source": "planner_journey",
                        }
                        self.pending_followups.pop(iid, None)
                        self.given_up.pop(iid, None)
                elif status == "given_up":
                    reason = j.get("given_up_reason") or "(planner gave up)"
                    if iid not in self.resolved:
                        self.given_up[iid] = reason
                        self.pending_followups.pop(iid, None)

    def unresolved_iids(self) -> set[str]:
        return {
            iid for iid in self.active_iids
            if iid not in self.resolved
            and iid not in self.given_up
            and iid not in self.skipped
        }

    def skipped_lines(self) -> str:
        if not self.skipped:
            return "  (none)"
        lines = []
        for iid in sorted(self.skipped):
            reason = self.skipped[iid].get("reasoning") or "(no reason recorded)"
            lines.append(f"  - {iid}: {reason}")
        return "\n".join(lines)

    def resolved_lines(self) -> str:
        if not self.resolved:
            return "  (none yet)"
        lines = []
        for iid in sorted(self.resolved):
            r = self.resolved[iid]
            lines.append(f"  - {iid} [{r['status']}] remaining={r['remaining']} (via {r['source']})")
        return "\n".join(lines)

    def unresolved_lines(self) -> str:
        unresolved = self.unresolved_iids()
        if not unresolved:
            return "  (all resolved or given up)"
        lines = []
        for iid in sorted(unresolved):
            entry = self.latest.get(iid)
            followups = self.pending_followups.get(iid, [])
            followup_str = f"  followup: {followups[-1]}" if followups else ""
            if entry is None:
                lines.append(f"  - {iid}: no observer read yet{followup_str}")
            else:
                status = entry.get("handling_status", "?")
                rem = entry.get("remaining")
                lines.append(
                    f"  - {iid} [{status}] remaining={rem}{followup_str}"
                )
        return "\n".join(lines)

    def given_up_lines(self) -> str:
        if not self.given_up:
            return "  (none)"
        lines = []
        for iid in sorted(self.given_up):
            lines.append(f"  - {iid}: {self.given_up[iid]}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Session pipeline (iterative loop)
# ---------------------------------------------------------------------------

def process_session(
    participant: str,
    session: str,
    client,
    ledger: dict,
    model: str,
    fps: float,
    max_frames: int,
    min_score: float,
    max_rounds: int,
    planner_only: bool = False,
    verbose: bool = False,
    model_tag: str = "default",
    run_tag: str = "default",
    inventory_scope: str = "full",
    transparency_by_iid: dict | None = None,
    evidence_mode: str = "per_frame",
    seg_tau_siglip: float = 0.15,
    seg_tau_dino: float = 0.15,
    seg_gap_close: float = 2.0,
    seg_min_duration: float = 1.5,
    flicker_min_score: float = 0.15,
    flicker_min_hits: int = 2,
    flicker_peak_score: float = 0.25,
) -> tuple[list[dict], dict]:
    session_log: dict = {
        "session": session,
        "planner_rounds": [],
        "observer_rounds": [],
        "skipped_items": [],
        "final_journeys": [],
    }

    cache_dir = CACHE_DIR / participant / session / model_tag / run_tag
    cache_dir.mkdir(parents=True, exist_ok=True)

    inventory = load_inventory(participant, session, scope=inventory_scope)
    if not inventory:
        print(f"  {session}: no inventory for scope={inventory_scope}")
        return [], session_log

    try:
        all_ts, hoi_ts = load_hoi_timestamps(participant, session)
    except FileNotFoundError as e:
        print(f"  {session}: missing hands23 results ({e}) — skipping")
        return [], session_log
    if not hoi_ts:
        print(f"  {session}: no HOI-contact frames — skipping")
        return [], session_log

    siglip_by_t = load_siglip_by_t(participant, session)
    dino_by_t = load_dino_by_t(participant, session)
    scene_by_t = load_owlv2_scene_by_t(participant, session)
    hoi_details_by_t = load_hoi_details_by_t(participant, session)

    hoi_sorted = sorted(hoi_ts)
    print(f"  {session}: {len(all_ts)} frames, {len(hoi_sorted)} HOI-contact, "
          f"{len(inventory)} inventory items ({inventory_scope} scope), "
          f"{len(scene_by_t)} OWLv2 scene tags")

    inv_by_iid = {inv["instance_id"]: inv for inv in inventory}
    inv_by_vc: dict[str, list[str]] = defaultdict(list)
    for inv in inventory:
        inv_by_vc[inv["visual_class"]].append(inv["instance_id"])

    if evidence_mode == "segments":
        evidence_text, ev_stats, active_vcs, active_iids = format_per_item_segments_evidence(
            participant, session, inventory, transparency_by_iid,
            min_score=min_score,
            siglip_by_t=siglip_by_t,
            dino_by_t=dino_by_t,
            hoi_sorted=hoi_sorted,
            seg_tau_siglip=seg_tau_siglip,
            seg_tau_dino=seg_tau_dino,
            seg_gap_close=seg_gap_close,
            seg_min_duration=seg_min_duration,
            inventory_scope=inventory_scope,
            flicker_min_score=flicker_min_score,
            flicker_min_hits=flicker_min_hits,
            flicker_peak_score=flicker_peak_score,
        )
    elif evidence_mode == "chrono":
        evidence_text, ev_stats, active_vcs, active_iids = format_chronological_segments_evidence(
            participant, session, inventory, transparency_by_iid,
            min_score=min_score,
            siglip_by_t=siglip_by_t,
            dino_by_t=dino_by_t,
            hoi_sorted=hoi_sorted,
            seg_tau_siglip=seg_tau_siglip,
            seg_tau_dino=seg_tau_dino,
            seg_gap_close=seg_gap_close,
            seg_min_duration=seg_min_duration,
            inventory_scope=inventory_scope,
            flicker_min_score=flicker_min_score,
            flicker_min_hits=flicker_min_hits,
            flicker_peak_score=flicker_peak_score,
        )
    else:
        evidence_text, ev_stats, active_vcs, active_iids = format_per_item_evidence(
            hoi_sorted, siglip_by_t, dino_by_t, scene_by_t, hoi_details_by_t,
            inventory,
            min_score=min_score,
            transparency_by_iid=transparency_by_iid,
        )
    inventory_text = format_inventory_for_prompt(inventory, transparency_by_iid)

    print(f"  Evidence ({evidence_mode}): {ev_stats['n_items_emitted']} items with hits "
          f"({ev_stats['n_rows_emitted']} rows across {ev_stats['n_hoi_frames_total']} HOI frames)")

    system_prompt_text = PLANNER_SYSTEM_PROMPT.format(
        min_score=f"{min_score:.2f}",
        max_rounds=max_rounds,
        max_frames=max_frames,
    )
    if evidence_mode == "segments":
        evidence_format_note = (
            "HOI-gated temporal SEGMENTS per visual_class — one row per "
            "coherent on-contact window: [start-end] dur hot/total "
            "sig_peak dino_peak scene=<top1:n,top2:n,unk:n> iid. "
            "`hot` = frames in the window where hands23 reported hand-on-item "
            "contact AND SigLIP/DINO for this vc/iid crossed its threshold. "
            "A `FLICKER ONLY` row means the item had per-frame HOI hits but "
            "no segment formed under the morphology thresholds — treat as "
            "weaker evidence but still worth a food_journey."
        )
    elif evidence_mode == "chrono":
        evidence_format_note = (
            "CHRONOLOGICAL timeline of HOI-gated segments — one row per "
            "(item × interval), sorted by start time: "
            "[start-end] dur=<s> dino=<peak> scene=<top1:n,top2:n,unk:n> \"<visual_class>\". "
            "Multiple items active in the same time window appear as "
            "consecutive rows — the timeline is NOT de-duplicated, so use "
            "DINO score + scene to judge which item is the actual focus. "
            "Look up package_type and instance_ids in the Session Inventory "
            "section above (each visual_class lists its iids and pkg)."
        )
    else:
        evidence_format_note = (
            "grouped by visual_class; chronological per item; HOI contact only"
        )
    round0_user_text = PLANNER_ROUND0_USER.format(
        inventory=inventory_text,
        evidence=evidence_text,
        max_frames=max_frames,
        evidence_format_note=evidence_format_note,
    )
    messages: list[dict] = [
        {"role": "system", "content": [{"type": "input_text", "text": system_prompt_text}]},
        {"role": "user",   "content": [{"type": "input_text", "text": round0_user_text}]},
    ]
    (cache_dir / "planner_round0_prompt.txt").write_text(
        system_prompt_text + "\n\n---USER---\n\n" + round0_user_text
    )

    clips = get_session_clips(participant, session) if not planner_only else []
    video_durations = [(path, dur) for _, path, dur in clips]

    state = SessionState(inventory, active_iids)
    prior_observations: list[str] = []

    planner_model = "gpt-5.4"
    vllm_endpoint = VLLM_ENDPOINTS.get(model.lower())
    use_vllm = vllm_endpoint is not None or model.lower().startswith("qwen") or model.lower().startswith("gemma")

    rounds_used = 0
    done = False
    final_parsed: dict | None = None

    while rounds_used < max_rounds and not done:
        round_idx = rounds_used + 1
        print(f"  Round {round_idx}/{max_rounds}: planner...", end="", flush=True)

        planner_text, planner_stats = call_planner(client, messages, planner_model)
        (cache_dir / f"planner_round{round_idx}_response.txt").write_text(planner_text or "")
        parsed = _parse_planner_json(planner_text)

        session_log["planner_rounds"].append({
            "round": round_idx,
            "stats": planner_stats,
            "raw_response": planner_text,
            "parsed": parsed,
        })

        if parsed is None:
            print(f" parse failed — aborting session")
            break

        final_parsed = parsed
        messages.append({
            "role": "assistant",
            "content": [{"type": "output_text", "text": planner_text}],
        })

        # Absorb planner's food_journeys into state (may flip to resolved/given_up).
        # Important ordering: apply skips FIRST so journeys can revive an iid
        # (apply_planner_journeys drops revived iids from state.skipped).
        state.apply_planner_skips(parsed.get("skipped_items") or [])
        state.apply_planner_journeys(parsed.get("food_journeys") or [])

        action = parsed.get("action") or {}
        action_type = action.get("type")

        if action_type == "stop":
            print(f" STOP ({state.total_frames_used} frames used)")
            done = True
            break

        if action_type != "observe":
            print(f" unexpected action.type={action_type} — aborting")
            break

        # --- Parse & validate the batch of observation_windows --------------
        windows_raw = action.get("observation_windows") or []
        if not isinstance(windows_raw, list) or not windows_raw:
            print(f" no observation_windows in batch — aborting")
            break

        validated_windows: list[dict] = []
        for w_raw in windows_raw:
            w, err = _validate_window(w_raw, inv_by_iid, inv_by_vc)
            if w is None:
                if verbose:
                    print(f"\n    DROP invalid window: {err}")
                continue
            validated_windows.append(w)

        if not validated_windows:
            print(f" all windows in batch invalid — aborting")
            break

        total_segments = sum(len(w["segments"]) for w in validated_windows)
        print(f" emitted {len(validated_windows)}-window batch "
              f"({total_segments} segments total)")

        if planner_only:
            # Record the planned batch for inspection, but don't run observers.
            for w in validated_windows:
                session_log["observer_rounds"].append({
                    "round": round_idx,
                    "window": w,
                    "window_observation": "(planner_only — skipped)",
                    "per_instance": [],
                    "needs_followup": [],
                    "stats": {},
                    "n_frames": 0,
                    "raw_response": "",
                })
            rounds_used += 1
            continue

        if not clips:
            print(f"    {session}: no video clips found; aborting observer")
            break

        # --- Run observers for every window in the batch --------------------
        batch_results: list[dict] = []  # parallel to validated_windows
        frames_this_round = 0
        for w_idx, window in enumerate(validated_windows):
            segs = window["segments"]
            vc = window["visual_class"]
            cand_iids = window["candidate_instance_ids"]
            candidates = [inv_by_iid[c] for c in cand_iids if c in inv_by_iid]
            if not candidates:
                continue

            frames, frame_ts = extract_window_frames(
                segs, video_durations,
                padding=2.0, fps=fps,
                max_frames=max_frames,
            )
            if not frames:
                print(f"    observer {vc}: no frames extracted")
                continue

            vc_slug = re.sub(r"[^A-Za-z0-9]+", "_", vc).strip("_").lower() or "obs"
            obs_prompt_path = cache_dir / f"round{round_idx}_w{w_idx:02d}_{vc_slug}_observer_prompt.txt"
            obs_response_path = cache_dir / f"round{round_idx}_w{w_idx:02d}_{vc_slug}_observer_response.txt"

            segs_str = "+".join(f"{s[0]:.0f}-{s[1]:.0f}" for s in segs)
            print(f"    observer[{w_idx+1}/{len(validated_windows)}]: {vc} "
                  f"[{segs_str}s] {len(frames)} frames...",
                  end="", flush=True)

            if use_vllm:
                if vllm_endpoint:
                    obs_url, obs_model = vllm_endpoint
                else:
                    obs_url, obs_model = QWEN_URL, model
                obs_text, obs_prompt, obs_stats = run_observer_qwen(
                    frames, frame_ts, candidates, window, prior_observations,
                    qwen_url=obs_url, qwen_model=obs_model,
                    prompt_save_path=obs_prompt_path,
                )
            else:
                obs_text, obs_prompt, obs_stats = run_observer(
                    client, frames, frame_ts, candidates, window, prior_observations,
                    model, prompt_save_path=obs_prompt_path,
                )
            obs_response_path.write_text(obs_text or "")
            window_observation, per_instance, needs_followup = parse_observer_response(obs_text)

            if not per_instance:
                print(f" parse failed / empty per_instance")
                state.observer_rounds.append({
                    "round": round_idx,
                    "window": window,
                    "window_observation": window_observation or "(parse failed)",
                    "per_instance": [],
                    "needs_followup": [],
                    "stats": obs_stats,
                    "n_frames": len(frames),
                    "raw_response": obs_text,
                })
                state.total_frames_used += len(frames)
                frames_this_round += len(frames)
                batch_results.append({
                    "window": window,
                    "window_observation": window_observation or "(parse failed)",
                    "per_instance": [],
                    "needs_followup": [],
                })
                continue

            state.apply_observer_round(
                window, per_instance, needs_followup, window_observation, len(frames),
            )
            frames_this_round += len(frames)
            emitted = []
            for entry in per_instance:
                emitted.append(
                    f"{entry['instance_id']}:{entry['handling_status']}={entry['remaining']}"
                )
            print(f" {' '.join(emitted)}")
            if window_observation:
                prior_observations.append(
                    f"[{segs_str}s vc={vc}] {window_observation}"
                )
            session_log["observer_rounds"].append({
                "round": round_idx,
                "window": window,
                "window_observation": window_observation,
                "per_instance": per_instance,
                "needs_followup": needs_followup,
                "stats": obs_stats,
                "n_frames": len(frames),
                "raw_response": obs_text,
            })
            batch_results.append({
                "window": window,
                "window_observation": window_observation,
                "per_instance": per_instance,
                "needs_followup": needs_followup,
            })

        # --- Build the follow-up user turn with the aggregated batch -------
        observer_batch_json_str = json.dumps(batch_results, indent=2)
        followup_user_text = PLANNER_FOLLOWUP_USER.format(
            round_idx=round_idx,
            n_windows=len(batch_results),
            n_frames_round=frames_this_round,
            observer_batch_json=observer_batch_json_str,
            resolved_lines=state.resolved_lines(),
            unresolved_lines=state.unresolved_lines(),
            given_up_lines=state.given_up_lines(),
            skipped_lines=state.skipped_lines(),
            rounds_used=round_idx,
            max_rounds=max_rounds,
            frames_used=state.total_frames_used,
        )
        messages.append({
            "role": "user",
            "content": [{"type": "input_text", "text": followup_user_text}],
        })
        (cache_dir / f"planner_round{round_idx + 1}_user.txt").write_text(followup_user_text)

        rounds_used += 1

        # Short-circuit: if no unresolved items remain, don't spend another
        # planner call just to get a `stop` — but only after at least one
        # round of observers (round 0 may emit a batch that instantly
        # resolves everything).
        if not state.unresolved_iids() and round_idx >= 1:
            print(f"    all items resolved/given_up; exiting loop without final planner call")
            done = True
            break

    if not done and rounds_used >= max_rounds:
        print(f"  max_rounds ({max_rounds}) reached without planner saying stop")

    skipped_items = list(state.skipped.values())
    session_log["skipped_items"] = skipped_items
    if final_parsed is not None:
        session_log["final_journeys"] = final_parsed.get("food_journeys") or []

    # Build predictions from resolved iids (observer or planner-journey source).
    predictions: list[dict] = []
    for iid, r in state.resolved.items():
        inv = inv_by_iid.get(iid)
        if inv is None:
            continue
        status = r["status"]
        rem = r["remaining"]
        # visible_untouched ⇒ fall back to ledger carry-forward; omit from preds.
        if status == "visible_untouched":
            continue
        latest_obs = state.latest.get(iid, {})
        # Collect segments from all observer rounds that touched this iid.
        seg_list = []
        for rr in state.observer_rounds:
            if any(p.get("instance_id") == iid for p in rr.get("per_instance", [])):
                w = rr.get("window", {})
                if "start" in w and "end" in w:
                    seg_list.append([w["start"], w["end"]])
        predictions.append({
            "session": session,
            "item": inv["visual_class"],
            "instance_id": iid,
            "amount_remaining": rem,
            "amount_starting": r.get("starting"),
            "amount_dispensed": r.get("dispensed"),
            "handling_status": status,
            "reasoning": latest_obs.get("reasoning", ""),
            "evidence_frames": latest_obs.get("evidence_frames", []),
            "segments": seg_list,
            "resolution_source": r["source"],
            "stats": {
                "n_planner_rounds": len(session_log["planner_rounds"]),
                "n_observer_rounds": len(session_log["observer_rounds"]),
                "total_observer_frames": state.total_frames_used,
            },
        })

    print(f"  {session}: {len(predictions)} predictions, "
          f"{len(state.resolved)} resolved iids, "
          f"{len(state.given_up)} given_up, "
          f"{len(state.unresolved_iids())} still-unresolved; "
          f"{state.total_frames_used} frames, "
          f"{len(session_log['planner_rounds'])} planner rounds, "
          f"{len(session_log['observer_rounds'])} observer rounds")
    return predictions, session_log


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="AVP Round 1 (remaining-only, Iterative): batched "
                    "plan-observe loop rebased on CandList_HOI_PerItem. "
                    "Planner runs in a persistent chat. Round 0 emits a "
                    "full batch of observation windows covering all active "
                    "visual_classes; later rounds emit smaller batches for "
                    "still-unresolved items. Stops when all items are "
                    "resolved or given_up."
    )
    parser.add_argument("--participant", default="kailai")
    parser.add_argument("--tag", required=True,
                        help="Short label for this run (e.g. 'Iterative_v1').")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--session", help="Single session")
    group.add_argument("--all", action="store_true", help="All sessions")
    parser.add_argument("--model", default="gpt-5.4",
                        help="Observer model (planner always uses gpt-5.4)")
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--max-frames", type=int, default=MAX_IMAGES,
                        help="Max frames per single observer call.")
    parser.add_argument("--max-rounds", type=int, default=DEFAULT_MAX_ROUNDS,
                        help="Max plan-observe cycles per session.")
    parser.add_argument("--min-score", type=float, default=0.15,
                        help="Only include per-frame detections with DINO or SigLIP >= this.")
    parser.add_argument("--planner-only", action="store_true",
                        help="Run planner rounds only; skip observer calls.")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", action="store_true",
                        help="Resume a previously-interrupted run with the same --tag.")
    parser.add_argument("--inventory-scope", choices=["full", "session"], default="full")
    parser.add_argument(
        "--evidence-mode",
        choices=["segments", "chrono", "per_frame"],
        default="segments",
        help="How to render per-item detections in the round-0 planner prompt. "
             "'segments' (default) groups segment rows under each visual_class. "
             "'chrono' emits a single chronological timeline of segments (one row per "
             "(item × interval), sorted by start). 'per_frame' falls back to the dense "
             "row-per-frame format used by CandList_HOI_PerItem.",
    )
    parser.add_argument("--seg-tau-siglip", type=float, default=0.15,
                        help="SigLIP threshold for 05b segment activation (segments mode).")
    parser.add_argument("--seg-tau-dino", type=float, default=0.15,
                        help="DINO threshold for 05b segment activation (segments mode).")
    parser.add_argument("--seg-gap-close", type=float, default=2.0,
                        help="Seconds between active runs to merge into one segment (segments mode).")
    parser.add_argument("--seg-min-duration", type=float, default=1.5,
                        help="Minimum segment duration in seconds (segments mode).")
    parser.add_argument("--flicker-min-score", type=float, default=0.15,
                        help="SigLIP/DINO score floor for the Tier 2 flicker tail. "
                             "Held independently of --min-score; the flicker gate uses "
                             "max(min_score, flicker_min_score). Raise to suppress the "
                             "long tail of low-confidence single-frame hits.")
    parser.add_argument("--flicker-min-hits", type=int, default=2,
                        help="Minimum number of per-frame hits required for a Tier 2 "
                             "flicker cluster to be emitted (segments and chrono modes). "
                             "Default 2 drops singleton 0-duration rows (the dominant "
                             "noise source under HOI-gated FPS=1). Raising to 3 disables "
                             "Tier 2 entirely on the current dataset.")
    parser.add_argument("--flicker-peak-score", type=float, default=0.25,
                        help="Required peak SigLIP/DINO score within a Tier 2 cluster. "
                             "Default 0.25 keeps clusters that have at least one hit at "
                             "moderate confidence and drops weak-everywhere clusters.")
    args = parser.parse_args()

    sessions = (
        [args.session] if args.session
        else get_sessions(args.participant) if args.all
        else get_sessions(args.participant)
    )

    model_tag = args.model.replace("/", "_")
    run_tag = args.tag.replace("/", "_")

    print(f"{'=' * 70}")
    print(f"AVP Round 1 (Remaining-Only, Iterative — on CandList_HOI_PerItem)")
    print(f"{'=' * 70}")
    print(f"Participant:  {args.participant}")
    print(f"Model:        {args.model}")
    print(f"Tag:          {run_tag}")
    print(f"FPS:          {args.fps}")
    print(f"Max frames:   {args.max_frames} per observer call")
    print(f"Max rounds:   {args.max_rounds}")
    print(f"Min score:    {args.min_score}")
    print(f"Flicker:      min_score={args.flicker_min_score} min_hits={args.flicker_min_hits} peak_score={args.flicker_peak_score}")
    print(f"Planner only: {args.planner_only}")
    print(f"Sessions:     {len(sessions)}")
    print()

    client = make_client()
    ledger = load_ledger(args.participant)
    transparency_by_iid = load_transparency_profile(args.participant)
    print(f"Transparency profile: {len(transparency_by_iid)} items tagged "
          f"(from confusable_profile.json)")

    all_predictions: list[dict] = []
    all_logs: list[dict] = []

    out_dir = participant_dir(args.participant) / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    preds_path = args.output or out_dir / f"{OUTPUT_PREFIX}_{model_tag}_{run_tag}_preds.json"
    status_path = out_dir / f"{OUTPUT_PREFIX}_{model_tag}_{run_tag}_status.json"

    status: dict = {"completed_sessions": [], "failed_sessions": []}
    if args.resume and preds_path.exists() and status_path.exists():
        all_predictions = json.loads(preds_path.read_text())
        status = json.loads(status_path.read_text())
        for s in status.get("completed_sessions", []):
            sess_log_path = outputs_dir(args.participant, s) / f"{OUTPUT_PREFIX}_{model_tag}_{run_tag}_planner.json"
            if sess_log_path.exists():
                try:
                    sess_data = json.loads(sess_log_path.read_text())
                    if "session" in sess_data:
                        all_logs.append(sess_data["session"])
                except json.JSONDecodeError:
                    pass
        completed = set(status.get("completed_sessions", []))
        pending = [s for s in sessions if s not in completed]
        print(f"\nRESUME: {len(completed)} session(s) already complete, "
              f"{len(pending)} pending (of {len(sessions)} total).")
        sessions = pending
    elif args.resume:
        print("\nRESUME requested but no existing predictions/status found — starting fresh.")

    failed_sessions: list[tuple[str, str]] = []
    for session in sessions:
        session_failed = False
        try:
            preds, log = process_session(
                participant=args.participant,
                session=session,
                client=client,
                ledger=ledger,
                model=args.model,
                fps=args.fps,
                max_frames=args.max_frames,
                min_score=args.min_score,
                max_rounds=args.max_rounds,
                planner_only=args.planner_only,
                verbose=args.verbose,
                model_tag=model_tag,
                run_tag=run_tag,
                inventory_scope=args.inventory_scope,
                transparency_by_iid=transparency_by_iid,
                evidence_mode=args.evidence_mode,
                seg_tau_siglip=args.seg_tau_siglip,
                seg_tau_dino=args.seg_tau_dino,
                seg_gap_close=args.seg_gap_close,
                seg_min_duration=args.seg_min_duration,
                flicker_min_score=args.flicker_min_score,
                flicker_min_hits=args.flicker_min_hits,
                flicker_peak_score=args.flicker_peak_score,
            )
        except KeyboardInterrupt:
            raise
        except Exception as e:
            import traceback
            print(f"\n  ERROR in session {session}: {e}")
            traceback.print_exc()
            failed_sessions.append((session, str(e)[:200]))
            preds, log = [], {"session": session, "planner_rounds": [], "observer_rounds": [],
                              "error": str(e)[:500]}
            session_failed = True
            status["failed_sessions"] = [
                f for f in status["failed_sessions"] if f.get("session") != session
            ] + [{"session": session, "error": str(e)[:200]}]
            status_path.write_text(json.dumps(status, indent=2) + "\n")
            all_logs.append(log)
            if not args.planner_only:
                with open(preds_path, "w", encoding="utf-8") as f:
                    json.dump(all_predictions, f, indent=2)
            print(f"\n  HALTED at session {session}. Fix and re-run with --resume.")
            break

        all_predictions.extend(preds)
        all_logs.append(log)

        if not session_failed:
            if session not in status["completed_sessions"]:
                status["completed_sessions"].append(session)
            status["failed_sessions"] = [
                f for f in status["failed_sessions"] if f.get("session") != session
            ]
            status_path.write_text(json.dumps(status, indent=2) + "\n")

        if not args.planner_only:
            with open(preds_path, "w", encoding="utf-8") as f:
                json.dump(all_predictions, f, indent=2)

        sess_out = outputs_dir(args.participant, session)
        sess_out.mkdir(parents=True, exist_ok=True)
        planner_sess_path = sess_out / f"{OUTPUT_PREFIX}_{model_tag}_{run_tag}_planner.json"
        with open(planner_sess_path, "w", encoding="utf-8") as f:
            json.dump({
                "participant": args.participant,
                "timestamp": datetime.now().isoformat(),
                "model": args.model,
                "tag": run_tag,
                "session": log,
            }, f, indent=2)

        if not args.planner_only:
            preds_sess_path = sess_out / f"{OUTPUT_PREFIX}_{model_tag}_{run_tag}_preds.json"
            with open(preds_sess_path, "w", encoding="utf-8") as f:
                json.dump(preds, f, indent=2)
            print(f"  Saved: {preds_sess_path}")
        print()

    planner_path = out_dir / f"{OUTPUT_PREFIX}_{model_tag}_{run_tag}_planner.json"
    with open(planner_path, "w", encoding="utf-8") as f:
        json.dump({
            "participant": args.participant,
            "timestamp": datetime.now().isoformat(),
            "model": args.model,
            "tag": run_tag,
            "sessions": all_logs,
        }, f, indent=2)
    print(f"Saved: {planner_path}")

    if not args.planner_only:
        print(f"Saved: {preds_path}")
    print(f"Status saved to {status_path}")

    if not args.planner_only and all_predictions:
        print(f"\n{'=' * 70}")
        print(f"Predictions ({len(all_predictions)} items):")
        for p in all_predictions:
            unit = "count" if ledger["items"][p["instance_id"]]["unit"] == "count" else "g"
            print(f"  [{p['session']}] {p['item']}: "
                  f"remaining={p['amount_remaining']} {unit} "
                  f"[{p['handling_status']}] via {p['resolution_source']}")

    if failed_sessions:
        print(f"\n{len(failed_sessions)} session(s) failed in this run:")
        for s, err in failed_sessions:
            print(f"  {s}: {err}")
        print(f"\nRe-run with --resume to retry the failed sessions only.")


if __name__ == "__main__":
    main()

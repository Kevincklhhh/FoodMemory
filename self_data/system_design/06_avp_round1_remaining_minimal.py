#!/usr/bin/env python3
"""AVP Round 1 (remaining-only, minimal): single-shot plan + observe.

Strips the iterative loop from `06_avp_round1_remaining_Iterative.py` while
keeping its richer planner context, and pairs it with the simpler observer
prompt from `06_avp_round1_remaining_noTAD.py`.

Pipeline:
  Step 1 (Planner, text-only):
    Input — same as Iterative round 0 except SigLIP is dropped entirely
    (not a useful signal in practice): session inventory + per-item HOI /
    DINO evidence (segments / chrono / per_frame mode). The planner runs
    ONCE and emits one decision entry per inventory iid whose visual_class
    has any detection burst.
  Step 2 (Observer, noTAD-style):
    For each iid the planner marked `observe`, run a single observer call
    on the planner-supplied segments. The observer returns the simple
    {item_confirmed, reasoning, evidence_frames, amount_remaining} schema
    — no candidate disambiguation, no needs_followup.

What's intentionally dropped vs Iterative:
  - food_journeys / usage_status / observation rounds 1+ / persistent chat
  - candidate-aware per_instance observer + needs_followup
  - replanning playbook

What's kept vs Iterative:
  - inventory + transparency tags
  - per-item HOI+DINO evidence (all three render modes)
  - HOI-gated segment compression via 05b_per_item_segments

NOTE: the evidence loaders / formatters below are duplicated from
`06_avp_round1_remaining_Iterative.py` rather than imported, so this branch
can be edited independently of the Iterative variant.

Usage:
  python system_design/06_avp_round1_remaining_minimal.py \
      --participant kailai --tag minimal_v1
  python system_design/06_avp_round1_remaining_minimal.py \
      --participant kailai --session 20260318-181229 \
      --tag minimal_smoke_v1 --evidence-mode segments
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

CACHE_DIR = Path(__file__).resolve().parent.parent / "cache" / "avp_remaining_minimal"

QWEN_URL = "http://saltyfish.eecs.umich.edu:8000/v1/chat/completions"
QWEN_MODEL_DEFAULT = "Qwen/Qwen3-VL-30B-A3B-Instruct"

VLLM_ENDPOINTS = {
    "qwen": ("http://saltyfish.eecs.umich.edu:8000/v1/chat/completions", "Qwen/Qwen3-VL-30B-A3B-Instruct"),
    "gemma": ("http://saltyfish.eecs.umich.edu:8000/v1/chat/completions", "google/gemma-4-31B-it"),
}

OUTPUT_PREFIX = "avp_minimal_remaining"


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


def make_observer_client(model: str):
    """Return the right SDK client for the observer model.

    - `gemini-*` → google-genai client (handles 100+ frames cheaply, faster
      reasoning, much lower $ under canonical Gemini paid-tier pricing).
    - anything else → Azure OpenAI client (gpt-5.x).
    """
    if model.startswith("gemini-"):
        from google import genai
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "Missing GOOGLE_API_KEY — required for Gemini observer model "
                f"'{model}'. Set GOOGLE_API_KEY or pass --model gpt-5.4."
            )
        return genai.Client(api_key=api_key)
    return make_client(model)


def _observer_api_call(observer_client, model: str, prompt: str,
                       frames_b64: list, timestamps: list) -> tuple[str, int, int]:
    """Single observer call. Dispatches on model prefix.

    Returns (response_text, input_tokens, output_tokens). Per the canonical
    cost convention (project_avp_experiment_baseline.md), output_tokens
    INCLUDES reasoning/thinking tokens for both providers.
    """
    if model.startswith("gemini-"):
        from google.genai import types
        import base64 as _b64
        contents = [types.Part.from_text(text=prompt)]
        for i, (fb64, ts) in enumerate(zip(frames_b64, timestamps), 1):
            contents.append(types.Part.from_text(text=f"[t={ts:.1f}s]"))
            contents.append(types.Part.from_bytes(
                data=_b64.b64decode(fb64), mime_type="image/jpeg",
            ))
        resp = observer_client.models.generate_content(
            model=model, contents=contents,
            config=types.GenerateContentConfig(
                # Sweep observer can return 10+ items each with multi-sentence
                # reasoning + a per_segment_observations block. 8192 truncates
                # around 6 items mid-reasoning, leaving the JSON unparseable.
                # 32768 is comfortable headroom and gemini-2.5-pro caps higher.
                temperature=0.3, max_output_tokens=32768,
                thinking_config=types.ThinkingConfig(
                    thinking_budget=16384, include_thoughts=True,
                ),
            ),
        )
        # Strip thought parts from the user-visible response; keep all parts
        # for token accounting via usage_metadata.
        text = ""
        if resp.candidates and resp.candidates[0].content:
            for part in resp.candidates[0].content.parts or []:
                if hasattr(part, "thought") and part.thought:
                    continue
                text += (part.text or "")
        um = resp.usage_metadata
        in_tok = (um.prompt_token_count if um else 0) or 0
        out_tok = (um.candidates_token_count if um else 0) or 0
        # Per convention: include thoughts in output tokens.
        thoughts = getattr(um, "thoughts_token_count", None) if um else None
        if thoughts:
            out_tok += thoughts
        return text, int(in_tok), int(out_tok)

    # Azure OpenAI (gpt-5.x) path — output_tokens already includes reasoning.
    content = [{"type": "input_text", "text": prompt}]
    for i, (fb64, ts) in enumerate(zip(frames_b64, timestamps), 1):
        content.append({"type": "input_text", "text": f"[t={ts:.1f}s]"})
        content.append({
            "type": "input_image",
            "image_url": f"data:image/jpeg;base64,{fb64}",
            "detail": "high",
        })
    resp = observer_client.responses.create(
        model=model,
        input=[{"role": "user", "content": content}],
        reasoning={"effort": "medium"},
    )
    text = resp.output_text or ""
    u = resp.usage
    return text, int(u.input_tokens), int(u.output_tokens)


# ---------------------------------------------------------------------------
# Per-frame data loading (duplicated from Iterative)
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
# Per-item evidence formatter (duplicated from Iterative)
# ---------------------------------------------------------------------------

_HAND_LABEL = {"left_hand": "L", "right_hand": "R"}
_HIDDEN_OBJ_TOUCH = {"neither_held", "neither_touched", None, ""}


def format_per_item_evidence(
    hoi_ts_sorted: list[float],
    dino_by_t: dict[float, dict[str, dict[str, float]]],
    scene_by_t: dict[float, str],
    hoi_details_by_t: dict[float, dict[str, dict]],
    inventory: list[dict],
    min_score: float = 0.15,
    transparency_by_iid: dict | None = None,
) -> tuple[str, dict, set[str], set[str]]:
    """Group HOI-gated DINO detections per visual_class, chronological per item."""
    transparency_by_iid = transparency_by_iid or {}

    inv_iids = {inv["instance_id"] for inv in inventory}
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
        hands_detail = hoi_details_by_t.get(t) or {}
        scene = scene_by_t.get(t, "unknown") or "unknown"

        for hs, detail in hands_detail.items():
            g = detail.get("grasp")
            ot = detail.get("obj_touch")
            if ot in _HIDDEN_OBJ_TOUCH:
                ot = None
            t_hand_hoi.setdefault(t, {})[hs] = (g, ot)

        all_hands = set(dino_by_hand) | set(hands_detail)
        for hs in all_hands:
            dino_vc: dict[str, float] = {}
            for iid, s in (dino_by_hand.get(hs) or {}).items():
                if iid not in inv_iids or s < min_score:
                    continue
                vc = iid_to_vc.get(iid)
                if not vc:
                    continue
                dino_vc[vc] = max(dino_vc.get(vc, 0.0), s)
            if not dino_vc:
                continue

            primary_vc = max(dino_vc, key=lambda k: dino_vc[k])
            primary_score = dino_vc[primary_vc]
            t_hand_primary.setdefault(t, {})[hs] = (primary_vc, primary_score)

            detail = hands_detail.get(hs) or {}
            grasp = detail.get("grasp")
            obj_touch = detail.get("obj_touch")
            if obj_touch in _HIDDEN_OBJ_TOUCH:
                obj_touch = None

            for vc, s in dino_vc.items():
                rows_by_vc[vc].append({
                    "t": t,
                    "hand": hs,
                    "scene": scene,
                    "grasp": grasp,
                    "touch": obj_touch,
                    "dino": s,
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

            scores_str = f"dino={r['dino']:.2f}"

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
# Segment-compressed evidence formatters (duplicated from Iterative)
# ---------------------------------------------------------------------------

def _scene_distribution_in_segment(
    start: float,
    end: float,
    owl_points: list[tuple[float, str]],
    pad: float = 0.5,
) -> tuple[str | None, str]:
    from collections import Counter
    window = [s for (t, s) in owl_points if start - pad <= t <= end + pad]
    if not window:
        return None, ""
    c = Counter(window)
    real = {s: v for s, v in c.items() if s != "unknown"}
    top = max(real, key=lambda k: real[k]) if real else None
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
    dino_by_t: dict[float, dict[str, dict[str, float]]],
    hoi_sorted: list[float],
    seg_tau_dino: float,
    seg_gap_close: float,
    seg_min_duration: float,
    inventory_scope: str = "full",
    flicker_min_score: float = 0.15,
    flicker_min_hits: int = 2,
    flicker_peak_score: float = 0.25,
) -> tuple[str, dict, set[str], set[str]]:
    """Segment-compressed counterpart to `format_per_item_evidence`."""
    transparency_by_iid = transparency_by_iid or {}

    # SigLIP is intentionally disabled — pass tau_siglip=1.1 so 05b never
    # activates a segment based on SigLIP alone.
    _, per_item = _PER_ITEM_SEGMENTS.build_session_segments(
        participant, session,
        tau_siglip=1.1,
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

    iid_to_vc = {inv["instance_id"]: inv["visual_class"] for inv in inventory}
    flicker_floor = max(min_score, flicker_min_score)
    flicker_hits: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for t in hoi_sorted:
        dino_by_hand = dino_by_t.get(t) or {}
        best_dino: dict[str, float] = {}
        for hs in dino_by_hand:
            for iid, s in (dino_by_hand.get(hs) or {}).items():
                if iid not in inv_iids or s < flicker_floor:
                    continue
                vc = iid_to_vc.get(iid)
                if not vc:
                    continue
                best_dino[vc] = max(best_dino.get(vc, 0.0), s)
        for vc, d in best_dino.items():
            flicker_hits[vc].append((t, d))

    lines: list[str] = []
    n_items_emitted = 0
    n_rows_emitted = 0
    active_vcs: set[str] = set()
    active_iids: set[str] = set()

    for vc in ordered_vcs:
        iids = vc_to_iids[vc]
        vc_segs: list[tuple[str, dict]] = []
        for iid in iids:
            _, segs = per_item.get(iid, ({}, []))
            for s in segs:
                vc_segs.append((iid, s))
        flickers = flicker_hits.get(vc, [])
        flicker_peak = max((d for _, d in flickers), default=0.0)

        if not vc_segs and (
            len(flickers) < flicker_min_hits
            or flicker_peak < flicker_peak_score
        ):
            continue
        n_items_emitted += 1
        active_vcs.add(vc)
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
                    f"dino_peak={s['peak_dino']:.2f} "
                    f"scene={scene_str}  iid={iid}"
                )
                n_rows_emitted += 1
        else:
            ts = sorted(t for t, _ in flickers)
            peak_dino = max((d for _, d in flickers), default=0.0)
            span = f"{ts[0]:.1f}-{ts[-1]:.1f}s" if ts else "?"
            sample = ", ".join(f"{t:.1f}" for t in ts[:6])
            if len(ts) > 6:
                sample += ", ..."
            lines.append(
                f"  FLICKER ONLY: {len(ts)} HOI-contact hits over {span} "
                f"(dino_peak={peak_dino:.2f}; sample ts: {sample}) — "
                f"no coherent segment formed under tau_dino={seg_tau_dino}, "
                f"gap_close={seg_gap_close}s, min_dur={seg_min_duration}s"
            )
            n_rows_emitted += 1
        lines.append("")

    stats = {
        "n_hoi_frames_total": len(hoi_sorted),
        "n_items_emitted": n_items_emitted,
        "n_rows_emitted": n_rows_emitted,
        "min_score": min_score,
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
    dino_by_t: dict[float, dict[str, dict[str, float]]],
    hoi_sorted: list[float],
    seg_tau_dino: float,
    seg_gap_close: float,
    seg_min_duration: float,
    inventory_scope: str = "full",
    flicker_min_score: float = 0.15,
    flicker_min_hits: int = 2,
    flicker_peak_score: float = 0.25,
) -> tuple[str, dict, set[str], set[str]]:
    """Chronological counterpart to `format_per_item_segments_evidence`."""
    transparency_by_iid = transparency_by_iid or {}

    # SigLIP is intentionally disabled — pass tau_siglip=1.1 so 05b never
    # activates a segment based on SigLIP alone.
    _, per_item = _PER_ITEM_SEGMENTS.build_session_segments(
        participant, session,
        tau_siglip=1.1,
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

    flicker_floor = max(min_score, flicker_min_score)
    flicker_hits: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for t in hoi_sorted:
        dino_by_hand = dino_by_t.get(t) or {}
        best_dino: dict[str, float] = {}
        for hs in dino_by_hand:
            for iid, s in (dino_by_hand.get(hs) or {}).items():
                if iid not in inv_iids or s < flicker_floor:
                    continue
                vc = iid_to_vc.get(iid)
                if not vc:
                    continue
                best_dino[vc] = max(best_dino.get(vc, 0.0), s)
        for vc, d in best_dino.items():
            flicker_hits[vc].append((t, d))

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
        outside_hits = [
            h for h in hits
            if not any(s <= h[0] <= e for (s, e) in intervals)
        ]
        if not outside_hits:
            continue
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
            cluster_peak = max(h[1] for h in cluster_hits)
            if cluster_peak < flicker_peak_score:
                continue
            ts_c = [h[0] for h in cluster_hits]
            peak_dino_c = cluster_peak
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
        "seg_tau_dino": seg_tau_dino,
        "seg_gap_close": seg_gap_close,
        "seg_min_duration": seg_min_duration,
    }
    return "\n".join(lines), stats, active_vcs, active_iids


# ---------------------------------------------------------------------------
# Block-merged evidence formatter
#
# Activity-block clustering: take all per-iid HOI-gated DINO segments + flicker
# clusters, build connected components under transitive interval overlap, and
# render each component as ONE block header + indented per-visual_class
# sub-rows sorted by peak DINO desc.
#
# Compared to chrono mode (one row per item × interval), this collapses the
# repeated-row noise that dominates cooking phases: a 30s pasta cook with
# carrots bleeding cross-talk becomes ONE block with two sub-rows, peak DINOs
# side-by-side, instead of 5–10 chronological rows the planner has to manually
# de-duplicate.
#
# This formatter changes the planner's CONTEXT only — the planner still emits
# the same item_decisions schema and the observer is untouched.
# ---------------------------------------------------------------------------

def _build_flicker_clusters_by_vc(
    inventory: list[dict],
    dino_by_t: dict[float, dict[str, dict[str, float]]],
    hoi_sorted: list[float],
    seg_gap_close: float,
    flicker_floor: float,
    flicker_min_hits: int,
    flicker_peak_score: float,
    vc_segment_intervals: dict[str, list[tuple[float, float]]],
) -> dict[str, list[dict]]:
    """Re-derive flicker clusters per visual_class outside existing segments.

    Mirrors `format_chronological_segments_evidence` flicker logic so block
    mode and chrono mode see the same flicker tail under identical params.
    """
    inv_iids = {inv["instance_id"] for inv in inventory}
    iid_to_vc = {inv["instance_id"]: inv["visual_class"] for inv in inventory}
    flicker_hits: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for t in hoi_sorted:
        dino_by_hand = dino_by_t.get(t) or {}
        best_dino: dict[str, float] = {}
        for hs in dino_by_hand:
            for iid, s in (dino_by_hand.get(hs) or {}).items():
                if iid not in inv_iids or s < flicker_floor:
                    continue
                vc = iid_to_vc.get(iid)
                if not vc:
                    continue
                best_dino[vc] = max(best_dino.get(vc, 0.0), s)
        for vc, d in best_dino.items():
            flicker_hits[vc].append((t, d))

    out: dict[str, list[dict]] = {}
    for vc, hits in flicker_hits.items():
        if not hits:
            continue
        intervals = vc_segment_intervals.get(vc, [])
        outside = [h for h in hits if not any(s <= h[0] <= e for (s, e) in intervals)]
        if not outside:
            continue
        outside.sort(key=lambda x: x[0])
        clusters: list[list[tuple[float, float]]] = [[outside[0]]]
        for h in outside[1:]:
            if h[0] - clusters[-1][-1][0] <= seg_gap_close:
                clusters[-1].append(h)
            else:
                clusters.append([h])
        kept: list[dict] = []
        for cluster_hits in clusters:
            if len(cluster_hits) < flicker_min_hits:
                continue
            peak = max(d for _, d in cluster_hits)
            if peak < flicker_peak_score:
                continue
            ts_c = [h[0] for h in cluster_hits]
            kept.append({
                "start": ts_c[0],
                "end": ts_c[-1],
                "peak_dino": peak,
                "n_hits": len(cluster_hits),
            })
        if kept:
            out[vc] = kept
    return out


def _format_subspans(
    subspans: list[tuple[float, float]],
    block_start: float,
    block_end: float,
    max_subspans: int = 3,
) -> str:
    """Render an item's active sub-ranges relative to its parent block.

    Returns 'full' when union ≈ block; otherwise 'a-bs' (relative to block
    start, integer seconds), comma-separated, capped at top-N by duration.
    """
    if not subspans:
        return "—"
    block_dur = max(block_end - block_start, 1e-3)
    # Merge any touching/overlapping subspans first.
    merged = []
    for s, e in sorted(subspans):
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    total = sum(e - s for s, e in merged)
    if total / block_dur >= 0.9 and len(merged) == 1:
        return "full"
    # Sort by duration desc, keep top-N, then re-sort by time for output.
    merged_sorted = sorted(merged, key=lambda r: -(r[1] - r[0]))
    keep = merged_sorted[:max_subspans]
    extra = len(merged_sorted) - len(keep)
    keep.sort(key=lambda r: r[0])
    parts = [
        f"{s - block_start:.0f}-{e - block_start:.0f}s"
        for s, e in keep
    ]
    if extra > 0:
        parts.append(f"+{extra} more")
    return ",".join(parts)


def format_blocks_evidence(
    participant: str,
    session: str,
    inventory: list[dict],
    transparency_by_iid: dict | None,
    min_score: float,
    dino_by_t: dict[float, dict[str, dict[str, float]]],
    hoi_sorted: list[float],
    seg_tau_dino: float,
    seg_gap_close: float,
    seg_min_duration: float,
    inventory_scope: str = "full",
    flicker_min_score: float = 0.15,
    flicker_min_hits: int = 2,
    flicker_peak_score: float = 0.25,
    block_gap_close: float = 0.0,
    max_block_s: float | None = None,
    crosstalk_dino_ratio: float = 0.4,
    crosstalk_cov_floor: float = 0.7,
) -> tuple[str, dict, set[str], set[str]]:
    """Activity-block counterpart to chrono / segments evidence modes.

    Pipeline:
      1. Pull per-iid HOI-gated DINO segments via 05b (tau_siglip=1.1 disables
         SigLIP-based activation).
      2. Compute flicker clusters per vc (same gates as chrono mode).
      3. Build a flat interval list: (start, end, vc, iid_or_None, peak_dino,
         is_flicker, n_hits_or_segframes).
      4. Cluster intervals into activity blocks via transitive overlap (any
         two intervals with `interval_overlap > -block_gap_close` join).
      5. Optionally split blocks longer than `max_block_s` at the largest
         internal gap.
      6. Per block: group by vc → peak_dino, union subspans, iid list. Emit
         a header + indented sub-rows sorted by peak_dino desc.

    Returns (text, stats, active_vcs, active_iids).
    """
    transparency_by_iid = transparency_by_iid or {}

    _, per_item = _PER_ITEM_SEGMENTS.build_session_segments(
        participant, session,
        tau_siglip=1.1,
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

    # ── Flat interval list ────────────────────────────────────────────────
    intervals: list[dict] = []
    vc_segment_intervals: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for iid, (_inv, segs) in per_item.items():
        if iid not in inv_iids:
            continue
        vc = iid_to_vc[iid]
        for s in segs:
            intervals.append({
                "start": s["start"],
                "end": s["end"],
                "vc": vc,
                "iid": iid,
                "peak_dino": s["peak_dino"],
                "is_flicker": False,
                "duration": s["duration"],
            })
            vc_segment_intervals[vc].append((s["start"], s["end"]))

    flicker_by_vc = _build_flicker_clusters_by_vc(
        inventory, dino_by_t, hoi_sorted,
        seg_gap_close=seg_gap_close,
        flicker_floor=max(min_score, flicker_min_score),
        flicker_min_hits=flicker_min_hits,
        flicker_peak_score=flicker_peak_score,
        vc_segment_intervals=vc_segment_intervals,
    )
    for vc, clusters in flicker_by_vc.items():
        for c in clusters:
            intervals.append({
                "start": c["start"],
                "end": c["end"],
                "vc": vc,
                "iid": None,  # flicker is vc-level, not iid-level
                "peak_dino": c["peak_dino"],
                "is_flicker": True,
                "duration": max(c["end"] - c["start"], 0.0),
            })

    if not intervals:
        return "(no detection bursts above thresholds)", {
            "n_hoi_frames_total": len(hoi_sorted),
            "n_items_emitted": 0,
            "n_rows_emitted": 0,
            "n_blocks": 0,
            "min_score": min_score,
            "seg_tau_dino": seg_tau_dino,
        }, set(), set()

    # ── Cluster intervals into blocks (transitive overlap) ─────────────────
    intervals.sort(key=lambda r: (r["start"], r["end"]))
    raw_blocks: list[list[dict]] = []
    cur: list[dict] = []
    cur_max_end = -1.0
    for it in intervals:
        if not cur or it["start"] > cur_max_end + block_gap_close:
            if cur:
                raw_blocks.append(cur)
            cur = [it]
            cur_max_end = it["end"]
        else:
            cur.append(it)
            cur_max_end = max(cur_max_end, it["end"])
    if cur:
        raw_blocks.append(cur)

    # Optional cap on block length: split at largest internal gap.
    blocks: list[list[dict]] = []
    if max_block_s is not None and max_block_s > 0:
        for b in raw_blocks:
            blocks.extend(_split_long_block(b, max_block_s))
    else:
        blocks = raw_blocks

    # ── Render ─────────────────────────────────────────────────────────────
    lines: list[str] = []
    n_rows_emitted = 0
    n_filtered = 0
    active_vcs: set[str] = set()
    active_iids: set[str] = set()

    for b in blocks:
        b_start = min(it["start"] for it in b)
        b_end = max(it["end"] for it in b)
        b_dur = b_end - b_start
        _, scene_dist = _scene_distribution_in_segment(b_start, b_end, owl_points)
        scene_str = scene_dist if scene_dist else "unknown"

        # Group by vc inside the block.
        by_vc: dict[str, dict] = {}
        for it in b:
            vc = it["vc"]
            slot = by_vc.setdefault(vc, {
                "iids": set(),
                "peak_dino": 0.0,
                "subspans": [],
                "is_flicker_only": True,
                "n_hot_frames": 0,
            })
            if it["iid"]:
                slot["iids"].add(it["iid"])
            slot["peak_dino"] = max(slot["peak_dino"], it["peak_dino"])
            slot["subspans"].append((it["start"], it["end"]))
            if not it["is_flicker"]:
                slot["is_flicker_only"] = False
        # Backfill iid list for vcs that contributed only flicker (no iid).
        for vc, slot in by_vc.items():
            if not slot["iids"]:
                slot["iids"] = set(vc_to_iids.get(vc, []))

        # Header. Single-item blocks collapse onto the header line — saves
        # ~half the lines on sessions where most blocks have only one item.
        n_items = len(by_vc)
        if n_items == 1:
            (only_vc,) = by_vc.keys()
            slot = by_vc[only_vc]
            for i in slot["iids"]:
                active_iids.add(i)
            active_vcs.add(only_vc)
            flicker_tag = " [flicker]" if slot["is_flicker_only"] else ""
            lines.append(
                f"[{b_start:6.1f}-{b_end:6.1f}s] dur={b_dur:5.1f}s "
                f"scene={scene_str}  \"{only_vc}\" "
                f"dino={slot['peak_dino']:.2f}{flicker_tag}"
            )
            n_rows_emitted += 1
            lines.append("")
            continue
        # Cross-talk filter: drop sub-rows whose peak DINO is < ratio × the
        # block's max peak DINO AND whose coverage is ≥ floor. The pattern
        # this catches: an item bleeding into the whole block at low DINO
        # while a different item dominates with a high peak. We only filter
        # in multi-item blocks; single-item blocks already collapsed above.
        block_max_dino = max(by_vc[v]["peak_dino"] for v in by_vc)
        kept_vcs: list[str] = []
        n_filtered_in_block = 0
        for vc in sorted(by_vc, key=lambda v: -by_vc[v]["peak_dino"]):
            slot = by_vc[vc]
            cov_frac = sum(e - s for s, e in slot["subspans"]) / max(b_dur, 1e-3)
            ratio = slot["peak_dino"] / max(block_max_dino, 1e-6)
            if (
                vc != max(by_vc, key=lambda v: by_vc[v]["peak_dino"])  # never filter the focus item
                and ratio < crosstalk_dino_ratio
                and cov_frac >= crosstalk_cov_floor
            ):
                n_filtered_in_block += 1
                continue
            kept_vcs.append(vc)
        n_filtered += n_filtered_in_block

        # Header reflects the post-filter item count.
        kept_n = len(kept_vcs)
        if kept_n == 1:
            only_vc = kept_vcs[0]
            slot = by_vc[only_vc]
            for i in slot["iids"]:
                active_iids.add(i)
            active_vcs.add(only_vc)
            flicker_tag = " [flicker]" if slot["is_flicker_only"] else ""
            ct_tag = f" (-{n_filtered_in_block} cross-talk)" if n_filtered_in_block else ""
            lines.append(
                f"[{b_start:6.1f}-{b_end:6.1f}s] dur={b_dur:5.1f}s "
                f"scene={scene_str}  \"{only_vc}\" "
                f"dino={slot['peak_dino']:.2f}{flicker_tag}{ct_tag}"
            )
            n_rows_emitted += 1
            lines.append("")
            continue
        ct_tag = f", -{n_filtered_in_block} cross-talk" if n_filtered_in_block else ""
        lines.append(
            f"[{b_start:6.1f}-{b_end:6.1f}s] dur={b_dur:5.1f}s "
            f"scene={scene_str} ({kept_n} items{ct_tag})"
        )
        # Sub-rows sorted by peak DINO desc. Metadata (transparency, pkg,
        # iid list) lives in the Session Inventory section above — repeating
        # it per block roughly doubles the prompt and adds no signal.
        for vc in kept_vcs:
            slot = by_vc[vc]
            iids_list = sorted(slot["iids"])
            for i in iids_list:
                active_iids.add(i)
            active_vcs.add(vc)

            active_str = _format_subspans(slot["subspans"], b_start, b_end)
            cov = sum(e - s for s, e in slot["subspans"]) / max(b_dur, 1e-3) * 100.0
            flicker_tag = " [flicker]" if slot["is_flicker_only"] else ""

            lines.append(
                f'  "{vc}" dino={slot["peak_dino"]:.2f} '
                f'active={active_str} cov={cov:3.0f}%{flicker_tag}'
            )
            n_rows_emitted += 1
        lines.append("")

    stats = {
        "n_hoi_frames_total": len(hoi_sorted),
        "n_items_emitted": len(active_vcs),
        "n_rows_emitted": n_rows_emitted,
        "n_blocks": len(blocks),
        "n_crosstalk_filtered": n_filtered,
        "min_score": min_score,
        "seg_tau_dino": seg_tau_dino,
        "seg_gap_close": seg_gap_close,
        "seg_min_duration": seg_min_duration,
        "block_gap_close": block_gap_close,
        "max_block_s": max_block_s,
        "crosstalk_dino_ratio": crosstalk_dino_ratio,
        "crosstalk_cov_floor": crosstalk_cov_floor,
    }
    return "\n".join(lines), stats, active_vcs, active_iids


def _vc_transparency_tag(
    vc: str,
    vc_to_iids: dict[str, list[str]],
    inv_by_iid: dict,
    transparency_by_iid: dict,
) -> str:
    iids = vc_to_iids.get(vc, [])
    n_t = 0
    n_o = 0
    for iid in iids:
        if iid in transparency_by_iid:
            if transparency_by_iid[iid]:
                n_t += 1
            else:
                n_o += 1
        else:
            vis = inv_by_iid.get(iid, {}).get("visible_during_interaction", True)
            if vis:
                n_t += 1
            else:
                n_o += 1
    return "transparent" if n_t >= n_o else "opaque"


def _split_long_block(block: list[dict], max_s: float) -> list[list[dict]]:
    """Recursively split a block at its largest internal coverage gap until
    every part is <= max_s. Returns list of sub-blocks, each a non-empty list
    of intervals.
    """
    b_start = min(it["start"] for it in block)
    b_end = max(it["end"] for it in block)
    if b_end - b_start <= max_s or len(block) < 2:
        return [block]

    # Find the largest gap in the block's covered timeline.
    spans = sorted([(it["start"], it["end"]) for it in block])
    merged = [spans[0]]
    for s, e in spans[1:]:
        if s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    if len(merged) < 2:
        # Fully contiguous coverage — split at midpoint.
        mid = (b_start + b_end) / 2.0
    else:
        gaps = [
            (merged[i + 1][0] - merged[i][1], (merged[i][1] + merged[i + 1][0]) / 2.0)
            for i in range(len(merged) - 1)
        ]
        gaps.sort(reverse=True)
        mid = gaps[0][1]

    left = [it for it in block if it["start"] < mid]
    right = [it for it in block if it["start"] >= mid]
    if not left or not right:
        # Couldn't split meaningfully; return as-is.
        return [block]
    return _split_long_block(left, max_s) + _split_long_block(right, max_s)


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
# Step 1: Planner (one-shot, text-only)
#
# Reads the Iterative-style inventory + per-item evidence section but emits
# the noTAD-style `item_decisions` schema — one entry per active iid, with
# observe/no_observation + planner-picked segments.
# ---------------------------------------------------------------------------

PLANNER_PROMPT = """\
You are an expert kitchen activity analyst. You are given an inventory and \
per-item hand-object-interaction (HOI) detection evidence for a single \
egocentric cooking session. Your job is to decide, for each inventory \
iid that has any detection burst in the evidence, whether a vision model \
should observe it to estimate its remaining amount after this session — \
and to pick the short segments to feed that observer.

## Signals
- **dino**: image-to-image similarity between the cropped hand-held object \
and the item's reference product photo. Generally reliable for branded \
packages, but can still fire on visually generic items (e.g. a meat \
reference matching pink/marbled surfaces during cooking).
- A detection row is listed whenever, during a hand-object contact event, \
the DINO score crossed its threshold and (for segment / chrono modes) \
stayed active across a short window.

## Session Inventory
{inventory}

## Per-Item Detections ({evidence_format_note})
{evidence}

## Task

Return **one decision entry for every distinct inventory iid whose \
visual_class appears at least once in the Per-Item Detections section \
above.** Items whose visual_class never appears are NOT in the candidate \
set — omit them.

**Duplicate iids of the same visual_class:** when two or more inventory \
entries share the same `visual_class` (e.g. two purchase instances of \
"Large White Eggs" or "Whole Milk Gallon"), the evidence usually cannot \
distinguish them — two sealed cartons of the same product score \
identically. Do NOT refuse observation of all duplicates. Treat the \
group as one physical package: emit `observe` for exactly ONE \
instance_id (the earliest trailing `YYYYMMDD` — FIFO, oldest carton \
consumed first), and `no_observation` for the others with reasoning \
citing the duplication.

## Step 1 — decide `observe` or `no_observation` for each detected iid

Your default decision is `observe`. Choose `no_observation` only when you \
can articulate a specific reason the evidence does not reflect real use of \
the item.

Walk the iid's bursts in time order:

- Multiple bursts over time, or any burst with sustained high DINO score, \
almost certainly indicate a real interaction — `observe`.
- A single brief burst with modest scores may still be a real quick \
interaction (staples like oil, sauce, eggs are handled briefly). Default \
to `observe` unless you have a concrete reason to reject it.
- `no_observation` is appropriate when the bursts plausibly result from \
visual cross-talk — e.g. a meat-reference image firing weakly on unrelated \
cooking frames, a generic bottle reference matching a different container \
on the counter. Typical pattern: multiple short, low-DINO bursts \
scattered across the cooking phase without clustering, and another \
visual_class with a much stronger overlapping burst.
- Returning `observe` for every detected iid is acceptable if the \
evidence supports it.

Write one sentence of `reasoning` for every decision, explaining what in \
the evidence drove your choice.

## Step 2 — pick observation segments for every `observe` iid

Walk the bursts in time order. Most items follow a retrieval → use → \
put-back arc. Pick short segments that let the observer piece this story \
together:

- Include at least one **early** burst when one exists (retrieval / \
first access) — this often shows the starting package and fill level best.
- Include the **most informative middle burst** (the one with the \
highest sustained DINO score for this item).
- Include a **final** burst if one is present — often the clearest view \
of what remains.
- For items handled only briefly, 1–2 short segments is enough.
- For items with a long cluster (many bursts across minutes), 3–5 short \
segments covering first → middle → last is appropriate.
- Prefer short segments (~2–5s). The observer sees them concatenated, so \
several short clips beat one long clip.

**Frame budget:** the observer has up to {max_frames} frames total per \
call (one call per `observe` iid). Keep segments short.

## Output Format

Return ONLY JSON (no other text):
```json
{{
  "item_decisions": [
    {{
      "item": "<visual_class exactly as in inventory>",
      "instance_id": "<instance_id from inventory>",
      "decision": "observe" | "no_observation",
      "reasoning": "<one sentence: what in the evidence drove this decision>",
      "confidence": "high" | "medium" | "low",
      "segments": [[start1, end1], [start2, end2]]
    }}
  ]
}}
```

Rules:
- You MUST return an entry for every distinct inventory iid whose \
visual_class appears at least once in Per-Item Detections. Omit items whose \
visual_class never appears.
- Use EXACT `visual_class` and `instance_id` strings from the Session \
Inventory section.
- `segments` is REQUIRED when `decision == "observe"` and MUST be omitted \
or empty when `decision == "no_observation"`.
- Multiple short segments (2–5s each) are preferred over one long segment.
- Do NOT bias observation windows toward the last burst only — earlier \
bursts often show the package and starting fill better than cooking-side \
bursts.
- `confidence` reflects how certain you are in the decision itself (not \
in the eventual amount estimate).
"""


# ---------------------------------------------------------------------------
# Sweep-only planner prompt (TEMPORARY): asks ONLY for sparse sweep
# timestamps + per-iid observe decision; no per-item segments. Pairs with
# the multi-item sweep observer below — per-item observer is disabled.
# ---------------------------------------------------------------------------

PLANNER_SWEEP_ONLY_PROMPT = """\
You are an expert kitchen activity analyst. You are given an inventory and \
per-item hand-object-interaction (HOI) detection evidence for a single \
egocentric cooking session. Your job is to:

1. Decide, for each inventory iid that has any detection burst in the \
evidence, whether a vision model should verify it (`observe`) or skip it \
(`no_observation`).
2. Pick TWO complementary kinds of frame samples and, for each, list \
which candidate items the sweep observer should look for in those \
frames. The two kinds are merged (sorted by timestamp) and fed to ONE \
multi-item sweep observer call.

## Signals
- **dino**: image-to-image similarity between the cropped hand-held object \
and the item's reference product photo. Generally reliable for branded \
packages, but can still fire on visually generic items (e.g. a meat \
reference matching pink/marbled surfaces during cooking).
- A detection row is listed whenever, during a hand-object contact event, \
the DINO score crossed its threshold and (for segment / chrono modes) \
stayed active across a short window.

## Session Inventory
{inventory}

## Per-Item Detections ({evidence_format_note})
{evidence}

## Task

### Step 1 — `observe` vs `no_observation` for each detected iid

Return **one decision entry for every distinct inventory iid whose \
visual_class appears at least once in the Per-Item Detections section \
above.** Items whose visual_class never appears are NOT in the candidate \
set — omit them.

**Duplicate iids of the same visual_class:** when two or more inventory \
entries share the same `visual_class`, the evidence usually cannot \
distinguish them. Emit `observe` for exactly ONE instance_id (the \
earliest trailing `YYYYMMDD` — FIFO, oldest carton consumed first), and \
`no_observation` for the others with reasoning citing the duplication.

Default decision is `observe`. Choose `no_observation` only when you can \
articulate a specific reason the evidence does not reflect real use of \
the item (e.g. multiple short, low-DINO bursts plausibly explained by \
visual cross-talk against another visual_class with much stronger \
overlapping bursts).

### Step 2 — pick `journey_samples` AND `dense_windows`

Two complementary sample kinds (BOTH should be emitted for actively-used \
items):

- **`journey_samples`** — sparse INDIVIDUAL timestamps (each yields ~1 \
frame) that span an item's journey through the session: retrieval from \
storage → transit/handling → moment it leaves the original package → \
return / last sighting. These give the observer CONTEXT: where the \
item came from, where it ended up, and whether it was actually used or \
just briefly handled. 3–6 timestamps per item is typical. They cost \
little and let the observer rule out look-alikes and confirm \
not-used-this-session.
- **`dense_windows`** — short continuous bursts ([start, end], typically \
3–10 s each) centred on the moments when the item LEAVES its original \
package (pour / scoop / squeeze / crack / cut). These are the windows \
where the observer reads the **stock container's fill level** \
(pre-dispense view, fill drop during dispense, post-dispense or \
put-back view).

Guidelines:
- Use EXACT `instance_id` strings in `target_items`.
- Each entry's `target_items` lists the candidate items the observer \
should look for in those frames; multi-target entries are encouraged \
when several items share a window.
- Spread `journey_samples` across early / middle / late so the observer \
gets full session context for each item.
- Anchor `dense_windows` on the strongest evidence bursts — high DINO + \
HOI contact. Prefer the moment around dispensing rather than transit.
- For items where the only useful read is the put-back / final shelf \
view (transparent jar, etc.), include that as a dense window.
- An item MAY appear in multiple journey samples and multiple dense \
windows.
- Every item with `decision == "observe"` MUST appear in `target_items` \
of at least one entry (journey OR dense).
- Budgets: at most {journey_budget} `journey_samples` total and at most \
{dense_budget} `dense_windows` total across the session — stay within \
both, but use as many as you need for the items you're observing.
- If every item is `no_observation`, return both lists empty.

## Output Format

Return ONLY JSON (no other text):
```json
{{
  "item_decisions": [
    {{
      "item": "<visual_class exactly as in inventory>",
      "instance_id": "<instance_id from inventory>",
      "decision": "observe" | "no_observation",
      "reasoning": "<one sentence: what in the evidence drove this decision>",
      "confidence": "high" | "medium" | "low"
    }}
  ],
  "journey_samples": [
    {{
      "t": <seconds>,
      "target_items": ["<instance_id1>", "<instance_id2>", ...]
    }}
  ],
  "dense_windows": [
    {{
      "start": <seconds>,
      "end": <seconds>,
      "target_items": ["<instance_id1>", "<instance_id2>", ...]
    }}
  ]
}}
```

Rules:
- One decision entry per distinct inventory iid whose visual_class appears \
in Per-Item Detections. Omit items whose visual_class never appears.
- Use EXACT `visual_class` and `instance_id` strings from the Session \
Inventory section.
- `confidence` reflects how certain you are in the decision itself.
"""


def _is_refusal(text: str) -> bool:
    low = text.strip().lower()
    return low.startswith("i'm sorry") or low.startswith("i cannot") or "cannot assist" in low


def build_planner_prompt(
    participant: str,
    session: str,
    inventory: list[dict],
    transparency_by_iid: dict | None,
    dino_by_t: dict,
    scene_by_t: dict,
    hoi_details_by_t: dict,
    hoi_sorted: list[float],
    evidence_mode: str,
    min_score: float,
    max_frames: int,
    inventory_scope: str,
    seg_tau_dino: float,
    seg_gap_close: float,
    seg_min_duration: float,
    flicker_min_score: float,
    flicker_min_hits: int,
    flicker_peak_score: float,
    block_gap_close: float = 0.0,
    max_block_s: float | None = None,
    crosstalk_dino_ratio: float = 0.4,
    crosstalk_cov_floor: float = 0.7,
) -> tuple[str, dict, set[str], set[str]]:
    if evidence_mode == "blocks":
        evidence_text, ev_stats, active_vcs, active_iids = format_blocks_evidence(
            participant, session, inventory, transparency_by_iid,
            min_score=min_score,
            dino_by_t=dino_by_t,
            hoi_sorted=hoi_sorted,
            seg_tau_dino=seg_tau_dino,
            seg_gap_close=seg_gap_close,
            seg_min_duration=seg_min_duration,
            inventory_scope=inventory_scope,
            flicker_min_score=flicker_min_score,
            flicker_min_hits=flicker_min_hits,
            flicker_peak_score=flicker_peak_score,
            block_gap_close=block_gap_close,
            max_block_s=max_block_s,
            crosstalk_dino_ratio=crosstalk_dino_ratio,
            crosstalk_cov_floor=crosstalk_cov_floor,
        )
        evidence_format_note = (
            "ACTIVITY BLOCKS — segments + flicker clusters merged into "
            "connected components (transitive interval overlap). Each block "
            "is one chronological 'moment' with co-active items grouped. "
            "Single-item blocks collapse to one line: "
            "`[start-end] dur=Xs scene=<dist>  \"<vc>\" dino=<peak>`. "
            "Multi-item blocks emit a header `[start-end] dur=Xs "
            "scene=<dist> (N items)` followed by indented per-visual_class "
            "sub-rows sorted by peak DINO desc: `\"<vc>\" dino=<peak> "
            "active=<offsets> cov=<%>`. `active=full` means the item spans "
            "the whole block; otherwise offsets are SECONDS RELATIVE TO "
            "block start. `cov` is the fraction of the block this item was "
            "active for. A `[flicker]` tag means this item joined the "
            "block via per-frame HOI hits that didn't form a coherent "
            "segment under the morphology thresholds — weaker evidence but "
            "still worth considering. Use `dino` to identify the focus "
            "item in each block; co-active items with much lower DINO + "
            "high coverage are typically visual-similarity bleed. A "
            "pre-filter has already dropped sub-rows whose peak DINO was "
            "<{cdr:.0%} of the block's max AND whose coverage was ≥{ccf:.0%} "
            "(those are flagged in the header as `-N cross-talk`); rows you "
            "see survived that gate. Look up package_type, iids, and unit "
            "in the Session Inventory section."
        ).format(cdr=crosstalk_dino_ratio, ccf=crosstalk_cov_floor)
    elif evidence_mode == "segments":
        evidence_text, ev_stats, active_vcs, active_iids = format_per_item_segments_evidence(
            participant, session, inventory, transparency_by_iid,
            min_score=min_score,
            dino_by_t=dino_by_t,
            hoi_sorted=hoi_sorted,
            seg_tau_dino=seg_tau_dino,
            seg_gap_close=seg_gap_close,
            seg_min_duration=seg_min_duration,
            inventory_scope=inventory_scope,
            flicker_min_score=flicker_min_score,
            flicker_min_hits=flicker_min_hits,
            flicker_peak_score=flicker_peak_score,
        )
        evidence_format_note = (
            "HOI-gated temporal SEGMENTS per visual_class — one row per "
            "coherent on-contact window: [start-end] dur hot/total "
            "dino_peak scene=<top1:n,top2:n,unk:n> iid. "
            "`hot` = frames in the window where hands23 reported hand-on-item "
            "contact AND DINO for this vc/iid crossed its threshold. "
            "A `FLICKER ONLY` row means the item had per-frame HOI hits but "
            "no segment formed under the morphology thresholds — treat as "
            "weaker evidence but still worth observing."
        )
    elif evidence_mode == "chrono":
        evidence_text, ev_stats, active_vcs, active_iids = format_chronological_segments_evidence(
            participant, session, inventory, transparency_by_iid,
            min_score=min_score,
            dino_by_t=dino_by_t,
            hoi_sorted=hoi_sorted,
            seg_tau_dino=seg_tau_dino,
            seg_gap_close=seg_gap_close,
            seg_min_duration=seg_min_duration,
            inventory_scope=inventory_scope,
            flicker_min_score=flicker_min_score,
            flicker_min_hits=flicker_min_hits,
            flicker_peak_score=flicker_peak_score,
        )
        evidence_format_note = (
            "CHRONOLOGICAL timeline of HOI-gated segments — one row per "
            "(item × interval), sorted by start time: "
            "[start-end] dur=<s> dino=<peak> scene=<top1:n,top2:n,unk:n> \"<visual_class>\". "
            "Multiple items active in the same window appear as consecutive "
            "rows — the timeline is NOT de-duplicated, so use DINO score + "
            "scene to judge which item is the actual focus. Look up "
            "package_type and instance_ids in the Session Inventory section "
            "(each visual_class lists its iids and pkg)."
        )
    else:
        evidence_text, ev_stats, active_vcs, active_iids = format_per_item_evidence(
            hoi_sorted, dino_by_t, scene_by_t, hoi_details_by_t,
            inventory,
            min_score=min_score,
            transparency_by_iid=transparency_by_iid,
        )
        evidence_format_note = (
            "grouped by visual_class; chronological per item; HOI contact only"
        )

    inventory_text = format_inventory_for_prompt(inventory, transparency_by_iid)

    prompt = PLANNER_PROMPT.format(
        inventory=inventory_text,
        evidence=evidence_text,
        evidence_format_note=evidence_format_note,
        max_frames=max_frames,
    )
    return prompt, ev_stats, active_vcs, active_iids


def build_planner_prompt_sweep_only(
    participant: str,
    session: str,
    inventory: list[dict],
    transparency_by_iid: dict | None,
    dino_by_t: dict,
    scene_by_t: dict,
    hoi_details_by_t: dict,
    hoi_sorted: list[float],
    evidence_mode: str,
    min_score: float,
    inventory_scope: str,
    seg_tau_dino: float,
    seg_gap_close: float,
    seg_min_duration: float,
    flicker_min_score: float,
    flicker_min_hits: int,
    flicker_peak_score: float,
    block_gap_close: float = 0.0,
    max_block_s: float | None = None,
    crosstalk_dino_ratio: float = 0.4,
    crosstalk_cov_floor: float = 0.7,
    sweep_budget: int = 20,
) -> tuple[str, dict, set[str], set[str]]:
    """Build the sweep-only planner prompt. Same evidence rendering as the
    per-item planner, but the prompt asks only for `item_decisions` + a
    shared `sweep_timestamps` list — no per-item segments.
    """
    if evidence_mode == "blocks":
        evidence_text, ev_stats, active_vcs, active_iids = format_blocks_evidence(
            participant, session, inventory, transparency_by_iid,
            min_score=min_score,
            dino_by_t=dino_by_t,
            hoi_sorted=hoi_sorted,
            seg_tau_dino=seg_tau_dino,
            seg_gap_close=seg_gap_close,
            seg_min_duration=seg_min_duration,
            inventory_scope=inventory_scope,
            flicker_min_score=flicker_min_score,
            flicker_min_hits=flicker_min_hits,
            flicker_peak_score=flicker_peak_score,
            block_gap_close=block_gap_close,
            max_block_s=max_block_s,
            crosstalk_dino_ratio=crosstalk_dino_ratio,
            crosstalk_cov_floor=crosstalk_cov_floor,
        )
        evidence_format_note = (
            "ACTIVITY BLOCKS — segments + flicker clusters merged into "
            "connected components. Multi-item blocks emit a header followed "
            "by indented per-visual_class sub-rows sorted by peak DINO desc. "
            "Use `dino` to identify the focus item; co-active items with "
            "much lower DINO + high coverage are typically visual-similarity "
            "bleed."
        )
    elif evidence_mode == "segments":
        evidence_text, ev_stats, active_vcs, active_iids = format_per_item_segments_evidence(
            participant, session, inventory, transparency_by_iid,
            min_score=min_score,
            dino_by_t=dino_by_t,
            hoi_sorted=hoi_sorted,
            seg_tau_dino=seg_tau_dino,
            seg_gap_close=seg_gap_close,
            seg_min_duration=seg_min_duration,
            inventory_scope=inventory_scope,
            flicker_min_score=flicker_min_score,
            flicker_min_hits=flicker_min_hits,
            flicker_peak_score=flicker_peak_score,
        )
        evidence_format_note = (
            "HOI-gated temporal SEGMENTS per visual_class — one row per "
            "coherent on-contact window."
        )
    elif evidence_mode == "chrono":
        evidence_text, ev_stats, active_vcs, active_iids = format_chronological_segments_evidence(
            participant, session, inventory, transparency_by_iid,
            min_score=min_score,
            dino_by_t=dino_by_t,
            hoi_sorted=hoi_sorted,
            seg_tau_dino=seg_tau_dino,
            seg_gap_close=seg_gap_close,
            seg_min_duration=seg_min_duration,
            inventory_scope=inventory_scope,
            flicker_min_score=flicker_min_score,
            flicker_min_hits=flicker_min_hits,
            flicker_peak_score=flicker_peak_score,
        )
        evidence_format_note = (
            "CHRONOLOGICAL timeline of HOI-gated segments — one row per "
            "(item × interval), sorted by start time."
        )
    else:
        evidence_text, ev_stats, active_vcs, active_iids = format_per_item_evidence(
            hoi_sorted, dino_by_t, scene_by_t, hoi_details_by_t,
            inventory,
            min_score=min_score,
            transparency_by_iid=transparency_by_iid,
        )
        evidence_format_note = (
            "grouped by visual_class; chronological per item; HOI contact only"
        )

    inventory_text = format_inventory_for_prompt(inventory, transparency_by_iid)
    # journey budget is intentionally generous (sparse 1-frame samples are
    # cheap context); dense windows are the costlier kind, so capped tighter.
    journey_budget = max(sweep_budget * 2, sweep_budget + 6)
    dense_budget = sweep_budget
    prompt = PLANNER_SWEEP_ONLY_PROMPT.format(
        inventory=inventory_text,
        evidence=evidence_text,
        evidence_format_note=evidence_format_note,
        journey_budget=journey_budget,
        dense_budget=dense_budget,
    )
    return prompt, ev_stats, active_vcs, active_iids


# ---------------------------------------------------------------------------
# Round-2 planner: gap-fill replan under the new journey/dense + 4-field schema.
# Each candidate item was confirmed USED in R1 but R1 could not produce a
# session-end remaining estimate (no direct amount_remaining and not both
# starting+derivative). The planner reads R1's per-segment observations to
# infer where dispensing actually happened, then proposes new windows that
# fill the missing amount field(s).
# ---------------------------------------------------------------------------

PLANNER_ROUND_2_PROMPT = """\
You are an expert kitchen activity analyst running ROUND 2 of a 2-round \
journey-aware planning pipeline for an egocentric kitchen video.

Each candidate item below was confirmed `status: used` in R1, but R1 \
could not produce a session-end remaining estimate. Either R1 read no \
package fill at all, or it read only the pre-dispense / post-dispense \
side, and the script could not derive `amount_remaining` from \
`amount_starting − amount_derivative`. Your job: pick NEW frame samples \
(journey + dense) that fill the SPECIFIC amount field(s) still missing \
for each item.

## Per-item state (what R1 found, what's still missing)

For each unresolved item we list:
- The R1 amount triple — `amount_starting`, `amount_remaining`, \
`amount_derivative` (any may be `null`).
- An explicit `needs:` line saying WHICH amount field(s) still need to \
be filled to derive remaining.
- The R1 segments whose `target_items` included this item, AND the R1 \
observer's per-segment narration. Use this to infer:
  * WHEN dispensing actually happened for this item (the `[stage: ...]` \
tag in each observation tells you pre / during / post / none).
  * Whether the package was ever visible at all (and why the fill was \
not readable: occluded? back-of-package? motion blur? out of frame?).
  * Which timeline regions are still UNSAMPLED for this item.

{per_item_history}

## R1 full session timeline (avoid overlap)

Below is the FULL R1 segment list (journey + dense, sorted by time) \
with the observer's narration. Your R2 segments MUST NOT overlap any \
R1 segment listed here. Use these notes to re-anchor: if R1 only \
caught the retrieval frame, look LATER (during or after dispense). If \
R1 only caught a portion-on-board view with no package, look EARLIER \
(retrieval) or LATER (put-back).

{r1_session_timeline}

## Per-Item Detections ({evidence_format_note}, unresolved items only)
Re-rendered for the still-unresolved items — this is the FULL evidence \
pool to choose unsampled bursts from.

{evidence}

## Goals per item (what to target)

- If `needs: starting AND remaining` (no fill read at all in R1): emit \
ONE journey sample at the earliest unsampled retrieval moment AND one \
dense window over the dispense action; OR a journey sample at \
retrieval + a journey sample at the put-back / last-sighting frame.
- If `needs: remaining` (R1 has starting only): a dense window \
catching the END of the dispense action, OR a journey sample at \
put-back / final shelf view.
- If `needs: starting` (R1 has derivative only): an EARLIER unsampled \
window where the package is first picked up / pre-dispense.
- If `needs: any fill view` (R1 has only derivative, no package was \
seen): any unsampled visibility burst — earliest preferred.

Spread budget across the items that have the strongest unsampled \
evidence. Items where ALL evidence bursts already overlap an R1 \
segment should get `decision: "no_observation"` with reasoning \
"evidence exhausted".

## Output Format

Return ONLY JSON (no other text):
```json
{{
  "item_decisions": [
    {{
      "item": "<visual_class>",
      "instance_id": "<instance_id>",
      "decision": "observe" | "no_observation",
      "reasoning": "<one sentence: WHY this window / why exhausted; \
reference what R1 saw and what gap you're filling>",
      "confidence": "high" | "medium" | "low"
    }}
  ],
  "journey_samples": [
    {{ "t": <seconds>, "target_items": ["<iid>", ...] }}
  ],
  "dense_windows": [
    {{ "start": <seconds>, "end": <seconds>, "target_items": ["<iid>", ...] }}
  ]
}}
```

Hard rules:
- Each journey sample timestamp and each dense window MUST NOT overlap \
any R1 segment in the timeline above.
- Use EXACT `instance_id` strings.
- Multi-target entries are fine when several items share a moment.
- Budget: at most {journey_budget} journey samples and {dense_budget} \
dense windows total.
- If an item is `no_observation`, omit it from all `target_items` lists.
"""


def _format_r1_session_timeline(r1_segments: list[dict], r1_pso: list[dict]) -> str:
    """Render the R1 unified segment list with per-segment observations."""
    obs_by_idx: dict[int, str] = {}
    for e in r1_pso or []:
        if isinstance(e, dict) and e.get("segment_idx") is not None:
            try:
                obs_by_idx[int(e["segment_idx"])] = str(e.get("observation", ""))
            except (TypeError, ValueError):
                continue
    lines: list[str] = []
    for i, s in enumerate(r1_segments or []):
        kind = s.get("kind", "dense")
        kind_tag = "[journey]" if kind == "journey" else "[dense]"
        if kind == "journey" or s.get("end") == s.get("start"):
            t_str = f"t={s['start']:.1f}s"
        else:
            t_str = f"t={s['start']:.1f}–{s['end']:.1f}s"
        targets = list(s.get("target_items", []))
        obs = obs_by_idx.get(i + 1, "")
        obs_suffix = f"\n      observed: {obs}" if obs else ""
        lines.append(
            f"  - R1 Seg {i+1} {kind_tag}: {t_str}  targets={targets}{obs_suffix}"
        )
    return "\n".join(lines) if lines else "  (no R1 segments)"


def _r1_needs_str(s, r, d) -> str:
    """Classify what R1 still needs to derive a session-end remaining."""
    if r is not None:
        return "(already resolved — should not be in unresolved list)"
    if s is not None and d is not None:
        return "(already resolvable as starting−derivative — should not be in unresolved list)"
    if s is None and d is None:
        return "starting AND remaining (no fill read AT ALL in R1)"
    if s is not None and d is None:
        return "remaining (R1 has starting only — find a POST-dispense window)"
    if s is None and d is not None:
        return ("starting (R1 has derivative only — find an EARLIER pre-dispense window) "
                "OR any fill view (post-dispense / put-back) for direct remaining")
    return "(unspecified gap)"


def _format_per_item_r1_history(
    target_inventory: list[dict],
    r1_items_by_iid: dict[str, dict],
    r1_segments: list[dict],
    r1_pso: list[dict],
) -> str:
    """For each unresolved item, render R1 amount triple + the segments that
    targeted it (with the observer's per-segment notes).
    """
    obs_by_idx: dict[int, str] = {}
    for e in r1_pso or []:
        if isinstance(e, dict) and e.get("segment_idx") is not None:
            try:
                obs_by_idx[int(e["segment_idx"])] = str(e.get("observation", ""))
            except (TypeError, ValueError):
                continue

    sections: list[str] = []
    for inv in target_inventory:
        iid = inv["instance_id"]
        vc = inv.get("visual_class", iid)
        unit = "g" if inv.get("unit") == "g" else "count"
        sw = r1_items_by_iid.get(iid) or {}
        s = sw.get("amount_starting")
        r = sw.get("amount_remaining")
        d = sw.get("amount_derivative")
        needs = _r1_needs_str(s, r, d)
        seg_lines: list[str] = []
        for i, seg in enumerate(r1_segments or []):
            tgts = seg.get("target_items", []) or []
            if iid not in tgts:
                continue
            kind = seg.get("kind", "dense")
            kind_tag = "[journey]" if kind == "journey" else "[dense]"
            if kind == "journey" or seg.get("end") == seg.get("start"):
                t_str = f"t={seg['start']:.1f}s"
            else:
                t_str = f"t={seg['start']:.1f}–{seg['end']:.1f}s"
            obs = obs_by_idx.get(i + 1, "")
            obs_suffix = f"  observed: {obs}" if obs else ""
            seg_lines.append(f"      R1 Seg {i+1} {kind_tag}: {t_str}{obs_suffix}")
        seg_block = "\n".join(seg_lines) if seg_lines else "      (no R1 segment targeted this item)"
        ev = sw.get("evidence_frames", []) or []
        ev_str = ", ".join(f"{float(t):.1f}s" for t in ev) if ev else "(none)"
        reasoning = (sw.get("reasoning") or "").strip()
        sections.append(
            f'  - iid=`{iid}` "{vc}" ({unit}):\n'
            f"      R1 amounts: starting={s}  remaining={r}  derivative={d}\n"
            f"      needs: {needs}\n"
            f"      R1 evidence frames: {ev_str}\n"
            f"      R1 observer reasoning: {reasoning[:240]}\n"
            f"      R1 segments targeting this item:\n"
            f"{seg_block}"
        )
    return "\n\n".join(sections) if sections else "  (no unresolved items)"


def build_planner_prompt_round_2(
    participant: str,
    session: str,
    target_inventory: list[dict],
    r1_items_by_iid: dict[str, dict],
    r1_segments: list[dict],
    r1_per_segment_observations: list[dict],
    transparency_by_iid: dict | None,
    dino_by_t: dict,
    scene_by_t: dict,
    hoi_details_by_t: dict,
    hoi_sorted: list[float],
    evidence_mode: str,
    min_score: float,
    inventory_scope: str,
    seg_tau_dino: float,
    seg_gap_close: float,
    seg_min_duration: float,
    flicker_min_score: float,
    flicker_min_hits: int,
    flicker_peak_score: float,
    block_gap_close: float = 0.0,
    max_block_s: float | None = None,
    crosstalk_dino_ratio: float = 0.4,
    crosstalk_cov_floor: float = 0.7,
    sweep_budget: int = 8,
) -> tuple[str, dict, set[str], set[str]]:
    """Build the R2 gap-fill planner prompt for items R1 left unresolved.

    target_inventory:           inventory entries for the unresolved iids.
    r1_items_by_iid:            R1 sweep items (the 4-field schema dicts).
    r1_segments:                R1 unified segment list (interleaved
                                 journey + dense, sorted by start time).
    r1_per_segment_observations: R1 observer's per-segment narration
                                 [{segment_idx, observation}, ...].
    """
    if evidence_mode == "blocks":
        evidence_text, ev_stats, active_vcs, active_iids = format_blocks_evidence(
            participant, session, target_inventory, transparency_by_iid,
            min_score=min_score, dino_by_t=dino_by_t, hoi_sorted=hoi_sorted,
            seg_tau_dino=seg_tau_dino, seg_gap_close=seg_gap_close,
            seg_min_duration=seg_min_duration, inventory_scope=inventory_scope,
            flicker_min_score=flicker_min_score, flicker_min_hits=flicker_min_hits,
            flicker_peak_score=flicker_peak_score, block_gap_close=block_gap_close,
            max_block_s=max_block_s, crosstalk_dino_ratio=crosstalk_dino_ratio,
            crosstalk_cov_floor=crosstalk_cov_floor,
        )
        evidence_format_note = "ACTIVITY BLOCKS (filtered to round-N items)"
    elif evidence_mode == "segments":
        evidence_text, ev_stats, active_vcs, active_iids = format_per_item_segments_evidence(
            participant, session, target_inventory, transparency_by_iid,
            min_score=min_score, dino_by_t=dino_by_t, hoi_sorted=hoi_sorted,
            seg_tau_dino=seg_tau_dino, seg_gap_close=seg_gap_close,
            seg_min_duration=seg_min_duration, inventory_scope=inventory_scope,
            flicker_min_score=flicker_min_score, flicker_min_hits=flicker_min_hits,
            flicker_peak_score=flicker_peak_score,
        )
        evidence_format_note = "HOI-gated SEGMENTS per visual_class"
    elif evidence_mode == "chrono":
        evidence_text, ev_stats, active_vcs, active_iids = format_chronological_segments_evidence(
            participant, session, target_inventory, transparency_by_iid,
            min_score=min_score, dino_by_t=dino_by_t, hoi_sorted=hoi_sorted,
            seg_tau_dino=seg_tau_dino, seg_gap_close=seg_gap_close,
            seg_min_duration=seg_min_duration, inventory_scope=inventory_scope,
            flicker_min_score=flicker_min_score, flicker_min_hits=flicker_min_hits,
            flicker_peak_score=flicker_peak_score,
        )
        evidence_format_note = "CHRONOLOGICAL timeline of HOI-gated segments"
    else:
        evidence_text, ev_stats, active_vcs, active_iids = format_per_item_evidence(
            hoi_sorted, dino_by_t, scene_by_t, hoi_details_by_t,
            target_inventory, min_score=min_score,
            transparency_by_iid=transparency_by_iid,
        )
        evidence_format_note = "per-frame HOI grouped by visual_class"

    per_item_history = _format_per_item_r1_history(
        target_inventory, r1_items_by_iid, r1_segments, r1_per_segment_observations,
    )
    r1_session_timeline = _format_r1_session_timeline(
        r1_segments, r1_per_segment_observations,
    )
    journey_budget = max(sweep_budget * 2, sweep_budget + 6)
    dense_budget = sweep_budget

    prompt = PLANNER_ROUND_2_PROMPT.format(
        per_item_history=per_item_history,
        r1_session_timeline=r1_session_timeline,
        evidence=evidence_text,
        evidence_format_note=evidence_format_note,
        journey_budget=journey_budget,
        dense_budget=dense_budget,
    )
    return prompt, ev_stats, active_vcs, active_iids


def run_planner(
    client,
    prompt: str,
    model: str,
    max_retries: int = 5,
    prompt_save_path: Path | None = None,
) -> tuple[str, dict]:
    if prompt_save_path is not None:
        prompt_save_path.parent.mkdir(parents=True, exist_ok=True)
        prompt_save_path.write_text(prompt)

    for attempt in range(max_retries):
        t0 = time.time()
        try:
            response = client.responses.create(
                model=model,
                input=[{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
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
            response_text and "item_decisions" not in response_text
        ):
            print(f"  Planner refusal/truncated (attempt {attempt + 1}), retrying...")
            if attempt < max_retries - 1:
                time.sleep(5 * (attempt + 1))
                continue

        return response_text, stats

    return "", {"error": "max retries exceeded"}


def parse_planner_response(response_text: str) -> tuple[list[dict], list[dict]]:
    """Parse planner JSON. Returns (item_decisions, observation_plan).

    observation_plan is the subset with decision=="observe" and non-empty
    segments, shaped for the observer loop.
    """
    parsed = None
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response_text, re.DOTALL)
    if fence:
        try:
            parsed = json.loads(fence.group(1))
        except json.JSONDecodeError:
            parsed = None

    if parsed is None:
        obj_match = re.search(r"\{.*\"item_decisions\".*\}", response_text, re.DOTALL)
        if obj_match:
            try:
                parsed = json.loads(obj_match.group())
            except json.JSONDecodeError:
                parsed = None

    if not isinstance(parsed, dict):
        return [], []

    decisions = parsed.get("item_decisions") or []
    if not isinstance(decisions, list):
        return [], []

    observation_plan: list[dict] = []
    for d in decisions:
        if not isinstance(d, dict):
            continue
        if d.get("decision") != "observe":
            continue
        segs_raw = d.get("segments") or []
        segs: list[list[float]] = []
        for seg in segs_raw:
            if not isinstance(seg, (list, tuple)) or len(seg) < 2:
                continue
            try:
                s, e = float(seg[0]), float(seg[1])
            except (TypeError, ValueError):
                continue
            if e <= s:
                continue
            segs.append([s, e])
        if not segs:
            continue
        observation_plan.append({
            "item": d.get("item"),
            "instance_id": d.get("instance_id"),
            "segments": segs,
            "confidence": d.get("confidence"),
            "reasoning": d.get("reasoning", ""),
        })
    return decisions, observation_plan


def parse_planner_sweep_response(
    response_text: str,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Parse sweep-only planner JSON.

    Returns (item_decisions, journey_samples, dense_windows).
    - item_decisions: one entry per detected iid with decision /
      reasoning / confidence.
    - journey_samples: list of {t, target_items} sparse single-frame
      timestamps (item-journey context).
    - dense_windows: list of {start, end, target_items} continuous windows
      (fill-level reads).

    Backward compat: if the planner emits the legacy `sweep_segments`
    list, those entries are interpreted as `dense_windows` so older
    cached responses still work.
    """
    parsed = None
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response_text, re.DOTALL)
    if fence:
        try:
            parsed = json.loads(fence.group(1))
        except json.JSONDecodeError:
            parsed = None

    if parsed is None:
        obj_match = re.search(r"\{.*\"item_decisions\".*\}", response_text, re.DOTALL)
        if obj_match:
            try:
                parsed = json.loads(obj_match.group())
            except json.JSONDecodeError:
                parsed = None

    if not isinstance(parsed, dict):
        return [], [], []

    decisions_raw = parsed.get("item_decisions") or []
    decisions: list[dict] = []
    if isinstance(decisions_raw, list):
        for d in decisions_raw:
            if not isinstance(d, dict):
                continue
            decisions.append({
                "item": d.get("item"),
                "instance_id": d.get("instance_id"),
                "decision": d.get("decision"),
                "reasoning": d.get("reasoning", ""),
                "confidence": d.get("confidence"),
            })

    def _clean_targets(raw) -> list[str]:
        if not isinstance(raw, list):
            return []
        return [str(x) for x in raw if isinstance(x, (str, int))]

    journey_samples: list[dict] = []
    for js in parsed.get("journey_samples") or []:
        if not isinstance(js, dict):
            continue
        try:
            t = float(js.get("t"))
        except (TypeError, ValueError):
            continue
        if t < 0:
            continue
        targets = _clean_targets(js.get("target_items"))
        if not targets:
            continue
        journey_samples.append({"t": round(t, 2), "target_items": targets})
    journey_samples.sort(key=lambda s: s["t"])

    dense_windows: list[dict] = []
    for dw in parsed.get("dense_windows") or []:
        if not isinstance(dw, dict):
            continue
        try:
            start = float(dw.get("start"))
            end = float(dw.get("end"))
        except (TypeError, ValueError):
            continue
        if not (end > start >= 0):
            continue
        targets = _clean_targets(dw.get("target_items"))
        if not targets:
            continue
        dense_windows.append({
            "start": round(start, 2),
            "end": round(end, 2),
            "target_items": targets,
        })

    # Legacy fallback: old planner caches still emit `sweep_segments`.
    if not journey_samples and not dense_windows:
        for seg in parsed.get("sweep_segments") or []:
            if not isinstance(seg, dict):
                continue
            try:
                start = float(seg.get("start"))
                end = float(seg.get("end"))
            except (TypeError, ValueError):
                continue
            if not (end > start >= 0):
                continue
            targets = _clean_targets(seg.get("target_items"))
            if not targets:
                continue
            dense_windows.append({
                "start": round(start, 2),
                "end": round(end, 2),
                "target_items": targets,
            })

    dense_windows.sort(key=lambda s: s["start"])
    return decisions, journey_samples, dense_windows


# ---------------------------------------------------------------------------
# Step 2: Observer (noTAD-style, single-item remaining)
# ---------------------------------------------------------------------------

OBSERVER_PROMPT = """\
You are analyzing frames from an egocentric kitchen video recorded with smart glasses.

These {n_frames} frames are extracted from **multiple separate time segments** within \
a cooking session — they are NOT continuous footage. Each frame is labeled with its \
session timestamp. There may be time gaps between frames.

Frame timestamps: {frame_timestamps}

## Target Item
You are looking for: "{item_name}"
- Unit: {unit_label}
{item_context_line}

## Context from action detection
An action detector identified these activities involving this item:
{segment_descriptions}

## Task
1. Confirm whether "{item_name}" is visible in the frames.
2. Track the item across ALL frames (not just the latest). Distinguish between:
   - The **stock container/package** (carton, bottle, bag, block, tub, etc.) \
that holds the remaining inventory of this item.
   - **Portions that have been taken out** for use in this session (e.g., eggs \
cracked into a bowl, cheese grated onto a plate, oil poured into a pan, \
vegetables chopped onto a cutting board).
   Portions taken out are NOT remaining — they have already been consumed or \
are about to be consumed in this session.
3. Estimate the **remaining amount** = amount still in the stock \
container/package at the end of the session, available for future use.
   - Use the latest frame in which the stock container is visible to read its \
fill level (carton count, bottle fill line, bag fullness, block size).
   - If the stock container is no longer visible in the final frames, carry \
forward its last observed fill.
   - Do NOT add loose portions on plates/bowls/pans to the remaining amount; \
those are used, not remaining.
4. Cite which frames support your estimate (prefer frames showing the stock \
container).

Think step by step:
- Identify the stock container/package across the frames. When is it last visible?
- What portion appears to remain inside the stock container in its last visible frame?
- Are there any portions already taken out onto plates/bowls/pans/cutting \
boards? Note them as used, not remaining.
- If the container disappears from view before the end of the session, your \
estimate is the last observed fill of the container — not what sits on the \
counter afterward.

Output ONLY JSON:
```json
{{
  "item_confirmed": true or false,
  "reasoning": "<your step-by-step reasoning>",
  "evidence_frames": [<list of timestamp values of key frames supporting your estimate>],
  "amount_remaining": <number or null if not confirmed>
}}
```"""


def extract_segments_frames(
    segments: list[list[float]],
    video_durations: list[tuple[Path, float]],
    padding: float = 2.0,
    fps: float = 1.0,
    max_frames: int = MAX_IMAGES,
) -> tuple[list[str], list[float]]:
    return _extract_segments_frames(
        [(s[0], s[1]) for s in segments],
        video_durations,
        padding=padding,
        max_frames=max_frames,
        target_fps=fps,
    )


def _build_observer_prompt(
    n_frames: int,
    timestamps: list[float],
    item_info: dict,
    plan_entry: dict,
) -> str:
    unit_label = "grams" if item_info["unit"] == "g" else "count"

    seg_descs = []
    for s in plan_entry.get("segments", []):
        seg_descs.append(f"- {s[0]:.1f}–{s[1]:.1f}s")
    if plan_entry.get("reasoning"):
        seg_descs.append(f"Planner note: {plan_entry['reasoning']}")

    item_context = f"- Package capacity: {item_info['package_amount']}"
    frame_ts_str = ", ".join(f"{t:.1f}s" for t in timestamps)

    return OBSERVER_PROMPT.format(
        n_frames=n_frames,
        frame_timestamps=frame_ts_str,
        item_name=item_info["visual_class"],
        unit_label=unit_label,
        item_context_line=item_context,
        segment_descriptions="\n".join(seg_descs),
    )


def run_observer(
    client,
    frames_b64: list[str],
    timestamps: list[float],
    item_info: dict,
    plan_entry: dict,
    model: str,
    prompt_save_path: Path | None = None,
) -> tuple[str, str, dict]:
    prompt = _build_observer_prompt(len(frames_b64), timestamps, item_info, plan_entry)
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
            print(f"  Observer ERROR ({item_info['visual_class']}): {e}")
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
    item_info: dict,
    plan_entry: dict,
    qwen_url: str = QWEN_URL,
    qwen_model: str = QWEN_MODEL_DEFAULT,
    prompt_save_path: Path | None = None,
) -> tuple[str, str, dict]:
    import requests

    prompt = _build_observer_prompt(len(frames_b64), timestamps, item_info, plan_entry)
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


def parse_observer_response(response_text: str) -> dict:
    result = {
        "item_confirmed": False, "reasoning": "",
        "evidence_frames": [], "amount_remaining": None,
    }
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response_text, re.DOTALL)
    text_to_parse = fence.group(1) if fence else response_text
    match = re.search(r"\{[^{}]*\}", text_to_parse, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            result["item_confirmed"] = parsed.get("item_confirmed", False)
            result["reasoning"] = parsed.get("reasoning", "")
            result["evidence_frames"] = parsed.get("evidence_frames", [])
            result["amount_remaining"] = parsed.get("amount_remaining")
        except json.JSONDecodeError:
            pass
    return result


# ---------------------------------------------------------------------------
# Step 2b: Sweep observer (multi-item pre-filter on sparse timestamps)
# ---------------------------------------------------------------------------

SWEEP_PROMPT = """\
You are analyzing frames from an egocentric kitchen video recorded with \
smart glasses.

These {n_frames} frames come from {n_segments} short SEGMENTS sampled \
across a single cooking session — they are NOT continuous footage. \
Frames were sampled in two complementary ways and are intermixed below \
in chronological order:

- **Journey samples** — sparse INDIVIDUAL frames spanning each item's \
journey through the session (retrieval from storage, transit, leaving \
the original package, return / last sighting). They give CONTEXT: \
where the item came from, where it ended up, and whether it was \
actually used.
- **Dense windows** — short continuous bursts of frames centred on \
moments when an item left its original package. Read the **stock \
container's fill level** here (pre-dispense view, fill drop during \
dispense, post-dispense or put-back view).

Each segment in the list below is tagged `[journey]` or `[dense]` and \
lists the candidate items the planner pre-associated with it. Look at \
that segment's target items first; you MAY use any of the {n_frames} \
frames if it helps.

## Sweep Segments
{segment_block}

## Candidate Items
{candidate_block}

## Per-Item Decision

For each candidate item, fill the four fields:

- **`status`** — `"used"` if the item was actually consumed in this \
session (any visible dispensing, portion taken out, or use of the \
contents); `"not_used"` if the item is absent OR is only visible \
being moved / put away without any dispensing this session.
- **`amount_starting`** — fill level of the stock container BEFORE \
dispensing began (pre-dispense / retrieval view). Number if any frame \
gives a readable starting fill; otherwise `null`.
- **`amount_remaining`** — fill level of the stock container AFTER \
dispensing (post-dispense view, put-back / storage-return frame, or \
the last sighting if the package disappears late). Number if readable \
in any frame; otherwise `null`. **This is the ground-truth target — \
fill it whenever you can.**
- **`amount_derivative`** — how much was TAKEN OUT / consumed \
(visible portion on a plate / bowl / pan / cutting board, or amount \
poured / scooped / squeezed during dispense). Number in the same unit \
as the others; `null` if not estimable.

You are not forced to pick one — fill any combination of the three \
amount fields that the frames support. They are independent \
measurements. If only one is readable, fill that one and leave the \
others `null`. If `status == "not_used"`, all three amounts should \
typically be `null` (or just `amount_remaining` reflecting an \
unchanged package fill — your call).

"Remaining" / "starting" reads count only what is still inside the \
stock container; loose portions on plates / bowls / pans / cutting \
boards count toward `amount_derivative`, not the others.

## Per-Segment Narrative

In addition to the per-item decisions, write ONE short observation per \
segment in `per_segment_observations`. Use verbs from the action \
vocabulary (retrieval, access / open package, dispensing / pour / \
scoop / squeeze, visible-portion-on-board, restocking / put-away, \
idle), name the candidate item(s) involved, and indicate the dispense \
stage (pre / during / post / none).

## Output Format

Return ONLY JSON (no other text):
```json
{{
  "items": [
    {{
      "instance_id": "<exact instance_id from candidate list>",
      "visual_class": "<visual_class>",
      "status": "used" | "not_used",
      "amount_starting": <number or null>,
      "amount_remaining": <number or null>,
      "amount_derivative": <number or null>,
      "evidence_frames": [<timestamp values of key frames>],
      "reasoning": "<one short sentence: which frame(s) gave each amount, and why you assigned this status>"
    }}
  ],
  "per_segment_observations": [
    {{
      "segment_idx": <1-based segment number from the Sweep Segments list>,
      "observation": "<one sentence: what is happening in this segment, which item, which stage>"
    }}
  ]
}}
```

Rules:
- Emit one entry per candidate item. Do NOT skip any.
- Each amount field is independent: fill any combination of starting / \
remaining / derivative. Use `null` for ones the frames do not support.
- `evidence_frames` should cite at least one frame when any amount is \
filled or when `status == "used"`; can be empty for `status == "not_used"`.
- `per_segment_observations` MUST include one entry per segment listed \
above (use `segment_idx` 1..N matching the list). If nothing \
meaningful happens in a segment, say so briefly.
"""


def _build_sweep_prompt(
    n_frames: int,
    timestamps: list[float],
    candidates: list[dict],
    sweep_segments: list[dict],
    frame_seg_idx: list[int],
) -> str:
    iid_to_vc = {c["instance_id"]: c["visual_class"] for c in candidates}

    # Per-segment timestamp ranges of the frames actually extracted from
    # this segment. Surfacing TIMESTAMPS (not 1-based indices) keeps a
    # single identifier in play across the prompt + per-frame labels +
    # `evidence_frames` field — gemini-2.5-pro garbles floats when forced
    # to translate "frame 13" → "172.3s" inside its JSON output.
    seg_ts: dict[int, list[float]] = {i: [] for i in range(len(sweep_segments))}
    for fi, si in enumerate(frame_seg_idx):
        if 0 <= si < len(sweep_segments) and fi < len(timestamps):
            seg_ts[si].append(float(timestamps[fi]))

    seg_lines = []
    iids_in_segments: dict[str, list[int]] = {}
    for si, seg in enumerate(sweep_segments):
        ts_list = seg_ts.get(si, [])
        if ts_list:
            t_lo, t_hi = ts_list[0], ts_list[-1]
            frame_str = (
                f"frames at t∈[{t_lo:.1f}–{t_hi:.1f}s]"
                if t_lo != t_hi else f"frame at t={t_lo:.1f}s"
            )
        else:
            frame_str = "frames (none extracted)"
        target_iids = seg.get("target_items", [])
        target_vcs = []
        for iid in target_iids:
            vc = iid_to_vc.get(iid)
            if vc:
                target_vcs.append(vc)
                iids_in_segments.setdefault(iid, []).append(si + 1)
        target_str = ", ".join(target_vcs) if target_vcs else "(no candidates)"
        kind = seg.get("kind", "dense")
        kind_tag = "[journey]" if kind == "journey" else "[dense]"
        if kind == "journey" or seg.get("end") == seg.get("start"):
            t_str = f"t={seg['start']:.1f}s"
        else:
            t_str = f"t={seg['start']:.1f}–{seg['end']:.1f}s"
        seg_lines.append(
            f"- Segment {si+1} {kind_tag}: window {t_str}, "
            f"{frame_str}\n    target items: {target_str}"
        )
    segment_block = "\n".join(seg_lines) if seg_lines else "(no segments)"

    cand_lines = []
    for c in candidates:
        unit = "grams" if c["unit"] == "g" else "count"
        segs_for_c = iids_in_segments.get(c["instance_id"], [])
        seg_str = (
            ", ".join(f"{i}" for i in segs_for_c) if segs_for_c else "(none)"
        )
        cand_lines.append(
            f"- instance_id=`{c['instance_id']}`  "
            f"visual_class=\"{c['visual_class']}\"  "
            f"unit={unit}  package={c.get('package_amount', '')}\n"
            f"    segments: {seg_str}"
        )
    candidate_block = "\n".join(cand_lines) if cand_lines else "(no candidates)"

    return SWEEP_PROMPT.format(
        n_frames=n_frames,
        n_segments=len(sweep_segments),
        segment_block=segment_block,
        candidate_block=candidate_block,
    )


def _extract_sweep_journey_dense_frames(
    journey_samples: list[dict],
    dense_windows: list[dict],
    video_durations: list[tuple[Path, float]],
    fps: float,
    max_frames: int,
    sample_half_width: float = 0.5,
) -> tuple[list[str], list[float], list[int], list[str]]:
    """Extract frames from journey samples + dense windows.

    Returns (frames_b64, timestamps, seg_idx, seg_kinds) where:
    - `seg_kinds[i]` is "journey" or "dense" for the i-th unified segment.
    - `seg_idx[j]` indexes into seg_kinds for frame j.
    Frames are returned in chronological order. The unified segment list
    interleaves journey + dense entries, sorted by start time, so the
    observer prompt can render them as one numbered list.
    """
    if not journey_samples and not dense_windows:
        return [], [], [], []

    n_journey = len(journey_samples)
    n_dense = len(dense_windows)
    # Allocate the budget: journey samples are cheap (~1 frame each); dense
    # windows get the remainder, proportional to duration.
    journey_cap_each = 1
    journey_total = min(max_frames, n_journey * journey_cap_each)
    dense_budget = max(0, max_frames - journey_total)

    if dense_budget > 0 and n_dense > 0:
        durations = [max(0.0, dw["end"] - dw["start"]) for dw in dense_windows]
        total_dur = sum(durations) or 1.0
        dense_caps = [max(1, round(dense_budget * d / total_dur)) for d in durations]
        while sum(dense_caps) > dense_budget:
            i = max(range(len(dense_caps)), key=lambda k: dense_caps[k])
            if dense_caps[i] <= 1:
                break
            dense_caps[i] -= 1
    else:
        dense_caps = [0] * n_dense

    # Build the unified segment list (sorted by start time), keeping kind +
    # original entry available so we can render and extract per-entry.
    unified: list[tuple[float, str, int, dict]] = []
    for ji, js in enumerate(journey_samples):
        unified.append((float(js["t"]), "journey", ji, js))
    for di, dw in enumerate(dense_windows):
        unified.append((float(dw["start"]), "dense", di, dw))
    unified.sort(key=lambda x: x[0])

    frames_b64: list[str] = []
    timestamps: list[float] = []
    seg_idx: list[int] = []
    seg_kinds: list[str] = []

    for unified_idx, (_t0, kind, orig_i, entry) in enumerate(unified):
        seg_kinds.append(kind)
        if kind == "journey":
            t = float(entry["t"])
            f, ts = _extract_segments_frames(
                [(max(0.0, t - sample_half_width), t + sample_half_width)],
                video_durations,
                padding=0.0, max_frames=1, target_fps=fps,
            )
        else:  # dense
            cap = dense_caps[orig_i]
            if cap <= 0:
                continue
            f, ts = _extract_segments_frames(
                [(entry["start"], entry["end"])], video_durations,
                padding=0.5, max_frames=cap, target_fps=fps,
            )
        for fb64, t in zip(f, ts):
            frames_b64.append(fb64)
            timestamps.append(t)
            seg_idx.append(unified_idx)

    if len(frames_b64) > max_frames:
        # Down-trim to budget while preserving chronological order and
        # keeping at least the first frame of each unified entry.
        keep = sorted(range(len(timestamps)), key=lambda k: timestamps[k])[:max_frames]
        keep_set = set(keep)
        frames_b64 = [frames_b64[k] for k in range(len(frames_b64)) if k in keep_set]
        timestamps = [timestamps[k] for k in range(len(timestamps)) if k in keep_set]
        seg_idx = [seg_idx[k] for k in range(len(seg_idx)) if k in keep_set]

    order = sorted(range(len(timestamps)), key=lambda i: timestamps[i])
    return (
        [frames_b64[i] for i in order],
        [timestamps[i] for i in order],
        [seg_idx[i] for i in order],
        seg_kinds,
    )


def run_sweep_observer(
    client,
    frames_b64: list[str],
    timestamps: list[float],
    candidates: list[dict],
    sweep_segments: list[dict],
    frame_seg_idx: list[int],
    model: str,
    prompt_save_path: Path | None = None,
) -> tuple[str, str, dict]:
    prompt = _build_sweep_prompt(
        len(frames_b64), timestamps, candidates,
        sweep_segments=sweep_segments,
        frame_seg_idx=frame_seg_idx,
    )
    if prompt_save_path is not None:
        prompt_save_path.parent.mkdir(parents=True, exist_ok=True)
        prompt_save_path.write_text(prompt)

    max_retries = 5
    for attempt in range(max_retries):
        t0 = time.time()
        try:
            response_text, in_tok, out_tok = _observer_api_call(
                client, model, prompt, frames_b64, timestamps,
            )
            stats = {
                "inference_time_s": round(time.time() - t0, 2),
                "model": model,
                "num_frames": len(frames_b64),
                "num_candidates": len(candidates),
                "input_tokens": in_tok,
                "output_tokens": out_tok,
                "attempt": attempt + 1,
            }
        except Exception as e:
            err_str = str(e)
            transient = any(m in err_str for m in (
                "503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED",
                "500", "INTERNAL", "deadline", "timeout", "connection", "Connection",
            ))
            if "content_policy_violation" in err_str and attempt < max_retries - 1:
                print(f" sweep content filter (attempt {attempt + 1}), retrying...", end="", flush=True)
                time.sleep(5 * (attempt + 1))
                continue
            if transient and attempt < max_retries - 1:
                wait = 4 * (2 ** attempt)
                print(f" sweep transient API error (attempt {attempt + 1}), retrying in {wait}s...", end="", flush=True)
                time.sleep(wait)
                continue
            print(f"  Sweep ERROR: {e}")
            return "", prompt, {"error": err_str, "inference_time_s": round(time.time() - t0, 2)}

        if _is_refusal(response_text) and attempt < max_retries - 1:
            print(f" sweep refusal (attempt {attempt + 1}), retrying...", end="", flush=True)
            time.sleep(5)
            continue

        return response_text, prompt, stats

    return "", prompt, {"error": "max retries exceeded"}


def parse_sweep_response(response_text: str) -> tuple[list[dict], list[dict]]:
    """Parse sweep JSON. Returns (items, per_segment_observations).

    `items` is the per-item list; `per_segment_observations` is the
    parallel list (entries `{segment_idx, observation}`). Either may be
    empty if the observer omits the field.
    """
    parsed = None
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response_text, re.DOTALL)
    if fence:
        try:
            parsed = json.loads(fence.group(1))
        except json.JSONDecodeError:
            parsed = None

    if parsed is None:
        obj_match = re.search(r"\{.*\"items\".*\}", response_text, re.DOTALL)
        if obj_match:
            try:
                parsed = json.loads(obj_match.group())
            except json.JSONDecodeError:
                parsed = None

    if not isinstance(parsed, dict):
        return [], []

    items = parsed.get("items") or []
    if not isinstance(items, list):
        return [], []

    def _to_float(v):
        try:
            return float(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    out: list[dict] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        status = it.get("status")
        if status not in {"used", "not_used"}:
            continue
        out.append({
            "instance_id": it.get("instance_id", ""),
            "visual_class": it.get("visual_class", ""),
            "status": status,
            "amount_starting": _to_float(it.get("amount_starting")),
            "amount_remaining": _to_float(it.get("amount_remaining")),
            "amount_derivative": _to_float(it.get("amount_derivative")),
            "evidence_frames": it.get("evidence_frames", []),
            "reasoning": it.get("reasoning", ""),
        })

    pso_raw = parsed.get("per_segment_observations") or []
    per_seg: list[dict] = []
    if isinstance(pso_raw, list):
        for entry in pso_raw:
            if not isinstance(entry, dict):
                continue
            try:
                idx = int(entry.get("segment_idx"))
            except (TypeError, ValueError):
                continue
            obs = str(entry.get("observation") or "").strip()
            if not obs:
                continue
            per_seg.append({"segment_idx": idx, "observation": obs})
    per_seg.sort(key=lambda e: e["segment_idx"])
    return out, per_seg


# ---------------------------------------------------------------------------
# Round-2 sweep observer (gap-fill verification under the new 4-field schema).
# Each candidate was confirmed `status: used` in R1 but R1 left at least one
# amount field null. The observer's job is to fill the missing field(s)
# from the new frames.
# ---------------------------------------------------------------------------

SWEEP_ROUND_2_PROMPT = """\
You are running ROUND 2 of a 2-round sweep observer pass on an \
egocentric kitchen video.

These {n_frames} frames come from {n_segments} new SEGMENTS the R2 \
planner picked specifically to fill amount fields the R1 observer \
could not read for these candidates. Frames are intermixed below in \
chronological order. Each segment is tagged `[journey]` (single-frame \
context point) or `[dense]` (continuous burst centred on a dispense \
moment).

R1 already confirmed each candidate's `status: used`. **Do not \
re-confirm usage.** Treat status as fixed at `used` for every \
candidate unless the new frames CLEARLY contradict R1 (in which case \
flip to `not_used` and explain).

## Sweep Segments (Round 2)
{segment_block}

## Candidate Items (with R1 amount triple — gaps to fill)
For each candidate the line below shows what R1 already filled and \
what `needs:` to be filled by THIS round.

{candidate_block}

## Per-Item Decision

Output the same 4-field schema as R1, with the same independent-amount \
rules:

- **`status`** — keep `"used"` (R1's confirmed state) unless the new \
frames clearly contradict it.
- **`amount_starting`** — fill level of the stock package BEFORE \
dispensing began. Number if any new frame gives a readable starting \
fill; otherwise `null`.
- **`amount_remaining`** — fill level of the stock package AFTER \
dispensing (post-dispense view, put-back / storage-return frame, \
late visibility). Number if readable in any new frame; otherwise \
`null`. **This is the most important field — fill it whenever you can.**
- **`amount_derivative`** — how much was TAKEN OUT / consumed (visible \
portion on a plate / bowl / pan / cutting board, or amount poured / \
scooped / squeezed). Number if estimable from the new frames; \
otherwise `null`.

You are NOT required to fill the same field R1 filled — you may fill \
the same field again if you have a better read, or fill a different \
field entirely. The script will merge per-field, preferring R1's value \
when it is set.

"Remaining" / "starting" reads count only what is still inside the \
stock container; loose portions on plates / bowls / pans / cutting \
boards count toward `amount_derivative`, not the others.

## Per-Segment Narrative

Write ONE short observation per segment in `per_segment_observations`. \
Use verbs from the action vocabulary (retrieval, access / open \
package, dispensing / pour / scoop / squeeze, visible-portion-on-board, \
restocking / put-away, idle), name the candidate item(s), and indicate \
the dispense stage (pre / during / post / none).

## Output Format

Return ONLY JSON (no other text):
```json
{{
  "items": [
    {{
      "instance_id": "<exact instance_id>",
      "visual_class": "<visual_class>",
      "status": "used" | "not_used",
      "amount_starting": <number or null>,
      "amount_remaining": <number or null>,
      "amount_derivative": <number or null>,
      "evidence_frames": [<timestamp values of key frames>],
      "reasoning": "<one short sentence: which frame(s) gave each amount, and which gap from R1 you filled>"
    }}
  ],
  "per_segment_observations": [
    {{
      "segment_idx": <1-based segment number from the Sweep Segments list>,
      "observation": "<one sentence: what is happening in this segment, which item, which stage>"
    }}
  ]
}}
```

Rules:
- Emit one entry per candidate item. Do NOT skip any.
- Each amount field is independent: fill any combination of starting / \
remaining / derivative. Use `null` for ones the new frames do not \
support. (The script keeps R1's value for any field you leave null.)
- `evidence_frames` should cite at least one frame when any amount is \
filled; can be empty if no new fill could be read.
- `per_segment_observations` MUST include one entry per R2 segment \
listed above (use `segment_idx` 1..N).
"""


def _build_sweep_prompt_round_2(
    n_frames: int,
    timestamps: list[float],
    candidates: list[dict],
    sweep_segments: list[dict],
    frame_seg_idx: list[int],
    r1_items_by_iid: dict[str, dict],
) -> str:
    """Build the R2 sweep observer prompt (4-field gap-fill schema)."""
    iid_to_vc = {c["instance_id"]: c["visual_class"] for c in candidates}

    seg_frames: dict[int, list[int]] = {i: [] for i in range(len(sweep_segments))}
    for fi, si in enumerate(frame_seg_idx):
        if 0 <= si < len(sweep_segments):
            seg_frames[si].append(fi + 1)

    seg_lines = []
    iids_in_segments: dict[str, list[int]] = {}
    for si, seg in enumerate(sweep_segments):
        frames = seg_frames.get(si, [])
        if frames:
            f_lo, f_hi = frames[0], frames[-1]
            frame_str = f"frames {f_lo}–{f_hi}" if f_lo != f_hi else f"frame {f_lo}"
        else:
            frame_str = "frames (none extracted)"
        target_iids = seg.get("target_items", [])
        target_vcs = []
        for iid in target_iids:
            vc = iid_to_vc.get(iid)
            if vc:
                target_vcs.append(vc)
                iids_in_segments.setdefault(iid, []).append(si + 1)
        target_str = ", ".join(target_vcs) if target_vcs else "(no candidates)"
        kind = seg.get("kind", "dense")
        kind_tag = "[journey]" if kind == "journey" else "[dense]"
        if kind == "journey" or seg.get("end") == seg.get("start"):
            t_str = f"t={seg['start']:.1f}s"
        else:
            t_str = f"t={seg['start']:.1f}–{seg['end']:.1f}s"
        seg_lines.append(
            f"- Segment {si+1} {kind_tag}: {t_str}, "
            f"{frame_str}\n    target items: {target_str}"
        )
    segment_block = "\n".join(seg_lines) if seg_lines else "(no segments)"

    cand_lines = []
    for c in candidates:
        unit = "grams" if c["unit"] == "g" else "count"
        segs_for_c = iids_in_segments.get(c["instance_id"], [])
        seg_str = (
            ", ".join(f"{i}" for i in segs_for_c) if segs_for_c else "(none)"
        )
        sw_r1 = r1_items_by_iid.get(c["instance_id"]) or {}
        s1, r1_, d1 = (
            sw_r1.get("amount_starting"),
            sw_r1.get("amount_remaining"),
            sw_r1.get("amount_derivative"),
        )
        needs = _r1_needs_str(s1, r1_, d1)
        cand_lines.append(
            f"- instance_id=`{c['instance_id']}`  "
            f"visual_class=\"{c['visual_class']}\"  "
            f"unit={unit}  package={c.get('package_amount', '')}\n"
            f"    R2 segments: {seg_str}\n"
            f"    R1 amounts: starting={s1}  remaining={r1_}  derivative={d1}\n"
            f"    needs: {needs}"
        )
    candidate_block = "\n".join(cand_lines) if cand_lines else "(no candidates)"

    return SWEEP_ROUND_2_PROMPT.format(
        n_frames=n_frames,
        n_segments=len(sweep_segments),
        segment_block=segment_block,
        candidate_block=candidate_block,
    )


def run_sweep_observer_round_2(
    client,
    frames_b64: list[str],
    timestamps: list[float],
    candidates: list[dict],
    sweep_segments: list[dict],
    frame_seg_idx: list[int],
    r1_items_by_iid: dict[str, dict],
    model: str,
    prompt_save_path: Path | None = None,
) -> tuple[str, str, dict]:
    prompt = _build_sweep_prompt_round_2(
        len(frames_b64), timestamps, candidates,
        sweep_segments=sweep_segments,
        frame_seg_idx=frame_seg_idx,
        r1_items_by_iid=r1_items_by_iid,
    )
    if prompt_save_path is not None:
        prompt_save_path.parent.mkdir(parents=True, exist_ok=True)
        prompt_save_path.write_text(prompt)

    max_retries = 5
    for attempt in range(max_retries):
        t0 = time.time()
        try:
            response_text, in_tok, out_tok = _observer_api_call(
                client, model, prompt, frames_b64, timestamps,
            )
            stats = {
                "inference_time_s": round(time.time() - t0, 2),
                "model": model,
                "num_frames": len(frames_b64),
                "num_candidates": len(candidates),
                "input_tokens": in_tok,
                "output_tokens": out_tok,
                "attempt": attempt + 1,
                "round": 2,
            }
        except Exception as e:
            err_str = str(e)
            transient = any(m in err_str for m in (
                "503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED",
                "500", "INTERNAL", "deadline", "timeout", "connection", "Connection",
            ))
            if "content_policy_violation" in err_str and attempt < max_retries - 1:
                print(f" R{2} content filter (attempt {attempt + 1}), retrying...", end="", flush=True)
                time.sleep(5 * (attempt + 1))
                continue
            if transient and attempt < max_retries - 1:
                wait = 4 * (2 ** attempt)
                print(f" R{2} transient API error (attempt {attempt + 1}), retrying in {wait}s...", end="", flush=True)
                time.sleep(wait)
                continue
            print(f"  R{2} Sweep ERROR: {e}")
            return "", prompt, {"error": err_str, "inference_time_s": round(time.time() - t0, 2)}

        if _is_refusal(response_text) and attempt < max_retries - 1:
            print(f" R{2} refusal (attempt {attempt + 1}), retrying...", end="", flush=True)
            time.sleep(5)
            continue

        return response_text, prompt, stats

    return "", prompt, {"error": "max retries exceeded"}


# ---------------------------------------------------------------------------
# Session pipeline (single planner call → observer per planned iid)
# ---------------------------------------------------------------------------

def process_session(
    participant: str,
    session: str,
    client,
    observer_client,
    ledger: dict,
    model: str,
    fps: float,
    max_frames: int,
    min_score: float,
    planner_only: bool = False,
    observer_only: bool = False,
    verbose: bool = False,
    model_tag: str = "default",
    run_tag: str = "default",
    inventory_scope: str = "full",
    transparency_by_iid: dict | None = None,
    evidence_mode: str = "chrono",
    seg_tau_dino: float = 0.15,
    seg_gap_close: float = 2.0,
    seg_min_duration: float = 1.5,
    flicker_min_score: float = 0.15,
    flicker_min_hits: int = 2,
    flicker_peak_score: float = 0.25,
    block_gap_close: float = 0.0,
    max_block_s: float | None = None,
    crosstalk_dino_ratio: float = 0.4,
    crosstalk_cov_floor: float = 0.7,
    sweep_budget: int = 20,
    enable_per_item_observer: bool = False,
    rounds: int = 1,
    sweep_budget_r2: int = 8,
) -> tuple[list[dict], dict]:
    session_log: dict = {
        "session": session, "planner": {}, "sweep": {},
        "planner_r2": {}, "sweep_r2": {}, "observer": [],
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

    dino_by_t = load_dino_by_t(participant, session)
    scene_by_t = load_owlv2_scene_by_t(participant, session)
    hoi_details_by_t = load_hoi_details_by_t(participant, session)

    hoi_sorted = sorted(hoi_ts)
    print(f"  {session}: {len(all_ts)} frames, {len(hoi_sorted)} HOI-contact, "
          f"{len(inventory)} inventory items ({inventory_scope} scope), "
          f"{len(scene_by_t)} OWLv2 scene tags")

    # ── Step 1: Sweep-only Planner ──
    # NOTE: per-item observer is temporarily disabled. The planner now uses
    # PLANNER_SWEEP_ONLY_PROMPT, which asks only for `item_decisions` +
    # shared `sweep_timestamps`. No per-item segments are produced.
    planner_model = "gpt-5.4"
    if observer_only:
        planner_sess_path = (
            outputs_dir(participant, session)
            / f"{OUTPUT_PREFIX}_{model_tag}_{run_tag}_planner.json"
        )
        if not planner_sess_path.exists():
            print(f"  {session}: --observer-only but no saved plan at "
                  f"{planner_sess_path} — skipping")
            return [], session_log
        try:
            saved = json.loads(planner_sess_path.read_text())
        except json.JSONDecodeError as e:
            print(f"  {session}: failed to read saved plan ({e}) — skipping")
            return [], session_log
        saved_log = saved.get("session", {})
        saved_planner = saved_log.get("planner", {}) or {}
        item_decisions = saved_planner.get("item_decisions") or []
        journey_samples = saved_planner.get("journey_samples") or []
        dense_windows = saved_planner.get("dense_windows") or []
        # Legacy plan files only stored `sweep_segments`; treat them as
        # dense windows so old caches still work.
        if not journey_samples and not dense_windows:
            for seg in saved_planner.get("sweep_segments") or []:
                dense_windows.append({
                    "start": seg["start"], "end": seg["end"],
                    "target_items": seg["target_items"],
                })
        session_log["planner"] = saved_planner
        print(f"  Step 1 (Planner): LOADED {len(item_decisions)} decisions, "
              f"{len(journey_samples)} journey samples, "
              f"{len(dense_windows)} dense windows from saved plan")
        planner_stats = saved_planner.get("stats", {})
    else:
        prompt, ev_stats, _active_vcs, _active_iids = build_planner_prompt_sweep_only(
            participant=participant,
            session=session,
            inventory=inventory,
            transparency_by_iid=transparency_by_iid,
            dino_by_t=dino_by_t,
            scene_by_t=scene_by_t,
            hoi_details_by_t=hoi_details_by_t,
            hoi_sorted=hoi_sorted,
            evidence_mode=evidence_mode,
            min_score=min_score,
            inventory_scope=inventory_scope,
            seg_tau_dino=seg_tau_dino,
            seg_gap_close=seg_gap_close,
            seg_min_duration=seg_min_duration,
            flicker_min_score=flicker_min_score,
            flicker_min_hits=flicker_min_hits,
            flicker_peak_score=flicker_peak_score,
            block_gap_close=block_gap_close,
            max_block_s=max_block_s,
            crosstalk_dino_ratio=crosstalk_dino_ratio,
            crosstalk_cov_floor=crosstalk_cov_floor,
            sweep_budget=sweep_budget,
        )
        print(f"  Evidence ({evidence_mode}): {ev_stats['n_items_emitted']} items with hits "
              f"({ev_stats['n_rows_emitted']} rows across {ev_stats['n_hoi_frames_total']} HOI frames)")

        print(f"  Step 1 (Sweep-only Planner): sending evidence to {planner_model}...")
        planner_text, planner_stats = run_planner(
            client, prompt, planner_model,
            prompt_save_path=cache_dir / "planner_prompt.txt",
        )
        (cache_dir / "planner_response.txt").write_text(planner_text or "")
        item_decisions, journey_samples, dense_windows = parse_planner_sweep_response(planner_text)
        n_observe = sum(1 for d in item_decisions if d.get("decision") == "observe")
        n_skip = sum(1 for d in item_decisions if d.get("decision") == "no_observation")
        print(f"  Planner decisions: {len(item_decisions)} total — "
              f"{n_observe} observe, {n_skip} no_observation; "
              f"{len(journey_samples)} journey samples, "
              f"{len(dense_windows)} dense windows")
        if n_skip:
            for d in item_decisions:
                if d.get("decision") == "no_observation":
                    print(f"    SKIP {d.get('item')}: {d.get('reasoning', '')[:120]}")
        session_log["planner"] = {
            "n_decisions_total": len(item_decisions),
            "n_decisions_observe": n_observe,
            "n_decisions_no_observation": n_skip,
            "evidence_mode": evidence_mode,
            "evidence_stats": ev_stats,
            "item_decisions": item_decisions,
            "journey_samples": journey_samples,
            "dense_windows": dense_windows,
            "stats": planner_stats,
            "prompt": prompt,
            "raw_response": planner_text,
        }

    inv_by_iid = {inv["instance_id"]: inv for inv in inventory}
    candidates: list[dict] = []
    for d in item_decisions:
        if d.get("decision") != "observe":
            continue
        iid = d.get("instance_id") or ""
        if iid not in inv_by_iid:
            if verbose:
                print(f"    SKIP: planner proposed '{iid}' not in inventory")
            continue
        candidates.append(inv_by_iid[iid])

    cand_str = ", ".join(c["visual_class"] for c in candidates)
    print(f"  Planner candidates ({len(candidates)}): {cand_str}")

    if planner_only:
        return [], session_log

    if not candidates or (not journey_samples and not dense_windows):
        print(f"  Planner returned empty candidates or empty samples — "
              f"nothing to sweep")
        return [], session_log

    # Filter target_items down to valid candidate iids in both lists.
    valid_iids = {c["instance_id"] for c in candidates}

    def _filter_targets(entry: dict) -> dict | None:
        kept = [iid for iid in entry.get("target_items", []) if iid in valid_iids]
        if not kept:
            return None
        out = dict(entry)
        out["target_items"] = kept
        return out

    filtered_journey = [e for e in (_filter_targets(j) for j in journey_samples) if e]
    filtered_dense = [e for e in (_filter_targets(d) for d in dense_windows) if e]

    if not filtered_journey and not filtered_dense:
        print(f"  No samples survived target_items validation — nothing to sweep")
        return [], session_log

    # ── Step 2: Sweep observer (single multi-item call on segment frames) ──
    if enable_per_item_observer:
        print("  WARNING: --enable-per-item-observer set but the sweep-only "
              "planner does not emit per-item segments; per-item observer "
              "stays disabled in this build.")

    clips = get_session_clips(participant, session)
    if not clips:
        print(f"  {session}: no video clips found")
        return [], session_log
    video_durations = [(path, dur) for _, path, dur in clips]

    sweep_frames, sweep_frame_ts, frame_seg_idx, seg_kinds = (
        _extract_sweep_journey_dense_frames(
            filtered_journey, filtered_dense,
            video_durations, fps=fps, max_frames=max_frames,
        )
    )
    if not sweep_frames:
        print(f"  Sweep: no frames could be extracted from planner samples")
        return [], session_log

    # Build the unified segment list (interleaved journey + dense, sorted
    # by start time) used for prompt rendering and per-segment narrative.
    unified_segments: list[dict] = []
    for js in filtered_journey:
        unified_segments.append({
            "kind": "journey",
            "start": js["t"], "end": js["t"],
            "target_items": js["target_items"],
        })
    for dw in filtered_dense:
        unified_segments.append({
            "kind": "dense",
            "start": dw["start"], "end": dw["end"],
            "target_items": dw["target_items"],
        })
    unified_segments.sort(key=lambda s: s["start"])

    print(f"    Sweep: {len(sweep_frames)} frames across "
          f"{len(filtered_journey)} journey + {len(filtered_dense)} dense "
          f"× {len(candidates)} candidates...",
          end="", flush=True)
    sweep_text, sweep_prompt, sweep_stats = run_sweep_observer(
        observer_client, sweep_frames, sweep_frame_ts, candidates,
        sweep_segments=unified_segments,
        frame_seg_idx=frame_seg_idx,
        model=model,
        prompt_save_path=cache_dir / "sweep_prompt.txt",
    )
    (cache_dir / "sweep_response.txt").write_text(sweep_text or "")
    sweep_items, sweep_per_seg = parse_sweep_response(sweep_text)
    print(f" returned {len(sweep_items)} item entries, "
          f"{len(sweep_per_seg)} per-segment observations")

    sweep_by_iid = {it["instance_id"]: it for it in sweep_items if it.get("instance_id")}

    session_log["sweep"] = {
        "n_frames": len(sweep_frame_ts),
        "frame_timestamps": sweep_frame_ts,
        "frame_seg_idx": frame_seg_idx,
        "journey_samples": filtered_journey,
        "dense_windows": filtered_dense,
        "unified_segments": unified_segments,
        "n_candidates": len(candidates),
        "items": sweep_items,
        "per_segment_observations": sweep_per_seg,
        "stats": sweep_stats,
        "prompt": sweep_prompt,
        "raw_response": sweep_text,
    }

    # ── Step 3: Round-N replan loop (dispensal-window for still-derivative items) ──
    final_by_iid: dict[str, dict] = dict(sweep_by_iid)
    # Track the round (>=2) that finally produced each item's value, if any.
    item_resolved_round: dict[str, int] = {}
    # Plan-observe history seeded with R1.
    prior_rounds_log: list[dict] = [{
        "round": 1,
        "segments": unified_segments,
        "items_by_iid": sweep_by_iid,
        "per_segment_observations": sweep_per_seg,
    }]
    # All sampled segments so far (for non-overlap filtering against new rounds).
    all_prior_segments: list[dict] = list(unified_segments)
    # Per-round stats so we can attach them to predictions later.
    extra_stats_log: list[dict] = []  # entries: {"round": n, "planner": {...}, "sweep": {...}}

    # ── Round 2 (gap-fill) — only fires if --rounds >= 2 and unresolved items exist ──
    # An item is "unresolved" iff status=used AND we cannot derive a remaining
    # estimate from R1's amount triple. Resolvable iff:
    #   amount_remaining is not None  OR  (amount_starting AND amount_derivative both set)
    if rounds >= 2:
        unresolved_iids: list[str] = []
        for iid, sw in final_by_iid.items():
            if sw.get("status") != "used":
                continue
            r = sw.get("amount_remaining")
            s = sw.get("amount_starting")
            d = sw.get("amount_derivative")
            if r is not None:
                continue
            if s is not None and d is not None:
                continue
            unresolved_iids.append(iid)

        if not unresolved_iids:
            print("  Round 2: every R1 item is resolvable (direct remaining or "
                  "starting+derivative pair) — skipping R2")
        else:
            target_inventory = [
                inv_by_iid[iid] for iid in unresolved_iids if iid in inv_by_iid
            ]
            print(f"  Round 2: {len(target_inventory)} unresolved item(s) — "
                  f"{', '.join(inv['visual_class'] for inv in target_inventory)}")

            r2_prompt, r2_ev_stats, _, _ = build_planner_prompt_round_2(
                participant=participant,
                session=session,
                target_inventory=target_inventory,
                r1_items_by_iid=sweep_by_iid,
                r1_segments=unified_segments,
                r1_per_segment_observations=sweep_per_seg,
                transparency_by_iid=transparency_by_iid,
                dino_by_t=dino_by_t,
                scene_by_t=scene_by_t,
                hoi_details_by_t=hoi_details_by_t,
                hoi_sorted=hoi_sorted,
                evidence_mode=evidence_mode,
                min_score=min_score,
                inventory_scope=inventory_scope,
                seg_tau_dino=seg_tau_dino,
                seg_gap_close=seg_gap_close,
                seg_min_duration=seg_min_duration,
                flicker_min_score=flicker_min_score,
                flicker_min_hits=flicker_min_hits,
                flicker_peak_score=flicker_peak_score,
                block_gap_close=block_gap_close,
                max_block_s=max_block_s,
                crosstalk_dino_ratio=crosstalk_dino_ratio,
                crosstalk_cov_floor=crosstalk_cov_floor,
                sweep_budget=sweep_budget_r2,
            )
            r2_planner_text, r2_planner_stats = run_planner(
                client, r2_prompt, planner_model,
                prompt_save_path=cache_dir / "planner_r2_prompt.txt",
            )
            (cache_dir / "planner_r2_response.txt").write_text(r2_planner_text or "")
            r2_decisions, r2_journey, r2_dense = parse_planner_sweep_response(r2_planner_text)
            print(f"  R2 planner: {len(r2_decisions)} decisions, "
                  f"{len(r2_journey)} journey samples, {len(r2_dense)} dense windows")

            session_log["planner_r2"] = {
                "n_decisions_total": len(r2_decisions),
                "evidence_stats": r2_ev_stats,
                "item_decisions": r2_decisions,
                "journey_samples": r2_journey,
                "dense_windows": r2_dense,
                "stats": r2_planner_stats,
                "prompt": r2_prompt,
                "raw_response": r2_planner_text,
            }

            valid_r2_iids = {inv["instance_id"] for inv in target_inventory}

            def _filter_targets_r2(entry: dict) -> dict | None:
                kept = [iid for iid in entry.get("target_items", []) if iid in valid_r2_iids]
                if not kept:
                    return None
                out = dict(entry)
                out["target_items"] = kept
                return out

            r2_filtered_journey: list[dict] = []
            for js in r2_journey:
                e = _filter_targets_r2(js)
                if not e:
                    continue
                t = float(e["t"])
                # Drop any journey sample inside an R1 segment.
                overlap = any(p["start"] <= t <= p["end"] for p in all_prior_segments)
                if overlap:
                    print(f"    R2 journey t={t:.1f}s overlaps an R1 segment — dropping")
                    continue
                r2_filtered_journey.append(e)

            r2_filtered_dense: list[dict] = []
            for dw in r2_dense:
                e = _filter_targets_r2(dw)
                if not e:
                    continue
                overlap = any(
                    not (e["end"] <= p["start"] or e["start"] >= p["end"])
                    for p in all_prior_segments
                )
                if overlap:
                    print(f"    R2 dense t={e['start']:.1f}-{e['end']:.1f}s "
                          f"overlaps an R1 segment — dropping")
                    continue
                r2_filtered_dense.append(e)

            if not r2_filtered_journey and not r2_filtered_dense:
                print("  R2 planner returned no usable samples — "
                      "evidence likely exhausted; skipping R2 observer")
                extra_stats_log.append({"round": 2, "planner": r2_planner_stats, "sweep": None})
            else:
                r2_candidates = list(target_inventory)
                r2_frames, r2_ts, r2_seg_idx, r2_seg_kinds = (
                    _extract_sweep_journey_dense_frames(
                        r2_filtered_journey, r2_filtered_dense,
                        video_durations, fps=fps, max_frames=max_frames,
                    )
                )
                if not r2_frames:
                    print("    R2 Sweep: no frames extracted — skipping observer")
                    extra_stats_log.append({"round": 2, "planner": r2_planner_stats, "sweep": None})
                else:
                    r2_unified: list[dict] = []
                    for js in r2_filtered_journey:
                        r2_unified.append({
                            "kind": "journey",
                            "start": js["t"], "end": js["t"],
                            "target_items": js["target_items"],
                        })
                    for dw in r2_filtered_dense:
                        r2_unified.append({
                            "kind": "dense",
                            "start": dw["start"], "end": dw["end"],
                            "target_items": dw["target_items"],
                        })
                    r2_unified.sort(key=lambda s: s["start"])

                    print(f"    R2 Sweep: {len(r2_frames)} frames across "
                          f"{len(r2_filtered_journey)} journey + "
                          f"{len(r2_filtered_dense)} dense × "
                          f"{len(r2_candidates)} candidates...",
                          end="", flush=True)
                    r2_text, r2_prompt_str, r2_sweep_stats = run_sweep_observer_round_2(
                        observer_client, r2_frames, r2_ts, r2_candidates,
                        sweep_segments=r2_unified,
                        frame_seg_idx=r2_seg_idx,
                        r1_items_by_iid=sweep_by_iid,
                        model=model,
                        prompt_save_path=cache_dir / "sweep_r2_prompt.txt",
                    )
                    (cache_dir / "sweep_r2_response.txt").write_text(r2_text or "")
                    r2_items, r2_per_seg = parse_sweep_response(r2_text)
                    print(f" returned {len(r2_items)} item entries, "
                          f"{len(r2_per_seg)} per-segment observations")

                    r2_by_iid = {it["instance_id"]: it for it in r2_items if it.get("instance_id")}
                    session_log["sweep_r2"] = {
                        "n_frames": len(r2_ts),
                        "frame_timestamps": r2_ts,
                        "frame_seg_idx": r2_seg_idx,
                        "journey_samples": r2_filtered_journey,
                        "dense_windows": r2_filtered_dense,
                        "unified_segments": r2_unified,
                        "n_candidates": len(r2_candidates),
                        "items": r2_items,
                        "per_segment_observations": r2_per_seg,
                        "stats": r2_sweep_stats,
                        "prompt": r2_prompt_str,
                        "raw_response": r2_text,
                    }
                    extra_stats_log.append({"round": 2, "planner": r2_planner_stats, "sweep": r2_sweep_stats})

                    # ── Per-field merge: R1 wins for any field already filled;
                    # R2 fills nulls. Status stays at R1's "used" unless R2
                    # explicitly returns "not_used" (rare). evidence_frames
                    # and reasoning are concatenated with an R2: prefix.
                    filled_count = 0
                    for iid in unresolved_iids:
                        r2_sw = r2_by_iid.get(iid)
                        if not r2_sw:
                            continue
                        prior = final_by_iid.get(iid, {})
                        merged = dict(prior)
                        gained_fields = []
                        for fld in ("amount_starting", "amount_remaining", "amount_derivative"):
                            if merged.get(fld) is None and r2_sw.get(fld) is not None:
                                merged[fld] = r2_sw.get(fld)
                                gained_fields.append(fld)
                        # Carry R2 evidence frames forward separately for transparency.
                        merged["evidence_frames_r2"] = r2_sw.get("evidence_frames", [])
                        merged["reasoning_r2"] = r2_sw.get("reasoning", "")
                        # Allow a clear contradiction to flip status.
                        if r2_sw.get("status") == "not_used":
                            merged["status"] = "not_used"
                        if gained_fields:
                            item_resolved_round[iid] = 2
                            filled_count += 1
                            inv = inv_by_iid.get(iid, {})
                            print(f"    R2 filled {gained_fields} for "
                                  f"{inv.get('visual_class', iid)}")
                        final_by_iid[iid] = merged
                    print(f"  R2: filled missing field(s) for {filled_count} item(s)")

    predictions: list[dict] = []
    for cand in candidates:
        iid = cand["instance_id"]
        sw = final_by_iid.get(iid)
        if not sw:
            print(f"    {cand['visual_class']}: missing from sweep response")
            continue
        status = sw["status"]
        if status == "not_used":
            print(f"    {cand['visual_class']}: not_used "
                  f"({sw.get('reasoning', '')[:80]})")
            continue

        amt_s = sw.get("amount_starting")
        amt_r = sw.get("amount_remaining")
        amt_d = sw.get("amount_derivative")

        # Pick the best available remaining estimate.
        # Priority: direct remaining > starting − derivative > derivative-only > starting-only.
        if amt_r is not None:
            amount = amt_r
            kind = "remaining"
        elif amt_s is not None and amt_d is not None:
            amount = max(0.0, amt_s - amt_d)
            kind = "computed_remaining"
        elif amt_d is not None:
            amount = amt_d
            kind = "derivative"
        elif amt_s is not None:
            amount = amt_s
            kind = "starting_only"
        else:
            print(f"    {cand['visual_class']}: status=used but no amount fields")
            continue

        round_src = item_resolved_round.get(iid, 1)
        marker = f" (R{round_src})" if round_src > 1 else ""
        print(f"    {cand['visual_class']}: {kind}={amount}{marker} "
              f"[s={amt_s} r={amt_r} d={amt_d}]")

        planner_entry = next(
            (d for d in item_decisions if d.get("instance_id") == iid), {}
        )
        stats_block = {"planner": planner_stats, "sweep": sweep_stats}
        for s in extra_stats_log:
            rn = s["round"]
            if s.get("planner"):
                stats_block[f"planner_r{rn}"] = s["planner"]
            if s.get("sweep"):
                stats_block[f"sweep_r{rn}"] = s["sweep"]
        predictions.append({
            "session": session,
            "item": cand["visual_class"],
            "instance_id": iid,
            "amount_starting": amt_s,
            "amount_remaining": amount if kind in {"remaining", "computed_remaining"} else None,
            "amount_derivative": amt_d,
            "amount_remaining_raw": amt_r,
            "amount_kind": kind,
            "status": status,
            "round_source": f"r{round_src}",
            "reasoning": sw.get("reasoning", ""),
            "evidence_frames": sw.get("evidence_frames", []),
            "planner_reasoning": planner_entry.get("reasoning", ""),
            "planner_confidence": planner_entry.get("confidence", ""),
            "sweep_frame_timestamps": sweep_frame_ts,
            "stats": stats_block,
        })

    print(f"  {session}: {len(predictions)} predictions from "
          f"{len(candidates)} candidates via sweep "
          f"(rounds={rounds})")
    return predictions, session_log


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="AVP Round 1 (remaining-only, minimal): single-shot "
                    "plan + observe. Planner reads Iterative-style inventory "
                    "+ per-item HOI/DINO evidence (SigLIP dropped — not a "
                    "useful signal in practice) and emits one decision per "
                    "active iid; observer is the noTAD-style per-item "
                    "remaining-amount estimator."
    )
    parser.add_argument("--participant", default="kailai")
    parser.add_argument("--tag", required=True,
                        help="Short label for this run (e.g. 'minimal_v1').")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--session", help="Single session")
    group.add_argument("--all", action="store_true", help="All sessions")
    parser.add_argument("--model", default="gemini-2.5-pro",
                        help="Observer model (planner always uses gpt-5.4). "
                             "Models prefixed `gemini-` route through the "
                             "google-genai SDK (cheap, fast, 100+ frames OK). "
                             "Anything else routes through Azure OpenAI "
                             "(gpt-5.x; 50-frame hard cap, slower, costlier).")
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--max-frames", type=int, default=100,
                        help="Max frames per single observer call. Default 100 "
                             "is sized for gemini-2.5-pro + journey/dense "
                             "split. With --model gpt-5.x, cap at 50 (Azure "
                             "image limit).")
    parser.add_argument("--min-score", type=float, default=0.15,
                        help="Only include per-frame detections with DINO >= this.")
    parser.add_argument("--planner-only", action="store_true")
    parser.add_argument("--observer-only", action="store_true",
                        help="Skip planner; load saved plan (same --tag) and run observer only.")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", action="store_true",
                        help="Resume a previously-interrupted run with the same --tag.")
    parser.add_argument("--inventory-scope", choices=["full", "session"], default="full",
                        help="Which items to list in the planner's inventory. "
                             "'full' = all items in stock at session time (default, "
                             "matches real home deployment); "
                             "'session' = GT-annotated subset only.")
    parser.add_argument(
        "--evidence-mode",
        choices=["blocks", "segments", "chrono", "per_frame"],
        default="chrono",
        help="How to render per-item detections in the planner prompt. "
             "'chrono' (default) emits a single chronological timeline of "
             "HOI-gated DINO segments — one row per (item × interval). "
             "'blocks' merges segments + flicker clusters into activity "
             "blocks (transitive interval overlap) and emits a header per "
             "block with per-visual_class sub-rows sorted by peak DINO — "
             "fewer rows in cooking phases at the cost of fine-grained "
             "starts/ends. "
             "'segments' groups segment rows under each visual_class. "
             "'per_frame' falls back to the dense row-per-frame format.",
    )
    parser.add_argument("--block-gap-close", type=float, default=0.0,
                        help="Seconds to fuse near-touching intervals into "
                             "the same activity block (blocks mode). 0 = "
                             "strict transitive overlap.")
    parser.add_argument("--max-block-s", type=float, default=None,
                        help="If set, split blocks longer than this many "
                             "seconds at the largest internal coverage gap "
                             "(blocks mode). Default: no cap.")
    parser.add_argument("--crosstalk-dino-ratio", type=float, default=0.4,
                        help="Cross-talk filter (blocks mode): drop a "
                             "non-focus sub-row when its peak DINO is below "
                             "this fraction of the block's max peak DINO "
                             "AND its coverage is ≥ --crosstalk-cov-floor. "
                             "0 disables the filter; 1.0 keeps only the "
                             "block's max-DINO item. Default 0.4.")
    parser.add_argument("--crosstalk-cov-floor", type=float, default=0.7,
                        help="Coverage floor for the cross-talk filter "
                             "(blocks mode). An item with coverage below "
                             "this fraction is kept regardless of DINO "
                             "ratio (it's a brief co-occurrence, not bleed). "
                             "Default 0.7.")
    parser.add_argument("--seg-tau-dino", type=float, default=0.15,
                        help="DINO threshold for 05b segment activation (segments mode). "
                             "SigLIP-based activation is hardcoded off (tau_siglip=1.1).")
    parser.add_argument("--seg-gap-close", type=float, default=2.0,
                        help="Seconds between active runs to merge into one segment (segments mode).")
    parser.add_argument("--seg-min-duration", type=float, default=1.5,
                        help="Minimum segment duration in seconds (segments mode).")
    parser.add_argument("--flicker-min-score", type=float, default=0.15)
    parser.add_argument("--flicker-min-hits", type=int, default=2)
    parser.add_argument("--flicker-peak-score", type=float, default=0.25)
    parser.add_argument("--sweep-budget", type=int, default=12,
                        help="Cap on the number of dense_windows the "
                             "planner may emit. journey_samples gets a "
                             "more generous cap (≈ 2× sweep-budget). Total "
                             "sweep frames are capped separately at "
                             "--max-frames.")
    parser.add_argument("--enable-per-item-observer", action="store_true",
                        help="(Currently a no-op.) Per-item observer is "
                             "disabled in the sweep-only branch — the planner "
                             "no longer emits per-item segments.")
    parser.add_argument("--rounds", type=int, default=1,
                        help="Number of plan-observe rounds: 1 = R1 sweep "
                             "only; 2 = R2 gap-fill replan on items the "
                             "script could not derive a remaining estimate "
                             "for after R1 (no direct amount_remaining and "
                             "not both amount_starting + amount_derivative). "
                             "R2 sees R1's per-segment observations and "
                             "proposes new journey/dense windows that "
                             "target the missing amount field(s). Capped "
                             "at 2 by design.")
    parser.add_argument("--sweep-budget-r2", type=int, default=8,
                        help="Cap on the number of segments each round-N "
                             "dispensal-window planner (N>=2) may emit. "
                             "Same budget applies to all replan rounds.")
    args = parser.parse_args()

    if args.planner_only and args.observer_only:
        parser.error("--planner-only and --observer-only are mutually exclusive.")
    if args.rounds > 2:
        parser.error("--rounds is capped at 2 by design (R2 is a single gap-fill pass).")

    sessions = (
        [args.session] if args.session
        else get_sessions(args.participant) if args.all
        else get_sessions(args.participant)
    )

    model_tag = args.model.replace("/", "_")
    run_tag = args.tag.replace("/", "_")

    print(f"{'=' * 70}")
    print(f"AVP Round 1 (Remaining-Only, minimal)")
    print(f"{'=' * 70}")
    print(f"Participant:  {args.participant}")
    print(f"Model:        {args.model}")
    print(f"Tag:          {run_tag}")
    print(f"FPS:          {args.fps}")
    print(f"Max frames:   {args.max_frames} per observer call")
    print(f"Min score:    {args.min_score}")
    print(f"Evidence:     {args.evidence_mode}")
    print(f"Sweep budget: {args.sweep_budget} segments (R1) / {args.sweep_budget_r2} (R2)")
    print(f"Rounds:       {args.rounds}")
    print(f"Per-item obs: DISABLED (sweep-only branch)")
    print(f"Planner only: {args.planner_only}")
    print(f"Observer only:{args.observer_only}")
    print(f"Sessions:     {len(sessions)}")
    print()

    client = make_client()  # planner (always Azure / gpt-5.4)
    observer_client = make_observer_client(args.model)
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
    if args.observer_only:
        print("\nOBSERVER-ONLY: loading saved plans and re-running observer "
              "for all requested sessions; prior preds/status will be overwritten.")
    elif args.resume and preds_path.exists() and status_path.exists():
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
                observer_client=observer_client,
                ledger=ledger,
                model=args.model,
                fps=args.fps,
                max_frames=args.max_frames,
                min_score=args.min_score,
                planner_only=args.planner_only,
                observer_only=args.observer_only,
                verbose=args.verbose,
                model_tag=model_tag,
                run_tag=run_tag,
                inventory_scope=args.inventory_scope,
                transparency_by_iid=transparency_by_iid,
                evidence_mode=args.evidence_mode,
                seg_tau_dino=args.seg_tau_dino,
                seg_gap_close=args.seg_gap_close,
                seg_min_duration=args.seg_min_duration,
                flicker_min_score=args.flicker_min_score,
                flicker_min_hits=args.flicker_min_hits,
                flicker_peak_score=args.flicker_peak_score,
                block_gap_close=args.block_gap_close,
                max_block_s=args.max_block_s,
                crosstalk_dino_ratio=args.crosstalk_dino_ratio,
                crosstalk_cov_floor=args.crosstalk_cov_floor,
                sweep_budget=args.sweep_budget,
                enable_per_item_observer=args.enable_per_item_observer,
                rounds=args.rounds,
                sweep_budget_r2=args.sweep_budget_r2,
            )
        except KeyboardInterrupt:
            raise
        except Exception as e:
            import traceback
            print(f"\n  ERROR in session {session}: {e}")
            traceback.print_exc()
            failed_sessions.append((session, str(e)[:200]))
            preds, log = [], {"session": session, "planner": {}, "sweep": {},
                              "planner_r2": {}, "sweep_r2": {},
                              "observer": [], "error": str(e)[:500]}
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
            kind = p.get("amount_kind", "?")
            amt = (
                p.get("amount_remaining") if kind in {"remaining", "computed_remaining"}
                else p.get("amount_derivative") if kind == "derivative"
                else p.get("amount_starting") if kind == "starting_only"
                else None
            )
            print(
                f"  [{p['session']}] {p['item']}: {kind}={amt} {unit} "
                f"[s={p.get('amount_starting')} r={p.get('amount_remaining_raw')} "
                f"d={p.get('amount_derivative')}] "
                f"({p.get('planner_confidence', '')})"
            )

    if failed_sessions:
        print(f"\n{len(failed_sessions)} session(s) failed in this run:")
        for s, err in failed_sessions:
            print(f"  {s}: {err}")
        print(f"\nRe-run with --resume to retry the failed sessions only.")


if __name__ == "__main__":
    main()

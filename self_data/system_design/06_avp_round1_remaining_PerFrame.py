#!/usr/bin/env python3
"""AVP Round 1 (remaining-only, PerFrame): planner reads raw per-frame HOI +
DINO/SigLIP similarity scores + OWLv2 scene tags directly, bypassing
05b_per_item_segments.py.

Motivation: 05b's morphological filter drops items whose detections are
scattered in time (gap_close / min_duration), and its visual_class-keyed dict
silently overwrites duplicate-purchase instances of the same product.  By
sending the raw per-frame evidence the planner sees every detection and can
decide for itself which are real; there is no upstream temporal filter.

Context-window sanity check (kailai, 69 sessions):
  mean 567 HOI frames × 1.56 DINO matches/frame ≈ 880 tuples per session
  ≈ 6K planner tokens per session, well under any LLM budget.
  Worst observed (4 299 s session) ≈ 30K tokens.

Two-step pipeline (same as SceneTransp/noTAD):
  Step 1 (Planner): text-only — reads per-frame evidence, picks observation
    windows.  Source: hands23 + SigLIP + DINO + OWLv2 per-frame JSONs.
  Step 2 (Observer): VLM with frames — asks ONLY for remaining amount.

Per-frame line format in the planner prompt:
    [   12.0s]  stove      large_white_eggs_20260403=0.42  whole_milk_gallon_20260318=0.31
    [   14.0s]  storage    whole_milk_gallon_20260318=0.41
Each line is one HOI-contact frame.  `scene` is the OWLv2 tag
({storage|sink|stove|unknown}).  Items are shown as
`instance_id=dino_score` (optionally `:sig=siglip_score`) when the raw
similarity is >= --min-score.

Usage:
  python system_design/06_avp_round1_remaining_PerFrame.py \
      --participant kailai --tag PerFrame_v1
  python system_design/06_avp_round1_remaining_PerFrame.py \
      --participant kailai --session 20260310-195710 --tag PerFrame_v1
"""

import argparse
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

CACHE_DIR = Path(__file__).resolve().parent.parent / "cache" / "avp_remaining_PerFrame"

QWEN_URL = "http://saltyfish.eecs.umich.edu:8000/v1/chat/completions"
QWEN_MODEL_DEFAULT = "Qwen/Qwen3-VL-30B-A3B-Instruct"

VLLM_ENDPOINTS = {
    "qwen": ("http://saltyfish.eecs.umich.edu:8000/v1/chat/completions", "Qwen/Qwen3-VL-30B-A3B-Instruct"),
    "gemma": ("http://saltyfish.eecs.umich.edu:8000/v1/chat/completions", "google/gemma-4-31B-it"),
}

OUTPUT_PREFIX = "avp_PerFrame_remaining"


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
# Per-frame data loading (bypasses 05b)
# ---------------------------------------------------------------------------

def load_hoi_timestamps(participant: str, session: str) -> tuple[list[float], set[float]]:
    """Return (sorted_all_timestamps, set_of_hoi_contact_timestamps)."""
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


def load_siglip_by_t(participant: str, session: str) -> dict[float, dict[str, float]]:
    """timestamp -> {visual_class: max_similarity}."""
    path = hands23_dir(participant, session) / f"{participant}_{session}_siglip_matches.json"
    if not path.exists():
        return {}
    d = json.loads(path.read_text())
    out: dict[float, dict[str, float]] = defaultdict(dict)
    for v in d["videos"]:
        for m in v["matches"]:
            t = round(m["timestamp"], 2)
            for tm in m.get("top_matches") or []:
                name = tm["food_name"]
                sim = float(tm["similarity"])
                out[t][name] = max(out[t].get(name, 0.0), sim)
    return dict(out)


def load_dino_by_t(participant: str, session: str) -> dict[float, dict[str, float]]:
    """timestamp -> {instance_id: max_similarity}."""
    path = hands23_dir(participant, session) / f"{participant}_{session}_dino_matches.json"
    if not path.exists():
        return {}
    d = json.loads(path.read_text())
    out: dict[float, dict[str, float]] = defaultdict(dict)
    for v in d["videos"]:
        for m in v["matches"]:
            t = round(m["timestamp"], 2)
            for tm in m.get("top_matches") or []:
                iid = tm["instance_id"]
                sim = float(tm["similarity"])
                out[t][iid] = max(out[t].get(iid, 0.0), sim)
    return dict(out)


def load_owlv2_scene_by_t(participant: str, session: str) -> dict[float, str]:
    """timestamp -> scene_tag (one of storage|sink|stove|unknown). Empty if missing."""
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
    """Load confusable_profile.json; return {instance_id: is_transparent_package}."""
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
# Per-frame prompt formatter
# ---------------------------------------------------------------------------

def format_per_frame_evidence(
    hoi_ts_sorted: list[float],
    siglip_by_t: dict[float, dict[str, float]],
    dino_by_t: dict[float, dict[str, float]],
    scene_by_t: dict[float, str],
    inventory: list[dict],
    min_score: float = 0.15,
) -> tuple[str, dict]:
    """Render one line per HOI frame:

        [   12.0s]  stove      iid_a=0.42  iid_b=0.31:sig=0.18

    Only lines with at least one score >= min_score are emitted.  If SigLIP
    fires for a visual_class, the score is appended as `:sig=<value>` to the
    matching DINO iid entry (or shown as a bare `vc:sig=<value>` token when no
    DINO hit).
    """
    inv_iids = {inv["instance_id"] for inv in inventory}
    inv_vcs = {inv["visual_class"] for inv in inventory}
    vc_to_iids: dict[str, list[str]] = defaultdict(list)
    for inv in inventory:
        vc_to_iids[inv["visual_class"]].append(inv["instance_id"])

    lines: list[str] = []
    n_lines_with_hit = 0
    n_dino_tokens = 0
    n_sig_tokens = 0
    for t in hoi_ts_sorted:
        dino_hits: dict[str, float] = {
            iid: s for iid, s in (dino_by_t.get(t) or {}).items()
            if iid in inv_iids and s >= min_score
        }
        sig_hits: dict[str, float] = {
            vc: s for vc, s in (siglip_by_t.get(t) or {}).items()
            if vc in inv_vcs and s >= min_score
        }
        if not dino_hits and not sig_hits:
            continue

        scene = scene_by_t.get(t, "unknown") or "unknown"

        tokens: list[str] = []
        emitted_sig_for_vc: set[str] = set()
        for iid, ds in sorted(dino_hits.items(), key=lambda x: -x[1]):
            # If any SigLIP hit's visual_class maps to this iid, fuse the score.
            attached_sig = None
            for vc, iids in vc_to_iids.items():
                if iid in iids and vc in sig_hits and vc not in emitted_sig_for_vc:
                    attached_sig = sig_hits[vc]
                    emitted_sig_for_vc.add(vc)
                    break
            if attached_sig is not None:
                tokens.append(f"{iid}={ds:.2f}:sig={attached_sig:.2f}")
                n_sig_tokens += 1
            else:
                tokens.append(f"{iid}={ds:.2f}")
            n_dino_tokens += 1

        # Emit any SigLIP-only hits whose visual_class didn't attach to a DINO iid.
        for vc, ss in sorted(sig_hits.items(), key=lambda x: -x[1]):
            if vc in emitted_sig_for_vc:
                continue
            tokens.append(f"{vc}:sig={ss:.2f}")
            n_sig_tokens += 1

        lines.append(f"[{t:7.1f}s]  {scene:<8} {'  '.join(tokens)}")
        n_lines_with_hit += 1

    stats = {
        "n_hoi_frames_total": len(hoi_ts_sorted),
        "n_frames_emitted": n_lines_with_hit,
        "n_dino_tokens": n_dino_tokens,
        "n_sig_tokens": n_sig_tokens,
        "min_score": min_score,
    }
    return "\n".join(lines), stats


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
# Step 1: Planner prompt
# ---------------------------------------------------------------------------

PLANNER_PROMPT = """\
You are an expert kitchen activity analyst.  For every inventory item that \
shows clear evidence in the per-frame detections below, decide whether a \
vision model should observe it and, if so, pick short observation windows \
that let the observer estimate the **remaining amount** in the stock \
container at the end of the session.

## What you are looking at

Each line is one hand-object-contact frame.  Format:

    [timestamp]  scene      iid=dino_score[:sig=siglip_score] ...

- **timestamp** — seconds from session start.
- **scene** — OWLv2 tag, one of `storage` (fridge / cabinet), `sink`, \
`stove`, or `unknown` (counter-side prep or unmatched).
- **iid=score** — DINO image-to-image similarity to that instance's product \
reference photo.  Usually the strongest signal.
- **:sig=score** — SigLIP text-to-image similarity for the item's \
visual_class (appended when available).

Only scores ≥ {min_score} are printed.  Multiple items on one line means \
they're co-visible in that HOI crop (or cross-talk between similar-looking \
items).

## Signals to weigh

- **Scene context.**  Items retrieved from `storage`, washed/cut at `sink`, \
and cooked at `stove` form a plausible journey.  An item with only `stove` \
hits and no earlier `storage` retrieval may be confusion with a similar \
item.
- **Transparency tag** on each inventory line: `[transparent]` = fill level \
readable through the package; `[opaque]` = not readable, so remaining amount \
has to be inferred from the dispensing action.

Treat the scene tag and transparency as **clues**, not rules.

## Duplicate instance_ids of the same visual_class

When two or more inventory entries share the same `visual_class` (e.g. two \
purchase instances of "Large White Eggs" or "Whole Milk Gallon"), the \
per-frame evidence usually cannot distinguish them — two sealed cartons of \
the same product score identically against both reference embeddings and \
fire on the same frames.  **Do not refuse observation of all duplicates.** \
Instead treat the group as **one physical package**:
- Emit `observe` for exactly ONE instance_id in the group (the one with the \
earliest trailing `YYYYMMDD` date in the instance_id — FIFO convention, \
since the oldest carton is physically consumed first).
- Emit `no_observation` for the other duplicate(s), with reasoning briefly \
noting that the evidence is indistinguishable from the chosen sibling.
- Use the SAME windows for the chosen one as you would if there were no \
duplication.

## Session Inventory
{inventory}

## Per-Frame Detections (chronological, HOI contact only)
{evidence}

## What to return

For every distinct inventory item that has meaningful evidence in the \
per-frame detections, return one entry.

1. **observe vs. no_observation.**  Default to `observe` when the \
per-frame bursts look like real use.  Pick `no_observation` when the \
evidence plausibly reflects cross-talk (scattered low-score hits with no \
coherent burst).
2. **If observing, pick windows.**  Walk the detections in time order and \
build a short narrative: where did the item come from, how was it used, \
where did it end up.  Select windows that best \
support a remaining-amount estimate for this item — favour windows where \
the stock container itself is likely in view, using the transparency tag \
to decide whether reading the fill level or watching the dispensing action \
will be more informative.
3. Keep windows short.  The observer has a budget of ~50 frames across all \
observed items, so weight the budget toward items and moments where it \
will change the answer.

## Output — JSON only, no other text
```json
{{
  "item_decisions": [
    {{
      "item": "<visual_class exactly as in inventory>",
      "instance_id": "<instance_id from inventory>",
      "decision": "observe" | "no_observation",
      "reasoning": "<one sentence citing what in the per-frame evidence \
(scores, timing, scene, transparency) drove the decision and window choice>",
      "confidence": "high" | "medium" | "low",
      "segments": [[start1, end1], [start2, end2]]
    }}
  ]
}}
```

Rules:
- One entry per distinct inventory item that has any evidence in the \
detections; omit items that never appear.
- Use EXACT `visual_class` and `instance_id` from the inventory.
- `segments` is required for `observe` and must be omitted or empty for \
`no_observation`.
- `confidence` is about the decision itself, not the eventual amount \
estimate.
"""


def _is_refusal(text: str) -> bool:
    low = text.strip().lower()
    return low.startswith("i'm sorry") or low.startswith("i cannot") or "cannot assist" in low


def run_planner(
    client,
    hoi_ts_sorted: list[float],
    siglip_by_t: dict,
    dino_by_t: dict,
    scene_by_t: dict,
    inventory: list[dict],
    model: str,
    min_score: float,
    max_retries: int = 5,
    prompt_save_path: Path | None = None,
    transparency_by_iid: dict | None = None,
) -> tuple[str, str, dict, dict]:
    """Run Step 1: Planner. Returns (response_text, prompt, stats, evidence_stats)."""
    evidence_text, ev_stats = format_per_frame_evidence(
        hoi_ts_sorted, siglip_by_t, dino_by_t, scene_by_t, inventory,
        min_score=min_score,
    )
    inventory_text = format_inventory_for_prompt(inventory, transparency_by_iid)

    prompt = PLANNER_PROMPT.format(
        inventory=inventory_text,
        evidence=evidence_text,
        min_score=f"{min_score:.2f}",
    )

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
            return "", prompt, {"error": str(e), "inference_time_s": round(time.time() - t0, 2)}, ev_stats

        if _is_refusal(response_text) or (
            response_text and "item_decisions" not in response_text
        ):
            filter_info = ""
            try:
                if hasattr(response, "model_dump"):
                    dump = response.model_dump()
                    for key in ("content_filter_results", "incomplete_details", "status"):
                        if key in dump and dump[key]:
                            filter_info += f" {key}={dump[key]}"
            except Exception:
                pass
            print(f"  Planner refusal/truncated (attempt {attempt + 1}), retrying...{filter_info}")
            if attempt < max_retries - 1:
                time.sleep(5 * (attempt + 1))
                continue

        return response_text, prompt, stats, ev_stats

    # Fallback: Qwen
    print(f"  Planner: GPT failed {max_retries} times, falling back to Qwen...")
    import requests
    try:
        t0 = time.time()
        resp = requests.post(QWEN_URL, json={
            "model": QWEN_MODEL_DEFAULT,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 16384,
            "temperature": 0.3,
        }, timeout=600)
        resp.raise_for_status()
        result = resp.json()
        response_text = result["choices"][0]["message"]["content"]
        usage = result.get("usage", {})
        stats = {
            "inference_time_s": round(time.time() - t0, 2),
            "model": f"qwen_fallback:{QWEN_MODEL_DEFAULT}",
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        }
        return response_text, prompt, stats, ev_stats
    except Exception as e:
        print(f"  Qwen fallback also failed: {e}")
        return "", prompt, {"error": f"all backends failed: {e}", "inference_time_s": 0}, ev_stats


def parse_planner_response(response_text: str) -> tuple[list[dict], list[dict]]:
    """Parse planner JSON. Returns (item_decisions, observation_plan)."""
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

    observation_plan = []
    for d in decisions:
        if not isinstance(d, dict):
            continue
        if d.get("decision") != "observe":
            continue
        segs = d.get("segments") or []
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


# ---------------------------------------------------------------------------
# Step 2: Observer (unchanged remaining-only prompt, copied from SceneTransp)
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
container/package at the end of the session.
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


def run_observer(
    client,
    frames_b64: list[str],
    timestamps: list[float],
    item_info: dict,
    plan_entry: dict,
    model: str,
    prompt_save_path: Path | None = None,
) -> tuple[str, str, dict]:
    unit_label = "grams" if item_info["unit"] == "g" else "count"

    seg_descs = []
    for s in plan_entry.get("segments", []):
        seg_descs.append(f"- {s[0]:.1f}–{s[1]:.1f}s")
    if plan_entry.get("reasoning"):
        seg_descs.append(f"Planner note: {plan_entry['reasoning']}")

    frame_ts_str = ", ".join(f"{t:.1f}s" for t in timestamps)
    item_context = f"- Package capacity: {item_info['package_amount']}"

    prompt = OBSERVER_PROMPT.format(
        n_frames=len(frames_b64),
        frame_timestamps=frame_ts_str,
        item_name=item_info["visual_class"],
        unit_label=unit_label,
        item_context_line=item_context,
        segment_descriptions="\n".join(seg_descs),
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

    unit_label = "grams" if item_info["unit"] == "g" else "count"

    seg_descs = []
    for s in plan_entry.get("segments", []):
        seg_descs.append(f"- {s[0]:.1f}–{s[1]:.1f}s")
    if plan_entry.get("reasoning"):
        seg_descs.append(f"Planner note: {plan_entry['reasoning']}")

    item_context = f"- Package capacity: {item_info['package_amount']}"

    frame_ts_str = ", ".join(f"{t:.1f}s" for t in timestamps)
    prompt = OBSERVER_PROMPT.format(
        n_frames=len(frames_b64),
        frame_timestamps=frame_ts_str,
        item_name=item_info["visual_class"],
        unit_label=unit_label,
        item_context_line=item_context,
        segment_descriptions="\n".join(seg_descs),
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
# Session pipeline
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
    planner_only: bool = False,
    observer_only: bool = False,
    verbose: bool = False,
    model_tag: str = "default",
    run_tag: str = "default",
    inventory_scope: str = "full",
    transparency_by_iid: dict | None = None,
) -> tuple[list[dict], dict]:
    session_log: dict = {"session": session, "planner": {}, "observer": []}

    cache_dir = CACHE_DIR / participant / session / model_tag / run_tag
    cache_dir.mkdir(parents=True, exist_ok=True)

    inventory = load_inventory(participant, session, scope=inventory_scope)
    if not inventory:
        print(f"  {session}: no inventory for scope={inventory_scope}")
        return [], session_log

    # Load raw per-frame signals directly (no 05b).
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

    hoi_sorted = sorted(hoi_ts)
    print(f"  {session}: {len(all_ts)} frames, {len(hoi_sorted)} HOI-contact, "
          f"{len(inventory)} inventory items ({inventory_scope} scope), "
          f"{len(scene_by_t)} OWLv2 scene tags")

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
        observation_plan = saved_planner.get("observation_plan") or []
        session_log["planner"] = saved_planner
        print(f"  Step 1 (Planner): LOADED {len(observation_plan)} items "
              f"from saved plan (observer-only mode)")
        planner_stats = saved_planner.get("stats", {})
    else:
        print(f"  Step 1 (Planner): sending per-frame evidence to {planner_model}...")
        planner_text, planner_prompt, planner_stats, ev_stats = run_planner(
            client, hoi_sorted, siglip_by_t, dino_by_t, scene_by_t,
            inventory, planner_model,
            min_score=min_score,
            prompt_save_path=cache_dir / "planner_prompt.txt",
            transparency_by_iid=transparency_by_iid,
        )
        (cache_dir / "planner_response.txt").write_text(planner_text or "")
        item_decisions, observation_plan = parse_planner_response(planner_text)
        n_observe = sum(1 for d in item_decisions if d.get("decision") == "observe")
        n_skip = sum(1 for d in item_decisions if d.get("decision") == "no_observation")
        print(f"  Evidence: {ev_stats['n_frames_emitted']}/{ev_stats['n_hoi_frames_total']} "
              f"HOI frames had ≥{min_score} hit ({ev_stats['n_dino_tokens']} dino tokens, "
              f"{ev_stats['n_sig_tokens']} sig tokens)")
        print(f"  Planner decisions: {len(item_decisions)} total — "
              f"{n_observe} observe, {n_skip} no_observation")
        if n_skip:
            for d in item_decisions:
                if d.get("decision") == "no_observation":
                    print(f"    SKIP {d.get('item')}: {d.get('reasoning', '')[:120]}")
        session_log["planner"] = {
            "n_hoi_frames": len(hoi_sorted),
            "evidence_stats": ev_stats,
            "n_items_planned": len(observation_plan),
            "n_decisions_total": len(item_decisions),
            "n_decisions_observe": n_observe,
            "n_decisions_no_observation": n_skip,
            "item_decisions": item_decisions,
            "observation_plan": observation_plan,
            "stats": planner_stats,
            "prompt": planner_prompt,
            "raw_response": planner_text,
        }

    if not observation_plan:
        print(f"  Planner returned empty plan")
        return [], session_log

    inv_by_iid = {inv["instance_id"]: inv for inv in inventory}
    valid_plan = []
    for entry in observation_plan:
        iid = entry.get("instance_id", "")
        if iid not in inv_by_iid:
            if verbose:
                print(f"    SKIP: planner proposed '{iid}' not in inventory")
            continue
        valid_plan.append(entry)

    items_str = ", ".join(
        f"{e.get('item', '?')}({e.get('confidence', '?')})" for e in valid_plan
    )
    print(f"  Planner identified {len(valid_plan)} items: {items_str}")

    if planner_only:
        return [], session_log

    clips = get_session_clips(participant, session)
    if not clips:
        print(f"  {session}: no video clips found")
        return [], session_log
    video_durations = [(path, dur) for _, path, dur in clips]

    predictions = []
    for entry in valid_plan:
        iid = entry["instance_id"]
        inv = inv_by_iid[iid]

        segs = entry.get("segments", [])
        if not segs:
            continue
        seg_str = "+".join(f"{s[0]:.0f}-{s[1]:.0f}" for s in segs)
        print(f"    Observer: {inv['visual_class']} [{seg_str}s]...", end="", flush=True)

        vllm_endpoint = VLLM_ENDPOINTS.get(model.lower())
        use_vllm = vllm_endpoint is not None or model.lower().startswith("qwen") or model.lower().startswith("gemma")

        frames, frame_ts = extract_segments_frames(segs, video_durations, padding=2.0, fps=fps, max_frames=max_frames)
        if not frames:
            print(f" no frames")
            continue

        obs_prompt_path = cache_dir / f"{iid}_observer_prompt.txt"
        if use_vllm:
            if vllm_endpoint:
                obs_url, obs_model = vllm_endpoint
            else:
                obs_url, obs_model = QWEN_URL, model
            obs_text, obs_prompt, obs_stats = run_observer_qwen(
                frames, frame_ts, inv, entry,
                qwen_url=obs_url,
                qwen_model=obs_model,
                prompt_save_path=obs_prompt_path,
            )
        else:
            obs_text, obs_prompt, obs_stats = run_observer(
                client, frames, frame_ts, inv, entry, model,
                prompt_save_path=obs_prompt_path,
            )
        (cache_dir / f"{iid}_observer_response.txt").write_text(obs_text or "")
        parsed = parse_observer_response(obs_text)

        obs_log = {
            "instance_id": iid,
            "visual_class": inv["visual_class"],
            "n_frames": len(frame_ts) if frame_ts else 0,
            "item_confirmed": parsed["item_confirmed"],
            "reasoning": parsed.get("reasoning", ""),
            "stats": obs_stats,
            "raw_response": obs_text,
        }
        session_log["observer"].append(obs_log)

        if parsed["item_confirmed"] and parsed["amount_remaining"] is not None:
            print(f" remaining={parsed['amount_remaining']}")
            predictions.append({
                "session": session,
                "item": inv["visual_class"],
                "instance_id": iid,
                "amount_remaining": parsed["amount_remaining"],
                "reasoning": parsed["reasoning"],
                "evidence_frames": parsed.get("evidence_frames", []),
                "planner_reasoning": entry.get("reasoning", ""),
                "planner_confidence": entry.get("confidence", ""),
                "segments": entry.get("segments", []),
                "stats": {
                    "planner": planner_stats,
                    "observer": obs_stats,
                },
            })
        else:
            if _is_refusal(obs_text):
                print(f" REFUSAL (all retries exhausted)")
            elif not parsed["item_confirmed"]:
                print(f" not confirmed: {parsed.get('reasoning', '')[:80]}")
            else:
                print(f" no amount estimated")

    print(f"  {session}: {len(predictions)} predictions from {len(valid_plan)} planned items")
    return predictions, session_log


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="AVP Round 1 (remaining-only, PerFrame): agentic remaining-"
                    "amount estimation with raw per-frame HOI + DINO/SigLIP + "
                    "OWLv2 scene evidence as planner context (no 05b)."
    )
    parser.add_argument("--participant", default="kailai")
    parser.add_argument("--tag", required=True,
                        help="Short label for this run (e.g. 'PerFrame_v1').")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--session", help="Single session")
    group.add_argument("--all", action="store_true", help="All sessions")
    parser.add_argument("--model", default="gpt-5.4", help="Observer model (planner always uses gpt-5.4)")
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--max-frames", type=int, default=MAX_IMAGES)
    parser.add_argument("--min-score", type=float, default=0.15,
                        help="Only include per-frame detections with DINO or SigLIP >= this threshold.")
    parser.add_argument("--planner-only", action="store_true")
    parser.add_argument("--observer-only", action="store_true",
                        help="Skip planner; load saved plan (same --tag) and run observer only.")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", action="store_true",
                        help="Resume a previously-interrupted run with the same --tag.")
    parser.add_argument("--inventory-scope", choices=["full", "session"], default="full",
                        help="Which items to list in the planner's inventory.")
    args = parser.parse_args()

    sessions = (
        [args.session] if args.session
        else get_sessions(args.participant) if args.all
        else get_sessions(args.participant)
    )

    model_tag = args.model.replace("/", "_")
    run_tag = args.tag.replace("/", "_")

    print(f"{'=' * 70}")
    print(f"AVP Round 1 (Remaining-Only, PerFrame)")
    print(f"{'=' * 70}")
    print(f"Participant:  {args.participant}")
    print(f"Model:        {args.model}")
    print(f"Tag:          {run_tag}")
    print(f"FPS:          {args.fps}")
    print(f"Max frames:   {args.max_frames}")
    print(f"Min score:    {args.min_score}")
    print(f"Planner only: {args.planner_only}")
    print(f"Observer only:{args.observer_only}")
    if args.planner_only and args.observer_only:
        parser.error("--planner-only and --observer-only are mutually exclusive.")
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
            )
        except KeyboardInterrupt:
            raise
        except Exception as e:
            import traceback
            print(f"\n  ERROR in session {session}: {e}")
            traceback.print_exc()
            failed_sessions.append((session, str(e)[:200]))
            preds, log = [], {"session": session, "planner": {}, "observer": [], "error": str(e)[:500]}
            session_failed = True
            status["failed_sessions"] = [
                f for f in status["failed_sessions"] if f.get("session") != session
            ] + [{"session": session, "error": str(e)[:200]}]
            status_path.write_text(json.dumps(status, indent=2) + "\n")
            all_logs.append(log)
            if not args.planner_only:
                with open(preds_path, "w", encoding="utf-8") as f:
                    json.dump(all_predictions, f, indent=2)
            print(f"\n  HALTED at session {session}. Fix the issue and re-run with --resume.")
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
    planner_output = {
        "participant": args.participant,
        "timestamp": datetime.now().isoformat(),
        "model": args.model,
        "tag": run_tag,
        "sessions": all_logs,
    }
    with open(planner_path, "w", encoding="utf-8") as f:
        json.dump(planner_output, f, indent=2)
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
                  f"({p['planner_confidence']})")

    if failed_sessions:
        print(f"\n{len(failed_sessions)} session(s) failed in this run:")
        for s, err in failed_sessions:
            print(f"  {s}: {err}")
        print(f"\nRe-run with --resume to retry the failed sessions only.")


if __name__ == "__main__":
    main()

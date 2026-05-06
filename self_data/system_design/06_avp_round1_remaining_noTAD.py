#!/usr/bin/env python3
"""AVP Round 1 (remaining-only, noTAD): planner reads per-item temporal
episodes from HOI + SigLIP + DINO aggregation instead of AdaTAD.

Same two-step pipeline as 06_avp_round1_remaining.py:
  Step 1 (Planner): Text-only — identifies which items were actually used and
    selects observation windows. Context is a chronological list of per-item
    candidate episodes (no verb labels, no scene tags). Source:
    per_item_segments.json, produced by 05b_per_item_segments.py --write.
  Step 2 (Observer): VLM with frames — asks ONLY for remaining amount at the
    end of the session. Unchanged.

Usage:
  python 06_avp_round1_remaining_noTAD.py --participant kailai --tag noTAD_v1
  python 06_avp_round1_remaining_noTAD.py --participant kailai --session 20260310-195710 --tag noTAD_v1
"""

import argparse
import json
import os
import re
import sys
import time
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
    load_inventory,
    load_ledger,
    outputs_dir,
    participant_dir,
)

_KITCHEN_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(_KITCHEN_DIR / ".env")

MAX_IMAGES = 50

CACHE_DIR = Path(__file__).resolve().parent.parent / "cache" / "avp_remaining_noTAD"

QWEN_URL = "http://saltyfish.eecs.umich.edu:8000/v1/chat/completions"
QWEN_MODEL_DEFAULT = "Qwen/Qwen3-VL-30B-A3B-Instruct"

VLLM_ENDPOINTS = {
    "qwen": ("http://saltyfish.eecs.umich.edu:8000/v1/chat/completions", "Qwen/Qwen3-VL-30B-A3B-Instruct"),
    "gemma": ("http://saltyfish.eecs.umich.edu:8000/v1/chat/completions", "google/gemma-4-31B-it"),
}


# ---------------------------------------------------------------------------
# Clients (reused)
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
# Data loading
# ---------------------------------------------------------------------------

def load_per_item_segments_file(participant: str, session: str) -> dict | None:
    """Load per_item_segments.json produced by 05b_per_item_segments.py."""
    path = outputs_dir(participant, session) / "per_item_segments.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def get_video_durations(session_clips: list) -> list[tuple[Path, float]]:
    return [(path, dur) for _, path, dur in session_clips]


# ---------------------------------------------------------------------------
# Frame extraction adapter
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Step 1: Planner (text-only) — per-item episode timeline, no verb, no scene.
# ---------------------------------------------------------------------------

PLANNER_PROMPT = """\
You are an expert kitchen activity analyst. You are given an **item-centric \
timeline** of candidate interaction episodes extracted from an egocentric \
kitchen video. Each line is an episode for a SINGLE inventory item, derived \
from per-frame hand-object-interaction (HOI) detections combined with two \
visual matching signals:

- **siglip**: zero-shot text-to-image similarity against the item's name. \
Works for generic foods (rice, vegetables, garlic, oil, sauce) that have \
descriptive text labels.
- **dino**: image-to-image similarity against the item's reference product \
photo. Generally more reliable than siglip, especially for branded packages. Dino can still fire on \
visually generic items — e.g. a meat reference image matching pink/marbled \
surfaces during cooking.
- An episode is listed whenever, during a hand-object contact event, at least \
one score crossed its threshold and stayed active across a short window.


## Session Inventory
{inventory}

## Detected Item Episodes (chronological)
Each line: `[start–end] item — siglip=X dino=Y`

- Episodes are sorted by start time.
- Multiple items can share overlapping time ranges. Usually this means several \
items are in view together (cutting board, cooking surface, side-by-side in \
fridge). It can also reflect visual cross-talk (one crop scores for two \
similar-looking items). Treat overlaps as co-occurrence, not as duplicates.

{segments}

## Task

Your job is to decide, for every inventory item that appears in the **Detected \
Item Episodes** section above, whether a vision model should observe it to \
estimate its remaining amount after this session.

You must return **one decision entry for every distinct item that appears at \
least once in the Detected Item Episodes list.** Items that never appear in \
the episodes list are NOT in the candidate set — omit them.

**Duplicate instance_ids of the same visual_class:** when two or more \
inventory entries share the same `visual_class` (e.g. two purchase \
instances of "Large White Eggs" or "Whole Milk Gallon"), the episode \
evidence usually cannot distinguish them — two sealed cartons of the same \
product score identically. Do NOT refuse observation of all duplicates. \
Treat the group as one physical package: emit `observe` for exactly ONE \
instance_id (the earliest trailing `YYYYMMDD` — FIFO, oldest carton \
consumed first), and `no_observation` for the others with reasoning \
citing the duplication.

## Step 1 — decide `observe` or `no_observation` for each detected item

Your default decision is `observe`. Choose `no_observation` only when you can \
articulate a specific reason the episodes do not reflect real use of the item.

Think about each item by walking its episodes in the order they appear:

- If the item has multiple episodes over time, or any episode where siglip \
and dino fire together, it is almost certainly a real interaction — \
`observe`.
- If the item has a single brief episode with modest scores, it may still be \
a real quick interaction (staples like oil, sauce, eggs are handled briefly). \
Default to `observe` unless you have a concrete reason to reject it.
- `no_observation` is appropriate when the episodes plausibly result from \
visual cross-talk — e.g. a meat-reference image firing weakly on unrelated \
cooking frames, a generic bottle reference matching a different container on \
the counter. Typical pattern: multiple short, dino-only episodes scattered \
across the cooking phase without clustering, with no siglip support.
- You are not required to skip any item. Returning `observe` for every \
detected item is acceptable if the evidence supports it.

Write one sentence of `reasoning` for every decision, explaining what in the \
episodes drove your choice.

## Step 2 — for each `observe` item, reconstruct its journey

Walk the item's episodes in time order. Most items follow an arc:

1. **Retrieval** — the first episode(s). Container taken out of storage. \
Starting fill level is likely visible.
2. **Use** — middle episode(s). The item is handled, poured, cut, scooped. \
Content is being dispensed.
3. **Put-back / set-down** — the last episode(s). Container placed back or \
left out. Final fill level is often visible here.

Because there are no verb labels and no scene labels, you must infer stage \
purely from **position within the item's own episode sequence** (first vs. \
middle vs. last). The first and last episodes are usually the most \
informative for remaining-amount estimation.

## Step 3 — for each `observe` item, pick observation windows

Return the short segments spanning the item's journey so the observer can \
piece together the story.

- Include at least one **early** episode when it exists (retrieval / first \
access) — this often shows the starting package best.
- Include the **most informative middle episodes** (especially ones where \
both siglip and dino fire, or ones with the highest scores for this item).
- Include a **final** episode if one is present — often the clearest view of \
what remains.
- For items handled only briefly, 1–2 short episodes is enough.
- For items with a long cluster (many episodes across minutes), **3–5 short \
windows** covering first episode → a couple of middle moments → last episode \
is appropriate.
- Prefer short windows (roughly 2–5 seconds). The observer will see them \
concatenated, so several short clips beat one long clip.

**Frame budget:** the observer has ~50 frames across ALL observed items. \
Keep each window short so the budget stretches across the items that matter.

## Output Format

Return ONLY JSON (no other text):
```json
{{
  "item_decisions": [
    {{
      "item": "<visual_class exactly as in inventory>",
      "instance_id": "<instance_id from inventory>",
      "decision": "observe" | "no_observation",
      "reasoning": "<one sentence: what in the episodes drove this decision>",
      "confidence": "high" | "medium" | "low",
      "segments": [[start1, end1], [start2, end2]]
    }}
  ]
}}
```

Rules:
- You MUST return an entry for every distinct item that appears in the \
Detected Item Episodes list. Omit items that never appear.
- Use EXACT visual_class and instance_id from the inventory list.
- `segments` is REQUIRED when `decision == "observe"` and MUST be omitted or \
empty when `decision == "no_observation"`.
- Multiple short segments (2–5s each) are preferred over one long segment.
- Do NOT bias observation windows toward the last episode only — earlier \
episodes often show the package and starting fill level better than \
cooking-side episodes.
- `confidence` reflects how certain you are in the decision itself (not in \
the eventual amount estimate).
"""


def format_per_item_segments_for_planner(
    peritem_data: dict,
    inventory: list[dict],
    min_score: float = 0.15,
) -> str:
    """Render per-item episode timeline as chronological text for the planner."""
    inv_ids = {inv["instance_id"] for inv in inventory}

    lines = []
    for seg in peritem_data.get("segments", []):
        if seg.get("instance_id") not in inv_ids:
            continue
        bits = []
        peak_siglip = float(seg.get("peak_siglip", 0.0) or 0.0)
        peak_dino = float(seg.get("peak_dino", 0.0) or 0.0)
        if peak_siglip >= min_score:
            bits.append(f"siglip={peak_siglip:.2f}")
        if peak_dino >= min_score:
            bits.append(f"dino={peak_dino:.2f}")
        if not bits:
            continue

        scores_str = ", ".join(bits)
        lines.append(
            f"[{seg['start']:.1f}–{seg['end']:.1f}s] {seg['visual_class']}"
            f" — {scores_str}"
        )
    return "\n".join(lines)


def format_inventory_for_prompt(inventory: list[dict]) -> str:
    lines = []
    for inv in inventory:
        unit_label = "grams" if inv["unit"] == "g" else "count"
        visible = inv.get("visible_during_interaction", True)
        vis_tag = "visible" if visible else "opaque"
        lines.append(
            f"- {inv['instance_id']}: \"{inv['visual_class']}\" "
            f"({unit_label}, package={inv['package_amount']}, {vis_tag})"
        )
    return "\n".join(lines)


def _is_refusal(text: str) -> bool:
    low = text.strip().lower()
    return low.startswith("i'm sorry") or low.startswith("i cannot") or "cannot assist" in low


def run_planner(
    client, peritem_data: dict, inventory: list[dict], model: str,
    max_retries: int = 5, safe_mode: bool = False,
    prompt_save_path: Path | None = None,
) -> tuple[str, str, dict]:
    """Run Step 1: Planner. Returns (response_text, prompt, stats)."""
    segments_text = format_per_item_segments_for_planner(peritem_data, inventory)
    inventory_text = format_inventory_for_prompt(inventory)

    prompt = PLANNER_PROMPT.format(
        inventory=inventory_text,
        segments=segments_text,
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
            return "", prompt, {"error": str(e), "inference_time_s": round(time.time() - t0, 2)}

        if _is_refusal(response_text) or (
            response_text and "item_decisions" not in response_text
        ):
            filter_info = ""
            try:
                if hasattr(response, 'model_dump'):
                    dump = response.model_dump()
                    for key in ('content_filter_results', 'incomplete_details', 'status'):
                        if key in dump and dump[key]:
                            filter_info += f" {key}={dump[key]}"
            except Exception:
                pass
            print(f"  Planner refusal/truncated (attempt {attempt + 1}), retrying...{filter_info}")
            if attempt < max_retries - 1:
                time.sleep(5 * (attempt + 1))
                continue

        return response_text, prompt, stats

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
        return response_text, prompt, stats
    except Exception as e:
        print(f"  Qwen fallback also failed: {e}")
        return "", prompt, {"error": f"all backends failed: {e}", "inference_time_s": 0}


def parse_planner_response(response_text: str) -> tuple[list[dict], list[dict]]:
    """Parse planner JSON. Returns (item_decisions, observation_plan).

    item_decisions contains every entry emitted by the planner (observe +
    no_observation). observation_plan is the subset with decision=="observe",
    shaped like the older schema so the observer loop consumes it unchanged.
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
# Step 2: Observer (unchanged — remaining-only prompt)
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

OUTPUT_PREFIX = "avp_noTAD_remaining"


def process_session(
    participant: str,
    session: str,
    client,
    ledger: dict,
    model: str,
    fps: float,
    max_frames: int,
    planner_only: bool = False,
    observer_only: bool = False,
    verbose: bool = False,
    model_tag: str = "default",
    run_tag: str = "default",
    inventory_scope: str = "full",
) -> tuple[list[dict], dict]:
    session_log: dict = {"session": session, "planner": {}, "observer": []}

    cache_dir = CACHE_DIR / participant / session / model_tag / run_tag
    cache_dir.mkdir(parents=True, exist_ok=True)

    peritem_data = load_per_item_segments_file(participant, session)
    if not peritem_data:
        print(f"  {session}: no per_item_segments.json "
              f"(run 05b_per_item_segments.py --write first)")
        return [], session_log

    inventory = load_inventory(participant, session, scope=inventory_scope)
    if not inventory:
        print(f"  {session}: no inventory for scope={inventory_scope}")
        return [], session_log

    n_segs = len(peritem_data.get("segments", []))
    print(f"  {session}: {n_segs} per-item episodes, "
          f"{len(inventory)} inventory items ({inventory_scope} scope)")

    # ── Step 1: Planner ──
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
        print(f"  Step 1 (Planner): sending {n_segs} episodes to {planner_model}...")
        planner_text, planner_prompt, planner_stats = run_planner(
            client, peritem_data, inventory, planner_model,
            prompt_save_path=cache_dir / "planner_prompt.txt",
        )
        (cache_dir / "planner_response.txt").write_text(planner_text or "")
        item_decisions, observation_plan = parse_planner_response(planner_text)
        n_observe = sum(1 for d in item_decisions if d.get("decision") == "observe")
        n_skip = sum(1 for d in item_decisions if d.get("decision") == "no_observation")
        print(f"  Planner decisions: {len(item_decisions)} total — "
              f"{n_observe} observe, {n_skip} no_observation")
        if n_skip:
            for d in item_decisions:
                if d.get("decision") == "no_observation":
                    print(f"    SKIP {d.get('item')}: {d.get('reasoning', '')[:120]}")
        session_log["planner"] = {
            "n_segments_input": n_segs,
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

    # ── Step 2: Observer ──
    clips = get_session_clips(participant, session)
    if not clips:
        print(f"  {session}: no video clips found")
        return [], session_log
    video_durations = get_video_durations(clips)

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
        description="AVP Round 1 (remaining-only, noTAD): "
                    "agentic food remaining-amount estimation with per-item "
                    "HOI+SigLIP+DINO episodes as planner context (no AdaTAD)."
    )
    parser.add_argument("--participant", default="kailai")
    parser.add_argument("--tag", required=True,
                        help="Short label for this run (e.g. 'noTAD_v1').")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--session", help="Single session")
    group.add_argument("--all", action="store_true", help="All sessions")
    parser.add_argument("--model", default="gpt-5.4", help="Observer model (planner always uses gpt-5.4)")
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--max-frames", type=int, default=MAX_IMAGES)
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
    args = parser.parse_args()

    sessions = (
        [args.session] if args.session
        else get_sessions(args.participant) if args.all
        else get_sessions(args.participant)
    )

    model_tag = args.model.replace("/", "_")
    run_tag = args.tag.replace("/", "_")

    print(f"{'=' * 70}")
    print(f"AVP Round 1 (Remaining-Only, noTAD)")
    print(f"{'=' * 70}")
    print(f"Participant:  {args.participant}")
    print(f"Model:        {args.model}")
    print(f"Tag:          {run_tag}")
    print(f"FPS:          {args.fps}")
    print(f"Max frames:   {args.max_frames}")
    print(f"Planner only: {args.planner_only}")
    print(f"Observer only:{args.observer_only}")
    if args.planner_only and args.observer_only:
        parser.error("--planner-only and --observer-only are mutually exclusive.")
    print(f"Sessions:     {len(sessions)}")
    print()

    client = make_client()
    ledger = load_ledger(args.participant)

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
                planner_only=args.planner_only,
                observer_only=args.observer_only,
                verbose=args.verbose,
                model_tag=model_tag,
                run_tag=run_tag,
                inventory_scope=args.inventory_scope,
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

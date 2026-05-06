#!/usr/bin/env python3
"""AVP Round 1: Agentic pipeline for food amount estimation.

Two-step pipeline using GPT 5.4 (Azure OpenAI):
  Step 1 (Planner): Text-only — reasons about AdaTAD+DINOv2 segments + inventory,
    groups overlapping detections, filters noise, outputs refined observation plan.
  Step 2 (Observer): VLM with frames — for each planned item, extracts frames from
    the observation window and estimates amount_used + amount_remaining.

Prerequisites:
  - AdaTAD item labels: participants/{P}/outputs/adatad_item_labels.json
    (from 05a_adatad_item_label.py)
  - Ledger with snapshots: participants/{P}/ledger.json

Usage:
  python 06_avp_round1.py --participant kailai --session 20260310-195710
  python 06_avp_round1.py --participant kailai --all
  python 06_avp_round1.py --participant kailai --planner-only
"""

import argparse
import base64
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
    cumulative_to_video_offset,
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

MAX_IMAGES = 50  # Azure hard limit per request

# Per-(model, run) cache for raw prompts and responses. The structure is
# cache/avp/{participant}/{session}/{model_tag}/{run_tag}/
#   planner_prompt.txt              ← saved before the planner API call
#   planner_response.txt            ← saved after the planner returns
#   {iid}_observer_prompt.txt       ← saved before each observer API call
#   {iid}_observer_response.txt     ← saved after each observer returns
# Survives crashes/timeouts and gives the user a forensic record.
CACHE_DIR = Path(__file__).resolve().parent.parent / "cache" / "avp"

QWEN_URL = "http://saltyfish.eecs.umich.edu:8000/v1/chat/completions"
QWEN_MODEL_DEFAULT = "Qwen/Qwen2.5-VL-32B-Instruct"

# Model URL registry: model name → vLLM endpoint
VLLM_ENDPOINTS = {
    "qwen": ("http://saltyfish.eecs.umich.edu:8000/v1/chat/completions", "Qwen/Qwen3-VL-30B-A3B-Instruct"),
    "gemma": ("http://saltyfish.eecs.umich.edu:8000/v1/chat/completions", "google/gemma-4-31B-it"),
}


# ---------------------------------------------------------------------------
# Clients
# ---------------------------------------------------------------------------

def make_client(model: str = "gpt-5.4"):
    """Create the appropriate client for planner (Azure GPT) or observer."""
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

def load_adatad_session(participant: str, session: str) -> dict | None:
    """Load AdaTAD item labels for a specific session."""
    path = participant_dir(participant) / "outputs" / "adatad_item_labels.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    for s in data["sessions"]:
        if s["session"] == session:
            return s
    return None


def get_video_durations(session_clips: list) -> list[tuple[Path, float]]:
    """Build [(video_path, duration)] from get_session_clips() output."""
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
    """Adapter that delegates to the shared frame_sampling.extract_segments_frames.
    Accepts segments as [[start, end], ...] (planner output format).
    `fps` is forwarded as `target_fps` (cap on per-second sampling rate)."""
    return _extract_segments_frames(
        [(s[0], s[1]) for s in segments],
        video_durations,
        padding=padding,
        max_frames=max_frames,
        target_fps=fps,
    )


# ---------------------------------------------------------------------------
# Step 1: Planner (text-only)
# ---------------------------------------------------------------------------

PLANNER_PROMPT = """\
You are an expert kitchen activity analyst. You are given a chronological sequence of \
action segments detected by a temporal action detector in an egocentric kitchen video. \
Each segment has a verb label and optional visual item-matching scores from a separate \
vision model (DINOv2). The scores are noisy — treat them as weak hints, not ground truth.

## Session Inventory
These items are present in the kitchen at the start of this session:
{inventory}

## Detected Action Timeline
Each line: [time range] verb (confidence%) — [item score(source), ...]
Two item identification sources are shown:
- **(visual)**: DINOv2 image-to-image matching against product reference photos. \
Good for branded packages (tofu, juice bottle, cheese bag).
- **(text)**: SigLIP zero-shot text-to-image matching. Good for generic foods \
(rice, soy sauce, olive oil, vegetables).
- Only scores >= 0.10 are shown from either method.
- **(visual+text:X.XX)**: Both methods detected the same item — stronger signal.
- Empty brackets = no match from either method.
- All scores are noisy; consistent appearance across adjacent segments is more \
reliable than any single score.

{segments}

## Task

For each inventory item you believe was actually USED (consumed, cooked with, poured, \
etc.), specify the **minimum video segments** a vision model needs to estimate \
how much was used.

**Segment selection priority — focus on moments where amount change is visible:**
- **BEST: dispensing/transfer moments** — pouring liquid, scooping food out, cracking \
eggs, cutting a portion off. These show the amount leaving the container.
- **GOOD: before/after container state** — the container just before use (shows \
starting fill level) and just after (shows remaining). One frame of each is enough.
- **SKIP: handling/transport** — taking from fridge, opening lid, closing, putting \
away. These confirm the item was used but do NOT help estimate amount. Only include \
if no dispensing moment is detected.

**Keep the total segment duration short.** The vision model has a limited \
frame budget (50 frames) — concentrating frames before and after the dispensing moment gives it \
better temporal resolution where it matters.

Use these signals to identify items:
- **Verb semantics** (strong): "pour" → liquids; "crack" → eggs; "cut"/"slice" → \
solids; "scoop" → semi-solids; "sprinkle"/"shake" → seasonings
- **Visual scores** (weak but aggregatable): same item across multiple adjacent \
segments = stronger signal
- **Temporal patterns**: cooking follows sequences (take → open → pour → close; \
take → cut → put-in; take → peel → slice)
- **Inventory context**: if the verb sequence implies an item (e.g., repeated \
pouring near a stove suggests oil/sauce) but no visual match exists, you may \
still propose it with low confidence

## Output Format

Return ONLY JSON (no other text):
```json
{{
  "observation_plan": [
    {{
      "item": "<visual_class exactly as in inventory>",
      "instance_id": "<instance_id from inventory>",
      "segments": [[start1, end1], [start2, end2]],
      "confidence": "high" | "medium" | "low"{planner_reasoning_field}
    }}
  ]
}}
```

Rules:
- Use EXACT visual_class and instance_id from the inventory list
- Merge all segments for the same item into one entry
- Only include items you believe were actually USED — not merely touched or moved
- Prefer fewer, shorter segments focused on dispensing over long sequences
- If uncertain between two items, pick the more plausible one and note uncertainty
"""


def format_segments_for_planner(
    session_data: dict, inventory: list[dict],
    min_score: float = 0.15,
) -> str:
    """Serialize AdaTAD segments with item identification scores.

    Reads from all_scores/siglip_scores when available,
    otherwise falls back to the items list (always present).

    Format: [time] verb (conf%) — [item score(source), ...]
    Only shows items with score >= min_score.
    """
    inv_ids = {inv["instance_id"] for inv in inventory}
    id_to_name = {inv["instance_id"]: inv["visual_class"] for inv in inventory}
    inv_names = {inv["visual_class"].lower(): inv["visual_class"] for inv in inventory}

    lines = []
    for seg in session_data["segments"]:
        candidates = []

        has_verbose_scores = seg.get("all_scores") or seg.get("siglip_scores")

        if has_verbose_scores:
            # Verbose mode: use detailed per-source scores
            for iid, sc in seg.get("all_scores", {}).items():
                if iid in inv_ids and sc >= min_score:
                    candidates.append((id_to_name.get(iid, iid), sc, "visual"))

            dino_names = {c[0].lower() for c in candidates}
            for food_name, sc in seg.get("siglip_scores", {}).items():
                if sc >= min_score and food_name.lower() in inv_names:
                    display = inv_names[food_name.lower()]
                    if display.lower() not in dino_names:
                        candidates.append((display, sc, "text"))
                    else:
                        for i, (name, dsc, src) in enumerate(candidates):
                            if name.lower() == display.lower() and src == "visual":
                                candidates[i] = (name, dsc, f"visual+text:{sc:.2f}")
                                break
        else:
            # Standard mode: use items list from 05
            for item in seg.get("items", []):
                name = item.get("visual_class") or item.get("food_name", "?")
                sc = item.get("score", 0.0)
                src = item.get("source", "?")
                if sc >= min_score:
                    candidates.append((name, sc, src))

        candidates.sort(key=lambda x: -x[1])

        items_str = ", ".join(
            f"{name} {sc:.2f}({src})" for name, sc, src in candidates
        )

        s, e = seg["segment"]
        lines.append(
            f"[{s:.1f}–{e:.1f}s] {seg['verb']} ({seg['verb_score']:.0%})"
            f" — [{items_str}]"
        )
    return "\n".join(lines)


def format_inventory_for_prompt(inventory: list[dict]) -> str:
    """Serialize session inventory for the planner prompt.

    Only provides W0 (package capacity from receipt), not GT starting amount.
    """
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
    """Check if the response is a content-filter refusal."""
    low = text.strip().lower()
    return low.startswith("i'm sorry") or low.startswith("i cannot") or "cannot assist" in low


def run_planner(
    client, session_data: dict, inventory: list[dict], model: str,
    max_retries: int = 5, safe_mode: bool = False,
    prompt_save_path: Path | None = None,
) -> tuple[str, str, dict]:
    """Run Step 1: Planner. Returns (response_text, prompt, stats).

    Retries on content-filter refusals or empty observation_plan.
    safe_mode: omit reasoning field from output schema to avoid content filter.
    prompt_save_path: if provided, write the raw prompt to disk BEFORE the
    API call so it survives crashes/timeouts/refusals.
    """
    segments_text = format_segments_for_planner(session_data, inventory)
    inventory_text = format_inventory_for_prompt(inventory)

    reasoning_field = ',\n      "reasoning": "<brief: why this item, which segments show amount change>"' if not safe_mode else ""
    prompt = PLANNER_PROMPT.format(
        inventory=inventory_text,
        segments=segments_text,
        planner_reasoning_field=reasoning_field,
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

        # Check for refusal or truncated response
        if _is_refusal(response_text) or (
            response_text and "observation_plan" not in response_text and "segments" not in response_text
        ):
            # Log filter details if available
            filter_info = ""
            try:
                if hasattr(response, 'model_dump'):
                    dump = response.model_dump()
                    # Azure may include content_filter_results or finish_reason
                    for key in ('content_filter_results', 'incomplete_details', 'status'):
                        if key in dump and dump[key]:
                            filter_info += f" {key}={dump[key]}"
            except Exception:
                pass
            print(f"  Planner refusal/truncated (attempt {attempt + 1}), retrying...{filter_info}")
            if attempt < max_retries - 1:
                # After 2 failures, switch to safe_mode (no reasoning in output)
                if attempt == 1 and not safe_mode:
                    print("  Switching to safe_mode (no reasoning output)...")
                    reasoning_field = ""
                    prompt = PLANNER_PROMPT.format(
                        inventory=inventory_text,
                        segments=segments_text,
                        planner_reasoning_field=reasoning_field,
                    )
                time.sleep(5 * (attempt + 1))
                continue

        return response_text, prompt, stats

    # Fallback: try Qwen for planner (text-only, no content filter)
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


def parse_planner_response(response_text: str) -> tuple[list[dict], str]:
    """Extract observation plan and activity summary from planner response.

    Returns (observation_plan, activity_summary).
    Handles both new format {activity_summary, observation_plan} and
    legacy format (bare JSON array).
    """
    activity_summary = ""

    # Try markdown code fence with JSON object
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response_text, re.DOTALL)
    if fence:
        try:
            parsed = json.loads(fence.group(1))
            if isinstance(parsed, dict) and "observation_plan" in parsed:
                activity_summary = parsed.get("activity_summary", "")
                return parsed["observation_plan"], activity_summary
        except json.JSONDecodeError:
            pass

    # Try raw JSON object
    # Find outermost { ... } that contains "observation_plan"
    obj_match = re.search(r"\{.*\"observation_plan\".*\}", response_text, re.DOTALL)
    if obj_match:
        try:
            parsed = json.loads(obj_match.group())
            activity_summary = parsed.get("activity_summary", "")
            return parsed.get("observation_plan", []), activity_summary
        except json.JSONDecodeError:
            pass

    # Legacy: bare JSON array (no activity_summary)
    fence_arr = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", response_text, re.DOTALL)
    if fence_arr:
        try:
            return json.loads(fence_arr.group(1)), ""
        except json.JSONDecodeError:
            pass
    match = re.search(r"\[.*\]", response_text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group()), ""
        except json.JSONDecodeError:
            pass
    return [], ""


# ---------------------------------------------------------------------------
# Step 2: Observer (VLM with frames)
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
{prior_estimate_line}
## Context from action detection
An action detector identified these activities involving this item:
{segment_descriptions}

## Task
1. Confirm whether "{item_name}" is visible and actually being used (consumed, \
poured, cut, cooked with, etc.)
2. {observation_instruction}
3. Estimate how much was used in this session and how much remains after.
4. Cite which frames support your estimate.

Think step by step:
- In which frames can you see the item or its container/packaging?
- What action is being performed with it?
- How much is being taken out, poured, cut off, scooped, etc.?
- {remaining_instruction}

Output ONLY JSON:
```json
{{
  "item_confirmed": true or false,
  "reasoning": "<your step-by-step reasoning>",
  "evidence_frames": [<list of timestamp values of key frames supporting your estimate>],
  "amount_used": <number or null if not confirmed>,
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
    prior_remaining: float | None = None,
    prompt_save_path: Path | None = None,
) -> tuple[str, str, dict]:
    """Run Step 2: Observer for one item. Returns (response_text, prompt, stats).

    prompt_save_path: if provided, write the raw prompt to disk BEFORE the
    API call so it survives crashes/timeouts/refusals.
    """
    unit_label = "grams" if item_info["unit"] == "g" else "count"
    visible = item_info.get("visible_during_interaction", True)

    # Build segment descriptions from planner output
    seg_descs = []
    for s in plan_entry.get("segments", []):
        seg_descs.append(f"- {s[0]:.1f}–{s[1]:.1f}s")
    if plan_entry.get("reasoning"):
        seg_descs.append(f"Planner note: {plan_entry['reasoning']}")

    # Format timestamps for the prompt
    frame_ts_str = ", ".join(f"{t:.1f}s" for t in timestamps)

    # Build context and instructions based on visibility.
    # package_amount is shown only as a static "package capacity" reference,
    # never as a measured/last-known amount. prior_remaining (a previous VLM
    # estimate) is the only thing shown as a measurement — and it must be
    # framed as a noisy hint, not ground truth, so the model doesn't anchor.
    item_context = f"- Package capacity: {item_info['package_amount']}"
    if prior_remaining is not None:
        if visible:
            prior_line = (
                f"- Last estimate (previous VLM run): ~{prior_remaining:.0f} {unit_label}. "
                f"This is a NOISY PRIOR, not a measurement — use the current frames as the "
                f"primary evidence and override the prior if what you see disagrees.\n"
            )
        else:
            prior_line = (
                f"- Last estimate (previous VLM run): ~{prior_remaining:.0f} {unit_label}. "
                f"The container is opaque so you cannot directly read remaining contents — "
                f"start from this prior and adjust by how much you observe being dispensed "
                f"in this session.\n"
            )
    else:
        if visible:
            prior_line = "- Not yet observed in any prior session — read the current fill level directly from the frames.\n"
        else:
            prior_line = (
                "- Not yet observed in any prior session — assume the container is at "
                "package capacity and estimate dispensed amount from the action.\n"
            )

    if visible:
        obs_instruction = (
            "Observe the amount change and remaining amount directly from the video frames. "
            "If a prior estimate was provided, treat it as a soft hint only — your reading "
            "of the current frames takes precedence."
        )
        rem_instruction = "What portion appears to remain in the container based on what you see?"
    else:
        obs_instruction = (
            "The container is opaque — estimate usage from the action you observe "
            "(pouring duration, scooping count, etc.). Anchor on the prior estimate "
            "if one was provided, then subtract what you see being dispensed."
        )
        rem_instruction = "Based on the action, how much was likely dispensed from the opaque container?"

    prompt = OBSERVER_PROMPT.format(
        n_frames=len(frames_b64),
        frame_timestamps=frame_ts_str,
        item_name=item_info["visual_class"],
        unit_label=unit_label,
        item_context_line=item_context,
        prior_estimate_line=prior_line,
        segment_descriptions="\n".join(seg_descs),
        observation_instruction=obs_instruction,
        remaining_instruction=rem_instruction,
    )

    if prompt_save_path is not None:
        prompt_save_path.parent.mkdir(parents=True, exist_ok=True)
        prompt_save_path.write_text(prompt)

    # Build content: interleave text labels with frames
    content: list[dict] = [{"type": "input_text", "text": prompt}]
    for i, (fb64, ts) in enumerate(zip(frames_b64, timestamps)):
        content.append({"type": "input_text", "text": f"[Frame {i+1}, t={ts:.1f}s]"})
        content.append({
            "type": "input_image",
            "image_url": f"data:image/jpeg;base64,{fb64}",
            "detail": "low",
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

        # Retry on refusal
        if _is_refusal(response_text) and attempt < max_retries - 1:
            print(f" refusal (attempt {attempt + 1}), retrying...", end="", flush=True)
            time.sleep(5)
            continue

        return response_text, prompt, stats

    return "", prompt, {"error": "max retries exceeded"}


# ---------------------------------------------------------------------------
# Qwen observer (frame-based, same interface as GPT observer)
# ---------------------------------------------------------------------------

def run_observer_qwen(
    frames_b64: list[str],
    timestamps: list[float],
    item_info: dict,
    plan_entry: dict,
    qwen_url: str = QWEN_URL,
    qwen_model: str = QWEN_MODEL_DEFAULT,
    prior_remaining: float | None = None,
    prompt_save_path: Path | None = None,
) -> tuple[str, str, dict]:
    """Run Qwen observer with sampled frames + interleaved [Frame N, t=Xs] labels.

    Uses the same OBSERVER_PROMPT and frame-based interface as run_observer
    (the GPT path) — no concat, no native video upload. Frames are read from
    source clips by extract_segments_frames upstream.
    """
    import requests

    unit_label = "grams" if item_info["unit"] == "g" else "count"
    visible = item_info.get("visible_during_interaction", True)

    seg_descs = []
    for s in plan_entry.get("segments", []):
        seg_descs.append(f"- {s[0]:.1f}–{s[1]:.1f}s")
    if plan_entry.get("reasoning"):
        seg_descs.append(f"Planner note: {plan_entry['reasoning']}")

    # package_amount is shown only as a static "package capacity" reference,
    # never as a measured amount. prior_remaining (a previous VLM estimate) is
    # the only thing shown as a measurement — and it must be framed as a
    # noisy hint, not ground truth, so the model doesn't anchor.
    item_context = f"- Package capacity: {item_info['package_amount']}"
    if prior_remaining is not None:
        if visible:
            prior_line = (
                f"- Last estimate (previous VLM run): ~{prior_remaining:.0f} {unit_label}. "
                f"This is a NOISY PRIOR, not a measurement — use the current frames as the "
                f"primary evidence and override the prior if what you see disagrees.\n"
            )
        else:
            prior_line = (
                f"- Last estimate (previous VLM run): ~{prior_remaining:.0f} {unit_label}. "
                f"The container is opaque so you cannot directly read remaining contents — "
                f"start from this prior and adjust by how much you observe being dispensed "
                f"in this session.\n"
            )
    else:
        if visible:
            prior_line = "- Not yet observed in any prior session — read the current fill level directly from the frames.\n"
        else:
            prior_line = (
                "- Not yet observed in any prior session — assume the container is at "
                "package capacity and estimate dispensed amount from the action.\n"
            )

    if visible:
        obs_instruction = (
            "Observe the amount change and remaining amount directly from the frames. "
            "If a prior estimate was provided, treat it as a soft hint only — your reading "
            "of the current frames takes precedence."
        )
        rem_instruction = "What portion appears to remain in the container based on what you see?"
    else:
        obs_instruction = (
            "The container is opaque — estimate usage from the action you observe "
            "(pouring duration, scooping count, etc.). Anchor on the prior estimate "
            "if one was provided, then subtract what you see being dispensed."
        )
        rem_instruction = "Based on the action, how much was likely dispensed from the opaque container?"

    frame_ts_str = ", ".join(f"{t:.1f}s" for t in timestamps)
    prompt = OBSERVER_PROMPT.format(
        n_frames=len(frames_b64),
        frame_timestamps=frame_ts_str,
        item_name=item_info["visual_class"],
        unit_label=unit_label,
        item_context_line=item_context,
        prior_estimate_line=prior_line,
        segment_descriptions="\n".join(seg_descs),
        observation_instruction=obs_instruction,
        remaining_instruction=rem_instruction,
    )

    if prompt_save_path is not None:
        prompt_save_path.parent.mkdir(parents=True, exist_ok=True)
        prompt_save_path.write_text(prompt)

    # Build content: prompt + interleaved [Frame N, t=X.Xs] labels and images
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
    last_err: str | None = None
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
    """Extract JSON from observer response."""
    result = {
        "item_confirmed": False, "reasoning": "",
        "evidence_frames": [], "amount_used": None, "amount_remaining": None,
    }
    # Try markdown code fence
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response_text, re.DOTALL)
    text_to_parse = fence.group(1) if fence else response_text
    # Try raw JSON object
    match = re.search(r"\{[^{}]*\}", text_to_parse, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            result["item_confirmed"] = parsed.get("item_confirmed", False)
            result["reasoning"] = parsed.get("reasoning", "")
            result["evidence_frames"] = parsed.get("evidence_frames", [])
            result["amount_used"] = parsed.get("amount_used")
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
    planner_only: bool = False,
    verbose: bool = False,
    prior_estimates: dict | None = None,
    model_tag: str = "default",
    run_tag: str = "default",
    inventory_scope: str = "full",
) -> tuple[list[dict], dict]:
    """Process one session through the AVP Round 1 pipeline.

    Args:
        prior_estimates: {instance_id: remaining_amount} from previous sessions.
            Used to give the observer context about expected starting amounts.
        model_tag, run_tag: combined into the per-call cache directory at
            cache/avp/{participant}/{session}/{model_tag}/{run_tag}/
            so different (model, prompt-version) combos never overwrite.

    Returns (predictions, session_log).
    """
    if prior_estimates is None:
        prior_estimates = {}
    session_log: dict = {"session": session, "planner": {}, "observer": []}

    # Per-(model, run) cache for raw prompts/responses (forensics + crash safety).
    cache_dir = CACHE_DIR / participant / session / model_tag / run_tag
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Load inputs
    session_data = load_adatad_session(participant, session)
    if not session_data:
        print(f"  {session}: no AdaTAD item labels")
        return [], session_log

    inventory = load_inventory(participant, session, scope=inventory_scope)
    if not inventory:
        print(f"  {session}: no inventory for scope={inventory_scope}")
        return [], session_log

    n_segs = len(session_data["segments"])
    print(f"  {session}: {n_segs} AdaTAD segments, "
          f"{len(inventory)} inventory items ({inventory_scope} scope)")

    # ── Step 1: Planner (always GPT 5.4, text-only) ──
    planner_model = "gpt-5.4"
    print(f"  Step 1 (Planner): sending {n_segs} segments to {planner_model}...")
    planner_text, planner_prompt, planner_stats = run_planner(
        client, session_data, inventory, planner_model,
        prompt_save_path=cache_dir / "planner_prompt.txt",
    )
    (cache_dir / "planner_response.txt").write_text(planner_text or "")
    observation_plan, activity_summary = parse_planner_response(planner_text)
    session_log["planner"] = {
        "n_segments_input": n_segs,
        "n_items_planned": len(observation_plan),
        "activity_summary": activity_summary,
        "observation_plan": observation_plan,
        "stats": planner_stats,
        "prompt": planner_prompt,
        "raw_response": planner_text,
    }

    if not observation_plan:
        print(f"  Planner returned empty plan")
        return [], session_log

    # Validate plan entries against inventory
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
        # Save planner output and return
        return [], session_log

    # ── Step 2: Observer ──
    clips = get_session_clips(participant, session)
    if not clips:
        print(f"  {session}: no video clips found")
        return [], session_log
    video_durations = get_video_durations(clips)
    total_duration = sum(dur for _, dur in video_durations)

    predictions = []
    for entry in valid_plan:
        iid = entry["instance_id"]
        inv = inv_by_iid[iid]

        segs = entry.get("segments", [])
        if not segs:
            continue
        seg_str = "+".join(f"{s[0]:.0f}-{s[1]:.0f}" for s in segs)
        print(f"    Observer: {inv['visual_class']} [{seg_str}s]...", end="", flush=True)

        # Resolve model → vLLM endpoint (if registered) or use GPT
        vllm_endpoint = VLLM_ENDPOINTS.get(model.lower())
        use_vllm = vllm_endpoint is not None or model.lower().startswith("qwen") or model.lower().startswith("gemma")

        # Look up prior estimate for this item (most recent observer output
        # for this iid from earlier sessions in this run).
        prior_rem = prior_estimates.get(iid)
        if prior_rem is not None:
            print(f"(prior ~{prior_rem:.0f})...", end="", flush=True)

        # Both GPT and vLLM (Qwen/Gemma) backends use the same frame-based path:
        # sample frames directly from source clips, interleave with [Frame N, t=Xs] labels.
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
                prior_remaining=prior_rem,
                prompt_save_path=obs_prompt_path,
            )
        else:
            obs_text, obs_prompt, obs_stats = run_observer(
                client, frames, frame_ts, inv, entry, model,
                prior_remaining=prior_rem,
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

        if parsed["item_confirmed"] and parsed["amount_used"] is not None:
            print(f" used={parsed['amount_used']}, remaining={parsed['amount_remaining']}")
            predictions.append({
                "session": session,
                "item": inv["visual_class"],
                "instance_id": iid,
                "amount_used": parsed["amount_used"],
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
        description="AVP Round 1: agentic food amount estimation"
    )
    parser.add_argument("--participant", default="kailai")
    parser.add_argument("--tag", required=True,
                        help="REQUIRED short label for this run (e.g. 'noisyprior_v1', "
                             "'baseline_2026_04'). Embedded in output filenames and cache "
                             "paths so different prompt iterations never overwrite each other.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--session", help="Single session")
    group.add_argument("--all", action="store_true", help="All sessions")
    parser.add_argument("--model", default="gpt-5.4", help="Observer model (planner always uses gpt-5.4)")
    parser.add_argument("--fps", type=float, default=1.0, help="Frame extraction rate")
    parser.add_argument("--max-frames", type=int, default=MAX_IMAGES)
    parser.add_argument("--planner-only", action="store_true",
                        help="Run only Step 1 (Planner), skip Observer")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--output", type=Path, help="Override output path")
    parser.add_argument("--resume", action="store_true",
                        help="Resume a previously-interrupted run with the same --tag: "
                             "load existing predictions, skip sessions already marked "
                             "complete in the sidecar status file, and rebuild the prior "
                             "chain from the saved predictions.")
    parser.add_argument("--inventory-scope", choices=["full", "session"], default="full",
                        help="Which items to list in the planner's inventory. "
                             "'full' = all items in stock at session time (default); "
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
    print(f"AVP Round 1: Agentic Food Amount Estimation")
    print(f"{'=' * 70}")
    print(f"Participant:  {args.participant}")
    print(f"Model:        {args.model}")
    print(f"Tag:          {run_tag}")
    print(f"FPS:          {args.fps}")
    print(f"Max frames:   {args.max_frames}")
    print(f"Planner only: {args.planner_only}")
    print(f"Sessions:     {len(sessions)}")
    print()

    client = make_client()
    ledger = load_ledger(args.participant)

    all_predictions: list[dict] = []
    all_logs: list[dict] = []

    # Running ledger of estimated remaining amounts across sessions.
    # Starts empty: package_amount is a static reference label, never used as
    # an "amount remaining" estimate. Session 1 has no prior; subsequent
    # sessions inherit the observer's predicted_remaining from earlier runs.
    running_estimates: dict[str, float] = {}  # {instance_id: remaining}

    # Resume sidecar (lives in the participant's outputs/ dir, tagged).
    out_dir = participant_dir(args.participant) / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    preds_path = args.output or out_dir / f"avp_{model_tag}_{run_tag}_preds.json"
    status_path = out_dir / f"avp_{model_tag}_{run_tag}_status.json"

    status: dict = {"completed_sessions": [], "failed_sessions": []}
    if args.resume and preds_path.exists() and status_path.exists():
        all_predictions = json.loads(preds_path.read_text())
        status = json.loads(status_path.read_text())
        # Rebuild aggregated planner trace from per-session files where they exist.
        for s in status.get("completed_sessions", []):
            sess_log_path = outputs_dir(args.participant, s) / f"avp_{model_tag}_{run_tag}_planner.json"
            if sess_log_path.exists():
                try:
                    sess_data = json.loads(sess_log_path.read_text())
                    if "session" in sess_data:
                        all_logs.append(sess_data["session"])
                except json.JSONDecodeError:
                    pass
        # Rebuild running_estimates in chronological order
        for p in sorted(all_predictions, key=lambda x: x.get("session", "")):
            if p.get("amount_remaining") is not None:
                running_estimates[p["instance_id"]] = p["amount_remaining"]
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
                verbose=args.verbose,
                prior_estimates=running_estimates,
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
            # Persist predictions snapshot before halting.
            if not args.planner_only:
                with open(preds_path, "w", encoding="utf-8") as f:
                    json.dump(all_predictions, f, indent=2)
            # HALT on first failure: session N+1 may depend on session N's
            # prior chain, so a silent skip would corrupt downstream priors.
            # User must fix the issue (or wait out a transient outage) and
            # re-run with --resume.
            print(f"\n  HALTED at session {session}. Fix the issue and re-run with --resume.")
            break

        # Update running estimates with this session's predictions
        for p in preds:
            iid = p["instance_id"]
            if p.get("amount_remaining") is not None:
                running_estimates[iid] = p["amount_remaining"]

        all_predictions.extend(preds)
        all_logs.append(log)

        if not session_failed:
            if session not in status["completed_sessions"]:
                status["completed_sessions"].append(session)
            status["failed_sessions"] = [
                f for f in status["failed_sessions"] if f.get("session") != session
            ]
            status_path.write_text(json.dumps(status, indent=2) + "\n")

        # Persist aggregated predictions after every session so resume + crash
        # recovery sees the latest snapshot.
        if not args.planner_only:
            with open(preds_path, "w", encoding="utf-8") as f:
                json.dump(all_predictions, f, indent=2)

        # Save per-session files immediately (no data loss on interruption)
        sess_out = outputs_dir(args.participant, session)
        sess_out.mkdir(parents=True, exist_ok=True)

        # Per-session planner trace (tagged so different runs don't collide)
        planner_sess_path = sess_out / f"avp_{model_tag}_{run_tag}_planner.json"
        with open(planner_sess_path, "w", encoding="utf-8") as f:
            json.dump({
                "participant": args.participant,
                "timestamp": datetime.now().isoformat(),
                "model": args.model,
                "tag": run_tag,
                "session": log,
            }, f, indent=2)

        # Per-session predictions (tagged)
        if not args.planner_only:
            preds_sess_path = sess_out / f"avp_{model_tag}_{run_tag}_preds.json"
            with open(preds_sess_path, "w", encoding="utf-8") as f:
                json.dump(preds, f, indent=2)
            print(f"  Saved: {preds_sess_path}")

        print()

    # Aggregated planner trace (tagged). out_dir/preds_path already defined above.
    planner_path = out_dir / f"avp_{model_tag}_{run_tag}_planner.json"
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
        # preds_path was written incrementally after every session.
        print(f"Saved: {preds_path}")
    print(f"Status saved to {status_path}")

    # Summary
    if not args.planner_only and all_predictions:
        print(f"\n{'=' * 70}")
        print(f"Predictions ({len(all_predictions)} items):")
        for p in all_predictions:
            unit = "count" if ledger["items"][p["instance_id"]]["unit"] == "count" else "g"
            print(f"  [{p['session']}] {p['item']}: "
                  f"used={p['amount_used']} {unit}, "
                  f"remaining={p['amount_remaining']} {unit} "
                  f"({p['planner_confidence']})")

    if failed_sessions:
        print(f"\n{len(failed_sessions)} session(s) failed in this run:")
        for s, err in failed_sessions:
            print(f"  {s}: {err}")
        print(f"\nRe-run with --resume to retry the failed sessions only.")


if __name__ == "__main__":
    main()

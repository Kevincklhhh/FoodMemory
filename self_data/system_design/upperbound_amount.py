#!/usr/bin/env python3
"""Upper-bound VLM amount estimation: send per-item GT segments to VLM.

Uses ground-truth action segments (not AdaTAD proposals) to extract per-item
video clips, providing the VLM with oracle temporal boundaries. This represents
the ceiling for our pipeline — experiment with different models and prompts here.

For each session, for each GT item with action annotations:
  1. Extract video segments corresponding to that item's actions
  2. Concatenate into one clip
  3. Send to VLM with amount estimation prompt
  4. Collect predictions → predictions.json

Usage:
  # Run on one session
  python upperbound_amount.py --participant kailai --session 20260310-195710

  # Run on all sessions
  python upperbound_amount.py --participant kailai

  # Evaluate results
  python evaluate_amount.py --participant kailai --predictions outputs/upperbound_preds.json
"""

import argparse
import base64
import json
import os
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

from frame_sampling import (
    cumulative_to_video_offset,
    extract_segments_frames as _extract_segments_frames,
    get_video_durations,
)
from utils import (
    load_actions,
    load_ledger,
    load_session_inventory,
    participant_dir,
)

_KITCHEN_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(_KITCHEN_DIR / ".env")

CACHE_DIR = Path(__file__).resolve().parent.parent / "cache" / "upperbound"


# Transient errors that should be retried with exponential backoff (the API
# is up but momentarily refusing — 503/429/500/network/timeout). Auth errors,
# invalid-request errors, etc. should NOT be retried — they bubble up.
_TRANSIENT_MARKERS = (
    "503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED",
    "500", "INTERNAL", "deadline", "timeout", "TimeoutError",
    "connection", "Connection",
)


def _is_transient_error(exc: Exception) -> bool:
    s = str(exc)
    return any(m in s for m in _TRANSIENT_MARKERS)


def _retry_call(call, *, label: str, max_retries: int = 6, base_delay: float = 4.0):
    """Run `call()`, retrying on transient errors with exponential backoff.

    Re-raises non-transient errors immediately. Re-raises after max_retries.
    """
    import random
    for attempt in range(max_retries):
        try:
            return call()
        except Exception as e:
            if not _is_transient_error(e) or attempt == max_retries - 1:
                raise
            wait = base_delay * (2 ** attempt) + random.random()
            print(f"\n  {label}: transient error (attempt {attempt+1}/{max_retries}): "
                  f"{str(e)[:120]}\n  retrying in {wait:.1f}s...", flush=True)
            time.sleep(wait)


# ---------------------------------------------------------------------------
# Frame extraction adapter
# ---------------------------------------------------------------------------

def extract_segments_frames(
    item_actions: list[dict],
    video_durations: list[tuple[Path, float]],
    padding: float = 2.0,
    max_frames: int = 50,
    target_fps: float = 1.0,
) -> tuple[list[str], list[float]]:
    """Adapter that converts annotated actions to (start, end) tuples and
    delegates to the shared frame_sampling.extract_segments_frames."""
    return _extract_segments_frames(
        [(a["start"], a["end"]) for a in item_actions],
        video_durations,
        padding=padding,
        max_frames=max_frames,
        target_fps=target_fps,
    )


# ---------------------------------------------------------------------------
# Shared prompt builder
# ---------------------------------------------------------------------------

def build_item_prompt(item_name: str, unit: str, package_amount: str | None,
                      n_frames: int,
                      prior_remaining: float | None = None,
                      visible_during_interaction: bool = True,
                      segments: list[dict] | None = None,
                      timestamps: list[float] | None = None) -> str:
    """Build the prompt for per-item amount estimation (frame-based).

    The model is told it's looking at sampled frames from disjoint moments of a
    cooking session — not continuous footage. Each frame will be labeled with
    its session timestamp inline by the caller. The prompt also lists the
    original session segments so the model can ground frame timestamps to
    annotated actions.

    `package_amount` is shown verbatim as a static reference label (e.g.
    "489g", "16 fl oz", "1 lb"). It is NEVER used as a measured/remaining
    amount. `prior_remaining` (a previous VLM estimate, in the item's natural
    unit) is the only thing shown as a measurement-style hint.
    """
    unit_label = "grams" if unit == "g" else "count"

    context_lines = []
    if package_amount:
        context_lines.append(f'Package size (label, full container): {package_amount}.')
    if prior_remaining is not None:
        if visible_during_interaction:
            context_lines.append(
                f'Last estimate (from a previous VLM run): ~{prior_remaining:.0f} {unit_label}. '
                f'This is a NOISY PRIOR, not a measurement — use the current frames as the '
                f'primary evidence and override the prior if what you see disagrees.'
            )
        else:
            context_lines.append(
                f'Last estimate (from a previous VLM run): ~{prior_remaining:.0f} {unit_label}. '
                f'The container is opaque so you cannot directly read remaining contents — '
                f'start from this prior and adjust by how much you observe being dispensed '
                f'in this session.'
            )
    else:
        if visible_during_interaction:
            context_lines.append('Not yet observed in any prior session — read the current fill level directly from the frames.')
        else:
            context_lines.append(
                'Not yet observed in any prior session — assume the container is at package capacity '
                'and estimate dispensed amount from the action.'
            )
    context_line = "\n".join(context_lines)

    seg_block = ""
    if segments:
        n_seg = len(segments)
        seg_lines = []
        for i, s in enumerate(segments, 1):
            label = s.get("action", "").strip() or "(no label)"
            seg_lines.append(
                f"  {i}. session t={s['start']:.1f}s–{s['end']:.1f}s — {label}"
            )
        seg_block = (
            f"\nThese frames come from {n_seg} annotated moments where "
            f'"{item_name}" was handled (in chronological order):\n'
            + "\n".join(seg_lines)
        )

    ts_line = ""
    if timestamps:
        ts_str = ", ".join(f"{t:.1f}s" for t in timestamps)
        ts_line = f"\nFrame timestamps (session time): {ts_str}"

    intro = (
        f'You are analyzing frames from an egocentric kitchen video recorded with smart glasses.\n'
        f'You will be shown {n_frames} frames sampled from a cooking session. Each frame is '
        f'labeled inline with its session timestamp like "[Frame N, t=X.Xs]". The frames are '
        f'NOT continuous footage — there are time jumps between them, both within an action '
        f'and between different actions.{seg_block}{ts_line}'
    )

    obs_instruction = (
        "Observe the amount change and remaining amount directly from the frames. "
        "If a prior estimate was provided, treat it as a soft hint only — your reading "
        "of the current frames takes precedence."
        if visible_during_interaction
        else "The container is opaque — estimate usage from the actions you observe "
             "(pouring duration, scoop count, etc.). Anchor on the prior estimate if one "
             "was provided, then subtract what you see being dispensed."
    )
    rem_instruction = (
        "What portion appears to remain in the container?"
        if visible_during_interaction
        else "Based on the action, how much was likely dispensed?"
    )

    return f"""{intro}
{context_line}
{obs_instruction}

Estimate:
1. How much of "{item_name}" was used (in {unit_label})
2. How much of "{item_name}" remains after this session (in {unit_label})

Think step by step about what you see:
- In which frames can you see the item or its container/packaging?
- What actions are performed with this item?
- How much is being taken out, poured, cut off, etc.?
- {rem_instruction}

Output your answer as JSON:
```json
{{
  "reasoning": "<your step-by-step reasoning>",
  "evidence_frames": [<list of frame numbers (1-indexed) supporting your estimate>],
  "amount_used": <number>,
  "amount_remaining": <number>
}}
```"""


def parse_item_response(response_text: str) -> dict:
    """Parse JSON from VLM response text."""
    result = {"predicted_used": None, "predicted_remaining": None, "reasoning": ""}
    json_match = re.search(r"\{[^{}]*\}", response_text, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group())
            result["predicted_used"] = parsed.get("amount_used")
            result["predicted_remaining"] = parsed.get("amount_remaining")
            result["reasoning"] = parsed.get("reasoning", "")
        except json.JSONDecodeError:
            pass
    return result


# ---------------------------------------------------------------------------
# VLM Clients
# ---------------------------------------------------------------------------

# All clients use the same frame-based interface:
#   estimate_amount(frames_b64, timestamps, item_name, unit, ...)
# Frames are sampled directly from source clips (extract_segments_frames) and
# interleaved with [Frame N, t=X.Xs] text labels in the API call.
#
# Frame budgets per backend (input token constraints):
#   Gemini 2.5 Pro/Flash: ~1M tokens — 200+ frames easily
#   Qwen2.5-VL (local vLLM): ~32k tokens — ~60 frames at low res
#   GPT-5.4 (Azure): 50 image hard limit


class GeminiAmountClient:
    """Gemini: inline JPEG frames + interleaved timestamp labels."""

    def __init__(self, model: str = "gemini-2.5-pro", max_frames: int = 200):
        from google import genai
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Missing GOOGLE_API_KEY")
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.max_frames = max_frames

    def estimate_amount(self, frames_b64, timestamps, item_name, unit, package_amount=None,
                        thinking_budget=8192, prior_remaining=None,
                        visible_during_interaction=True, segments=None, **_kw):
        from google.genai import types

        prompt = build_item_prompt(
            item_name, unit, package_amount,
            n_frames=len(frames_b64),
            prior_remaining=prior_remaining,
            visible_during_interaction=visible_during_interaction,
            segments=segments,
            timestamps=timestamps,
        )

        contents = [types.Part.from_text(text=prompt)]
        for i, (fb64, ts) in enumerate(zip(frames_b64, timestamps), 1):
            contents.append(types.Part.from_text(text=f"[Frame {i}, t={ts:.1f}s]"))
            contents.append(types.Part.from_bytes(
                data=base64.b64decode(fb64),
                mime_type="image/jpeg",
            ))

        t0 = time.time()
        response = _retry_call(
            lambda: self.client.models.generate_content(
                model=self.model, contents=contents,
                config=types.GenerateContentConfig(
                    temperature=0.3, max_output_tokens=4096,
                    thinking_config=types.ThinkingConfig(thinking_budget=thinking_budget, include_thoughts=True),
                ),
            ),
            label=f"Gemini {self.model}",
        )
        stats = {"inference_time_s": round(time.time() - t0, 2), "model": self.model,
                 "num_frames": len(frames_b64)}
        um = response.usage_metadata
        if um:
            stats["input_tokens"] = um.prompt_token_count
            stats["output_tokens"] = um.candidates_token_count
            stats["total_tokens"] = um.total_token_count

        thinking_text, response_text = "", ""
        if response.candidates and response.candidates[0].content:
            for part in response.candidates[0].content.parts:
                if hasattr(part, "thought") and part.thought:
                    thinking_text += (part.text or "")
                else:
                    response_text += (part.text or "")

        parsed = parse_item_response(response_text)
        return {**parsed, "thinking": thinking_text, "raw_response": response_text, "prompt": prompt, "stats": stats}


class VLLMAmountClient:
    """Generic OpenAI-compat vLLM client (Qwen, Gemma, etc.).

    Sends inline JPEG frames + interleaved [Frame N, t=Xs] labels.
    """

    DEFAULT_URL = "http://saltyfish.eecs.umich.edu:8000/v1/chat/completions"

    def __init__(self, model: str, url: str = DEFAULT_URL, max_frames: int = 60):
        self.model = model
        self.url = url
        self.max_frames = max_frames

    def estimate_amount(self, frames_b64, timestamps, item_name, unit, package_amount=None,
                        prior_remaining=None, visible_during_interaction=True,
                        segments=None, **_kw):
        import requests

        prompt = build_item_prompt(
            item_name, unit, package_amount,
            n_frames=len(frames_b64),
            prior_remaining=prior_remaining,
            visible_during_interaction=visible_during_interaction,
            segments=segments,
            timestamps=timestamps,
        )

        content = [{"type": "text", "text": prompt}]
        for i, (fb64, ts) in enumerate(zip(frames_b64, timestamps), 1):
            content.append({"type": "text", "text": f"[Frame {i}, t={ts:.1f}s]"})
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{fb64}"},
            })

        messages = [{"role": "user", "content": content}]
        t0 = time.time()
        try:
            resp = requests.post(self.url, json={
                "model": self.model,
                "messages": messages,
                "max_tokens": 4096,
                "temperature": 0.3,
            }, timeout=600)
            resp.raise_for_status()
            data = resp.json()
            response_text = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            stats = {
                "inference_time_s": round(time.time() - t0, 2), "model": self.model,
                "num_frames": len(frames_b64),
                "input_tokens": usage.get("prompt_tokens"),
                "output_tokens": usage.get("completion_tokens"),
                "total_tokens": usage.get("total_tokens"),
            }
        except Exception as e:
            print(f"  vLLM ERROR: {e}")
            return {"predicted_used": None, "predicted_remaining": None, "reasoning": "",
                    "thinking": "", "raw_response": "", "prompt": prompt,
                    "stats": {"inference_time_s": round(time.time() - t0, 2), "model": self.model, "error": str(e)}}

        parsed = parse_item_response(response_text)
        return {**parsed, "thinking": "", "raw_response": response_text, "prompt": prompt, "stats": stats}


# Backward-compat alias
QwenAmountClient = VLLMAmountClient


class GPTAmountClient:
    """GPT: inline JPEG frames + interleaved timestamp labels via Azure Responses API.

    Azure hard limit: 50 images per request.
    """

    MAX_IMAGES = 50

    def __init__(self, model: str = "gpt-5.4", max_frames: int = 50):
        from openai import AzureOpenAI
        api_key = os.getenv("AZURE_OPENAI_API_KEY_2") or os.getenv("AZURE_OPENAI_API_KEY")
        endpoint = (os.getenv("AZURE_OPENAI_ENDPOINT_2") or os.getenv("AZURE_OPENAI_ENDPOINT") or "").strip()
        if not api_key or not endpoint:
            raise ValueError("Missing Azure OpenAI API credentials")
        self.client = AzureOpenAI(
            azure_endpoint=endpoint, api_key=api_key, api_version="2025-03-01-preview",
        )
        self.model = model
        self.max_frames = min(max_frames, self.MAX_IMAGES)

    def estimate_amount(self, frames_b64, timestamps, item_name, unit, package_amount=None,
                        prior_remaining=None, visible_during_interaction=True,
                        segments=None, **_kw):
        prompt = build_item_prompt(
            item_name, unit, package_amount,
            n_frames=len(frames_b64),
            prior_remaining=prior_remaining,
            visible_during_interaction=visible_during_interaction,
            segments=segments,
            timestamps=timestamps,
        )

        content = [{"type": "input_text", "text": prompt}]
        for i, (fb64, ts) in enumerate(zip(frames_b64, timestamps), 1):
            content.append({"type": "input_text", "text": f"[Frame {i}, t={ts:.1f}s]"})
            content.append({
                "type": "input_image",
                "image_url": f"data:image/jpeg;base64,{fb64}",
                "detail": "low",
            })

        t0 = time.time()
        try:
            response = self.client.responses.create(
                model=self.model,
                input=[{"role": "user", "content": content}],
                reasoning={"effort": "medium"},
            )
            response_text = response.output_text or ""
            usage = response.usage
            stats = {
                "inference_time_s": round(time.time() - t0, 2), "model": self.model,
                "num_frames": len(frames_b64),
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "total_tokens": getattr(usage, "total_tokens", None),
            }
        except Exception as e:
            print(f"  GPT ERROR: {e}")
            return {"predicted_used": None, "predicted_remaining": None, "reasoning": "",
                    "thinking": "", "raw_response": "", "prompt": prompt,
                    "stats": {"inference_time_s": round(time.time() - t0, 2), "model": self.model, "error": str(e)}}

        parsed = parse_item_response(response_text)
        return {**parsed, "thinking": "", "raw_response": response_text, "prompt": prompt, "stats": stats}


VLLM_ENDPOINTS = {
    "qwen": ("http://saltyfish.eecs.umich.edu:8000/v1/chat/completions", "Qwen/Qwen3-VL-30B-A3B-Instruct"),
    "gemma": ("http://saltyfish.eecs.umich.edu:8000/v1/chat/completions", "google/gemma-4-31B-it"),
    "gemma4": ("http://saltyfish.eecs.umich.edu:8000/v1/chat/completions", "google/gemma-4-31B-it"),
}


def make_client(model: str):
    """Create the appropriate VLM client for the given model name."""
    m = model.lower()
    if m.startswith("gemini"):
        return GeminiAmountClient(model=model)
    if m.startswith("gpt"):
        return GPTAmountClient(model=model)
    if m in VLLM_ENDPOINTS:
        url, mdl = VLLM_ENDPOINTS[m]
        return VLLMAmountClient(model=mdl, url=url)
    if "qwen" in m or "gemma" in m:
        # Treat unknown qwen/gemma model strings as raw model IDs at the default vLLM URL
        return VLLMAmountClient(model=model)
    raise ValueError(f"Unknown model: {model}. Use gemini-*, gpt-*, qwen, or gemma.")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def process_session(
    participant: str,
    session: str,
    client,
    ledger: dict,
    prior_estimates: dict | None = None,
    model_tag: str = "default",
    run_tag: str = "default",
    only_item: str | None = None,
) -> list[dict]:
    """Process one session: extract per-item clips, query Gemini, return predictions.

    Per-item logs are written to
    cache/upperbound/{participant}/{session}/{model_tag}/{run_tag}/
    so different (model, prompt-version) combos never overwrite each other.
    The run_tag is required at the CLI layer and is what the user types when
    starting a run, so they can group related results.
    """
    if prior_estimates is None:
        prior_estimates = {}
    try:
        actions = load_actions(participant, session)
    except FileNotFoundError:
        print(f"  SKIP {session}: no actions.json")
        return []
    if not actions:
        print(f"  SKIP {session}: empty actions.json")
        return []

    # Get GT items for this session
    snap = ledger.get("snapshots", {}).get(session, {})
    if not snap:
        print(f"  SKIP {session}: no snapshot")
        return []

    # Get video files and durations
    vdir = participant_dir(participant) / "videos" / session
    session_videos = sorted(vdir.glob("*.mp4"))
    if not session_videos:
        print(f"  SKIP {session}: no videos")
        return []

    video_durations = get_video_durations(session_videos)
    total_dur = sum(d for _, d in video_durations)
    print(f"  {session}: {len(session_videos)} videos ({total_dur:.0f}s), {len(actions)} actions")

    # Per-(model, run) log dir so different model runs and different prompt
    # iterations don't overwrite each other.
    cache = CACHE_DIR / participant / session / model_tag / run_tag
    cache.mkdir(parents=True, exist_ok=True)

    predictions = []
    items = ledger["items"]

    for iid, state in snap.items():
        if only_item is not None and iid != only_item:
            continue
        used = state.get("used")
        remaining_gt = state.get("remaining")
        # Skip only when neither metric has a scorable GT. Remaining-only
        # annotations (used=None, remaining=<val>) still yield CNPE_rem even
        # though CNPE_used can't be computed.
        if used is None and remaining_gt is None:
            continue

        item = items.get(iid, {})
        visual_class = item.get("visual_class", iid)
        unit = item.get("unit", "g")

        # Check if this item has actions. Skip stage=="VISIBLE_PORTION" segments:
        # those are "item visible in hand/on counter" frames, not amount-changing
        # actions, so they bloat the frame budget without helping the model
        # estimate dispensed/remaining amounts.
        item_actions = [
            a for a in actions
            if a["item"] == iid
            and (a.get("stage") or "").upper() != "VISIBLE_PORTION"
        ]
        if not item_actions:
            print(f"    {visual_class}: no amount-changing action segments, skipping")
            continue

        total_seg_dur = sum(a["end"] - a["start"] for a in item_actions)
        # package_amount is a free-form reference label ("489g", "16 fl oz", ...)
        # passed to the prompt as-is. Eval reads it separately for CNPE.
        pkg = item.get("package_amount", "")

        # Starting amount at this session
        starting_amount = state.get("starting")

        print(f"    {visual_class}: {len(item_actions)} segments ({total_seg_dur:.0f}s)...", end=" ", flush=True)

        # Sample frames directly from source clips (per-clip clamped padding)
        max_frames = getattr(client, "max_frames", 50)
        frames_b64, timestamps = extract_segments_frames(
            item_actions, video_durations, padding=2.0, max_frames=max_frames,
        )
        if not frames_b64:
            print("FAIL (frame extraction)")
            continue

        # Query VLM
        prior_rem = prior_estimates.get(iid)
        visible = item.get("visible_during_interaction", True)
        segments_meta = [
            {"start": a["start"], "end": a["end"], "action": a.get("action", "")}
            for a in item_actions
        ]

        # Persist the raw prompt to disk BEFORE the API call so it survives
        # crashes/timeouts. The client's estimate_amount() rebuilds the same
        # string internally; this is duplicated work but cheap (pure string
        # formatting) and gives us a forensic record on failure.
        prompt_text = build_item_prompt(
            visual_class, unit, pkg,
            n_frames=len(frames_b64),
            prior_remaining=prior_rem,
            visible_during_interaction=visible,
            segments=segments_meta,
            timestamps=timestamps,
        )
        (cache / f"{iid}_prompt.txt").write_text(prompt_text)

        try:
            result = client.estimate_amount(
                frames_b64, timestamps, visual_class, unit, package_amount=pkg,
                thinking_budget=getattr(client, '_thinking_budget', 8192),
                prior_remaining=prior_rem,
                visible_during_interaction=visible,
                segments=segments_meta,
            )
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"FAIL: {e}")
            continue
        stats = result["stats"]
        print(f"used={result['predicted_used']}, rem={result['predicted_remaining']} ({stats.get('inference_time_s', '?')}s)")

        pred_entry = {
            "session": session,
            "item": visual_class,
            "instance_id": iid,
            "amount_used": result["predicted_used"],
            "amount_remaining": result["predicted_remaining"],
            "reasoning": result["reasoning"],
            "thinking": result["thinking"],
            "stats": stats,
            "segments": [
                {"start": a["start"], "end": a["end"], "action": a.get("action", "")}
                for a in item_actions
            ],
        }
        predictions.append(pred_entry)

        # Save per-item log (prompt, response, thinking)
        log_entry = {
            **pred_entry,
            "prompt": result["prompt"],
            "raw_response": result["raw_response"],
            "num_frames": len(frames_b64),
            "frame_timestamps": timestamps,
            "gt_used": state.get("used"),
            "gt_remaining": state.get("remaining"),
            "package_amount": pkg,
            "starting_amount": starting_amount,
        }
        log_path = cache / f"{iid}_log.json"
        log_path.write_text(json.dumps(log_entry, indent=2, ensure_ascii=False) + "\n")

    return predictions


def main():
    parser = argparse.ArgumentParser(description="Baseline VLM amount estimation")
    parser.add_argument("--participant", required=True)
    parser.add_argument("--tag", required=True,
                        help="REQUIRED short label for this run (e.g. 'noisyprior_v1', "
                             "'no_prior_anchor'). Embedded in output filenames and cache "
                             "paths so different prompt iterations never overwrite each other.")
    parser.add_argument("--session", help="Process single session (default: all)")
    parser.add_argument("--until", help="Run sessions up to and including this one")
    parser.add_argument("--item",
                        help="Run only this instance_id. Finds every session that "
                             "uses the item and processes them in chronological order. "
                             "If --tag matches an existing run, ONLY this item's entries "
                             "are overridden in the predictions file; all other items are "
                             "left untouched. Compatible with --session/--until to further "
                             "restrict the session set.")
    parser.add_argument("--model", default="gemini-2.5-pro", help="Gemini model name")
    parser.add_argument("--output", type=Path, help="Output predictions JSON path")
    parser.add_argument("--thinking-budget", type=int, default=8192)
    parser.add_argument("--resume", action="store_true",
                        help="Resume a previously-interrupted run with the same --tag: "
                             "load existing predictions, skip sessions already marked "
                             "complete in the sidecar status file, and rebuild the prior "
                             "chain from the saved predictions.")
    args = parser.parse_args()

    ledger = load_ledger(args.participant)
    snapshots = ledger.get("snapshots", {})

    if args.session:
        sessions = [args.session]
    else:
        sessions = sorted(snapshots.keys())
        if args.until:
            sessions = [s for s in sessions if s <= args.until]

    # --item mode: restrict to sessions where the target item actually has a
    # snapshot entry with a recorded `used` value (i.e. the sessions a normal
    # full run would have processed for this item).
    if args.item:
        target_sessions = [
            s for s in sessions
            if args.item in snapshots.get(s, {})
            and (snapshots[s][args.item].get("used") is not None
                 or snapshots[s][args.item].get("remaining") is not None)
        ]
        if not target_sessions:
            print(f"No sessions found for item {args.item} in participant {args.participant}.")
            return
        print(f"--item {args.item}: {len(target_sessions)} session(s) -> {target_sessions}")
        sessions = target_sessions

    # Model name + run tag both go into output path so different model runs
    # AND different prompt iterations stay separated.
    model_tag = args.model.replace("/", "_")
    run_tag = args.tag.replace("/", "_")
    output_path = args.output or (
        participant_dir(args.participant) / "outputs"
        / f"upperbound_{model_tag}_{run_tag}_preds.json"
    )

    client = make_client(args.model)
    client._thinking_budget = args.thinking_budget

    # Running ledger of estimated remaining amounts across sessions.
    # Starts empty: session 1 has no prior estimate (the model sees package_amount
    # only as a static reference label in the prompt, never as an "amount remaining"
    # value). After each session, the VLM's predicted_remaining for an item is
    # carried forward as the prior_remaining hint for the next session that uses it.
    running_estimates: dict[str, float] = {}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    status_path = output_path.with_name(
        output_path.stem.replace("_preds", "_status") + ".json"
    )

    # ── Resume bookkeeping ──
    all_predictions: list[dict] = []
    status: dict = {"completed_sessions": [], "failed_sessions": []}
    if args.resume and output_path.exists() and status_path.exists():
        all_predictions = json.loads(output_path.read_text())
        status = json.loads(status_path.read_text())
        completed = set(status.get("completed_sessions", []))
        # Rebuild running_estimates from existing predictions in chronological order
        for p in sorted(all_predictions, key=lambda x: x.get("session", "")):
            if p.get("amount_remaining") is not None:
                running_estimates[p["instance_id"]] = p["amount_remaining"]
        pending = [s for s in sessions if s not in completed]
        print(f"\nRESUME: {len(completed)} session(s) already complete, "
              f"{len(pending)} pending (of {len(sessions)} total).")
        sessions = pending
    elif args.resume:
        print("\nRESUME requested but no existing predictions/status found — starting fresh.")

    # --item override mode: load existing preds (if any) so we can splice the
    # target item back in without touching other items, and rebuild the prior
    # chain for the target item from preds in sessions BEFORE the first one
    # we're about to recompute. Other items' priors are irrelevant since we
    # only process the target item.
    if args.item:
        if not args.resume and output_path.exists():
            all_predictions = json.loads(output_path.read_text())
        running_estimates = {}
        first_target = sessions[0]
        for p in sorted(all_predictions, key=lambda x: x.get("session", "")):
            if (p.get("instance_id") == args.item
                    and p.get("session", "") < first_target
                    and p.get("amount_remaining") is not None):
                running_estimates[args.item] = p["amount_remaining"]

    failed_sessions: list[tuple[str, str]] = []
    for session in sessions:
        try:
            preds = process_session(args.participant, session, client, ledger,
                                    prior_estimates=running_estimates,
                                    model_tag=model_tag, run_tag=run_tag,
                                    only_item=args.item)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            import traceback
            print(f"\n  ERROR in session {session}: {e}")
            traceback.print_exc()
            failed_sessions.append((session, str(e)[:200]))
            status["failed_sessions"] = [
                f for f in status["failed_sessions"] if f.get("session") != session
            ] + [{"session": session, "error": str(e)[:200]}]
            status_path.write_text(json.dumps(status, indent=2) + "\n")
            # HALT on first failure: session N+1 may depend on session N's
            # prior chain, so a silent skip would corrupt downstream priors.
            # User must fix the issue (or wait out a transient outage) and
            # re-run with --resume.
            print(f"\n  HALTED at session {session}. Fix the issue and re-run with --resume.")
            break

        for p in preds:
            if p.get("amount_remaining") is not None:
                running_estimates[p["instance_id"]] = p["amount_remaining"]
        if args.item:
            # Splice: drop any existing entries for (session, target item) and
            # append the freshly computed ones. Other items in the file are
            # left untouched. Don't update status — the run isn't a full
            # session pass and shouldn't interfere with future --resume runs.
            target_keys = {(p["session"], p["instance_id"]) for p in preds}
            all_predictions = [
                x for x in all_predictions
                if (x.get("session"), x.get("instance_id")) not in target_keys
            ]
            all_predictions.extend(preds)
            output_path.write_text(json.dumps(all_predictions, indent=2) + "\n")
        else:
            all_predictions.extend(preds)
            output_path.write_text(json.dumps(all_predictions, indent=2) + "\n")
            if session not in status["completed_sessions"]:
                status["completed_sessions"].append(session)
            status["failed_sessions"] = [
                f for f in status["failed_sessions"] if f.get("session") != session
            ]
            status_path.write_text(json.dumps(status, indent=2) + "\n")

    print(f"\n{len(all_predictions)} predictions saved to {output_path}")
    print(f"Status saved to {status_path}")
    if failed_sessions:
        print(f"\n{len(failed_sessions)} session(s) failed in this run:")
        for s, err in failed_sessions:
            print(f"  {s}: {err}")
        print(f"\nRe-run with --resume to retry the failed sessions only.")

    # Run evaluation and save report (CORRUPTED_ITEMS excluded automatically)
    from importlib import import_module
    eval_mod = import_module("evaluate_amount")
    gt = eval_mod.extract_ground_truth(ledger)
    report = eval_mod.evaluate(gt, all_predictions)

    eval_path = output_path.with_name(output_path.stem + "_eval.json")
    eval_mod.write_report(report, eval_path)

    eval_mod.print_eval_table(report)


if __name__ == "__main__":
    main()

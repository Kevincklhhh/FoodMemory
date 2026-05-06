#!/usr/bin/env python3
"""Upper-bound VLM remaining-amount estimation (observation only).

Like upperbound_amount.py but asks ONLY for the remaining amount of each item
at the end of the session. The prompt deliberately avoids any mention of
"usage", "consumed", or "amount_used" — the VLM's sole task is to observe the
final state of each item. This isolates perceptual estimation from
action-based guessing.

Usage:
  python upperbound_remaining.py --participant kailai --tag remaining_v1
  python upperbound_remaining.py --participant kailai --session 20260310-195710 --tag remaining_v1
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

CACHE_DIR = Path(__file__).resolve().parent.parent / "cache" / "upperbound_remaining"


_TRANSIENT_MARKERS = (
    "503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED",
    "500", "INTERNAL", "deadline", "timeout", "TimeoutError",
    "connection", "Connection",
)


def _is_transient_error(exc: Exception) -> bool:
    s = str(exc)
    return any(m in s for m in _TRANSIENT_MARKERS)


def _retry_call(call, *, label: str, max_retries: int = 6, base_delay: float = 4.0):
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
    return _extract_segments_frames(
        [(a["start"], a["end"]) for a in item_actions],
        video_durations,
        padding=padding,
        max_frames=max_frames,
        target_fps=target_fps,
    )


# ---------------------------------------------------------------------------
# Prompt builder — remaining-only, no usage language
# ---------------------------------------------------------------------------

def build_item_prompt(item_name: str, unit: str, package_amount: str | None,
                      n_frames: int,
                      segments: list[dict] | None = None,
                      timestamps: list[float] | None = None) -> str:
    """Build a remaining-only prompt for per-item amount estimation (frame-based).

    The model is asked to observe the final state of the item and report
    how much remains. No prior estimates are provided — the model must
    rely purely on what it sees. Only called for visible items.
    """
    unit_label = "grams" if unit == "g" else "count"

    context_lines = []
    if package_amount:
        context_lines.append(f'Package size (label, full container): {package_amount}.')
    context_lines.append('Read the current fill level directly from the frames.')
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
        "Focus on the latest frames where the item or its container is visible. "
        "Assess the fill level, quantity, or remaining contents at that point."
    )
    rem_instruction = "What portion appears to remain in the container in the latest frames?"

    return f"""{intro}
{context_line}
{obs_instruction}

Your task: estimate the **remaining amount** of "{item_name}" (in {unit_label}) at the end of this session.

Think step by step about what you see:
- In which frames can you see the item or its container/packaging?
- Focus on the last appearance of the item — what is its state?
- {rem_instruction}

Output your answer as JSON:
```json
{{
  "reasoning": "<your step-by-step reasoning>",
  "evidence_frames": [<list of timestamp values (from the t=X.Xs labels) of key frames supporting your estimate>],
  "amount_remaining": <number>
}}
```"""


def parse_item_response(response_text: str) -> dict:
    """Parse JSON from VLM response text."""
    result = {"predicted_remaining": None, "reasoning": "", "evidence_frames": []}
    json_match = re.search(r"\{[^{}]*\}", response_text, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group())
            result["predicted_remaining"] = parsed.get("amount_remaining")
            result["reasoning"] = parsed.get("reasoning", "")
            result["evidence_frames"] = parsed.get("evidence_frames", [])
        except json.JSONDecodeError:
            pass
    return result


# ---------------------------------------------------------------------------
# VLM Clients
# ---------------------------------------------------------------------------

class GeminiAmountClient:
    def __init__(self, model: str = "gemini-2.5-pro", max_frames: int = 200):
        from google import genai
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Missing GOOGLE_API_KEY")
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.max_frames = max_frames

    def estimate_amount(self, frames_b64, timestamps, item_name, unit, package_amount=None,
                        thinking_budget=8192, segments=None, **_kw):
        from google.genai import types

        prompt = build_item_prompt(
            item_name, unit, package_amount,
            n_frames=len(frames_b64),
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
    DEFAULT_URL = "http://saltyfish.eecs.umich.edu:8000/v1/chat/completions"

    def __init__(self, model: str, url: str = DEFAULT_URL, max_frames: int = 60):
        self.model = model
        self.url = url
        self.max_frames = max_frames

    def estimate_amount(self, frames_b64, timestamps, item_name, unit, package_amount=None,
                        segments=None, **_kw):
        import requests

        prompt = build_item_prompt(
            item_name, unit, package_amount,
            n_frames=len(frames_b64),
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
            return {"predicted_remaining": None, "reasoning": "",
                    "thinking": "", "raw_response": "", "prompt": prompt,
                    "stats": {"inference_time_s": round(time.time() - t0, 2), "model": self.model, "error": str(e)}}

        parsed = parse_item_response(response_text)
        return {**parsed, "thinking": "", "raw_response": response_text, "prompt": prompt, "stats": stats}


QwenAmountClient = VLLMAmountClient


class GPTAmountClient:
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
                        segments=None, **_kw):
        prompt = build_item_prompt(
            item_name, unit, package_amount,
            n_frames=len(frames_b64),
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
            return {"predicted_remaining": None, "reasoning": "",
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
    m = model.lower()
    if m.startswith("gemini"):
        return GeminiAmountClient(model=model)
    if m.startswith("gpt"):
        return GPTAmountClient(model=model)
    if m in VLLM_ENDPOINTS:
        url, mdl = VLLM_ENDPOINTS[m]
        return VLLMAmountClient(model=mdl, url=url)
    if "qwen" in m or "gemma" in m:
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
    model_tag: str = "default",
    run_tag: str = "default",
    only_item: str | None = None,
) -> list[dict]:
    """Process one session: extract per-item clips, query VLM for remaining amount.

    Only visible items are processed — opaque items are skipped since
    their remaining amount cannot be visually assessed.
    """
    try:
        actions = load_actions(participant, session)
    except FileNotFoundError:
        print(f"  SKIP {session}: no actions.json")
        return []
    if not actions:
        print(f"  SKIP {session}: empty actions.json")
        return []

    snap = ledger.get("snapshots", {}).get(session, {})
    if not snap:
        print(f"  SKIP {session}: no snapshot")
        return []

    vdir = participant_dir(participant) / "videos" / session
    session_videos = sorted(vdir.glob("*.mp4"))
    if not session_videos:
        print(f"  SKIP {session}: no videos")
        return []

    video_durations = get_video_durations(session_videos)
    total_dur = sum(d for _, d in video_durations)
    print(f"  {session}: {len(session_videos)} videos ({total_dur:.0f}s), {len(actions)} actions")

    cache = CACHE_DIR / participant / session / model_tag / run_tag
    cache.mkdir(parents=True, exist_ok=True)

    predictions = []
    items = ledger["items"]

    for iid, state in snap.items():
        if only_item is not None and iid != only_item:
            continue
        # Remaining-only script: the VLM predicts amount_remaining only, so
        # the only GT signal we need is `remaining`. `used` may be None.
        if state.get("remaining") is None:
            continue

        item = items.get(iid, {})
        visual_class = item.get("visual_class", iid)
        unit = item.get("unit", "g")

        # Skip opaque items — remaining amount cannot be visually assessed
        if not item.get("visible_during_interaction", True):
            print(f"    {visual_class}: opaque, skipping")
            continue

        item_actions = [
            a for a in actions
            if a["item"] == iid
            and (a.get("stage") or "").upper() != "VISIBLE_PORTION"
        ]
        if not item_actions:
            print(f"    {visual_class}: no amount-changing action segments, skipping")
            continue

        total_seg_dur = sum(a["end"] - a["start"] for a in item_actions)
        pkg = item.get("package_amount", "")

        print(f"    {visual_class}: {len(item_actions)} segments ({total_seg_dur:.0f}s)...", end=" ", flush=True)

        max_frames = getattr(client, "max_frames", 50)
        frames_b64, timestamps = extract_segments_frames(
            item_actions, video_durations, padding=2.0, max_frames=max_frames,
        )
        if not frames_b64:
            print("FAIL (frame extraction)")
            continue

        segments_meta = [
            {"start": a["start"], "end": a["end"], "action": a.get("action", "")}
            for a in item_actions
        ]

        prompt_text = build_item_prompt(
            visual_class, unit, pkg,
            n_frames=len(frames_b64),
            segments=segments_meta,
            timestamps=timestamps,
        )
        (cache / f"{iid}_prompt.txt").write_text(prompt_text)

        try:
            result = client.estimate_amount(
                frames_b64, timestamps, visual_class, unit, package_amount=pkg,
                thinking_budget=getattr(client, '_thinking_budget', 8192),
                segments=segments_meta,
            )
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"FAIL: {e}")
            continue
        stats = result["stats"]
        print(f"rem={result['predicted_remaining']} ({stats.get('inference_time_s', '?')}s)")

        pred_entry = {
            "session": session,
            "item": visual_class,
            "instance_id": iid,
            "amount_remaining": result["predicted_remaining"],
            "evidence_frames": result.get("evidence_frames", []),
            "reasoning": result["reasoning"],
            "thinking": result.get("thinking", ""),
            "stats": stats,
            "segments": [
                {"start": a["start"], "end": a["end"], "action": a.get("action", "")}
                for a in item_actions
            ],
        }
        predictions.append(pred_entry)

        log_entry = {
            **pred_entry,
            "prompt": result["prompt"],
            "raw_response": result["raw_response"],
            "num_frames": len(frames_b64),
            "frame_timestamps": timestamps,
            "gt_used": state.get("used"),
            "gt_remaining": state.get("remaining"),
            "package_amount": pkg,
            "starting_amount": state.get("starting"),
        }
        (cache / f"{iid}_log.json").write_text(
            json.dumps(log_entry, indent=2, ensure_ascii=False) + "\n"
        )

    return predictions


def main():
    parser = argparse.ArgumentParser(
        description="Upper-bound VLM remaining-amount estimation (observation only)")
    parser.add_argument("--participant", required=True)
    parser.add_argument("--tag", required=True,
                        help="Short label for this run (e.g. 'remaining_v1').")
    parser.add_argument("--session", help="Process single session (default: all)")
    parser.add_argument("--until", help="Run sessions up to and including this one")
    parser.add_argument("--item",
                        help="Run only this instance_id across matching sessions.")
    parser.add_argument("--model", default="gemini-2.5-pro")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--thinking-budget", type=int, default=8192)
    parser.add_argument("--resume", action="store_true",
                        help="Resume a previously-interrupted run with the same --tag.")
    args = parser.parse_args()

    ledger = load_ledger(args.participant)
    snapshots = ledger.get("snapshots", {})

    if args.session:
        sessions = [args.session]
    else:
        sessions = sorted(snapshots.keys())
        if args.until:
            sessions = [s for s in sessions if s <= args.until]

    if args.item:
        target_sessions = [
            s for s in sessions
            if args.item in snapshots.get(s, {})
            and snapshots[s][args.item].get("remaining") is not None
        ]
        if not target_sessions:
            print(f"No sessions found for item {args.item} in participant {args.participant}.")
            return
        print(f"--item {args.item}: {len(target_sessions)} session(s) -> {target_sessions}")
        sessions = target_sessions

    model_tag = args.model.replace("/", "_")
    run_tag = args.tag.replace("/", "_")
    output_path = args.output or (
        participant_dir(args.participant) / "outputs"
        / f"upperbound_remaining_{model_tag}_{run_tag}_preds.json"
    )

    client = make_client(args.model)
    client._thinking_budget = args.thinking_budget

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
        pending = [s for s in sessions if s not in completed]
        print(f"\nRESUME: {len(completed)} session(s) already complete, "
              f"{len(pending)} pending (of {len(sessions)} total).")
        sessions = pending
    elif args.resume:
        print("\nRESUME requested but no existing predictions/status found — starting fresh.")

    if args.item:
        if not args.resume and output_path.exists():
            all_predictions = json.loads(output_path.read_text())

    failed_sessions: list[tuple[str, str]] = []
    for session in sessions:
        try:
            preds = process_session(args.participant, session, client, ledger,
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
            print(f"\n  HALTED at session {session}. Fix the issue and re-run with --resume.")
            break

        if args.item:
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

    # Run evaluation
    from importlib import import_module
    eval_mod = import_module("evaluate_amount")
    gt = eval_mod.extract_ground_truth(ledger)
    report = eval_mod.evaluate(gt, all_predictions)

    eval_path = output_path.with_name(output_path.stem + "_eval.json")
    eval_mod.write_report(report, eval_path)

    eval_mod.print_eval_table(report)


if __name__ == "__main__":
    main()

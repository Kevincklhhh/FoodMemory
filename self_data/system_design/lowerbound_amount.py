#!/usr/bin/env python3
"""Lower-bound baseline: send whole session videos to VLM for amount estimation.

No per-item segmentation, no GT action hints. The VLM sees the full video and
must identify which items were used and estimate amounts for each. This is the
end-to-end floor — kept stable and re-run as the dataset expands.

Videos are converted to 1fps to reduce tokens, then uploaded to Gemini.

Usage:
  python lowerbound_amount.py --participant kailai --session 20260310-195710
  python lowerbound_amount.py --participant kailai  # all sessions
"""

import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path

from dotenv import load_dotenv

from utils import load_inventory, load_ledger, participant_dir
from lowerbound_remaining import concat_1fps_videos

_KITCHEN_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(_KITCHEN_DIR / ".env")

CACHE_DIR = Path(__file__).resolve().parent.parent / "cache" / "lowerbound"


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
# Video utilities
# ---------------------------------------------------------------------------

def convert_to_1fps(src: Path, dst: Path) -> bool:
    """Convert video to 1fps for token reduction."""
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", str(src), "-vf", "fps=1", "-c:v", "libx264",
         "-preset", "fast", "-crf", "28", "-an", str(dst)],
        capture_output=True,
    )
    return dst.exists() and dst.stat().st_size > 0


def get_video_duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True,
    )
    try:
        return float(out.stdout.strip())
    except ValueError:
        return 0.0


# ---------------------------------------------------------------------------
# Prompt builder (extracted so callers can persist the prompt to disk before
# the API call)
# ---------------------------------------------------------------------------

def build_session_prompt(item_list: list[dict], prior_estimates: dict | None = None) -> str:
    """Build the whole-session lower-bound prompt for one session.

    `item_list`: output of `load_session_inventory(...)` — one entry per item
    that was present at session start, with `instance_id`, `visual_class`,
    `unit`, `package_amount`, `visible_during_interaction`.
    `prior_estimates`: {instance_id: prev VLM amount_remaining}, or None.

    Visibility/opaque only affects the per-item `, VISIBLE)` / `, OPAQUE)`
    tag — the Rules section in the prompt body explains how the model should
    treat each.
    """
    if prior_estimates is None:
        prior_estimates = {}

    item_lines = []
    for inv in item_list:
        unit_label = "grams" if inv["unit"] == "g" else "count"
        pkg = inv.get("package_amount", "")
        iid = inv.get("instance_id", "")
        prior = prior_estimates.get(iid)
        visible = inv.get("visible_during_interaction", True)

        line = f"- {iid}: \"{inv['visual_class']}\" ({unit_label}"
        if pkg:
            line += f", package size: {pkg}"
        if prior is not None:
            line += f", last estimated remaining: ~{prior:.0f} {unit_label}"
        else:
            line += ", not yet observed in any prior session"
        line += ", VISIBLE)" if visible else ", OPAQUE)"
        item_lines.append(line)
    items_str = "\n".join(item_lines)

    return f"""You are a kitchen inventory auditor analyzing egocentric video from smart glasses.

The following food items are in the kitchen at the start of this session. Each line is formatted as `<instance_id>: "<display name>" (...)`. Multiple lines may share the same display name when there are several physical instances of the same product (e.g. two milk gallons) — they are different physical items and must be reported separately by `instance_id`.

{items_str}

Watch the entire video. For each instance that was used (taken out, consumed, cooked with, dispensed, etc.):
1. Observe the amount change and remaining amount directly from the video when the item is visible
2. For opaque containers where contents are not visible, estimate usage based on the action performed
3. Estimate the amount remaining after this session
4. If two instances share a display name, decide which physical instance is being handled in each event (e.g. an older opened bottle vs a newly retrieved sealed one) and assign usage to that specific `instance_id`. If you genuinely cannot tell them apart, report each instance separately with your best guess and lower the amounts accordingly.

Output a JSON array. Only include instances that were actually used (amount_used > 0).

```json
[
  {{
    "instance_id": "<instance_id exactly as listed above>",
    "item": "<display name exactly as listed above>",
    "evidence_timestamps": [<list of video timestamps in seconds where you observed usage>],
    "amount_used": <number>,
    "amount_remaining": <number>
  }}
]
```

Rules:
- Use grams for weight items and integer count for discrete items
- `instance_id` is REQUIRED and must match one of the IDs above exactly
- `evidence_timestamps` must list the video timestamps (in seconds, integers) of the key moments \
you used to deduce the amounts — e.g. dispensing actions, fill level observations, etc.
- Only report instances you actually see being used in the video
- For VISIBLE items: base your estimate on what you observe directly in the container. \
The "last estimated remaining" value (when shown) is a NOISY PRIOR from a previous VLM run, \
not a measurement — use the current video as the primary evidence and override the prior \
if what you see disagrees. If no prior is shown, read the current fill level directly.
- For OPAQUE items: you cannot directly see remaining contents. When a "last estimated \
remaining" value is shown, anchor on it and subtract what you see being dispensed in this \
session. When no prior is shown, assume the container is at package capacity and estimate \
the dispensed amount from the pouring/dispensing action duration and speed.
- If unsure about exact amounts, give your best estimate
- Return ONLY the JSON array, no other text"""


# ---------------------------------------------------------------------------
# Gemini client
# ---------------------------------------------------------------------------

class GeminiWholeVideoClient:
    def __init__(self, model: str = "gemini-2.5-flash"):
        from google import genai
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Missing GOOGLE_API_KEY")
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self._uploaded = {}

    def upload_video(self, video_path: Path):
        key = str(video_path)
        if key in self._uploaded:
            return self._uploaded[key]
        video_file = self.client.files.upload(file=str(video_path))
        while video_file.state == "PROCESSING":
            time.sleep(1)
            video_file = self.client.files.get(name=video_file.name)
        if video_file.state == "FAILED":
            raise ValueError(f"Video processing failed: {video_file.name}")
        self._uploaded[key] = video_file
        return video_file

    def query_session(
        self,
        video_paths: list[Path],
        item_list: list[dict],
        thinking_budget: int = 8192,
        prior_estimates: dict | None = None,
        prompt: str | None = None,
    ) -> dict:
        """Send full session videos + item list, get amount predictions for all items.

        If `prompt` is None, it is built internally from item_list +
        prior_estimates via build_session_prompt(). Callers that want to
        persist the prompt to disk before the API call should build it via
        build_session_prompt() and pass it in here.

        Returns: {predictions: [{item, amount_used, amount_remaining}], thinking, raw_response, prompt, stats}
        """
        from google.genai import types
        if prior_estimates is None:
            prior_estimates = {}
        if prompt is None:
            prompt = build_session_prompt(item_list, prior_estimates)

        # Build contents: all videos + prompt
        t0 = time.time()
        contents = []
        for vp in video_paths:
            vf = self.upload_video(vp)
            contents.append(
                types.Part.from_uri(file_uri=vf.uri, mime_type="video/mp4")
            )
        contents.append(prompt)

        response = _retry_call(
            lambda: self.client.models.generate_content(
                model=self.model,
                contents=contents,
                config=types.GenerateContentConfig(
                    temperature=0.3,
                    max_output_tokens=8192,
                    thinking_config=types.ThinkingConfig(
                        thinking_budget=thinking_budget,
                        include_thoughts=True,
                    ),
                ),
            ),
            label=f"Gemini {self.model}",
        )

        stats = {"inference_time_s": round(time.time() - t0, 2)}
        um = response.usage_metadata
        if um:
            stats["input_tokens"] = um.prompt_token_count
            stats["output_tokens"] = um.candidates_token_count
            stats["total_tokens"] = um.total_token_count

        # Extract thinking and response
        thinking_text = ""
        response_text = ""
        if response.candidates and response.candidates[0].content:
            for part in response.candidates[0].content.parts:
                if hasattr(part, "thought") and part.thought:
                    thinking_text += (part.text or "")
                else:
                    response_text += (part.text or "")

        # Parse JSON array from response
        predictions = []
        json_match = re.search(r"\[[\s\S]*\]", response_text)
        if json_match:
            try:
                predictions = json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        return {
            "predictions": predictions,
            "thinking": thinking_text,
            "raw_response": response_text,
            "prompt": prompt,
            "stats": stats,
        }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def process_session(
    participant: str,
    session: str,
    client: GeminiWholeVideoClient,
    ledger: dict,
    thinking_budget: int,
    prior_estimates: dict | None = None,
    model_tag: str = "default",
    run_tag: str = "default",
    inventory_scope: str = "full",
) -> list[dict]:
    """Process one session: send whole videos, return predictions.

    Per-(model, run) outputs go to
    cache/lowerbound/{participant}/{session}/{model_tag}/{run_tag}/
    The 1-fps re-encoded videos are cached at the parent level (shared across
    runs) since they don't depend on prompt/model.
    """
    snap = ledger.get("snapshots", {}).get(session, {})
    if not snap:
        print(f"  SKIP {session}: no snapshot")
        return []

    # Get inventory
    inventory = load_inventory(participant, session, scope=inventory_scope)
    if not inventory:
        print(f"  SKIP {session}: no inventory for scope={inventory_scope}")
        return []

    # Get videos
    vdir = participant_dir(participant) / "videos" / session
    session_videos = sorted(vdir.glob("*.mp4"))
    if not session_videos:
        print(f"  SKIP {session}: no videos")
        return []

    total_dur = sum(get_video_duration(v) for v in session_videos)
    print(f"  {session}: {len(session_videos)} videos ({total_dur:.0f}s), {len(inventory)} items")

    # 1-fps re-encodes are shared across all model/run tags (they only depend
    # on the source video). Per-run outputs (prompt, response, log) live in a
    # nested {model_tag}/{run_tag}/ subdir so different runs never collide.
    video_cache = CACHE_DIR / participant / session
    video_cache.mkdir(parents=True, exist_ok=True)
    run_cache = video_cache / model_tag / run_tag
    run_cache.mkdir(parents=True, exist_ok=True)

    fps_videos = []
    for vp in session_videos:
        fps_path = video_cache / f"{vp.stem}_1fps.mp4"
        if not fps_path.exists():
            print(f"    Converting {vp.name} to 1fps...", end=" ", flush=True)
            if convert_to_1fps(vp, fps_path):
                orig_mb = vp.stat().st_size / (1024 * 1024)
                new_mb = fps_path.stat().st_size / (1024 * 1024)
                print(f"{orig_mb:.1f}MB -> {new_mb:.1f}MB")
            else:
                print("FAILED")
                continue
        fps_videos.append(fps_path)

    if not fps_videos:
        print(f"  SKIP {session}: 1fps conversion failed")
        return []

    # Concatenate clips into a single contiguous mp4 so timestamps are
    # session-absolute and Gemini sees one video per session.
    concat_path = video_cache / f"{session}_1fps_concat.mp4"
    if not concat_path.exists():
        print(f"    Concatenating {len(fps_videos)} clip(s) -> {concat_path.name}...",
              end=" ", flush=True)
        if not concat_1fps_videos(fps_videos, concat_path):
            print("FAILED")
            return []
        print(f"OK ({concat_path.stat().st_size / (1024*1024):.1f}MB)")
    query_videos = [concat_path]

    # Build the prompt and persist it to disk BEFORE the API call so it
    # survives crashes/timeouts/refusals.
    prompt_text = build_session_prompt(inventory, prior_estimates or {})
    (run_cache / "prompt.txt").write_text(prompt_text)

    print(f"    Querying {client.model} with {len(query_videos)} clip (concatenated from {len(fps_videos)})...", end=" ", flush=True)
    result = client.query_session(query_videos, inventory, thinking_budget=thinking_budget,
                                   prior_estimates=prior_estimates or {},
                                   prompt=prompt_text)
    stats = result["stats"]
    print(f"{len(result['predictions'])} items detected ({stats.get('inference_time_s', '?')}s, {stats.get('total_tokens', '?')} tok)")

    # Convert to evaluation format. Predictions are addressed by instance_id;
    # fall back to visual_class only if the model omitted it AND there is exactly
    # one instance for that visual_class in this session.
    valid_iids = {inv["instance_id"] for inv in inventory}
    name_to_iids: dict[str, list[str]] = {}
    for inv in inventory:
        name_to_iids.setdefault(inv["visual_class"].lower(), []).append(inv["instance_id"])
    iid_to_name = {inv["instance_id"]: inv["visual_class"] for inv in inventory}

    predictions = []
    for pred in result["predictions"]:
        pred_iid = (pred.get("instance_id") or "").strip()
        pred_name = (pred.get("item") or "").strip()

        if pred_iid and pred_iid in valid_iids:
            resolved_iid = pred_iid
            resolved_name = iid_to_name[pred_iid]
        else:
            # Legacy fallback: only safe when the visual_class is unique in this session.
            candidates = name_to_iids.get(pred_name.lower(), [])
            if len(candidates) == 1:
                resolved_iid = candidates[0]
                resolved_name = iid_to_name[resolved_iid]
            else:
                resolved_iid = ""  # ambiguous or unknown — eval will mark as hallucinated
                resolved_name = pred_name
        predictions.append({
            "session": session,
            "item": resolved_name,
            "instance_id": resolved_iid,
            "evidence_timestamps": pred.get("evidence_timestamps", []),
            "amount_used": pred.get("amount_used"),
            "amount_remaining": pred.get("amount_remaining"),
        })

    # Save session log
    log = {
        "session": session,
        "model": client.model,
        "inventory": inventory,
        "predictions": result["predictions"],
        "thinking": result["thinking"],
        "raw_response": result["raw_response"],
        "prompt": result["prompt"],
        "stats": stats,
    }
    log_path = run_cache / "session_log.json"
    log_path.write_text(json.dumps(log, indent=2, ensure_ascii=False) + "\n")

    return predictions


def main():
    parser = argparse.ArgumentParser(description="Whole-video VLM amount estimation (lower bound)")
    parser.add_argument("--participant", required=True)
    parser.add_argument("--tag", required=True,
                        help="REQUIRED short label for this run (e.g. 'noisyprior_v1', "
                             "'baseline_2026_04'). Embedded in output filenames and cache "
                             "paths so different prompt iterations never overwrite each other.")
    parser.add_argument("--session", help="Single session (default: all)")
    parser.add_argument("--until", help="Run sessions up to and including this one (e.g. 20260323-185522)")
    parser.add_argument("--model", default="gemini-2.5-pro")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--thinking-budget", type=int, default=8192)
    parser.add_argument("--resume", action="store_true",
                        help="Resume a previously-interrupted run with the same --tag: "
                             "load existing predictions, skip sessions already marked "
                             "complete in the sidecar status file, and rebuild the prior "
                             "chain from the saved predictions.")
    parser.add_argument("--inventory-scope", choices=["full", "session"], default="full",
                        help="Which items to list in the Gemini prompt. "
                             "'full' = all items in stock at session time (default); "
                             "'session' = GT-annotated subset only.")
    args = parser.parse_args()

    ledger = load_ledger(args.participant)
    snapshots = ledger.get("snapshots", {})

    if args.session:
        sessions = [args.session]
    else:
        sessions = sorted(snapshots.keys())
        if args.until:
            sessions = [s for s in sessions if s <= args.until]

    model_tag = args.model.replace("/", "_")
    run_tag = args.tag.replace("/", "_")
    output_path = args.output or (
        participant_dir(args.participant) / "outputs"
        / f"lowerbound_{model_tag}_{run_tag}_preds.json"
    )

    client = GeminiWholeVideoClient(model=args.model)

    # Running ledger of estimated remaining amounts across sessions.
    # Starts empty: package_amount is a static reference, never used as an
    # amount-remaining estimate. The first session has no prior; subsequent
    # sessions inherit predicted_remaining from earlier VLM runs.
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

    failed_sessions: list[tuple[str, str]] = []
    for session in sessions:
        try:
            preds = process_session(args.participant, session, client, ledger,
                                    args.thinking_budget,
                                    prior_estimates=running_estimates,
                                    model_tag=model_tag, run_tag=run_tag,
                                    inventory_scope=args.inventory_scope)
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
        all_predictions.extend(preds)
        # Persist predictions and status after every session.
        output_path.write_text(json.dumps(all_predictions, indent=2) + "\n")
        if session not in status["completed_sessions"]:
            status["completed_sessions"].append(session)
        # Clear from failed list if a previous attempt had failed
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

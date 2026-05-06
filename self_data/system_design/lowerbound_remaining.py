#!/usr/bin/env python3
"""Lower-bound baseline (remaining-only): send whole session videos to VLM.

Unlike lowerbound_amount.py which asks the VLM to estimate both usage and
remaining amounts, this variant asks ONLY for the remaining amount of each
item at the end of the session.  The prompt deliberately avoids any mention
of "usage", "consumed", or "amount_used" — the VLM's sole task is to
observe the final state of each item.  This isolates perceptual estimation
from action-based guessing.

Usage:
  python lowerbound_remaining.py --participant kailai --tag remaining_v1
  python lowerbound_remaining.py --participant kailai --session 20260310-195710 --tag remaining_v1
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

_KITCHEN_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(_KITCHEN_DIR / ".env")

CACHE_DIR = Path(__file__).resolve().parent.parent / "cache" / "lowerbound_remaining"


# Transient errors that should be retried with exponential backoff.
_TRANSIENT_MARKERS = (
    "503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED",
    "500", "INTERNAL", "deadline", "timeout", "TimeoutError",
    "connection", "Connection",
)


def _is_transient_error(exc: Exception) -> bool:
    s = str(exc)
    return any(m in s for m in _TRANSIENT_MARKERS)


def _retry_call(call, *, label: str, max_retries: int = 6, base_delay: float = 4.0):
    """Run `call()`, retrying on transient errors with exponential backoff."""
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
# Video utilities (identical to lowerbound_amount.py)
# ---------------------------------------------------------------------------

def convert_to_1fps(src: Path, dst: Path) -> bool:
    # -g 1 forces every frame to be a keyframe, so downstream stream-copy
    # splits in split_video() are frame-accurate (no content lost at chunk
    # boundaries due to GOP alignment).
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", str(src), "-vf", "fps=1", "-c:v", "libx264",
         "-preset", "fast", "-crf", "28", "-g", "1", "-an", str(dst)],
        capture_output=True,
    )
    return dst.exists() and dst.stat().st_size > 0


def concat_1fps_videos(fps_videos: list[Path], dst: Path) -> bool:
    """Concatenate 1fps clips into one mp4 (re-encoded so timestamps are
    session-absolute and Gemini receives a single contiguous video).

    The ffmpeg `concat` filter requires identical width/height/SAR across all
    inputs. Recordings from the smart glasses can vary in resolution between
    clips (e.g. 1376x1840 vs 1504x2000), so we scale every input to the first
    clip's dimensions and force SAR=1 before concatenation. ffmpeg stderr is
    surfaced on failure so problems stop being silent (0-byte output).
    """
    if not fps_videos:
        return False
    if len(fps_videos) == 1:
        import shutil
        shutil.copy(fps_videos[0], dst)
        return dst.exists() and dst.stat().st_size > 0

    # Probe target dimensions from the first clip.
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0:s=x",
         str(fps_videos[0])],
        capture_output=True, text=True,
    )
    target = probe.stdout.strip()  # e.g. "1376x1840"
    if "x" not in target:
        print(f"  ffprobe failed for {fps_videos[0].name}: {probe.stderr.strip()}")
        return False
    tw, th = target.split("x")

    inputs: list[str] = []
    for vp in fps_videos:
        inputs += ["-i", str(vp)]
    scale_parts = "".join(
        f"[{i}:v:0]scale={tw}:{th},setsar=1[v{i}];" for i in range(len(fps_videos))
    )
    concat_inputs = "".join(f"[v{i}]" for i in range(len(fps_videos)))
    filter_complex = (
        f"{scale_parts}{concat_inputs}concat=n={len(fps_videos)}:v=1:a=0[outv]"
    )
    result = subprocess.run(
        ["ffmpeg", "-y", *inputs, "-filter_complex", filter_complex,
         "-map", "[outv]", "-c:v", "libx264", "-preset", "fast",
         "-crf", "28", "-g", "1", "-an", str(dst)],
        capture_output=True, text=True,
    )
    ok = dst.exists() and dst.stat().st_size > 0
    if not ok:
        # Surface ffmpeg's last few error lines so silent failures stop
        # being a debugging trap.
        tail = "\n".join(result.stderr.strip().splitlines()[-6:])
        print(f"  ffmpeg concat failed for {dst.name}:\n{tail}")
    return ok


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


def split_video(src: Path, max_seconds: float, output_dir: Path) -> list[Path]:
    """Split a video into segments of ≤max_seconds using ffmpeg stream copy."""
    import math
    dur = get_video_duration(src)
    if dur <= max_seconds:
        return [src]

    n_chunks = math.ceil(dur / max_seconds)
    chunk_len = math.ceil(dur / n_chunks)
    parts = []
    for i in range(n_chunks):
        start = i * chunk_len
        out_path = output_dir / f"{src.stem}_part{i}.mp4"
        if not out_path.exists():
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(src),
                 "-ss", str(start), "-t", str(chunk_len),
                 "-c", "copy", "-an", str(out_path)],
                capture_output=True,
            )
        if out_path.exists() and out_path.stat().st_size > 0:
            parts.append(out_path)
    return parts


def group_videos_by_duration(videos: list[Path], max_seconds: float) -> list[list[Path]]:
    """Greedily pack pre-existing video files into groups of ≤max_seconds total."""
    groups: list[list[Path]] = []
    current: list[Path] = []
    current_dur = 0.0
    for v in videos:
        d = get_video_duration(v)
        if current_dur + d > max_seconds and current:
            groups.append(current)
            current = []
            current_dur = 0.0
        current.append(v)
        current_dur += d
    if current:
        groups.append(current)
    return groups


# ---------------------------------------------------------------------------
# Prompt builder — remaining-only, no usage language
# ---------------------------------------------------------------------------

def _format_inventory_lines(item_list: list[dict]) -> str:
    """Shared inventory rendering for v1/v2 prompts."""
    item_lines = []
    for inv in item_list:
        unit_label = "grams" if inv["unit"] == "g" else "count"
        pkg = inv.get("package_amount", "")
        iid = inv.get("instance_id", "")
        line = f"- {iid}: \"{inv['visual_class']}\" ({unit_label}"
        if pkg:
            line += f", package size: {pkg}"
        line += ")"
        item_lines.append(line)
    return "\n".join(item_lines)


def build_session_prompt_v2(item_list: list[dict]) -> str:
    """LB prompt v2 — output schema matched to AVP-minimal sweep observer.

    Forces the model to produce: (1) an `inventory_journey` of activity
    blocks where one or more inventory items are interacted with (NOT a
    timeline of every kitchen action), and (2) per-item entries with
    status + 3-amount triple (starting / remaining / derivative) so the
    reasoning load is comparable to AVP's sweep call. Used for the
    canonical LB-vs-AVP comparison where reasoning-load symmetry is
    important.
    """
    items_str = _format_inventory_lines(item_list)
    return f"""You are a kitchen inventory auditor analyzing egocentric video from smart glasses.

The following food items are in the kitchen at the start of this session. Multiple lines may share the same display name when there are several physical instances of the same product (e.g. two milk gallons) — they are different physical items and must be reported separately by `instance_id`.

{items_str}

Watch the entire video. Your task has two outputs:

1. **inventory_journey** — emit one entry per activity block in which the wearer **interacts with one or more of the inventory items above** (retrieves, opens, dispenses from, cuts, transfers from, returns, puts away). DO NOT emit entries for activities that do not involve an inventory item — skip dishwashing, idle stirring of finished cookware, hand-washing, walking around, mid-cook waiting, eating from a plate, etc. Each entry carries:
   - `time`: `[start_seconds, end_seconds]` covering the block
   - `scene`: `storage` / `sink` / `stove` / `counter` / `unknown` (use `→` for short transitions, e.g. `storage→counter`)
   - `user_action`: ≤12-word verb-led description (e.g. "retrieves carton from fridge", "dispenses pasta into pot", "returns jug to fridge")
   - `items`: list of `visual_class` strings — every inventory item meaningfully involved in this block (handled, dispensed, returned). At least one inventory item is required for the entry to exist.

2. **items** — one entry per item from the inventory list above (cover EVERY iid, even `not_used`). For each:
   - `status`: `used` if material left the container during this session, `not_used` otherwise.
   - For `used` items, commit to as many of the 3-amount triple as you can defensibly read from the video:
     - `amount_starting`: amount in the container when the user first interacts with it.
     - `amount_remaining`: amount in the container at end of session (post-last-interaction view).
     - `amount_derivative`: amount that left the container across the session (sum of dispenses).
     Self-consistency: `amount_remaining ≈ amount_starting − amount_derivative`. Populate at least two of the three when possible; set the third to `null` if not directly observable. If you can read only one, populate that one and set the other two to `null`.
   - For `not_used` items, set all three amount fields to `null`.

`package_size` (in the inventory list) is the receipt amount, NOT the session-start fill — the container may have been partially used in earlier sessions. Read `amount_starting` directly from what you see in the container, not from `package_size`.

Output JSON only, no surrounding text:

```json
{{
  "inventory_journey": [
    {{
      "time": [<start_seconds>, <end_seconds>],
      "scene": "storage" | "sink" | "stove" | "counter" | "unknown",
      "user_action": "<≤12-word verb-led description>",
      "items": ["<visual_class>", ...]
    }}
  ],
  "items": [
    {{
      "instance_id": "<exact iid from above>",
      "item": "<exact display name from above>",
      "status": "used" | "not_used",
      "amount_starting": <number or null>,
      "amount_remaining": <number or null>,
      "amount_derivative": <number or null>,
      "evidence_timestamps": [<integer seconds>],
      "reasoning": "<≤25 words: cue used (visible fill / heft / pour duration) + brief journey summary>"
    }}
  ]
}}
```

Rules:
- Use grams for weight items and integer count for discrete items, for ALL three amount fields.
- `evidence_timestamps` must list integer seconds of the key moments you used (interactions + post-interaction state read).
- The `items[]` array must cover EVERY inventory iid above (one entry each), even if `not_used`.
- `inventory_journey` should be tight — one entry per distinct interaction block; typical 5–20 entries for a 5–25 min session. Skip everything that doesn't touch an inventory item.
- Return ONLY the JSON object, no other text."""


def build_session_prompt(item_list: list[dict]) -> str:
    """Build a remaining-only prompt for one session.

    The VLM is asked to observe the final state of each item and report
    how much remains.  No prior estimates are provided — the model must
    rely purely on what it sees.  Only visible items are included (opaque
    items are filtered out upstream).
    """
    items_str = _format_inventory_lines(item_list)

    return f"""You are a kitchen inventory auditor analyzing egocentric video from smart glasses.

The following food items are in the kitchen at the start of this session. Each line is formatted as `<instance_id>: "<display name>" (...)`. Multiple lines may share the same display name when there are several physical instances of the same product (e.g. two milk gallons) — they are different physical items and must be reported separately by `instance_id`.

{items_str}

Watch the entire video. Your task is to estimate the **remaining amount** of each item at the end of this session. For each item, identify every interaction with it (taking from the container, pouring, cutting, returning, etc.) and assess the item's state **after all interactions are complete**

For each item that you can assess:
1. Locate the moment after the final interaction has finished  — the container/item is set down, sealed, or otherwise no longer being manipulated
2. Observe the fill level, quantity, or remaining contents at that post-interaction state
3. Report your best estimate of the amount remaining
4. If two instances share a display name, identify which physical instance you are looking at (e.g. by condition, label wear, position) and report each separately by `instance_id`

Output a JSON array. Include every instance whose remaining amount you can estimate from the video.

```json
[
  {{
    "instance_id": "<instance_id exactly as listed above>",
    "item": "<display name exactly as listed above>",
    "evidence_timestamps": [<list of video timestamps in seconds where you observed the item>],
    "amount_remaining": <number>
  }}
]
```

Rules:
- Use grams for weight items and integer count for discrete items
- Only report instances whose remaining amount you can visually assess in the video
- Base your estimate on what you directly observe in the container — read the fill level, \
count the remaining items, or assess the visible quantity
- `evidence_timestamps` must list the video timestamps (in seconds, integers) of the key moments \
you used to deduce the remaining amount — e.g. the post-interaction view showing fill level, \
dispensing moments, etc.
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
        prompt: str | None = None,
    ) -> dict:
        from google.genai import types
        if prompt is None:
            prompt = build_session_prompt(item_list)

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

        thinking_text = ""
        response_text = ""
        if response.candidates and response.candidates[0].content:
            for part in response.candidates[0].content.parts:
                if hasattr(part, "thought") and part.thought:
                    thinking_text += (part.text or "")
                else:
                    response_text += (part.text or "")

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
    model_tag: str = "default",
    run_tag: str = "default",
    inventory_scope: str = "full",
) -> list[dict]:
    """Process one session: send whole videos, return remaining-amount predictions.

    Only visible items are included — opaque items are filtered out since
    their remaining amount cannot be visually assessed.
    """
    snap = ledger.get("snapshots", {}).get(session, {})
    if not snap:
        print(f"  SKIP {session}: no snapshot")
        return []

    inventory = load_inventory(participant, session, scope=inventory_scope)
    if not inventory:
        print(f"  SKIP {session}: no inventory for scope={inventory_scope}")
        return []

    # Filter out opaque items — remaining amount cannot be visually assessed
    visible_inventory = [inv for inv in inventory if inv.get("visible_during_interaction", True)]
    n_opaque = len(inventory) - len(visible_inventory)
    if not visible_inventory:
        print(f"  SKIP {session}: no visible items ({n_opaque} opaque filtered)")
        return []

    vdir = participant_dir(participant) / "videos" / session
    session_videos = sorted(vdir.glob("*.mp4"))
    if not session_videos:
        print(f"  SKIP {session}: no videos")
        return []

    total_dur = sum(get_video_duration(v) for v in session_videos)
    print(f"  {session}: {len(session_videos)} videos ({total_dur:.0f}s), "
          f"{len(visible_inventory)} visible items ({n_opaque} opaque filtered)")

    # 1-fps re-encodes shared across runs; per-run outputs in nested subdir
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

    prompt_text = build_session_prompt(visible_inventory)
    (run_cache / "prompt.txt").write_text(prompt_text)

    print(f"    Querying {client.model} with {len(query_videos)} clip (concatenated from {len(fps_videos)})...", end=" ", flush=True)
    result = client.query_session(query_videos, visible_inventory, thinking_budget=thinking_budget,
                                   prompt=prompt_text)
    stats = result["stats"]
    print(f"{len(result['predictions'])} items detected ({stats.get('inference_time_s', '?')}s, {stats.get('total_tokens', '?')} tok)")

    # Resolve instance_ids (only visible items were sent to VLM)
    valid_iids = {inv["instance_id"] for inv in visible_inventory}
    name_to_iids: dict[str, list[str]] = {}
    for inv in visible_inventory:
        name_to_iids.setdefault(inv["visual_class"].lower(), []).append(inv["instance_id"])
    iid_to_name = {inv["instance_id"]: inv["visual_class"] for inv in visible_inventory}

    predictions = []
    for pred in result["predictions"]:
        pred_iid = (pred.get("instance_id") or "").strip()
        pred_name = (pred.get("item") or "").strip()

        if pred_iid and pred_iid in valid_iids:
            resolved_iid = pred_iid
            resolved_name = iid_to_name[pred_iid]
        else:
            candidates = name_to_iids.get(pred_name.lower(), [])
            if len(candidates) == 1:
                resolved_iid = candidates[0]
                resolved_name = iid_to_name[resolved_iid]
            else:
                resolved_iid = ""
                resolved_name = pred_name
        predictions.append({
            "session": session,
            "item": resolved_name,
            "instance_id": resolved_iid,
            "evidence_timestamps": pred.get("evidence_timestamps", []),
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
    (run_cache / "session_log.json").write_text(
        json.dumps(log, indent=2, ensure_ascii=False) + "\n"
    )

    return predictions


def main():
    parser = argparse.ArgumentParser(
        description="Whole-video VLM remaining-amount estimation (observation only)")
    parser.add_argument("--participant", required=True)
    parser.add_argument("--tag", required=True,
                        help="Short label for this run (e.g. 'remaining_v1'). "
                             "Embedded in output filenames and cache paths.")
    parser.add_argument("--session", help="Single session (default: all)")
    parser.add_argument("--until", help="Run sessions up to and including this one")
    parser.add_argument("--model", default="gemini-2.5-pro")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--thinking-budget", type=int, default=8192)
    parser.add_argument("--resume", action="store_true",
                        help="Resume a previously-interrupted run with the same --tag.")
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
        / f"lowerbound_remaining_{model_tag}_{run_tag}_preds.json"
    )

    client = GeminiWholeVideoClient(model=args.model)

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

    failed_sessions: list[tuple[str, str]] = []
    for session in sessions:
        try:
            preds = process_session(args.participant, session, client, ledger,
                                    args.thinking_budget,
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
            print(f"\n  HALTED at session {session}. Fix the issue and re-run with --resume.")
            break

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

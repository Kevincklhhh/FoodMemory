#!/usr/bin/env python3
"""Upper-bound VLM amount estimation — CONCATENATED VIDEO variant.

This is a sibling of `upperbound_amount.py` that, instead of sampling frames
from each GT segment and sending them as inline JPEGs, extracts the segments
as short clips with ffmpeg, concatenates them into ONE video per item, and
uploads that single video to Gemini via the Files API.

Two implementation guarantees the user asked for explicitly:

  1. Multi-clip awareness in the prompt
     The model is told the video is a HARD-CUT concatenation of N short
     clips taken from a longer cooking session, and is given the original
     session-time range of each clip in chronological order. So a transition
     between clips is a time jump, not a continuous motion edit.

  2. Padding never crosses clip boundaries
     Each GT action segment is anchored to the source .mp4 file containing
     its RAW (unpadded) start. The padded window [start-pad, end+pad] is then
     clamped to that file's [0, duration]. If a single GT segment spans more
     than one source file, the right edge is clamped at the EOF of the
     starting clip rather than bleeding past it. This is the same per-clip
     clamping logic used in frame_sampling.extract_segments_frames.

Output filename: upperbound_video_{model}_{tag}_preds.json
(distinct from frame-based upper bound and whole-video lower bound)

Usage:
  python upperbound_amount_video.py --participant kailai --tag clipvideo_v1
  python upperbound_amount_video.py --participant kailai --tag clipvideo_v1 \
      --session 20260310-195710
  python upperbound_amount_video.py --participant kailai --tag clipvideo_v1 \
      --item large_white_eggs_20260310
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import subprocess
import time
from pathlib import Path

from dotenv import load_dotenv

from frame_sampling import cumulative_to_video_offset, get_video_durations
from utils import load_actions, load_ledger, participant_dir

_KITCHEN_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(_KITCHEN_DIR / ".env")

CACHE_DIR = Path(__file__).resolve().parent.parent / "cache" / "upperbound_video"


# ---------------------------------------------------------------------------
# Retry helper (same policy as upperbound_amount.py)
# ---------------------------------------------------------------------------

_TRANSIENT_MARKERS = (
    "503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED",
    "500", "INTERNAL", "deadline", "timeout", "TimeoutError",
    "connection", "Connection",
)


def _is_transient_error(exc: Exception) -> bool:
    s = str(exc)
    return any(m in s for m in _TRANSIENT_MARKERS)


def _retry_call(call, *, label: str, max_retries: int = 6, base_delay: float = 4.0):
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
# Per-clip merged interval builder (the boundary-safe padding logic)
# ---------------------------------------------------------------------------

def build_merged_clip_intervals(
    segments: list[tuple[float, float]],
    video_durations: list[tuple[Path, float]],
    padding: float = 2.0,
) -> list[dict]:
    """Convert session-time (start, end) segments into per-clip merged intervals.

    Output is a list of dicts, one per merged interval, in chronological
    session-time order:

        {
            "clip_path":   Path,    # source .mp4 file containing this interval
            "clip_start":  float,   # offset within clip_path  (seconds)
            "clip_end":    float,   # offset within clip_path  (seconds)
            "session_start": float, # absolute session-time of clip_start
            "session_end":   float, # absolute session-time of clip_end
        }

    Boundary safety:
      * Each raw segment is anchored to the .mp4 containing its RAW start.
      * `clip_start = max(0, raw_start - padding)` — clamped to that file.
      * `clip_end`:
          - if raw_end falls inside the same file, `min(file_dur, raw_end + padding)`
          - otherwise the segment crosses files; clamp to the file's EOF
            (ignore the cross-file remainder rather than bleed into the next
            clip).
      * Padded intervals on the SAME clip are merged if they overlap or
        touch.
      * No frame from clip A is ever used to pad a segment that started in
        clip B.
    """
    if not segments:
        return []

    cum_starts: list[float] = []
    el = 0.0
    for _, d in video_durations:
        cum_starts.append(el)
        el += d

    # path string -> list of (clip_start, clip_end) intervals
    by_clip: dict[str, list[tuple[float, float]]] = {}
    for raw_s, raw_e in segments:
        rs = cumulative_to_video_offset(raw_s, video_durations)
        if rs is None:
            continue
        vp, ros = rs
        vi = next(i for i, (p, _) in enumerate(video_durations) if p == vp)
        vd = video_durations[vi][1]

        clip_start = max(0.0, ros - padding)

        re_ = cumulative_to_video_offset(raw_e, video_durations)
        if re_ is not None and re_[0] == vp:
            clip_end = min(vd, re_[1] + padding)
        else:
            clip_end = vd  # spans clips → clamp to EOF of starting clip

        if clip_end <= clip_start:
            continue
        by_clip.setdefault(str(vp), []).append((clip_start, clip_end))

    if not by_clip:
        return []

    path_by_str = {str(p): p for p, _ in video_durations}

    merged: list[dict] = []
    for vk, ivs in by_clip.items():
        ivs.sort()
        merged_ivs: list[tuple[float, float]] = []
        for s, e in ivs:
            if merged_ivs and s <= merged_ivs[-1][1]:
                merged_ivs[-1] = (merged_ivs[-1][0], max(merged_ivs[-1][1], e))
            else:
                merged_ivs.append((s, e))
        clip_path = path_by_str[vk]
        vi = next(i for i, (p, _) in enumerate(video_durations) if p == clip_path)
        clip_cum = cum_starts[vi]
        for cs, ce in merged_ivs:
            merged.append({
                "clip_path": clip_path,
                "clip_start": cs,
                "clip_end": ce,
                "session_start": clip_cum + cs,
                "session_end": clip_cum + ce,
            })

    merged.sort(key=lambda x: x["session_start"])
    return merged


# ---------------------------------------------------------------------------
# ffmpeg: extract per-interval clips and concatenate
# ---------------------------------------------------------------------------

def _run_ffmpeg(cmd: list[str], label: str) -> None:
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed ({label}): {res.stderr[-400:].strip()}"
        )


def extract_and_concat(
    intervals: list[dict],
    out_dir: Path,
    *,
    target_fps: float = 2.0,
    crf: int = 28,
) -> tuple[Path, float]:
    """Extract each interval to its own .mp4 and concatenate them.

    Re-encodes everything with the same codec/params so the concat demuxer
    can stream-copy the result. Returns (concat_path, total_dur_seconds).

    Per-part files live in `out_dir/parts/`; the final concat lives at
    `out_dir/concat.mp4`. The concat list file is also kept for debugging.
    """
    parts_dir = out_dir / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    concat_list = out_dir / "concat_list.txt"
    concat_path = out_dir / "concat.mp4"

    # Wipe stale parts so the concat list is always consistent with what's
    # on disk (e.g. if the segment set changed due to an actions.json edit).
    for old in parts_dir.glob("part_*.mp4"):
        old.unlink()

    part_paths: list[Path] = []
    total_dur = 0.0
    for i, iv in enumerate(intervals, 1):
        dur = iv["clip_end"] - iv["clip_start"]
        if dur <= 0:
            continue
        part_path = parts_dir / f"part_{i:03d}.mp4"
        _run_ffmpeg(
            [
                "ffmpeg", "-y",
                # Fast seek before -i, then -t for length. Re-encode handles
                # keyframe-aligned cuts so the actual cut is accurate.
                "-ss", f"{iv['clip_start']:.3f}",
                "-i", str(iv["clip_path"]),
                "-t", f"{dur:.3f}",
                "-vf", f"fps={target_fps}",
                "-c:v", "libx264",
                "-preset", "veryfast",
                "-crf", str(crf),
                "-pix_fmt", "yuv420p",  # ensures concat compatibility
                "-an",
                str(part_path),
            ],
            label=f"extract part {i}",
        )
        part_paths.append(part_path)
        total_dur += dur

    if not part_paths:
        raise RuntimeError("no parts produced")

    # Build concat demuxer list (single-quoted absolute paths).
    lines = [f"file '{p.resolve()}'" for p in part_paths]
    concat_list.write_text("\n".join(lines) + "\n")

    _run_ffmpeg(
        [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", str(concat_list),
            "-c", "copy",
            str(concat_path),
        ],
        label="concat",
    )

    return concat_path, total_dur


# ---------------------------------------------------------------------------
# Prompt builder (multi-clip aware)
# ---------------------------------------------------------------------------

def build_item_video_prompt(
    item_name: str,
    unit: str,
    package_amount: str | None,
    intervals: list[dict],
    *,
    prior_remaining: float | None = None,
    visible_during_interaction: bool = True,
    actions: list[dict] | None = None,
) -> str:
    """Prompt for clip-video upper bound.

    Critical: explicitly tells the model that the attached video is a
    concatenation of N short clips taken from disjoint moments in a longer
    cooking session. Each clip's original session time range is listed in
    order so the model treats clip boundaries as time jumps.
    """
    unit_label = "grams" if unit == "g" else "count"

    n_clips = len(intervals)
    total_dur = sum(iv["session_end"] - iv["session_start"] for iv in intervals)

    # Map session-aligned action labels onto each clip so the model knows
    # what is happening at each cut, not just where the cut is in time.
    actions = actions or []
    clip_lines: list[str] = []
    for i, iv in enumerate(intervals, 1):
        labels = []
        for a in actions:
            # An action is associated with a clip if it overlaps the clip's
            # session-time range at all.
            if (a.get("end", 0) >= iv["session_start"]
                    and a.get("start", 0) <= iv["session_end"]):
                lab = (a.get("action") or "").strip()
                stage = (a.get("stage") or "").strip()
                if lab or stage:
                    labels.append(f"{lab} [{stage}]" if stage else lab)
        label_str = "; ".join(labels) if labels else "(no annotated action)"
        clip_lines.append(
            f"  Clip {i}: session t={iv['session_start']:.1f}s–{iv['session_end']:.1f}s "
            f"({iv['session_end']-iv['session_start']:.1f}s) — {label_str}"
        )
    clips_block = "\n".join(clip_lines)

    context_lines: list[str] = []
    if package_amount:
        context_lines.append(f'Package size (label, full container): {package_amount}.')
    if prior_remaining is not None:
        if visible_during_interaction:
            context_lines.append(
                f'Last estimate (from a previous VLM run): ~{prior_remaining:.0f} {unit_label}. '
                f'This is a NOISY PRIOR, not a measurement — use the current video as the '
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
            context_lines.append(
                'Not yet observed in any prior session — read the current fill level '
                'directly from the video.'
            )
        else:
            context_lines.append(
                'Not yet observed in any prior session — assume the container is at '
                'package capacity and estimate dispensed amount from the action.'
            )
    context_line = "\n".join(context_lines)

    obs_instruction = (
        "Observe the amount change and remaining amount directly from the video. "
        "If a prior estimate was provided, treat it as a soft hint only — your reading "
        "of the current video takes precedence."
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

    return f"""You are analyzing an egocentric kitchen video recorded with smart glasses.

IMPORTANT — about the attached video:
The attached video is NOT continuous footage. It is a HARD-CUT CONCATENATION of \
{n_clips} short clip(s) extracted from {total_dur:.1f}s of total footage out of a \
longer cooking session. Every transition between clips is a TIME JUMP — the kitchen \
state, the items in hand, and the camera pose can change abruptly across a cut. \
Do NOT treat motion across a cut as a single continuous action.

The clips appear in chronological session order. Their original session-time ranges \
(and the annotated actions inside each one) are:

{clips_block}

All {n_clips} clip(s) show moments where "{item_name}" was being handled.

{context_line}
{obs_instruction}

Estimate, for "{item_name}":
1. How much was used in this session (in {unit_label})
2. How much remains after this session (in {unit_label})

Think step by step:
- In which clip(s) can you see the item or its container/packaging?
- What actions are performed with this item?
- How much is being taken out, poured, cut off, etc.?
- {rem_instruction}

Output your answer as JSON:
```json
{{
  "reasoning": "<your step-by-step reasoning, referencing clip numbers>",
  "evidence_clips": [<list of clip numbers (1-indexed) supporting your estimate>],
  "amount_used": <number>,
  "amount_remaining": <number>
}}
```"""


def parse_item_response(response_text: str) -> dict:
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
# Gemini client (Files API video upload)
# ---------------------------------------------------------------------------

class GeminiVideoAmountClient:
    def __init__(self, model: str = "gemini-2.5-pro"):
        from google import genai
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Missing GOOGLE_API_KEY")
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def _upload(self, video_path: Path):
        video_file = self.client.files.upload(file=str(video_path))
        while video_file.state == "PROCESSING":
            time.sleep(1)
            video_file = self.client.files.get(name=video_file.name)
        if video_file.state == "FAILED":
            raise ValueError(f"Video processing failed: {video_file.name}")
        return video_file

    def estimate_amount(
        self,
        video_path: Path,
        prompt: str,
        *,
        thinking_budget: int = 8192,
    ) -> dict:
        from google.genai import types

        t0 = time.time()
        # Each item gets a fresh upload — we can't reuse across items because
        # each item has its own concat video. Cleanup happens implicitly via
        # Gemini's 48h Files API TTL.
        video_file = _retry_call(
            lambda: self._upload(video_path),
            label=f"upload {video_path.name}",
        )

        contents = [
            types.Part.from_uri(file_uri=video_file.uri, mime_type="video/mp4"),
            prompt,
        ]
        response = _retry_call(
            lambda: self.client.models.generate_content(
                model=self.model, contents=contents,
                config=types.GenerateContentConfig(
                    temperature=0.3, max_output_tokens=4096,
                    thinking_config=types.ThinkingConfig(
                        thinking_budget=thinking_budget, include_thoughts=True,
                    ),
                ),
            ),
            label=f"Gemini {self.model}",
        )

        stats = {
            "inference_time_s": round(time.time() - t0, 2),
            "model": self.model,
        }
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
        return {
            **parsed,
            "thinking": thinking_text,
            "raw_response": response_text,
            "prompt": prompt,
            "stats": stats,
        }


# ---------------------------------------------------------------------------
# Per-session pipeline
# ---------------------------------------------------------------------------

def process_session(
    participant: str,
    session: str,
    client: GeminiVideoAmountClient,
    ledger: dict,
    *,
    prior_estimates: dict | None = None,
    model_tag: str = "default",
    run_tag: str = "default",
    only_item: str | None = None,
    target_fps: float = 2.0,
    padding: float = 2.0,
) -> list[dict]:
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

    predictions: list[dict] = []
    items = ledger["items"]

    for iid, state in snap.items():
        if only_item is not None and iid != only_item:
            continue
        used = state.get("used")
        remaining_gt = state.get("remaining")
        # Remaining-only annotations still score CNPE_rem even without a
        # `used` value — keep them.
        if used is None and remaining_gt is None:
            continue

        item = items.get(iid, {})
        visual_class = item.get("visual_class", iid)
        unit = item.get("unit", "g")

        # Strip VISIBLE_PORTION segments — same policy as upperbound_amount.py.
        item_actions = [
            a for a in actions
            if a["item"] == iid
            and (a.get("stage") or "").upper() != "VISIBLE_PORTION"
        ]
        if not item_actions:
            print(f"    {visual_class}: no amount-changing action segments, skipping")
            continue

        intervals = build_merged_clip_intervals(
            [(a["start"], a["end"]) for a in item_actions],
            video_durations,
            padding=padding,
        )
        if not intervals:
            print(f"    {visual_class}: no valid intervals after clamping, skipping")
            continue

        total_clip_dur = sum(iv["session_end"] - iv["session_start"] for iv in intervals)
        print(
            f"    {visual_class}: {len(intervals)} merged clip(s) ({total_clip_dur:.1f}s)...",
            end=" ", flush=True,
        )

        item_dir = cache / iid
        item_dir.mkdir(parents=True, exist_ok=True)
        try:
            concat_path, _ = extract_and_concat(
                intervals, item_dir, target_fps=target_fps,
            )
        except Exception as e:
            print(f"FAIL (ffmpeg): {e}")
            continue

        pkg = item.get("package_amount", "")
        prior_rem = prior_estimates.get(iid)
        visible = item.get("visible_during_interaction", True)

        prompt_text = build_item_video_prompt(
            visual_class, unit, pkg, intervals,
            prior_remaining=prior_rem,
            visible_during_interaction=visible,
            actions=item_actions,
        )
        (item_dir / "prompt.txt").write_text(prompt_text)

        try:
            result = client.estimate_amount(
                concat_path, prompt_text,
                thinking_budget=getattr(client, "_thinking_budget", 8192),
            )
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"FAIL: {e}")
            continue

        stats = result["stats"]
        print(
            f"used={result['predicted_used']}, rem={result['predicted_remaining']} "
            f"({stats.get('inference_time_s','?')}s, {stats.get('total_tokens','?')} tok)"
        )

        intervals_meta = [
            {
                "clip_path": str(iv["clip_path"]),
                "clip_start": iv["clip_start"],
                "clip_end": iv["clip_end"],
                "session_start": iv["session_start"],
                "session_end": iv["session_end"],
            }
            for iv in intervals
        ]

        pred_entry = {
            "session": session,
            "item": visual_class,
            "instance_id": iid,
            "amount_used": result["predicted_used"],
            "amount_remaining": result["predicted_remaining"],
            "reasoning": result["reasoning"],
            "thinking": result["thinking"],
            "stats": stats,
            "intervals": intervals_meta,
        }
        predictions.append(pred_entry)

        log_entry = {
            **pred_entry,
            "prompt": result["prompt"],
            "raw_response": result["raw_response"],
            "concat_video": str(concat_path),
            "n_clips": len(intervals),
            "total_clip_duration_s": total_clip_dur,
            "gt_used": state.get("used"),
            "gt_remaining": state.get("remaining"),
            "package_amount": pkg,
            "starting_amount": state.get("starting"),
        }
        (item_dir / "log.json").write_text(
            json.dumps(log_entry, indent=2, ensure_ascii=False) + "\n"
        )

    return predictions


# ---------------------------------------------------------------------------
# Main CLI (mirrors upperbound_amount.py: --tag/--session/--until/--item/--resume)
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Upper-bound VLM amount estimation (concatenated video variant)"
    )
    parser.add_argument("--participant", required=True)
    parser.add_argument("--tag", required=True,
                        help="REQUIRED short label for this run; embedded in output "
                             "filenames and cache paths.")
    parser.add_argument("--session", help="Process single session (default: all)")
    parser.add_argument("--until", help="Run sessions up to and including this one")
    parser.add_argument("--item",
                        help="Run only this instance_id. Finds every session that "
                             "uses the item and processes them in chronological "
                             "order. Same --tag overrides ONLY this item's entries "
                             "in the predictions file; other items untouched.")
    parser.add_argument("--model", default="gemini-2.5-pro")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--thinking-budget", type=int, default=8192)
    parser.add_argument("--target-fps", type=float, default=2.0,
                        help="fps to re-encode each clip at before concat (default 2.0)")
    parser.add_argument("--padding", type=float, default=2.0,
                        help="seconds of padding around each GT segment, clamped to "
                             "the source clip's bounds (default 2.0)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume a previously-interrupted run with the same --tag.")
    args = parser.parse_args()

    # ffmpeg presence check — fail early with a useful message rather than
    # in the middle of the first session.
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise SystemExit("ffmpeg/ffprobe not found on PATH")

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
            and (snapshots[s][args.item].get("used") is not None
                 or snapshots[s][args.item].get("remaining") is not None)
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
        / f"upperbound_video_{model_tag}_{run_tag}_preds.json"
    )

    client = GeminiVideoAmountClient(model=args.model)
    client._thinking_budget = args.thinking_budget

    running_estimates: dict[str, float] = {}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    status_path = output_path.with_name(
        output_path.stem.replace("_preds", "_status") + ".json"
    )

    all_predictions: list[dict] = []
    status: dict = {"completed_sessions": [], "failed_sessions": []}
    if args.resume and output_path.exists() and status_path.exists():
        all_predictions = json.loads(output_path.read_text())
        status = json.loads(status_path.read_text())
        completed = set(status.get("completed_sessions", []))
        for p in sorted(all_predictions, key=lambda x: x.get("session", "")):
            if p.get("amount_remaining") is not None:
                running_estimates[p["instance_id"]] = p["amount_remaining"]
        pending = [s for s in sessions if s not in completed]
        print(f"\nRESUME: {len(completed)} session(s) already complete, "
              f"{len(pending)} pending (of {len(sessions)} total).")
        sessions = pending
    elif args.resume:
        print("\nRESUME requested but no existing predictions/status found — starting fresh.")

    # --item override mode (same semantics as upperbound_amount.py).
    if args.item:
        if not args.resume and output_path.exists():
            all_predictions = json.loads(output_path.read_text())
        running_estimates = {}
        first_target = sessions[0] if sessions else None
        if first_target is not None:
            for p in sorted(all_predictions, key=lambda x: x.get("session", "")):
                if (p.get("instance_id") == args.item
                        and p.get("session", "") < first_target
                        and p.get("amount_remaining") is not None):
                    running_estimates[args.item] = p["amount_remaining"]

    failed_sessions: list[tuple[str, str]] = []
    for session in sessions:
        try:
            preds = process_session(
                args.participant, session, client, ledger,
                prior_estimates=running_estimates,
                model_tag=model_tag, run_tag=run_tag,
                only_item=args.item,
                target_fps=args.target_fps,
                padding=args.padding,
            )
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
            print(f"\n  HALTED at session {session}. Fix and re-run with --resume.")
            break

        for p in preds:
            if p.get("amount_remaining") is not None:
                running_estimates[p["instance_id"]] = p["amount_remaining"]

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
        print("\nRe-run with --resume to retry the failed sessions only.")

    from importlib import import_module
    eval_mod = import_module("evaluate_amount")
    gt = eval_mod.extract_ground_truth(ledger)
    report = eval_mod.evaluate(gt, all_predictions)

    eval_path = output_path.with_name(output_path.stem + "_eval.json")
    eval_mod.write_report(report, eval_path)
    eval_mod.print_eval_table(report)


if __name__ == "__main__":
    main()

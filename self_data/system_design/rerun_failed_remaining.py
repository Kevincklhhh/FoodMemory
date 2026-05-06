#!/usr/bin/env python3
"""Re-run failed lowerbound_remaining sessions with a per-call video cap.

For sessions whose videos exceed --max-minutes (default 30), this script
splits the 1fps videos into ≤max-minutes chunks, sends each chunk as a
separate API call, and merges predictions (last observation wins per item).

After re-running, it patches the main predictions file and re-evaluates.

Usage:
  python rerun_failed_remaining.py --participant kailai --tag remaining_batch_v1 \
      --sessions 20260318-181229 20260402-173309 20260402-185400 20260404-192616
"""

import argparse
import json
import math
import os
import re
import subprocess
import time
from pathlib import Path

from dotenv import load_dotenv

from lowerbound_remaining import (
    CACHE_DIR,
    GeminiWholeVideoClient,
    build_session_prompt,
    convert_to_1fps,
    get_video_duration,
    group_videos_by_duration,
    split_video,
)
from lowerbound_remaining_batch import resolve_predictions
from utils import load_ledger, load_session_inventory, participant_dir

_KITCHEN_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(_KITCHEN_DIR / ".env")

MAX_MINUTES_DEFAULT = 30


def process_session_chunked(
    participant: str,
    session: str,
    client: GeminiWholeVideoClient,
    ledger: dict,
    max_seconds: float,
    thinking_budget: int = 8192,
) -> tuple[list[dict], list[dict]]:
    """Process one session with chunked video, return (resolved_preds, chunk_logs)."""
    snap = ledger.get("snapshots", {}).get(session, {})
    if not snap:
        print(f"  SKIP {session}: no snapshot")
        return [], []

    inventory = load_session_inventory(participant, session)
    if not inventory:
        print(f"  SKIP {session}: no inventory")
        return [], []

    visible_inventory = [inv for inv in inventory if inv.get("visible_during_interaction", True)]
    if not visible_inventory:
        print(f"  SKIP {session}: no visible items")
        return [], []

    # Get 1fps videos from cache
    video_cache = CACHE_DIR / participant / session
    video_cache.mkdir(parents=True, exist_ok=True)

    vdir = participant_dir(participant) / "videos" / session
    session_videos = sorted(vdir.glob("*.mp4"))
    if not session_videos:
        print(f"  SKIP {session}: no videos")
        return [], []

    fps_videos = []
    for vp in session_videos:
        fps_path = video_cache / f"{vp.stem}_1fps.mp4"
        if not fps_path.exists():
            print(f"    Converting {vp.name} to 1fps...", end=" ", flush=True)
            if convert_to_1fps(vp, fps_path):
                print("OK")
            else:
                print("FAILED")
                continue
        fps_videos.append(fps_path)

    if not fps_videos:
        print(f"  SKIP {session}: no 1fps videos")
        return [], []

    total_dur = sum(get_video_duration(v) for v in fps_videos)
    print(f"  {session}: {len(fps_videos)} 1fps videos ({total_dur:.0f}s), "
          f"{len(visible_inventory)} visible items")

    # Build chunks: split single long videos, or group multiple short ones
    split_dir = video_cache / "splits"
    split_dir.mkdir(exist_ok=True)

    all_parts: list[Path] = []
    for v in fps_videos:
        parts = split_video(v, max_seconds, split_dir)
        all_parts.extend(parts)

    chunks = group_videos_by_duration(all_parts, max_seconds)
    print(f"    Split into {len(chunks)} chunk(s): "
          + ", ".join(f"{sum(get_video_duration(v) for v in c):.0f}s" for c in chunks))

    # Query each chunk
    prompt_text = build_session_prompt(visible_inventory)
    chunk_logs = []
    all_raw_preds: dict[str, dict] = {}  # instance_id -> best prediction

    for i, chunk_videos in enumerate(chunks):
        chunk_dur = sum(get_video_duration(v) for v in chunk_videos)
        vnames = [v.name for v in chunk_videos]
        print(f"    Chunk {i+1}/{len(chunks)} ({chunk_dur:.0f}s, {len(chunk_videos)} files): "
              f"{vnames[0]}..{vnames[-1]}", end=" ", flush=True)

        result = client.query_session(
            chunk_videos, visible_inventory,
            thinking_budget=thinking_budget, prompt=prompt_text,
        )
        stats = result["stats"]
        print(f"-> {len(result['predictions'])} items "
              f"({stats.get('inference_time_s', '?')}s, "
              f"{stats.get('total_tokens', '?')} tok)")

        chunk_logs.append({
            "chunk": i,
            "videos": [str(v) for v in chunk_videos],
            "duration": chunk_dur,
            "predictions": result["predictions"],
            "thinking": result["thinking"],
            "raw_response": result["raw_response"],
            "stats": stats,
        })

        # Merge: later chunks overwrite earlier ones per instance_id
        for pred in result["predictions"]:
            iid = (pred.get("instance_id") or "").strip()
            if iid:
                all_raw_preds[iid] = pred
            else:
                name = (pred.get("item") or "").strip().lower()
                all_raw_preds[f"__name__{name}"] = pred

    # Resolve predictions
    merged_raw = list(all_raw_preds.values())
    resolved = resolve_predictions(session, merged_raw, visible_inventory)

    print(f"    Merged: {len(resolved)} predictions from {len(chunks)} chunks")
    return resolved, chunk_logs


def main():
    parser = argparse.ArgumentParser(description="Re-run failed lowerbound_remaining sessions with video cap")
    parser.add_argument("--participant", required=True)
    parser.add_argument("--tag", required=True, help="Original batch tag (e.g. remaining_batch_v1)")
    parser.add_argument("--sessions", nargs="+", required=True, help="Session IDs to re-run")
    parser.add_argument("--max-minutes", type=float, default=MAX_MINUTES_DEFAULT,
                        help=f"Max video duration per API call in minutes (default: {MAX_MINUTES_DEFAULT})")
    parser.add_argument("--model", default="gemini-2.5-pro")
    parser.add_argument("--thinking-budget", type=int, default=8192)
    args = parser.parse_args()

    ledger = load_ledger(args.participant)
    max_seconds = args.max_minutes * 60

    model_tag = args.model.replace("/", "_")
    run_tag = args.tag.replace("/", "_")

    # Main predictions file to patch
    output_path = (
        participant_dir(args.participant) / "outputs"
        / f"lowerbound_remaining_{model_tag}_{run_tag}_preds.json"
    )

    # Batch cache dir for saving logs alongside originals
    batch_dir = CACHE_DIR / args.participant / f"_batch_{model_tag}_{run_tag}"

    client = GeminiWholeVideoClient(model=args.model)

    new_predictions: list[dict] = []
    for session in args.sessions:
        print(f"\n{'='*60}")
        print(f"Re-running {session} (max {args.max_minutes}m per call)")
        print(f"{'='*60}")

        preds, chunk_logs = process_session_chunked(
            args.participant, session, client, ledger,
            max_seconds=max_seconds,
            thinking_budget=args.thinking_budget,
        )
        new_predictions.extend(preds)

        # Save chunk log
        log_path = batch_dir / f"{session}_rerun_log.json"
        log = {
            "session": session,
            "model": args.model,
            "max_minutes": args.max_minutes,
            "chunks": chunk_logs,
            "merged_predictions": [p for p in preds],
        }
        log_path.write_text(json.dumps(log, indent=2, ensure_ascii=False) + "\n")
        print(f"  Log saved: {log_path}")

    # Patch main predictions file
    if output_path.exists():
        existing = json.loads(output_path.read_text())
        rerun_sessions = set(args.sessions)
        # Remove old (empty) entries for these sessions, keep everything else
        kept = [p for p in existing if p["session"] not in rerun_sessions]
        patched = kept + new_predictions
        output_path.write_text(json.dumps(patched, indent=2) + "\n")
        print(f"\nPatched {output_path}: {len(kept)} existing + {len(new_predictions)} new = {len(patched)} total")
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(new_predictions, indent=2) + "\n")
        print(f"\nSaved {len(new_predictions)} predictions to {output_path}")

    # Re-run evaluation
    from importlib import import_module
    eval_mod = import_module("evaluate_amount")
    all_preds = json.loads(output_path.read_text())
    gt = eval_mod.extract_ground_truth(ledger)
    report = eval_mod.evaluate(gt, all_preds)

    eval_path = output_path.with_name(output_path.stem + "_eval.json")
    eval_mod.write_report(report, eval_path)
    eval_mod.print_eval_table(report)


if __name__ == "__main__":
    main()

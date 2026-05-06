#!/usr/bin/env python3
"""Batch version of lowerbound_remaining.py using Gemini Batch API.

Uploads all 1fps videos, builds a JSONL batch request (one per session),
submits to Gemini Batch API (50% cheaper), polls for completion, then
parses results and runs evaluation.

Sessions are independent (no prior chain), so all can run in one batch.

Usage:
  # Submit batch
  python lowerbound_remaining_batch.py --participant kailai --tag remaining_v1

  # Check status / collect results from a previous batch
  python lowerbound_remaining_batch.py --participant kailai --tag remaining_v1 --collect

  # Use specific model
  python lowerbound_remaining_batch.py --participant kailai --tag remaining_v1 --model gemini-2.5-flash
"""

import argparse
import json
import mimetypes
import os
import re
import subprocess
import time
from pathlib import Path

from dotenv import load_dotenv

from lowerbound_remaining import (
    CACHE_DIR,
    _retry_call,
    build_session_prompt,
    build_session_prompt_v2,
    concat_1fps_videos,
    convert_to_1fps,
    get_video_duration,
    group_videos_by_duration,
    split_video,
)
from utils import load_inventory, load_ledger, participant_dir

_KITCHEN_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(_KITCHEN_DIR / ".env")

# Workaround: genai SDK may not recognize .jsonl mime type
mimetypes.add_type("application/jsonl", ".jsonl")

MAX_MINUTES_DEFAULT = 30
THINKING_BUDGET_DEFAULT = 8192


# ---------------------------------------------------------------------------
# Phase 1: Prepare — convert videos, upload, build JSONL
# ---------------------------------------------------------------------------

def prepare_session(
    participant: str,
    session: str,
    ledger: dict,
    client,
    video_cache: Path,
    max_seconds: float,
    inventory_scope: str = "full",
    prompt_variant: str = "v1",
) -> dict | None:
    """Prepare one session: convert to 1fps, concat, split into ≤max_seconds
    chunks, upload each chunk, build per-chunk request metadata.

    Returns a dict with session metadata + chunk descriptors, or None to skip.
    """
    snap = ledger.get("snapshots", {}).get(session, {})
    if not snap:
        print(f"  SKIP {session}: no snapshot")
        return None

    inventory = load_inventory(participant, session, scope=inventory_scope)
    if not inventory:
        print(f"  SKIP {session}: no inventory for scope={inventory_scope}")
        return None

    visible_inventory = [inv for inv in inventory if inv.get("visible_during_interaction", True)]
    n_opaque = len(inventory) - len(visible_inventory)
    if not visible_inventory:
        print(f"  SKIP {session}: no visible items ({n_opaque} opaque filtered)")
        return None

    vdir = participant_dir(participant) / "videos" / session
    session_videos = sorted(vdir.glob("*.mp4"))
    if not session_videos:
        print(f"  SKIP {session}: no videos")
        return None

    total_dur = sum(get_video_duration(v) for v in session_videos)

    # Convert to 1fps (shared cache)
    sess_cache = video_cache / session
    sess_cache.mkdir(parents=True, exist_ok=True)

    fps_videos = []
    for vp in session_videos:
        fps_path = sess_cache / f"{vp.stem}_1fps.mp4"
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
        return None

    # Concatenate all clips into one mp4 so Gemini sees a single contiguous
    # video with session-absolute timestamps.
    concat_path = sess_cache / f"{session}_1fps_concat.mp4"
    if not concat_path.exists():
        print(f"    Concatenating {len(fps_videos)} clip(s) -> {concat_path.name}...",
              end=" ", flush=True)
        if not concat_1fps_videos(fps_videos, concat_path):
            print("FAILED")
            print(f"  SKIP {session}: concat failed")
            return None
        print(f"OK ({concat_path.stat().st_size / (1024*1024):.1f}MB)")

    # Split concatenated video into ≤max_seconds chunks (one if short enough)
    split_dir = sess_cache / "splits"
    split_dir.mkdir(exist_ok=True)
    parts = split_video(concat_path, max_seconds, split_dir)
    chunk_groups = group_videos_by_duration(parts, max_seconds)

    # Upload each chunk; one chunk = one Gemini API call. Each upload + the
    # subsequent state-polling go through `_retry_call`, which backs off on
    # 503/UNAVAILABLE / 429 / timeout / connection errors. Once all chunks are
    # uploaded for a session, the chunk descriptors (incl. file_uris) are
    # written to a checkpoint so a crash mid-batch doesn't force re-uploading
    # earlier sessions; the next run picks them up if the URIs are still valid
    # (Gemini Files API keeps files ~48h).
    upload_ckpt = sess_cache / "uploaded_chunks.json"
    cached_chunks: list[dict] | None = None
    if upload_ckpt.exists():
        try:
            cached = json.loads(upload_ckpt.read_text())
            if cached.get("max_seconds") == max_seconds and cached.get("chunks"):
                # Verify the first URI is still alive (Files API entries expire).
                probe_uri = cached["chunks"][0]["file_uris"][0]
                probe_name = probe_uri.rsplit("/", 1)[-1]
                try:
                    _retry_call(
                        lambda: client.files.get(name=f"files/{probe_name}"),
                        label=f"Verify cached upload {probe_name}",
                        max_retries=3, base_delay=2.0,
                    )
                    cached_chunks = cached["chunks"]
                    print(f"    Reusing {len(cached_chunks)} cached chunk upload(s)")
                except Exception:
                    print(f"    Cached uploads expired; re-uploading")
        except Exception:
            pass

    chunks: list[dict] = cached_chunks or []
    if not cached_chunks:
        t_cursor = 0.0
        for i, group in enumerate(chunk_groups):
            chunk_uris = []
            chunk_dur = 0.0
            for vp in group:
                print(f"    Uploading chunk {i} part {vp.name}...", end=" ", flush=True)
                vf = _retry_call(
                    lambda vp=vp: client.files.upload(file=str(vp)),
                    label=f"Upload {vp.name}",
                )
                while vf.state == "PROCESSING":
                    time.sleep(1)
                    vf = _retry_call(
                        lambda vf=vf: client.files.get(name=vf.name),
                        label=f"Poll {vf.name}",
                    )
                if vf.state == "FAILED":
                    print("FAILED")
                    print(f"  SKIP {session}: upload failed")
                    return None
                chunk_uris.append(vf.uri)
                chunk_dur += get_video_duration(vp)
                print(f"OK ({vf.name})")
            chunks.append({
                "index": i,
                "file_uris": chunk_uris,
                "duration": chunk_dur,
                "t_start": t_cursor,
                "t_end": t_cursor + chunk_dur,
                "videos": [str(v) for v in group],
            })
            t_cursor += chunk_dur

        # Persist URIs so a later crash doesn't waste this work.
        upload_ckpt.write_text(json.dumps(
            {"max_seconds": max_seconds, "chunks": chunks}, indent=2,
        ) + "\n")

    if prompt_variant == "v2":
        prompt_text = build_session_prompt_v2(visible_inventory)
    else:
        prompt_text = build_session_prompt(visible_inventory)

    print(f"  {session}: {len(session_videos)} videos ({total_dur:.0f}s), "
          f"{len(visible_inventory)} visible items ({n_opaque} opaque filtered), "
          f"{len(chunks)} chunk(s)")

    return {
        "session": session,
        "inventory": inventory,
        "visible_inventory": visible_inventory,
        "prompt": prompt_text,
        "chunks": chunks,
        "total_duration": total_dur,
        "n_videos": len(session_videos),
    }


def build_jsonl_requests(
    session_info: dict,
    thinking_budget: int,
    thinking_level: str | None = None,
    media_resolution: str | None = None,
) -> list[dict]:
    """Build one JSONL line per chunk for the batch request.

    Each chunk becomes a separate Gemini API call; key encodes session + chunk
    index as `{session}__chunk{i}` so results can be regrouped on collect.

    Gemini 3.x: pass `thinking_level` (low/medium/high) instead of
    `thinking_budget`; the two parameters are mutually exclusive (the API
    rejects requests that include both).

    `media_resolution` (Gemini 3 only) sets the per-frame token allocation for
    video parts. For our 1 fps footage, 'media_resolution_low' (= 70 tok/frame)
    is the right default; bump to 'media_resolution_high' (= 280 tok/frame)
    only if reading dense package text or fine fill-line detail demands it.
    """
    lines = []
    for chunk in session_info["chunks"]:
        parts = [
            {"file_data": {"mime_type": "video/mp4", "file_uri": uri}}
            for uri in chunk["file_uris"]
        ]
        parts.append({"text": session_info["prompt"]})

        # `thinking_level` is Gemini-3-only and the batch REST path is picky
        # about it. Use full camelCase + uppercase enum value (protobuf
        # convention): `thinkingConfig.thinkingLevel` = "HIGH" / "MEDIUM" /
        # "LOW". Verified working empirically; snake_case + lowercase value
        # both rejected with 400 INVALID_ARGUMENT.
        if thinking_level:
            thinking_cfg = {
                "thinkingLevel": thinking_level.upper(),
                "includeThoughts": True,
            }
        else:
            thinking_cfg = {
                "thinkingBudget": thinking_budget,
                "includeThoughts": True,
            }

        # Gemini-3: media_resolution is set GLOBALLY via generation_config.
        # Per-part media_resolution is rejected on the REST/batch path with
        # 400 INVALID_ARGUMENT. Camel-case key matches Gemini's protobuf
        # field name (`mediaResolution`). For 1 fps video the recommended
        # value is `media_resolution_high` (= 280 tok/frame).
        # On Gemini 3, max_output_tokens caps `visible + thinking` combined.
        # With thinking_level=high on whole-video LB inputs (1000+ frames),
        # the model's thinking can hit 31K+ tokens on long sessions and
        # truncate visible output mid-JSON. 65536 gives 2× headroom over
        # the empirical max-thinking observed, and costs nothing extra
        # (billed by actual usage). AVP observer keeps the 32K cap because
        # its per-call inputs are smaller (~85 frames, narrower scope).
        gen_cfg = {
            "temperature": 0.3,
            "max_output_tokens": 65536,
            "thinking_config": thinking_cfg,
        }
        if media_resolution:
            gen_cfg["mediaResolution"] = media_resolution.upper()

        lines.append({
            "key": f"{session_info['session']}__chunk{chunk['index']}",
            "request": {
                "contents": [{"role": "user", "parts": parts}],
                "generation_config": gen_cfg,
            },
        })
    return lines


# ---------------------------------------------------------------------------
# Phase 2: Submit batch
# ---------------------------------------------------------------------------

def submit_batch(client, jsonl_path: Path, model: str) -> str:
    """Upload JSONL and submit batch job. Returns batch job name."""
    from google.genai import types

    uploaded = client.files.upload(
        file=str(jsonl_path),
        config=types.UploadFileConfig(
            display_name=jsonl_path.stem,
            mime_type="application/jsonl",
        ),
    )
    print(f"  JSONL uploaded: {uploaded.name}")

    batch_job = client.batches.create(
        model=model,
        src=uploaded.name,
        config={"display_name": jsonl_path.stem},
    )
    print(f"  Batch job created: {batch_job.name}")
    print(f"  State: {batch_job.state}")
    return batch_job.name


# ---------------------------------------------------------------------------
# Phase 3: Poll and collect results
# ---------------------------------------------------------------------------

def poll_batch(client, batch_name: str, poll_interval: int = 60) -> object | None:
    """Poll batch job until terminal state. Returns completed batch or None.

    Tolerates transient server errors (503/UNAVAILABLE etc.) on the polling
    side: a single hiccup on `client.batches.get` retries with backoff
    instead of crashing the whole run while the job is still alive on
    Google's side.
    """
    from datetime import datetime

    consecutive_errs = 0
    while True:
        try:
            batch = client.batches.get(name=batch_name)
            consecutive_errs = 0
        except Exception as e:
            consecutive_errs += 1
            err_str = str(e)
            transient = any(m in err_str for m in (
                "503", "UNAVAILABLE", "500", "INTERNAL",
                "deadline", "timeout", "connection", "Connection",
                "RESOURCE_EXHAUSTED", "429",
            ))
            ts = datetime.now().strftime("%H:%M:%S")
            if transient and consecutive_errs <= 10:
                backoff = min(60 * consecutive_errs, 600)
                print(f"  [{ts}] Polling error (attempt {consecutive_errs}/10): "
                      f"{err_str[:160]}. Retrying in {backoff}s...", flush=True)
                time.sleep(backoff)
                continue
            # Non-transient or too many consecutive errs: surface.
            print(f"  [{ts}] Polling failed: {err_str[:160]}", flush=True)
            raise

        state = str(batch.state)
        if "SUCCEEDED" in state:
            print(f"\n  Batch SUCCEEDED: {batch_name}")
            return batch
        elif "FAILED" in state or "CANCELLED" in state:
            print(f"\n  Batch {state}: {batch_name}")
            return None

        ts = datetime.now().strftime("%H:%M:%S")
        print(f"  [{ts}] State: {state}. Waiting {poll_interval}s...", flush=True)
        time.sleep(poll_interval)


def collect_results(client, batch_name: str) -> dict[str, dict]:
    """Collect results from completed batch. Returns {session_key: response_dict}."""
    batch = client.batches.get(name=batch_name)
    state = str(batch.state)

    if "SUCCEEDED" not in state:
        print(f"  Batch not yet complete: {state}")
        return {}

    result_file_name = batch.dest.file_name
    content_bytes = client.files.download(file=result_file_name)
    content_str = content_bytes.decode("utf-8")

    results = {}
    for line in content_str.strip().split("\n"):
        if not line.strip():
            continue
        entry = json.loads(line)
        key = entry.get("key", "")
        results[key] = entry
    return results


def parse_batch_response(entry: dict) -> dict:
    """Parse one batch response entry into predictions + metadata.

    Handles both v1 (flat JSON array) and v2 (object with `inventory_journey`
    + `items` keys) prompt formats. For v2, the per-item entries come from
    the `items` field, the `inventory_journey` is preserved in metadata
    for forensics, and only `used` items contribute predictions.
    Output tokens include thinking (sum of candidates + thoughts) per the
    canonical convention.
    """
    response = entry.get("response", {})
    candidates = response.get("candidates", [])

    response_text = ""
    thinking_text = ""
    if candidates:
        content = candidates[0].get("content", {})
        for part in content.get("parts", []):
            if part.get("thought"):
                thinking_text += part.get("text", "")
            else:
                response_text += part.get("text", "")

    usage = response.get("usageMetadata", {})
    cands_tok = usage.get("candidatesTokenCount") or 0
    thoughts_tok = usage.get("thoughtsTokenCount") or 0
    stats = {
        "input_tokens": usage.get("promptTokenCount"),
        # Canonical: output includes thinking tokens.
        "output_tokens": cands_tok + thoughts_tok,
        "candidates_tokens": cands_tok,
        "thoughts_tokens": thoughts_tok,
        "total_tokens": usage.get("totalTokenCount"),
    }

    predictions = []
    inventory_journey = None

    # Try v2 first: an object with `items` and (optionally) `inventory_journey`.
    obj_match = re.search(r"\{[\s\S]*\"items\"[\s\S]*\}", response_text)
    if obj_match:
        try:
            obj = json.loads(obj_match.group())
            if isinstance(obj, dict) and isinstance(obj.get("items"), list):
                inventory_journey = obj.get("inventory_journey")
                for it in obj["items"]:
                    if not isinstance(it, dict):
                        continue
                    status = it.get("status")
                    # Skip not_used items — they have no remaining-amount prediction.
                    if status == "not_used":
                        continue
                    # Pick the best amount to commit to (mirrors AVP-minimal logic).
                    a_r = it.get("amount_remaining")
                    a_s = it.get("amount_starting")
                    a_d = it.get("amount_derivative")
                    if a_r is not None:
                        amount = a_r
                    elif a_s is not None and a_d is not None:
                        try:
                            amount = max(0.0, float(a_s) - float(a_d))
                        except (TypeError, ValueError):
                            amount = None
                    elif a_d is not None:
                        amount = a_d
                    elif a_s is not None:
                        amount = a_s
                    else:
                        # used but no amount fields — skip
                        continue
                    predictions.append({
                        "instance_id": it.get("instance_id", ""),
                        "item": it.get("item", ""),
                        "evidence_timestamps": it.get("evidence_timestamps", []) or [],
                        "amount_remaining": amount,
                    })
        except json.JSONDecodeError:
            predictions = []

    # Fall back to v1 (flat JSON array) if v2 didn't yield anything.
    if not predictions:
        json_match = re.search(r"\[[\s\S]*\]", response_text)
        if json_match:
            try:
                predictions = json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

    return {
        "predictions": predictions,
        "inventory_journey": inventory_journey,
        "thinking": thinking_text,
        "raw_response": response_text,
        "stats": stats,
    }


def resolve_predictions(session: str, raw_preds: list[dict], visible_inventory: list[dict]) -> list[dict]:
    """Resolve instance_ids from VLM predictions. Output schema matches the
    sequential script (`lowerbound_remaining.py::process_session`) exactly so
    the eval step is symmetric across batch/sequential runs.
    """
    valid_iids = {inv["instance_id"] for inv in visible_inventory}
    name_to_iids: dict[str, list[str]] = {}
    for inv in visible_inventory:
        name_to_iids.setdefault(inv["visual_class"].lower(), []).append(inv["instance_id"])
    iid_to_name = {inv["instance_id"]: inv["visual_class"] for inv in visible_inventory}

    predictions = []
    for pred in raw_preds:
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
    return predictions


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Batch whole-video VLM remaining-amount estimation")
    parser.add_argument("--participant", required=True)
    parser.add_argument("--tag", required=True,
                        help="Short label for this run (e.g. 'remaining_batch_v1').")
    parser.add_argument("--session", help="Single session (default: all)")
    parser.add_argument("--sessions",
                        help="Comma-separated list of sessions (overrides --session/--until).")
    parser.add_argument("--until", help="Run sessions up to and including this one")
    parser.add_argument("--model", default="gemini-2.5-pro")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--collect", action="store_true",
                        help="Skip submission, just poll/collect results from existing batch.")
    parser.add_argument("--poll-interval", type=int, default=60,
                        help="Seconds between status checks (default: 60)")
    parser.add_argument("--max-minutes", type=float, default=MAX_MINUTES_DEFAULT,
                        help=f"Max video duration per API call in minutes "
                             f"(default: {MAX_MINUTES_DEFAULT}). Sessions exceeding this "
                             f"are split into chunks; one Gemini call per chunk.")
    parser.add_argument("--thinking-budget", type=int, default=THINKING_BUDGET_DEFAULT,
                        help=f"Thinking token budget per chunk (default: {THINKING_BUDGET_DEFAULT}; "
                             f"matches lowerbound_remaining.py). Ignored when --thinking-level is set.")
    parser.add_argument("--thinking-level",
                        choices=["minimal", "low", "medium", "high"],
                        help="Gemini-3 thinking level (overrides --thinking-budget). "
                             "Mutually exclusive with thinking_budget per the API.")
    parser.add_argument("--media-resolution",
                        choices=["media_resolution_low", "media_resolution_medium",
                                 "media_resolution_high", "media_resolution_ultra_high"],
                        help="Gemini-3 per-frame video token allocation. "
                             "low/medium = 70 tok/frame (videos); high = 280 tok/frame. "
                             "Default: model-chosen.")
    parser.add_argument("--inventory-scope", choices=["full", "session"], default="full",
                        help="Which items to list in the Gemini prompt. "
                             "'full' = all items in stock at session time (default); "
                             "'session' = GT-annotated subset only.")
    parser.add_argument("--prompt-variant", choices=["v1", "v2"], default="v1",
                        help="LB prompt variant. v1: flat list, one number per item "
                             "(legacy). v2: AVP-equivalent schema — `inventory_journey` "
                             "of inventory-touching activity blocks + per-item entries "
                             "with status + 3-amount triple. Use v2 for fair "
                             "reasoning-load comparison against AVP-minimal.")
    args = parser.parse_args()
    max_seconds = args.max_minutes * 60

    from google import genai

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Missing GOOGLE_API_KEY")
    client = genai.Client(api_key=api_key)

    ledger = load_ledger(args.participant)
    snapshots = ledger.get("snapshots", {})

    if args.sessions:
        sessions = [s.strip() for s in args.sessions.split(",") if s.strip()]
    elif args.session:
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
    output_path.parent.mkdir(parents=True, exist_ok=True)

    batch_dir = CACHE_DIR / args.participant / f"_batch_{model_tag}_{run_tag}"
    batch_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = batch_dir / "batch_requests.jsonl"
    meta_path = batch_dir / "batch_meta.json"
    batch_name_path = batch_dir / "batch_name.txt"

    # Skip sessions that already have a completed log in this batch_dir
    # (re-runs with the same tag accumulate; delete a `_log.json` to force retry).
    if not args.collect:
        already_done = [s for s in sessions if (batch_dir / f"{s}_log.json").exists()]
        if already_done:
            print(f"  Skipping {len(already_done)} already-completed session(s) in {batch_dir.name}:")
            for s in already_done:
                print(f"    {s}")
        sessions = [s for s in sessions if s not in set(already_done)]
        if not sessions:
            print("\nAll requested sessions already completed. Re-emitting merged output.")
            _rebuild_merged_output(batch_dir, output_path, ledger)
            return

    # ── Collect mode: skip to results ──
    if args.collect:
        if not batch_name_path.exists():
            print("No batch_name.txt found. Run without --collect first.")
            return
        batch_name = batch_name_path.read_text().strip()
        print(f"Collecting results from batch: {batch_name}")

        batch = poll_batch(client, batch_name, poll_interval=args.poll_interval)
        if batch is None:
            print("Batch did not succeed.")
            return

        meta = json.loads(meta_path.read_text())
        _process_results(client, batch_name, meta, ledger, output_path, batch_dir)
        return

    # ── Phase 1: Prepare all sessions ──
    print(f"\n=== Phase 1: Prepare {len(sessions)} sessions ===\n")
    video_cache = CACHE_DIR / args.participant
    session_infos = []

    for session in sessions:
        info = prepare_session(args.participant, session, ledger, client, video_cache,
                               max_seconds=max_seconds,
                               inventory_scope=args.inventory_scope,
                               prompt_variant=args.prompt_variant)
        if info:
            session_infos.append(info)

    if not session_infos:
        print("No sessions to process.")
        return

    print(f"\n  {len(session_infos)} sessions prepared "
          f"(skipped {len(sessions) - len(session_infos)})")

    # ── Build JSONL ──
    print(f"\n=== Phase 2: Build JSONL & submit batch ===\n")
    n_lines = 0
    with open(jsonl_path, "w") as f:
        for info in session_infos:
            for line in build_jsonl_requests(
                info,
                thinking_budget=args.thinking_budget,
                thinking_level=args.thinking_level,
                media_resolution=args.media_resolution,
            ):
                f.write(json.dumps(line) + "\n")
                n_lines += 1
    print(f"  JSONL written: {jsonl_path} "
          f"({n_lines} chunk requests across {len(session_infos)} sessions)")

    # Save metadata for result collection
    meta = {
        "model": args.model,
        "participant": args.participant,
        "max_minutes": args.max_minutes,
        "thinking_budget": args.thinking_budget,
        "thinking_level": args.thinking_level,
        "media_resolution": args.media_resolution,
        "sessions": {
            info["session"]: {
                "inventory": info["inventory"],
                "visible_inventory": info["visible_inventory"],
                "prompt": info["prompt"],
                "n_videos": info["n_videos"],
                "total_duration": info["total_duration"],
                "chunks": info["chunks"],
            }
            for info in session_infos
        },
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")

    # ── Submit ──
    batch_name = submit_batch(client, jsonl_path, args.model)
    batch_name_path.write_text(batch_name)

    # ── Phase 3: Poll and collect ──
    print(f"\n=== Phase 3: Poll for completion ===\n")
    batch = poll_batch(client, batch_name, poll_interval=args.poll_interval)
    if batch is None:
        print("Batch did not succeed. Re-run with --collect to retry later.")
        return

    _process_results(client, batch_name, meta, ledger, output_path, batch_dir)


def _process_results(client, batch_name, meta, ledger, output_path, batch_dir):
    """Collect batch results, regroup chunks by session, resolve, evaluate."""
    print(f"\n=== Collecting results ===\n")
    results = collect_results(client, batch_name)
    print(f"  {len(results)} chunk responses received")

    all_predictions = []
    for session, session_meta in sorted(meta["sessions"].items()):
        chunk_meta = session_meta.get("chunks", [])
        n_chunks = len(chunk_meta) if chunk_meta else 1
        chunk_logs: list[dict] = []
        # Merge across chunks: last chunk to report an instance_id wins.
        merged_raw: dict[str, dict] = {}
        any_response = False

        for i in range(n_chunks):
            key = f"{session}__chunk{i}"
            entry = results.get(key)
            if not entry:
                print(f"  {key}: NO RESPONSE")
                chunk_logs.append({"chunk": i, "error": "no_response"})
                continue
            if "error" in entry:
                print(f"  {key}: ERROR — {entry['error']}")
                chunk_logs.append({"chunk": i, "error": entry["error"]})
                continue

            any_response = True
            parsed = parse_batch_response(entry)
            cm = chunk_meta[i] if chunk_meta else {}
            t_start = cm.get("t_start") or 0.0
            chunk_logs.append({
                "chunk": i,
                "videos": cm.get("videos", []),
                "duration": cm.get("duration"),
                "t_start": cm.get("t_start"),
                "t_end": cm.get("t_end"),
                # Raw chunk-local predictions kept here for forensics.
                "predictions": parsed["predictions"],
                "inventory_journey": parsed.get("inventory_journey"),
                "thinking": parsed["thinking"],
                "raw_response": parsed["raw_response"],
                "stats": parsed["stats"],
            })

            # Translate chunk-local `evidence_timestamps` to session-absolute
            # (concat-mp4 time) so the annotator's seek-to-time and any
            # downstream consumer can use them without per-chunk math.
            for pred in parsed["predictions"]:
                pred_abs = dict(pred)
                ts_local = pred.get("evidence_timestamps") or []
                pred_abs["evidence_timestamps"] = [
                    int(round(t + t_start)) for t in ts_local if isinstance(t, (int, float))
                ]
                iid = (pred_abs.get("instance_id") or "").strip()
                if iid:
                    merged_raw[iid] = pred_abs
                else:
                    name = (pred_abs.get("item") or "").strip().lower()
                    merged_raw[f"__name__{name}"] = pred_abs

        if not any_response:
            continue

        visible_inventory = session_meta["visible_inventory"]
        preds = resolve_predictions(session, list(merged_raw.values()), visible_inventory)
        all_predictions.extend(preds)

        # Per-session log mirrors the rerun-log shape: chunks: [...].
        log = {
            "session": session,
            "model": meta["model"],
            "max_minutes": meta.get("max_minutes"),
            "inventory": session_meta["inventory"],
            "prompt": session_meta["prompt"],
            "chunks": chunk_logs,
            "merged_predictions": preds,
        }
        log_path = batch_dir / f"{session}_log.json"
        log_path.write_text(json.dumps(log, indent=2, ensure_ascii=False) + "\n")

        toks = sum((c.get("stats") or {}).get("total_tokens") or 0 for c in chunk_logs)
        print(f"  {session}: {len(preds)} predictions across {len(chunk_logs)} chunk(s) ({toks} tokens)")

    # Fold in predictions from any prior runs that wrote `_log.json` files
    # in this batch_dir but were skipped from the current submission, so the
    # merged output reflects every completed session under this tag.
    sessions_this_run = set(meta["sessions"].keys())
    for log_path in sorted(batch_dir.glob("*_log.json")):
        sess = log_path.stem.removesuffix("_log")
        if sess in sessions_this_run:
            continue
        try:
            prior = json.loads(log_path.read_text())
        except Exception as e:
            print(f"  WARN could not read prior log {log_path.name}: {e}")
            continue
        prior_preds = prior.get("merged_predictions") or []
        if prior_preds:
            all_predictions.extend(prior_preds)
            print(f"  + carried forward {len(prior_preds)} preds from prior session {sess}")

    # Save predictions
    output_path.write_text(json.dumps(all_predictions, indent=2) + "\n")
    print(f"\n{len(all_predictions)} predictions saved to {output_path}")

    # Save raw batch results
    raw_path = batch_dir / "batch_results_raw.json"
    raw_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")

    # Run evaluation
    from importlib import import_module
    eval_mod = import_module("evaluate_amount")
    gt = eval_mod.extract_ground_truth(ledger)
    report = eval_mod.evaluate(gt, all_predictions)

    eval_path = output_path.with_name(output_path.stem + "_eval.json")
    eval_mod.write_report(report, eval_path)
    eval_mod.print_eval_table(report)


def _rebuild_merged_output(batch_dir: Path, output_path: Path, ledger: dict) -> None:
    """Rebuild the merged preds + eval files from already-written per-session logs.

    Used when every requested session already has a `_log.json` (no submission
    needed). Lets the user re-emit `output_path` after manually restoring or
    deleting individual logs.
    """
    all_predictions: list[dict] = []
    for log_path in sorted(batch_dir.glob("*_log.json")):
        try:
            prior = json.loads(log_path.read_text())
        except Exception as e:
            print(f"  WARN could not read {log_path.name}: {e}")
            continue
        all_predictions.extend(prior.get("merged_predictions") or [])

    output_path.write_text(json.dumps(all_predictions, indent=2) + "\n")
    print(f"\n{len(all_predictions)} predictions saved to {output_path}")

    from importlib import import_module
    eval_mod = import_module("evaluate_amount")
    gt = eval_mod.extract_ground_truth(ledger)
    report = eval_mod.evaluate(gt, all_predictions)

    eval_path = output_path.with_name(output_path.stem + "_eval.json")
    eval_mod.write_report(report, eval_path)
    eval_mod.print_eval_table(report)


if __name__ == "__main__":
    main()

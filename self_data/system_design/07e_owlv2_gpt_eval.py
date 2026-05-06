#!/usr/bin/env python3
"""
07e_owlv2_gpt_eval.py — Scene-tag agreement benchmark: OWLv2 (07d) vs GPT-5.4.

For each session, evaluates ALL HOI trigger frames (frames where SigLIP or
DINO matched a food item above `--min-food-score`, same selection 07d uses).
This puts the eval on the same set of frames that 07d, 06_avp_round1, and
the planner all see, so disagreement here directly explains downstream
divergence.

Each trigger frame is labeled two ways:
  1. GPT-5.4 — frames chunked into batches of `--gpt-batch-size` (50 by
     default) and sent across multiple API calls, labels concatenated
     in order. Outputs one of {sink, stove, storage, other}.
  2. OWLv2 multi-query detection (same queries as 07d, threshold from CLI)
     → {sink, stove, storage, unknown} (unknown remapped to `other` for matching)

Writes per-session JSON with the per-frame labels + agreement stats, plus an
aggregate file across all processed sessions.

Usage:
    python system_design/07e_owlv2_gpt_eval.py --participant kailai --session 20260310-195710
    python system_design/07e_owlv2_gpt_eval.py --participant kailai --all --resume
    python system_design/07e_owlv2_gpt_eval.py --all-participants --resume

Prerequisites:
    - SigLIP matches: hands23_detection/*_siglip_matches.json  (from 02)
    - DINO   matches: hands23_detection/*_dino_matches.json    (from 03)
    - Azure OpenAI credentials: AZURE_OPENAI_API_KEY(_2) + AZURE_OPENAI_ENDPOINT(_2)
"""

import argparse
import base64
import io
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from dotenv import load_dotenv
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from utils import (  # noqa: E402
    get_sessions,
    hands23_dir,
    outputs_dir,
    participant_dir,
)

# Load credentials from kitchen/.env (same pattern as 06_avp_round1_remaining.py)
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

# ---------------------------------------------------------------------------
# Scene query set — kept in sync with 07d.
# ---------------------------------------------------------------------------

QUERIES_MULTI: Dict[str, List[str]] = {
    "sink":    ["a sink faucet", "a stainless steel sink basin"],
    "stove":   ["a stove", "a pot on a stove"],
    "storage": ["a refrigerator", "a cabinet"],
}
OWL_MODEL_ID = "google/owlv2-base-patch16-ensemble"
OWL_THRESHOLD_DEFAULT = 0.40

GPT_MODEL_DEFAULT = "gpt-5.4"
GPT_BATCH_SIZE_DEFAULT = 50    # max frames per GPT request
MIN_FOOD_SCORE_DEFAULT = 0.15  # SigLIP/DINO similarity floor — same as 07d
JPEG_QUALITY = 85
# Longest-side resize before sending to GPT. ~800px keeps detail while cutting
# payload ~8x vs full fisheye native res.
GPT_IMAGE_MAX_SIDE = 800

SCENE_SET = ["sink", "stove", "storage", "other"]


# ---------------------------------------------------------------------------
# Frame selection — HOI trigger frames (same set as 07d).
#
# A frame qualifies if SigLIP or DINO matched a food item with similarity
# >= min_food_score on at least one HOI crop in that frame. Path layout:
#   outputs/{session}/hands23_detection/{clip_stem}/frames/frame_NNNNNNNN_tXX.XXs.jpg
# The siglip/dino match files reference these by `frame_path` relative to
# hands23_dir, with `timestamp` already in session-cumulative seconds.
# ---------------------------------------------------------------------------

def _collect_trigger_frames(
    participant: str, session: str, min_score: float,
) -> List[Tuple[float, Path]]:
    """Return [(session_time_s, frame_jpg_path)] sorted by time, for every
    HOI frame whose SigLIP or DINO top match reached `min_score`.

    Mirrors 07d's `collect_trigger_frames` so both scripts evaluate the
    same set of frames per session. SigLIP and DINO contribute to the same
    union (deduped by frame_path)."""
    h23 = hands23_dir(participant, session)
    trigger: Dict[str, float] = {}
    for pattern in ("*_siglip_matches.json", "*_dino_matches.json"):
        files = list(h23.glob(pattern))
        if not files:
            continue
        data = json.loads(files[0].read_text())
        for video in data.get("videos", []):
            for m in video.get("matches", []):
                if any(
                    t.get("similarity", 0) >= min_score
                    for t in m.get("top_matches", [])
                ):
                    trigger.setdefault(m["frame_path"], m["timestamp"])
    out: List[Tuple[float, Path]] = []
    for rel, ts in trigger.items():
        out.append((float(ts), h23 / rel))
    out.sort(key=lambda x: x[0])
    return out


def image_to_jpeg_b64(img: Image.Image, max_side: int = GPT_IMAGE_MAX_SIDE) -> str:
    """JPEG-encode (with optional downsize) and return base64 string."""
    w, h = img.size
    m = max(w, h)
    if m > max_side:
        scale = max_side / m
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=JPEG_QUALITY)
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ---------------------------------------------------------------------------
# OWLv2 (inline — same primitives as 07d, duplicated to avoid importing a
# numeric-prefixed module name).
# ---------------------------------------------------------------------------

def _build_owlv2(device: torch.device):
    from transformers import Owlv2ForObjectDetection, Owlv2Processor
    processor = Owlv2Processor.from_pretrained(OWL_MODEL_ID)
    model = Owlv2ForObjectDetection.from_pretrained(OWL_MODEL_ID).eval().to(device)
    return model, processor


def _owl_detect_batch(
    model, processor, images: List[Image.Image],
    query_texts: List[str], device: torch.device, threshold: float,
) -> List[List[dict]]:
    nested = [query_texts] * len(images)
    inputs = processor(text=nested, images=images, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    target_sizes = torch.tensor(
        [[img.size[1], img.size[0]] for img in images], device=device
    )
    results = processor.post_process_grounded_object_detection(
        outputs=outputs, target_sizes=target_sizes, threshold=threshold,
        text_labels=nested,
    )
    per_image: List[List[dict]] = []
    for r in results:
        scores = r["scores"].cpu().tolist()
        labels = r.get("text_labels") or [query_texts[i] for i in r["labels"].cpu().tolist()]
        boxes = r["boxes"].cpu().tolist()
        dets = [
            {"label": lbl, "score": round(float(s), 4),
             "box": [round(float(v), 1) for v in bx]}
            for s, lbl, bx in zip(scores, labels, boxes)
        ]
        per_image.append(dets)
    return per_image


def _derive_scene(
    detections: List[dict], query_to_scene: Dict[str, str],
) -> Tuple[str, Optional[dict], Dict[str, float]]:
    per_scene_max: Dict[str, float] = {}
    per_scene_top: Dict[str, dict] = {}
    for d in detections:
        scene = query_to_scene.get(d["label"])
        if scene is None:
            continue
        if d["score"] > per_scene_max.get(scene, -1.0):
            per_scene_max[scene] = d["score"]
            per_scene_top[scene] = d
    if not per_scene_max:
        return "unknown", None, {}
    top = max(per_scene_max, key=per_scene_max.get)
    return top, per_scene_top[top], per_scene_max


def owlv2_label_frames(
    images: List[Image.Image], model, processor,
    device: torch.device, threshold: float, batch_size: int = 4,
) -> List[dict]:
    query_texts: List[str] = []
    query_to_scene: Dict[str, str] = {}
    for scene, qs in QUERIES_MULTI.items():
        for q in qs:
            query_texts.append(q)
            query_to_scene[q] = scene

    results: List[dict] = []
    for start in range(0, len(images), batch_size):
        batch = images[start:start + batch_size]
        dets_per = _owl_detect_batch(
            model, processor, batch, query_texts, device, threshold
        )
        for dets in dets_per:
            scene, top, per_scene_max = _derive_scene(dets, query_to_scene)
            results.append({
                "owlv2_scene": scene,                              # sink/stove/storage/unknown
                "owlv2_top_label": top["label"] if top else None,  # raw query
                "owlv2_top_score": top["score"] if top else 0.0,
                "owlv2_per_scene_max": {s: round(v, 4) for s, v in per_scene_max.items()},
            })
    return results


# ---------------------------------------------------------------------------
# GPT-5.4 classifier — all N frames in one API call.
# ---------------------------------------------------------------------------

GPT_PROMPT = """You are labeling egocentric kitchen video frames from a \
person wearing smart glasses. Each frame will be provided as an image, \
numbered 1..{n}.

For each frame assign exactly one scene label from this set:
- "sink": sink basin or sink faucet is a dominant feature of the frame
- "stove": a stove burner, pan, or pot on a stove is a dominant feature
- "storage": interior of a refrigerator or pantry/cabinet is visible
- "other": none of the above (hallway, person, hands mid-air with no kitchen \
feature, countertop-only, blurry transition, etc.)

Return STRICT JSON only, with this exact schema:
{{"frames": [{{"i": 1, "label": "sink"}}, {{"i": 2, "label": "other"}}, ...]}}

Exactly {n} entries, one per frame, in order. No prose, no markdown, no \
explanations — ONLY the JSON object.
"""


def make_azure_client():
    from openai import AzureOpenAI
    api_key = os.getenv("AZURE_OPENAI_API_KEY_2") or os.getenv("AZURE_OPENAI_API_KEY")
    endpoint = (
        os.getenv("AZURE_OPENAI_ENDPOINT_2")
        or os.getenv("AZURE_OPENAI_ENDPOINT")
        or ""
    ).strip()
    if not api_key or not endpoint:
        raise ValueError(
            "Missing Azure OpenAI API credentials "
            "(AZURE_OPENAI_API_KEY[_2] / AZURE_OPENAI_ENDPOINT[_2])"
        )
    return AzureOpenAI(
        azure_endpoint=endpoint, api_key=api_key, api_version="2025-03-01-preview",
    )


def _parse_gpt_json(text: str, expected_n: int) -> Optional[List[dict]]:
    """Tolerant JSON extract. Falls back to the first {...} block if the
    response is wrapped in prose or code fences."""
    if not text:
        return None
    for candidate in (text, *_extract_json_blocks(text)):
        try:
            obj = json.loads(candidate)
            frames = obj.get("frames") if isinstance(obj, dict) else None
            if isinstance(frames, list) and len(frames) == expected_n:
                return frames
        except json.JSONDecodeError:
            continue
    return None


def _extract_json_blocks(text: str) -> List[str]:
    out = []
    # ```json ... ``` blocks
    for m in re.finditer(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL):
        out.append(m.group(1))
    # first balanced { ... } (greedy)
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        out.append(m.group(0))
    return out


def gpt_label_frames(
    client, images: List[Image.Image],
    model: str = GPT_MODEL_DEFAULT,
    reasoning_effort: str = "low",
) -> Tuple[List[str], dict, str]:
    """Send all images in one request. Returns (labels, stats, raw_response_text).

    raw_response_text is always returned for audit, even on successful parse.
    """
    frames_b64 = [image_to_jpeg_b64(img) for img in images]
    n = len(images)

    content: list[dict] = [{"type": "input_text", "text": GPT_PROMPT.format(n=n)}]
    for i, fb64 in enumerate(frames_b64):
        content.append({"type": "input_text", "text": f"[Frame {i+1}]"})
        content.append({
            "type": "input_image",
            "image_url": f"data:image/jpeg;base64,{fb64}",
            "detail": "low",  # scene classification doesn't need high-res tokens
        })

    max_retries = 3
    for attempt in range(max_retries):
        t0 = time.time()
        try:
            response = client.responses.create(
                model=model,
                input=[{"role": "user", "content": content}],
                reasoning={"effort": reasoning_effort},
            )
            response_text = response.output_text or ""
            usage = response.usage
            elapsed = round(time.time() - t0, 2)
            parsed = _parse_gpt_json(response_text, expected_n=n)
            if parsed is None:
                if attempt < max_retries - 1:
                    print(f"  GPT parse fail (attempt {attempt+1}), retrying...", flush=True)
                    time.sleep(3)
                    continue
                return [], {
                    "error": "parse_failed",
                    "inference_time_s": elapsed,
                    "input_tokens": usage.input_tokens,
                    "output_tokens": usage.output_tokens,
                }, response_text
            labels = [
                (f.get("label") or "other").lower() if isinstance(f, dict) else "other"
                for f in parsed
            ]
            # Normalize unknown labels to `other` for safety
            labels = [l if l in SCENE_SET else "other" for l in labels]
            stats = {
                "model": model,
                "inference_time_s": elapsed,
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "attempt": attempt + 1,
            }
            return labels, stats, response_text
        except Exception as e:
            err = str(e)
            transient = any(m in err for m in (
                "503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED",
                "500", "INTERNAL", "timeout", "connection", "Connection",
            ))
            if transient and attempt < max_retries - 1:
                wait = 4 * (2 ** attempt)
                print(f"  GPT transient error ({err[:60]}), retry in {wait}s", flush=True)
                time.sleep(wait)
                continue
            return [], {"error": err[:500], "inference_time_s": round(time.time() - t0, 2)}, ""

    return [], {"error": "max retries exceeded"}, ""


# ---------------------------------------------------------------------------
# Session evaluation
# ---------------------------------------------------------------------------

def _owl_to_matching(scene: str) -> str:
    """Remap OWLv2's `unknown` → `other` so labels align with GPT's vocabulary."""
    return "other" if scene == "unknown" else scene


def eval_session(
    participant: str,
    session: str,
    owl_model,
    owl_processor,
    gpt_client,
    device: torch.device,
    min_food_score: float,
    gpt_batch_size: int,
    owl_threshold: float,
    owl_batch_size: int,
    gpt_model: str,
    reasoning_effort: str,
) -> Optional[dict]:
    out_dir = outputs_dir(participant, session)
    out_dir.mkdir(parents=True, exist_ok=True)

    # HOI trigger frames — same set 07d evaluates.
    samples = _collect_trigger_frames(participant, session, min_food_score)
    if not samples:
        print(f"  {session}: no HOI trigger frames at min_food_score={min_food_score}, skipping")
        return None
    n_sampled = len(samples)
    source = "hoi_trigger"
    print(f"  {session}: {n_sampled} HOI trigger frames (min_food_score={min_food_score})")

    # Decode once — both OWLv2 and GPT need the pixel data.
    images: List[Image.Image] = []
    valid_samples: List[Tuple[float, Path]] = []
    for t, fp in tqdm(samples, desc="  Loading"):
        try:
            img = Image.open(fp).convert("RGB")
        except (FileNotFoundError, OSError):
            continue
        images.append(img)
        valid_samples.append((t, fp))
    if not images:
        print(f"  {session}: no frames loaded")
        return None
    n_sampled = len(images)

    # ---------- OWLv2 ----------
    t0 = time.time()
    owl_results = owlv2_label_frames(
        images, owl_model, owl_processor, device, owl_threshold, owl_batch_size,
    )
    owl_elapsed = time.time() - t0
    print(f"  OWLv2: {owl_elapsed:.1f}s ({owl_elapsed / n_sampled * 1000:.0f} ms/frame)")

    # ---------- GPT (chunked into batches of gpt_batch_size) ----------
    gpt_labels: List[str] = []
    gpt_batches: List[dict] = []
    total_input_toks = 0
    total_output_toks = 0
    total_gpt_time = 0.0
    n_batches = (n_sampled + gpt_batch_size - 1) // gpt_batch_size
    gpt_failed = False
    for b in range(n_batches):
        start = b * gpt_batch_size
        end = min(n_sampled, start + gpt_batch_size)
        batch_imgs = images[start:end]
        print(f"  GPT batch {b+1}/{n_batches}: frames {start+1}..{end} "
              f"({len(batch_imgs)} imgs)", flush=True)
        labels, stats, raw = gpt_label_frames(
            gpt_client, batch_imgs, model=gpt_model,
            reasoning_effort=reasoning_effort,
        )
        gpt_batches.append({
            "batch_index": b,
            "frame_start": start + 1, "frame_end": end,
            "n_frames": len(batch_imgs),
            "labels": labels,
            "stats": stats,
            "response": raw,
        })
        if not labels or len(labels) != len(batch_imgs):
            print(f"    failed: got {len(labels)} labels, expected {len(batch_imgs)} — stats={stats}")
            # Pad with `other` so downstream alignment still works, but mark
            # the session's GPT result as failed for reporting.
            gpt_failed = True
            labels = ["other"] * len(batch_imgs)
        gpt_labels.extend(labels)
        total_input_toks += stats.get("input_tokens", 0) or 0
        total_output_toks += stats.get("output_tokens", 0) or 0
        total_gpt_time += stats.get("inference_time_s", 0.0) or 0.0

    # Persist per-batch responses for audit.
    responses_path = out_dir / "owlv2_gpt_eval_gpt_responses.json"
    responses_path.write_text(json.dumps(gpt_batches, indent=2))

    print(f"  GPT total: {total_gpt_time:.1f}s, "
          f"{total_input_toks}→{total_output_toks} toks across {n_batches} batch(es)")

    gpt_stats = {
        "model": gpt_model,
        "reasoning_effort": reasoning_effort,
        "n_batches": n_batches,
        "batch_size": gpt_batch_size,
        "total_input_tokens": total_input_toks,
        "total_output_tokens": total_output_toks,
        "total_inference_time_s": round(total_gpt_time, 2),
        "any_batch_failed": gpt_failed,
    }

    # ---------- Per-frame comparison ----------
    frames: List[dict] = []
    confusion: Dict[str, Dict[str, int]] = {a: {b: 0 for b in SCENE_SET} for a in SCENE_SET}
    n_match = 0
    per_label_match: Dict[str, Dict[str, int]] = {l: {"n": 0, "match": 0} for l in SCENE_SET}
    for i, ((t, fp), owl, gpt_lbl) in enumerate(zip(valid_samples, owl_results, gpt_labels)):
        owl_matching = _owl_to_matching(owl["owlv2_scene"])
        is_match = owl_matching == gpt_lbl
        if is_match:
            n_match += 1
        confusion[gpt_lbl][owl_matching] += 1
        per_label_match[gpt_lbl]["n"] += 1
        if is_match:
            per_label_match[gpt_lbl]["match"] += 1
        # Record frame_file as a path relative to the participant's outputs
        # dir so audit tools can locate it regardless of whether it came from
        # the hands23 cache or the ffmpeg fallback.
        try:
            rel_frame = fp.relative_to(outputs_dir(participant, session))
        except ValueError:
            rel_frame = fp.name
        frames.append({
            "index": i + 1,
            "session_timestamp_s": round(t, 2),
            "frame_file": str(rel_frame),
            "gpt_label": gpt_lbl,
            **owl,
            "match": is_match,
        })

    agreement = {
        "n": n_sampled,
        "match": n_match,
        "accuracy": round(n_match / n_sampled, 4),
        "per_label": {
            l: {
                "n": per_label_match[l]["n"],
                "match": per_label_match[l]["match"],
                "recall": round(per_label_match[l]["match"] / per_label_match[l]["n"], 4)
                          if per_label_match[l]["n"] else None,
            }
            for l in SCENE_SET
        },
        "confusion_gpt_x_owlv2": confusion,
    }

    result = {
        "participant": participant,
        "session": session,
        "timestamp": datetime.now().isoformat(),
        "min_food_score": min_food_score,
        "gpt_batch_size": gpt_batch_size,
        "n_frames_sampled": n_sampled,
        "frame_source": source,
        "owlv2_threshold": owl_threshold,
        "owlv2_queries": QUERIES_MULTI,
        "gpt_model": gpt_model,
        "gpt_stats": gpt_stats,
        "gpt_responses_path": "owlv2_gpt_eval_gpt_responses.json",
        "agreement": agreement,
        "frames": frames,
    }

    out_file = out_dir / "owlv2_gpt_scene_eval.json"
    out_file.write_text(json.dumps(result, indent=2))
    print(f"  Agreement: {n_match}/{n_sampled} = {agreement['accuracy']:.1%}")
    print(f"  Output: {out_file}")
    return result


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------

def write_aggregate(
    runs: List[dict], output_path: Path,
) -> None:
    total_n = 0
    total_match = 0
    per_label: Dict[str, Dict[str, int]] = {l: {"n": 0, "match": 0} for l in SCENE_SET}
    confusion: Dict[str, Dict[str, int]] = {a: {b: 0 for b in SCENE_SET} for a in SCENE_SET}
    sessions_summary: List[dict] = []
    for r in runs:
        if "agreement" not in r:
            continue
        a = r["agreement"]
        total_n += a["n"]; total_match += a["match"]
        for l in SCENE_SET:
            per_label[l]["n"] += a["per_label"][l]["n"]
            per_label[l]["match"] += a["per_label"][l]["match"]
        for g, row in a["confusion_gpt_x_owlv2"].items():
            for o, v in row.items():
                confusion[g][o] += v
        sessions_summary.append({
            "participant": r["participant"], "session": r["session"],
            "n": a["n"], "match": a["match"], "accuracy": a["accuracy"],
        })

    aggregate = {
        "timestamp": datetime.now().isoformat(),
        "n_sessions": len(sessions_summary),
        "total_frames": total_n,
        "total_match": total_match,
        "accuracy": round(total_match / total_n, 4) if total_n else None,
        "per_label": {
            l: {
                "n": per_label[l]["n"],
                "match": per_label[l]["match"],
                "recall": round(per_label[l]["match"] / per_label[l]["n"], 4)
                          if per_label[l]["n"] else None,
            }
            for l in SCENE_SET
        },
        "confusion_gpt_x_owlv2": confusion,
        "sessions": sessions_summary,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(aggregate, indent=2))
    print(f"\nAggregate written: {output_path}")
    print(f"  sessions={len(sessions_summary)}  frames={total_n}  "
          f"agreement={aggregate['accuracy']}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="OWLv2 vs GPT-5.4 scene-tag agreement benchmark"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--participant", help="Single participant")
    group.add_argument("--all-participants", action="store_true")
    parser.add_argument("--session", help="Single session (requires --participant)")
    parser.add_argument("--all", action="store_true", help="All sessions for the participant")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--min-food-score", type=float, default=MIN_FOOD_SCORE_DEFAULT,
                        help=f"SigLIP/DINO similarity floor for trigger frames (default {MIN_FOOD_SCORE_DEFAULT}, same as 07d)")
    parser.add_argument("--gpt-batch-size", type=int, default=GPT_BATCH_SIZE_DEFAULT,
                        help=f"Max frames per GPT request (default {GPT_BATCH_SIZE_DEFAULT}); sessions are chunked into multiple calls as needed")
    parser.add_argument("--owl-threshold", type=float, default=OWL_THRESHOLD_DEFAULT)
    parser.add_argument("--owl-batch-size", type=int, default=4)
    parser.add_argument("--gpt-model", default=GPT_MODEL_DEFAULT)
    parser.add_argument("--reasoning", default="low",
                        choices=["minimal", "low", "medium", "high"],
                        help="GPT reasoning effort (default low — scene tagging is simple)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip sessions with existing owlv2_gpt_scene_eval.json")
    parser.add_argument("--aggregate-out", default=None,
                        help="Path to write aggregate JSON (default: outputs/owlv2_gpt_eval_aggregate.json)")
    args = parser.parse_args()

    if args.all_participants and args.session:
        parser.error("--session only valid with --participant")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Resolve target (participant, session) pairs
    pairs: List[Tuple[str, str]] = []
    if args.all_participants:
        participants_root = Path(__file__).resolve().parents[1] / "participants"
        participants = sorted(
            d.name for d in participants_root.iterdir()
            if d.is_dir() and (d / "videos").is_dir()
        )
        for p in participants:
            for s in get_sessions(p):
                pairs.append((p, s))
    else:
        if args.session:
            pairs = [(args.participant, args.session)]
        elif args.all:
            pairs = [(args.participant, s) for s in get_sessions(args.participant)]
        else:
            parser.error("Provide --session or --all when using --participant")

    print(f"\nTargets: {len(pairs)} (participant, session) pair(s)")
    print(f"Device: {device}")
    print(f"Trigger source: HOI frames @ min_food_score={args.min_food_score} (batch size {args.gpt_batch_size})")
    print(f"OWLv2 threshold: {args.owl_threshold}")
    print(f"GPT model: {args.gpt_model}, reasoning effort: {args.reasoning}")

    print(f"\nLoading OWLv2 ({OWL_MODEL_ID})...")
    owl_model, owl_processor = _build_owlv2(device)
    print("OWLv2 loaded")

    print("Building Azure OpenAI client...")
    gpt_client = make_azure_client()
    print("Client ready")

    runs: List[dict] = []
    for i, (p, s) in enumerate(pairs):
        print(f"\n{'#' * 70}")
        print(f"# [{i+1}/{len(pairs)}] {p}/{s}")
        print(f"{'#' * 70}")
        if args.resume:
            out = outputs_dir(p, s) / "owlv2_gpt_scene_eval.json"
            if out.exists():
                print("  SKIPPED (results exist)")
                try:
                    runs.append(json.loads(out.read_text()))
                except json.JSONDecodeError:
                    pass
                continue
        result = eval_session(
            participant=p, session=s,
            owl_model=owl_model, owl_processor=owl_processor,
            gpt_client=gpt_client, device=device,
            min_food_score=args.min_food_score,
            gpt_batch_size=args.gpt_batch_size,
            owl_threshold=args.owl_threshold,
            owl_batch_size=args.owl_batch_size,
            gpt_model=args.gpt_model,
            reasoning_effort=args.reasoning,
        )
        if result is not None:
            runs.append(result)

    # Aggregate
    if runs:
        agg_path = Path(args.aggregate_out) if args.aggregate_out else (
            Path(__file__).resolve().parents[1] / "outputs" / "owlv2_gpt_eval_aggregate.json"
        )
        write_aggregate(runs, agg_path)


if __name__ == "__main__":
    main()

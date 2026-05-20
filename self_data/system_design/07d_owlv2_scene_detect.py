#!/usr/bin/env python3
"""
07d_owlv2_scene_detect.py — Zero-shot object detection of fridge / sink / stove
on HOI trigger frames using OWLv2.

Alternative to the whole-frame classification approach (07b SigLIP2, 07c
EgoVLPv2). OWLv2 returns bounding boxes + per-query confidences, which is a
stronger signal than cosine similarity on a full frame: it localizes the
object, handles partial views, and is less fooled by dominant textures.

Scene tag derivation:
    For each frame we collect the highest-scoring detection per query,
    then pick the label with the top confidence above threshold. If no
    query clears threshold, tag as `unknown`.

Output: scene_tags_owlv2.json (parallel to 07b's scene_tags.json and 07c's
scene_tags_egovlpv2.json). Schema shaped so the annotator review bar can read
any of the three interchangeably.

Usage:
    python system_design/07d_owlv2_scene_detect.py --participant kailai --session 20260310-195710
    python system_design/07d_owlv2_scene_detect.py --participant kailai --all --device cuda:0
    python system_design/07d_owlv2_scene_detect.py --participant kailai --all --resume

Prerequisites:
    - SigLIP matches: hands23_detection/*_siglip_matches.json  (from 02)
    - DINO matches:   hands23_detection/*_dino_matches.json    (from 03)
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from utils import get_sessions, hands23_dir, outputs_dir  # noqa: E402

MODEL_ID = "google/owlv2-base-patch16-ensemble"

# Scene tag → list of OWLv2 query texts. Each scene is anchored on multiple
# object nouns; a frame's scene is the scene whose best query scored highest
# (so e.g. a frame where `a pantry` beats `a refrigerator` still tags as
# `storage`). All queries run in one detection batch; the grouping happens
# during derivation. No `prep` here because "prep" is an activity, not an
# object — the earlier single-query "a stove" version misfired when any
# vaguely-rectangular counter object got a stove detection. Multi-anchor
# `stove` (stove + pan + pot) is more conservative.
QUERIES_MULTI: Dict[str, List[str]] = {
    "sink":    ["a sink faucet", "a stainless steel sink basin"],
    "stove":   ["a stove", "a pot on a stove"],
    # "an open refrigerator" is complementary to "a refrigerator": the closed
    # fridge face and the door-open view of fridge contents activate the
    # model via different anchors. A/B probe on 4 kailai sessions (689
    # trigger frames) recovered 40 frames from `unknown` to `storage` with
    # no false transitions. Shelf / cabinet phrasings ("a shelf", "a
    # kitchen shelf", "an open cabinet", etc.) all fired 0 or ~1 times
    # across the same 689 frames and are not worth adding.
    "storage": ["a refrigerator", "an open refrigerator", "a cabinet"],
}

MIN_FOOD_SCORE = 0.15

# OWLv2 confidence floor. Anything below this is filtered *before* argmax,
# so a frame with only low-confidence detections tags as `unknown` rather
# than forcing a weak label. Each frame's `detections` list records ALL
# detections scoring above this threshold (not only the top one), so the
# scene argmax and raw detection evidence are both preserved. Tunable
# from CLI.
DETECTION_THRESHOLD = 0.40


# ---------------------------------------------------------------------------
# Trigger frames (mirrors 07b/07c)
# ---------------------------------------------------------------------------

def collect_trigger_frames(
    participant: str,
    session: str,
    min_score: float = MIN_FOOD_SCORE,
) -> Dict[str, float]:
    det_dir = hands23_dir(participant, session)
    trigger: Dict[str, float] = {}

    for pattern in ("*_siglip_matches.json", "*_dino_matches.json"):
        files = list(det_dir.glob(pattern))
        if not files:
            continue
        data = json.loads(files[0].read_text())
        for video in data.get("videos", []):
            for m in video.get("matches", []):
                if any(t.get("similarity", 0) >= min_score
                       for t in m.get("top_matches", [])):
                    trigger.setdefault(m["frame_path"], m["timestamp"])
    return trigger


# ---------------------------------------------------------------------------
# OWLv2 detection
# ---------------------------------------------------------------------------

def _run_detection_batch(
    model, processor, images: List[Image.Image],
    query_texts: List[str], device: torch.device,
    threshold: float,
) -> List[List[dict]]:
    """Run OWLv2 on a batch. Returns a list of per-image detection lists.
    Each detection: {"label": <query>, "score": float, "box": [x1,y1,x2,y2]}.
    Boxes are in pixel coords of the ORIGINAL image (via target_sizes).
    """
    nested_texts = [query_texts] * len(images)
    inputs = processor(text=nested_texts, images=images, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    # (H, W) per image — OWLv2 post-processing expects this order.
    target_sizes = torch.tensor(
        [[img.size[1], img.size[0]] for img in images], device=device
    )
    results = processor.post_process_grounded_object_detection(
        outputs=outputs, target_sizes=target_sizes, threshold=threshold,
        text_labels=nested_texts,
    )

    per_image: List[List[dict]] = []
    for r in results:
        dets = []
        scores = r["scores"].cpu().tolist()
        labels = r.get("text_labels") or [query_texts[i] for i in r["labels"].cpu().tolist()]
        boxes = r["boxes"].cpu().tolist()
        for score, label, box in zip(scores, labels, boxes):
            dets.append({
                "label": label,
                "score": round(float(score), 4),
                "box": [round(float(v), 1) for v in box],
            })
        per_image.append(dets)
    return per_image


def _derive_scene(
    detections: List[dict], query_to_scene: Dict[str, str],
) -> Tuple[str, Optional[dict], Dict[str, float]]:
    """Multi-query → scene mapping.

    For each detection, route it to its scene via query_to_scene. Scene's
    score is the max over its queries' detection scores. Final scene tag =
    argmax of per-scene scores; `unknown` if nothing detected at all.

    Returns (scene_tag, top_detection_dict or None, per_scene_max_scores).
    """
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
    top_scene = max(per_scene_max, key=per_scene_max.get)
    return top_scene, per_scene_top[top_scene], per_scene_max


# ---------------------------------------------------------------------------
# Session processing
# ---------------------------------------------------------------------------

def run_session(
    participant: str,
    session: str,
    model,
    processor,
    device: torch.device,
    batch_size: int = 4,
    min_score: float = MIN_FOOD_SCORE,
    threshold: float = DETECTION_THRESHOLD,
    queries_multi: Optional[Dict[str, List[str]]] = None,
    output_filename: str = "scene_tags_owlv2.json",
) -> Optional[dict]:
    if queries_multi is None:
        queries_multi = QUERIES_MULTI
    det_dir = hands23_dir(participant, session)
    out_dir = outputs_dir(participant, session)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_file = out_dir / output_filename

    trigger_frames = collect_trigger_frames(participant, session, min_score)
    if not trigger_frames:
        print(f"  {session}: no trigger frames (no food matches >= {min_score})")
        return None

    print(f"  {session}: {len(trigger_frames)} trigger frames")

    sorted_frames = sorted(trigger_frames.items(), key=lambda x: x[1])
    frame_paths = [fp for fp, _ in sorted_frames]
    timestamps = [ts for _, ts in sorted_frames]

    # Flatten {scene: [q1, q2, ...]} into a single query list + reverse map.
    query_texts: List[str] = []
    query_to_scene: Dict[str, str] = {}
    for scene, qs in queries_multi.items():
        for q in qs:
            query_texts.append(q)
            query_to_scene[q] = scene
    scene_names = list(queries_multi.keys())

    frames_dict = {}
    scene_counts = {s: 0 for s in scene_names + ["unknown"]}

    # Process in batches so the tqdm bar reflects GPU batches, not single images.
    skipped = 0
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for start in tqdm(range(0, len(frame_paths), batch_size), desc="  OWLv2 batches"):
        batch_paths = frame_paths[start:start + batch_size]
        batch_ts = timestamps[start:start + batch_size]
        images = []
        valid_paths = []
        valid_ts = []
        for fp, ts in zip(batch_paths, batch_ts):
            try:
                images.append(Image.open(det_dir / fp).convert("RGB"))
                valid_paths.append(fp)
                valid_ts.append(ts)
            except (FileNotFoundError, OSError):
                skipped += 1
        if not images:
            continue

        detections = _run_detection_batch(
            model, processor, images, query_texts, device, threshold
        )

        for fp, ts, dets in zip(valid_paths, valid_ts, detections):
            scene, top, per_scene_max = _derive_scene(dets, query_to_scene)
            frames_dict[fp] = {
                "timestamp": round(ts, 2),
                "scene": scene,
                "top_score": top["score"] if top else 0.0,
                "top_label": top["label"] if top else None,
                "top_box": top["box"] if top else None,
                "per_scene_max": {s: round(v, 4) for s, v in per_scene_max.items()},
                "detections": dets,
            }
            scene_counts[scene] += 1

    elapsed = time.time() - t0
    if skipped:
        print(f"  Skipped {skipped} unreadable frames")
    print(f"  Detected in {elapsed:.1f}s ({elapsed / max(len(frames_dict), 1) * 1000:.1f} ms/frame)")

    output = {
        "participant": participant,
        "session": session,
        "model": MODEL_ID,
        "queries": queries_multi,
        "threshold": threshold,
        "min_food_score": min_score,
        "timestamp": datetime.now().isoformat(),
        "stats": {
            "total_trigger_frames": len(frames_dict),
            "scene_counts": scene_counts,
            "detect_time_s": round(elapsed, 2),
            **(
                {
                    "peak_gpu_mem_allocated_gb": round(
                        torch.cuda.max_memory_allocated() / 1024**3, 3),
                    "peak_gpu_mem_reserved_gb": round(
                        torch.cuda.max_memory_reserved() / 1024**3, 3),
                    "gpu_device_name": torch.cuda.get_device_name(0),
                }
                if torch.cuda.is_available() else {}
            ),
        },
        "frames": frames_dict,
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"  Output: {output_file}")
    print(f"  Scene distribution: {scene_counts}")
    return output


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Zero-shot detect fridge/sink/stove on HOI trigger frames with OWLv2"
    )
    parser.add_argument("--participant", required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--session", help="Single session")
    group.add_argument("--all", action="store_true", help="All sessions")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--min-score", type=float, default=MIN_FOOD_SCORE,
                        help=f"Min food match score to trigger detection (default: {MIN_FOOD_SCORE})")
    parser.add_argument("--threshold", type=float, default=DETECTION_THRESHOLD,
                        help=f"OWLv2 confidence floor (default: {DETECTION_THRESHOLD})")
    parser.add_argument("--resume", action="store_true",
                        help="Skip sessions with existing output file")
    parser.add_argument("--probe-name", default=None,
                        help="Probe mode: scene tag name (e.g. 'trash'). Output "
                             "is written to scene_tags_owlv2_{probe-name}.json so "
                             "the canonical scene_tags_owlv2.json is untouched.")
    parser.add_argument("--probe-queries", nargs="+", default=None,
                        help="Probe mode: OWLv2 text queries to detect. "
                             "Required when --probe-name is set.")
    args = parser.parse_args()

    if args.probe_name and not args.probe_queries:
        parser.error("--probe-name requires --probe-queries")
    if args.probe_queries and not args.probe_name:
        parser.error("--probe-queries requires --probe-name")

    if args.probe_name:
        queries_multi = {args.probe_name: list(args.probe_queries)}
        output_filename = f"scene_tags_owlv2_{args.probe_name}.json"
    else:
        queries_multi = QUERIES_MULTI
        output_filename = "scene_tags_owlv2.json"

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    sessions = [args.session] if args.session else get_sessions(args.participant)

    print(f"\nParticipant: {args.participant} | Sessions: {len(sessions)}")
    print(f"Device: {device}")
    print(f"Queries: {queries_multi}")
    print(f"Threshold: {args.threshold}")
    print(f"Output: {output_filename}")

    print(f"\nLoading OWLv2: {MODEL_ID}")
    from transformers import Owlv2ForObjectDetection, Owlv2Processor
    processor = Owlv2Processor.from_pretrained(MODEL_ID)
    model = Owlv2ForObjectDetection.from_pretrained(MODEL_ID).eval().to(device)
    print("Model loaded")

    for i, session in enumerate(sessions):
        if args.all:
            print(f"\n{'#' * 70}")
            print(f"# SESSION {i + 1}/{len(sessions)}: {session}")
            print(f"{'#' * 70}")

        if args.resume:
            out_file = outputs_dir(args.participant, session) / output_filename
            if out_file.exists():
                print("  SKIPPED (results exist)")
                continue

        run_session(
            participant=args.participant,
            session=session,
            model=model,
            processor=processor,
            device=device,
            batch_size=args.batch_size,
            min_score=args.min_score,
            threshold=args.threshold,
            queries_multi=queries_multi,
            output_filename=output_filename,
        )


if __name__ == "__main__":
    main()

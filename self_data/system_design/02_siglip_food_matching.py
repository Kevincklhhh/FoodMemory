#!/usr/bin/env python3
"""
02_siglip_food_matching.py - Match detected hand-object crops to food items using SigLIP2.

Uses SigLIP2 zero-shot image-text matching to compare each detected object crop
(from hands23 detector output) against the participant's food items from ledger.json.

Usage:
    python system_design/02_siglip_food_matching.py --participant kailai --session 20260310-195710
    python system_design/02_siglip_food_matching.py --participant kailai --session 20260310-195710 --threshold 0.15

Prerequisites:
    - {P}_{session}_hands23_results.json from 01_extract_and_detect_hands.py
    - ledger.json with food item visual_class names
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

from utils import get_sessions, hands23_dir, load_food_items, load_session_food_items, load_session_siglip_labels

MODEL_ID = "google/siglip2-so400m-patch14-384"

TEXT_TEMPLATES = [
    "a photo of {food_name}",
    "a photo of {food_name} in a kitchen",
    "{food_name} on a kitchen counter",
    "a person holding {food_name}",
    "a close-up of {food_name}",
]

NON_FOOD_LABELS = [
    "a knife", "a cutting board", "a pan", "a pot", "a spoon",
    "a plate", "a bowl", "a mug", "a sink", "a towel",
]


# =============================================================================
# TEXT EMBEDDINGS
# =============================================================================

def _expand_name(name: str) -> List[str]:
    """Expand 'food (variant/variant)' into multiple text variants."""
    import re
    m = re.match(r"^(.+?)\s*\((.+)\)$", name.strip())
    if not m:
        return [name.strip()]
    base = m.group(1).strip()
    variants = [base]
    for part in re.split(r"[/;]", m.group(2)):
        part = part.strip()
        if part and part != base:
            variants.append(part)
    return variants


def build_text_embeddings(
    model, processor, labels: List[str], templates: List[str], device: torch.device
) -> torch.Tensor:
    """Build L2-normalized text embeddings via template ensemble averaging.

    Each label string is expanded via _expand_name (handles "food (variant)" syntax),
    then each variant is formatted through all templates. All resulting texts are
    averaged into one embedding per label.
    """
    tokenizer = processor.tokenizer
    all_embs = []
    for label in labels:
        texts = []
        for variant in _expand_name(label):
            texts.extend(t.format(food_name=variant) for t in templates)
        inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            embs = model.get_text_features(**inputs)
        all_embs.append(embs.mean(dim=0))
    embs = torch.stack(all_embs)
    return F.normalize(embs, p=2, dim=-1)


def build_text_embeddings_multi(
    model, processor, label_config: List[dict], templates: List[str], device: torch.device
) -> Tuple[torch.Tensor, List[str]]:
    """Build text embeddings from label config with custom siglip_labels support.

    Args:
        label_config: List of {"display_name": str, "labels": List[str]} dicts.
            Each item may have multiple label strings; all are expanded and averaged
            into one embedding per item.

    Returns:
        (embeddings tensor, display_names list) — aligned by index.
    """
    tokenizer = processor.tokenizer
    all_embs = []
    display_names = []
    for item in label_config:
        texts = []
        for label in item["labels"]:
            for variant in _expand_name(label):
                texts.extend(t.format(food_name=variant) for t in templates)
        inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            embs = model.get_text_features(**inputs)
        all_embs.append(embs.mean(dim=0))
        display_names.append(item["display_name"])
    embs = torch.stack(all_embs)
    return F.normalize(embs, p=2, dim=-1), display_names


# =============================================================================
# IMAGE PROCESSING
# =============================================================================

def _crop_bbox(image: Image.Image, bbox: List[float], padding: float = 0.2) -> Optional[Image.Image]:
    """Crop bbox with padding, return None if degenerate."""
    iw, ih = image.size
    x0, y0, x1, y1 = bbox
    w, h = x1 - x0, y1 - y0
    if w < 10 or h < 10:
        return None
    x0 = max(0, x0 - w * padding)
    y0 = max(0, y0 - h * padding)
    x1 = min(iw, x1 + w * padding)
    y1 = min(ih, y1 + h * padding)
    return image.crop((int(x0), int(y0), int(x1), int(y1)))


def collect_crop_jobs(hands23_data: Dict, include_second_obj: bool = True) -> List[Dict]:
    """Collect all bbox crop jobs from hands23 results."""
    jobs = []
    for video in hands23_data["videos"]:
        video_id = video["video_id"]
        for frame in video["frames"]:
            ts = frame.get("session_timestamp_s", frame.get("clip_timestamp_s", 0))
            for det in frame.get("detections", []):
                if det.get("obj_bbox") is not None:
                    jobs.append({
                        "video_id": video_id,
                        "frame_path": frame["frame_path"],
                        "timestamp": ts,
                        "hand_side": det["hand_side"],
                        "contact_state": det["contact_state"],
                        "obj_touch": det.get("obj_touch"),
                        "obj_bbox": det["obj_bbox"],
                        "obj_score": det.get("obj_score"),
                        "bbox_type": "primary",
                    })
                if include_second_obj and det.get("second_obj_bbox") is not None:
                    jobs.append({
                        "video_id": video_id,
                        "frame_path": frame["frame_path"],
                        "timestamp": ts,
                        "hand_side": det["hand_side"],
                        "contact_state": det["contact_state"],
                        "obj_touch": det.get("obj_touch"),
                        "obj_bbox": det["second_obj_bbox"],
                        "obj_score": det.get("second_obj_score"),
                        "bbox_type": "secondary",
                    })
    return jobs


def encode_crops(model, processor, crops: List[Image.Image], device: torch.device) -> torch.Tensor:
    inputs = processor(images=crops, return_tensors="pt").to(device)
    with torch.no_grad():
        embs = model.get_image_features(**inputs)
    return F.normalize(embs, p=2, dim=-1).cpu()


def score_embeddings(
    image_embs: torch.Tensor,
    model,
    food_embs: torch.Tensor,
    food_items: List[str],
    non_food_embs: torch.Tensor,
    top_k: int,
    threshold: float,
    device: torch.device,
    batch_size: int = 2048,
) -> List[Dict]:
    logit_scale = model.logit_scale.exp()
    logit_bias = model.logit_bias

    results = []
    for start in range(0, len(image_embs), batch_size):
        batch = image_embs[start:start + batch_size].to(device)
        food_probs = torch.sigmoid(batch @ food_embs.T * logit_scale + logit_bias)
        nf_probs = torch.sigmoid(batch @ non_food_embs.T * logit_scale + logit_bias)

        for i in range(len(batch)):
            top_vals, top_idxs = food_probs[i].topk(min(top_k, len(food_items)))
            top_matches = [
                {"food_name": food_items[idx], "similarity": round(val, 4)}
                for val, idx in zip(top_vals.tolist(), top_idxs.tolist())
                if val >= threshold
            ]
            best_nf_val, best_nf_idx = nf_probs[i].max(dim=0)
            results.append({
                "top_matches": top_matches,
                "best_non_food": {
                    "label": NON_FOOD_LABELS[best_nf_idx.item()],
                    "similarity": round(best_nf_val.item(), 4),
                },
            })
    return results


def process_batch(model, processor, crops, food_embs, food_items, non_food_embs, top_k, threshold, device, timing=None):
    t0 = time.time()
    image_embs = encode_crops(model, processor, crops, device)
    if timing is not None:
        timing["encode_s"] += time.time() - t0
    t1 = time.time()
    results = score_embeddings(image_embs, model, food_embs, food_items, non_food_embs, top_k, threshold, device)
    if timing is not None:
        timing["score_s"] += time.time() - t1
    return results, image_embs


# =============================================================================
# MAIN PROCESSING
# =============================================================================

def run_session(
    participant: str,
    session: str,
    model,
    processor,
    food_embs: torch.Tensor,
    food_items: List[str],
    non_food_embs: torch.Tensor,
    top_k: int,
    threshold: float,
    batch_size: int,
    bbox_padding: float,
    device: torch.device,
    verbose: bool = False,
    text_embed_s: Dict[str, float] = None,
) -> Optional[Dict]:
    det_dir = hands23_dir(participant, session)
    hands23_file = det_dir / f"{participant}_{session}_hands23_results.json"
    output_file = det_dir / f"{participant}_{session}_siglip_matches.json"
    cache_file = det_dir / f"{participant}_{session}_siglip_image_embeddings.pt"

    if not hands23_file.exists():
        print(f"  SKIP: {hands23_file.name} not found")
        return None

    print(f"\n  Loading hands23 results: {hands23_file.name}")
    with open(hands23_file) as f:
        hands23_data = json.load(f)

    jobs = collect_crop_jobs(hands23_data, include_second_obj=True)
    print(f"  Crop jobs: {len(jobs)} across {len(hands23_data['videos'])} clips")

    # Group jobs by video for efficient image loading
    video_jobs: Dict[str, List[Tuple[int, Dict]]] = {}
    for idx, job in enumerate(jobs):
        vid = job["video_id"]
        video_jobs.setdefault(vid, []).append((idx, job))

    source_mtime = hands23_file.stat().st_mtime
    match_results = [None] * len(jobs)
    start_time = time.time()
    total_crops = 0
    crops_skipped = 0
    timing = {
        "cache_load_s": 0.0,
        "image_load_s": 0.0,
        "crop_s": 0.0,
        "encode_s": 0.0,
        "score_s": 0.0,
        "text_embed_food_s": (text_embed_s or {}).get("food", 0.0),
        "text_embed_non_food_s": (text_embed_s or {}).get("non_food", 0.0),
    }

    # Check cache
    if cache_file.exists():
        t_cache = time.time()
        cache = torch.load(cache_file, map_location="cpu", weights_only=False)
        timing["cache_load_s"] += time.time() - t_cache
        if cache.get("source_mtime") == source_mtime and cache.get("total_jobs") == len(jobs):
            all_embs = cache["image_embs"]
            valid_indices = cache["valid_indices"]
            total_crops = len(valid_indices)
            print(f"  Using cached embeddings ({total_crops} crops)")
            t_score = time.time()
            scores = score_embeddings(
                all_embs, model, food_embs, food_items,
                non_food_embs, top_k, threshold, device,
            )
            timing["score_s"] += time.time() - t_score
            for i, idx in enumerate(valid_indices):
                match_results[idx] = scores[i]
            print(f"  Re-scored in {time.time() - start_time:.1f}s")
            cache = True  # mark as used
        else:
            print("  Cache stale, re-encoding...")
            cache = None
    else:
        cache = None

    if cache is None:
        all_embs_list = []
        all_valid_indices = []

        for video_id in tqdm(sorted(video_jobs.keys()), desc="  Clips"):
            # Group by frame_path
            frame_groups: Dict[str, List[Tuple[int, Dict]]] = {}
            for idx, job in video_jobs[video_id]:
                frame_groups.setdefault(job["frame_path"], []).append((idx, job))

            batch_crops = []
            batch_indices = []

            for frame_path_rel, frame_jobs in frame_groups.items():
                frame_path = det_dir / frame_path_rel
                t_io = time.time()
                try:
                    image = Image.open(frame_path).convert("RGB")
                except (FileNotFoundError, OSError) as e:
                    if verbose:
                        print(f"    WARN: {frame_path_rel}: {e}")
                    crops_skipped += len(frame_jobs)
                    continue
                timing["image_load_s"] += time.time() - t_io

                for idx, job in frame_jobs:
                    t_crop = time.time()
                    crop = _crop_bbox(image, job["obj_bbox"], padding=bbox_padding)
                    timing["crop_s"] += time.time() - t_crop
                    if crop is None:
                        crops_skipped += 1
                        continue
                    batch_crops.append(crop)
                    batch_indices.append(idx)

                    if len(batch_crops) >= batch_size:
                        res, embs = process_batch(
                            model, processor, batch_crops, food_embs, food_items,
                            non_food_embs, top_k, threshold, device, timing,
                        )
                        for bi, br in zip(batch_indices, res):
                            match_results[bi] = br
                        all_embs_list.append(embs)
                        all_valid_indices.extend(batch_indices)
                        total_crops += len(batch_crops)
                        batch_crops = []
                        batch_indices = []

            # Flush
            if batch_crops:
                try:
                    res, embs = process_batch(
                        model, processor, batch_crops, food_embs, food_items,
                        non_food_embs, top_k, threshold, device, timing,
                    )
                    for bi, br in zip(batch_indices, res):
                        match_results[bi] = br
                    all_embs_list.append(embs)
                    all_valid_indices.extend(batch_indices)
                    total_crops += len(batch_crops)
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print(f"    OOM, halving batch...")
                        torch.cuda.empty_cache()
                        half = len(batch_crops) // 2
                        for sc, si in [(batch_crops[:half], batch_indices[:half]),
                                       (batch_crops[half:], batch_indices[half:])]:
                            if sc:
                                r2, e2 = process_batch(
                                    model, processor, sc, food_embs, food_items,
                                    non_food_embs, top_k, threshold, device, timing,
                                )
                                for bi, br in zip(si, r2):
                                    match_results[bi] = br
                                all_embs_list.append(e2)
                                all_valid_indices.extend(si)
                                total_crops += len(sc)
                    else:
                        raise

        # Save embedding cache
        all_embs = torch.cat(all_embs_list, dim=0) if all_embs_list else torch.empty(0)
        torch.save({
            "image_embs": all_embs,
            "valid_indices": all_valid_indices,
            "total_jobs": len(jobs),
            "source_mtime": source_mtime,
        }, cache_file)
        print(f"  Saved embedding cache: {cache_file.name}")

    elapsed = time.time() - start_time

    # Build output grouped by video
    video_job_map: Dict[str, List[int]] = {}
    for idx, job in enumerate(jobs):
        video_job_map.setdefault(job["video_id"], []).append(idx)

    output_videos = []
    for video_id in sorted(video_job_map.keys()):
        matches = []
        for idx in video_job_map[video_id]:
            result = match_results[idx]
            if result is None:
                continue
            job = jobs[idx]
            matches.append({
                "frame_path": job["frame_path"],
                "timestamp": job["timestamp"],
                "hand_side": job["hand_side"],
                "contact_state": job["contact_state"],
                "obj_touch": job["obj_touch"],
                "obj_bbox": job["obj_bbox"],
                "obj_score": job["obj_score"],
                "bbox_type": job["bbox_type"],
                "top_matches": result["top_matches"],
                "best_non_food": result["best_non_food"],
            })
        output_videos.append({"video_id": video_id, "matches": matches})

    unique_frames = len(set(j["frame_path"] for j in jobs))
    output = {
        "participant": participant,
        "session": session,
        "model": MODEL_ID,
        "food_source": "ledger.json",
        "num_food_items": len(food_items),
        "food_items": food_items,
        "threshold": threshold,
        "top_k": top_k,
        "bbox_padding": bbox_padding,
        "timestamp": datetime.now().isoformat(),
        "stats": {
            "total_frames_processed": unique_frames,
            "total_crops_processed": total_crops,
            "crops_skipped": crops_skipped,
            "processing_time_s": round(elapsed, 1),
        },
        "timing": {
            **{k: round(v, 2) for k, v in timing.items()},
            "counts": {
                "num_crop_jobs": len(jobs),
                "num_crops_encoded": total_crops,
                "num_crops_skipped": crops_skipped,
                "num_unique_frames": unique_frames,
                "num_food_items": len(food_items),
                "num_non_food_labels": len(NON_FOOD_LABELS),
                "num_text_templates": len(TEXT_TEMPLATES),
            },
        },
        "videos": output_videos,
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"  Output: {output_file}")
    print(f"  Crops: {total_crops} processed, {crops_skipped} skipped | {elapsed:.1f}s")
    print(f"  Timing: image_load={timing['image_load_s']:.1f}s  "
          f"crop={timing['crop_s']:.2f}s  "
          f"encode={timing['encode_s']:.1f}s  "
          f"score={timing['score_s']:.2f}s  "
          f"cache_load={timing['cache_load_s']:.1f}s")
    return output


def main():
    parser = argparse.ArgumentParser(
        description="Match hand-object crops to food items using SigLIP2"
    )
    parser.add_argument("--participant", required=True, help="Participant ID (e.g., kailai)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--session", help="Session ID (e.g., 20260310-195710)")
    group.add_argument("--all", action="store_true", help="Process all sessions with hands23 results")
    parser.add_argument("--top-k", type=int, default=5, help="Top food matches to store (default: 5)")
    parser.add_argument("--threshold", type=float, default=0.1, help="Min similarity threshold (default: 0.1)")
    parser.add_argument("--batch-size", type=int, default=64, help="GPU batch size (default: 64)")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device (default: cuda:0)")
    parser.add_argument("--bbox-padding", type=float, default=0.2, help="Bbox crop padding (default: 0.2)")
    parser.add_argument("--resume", action="store_true", help="Skip sessions with existing SigLIP results")
    parser.add_argument("--inventory-scope", choices=["full", "session"], default="full",
                        help="Which items to build SigLIP labels for. "
                             "'full' = all items in stock at session time (default); "
                             "'session' = GT-annotated subset only.")
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    sessions = [args.session] if args.session else get_sessions(args.participant)

    print(f"\nParticipant: {args.participant} | Sessions: {len(sessions)}")
    print(f"Device: {device}")

    print(f"\nLoading SigLIP2: {MODEL_ID}")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID).eval().to(device)
    print("Model loaded")

    print("Building non-food distractor embeddings...")
    t_nf = time.time()
    non_food_embs = build_text_embeddings(
        model, processor, NON_FOOD_LABELS, ["a photo of {food_name}"], device
    )
    non_food_embed_s = time.time() - t_nf
    print(f"  non-food text embed: {non_food_embed_s:.2f}s")

    # Cache food embeddings — rebuild only when label config changes between sessions
    cached_label_config = None
    food_embs = None
    food_items = None  # display names aligned with food_embs

    for i, session in enumerate(sessions):
        if args.all:
            print(f"\n{'#'*70}")
            print(f"# SESSION {i+1}/{len(sessions)}: {session}")
            print(f"{'#'*70}")

        # Check if hands23 results exist
        det_dir = hands23_dir(args.participant, session)
        h23_file = det_dir / f"{args.participant}_{session}_hands23_results.json"
        if not h23_file.exists():
            print(f"  SKIP: no hands23 results for {session}")
            continue

        if args.resume:
            match_file = det_dir / f"{args.participant}_{session}_siglip_matches.json"
            if match_file.exists():
                print(f"  SKIPPED (results exist: {match_file.name})")
                continue

        # Per-session label config from ledger (uses siglip_labels when present)
        label_config = load_session_siglip_labels(
            args.participant, session, scope=args.inventory_scope,
        )
        print(f"  Food items for session ({args.inventory_scope} scope): {len(label_config)}")
        for lc in label_config:
            if lc["labels"] != [lc["display_name"]]:
                print(f"    {lc['display_name']} -> {lc['labels']}")
            else:
                print(f"    {lc['display_name']}")

        food_text_embed_s = 0.0
        if label_config != cached_label_config:
            print("  Building food text embeddings...")
            t_fe = time.time()
            food_embs, food_items = build_text_embeddings_multi(
                model, processor, label_config, TEXT_TEMPLATES, device
            )
            food_text_embed_s = time.time() - t_fe
            cached_label_config = label_config
            print(f"  food text embed: {food_text_embed_s:.2f}s")

        run_session(
            participant=args.participant,
            session=session,
            model=model,
            processor=processor,
            food_embs=food_embs,
            food_items=food_items,
            non_food_embs=non_food_embs,
            top_k=args.top_k,
            threshold=args.threshold,
            batch_size=args.batch_size,
            bbox_padding=args.bbox_padding,
            device=device,
            verbose=args.verbose,
            text_embed_s={"food": food_text_embed_s, "non_food": non_food_embed_s},
        )


if __name__ == "__main__":
    main()

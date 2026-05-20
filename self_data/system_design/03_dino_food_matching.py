#!/usr/bin/env python3
"""
03_dino_food_matching.py - Match HOI object crops to food items using DINOv3 image embeddings.

Computes DINOv3 embeddings for reference product images, then compares against
each HOI-detected object crop via cosine similarity.

Usage:
    python system_design/03_dino_food_matching.py --participant kailai --session 20260310-195710 --device cuda:7
    python system_design/03_dino_food_matching.py --participant kailai --all --device cuda:7 --resume

Prerequisites:
    - Reference images in participants/{P}/reference_images/{instance_id}/product.*
    - hands23 results from 01_extract_and_detect_hands.py
    - DINOv3 model at kitchen/HDEPIC/models/dinov3/
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from utils import (
    get_sessions,
    hands23_dir,
    load_ledger,
    participant_dir,
)

_SCRIPT_DIR = Path(__file__).parent
_KITCHEN_DIR = _SCRIPT_DIR.parent.parent
_DINOV3_DIR = _KITCHEN_DIR / "HDEPIC" / "models" / "dinov3"

sys.path.insert(0, str(_DINOV3_DIR))

# DINOv3 7B checkpoint (local)
DINOV3_WEIGHTS = next(_DINOV3_DIR.glob("dinov3_vit7b16_pretrain_lvd1689m*.pth*"), None)


# =============================================================================
# MODEL
# =============================================================================

def load_dinov3(device: torch.device, dtype: torch.dtype = torch.bfloat16):
    """Load DINOv3 ViT-7B backbone with local weights."""
    from dinov3.hub.backbones import dinov3_vit7b16

    print(f"  Loading DINOv3 ViT-7B from local weights...")
    # Create model without pretrained weights, then load manually
    model = dinov3_vit7b16(pretrained=False)
    state_dict = torch.load(str(DINOV3_WEIGHTS), map_location="cpu", weights_only=False)
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device=device, dtype=dtype).eval()
    print(f"  Model loaded on {device} ({dtype})")
    return model


def build_transform(img_size: int = 518):
    """Build DINOv3 image transform (center crop + normalize)."""
    from torchvision import transforms

    return transforms.Compose([
        transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


@torch.no_grad()
def encode_images(
    model, transform, images: List[Image.Image], device: torch.device,
    dtype: torch.dtype = torch.bfloat16, batch_size: int = 32,
) -> torch.Tensor:
    """Encode images into L2-normalized DINOv3 CLS embeddings."""
    all_embs = []
    for i in range(0, len(images), batch_size):
        batch_imgs = images[i:i + batch_size]
        tensors = torch.stack([transform(img) for img in batch_imgs])
        tensors = tensors.to(device=device, dtype=dtype)
        feats = model(tensors)
        # DINOv3 returns dict or tensor depending on version
        if isinstance(feats, dict):
            cls_emb = feats.get("x_norm_clstoken", feats.get("cls_token"))
        else:
            cls_emb = feats
        # If shape is (B, T, D), take CLS token (index 0)
        if cls_emb.dim() == 3:
            cls_emb = cls_emb[:, 0]
        all_embs.append(cls_emb.float().cpu())
    return F.normalize(torch.cat(all_embs, dim=0), p=2, dim=-1)


# =============================================================================
# REFERENCE EMBEDDINGS
# =============================================================================

def load_reference_images(participant: str) -> List[dict]:
    """Load reference product images from participants/{P}/reference_images/.

    Discovers references by scanning for {instance_id}/product.* directories.
    Returns list of {instance_id, visual_class, image_path, image}.
    """
    ref_dir = participant_dir(participant) / "reference_images"
    if not ref_dir.exists():
        print(f"  No reference_images/ directory found")
        return []

    ledger = load_ledger(participant)
    refs = []
    for item_dir in sorted(ref_dir.iterdir()):
        if not item_dir.is_dir():
            continue
        iid = item_dir.name
        if iid not in ledger["items"]:
            continue
        # Find product image (any extension)
        img_files = list(item_dir.glob("product.*"))
        if not img_files:
            continue
        img_path = img_files[0]
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"  WARN: Cannot open {img_path}: {e}")
            continue
        refs.append({
            "instance_id": iid,
            "visual_class": ledger["items"][iid].get("visual_class", iid),
            "image_path": str(img_path),
            "image": img,
        })
    return refs


def build_reference_embeddings(
    model, transform, refs: List[dict], device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
) -> Tuple[torch.Tensor, List[str], List[str]]:
    """Encode reference images. Returns (embeddings, instance_ids, visual_classes)."""
    images = [r["image"] for r in refs]
    iids = [r["instance_id"] for r in refs]
    vcs = [r["visual_class"] for r in refs]
    embs = encode_images(model, transform, images, device, dtype)
    return embs, iids, vcs


# =============================================================================
# CROP PROCESSING
# =============================================================================

def crop_bbox(image: Image.Image, bbox: List[float], padding: float = 0.2) -> Optional[Image.Image]:
    """Crop bbox region with padding. Returns None if degenerate."""
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


def collect_crop_jobs(hands23_data: Dict) -> List[Dict]:
    """Collect all object bbox crop jobs from hands23 results."""
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
                if det.get("second_obj_bbox") is not None:
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


# =============================================================================
# MATCHING
# =============================================================================

def match_crops(
    crop_embs: torch.Tensor,
    ref_embs: torch.Tensor,
    ref_iids: List[str],
    ref_vcs: List[str],
    top_k: int = 3,
    threshold: float = 0.0,
) -> List[Dict]:
    """Compute cosine similarity between crop and reference embeddings.

    Returns list of match results (one per crop).
    """
    # cosine similarity (both already L2-normalized)
    sims = crop_embs @ ref_embs.T  # (N_crops, N_refs)

    results = []
    for i in range(len(crop_embs)):
        top_vals, top_idxs = sims[i].topk(min(top_k, len(ref_iids)))
        matches = []
        for val, idx in zip(top_vals.tolist(), top_idxs.tolist()):
            if val >= threshold:
                matches.append({
                    "instance_id": ref_iids[idx],
                    "visual_class": ref_vcs[idx],
                    "similarity": round(val, 4),
                })
        results.append({"top_matches": matches})
    return results


# =============================================================================
# SESSION PROCESSING
# =============================================================================

def run_session(
    participant: str,
    session: str,
    model,
    transform,
    ref_embs: torch.Tensor,
    ref_iids: List[str],
    ref_vcs: List[str],
    device: torch.device,
    dtype: torch.dtype,
    top_k: int,
    threshold: float,
    batch_size: int,
    bbox_padding: float,
    ref_embed_s: float = 0.0,
    inventory_scope: str = "full",
) -> Optional[Dict]:
    det_dir = hands23_dir(participant, session)
    hands23_file = list(det_dir.glob("*_hands23_results.json"))
    if not hands23_file:
        print(f"  SKIP: no hands23 results for {session}")
        return None

    suffix = "" if inventory_scope == "full" else f"_{inventory_scope}"
    output_file = det_dir / f"{participant}_{session}_dino_matches{suffix}.json"
    cache_file = det_dir / f"{participant}_{session}_dino_image_embeddings.pt"

    print(f"\n  Loading hands23: {hands23_file[0].name}")
    hands23_data = json.loads(hands23_file[0].read_text())
    jobs = collect_crop_jobs(hands23_data)
    print(f"  Crop jobs: {len(jobs)}")

    if not jobs:
        print(f"  No crops to process")
        return None

    # Filter references to items present in kitchen at session time
    from utils import load_full_inventory
    inventory = load_full_inventory(
        participant, session,
        include_depleted=(inventory_scope == "purchased_all"),
    )
    if inventory:
        inv_ids = {inv["instance_id"] for inv in inventory}
        sess_mask = [i for i, iid in enumerate(ref_iids) if iid in inv_ids]
        if sess_mask:
            sess_ref_embs = ref_embs[sess_mask]
            sess_ref_iids = [ref_iids[i] for i in sess_mask]
            sess_ref_vcs = [ref_vcs[i] for i in sess_mask]
            print(f"  Session inventory: {len(sess_mask)}/{len(ref_iids)} reference items")
        else:
            sess_ref_embs, sess_ref_iids, sess_ref_vcs = ref_embs, ref_iids, ref_vcs
    else:
        sess_ref_embs, sess_ref_iids, sess_ref_vcs = ref_embs, ref_iids, ref_vcs

    # Group by frame_path for efficient image loading
    frame_groups: Dict[str, List[Tuple[int, Dict]]] = {}
    for idx, job in enumerate(jobs):
        frame_groups.setdefault(job["frame_path"], []).append((idx, job))

    t0 = time.time()
    source_mtime = hands23_file[0].stat().st_mtime
    timing = {
        "cache_load_s": 0.0,
        "image_load_s": 0.0,
        "crop_s": 0.0,
        "encode_s": 0.0,
        "match_s": 0.0,
        "ref_embed_s": ref_embed_s,
    }
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Check embedding cache
    crop_embs = None
    all_indices = None
    if cache_file.exists():
        t_cache = time.time()
        cache = torch.load(cache_file, map_location="cpu", weights_only=False)
        timing["cache_load_s"] += time.time() - t_cache
        if cache.get("source_mtime") == source_mtime and cache.get("total_jobs") == len(jobs):
            crop_embs = cache["image_embs"].to(device=device, dtype=dtype)
            all_indices = cache["valid_indices"]
            print(f"  Using cached embeddings ({len(all_indices)} crops)")
        else:
            print(f"  Cache stale, re-encoding...")

    if crop_embs is None:
        all_crops = []
        all_indices = []
        skipped = 0

        for frame_path_rel in tqdm(sorted(frame_groups.keys()), desc="  Cropping"):
            frame_path = det_dir / frame_path_rel
            t_io = time.time()
            try:
                image = Image.open(frame_path).convert("RGB")
            except (FileNotFoundError, OSError):
                skipped += len(frame_groups[frame_path_rel])
                continue
            timing["image_load_s"] += time.time() - t_io

            for idx, job in frame_groups[frame_path_rel]:
                t_crop = time.time()
                crop = crop_bbox(image, job["obj_bbox"], padding=bbox_padding)
                timing["crop_s"] += time.time() - t_crop
                if crop is None:
                    skipped += 1
                    continue
                all_crops.append(crop)
                all_indices.append(idx)

        print(f"  Crops: {len(all_crops)} valid, {skipped} skipped")

        if not all_crops:
            return None

        # Encode all crops
        print(f"  Encoding {len(all_crops)} crops...")
        t_enc = time.time()
        crop_embs = encode_images(model, transform, all_crops, device, dtype, batch_size)
        timing["encode_s"] += time.time() - t_enc

        # Save embedding cache
        torch.save({
            "image_embs": crop_embs.cpu(),
            "valid_indices": all_indices,
            "source_mtime": source_mtime,
            "total_jobs": len(jobs),
        }, cache_file)
        print(f"  Saved embedding cache: {cache_file.name}")

    # Match against session-filtered references (ensure same device)
    print(f"  Matching against {len(sess_ref_iids)} session items...")
    t_match = time.time()
    match_results = match_crops(
        crop_embs.to(device=device, dtype=dtype),
        sess_ref_embs.to(device=device, dtype=dtype),
        sess_ref_iids, sess_ref_vcs, top_k, threshold,
    )
    timing["match_s"] += time.time() - t_match

    elapsed = time.time() - t0

    # Build output grouped by video
    video_job_map: Dict[str, List[int]] = {}
    for idx, job in enumerate(jobs):
        video_job_map.setdefault(job["video_id"], []).append(idx)

    idx_to_match = {all_indices[i]: match_results[i] for i in range(len(all_indices))}

    output_videos = []
    for video_id in sorted(video_job_map.keys()):
        matches = []
        for idx in video_job_map[video_id]:
            result = idx_to_match.get(idx)
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
            })
        output_videos.append({"video_id": video_id, "matches": matches})

    output = {
        "participant": participant,
        "session": session,
        "model": "dinov3_vit7b16",
        "reference_source": "receipt_product_images",
        "num_reference_items": len(sess_ref_iids),
        "reference_items": sess_ref_vcs,
        "threshold": threshold,
        "top_k": top_k,
        "bbox_padding": bbox_padding,
        "timestamp": datetime.now().isoformat(),
        "stats": {
            "total_crops_processed": len(all_indices),
            "crops_skipped": len(jobs) - len(all_indices),
            "processing_time_s": round(elapsed, 1),
        },
        "timing": {
            **{k: round(v, 2) for k, v in timing.items()},
            "counts": {
                "num_crop_jobs": len(jobs),
                "num_crops_encoded": len(all_indices),
                "num_crops_skipped": len(jobs) - len(all_indices),
                "num_unique_frames": len(frame_groups),
                "num_reference_items_total": len(ref_iids),
                "num_reference_items_session": len(sess_ref_iids),
            },
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
        "videos": output_videos,
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"  Output: {output_file}")
    print(f"  Crops: {len(all_indices)} processed, {len(jobs) - len(all_indices)} skipped | {elapsed:.1f}s")
    print(f"  Timing: image_load={timing['image_load_s']:.1f}s  "
          f"crop={timing['crop_s']:.2f}s  "
          f"encode={timing['encode_s']:.1f}s  "
          f"match={timing['match_s']:.2f}s  "
          f"cache_load={timing['cache_load_s']:.1f}s")

    # Quick stats
    all_matches = [m for v in output_videos for m in v["matches"]]
    with_food = [m for m in all_matches if m["top_matches"]]
    if with_food:
        best = max(with_food, key=lambda m: m["top_matches"][0]["similarity"])
        print(f"  Matches above threshold: {len(with_food)}")
        print(f"  Best: {best['top_matches'][0]['visual_class']} sim={best['top_matches'][0]['similarity']:.3f} at t={best['timestamp']:.1f}s")

    return output


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Match HOI crops to food items using DINOv3")
    parser.add_argument("--participant", required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--session", help="Single session")
    group.add_argument("--all", action="store_true", help="All sessions")
    parser.add_argument("--device", default="cuda:7")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.0,
                        help="Min similarity threshold (default: 0.0 = keep all)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--bbox-padding", type=float, default=0.2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--inventory-scope", choices=["full", "purchased_all"],
                        default="full",
                        help="Reference pool: 'full' = purchased & not depleted "
                             "before session (default); 'purchased_all' = every "
                             "iid ever purchased before session, depletion ignored. "
                             "When != 'full', output is written with a '_<scope>' "
                             "suffix so it doesn't overwrite the canonical run.")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    # Load reference images
    print(f"Loading reference images for {args.participant}...")
    refs = load_reference_images(args.participant)
    if not refs:
        print("ERROR: No reference images found. Run extract_receipt_images.py first.")
        return
    print(f"  {len(refs)} reference items: {[r['visual_class'] for r in refs]}")

    # Load model
    print(f"\nLoading DINOv3 on {device}...")
    model = load_dinov3(device, dtype)
    transform = build_transform()

    # Build reference embeddings
    print("Building reference embeddings...")
    t_ref = time.time()
    ref_embs, ref_iids, ref_vcs = build_reference_embeddings(model, transform, refs, device, dtype)
    ref_embed_s = time.time() - t_ref
    print(f"  Reference embedding shape: {ref_embs.shape} ({ref_embed_s:.2f}s)")

    # Process sessions
    sessions = [args.session] if args.session else get_sessions(args.participant)
    print(f"\nSessions to process: {len(sessions)}")

    for i, session in enumerate(sessions):
        if args.all:
            print(f"\n{'#'*70}")
            print(f"# SESSION {i+1}/{len(sessions)}: {session}")
            print(f"{'#'*70}")

        if args.resume:
            det_dir = hands23_dir(args.participant, session)
            suffix = "" if args.inventory_scope == "full" else f"_{args.inventory_scope}"
            match_file = det_dir / f"{args.participant}_{session}_dino_matches{suffix}.json"
            if match_file.exists():
                print(f"  SKIPPED (results exist)")
                continue

        run_session(
            participant=args.participant,
            session=session,
            model=model,
            transform=transform,
            ref_embs=ref_embs,
            ref_iids=ref_iids,
            ref_vcs=ref_vcs,
            device=device,
            dtype=dtype,
            top_k=args.top_k,
            threshold=args.threshold,
            batch_size=args.batch_size,
            bbox_padding=args.bbox_padding,
            ref_embed_s=ref_embed_s,
            inventory_scope=args.inventory_scope,
        )


if __name__ == "__main__":
    main()

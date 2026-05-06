#!/usr/bin/env python3
"""
07b_siglip_scene_tags.py — Tag HOI trigger frames with scene context using SigLIP2.

For each frame where SigLIP or DINOv2 food matching produced an above-threshold
hit (the same frames that appear in the planner prompt), classifies the *whole
frame* (not the bbox crop) against a small set of egocentric scene prompts:
  fridge, cabinet, countertop, sink

Output feeds into 06_avp_round1_remaining.py to give the planner lifecycle
context (retrieval @ fridge/cabinet → use @ countertop/sink → put-back @
fridge/cabinet). Companion to 07a_siglip_pkg_vs_deriv_proto.py, which scores
the same frames for package-vs-derivative visibility per item.

Usage:
    python system_design/07b_siglip_scene_tags.py --participant kailai --session 20260310-195710
    python system_design/07b_siglip_scene_tags.py --participant kailai --all --device cuda:0
    python system_design/07b_siglip_scene_tags.py --participant kailai --all --resume

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
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from utils import get_sessions, hands23_dir, outputs_dir

MODEL_ID = "google/siglip2-so400m-patch14-384"

# Single noun-heavy descriptive prompts (one per label). Kept in sync with
# the first phrasing in ENSEMBLE_PROMPTS below for backwards compatibility.
SCENE_PROMPTS = {
    "fridge":     "Inside of an open kitchen refrigerator with food on shelves.",
    "cabinet":    "Inside an open kitchen cabinet showing stacked food packages on shelves.",
    "countertop": "Close up of a kitchen countertop surface.",
    "sink":       "Looking down into a kitchen sink basin.",
    "prep":       "A frying pan or pot with food cooking on a stove.",
}

# The distractor is supposed to absorb "none of the above" frames. The
# previous prompt ("A general view of a kitchen room.") matched everything
# because the frames ARE kitchen views; that pulled cosine above every
# scene label. Replaced with active-cooking phrasings, which is the most
# common "neither storage nor sink" state. Frames that genuinely show
# cooking will now land on `distractor` (i.e. → `unknown` tag) while
# storage/sink frames stay competitive.
SCENE_DISTRACTOR = "A close up of food being prepared in cooking utensils."

# Per-label ensemble — multiple noun-heavy phrasings averaged into one
# embedding per label (mirrors 02_siglip_food_matching.py's TEXT_TEMPLATES
# pattern). Each label's final text embedding is the L2-normalized mean of
# the embeddings of all phrasings below.
ENSEMBLE_PROMPTS = {
    "fridge": [
        "Inside of an open kitchen refrigerator with food on shelves.",
        "Inside a refrigerator.",
        "Open fridge shelves.",
        "Food stored in a kitchen fridge.",
    ],
    "cabinet": [
        "Inside an open kitchen cabinet showing stacked food packages on shelves.",
        "An open cabinet door revealing pantry items.",
        "Cabinet interior with packaged groceries.",
        "Looking into an open kitchen cupboard.",
    ],
    "countertop": [
        "Close up of a kitchen countertop surface.",
        "A kitchen countertop.",
        "A kitchen prep surface.",
        "Looking down at a kitchen counter.",
    ],
    "sink": [
        "A stainless steel kitchen sink basin with a faucet above.",
        "Water running from a faucet into a kitchen sink.",
        "Dirty dishes piled inside a kitchen sink.",
        "A kitchen sink drain with soap suds.",
    ],
    "prep": [
        "A frying pan sitting on a gas or electric stove burner.",
        "A saucepan or pot placed on a stovetop flame.",
        "Food sizzling inside a pan on a stove burner.",
        "A stovetop with a hot pan heating food over a flame.",
    ],
}
# Distractor must NOT overlap semantically with any scene label — its job is
# to absorb frames that match none of the storage / sink / prep cues. Earlier
# "food being prepared" phrasings collided with `prep`. These describe a
# close-up hand-and-food view that's semantically orthogonal to any of the
# named scene categories.
DISTRACTOR_ENSEMBLE = [
    "An out-of-focus or unclear first-person view.",
    "A close-up of a hand holding a food item mid-air, with no background.",
    "A blurry transitional frame with no recognizable kitchen feature.",
]

SCENE_LABELS = list(SCENE_PROMPTS.keys())

MIN_FOOD_SCORE = 0.15

# Tag a frame `unknown` if (best_label_cosine − distractor_cosine) < SCENE_FLOOR.
# Cosine units (raw image · text), not sigmoid units. Typical egocentric
# whole-frame cosines on these prompts are 0.01-0.05, so a margin of 0.005
# is a meaningful cutoff. Was 0.15 when scoring used sigmoid probs; lowered
# after switching to cosine ranking (sigmoid flatlines on whole frames).
SCENE_FLOOR = 0.005


# ---------------------------------------------------------------------------
# Trigger frame collection
# ---------------------------------------------------------------------------

def collect_trigger_frames(
    participant: str,
    session: str,
    min_score: float = MIN_FOOD_SCORE,
) -> Dict[str, float]:
    """Collect frame_path → timestamp for frames with a food match above threshold.

    Union of SigLIP and DINO match files. Returns dict keyed by frame_path
    (relative to hands23_dir) with session timestamp as value.
    """
    det_dir = hands23_dir(participant, session)
    trigger: Dict[str, float] = {}

    # SigLIP matches
    sig_files = list(det_dir.glob("*_siglip_matches.json"))
    if sig_files:
        data = json.loads(sig_files[0].read_text())
        for video in data.get("videos", []):
            for m in video.get("matches", []):
                if any(
                    t.get("similarity", 0) >= min_score
                    for t in m.get("top_matches", [])
                ):
                    trigger[m["frame_path"]] = m["timestamp"]

    # DINO matches
    dino_files = list(det_dir.glob("*_dino_matches.json"))
    if dino_files:
        data = json.loads(dino_files[0].read_text())
        for video in data.get("videos", []):
            for m in video.get("matches", []):
                if any(
                    t.get("similarity", 0) >= min_score
                    for t in m.get("top_matches", [])
                ):
                    trigger.setdefault(m["frame_path"], m["timestamp"])

    return trigger


# ---------------------------------------------------------------------------
# SigLIP scene scoring
# ---------------------------------------------------------------------------

def build_scene_text_embeddings(
    model, processor, device: torch.device,
) -> Tuple[torch.Tensor, List[str]]:
    """Build text embeddings for scene prompts + distractor.

    For each label, embed every phrasing in ENSEMBLE_PROMPTS[label],
    average, then L2-normalize into one vector. Same pattern for the
    distractor (averaged across DISTRACTOR_ENSEMBLE). This noticeably
    boosts cosine vs a single bare prompt.

    Returns (embeddings [N_labels+1, D], label_names [N_labels+1]).
    """
    tokenizer = processor.tokenizer

    def _embed_ensemble(phrasings: List[str]) -> torch.Tensor:
        """L2-normalize EACH phrasing, then average unit vectors, then
        re-normalize. Averaging un-normalized embeddings lets phrasings
        with larger raw norms dominate the mean direction — that biases
        labels whose ensemble happens to contain high-norm text, which
        showed up as the 'everything is sink' result.
        """
        inputs = tokenizer(phrasings, padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            e = model.get_text_features(**inputs)
        e = F.normalize(e, p=2, dim=-1)           # unit per-phrasing
        mean = e.mean(dim=0)                       # centroid of unit vectors
        return F.normalize(mean, p=2, dim=-1)      # re-unitize

    label_names: List[str] = []
    embs: List[torch.Tensor] = []
    for label in SCENE_LABELS:
        phrasings = ENSEMBLE_PROMPTS.get(label) or [SCENE_PROMPTS[label]]
        embs.append(_embed_ensemble(phrasings))
        label_names.append(label)
    embs.append(_embed_ensemble(DISTRACTOR_ENSEMBLE or [SCENE_DISTRACTOR]))
    label_names.append("distractor")

    return torch.stack(embs), label_names


def encode_full_frames(
    model, processor, images: List[Image.Image], device: torch.device,
    batch_size: int = 32,
) -> torch.Tensor:
    """Encode full frames (not crops) into L2-normalized SigLIP image embeddings."""
    all_embs = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size]
        inputs = processor(images=batch, return_tensors="pt").to(device)
        with torch.no_grad():
            embs = model.get_image_features(**inputs)
        all_embs.append(embs.cpu())
    return F.normalize(torch.cat(all_embs, dim=0), p=2, dim=-1)


def score_scene(
    image_embs: torch.Tensor,
    scene_embs: torch.Tensor,
    model,
    device: torch.device,
    scene_labels: List[str],
    scene_floor: float = SCENE_FLOOR,
) -> List[dict]:
    """Score each image embedding against scene prompts.

    Stores BOTH raw cosine and SigLIP sigmoid probabilities. Tag selection
    uses raw cosine argmax (with the cosine `scene_floor` interpreted as a
    minimum margin above distractor) — sigmoid on whole egocentric frames
    flatlines to ~0 because cos × scale + bias is deeply negative, so
    cosine ranking is the only reliable primitive.
    """
    logit_scale = model.logit_scale.exp()
    logit_bias = model.logit_bias

    results = []
    batch_size = 2048
    for start in range(0, len(image_embs), batch_size):
        batch = image_embs[start:start + batch_size].to(device)
        cos = batch @ scene_embs.T                                # raw cosine
        probs = torch.sigmoid(cos * logit_scale + logit_bias)     # SigLIP head

        for i in range(len(batch)):
            cos_all = cos[i].tolist()
            sig_all = probs[i].tolist()
            n_scene = len(SCENE_LABELS)
            scene_scores = {scene_labels[j]: sig_all[j] for j in range(n_scene)}
            scene_cos    = {scene_labels[j]: cos_all[j] for j in range(n_scene)}
            distractor_score = sig_all[n_scene]
            distractor_cos   = cos_all[n_scene]

            best_label = max(SCENE_LABELS, key=lambda l: scene_cos[l])
            best_cos = scene_cos[best_label]

            # Use cosine for tag selection: best label must beat distractor
            # by at least `scene_floor` (interpreted as a cosine margin) AND
            # have positive cosine.
            margin = best_cos - distractor_cos
            if best_cos <= 0 or margin < scene_floor:
                tag = "unknown"
            else:
                tag = best_label

            results.append({
                "scene": tag,
                "cosine": {**scene_cos, "distractor": distractor_cos},
                "margin_over_distractor": round(margin, 4),
                "scores": scene_scores,
                "distractor_score": round(distractor_score, 4),
            })

    return results


# ---------------------------------------------------------------------------
# Session processing
# ---------------------------------------------------------------------------

def run_session(
    participant: str,
    session: str,
    model,
    processor,
    scene_embs: torch.Tensor,
    scene_labels: List[str],
    device: torch.device,
    batch_size: int = 32,
    min_score: float = MIN_FOOD_SCORE,
    scene_floor: float = SCENE_FLOOR,
) -> Optional[dict]:
    det_dir = hands23_dir(participant, session)
    out_dir = outputs_dir(participant, session)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_file = out_dir / "scene_tags.json"
    cache_file = det_dir / f"{participant}_{session}_scene_image_embeddings.pt"

    trigger_frames = collect_trigger_frames(participant, session, min_score)
    if not trigger_frames:
        print(f"  {session}: no trigger frames (no food matches >= {min_score})")
        return None

    print(f"  {session}: {len(trigger_frames)} trigger frames")

    # Sort by timestamp for deterministic ordering
    sorted_frames = sorted(trigger_frames.items(), key=lambda x: x[1])
    frame_paths = [fp for fp, _ in sorted_frames]
    timestamps = [ts for _, ts in sorted_frames]

    # Check embedding cache
    # Source mtime: max of siglip + dino match files
    source_mtimes = []
    for pattern in ("*_siglip_matches.json", "*_dino_matches.json"):
        for f in det_dir.glob(pattern):
            source_mtimes.append(f.stat().st_mtime)
    source_mtime = max(source_mtimes) if source_mtimes else 0

    image_embs = None
    cache_frame_paths = None
    if cache_file.exists():
        cache = torch.load(cache_file, map_location="cpu", weights_only=False)
        if (
            cache.get("source_mtime") == source_mtime
            and cache.get("frame_paths") == frame_paths
        ):
            image_embs = cache["image_embs"]
            cache_frame_paths = cache["frame_paths"]
            print(f"  Using cached scene embeddings ({len(image_embs)} frames)")
        else:
            print(f"  Cache stale, re-encoding...")

    if image_embs is None:
        images = []
        valid_indices = []
        skipped = 0

        for i, fp in enumerate(tqdm(frame_paths, desc="  Loading frames")):
            abs_path = det_dir / fp
            try:
                img = Image.open(abs_path).convert("RGB")
                images.append(img)
                valid_indices.append(i)
            except (FileNotFoundError, OSError) as e:
                skipped += 1
                continue

        if not images:
            print(f"  {session}: no valid frames loaded")
            return None

        if skipped:
            print(f"  Skipped {skipped} unreadable frames")

        print(f"  Encoding {len(images)} full frames...")
        t0 = time.time()
        image_embs = encode_full_frames(model, processor, images, device, batch_size)
        encode_time = time.time() - t0
        print(f"  Encoded in {encode_time:.1f}s")

        # If some frames were skipped, filter frame_paths/timestamps to valid only
        if len(valid_indices) < len(frame_paths):
            frame_paths = [frame_paths[i] for i in valid_indices]
            timestamps = [timestamps[i] for i in valid_indices]

        # Save cache
        torch.save({
            "image_embs": image_embs,
            "frame_paths": frame_paths,
            "source_mtime": source_mtime,
        }, cache_file)
        print(f"  Saved embedding cache: {cache_file.name}")

    # Score against scene prompts
    t0 = time.time()
    scene_results = score_scene(
        image_embs, scene_embs, model, device, scene_labels, scene_floor,
    )
    score_time = time.time() - t0

    # Build output
    frames_dict = {}
    scene_counts = {l: 0 for l in SCENE_LABELS + ["unknown"]}
    for fp, ts, result in zip(frame_paths, timestamps, scene_results):
        frames_dict[fp] = {
            "timestamp": round(ts, 2),
            "scene": result["scene"],
            "scores": {k: round(v, 4) for k, v in result["scores"].items()},
            "distractor_score": result["distractor_score"],
            # Raw cosine (image · text). Tag selection uses this, not sigmoid.
            "cosine": {k: round(v, 4) for k, v in result["cosine"].items()},
            "margin_over_distractor": result["margin_over_distractor"],
        }
        scene_counts[result["scene"]] += 1

    output = {
        "participant": participant,
        "session": session,
        "model": MODEL_ID,
        "prompts": SCENE_PROMPTS,
        "distractor_prompt": SCENE_DISTRACTOR,
        "min_food_score": min_score,
        "scene_floor": scene_floor,
        "timestamp": datetime.now().isoformat(),
        "stats": {
            "total_trigger_frames": len(frames_dict),
            "scene_counts": scene_counts,
            "score_time_s": round(score_time, 2),
        },
        "frames": frames_dict,
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"  Output: {output_file}")
    print(f"  Scene distribution: {scene_counts}")
    print(f"  Score time: {score_time:.2f}s")
    return output


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Tag HOI trigger frames with scene context using SigLIP2"
    )
    parser.add_argument("--participant", required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--session", help="Single session")
    group.add_argument("--all", action="store_true", help="All sessions")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--min-score", type=float, default=MIN_FOOD_SCORE,
                        help=f"Min food match score to trigger scene tagging (default: {MIN_FOOD_SCORE})")
    parser.add_argument("--scene-floor", type=float, default=SCENE_FLOOR,
                        help=f"Min scene score to assign a tag (default: {SCENE_FLOOR})")
    parser.add_argument("--resume", action="store_true",
                        help="Skip sessions with existing scene_tags.json")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    sessions = [args.session] if args.session else get_sessions(args.participant)
    print(f"\nParticipant: {args.participant} | Sessions: {len(sessions)}")
    print(f"Device: {device}")

    print(f"\nLoading SigLIP2: {MODEL_ID}")
    from transformers import AutoModel, AutoProcessor
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID).eval().to(device)
    print("Model loaded")

    print("Building scene text embeddings...")
    scene_embs, scene_labels = build_scene_text_embeddings(model, processor, device)
    print(f"  {len(scene_labels)} scene labels: {scene_labels}")

    for i, session in enumerate(sessions):
        if args.all:
            print(f"\n{'#' * 70}")
            print(f"# SESSION {i + 1}/{len(sessions)}: {session}")
            print(f"{'#' * 70}")

        if args.resume:
            out_file = outputs_dir(args.participant, session) / "scene_tags.json"
            if out_file.exists():
                print(f"  SKIPPED (results exist)")
                continue

        run_session(
            participant=args.participant,
            session=session,
            model=model,
            processor=processor,
            scene_embs=scene_embs,
            scene_labels=scene_labels,
            device=device,
            batch_size=args.batch_size,
            min_score=args.min_score,
            scene_floor=args.scene_floor,
        )


if __name__ == "__main__":
    main()

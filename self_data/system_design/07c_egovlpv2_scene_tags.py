#!/usr/bin/env python3
"""
07c_egovlpv2_scene_tags.py — Tag HOI trigger frames with scene context using
EgoVLPv2 (egocentric video-language pretraining).

Parallel alternative to 07b (SigLIP2). Same trigger-frame selection and output
schema, different backbone. Motivation: EgoVLPv2 was pretrained on EgoClip
(Ego4D narrations) so its text encoder expects narrative, action-centric
prompts ("#C C opens the refrigerator") rather than noun phrases ("inside an
open refrigerator"). That's a better fit for egocentric fisheye frames, which
are OOD for SigLIP2.

Per-frame strategy: each trigger frame is replicated across 16 time steps
(still-frame clip), resized/normalized per the EgoVLPv2 recipe, encoded once
via compute_video, then cosine-compared to per-scene narrative-prompt
ensembles encoded via compute_text. Output matches 07b's scene_tags.json
schema but writes to scene_tags_egovlpv2.json so both can coexist.

Checkpoint (manual step — Cloudflare blocks wget on cis.jhu.edu):
    Browser-download http://www.cis.jhu.edu/~shraman/EgoVLPv2/ckpts/Pre-trained/EgoVLPv2.pth
    to kitchen/self_data/models/EgoVLPv2/checkpoints/EgoVLPv2.pth

Usage:
    python system_design/07c_egovlpv2_scene_tags.py --participant kailai --session 20260310-195710
    python system_design/07c_egovlpv2_scene_tags.py --participant kailai --all --device cuda:0
    python system_design/07c_egovlpv2_scene_tags.py --participant kailai --all --resume

Prerequisites:
    - SigLIP matches: hands23_detection/*_siglip_matches.json  (from 02)
    - DINO matches:   hands23_detection/*_dino_matches.json    (from 03)
    - Checkpoint: models/EgoVLPv2/checkpoints/EgoVLPv2.pth
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

# Load system_design/utils.py under a private name (`sd_utils`) so we don't
# squat the `utils` slot in sys.modules — EgoVLPv2 has its own `utils` package
# that must resolve to `from utils.util import state_dict_data_parallel_fix`.
import importlib.util as _ilu  # noqa: E402
_sd_utils_spec = _ilu.spec_from_file_location(
    "sd_utils", str(Path(__file__).parent / "utils.py")
)
_sd_utils = _ilu.module_from_spec(_sd_utils_spec)
_sd_utils_spec.loader.exec_module(_sd_utils)
get_sessions = _sd_utils.get_sessions
hands23_dir = _sd_utils.hands23_dir
outputs_dir = _sd_utils.outputs_dir

# ---------------------------------------------------------------------------
# EgoVLPv2 import (sys.path + sibling-module workaround)
# ---------------------------------------------------------------------------

_EGOVLP_ROOT = Path(__file__).resolve().parents[1] / "models" / "EgoVLPv2"
_EGOVLP_CODE = _EGOVLP_ROOT / "EgoVLPv2"
_EGOVLP_CKPT_DIR = _EGOVLP_ROOT / "checkpoints"
DEFAULT_CKPT = _EGOVLP_CKPT_DIR / "EgoVLPv2.pth"

sys.path.insert(0, str(_EGOVLP_CODE))


def _inject_base_module():
    """Pre-inject the `base` package so `from base import BaseModel` resolves
    without triggering `base.base_dataset` (which pulls in PyAV, a dataset-only
    dep we don't need for inference)."""
    import importlib.util
    import types

    if "base" in sys.modules and hasattr(sys.modules["base"], "BaseModel"):
        return

    base_dir = _EGOVLP_CODE / "base"
    pkg = types.ModuleType("base")
    pkg.__path__ = [str(base_dir)]
    sys.modules["base"] = pkg

    spec = importlib.util.spec_from_file_location(
        "base.base_model", str(base_dir / "base_model.py")
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["base.base_model"] = mod
    spec.loader.exec_module(mod)
    pkg.BaseModel = mod.BaseModel


_inject_base_module()

# ---------------------------------------------------------------------------
# Scene prompts — narrative, EgoClip-style (#C = camera-wearer). One
# ensemble per label; all phrasings averaged (unit per phrasing, then
# re-normalized) to mirror 07b.
# ---------------------------------------------------------------------------

SCENE_PROMPTS = {
    "fridge":     "#C C opens the refrigerator door to look inside.",
    "cabinet":    "#C C opens a kitchen cabinet to look at stored food.",
    "countertop": "#C C places an item on the kitchen countertop.",
    "sink":       "#C C stands at the kitchen sink over the basin.",
    "prep":       "#C C cooks food in a pan on the stove.",
}

ENSEMBLE_PROMPTS = {
    "fridge": [
        "#C C opens the refrigerator door to look inside.",
        "#C C takes an item from the refrigerator shelf.",
        "#C C puts an item back into the fridge.",
        "#C C looks into the open refrigerator.",
    ],
    "cabinet": [
        "#C C opens a kitchen cabinet to look at stored food.",
        "#C C takes a package from the cabinet shelf.",
        "#C C puts a package back inside a kitchen cupboard.",
        "#C C opens the pantry cabinet door.",
    ],
    "countertop": [
        "#C C places an item on the kitchen countertop.",
        "#C C picks up an item from the counter surface.",
        "#C C chops food on the kitchen counter.",
        "#C C works on the kitchen prep counter.",
    ],
    "sink": [
        "#C C stands at the kitchen sink over the basin.",
        "#C C washes a dish in the kitchen sink.",
        "#C C rinses food under the faucet in the sink.",
        "#C C drains water into the kitchen sink.",
    ],
    "prep": [
        "#C C cooks food in a pan on the stove.",
        "#C C stirs food in a pot on the stovetop.",
        "#C C places a pan on the stove burner.",
        "#C C heats food in a pan over the stove flame.",
    ],
}

DISTRACTOR_ENSEMBLE = [
    "#C C walks through the kitchen.",
    "#C C looks around with no object in hand.",
    "#C C stands still facing away from any work surface.",
]

SCENE_LABELS = list(SCENE_PROMPTS.keys())
MIN_FOOD_SCORE = 0.15
# Cosine units (not sigmoid). EgoVLPv2 projection cosines are typically small
# in magnitude (0.01-0.10); require the winning label to beat the distractor
# by this margin, else tag `unknown`.
SCENE_FLOOR = 0.005

# ---------------------------------------------------------------------------
# Video preprocessing — 16-frame "still clip" from a single frame. EgoVLPv2
# was pretrained with 4-16 frame clips at 224x224, ImageNet normalization.
# ---------------------------------------------------------------------------

NUM_FRAMES = 16
INPUT_RES = 224
IMNET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
IMNET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)


def _build_transform():
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize(INPUT_RES, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(INPUT_RES),
        transforms.ToTensor(),  # [0,1], shape (C,H,W)
    ])


def frames_to_clips(
    images: List[Image.Image],
    transform,
) -> torch.Tensor:
    """Each still frame → (num_frames, C, H, W) tensor, then stack batch.
    Returns (N, T, C, H, W), un-normalized [0,1]. Mean/std normalization
    is applied inside batched encoder (constant subtract)."""
    clips = []
    for img in images:
        t = transform(img)  # (C, H, W)
        clip = t.unsqueeze(0).repeat(NUM_FRAMES, 1, 1, 1)  # (T, C, H, W)
        clips.append(clip)
    return torch.stack(clips, dim=0)  # (N, T, C, H, W)


# ---------------------------------------------------------------------------
# Trigger frames (identical to 07b)
# ---------------------------------------------------------------------------

def collect_trigger_frames(
    participant: str,
    session: str,
    min_score: float = MIN_FOOD_SCORE,
) -> Dict[str, float]:
    det_dir = hands23_dir(participant, session)
    trigger: Dict[str, float] = {}

    sig_files = list(det_dir.glob("*_siglip_matches.json"))
    if sig_files:
        data = json.loads(sig_files[0].read_text())
        for video in data.get("videos", []):
            for m in video.get("matches", []):
                if any(t.get("similarity", 0) >= min_score for t in m.get("top_matches", [])):
                    trigger[m["frame_path"]] = m["timestamp"]

    dino_files = list(det_dir.glob("*_dino_matches.json"))
    if dino_files:
        data = json.loads(dino_files[0].read_text())
        for video in data.get("videos", []):
            for m in video.get("matches", []):
                if any(t.get("similarity", 0) >= min_score for t in m.get("top_matches", [])):
                    trigger.setdefault(m["frame_path"], m["timestamp"])

    return trigger


# ---------------------------------------------------------------------------
# EgoVLPv2 model wrapper
# ---------------------------------------------------------------------------

class EgoVLPv2Scorer:
    def __init__(self, checkpoint_path: Path, device: torch.device, num_frames: int = NUM_FRAMES):
        from transformers import RobertaTokenizerFast
        import model.model as egovlp_model  # noqa: E402  (from EgoVLPv2 code)

        self.device = device
        self.num_frames = num_frames

        video_params = {
            "model": "SpaceTimeTransformer",
            "arch_config": "base_patch16_224",
            "num_frames": num_frames,
            "pretrained": True,
            "time_init": "zeros",
        }
        text_params = {"model": "roberta-base", "pretrained": True, "input": "text"}

        # task_names='EgoNCE' skips ITM/MLM heads — we only need the dual
        # encoder for cosine scoring. That trims ~50% of the param count.
        net = egovlp_model.FrozenInTime(
            video_params=video_params,
            text_params=text_params,
            projection="minimal",
            load_checkpoint=str(checkpoint_path),
            task_names="EgoNCE",
        )
        # Disable gradient checkpointing — pointless in eval/no_grad and
        # triggers warnings under newer torch.
        net.config["use_checkpoint"] = False
        net.eval().to(device)
        self.net = net

        self.tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")

        self._mean = IMNET_MEAN.to(device)
        self._std = IMNET_STD.to(device)

    @torch.no_grad()
    def encode_text(self, phrasings: List[str]) -> torch.Tensor:
        """Returns (N, D) L2-normalized text projections."""
        batch = self.tokenizer(
            phrasings, padding=True, truncation=True, max_length=64, return_tensors="pt",
        ).to(self.device)
        text_data = {"input_ids": batch["input_ids"], "attention_mask": batch["attention_mask"]}
        embs = self.net.compute_text(text_data)
        return F.normalize(embs, p=2, dim=-1)

    @torch.no_grad()
    def encode_video_clips(self, clips: torch.Tensor, batch_size: int = 4) -> torch.Tensor:
        """clips: (N, T, C, H, W) un-normalized [0,1]. Returns (N, D) L2-normalized."""
        all_embs = []
        for start in range(0, len(clips), batch_size):
            batch = clips[start:start + batch_size].to(self.device)
            batch = (batch - self._mean) / self._std
            embs = self.net.compute_video(batch)
            all_embs.append(F.normalize(embs, p=2, dim=-1).cpu())
        return torch.cat(all_embs, dim=0)


def build_scene_text_embeddings(scorer: EgoVLPv2Scorer) -> Tuple[torch.Tensor, List[str]]:
    """Unit-per-phrasing → mean → re-unitize. Same pattern as 07b."""
    label_names: List[str] = []
    embs: List[torch.Tensor] = []
    for label in SCENE_LABELS:
        phrasings = ENSEMBLE_PROMPTS.get(label) or [SCENE_PROMPTS[label]]
        e = scorer.encode_text(phrasings)      # already L2-normed per phrasing
        centroid = F.normalize(e.mean(dim=0), p=2, dim=-1)
        embs.append(centroid.cpu())
        label_names.append(label)
    d = scorer.encode_text(DISTRACTOR_ENSEMBLE)
    embs.append(F.normalize(d.mean(dim=0), p=2, dim=-1).cpu())
    label_names.append("distractor")
    return torch.stack(embs), label_names


def score_scene(
    image_embs: torch.Tensor,
    scene_embs: torch.Tensor,
    scene_labels: List[str],
    scene_floor: float = SCENE_FLOOR,
) -> List[dict]:
    """Raw cosine, then best-label-vs-distractor margin. No sigmoid (EgoVLPv2
    has no calibrated sigmoid head)."""
    n_scene = len(SCENE_LABELS)
    results = []
    cos = image_embs @ scene_embs.T  # (N, n_scene+1)
    for i in range(len(image_embs)):
        cos_all = cos[i].tolist()
        scene_cos = {scene_labels[j]: cos_all[j] for j in range(n_scene)}
        distractor_cos = cos_all[n_scene]

        best_label = max(SCENE_LABELS, key=lambda l: scene_cos[l])
        best_cos = scene_cos[best_label]
        margin = best_cos - distractor_cos
        tag = "unknown" if (best_cos <= 0 or margin < scene_floor) else best_label

        results.append({
            "scene": tag,
            "cosine": {**scene_cos, "distractor": distractor_cos},
            "margin_over_distractor": round(margin, 4),
        })
    return results


# ---------------------------------------------------------------------------
# Session processing
# ---------------------------------------------------------------------------

def run_session(
    participant: str,
    session: str,
    scorer: EgoVLPv2Scorer,
    scene_embs: torch.Tensor,
    scene_labels: List[str],
    transform,
    batch_size: int = 4,
    min_score: float = MIN_FOOD_SCORE,
    scene_floor: float = SCENE_FLOOR,
) -> Optional[dict]:
    det_dir = hands23_dir(participant, session)
    out_dir = outputs_dir(participant, session)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_file = out_dir / "scene_tags_egovlpv2.json"
    cache_file = det_dir / f"{participant}_{session}_scene_egovlpv2_embeddings.pt"

    trigger_frames = collect_trigger_frames(participant, session, min_score)
    if not trigger_frames:
        print(f"  {session}: no trigger frames (no food matches >= {min_score})")
        return None

    print(f"  {session}: {len(trigger_frames)} trigger frames")

    sorted_frames = sorted(trigger_frames.items(), key=lambda x: x[1])
    frame_paths = [fp for fp, _ in sorted_frames]
    timestamps = [ts for _, ts in sorted_frames]

    source_mtimes = []
    for pattern in ("*_siglip_matches.json", "*_dino_matches.json"):
        for f in det_dir.glob(pattern):
            source_mtimes.append(f.stat().st_mtime)
    source_mtime = max(source_mtimes) if source_mtimes else 0

    image_embs = None
    if cache_file.exists():
        cache = torch.load(cache_file, map_location="cpu", weights_only=False)
        if cache.get("source_mtime") == source_mtime and cache.get("frame_paths") == frame_paths:
            image_embs = cache["image_embs"]
            print(f"  Using cached EgoVLPv2 embeddings ({len(image_embs)} frames)")
        else:
            print("  Cache stale, re-encoding...")

    if image_embs is None:
        images = []
        valid_indices = []
        skipped = 0
        for i, fp in enumerate(tqdm(frame_paths, desc="  Loading frames")):
            abs_path = det_dir / fp
            try:
                images.append(Image.open(abs_path).convert("RGB"))
                valid_indices.append(i)
            except (FileNotFoundError, OSError):
                skipped += 1
        if not images:
            print(f"  {session}: no valid frames loaded")
            return None
        if skipped:
            print(f"  Skipped {skipped} unreadable frames")

        print(f"  Building {len(images)} {NUM_FRAMES}-frame still-clips...")
        clips = frames_to_clips(images, transform)
        print(f"  Encoding clips on {scorer.device}...")
        t0 = time.time()
        image_embs = scorer.encode_video_clips(clips, batch_size=batch_size)
        encode_time = time.time() - t0
        print(f"  Encoded {len(image_embs)} clips in {encode_time:.1f}s "
              f"({encode_time / max(len(image_embs), 1) * 1000:.1f} ms/clip)")

        if len(valid_indices) < len(frame_paths):
            frame_paths = [frame_paths[i] for i in valid_indices]
            timestamps = [timestamps[i] for i in valid_indices]

        torch.save({
            "image_embs": image_embs,
            "frame_paths": frame_paths,
            "source_mtime": source_mtime,
        }, cache_file)
        print(f"  Saved embedding cache: {cache_file.name}")

    t0 = time.time()
    scene_results = score_scene(image_embs, scene_embs, scene_labels, scene_floor)
    score_time = time.time() - t0

    frames_dict = {}
    scene_counts = {l: 0 for l in SCENE_LABELS + ["unknown"]}
    for fp, ts, result in zip(frame_paths, timestamps, scene_results):
        frames_dict[fp] = {
            "timestamp": round(ts, 2),
            "scene": result["scene"],
            "cosine": {k: round(v, 4) for k, v in result["cosine"].items()},
            "margin_over_distractor": result["margin_over_distractor"],
        }
        scene_counts[result["scene"]] += 1

    output = {
        "participant": participant,
        "session": session,
        "model": "EgoVLPv2",
        "prompts": SCENE_PROMPTS,
        "distractor_prompts": DISTRACTOR_ENSEMBLE,
        "num_frames_per_clip": NUM_FRAMES,
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
    return output


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Tag HOI trigger frames with scene context using EgoVLPv2"
    )
    parser.add_argument("--participant", required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--session", help="Single session")
    group.add_argument("--all", action="store_true", help="All sessions")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CKPT),
                        help=f"Path to EgoVLPv2.pth (default: {DEFAULT_CKPT})")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Clips per batch (each clip is 16 frames)")
    parser.add_argument("--min-score", type=float, default=MIN_FOOD_SCORE)
    parser.add_argument("--scene-floor", type=float, default=SCENE_FLOOR)
    parser.add_argument("--resume", action="store_true",
                        help="Skip sessions with existing scene_tags_egovlpv2.json")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"ERROR: checkpoint not found at {ckpt_path}", file=sys.stderr)
        print("\nManual download required (Cloudflare blocks wget on cis.jhu.edu):", file=sys.stderr)
        print("  1. Open in a browser:", file=sys.stderr)
        print("     http://www.cis.jhu.edu/~shraman/EgoVLPv2/ckpts/Pre-trained/EgoVLPv2.pth", file=sys.stderr)
        print(f"  2. Save as: {ckpt_path}", file=sys.stderr)
        sys.exit(1)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    sessions = [args.session] if args.session else get_sessions(args.participant)

    print(f"\nParticipant: {args.participant} | Sessions: {len(sessions)}")
    print(f"Device: {device}")
    print(f"Checkpoint: {ckpt_path}")

    print("\nLoading EgoVLPv2...")
    scorer = EgoVLPv2Scorer(ckpt_path, device, num_frames=NUM_FRAMES)
    print("Model loaded")

    print("Building scene text embeddings...")
    scene_embs, scene_labels = build_scene_text_embeddings(scorer)
    print(f"  {len(scene_labels)} scene labels: {scene_labels}")

    transform = _build_transform()

    for i, session in enumerate(sessions):
        if args.all:
            print(f"\n{'#' * 70}")
            print(f"# SESSION {i + 1}/{len(sessions)}: {session}")
            print(f"{'#' * 70}")

        if args.resume:
            out_file = outputs_dir(args.participant, session) / "scene_tags_egovlpv2.json"
            if out_file.exists():
                print("  SKIPPED (results exist)")
                continue

        run_session(
            participant=args.participant,
            session=session,
            scorer=scorer,
            scene_embs=scene_embs,
            scene_labels=scene_labels,
            transform=transform,
            batch_size=args.batch_size,
            min_score=args.min_score,
            scene_floor=args.scene_floor,
        )


if __name__ == "__main__":
    main()

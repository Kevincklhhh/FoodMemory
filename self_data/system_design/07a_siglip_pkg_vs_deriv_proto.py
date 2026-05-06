#!/usr/bin/env python3
"""Prototype: SigLIP2 package-vs-derivative scoring on HOI crops.

For each (session, item) test case, finds the HOI crops that 02_siglip already
matched to this item, re-crops the same bbox from the source frame, and scores
the crop against item-specific *package* vs *derivative* prompts.

Why crops, not whole frames: SigLIP2 sigmoid scores are calibrated for image-
text pairs where the text describes the dominant content of the image. On
egocentric whole frames, raw cosine sits around 0.02–0.05 → sigmoid ≈ 1e−5 and
the rankings are noise. On HOI crops (the held object filling the crop), 02
gets 0.20–0.40 — useful signal.

Scene tagging is delegated to 07b_siglip_scene_tags.py.

Run:
    python system_design/07a_siglip_pkg_vs_deriv_proto.py --participant kailai
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModel, AutoProcessor

sys.path.insert(0, str(Path(__file__).parent))
from utils import hands23_dir, participant_dir

MODEL_ID = "google/siglip2-so400m-patch14-384"
BBOX_PADDING = 0.2  # same as 02_siglip_food_matching.py

# Verdict thresholds on pkg_margin = pkg_cos_max - deriv_cos_max
# (raw cosine units — typical magnitudes on egocentric crops are 0.02-0.08).
# SigLIP2's sigmoid head squashes everything to ~0 on these prompts because
# logit_scale (~110) and logit_bias (~-16) are calibrated for noun-only food
# labels. Relative cosine still ranks pkg vs deriv reliably.
PKG_VISIBLE_MARGIN = 0.010
DERIVATIVE_MARGIN = -0.010


# ── Item configs ─────────────────────────────────────────────────────────────

ITEM_CONFIGS = {
    "kashi_peanut_butter_cereal_20260317": {
        "visual_class": "Kashi Peanut Butter Cereal",
        "distinguishability": "high",
        "package_prompts": [
            "a cereal box",
            "a Kashi cereal box",
            "hand holding a cereal box",
            "a branded cereal package",
            "a sealed cereal box",
        ],
        "derivative_prompts": [
            "cereal in a bowl",
            "cereal flakes on a plate",
            "loose cereal pieces",
            "cereal and milk in a bowl",
            "scattered cereal",
        ],
    },
    "large_white_eggs_20260310": {
        "visual_class": "Large White Eggs",
        "distinguishability": "high",
        "package_prompts": [
            "an egg carton",
            "an open egg carton with eggs",
            "hand holding an egg carton",
            "a dozen-egg carton",
            "a closed egg carton",
        ],
        "derivative_prompts": [
            "an egg cracked into a pan",
            "an egg in a bowl",
            "scrambled eggs in a pan",
            "egg yolk in a bowl",
            "egg shell on the counter",
        ],
    },
    "large_white_eggs_20260403": {
        "visual_class": "Large White Eggs",
        "distinguishability": "high",
        "package_prompts": [
            "an egg carton",
            "an open egg carton with eggs",
            "hand holding an egg carton",
            "a dozen-egg carton",
            "a closed egg carton",
        ],
        "derivative_prompts": [
            "an egg cracked into a pan",
            "an egg in a bowl",
            "scrambled eggs in a pan",
            "egg yolk in a bowl",
            "egg shell on the counter",
        ],
    },
    "whole_milk_gallon_20260318": {
        "visual_class": "Whole Milk Gallon",
        "distinguishability": "high",
        "package_prompts": [
            "a milk gallon jug",
            "a white plastic milk jug",
            "hand holding a gallon of milk",
            "a sealed milk jug with cap",
            "a plastic gallon container",
        ],
        "derivative_prompts": [
            "milk poured in a glass",
            "milk in a cereal bowl",
            "milk in a measuring cup",
            "white liquid in a glass",
            "milk in a pan",
        ],
    },
}


TEST_CASES = [
    ("20260323-094219", "kashi_peanut_butter_cereal_20260317"),
    ("20260406-093613", "kashi_peanut_butter_cereal_20260317"),
    ("20260410-095737", "large_white_eggs_20260403"),
    ("20260323-094219", "whole_milk_gallon_20260318"),
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def crop_bbox(image: Image.Image, bbox: List[float], padding: float = BBOX_PADDING) -> Optional[Image.Image]:
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


def collect_item_crops(
    participant: str,
    session: str,
    visual_class: str,
    min_score: float = 0.15,
) -> List[dict]:
    """Iterate 02's siglip_matches + 03's dino_matches and return all crops
    where the top match's food/visual class equals `visual_class`.
    """
    det_dir = hands23_dir(participant, session)
    crops: Dict[tuple, dict] = {}  # (frame_path, bbox_tuple, hand_side, bbox_type) -> entry

    # SigLIP matches (text-to-image)
    for sig_file in det_dir.glob("*_siglip_matches.json"):
        data = json.loads(sig_file.read_text())
        for video in data.get("videos", []):
            for m in video.get("matches", []):
                tops = m.get("top_matches", [])
                hit = next(
                    (t for t in tops if t.get("food_name") == visual_class
                     and t.get("similarity", 0) >= min_score),
                    None,
                )
                if not hit:
                    continue
                key = (m["frame_path"], tuple(m["obj_bbox"]), m.get("hand_side"), m.get("bbox_type"))
                crops.setdefault(key, {
                    "frame_path": m["frame_path"],
                    "timestamp": m["timestamp"],
                    "obj_bbox": m["obj_bbox"],
                    "hand_side": m.get("hand_side"),
                    "contact_state": m.get("contact_state"),
                    "bbox_type": m.get("bbox_type"),
                    "siglip_score": hit["similarity"],
                    "dino_score": None,
                    "source": "siglip",
                })

    # DINO matches (image-to-image; uses visual_class)
    for dino_file in det_dir.glob("*_dino_matches.json"):
        data = json.loads(dino_file.read_text())
        for video in data.get("videos", []):
            for m in video.get("matches", []):
                tops = m.get("top_matches", [])
                hit = next(
                    (t for t in tops if t.get("visual_class") == visual_class
                     and t.get("similarity", 0) >= min_score),
                    None,
                )
                if not hit:
                    continue
                key = (m["frame_path"], tuple(m["obj_bbox"]), m.get("hand_side"), m.get("bbox_type"))
                if key in crops:
                    crops[key]["dino_score"] = hit["similarity"]
                    if crops[key]["source"] == "siglip":
                        crops[key]["source"] = "siglip+dino"
                else:
                    crops[key] = {
                        "frame_path": m["frame_path"],
                        "timestamp": m["timestamp"],
                        "obj_bbox": m["obj_bbox"],
                        "hand_side": m.get("hand_side"),
                        "contact_state": m.get("contact_state"),
                        "bbox_type": m.get("bbox_type"),
                        "siglip_score": None,
                        "dino_score": hit["similarity"],
                        "source": "dino",
                    }

    return sorted(crops.values(), key=lambda x: x["timestamp"])


# Same 5 templates as 02_siglip_food_matching.py — ensemble averaging across
# templates produces a noticeably higher-cosine text embedding than a single
# bare prompt string (which is why our earlier whole-frame and bare-crop
# attempts collapsed to sigmoid ~ 0).
TEXT_TEMPLATES = [
    "a photo of {food_name}",
    "a photo of {food_name} in a kitchen",
    "{food_name} on a kitchen counter",
    "a person holding {food_name}",
    "a close-up of {food_name}",
]


def embed_texts(model, processor, prompts: List[str], device) -> torch.Tensor:
    """Embed each prompt by ensembling it through TEXT_TEMPLATES and averaging.

    Each entry in `prompts` is treated as a {food_name}-style noun phrase
    (e.g. "an egg carton") and inserted into all 5 templates; the resulting
    embeddings are averaged and L2-normalized into one vector per prompt.
    """
    tokenizer = processor.tokenizer
    out = []
    for p in prompts:
        texts = [t.format(food_name=p) for t in TEXT_TEMPLATES]
        inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            embs = model.get_text_features(**inputs)
        out.append(embs.mean(dim=0))
    embs = torch.stack(out)
    return F.normalize(embs, p=2, dim=-1)


def encode_crops(model, processor, crops: List[Image.Image], device) -> torch.Tensor:
    inputs = processor(images=crops, return_tensors="pt").to(device)
    with torch.no_grad():
        embs = model.get_image_features(**inputs)
    return F.normalize(embs, p=2, dim=-1)


def score_crop(
    image_emb: torch.Tensor,
    pkg_embs: torch.Tensor,
    deriv_embs: torch.Tensor,
    logit_scale: torch.Tensor,
    logit_bias: torch.Tensor,
) -> dict:
    """Score a crop with raw cosine (skip sigmoid — see PKG_VISIBLE_MARGIN
    docstring for why). pkg_max / deriv_max are cosine values, not probs.
    """
    def _cos_scores(text_embs):
        cos = (image_emb @ text_embs.T).squeeze(0)
        if cos.dim() == 0:
            cos = cos.unsqueeze(0)
        vals = cos.tolist()
        return max(vals), vals

    pkg_max, pkg_scores = _cos_scores(pkg_embs)
    deriv_max, deriv_scores = _cos_scores(deriv_embs)
    margin = pkg_max - deriv_max
    if margin > PKG_VISIBLE_MARGIN:
        verdict = "pkg_visible"
    elif margin < DERIVATIVE_MARGIN:
        verdict = "derivative"
    else:
        verdict = "ambiguous"
    return {
        "pkg_max": round(pkg_max, 4),
        "deriv_max": round(deriv_max, 4),
        "pkg_margin": round(margin, 4),
        "pkg_scores": [round(s, 4) for s in pkg_scores],
        "deriv_scores": [round(s, 4) for s in deriv_scores],
        "verdict": verdict,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="SigLIP2 package-vs-derivative scoring on HOI crops")
    parser.add_argument("--participant", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--min-match-score", type=float, default=0.15,
                        help="Min SigLIP/DINO match score to include a crop (default: 0.15)")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Loading SigLIP2: {MODEL_ID}")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID).eval().to(device)
    logit_scale = model.logit_scale.exp()
    logit_bias = model.logit_bias

    all_results = []

    for session, instance_id in TEST_CASES:
        cfg = ITEM_CONFIGS.get(instance_id)
        if not cfg:
            print(f"\n  SKIP {instance_id}: no item config")
            continue

        print(f"\n{'='*70}")
        print(f"Session: {session}  |  Item: {cfg['visual_class']} ({instance_id})")
        print(f"Distinguishability: {cfg['distinguishability']}")
        print(f"{'='*70}")

        item_crops = collect_item_crops(
            args.participant, session, cfg["visual_class"], args.min_match_score)
        print(f"  {len(item_crops)} HOI crops with item match (min_score={args.min_match_score})")
        if not item_crops:
            continue

        det_dir = hands23_dir(args.participant, session)
        pkg_embs = embed_texts(model, processor, cfg["package_prompts"], device)
        deriv_embs = embed_texts(model, processor, cfg["derivative_prompts"], device)

        # Load frame + crop bbox; batch encode for speed
        crop_imgs: List[Image.Image] = []
        crop_meta: List[dict] = []
        for entry in item_crops:
            abs_path = det_dir / entry["frame_path"]
            if not abs_path.exists():
                continue
            try:
                img = Image.open(abs_path).convert("RGB")
            except Exception:
                continue
            cropped = crop_bbox(img, entry["obj_bbox"])
            if cropped is None:
                continue
            crop_imgs.append(cropped)
            crop_meta.append(entry)

        if not crop_imgs:
            print("  No usable crops")
            continue

        # Encode all crops in batches
        BATCH = 32
        all_embs = []
        for i in range(0, len(crop_imgs), BATCH):
            all_embs.append(encode_crops(model, processor, crop_imgs[i:i + BATCH], device))
        crop_embs = torch.cat(all_embs, dim=0)

        # Score one-by-one (small N) and emit
        for emb, entry in zip(crop_embs, crop_meta):
            pkg = score_crop(emb.unsqueeze(0), pkg_embs, deriv_embs, logit_scale, logit_bias)
            sig_s = entry.get("siglip_score")
            din_s = entry.get("dino_score")
            sig_str = f"sig={sig_s:.2f}" if sig_s is not None else "sig=-"
            din_str = f"dino={din_s:.2f}" if din_s is not None else "dino=-"
            print(f"  t={entry['timestamp']:6.1f} {entry['hand_side'][:1].upper()}/{entry['bbox_type'][:3]} "
                  f"{sig_str} {din_str}  "
                  f"pkg={pkg['pkg_max']:.3f} deriv={pkg['deriv_max']:.3f} "
                  f"margin={pkg['pkg_margin']:+.3f} → {pkg['verdict']}")
            all_results.append({
                "session": session,
                "instance_id": instance_id,
                "frame_path": entry["frame_path"],
                "timestamp": entry["timestamp"],
                "obj_bbox": entry["obj_bbox"],
                "hand_side": entry["hand_side"],
                "bbox_type": entry["bbox_type"],
                "match_source": entry.get("source"),
                "siglip_score": sig_s,
                "dino_score": din_s,
                **pkg,
            })

    # Save results
    out_dir = participant_dir(args.participant) / "outputs"
    out_path = out_dir / "scene_context_proto_results.json"
    out_path.write_text(json.dumps(all_results, indent=2) + "\n")
    print(f"\n\nSaved {len(all_results)} scored crops to {out_path}")

    # Per-(session, item) summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    for session, instance_id in TEST_CASES:
        entries = [r for r in all_results
                   if r["session"] == session and r["instance_id"] == instance_id]
        if not entries:
            continue
        cfg = ITEM_CONFIGS[instance_id]
        n_pkg = sum(1 for e in entries if e["verdict"] == "pkg_visible")
        n_deriv = sum(1 for e in entries if e["verdict"] == "derivative")
        n_amb = sum(1 for e in entries if e["verdict"] == "ambiguous")
        avg_margin = sum(e["pkg_margin"] for e in entries) / len(entries)
        print(f"\n{cfg['visual_class']} @ {session}: {len(entries)} crops")
        print(f"  pkg_visible={n_pkg}  derivative={n_deriv}  ambiguous={n_amb}  "
              f"mean_margin={avg_margin:+.3f}")
        # Top 5 most pkg-like and top 5 most deriv-like
        for label, key in [("most pkg-like", lambda e: -e["pkg_margin"]),
                           ("most deriv-like", lambda e: e["pkg_margin"])]:
            print(f"  {label}:")
            for e in sorted(entries, key=key)[:3]:
                print(f"    t={e['timestamp']:6.1f}  margin={e['pkg_margin']:+.3f}  "
                      f"pkg={e['pkg_max']:.3f}/{e['deriv_max']:.3f}=deriv")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Step 3 of the DINO pkg-vs-derivative filter pipeline: embed and filter.

For every item flagged by 07f and fetched by 07g, this script:
  1. Loads the item's package reference image (reference_images/<iid>/product.*)
  2. Loads the fetched derivative images (derivative_anchors/_query_cache/<q>/*.jpg)
  3. Embeds all images with DINOv3 ViT-7B (same model as 03_dino_food_matching.py)
  4. Runs a within-query self-consistency filter — drops images whose mean
     cosine to the rest of their own query pool is in the bottom fraction
     (default 0.2). This removes stray search hits like "ground sirloin
     burgers" inside a "raw ground sirloin on a plate" pool.
  5. Computes diagnostic scores per item:
        - item_cohesion: mean pairwise cos across all surviving deriv embs
        - pkg_deriv_sep: cos(pkg, mean(deriv_survivors))
        A healthy item has cohesion high (≳0.35) and pkg_deriv_sep low (≲0.5).
  6. Saves one torch file `derivative_anchors/anchors.pt` plus a human-readable
     BUILD_REPORT.json with per-item stats.

Usage:
    python system_design/07h_build_derivative_anchors.py --participant kailai
    python system_design/07h_build_derivative_anchors.py --participant kailai \
        --only fresh_blueberries_20260328 \
        --only blueberries_20260415 \
        --only jumbo_blueberries_20260402
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
# Reuse 03's DINOv3 loader + transform + encoder
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "_dino_mod",
    Path(__file__).parent / "03_dino_food_matching.py",
)
_dino_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_dino_mod)
load_dinov3 = _dino_mod.load_dinov3
build_transform = _dino_mod.build_transform
encode_images = _dino_mod.encode_images

from utils import load_ledger, participant_dir  # noqa: E402


DEFAULT_DROP_FRAC = 0.20          # drop worst 20% per-query
DEFAULT_MIN_PER_QUERY = 3         # stop dropping if a query would fall below this
DEFAULT_DTYPE = torch.bfloat16
CACHE_DIRNAME = "_query_cache"


def _load_image(path: Path) -> Optional[Image.Image]:
    try:
        return Image.open(path).convert("RGB")
    except Exception as e:
        print(f"    WARN: cannot open {path}: {e}")
        return None


def _pairwise_cos(embs: torch.Tensor) -> torch.Tensor:
    """(N, D) L2-normalized -> (N, N) cosine matrix."""
    return embs @ embs.T


def _mean_cos_to_others(embs: torch.Tensor) -> torch.Tensor:
    n = embs.shape[0]
    if n <= 1:
        return torch.ones(n)
    sim = _pairwise_cos(embs)
    # Subtract diag (self-similarity 1.0), divide by (n-1) for the mean
    return (sim.sum(dim=1) - 1.0) / (n - 1)


def _filter_query_pool(
    embs: torch.Tensor,
    drop_frac: float,
    min_keep: int,
) -> torch.Tensor:
    """Return indices (sorted ascending) of kept images within one query pool.

    Drops the worst `drop_frac` of images by mean-cos-to-others, but never
    below `min_keep` survivors.
    """
    n = embs.shape[0]
    if n <= min_keep:
        return torch.arange(n)
    scores = _mean_cos_to_others(embs)
    order = scores.argsort(descending=True)        # best first
    n_keep = max(min_keep, int(round(n * (1 - drop_frac))))
    n_keep = min(n_keep, n)
    keep_idx = order[:n_keep].sort().values
    return keep_idx


def _load_item_derivatives(
    anchors_root: Path, item_manifest_path: Path,
) -> List[dict]:
    """Return [{query, file_path, rel_path}] for a single item, grouping by query."""
    im = json.loads(item_manifest_path.read_text())
    out: List[dict] = []
    for q in im.get("queries", []):
        for rel in q.get("files", []):
            abs_path = anchors_root / rel
            out.append({
                "query": q["query"],
                "cache_dir": q["cache_dir"],
                "rel_path": rel,
                "abs_path": abs_path,
            })
    return out


def build_item(
    iid: str,
    entry: dict,
    participant: str,
    anchors_root: Path,
    model,
    transform,
    device: torch.device,
    dtype: torch.dtype,
    drop_frac: float,
    min_per_query: int,
) -> Optional[dict]:
    pdir = participant_dir(participant)

    # --- package ref ---
    pkg_rel = entry.get("image_file")  # from confusable_profile.json
    pkg_path = pdir / "reference_images" / pkg_rel if pkg_rel else None
    if not pkg_path or not pkg_path.exists():
        # Fall back to globbing
        item_dir = pdir / "reference_images" / iid
        cand = list(item_dir.glob("product.*")) if item_dir.exists() else []
        pkg_path = cand[0] if cand else None
    if not pkg_path or not pkg_path.exists():
        print(f"  SKIP {iid}: no package reference image")
        return None
    pkg_img = _load_image(pkg_path)
    if pkg_img is None:
        return None

    # --- derivative refs ---
    item_manifest = anchors_root / f"{iid}.json"
    if not item_manifest.exists():
        print(f"  SKIP {iid}: no item manifest in derivative_anchors/")
        return None
    deriv_entries = _load_item_derivatives(anchors_root, item_manifest)
    if not deriv_entries:
        print(f"  SKIP {iid}: empty derivative manifest")
        return None

    deriv_imgs: List[Image.Image] = []
    kept_entries: List[dict] = []
    for de in deriv_entries:
        img = _load_image(de["abs_path"])
        if img is None:
            continue
        deriv_imgs.append(img)
        kept_entries.append(de)

    print(f"  {iid}: 1 pkg + {len(kept_entries)} derivatives")

    # --- embed ---
    t0 = time.time()
    all_imgs = [pkg_img] + deriv_imgs
    embs = encode_images(model, transform, all_imgs, device, dtype=dtype)
    print(f"    embedded {len(all_imgs)} imgs in {time.time()-t0:.1f}s")

    pkg_emb = embs[:1]                      # (1, D)
    deriv_embs_all = embs[1:]               # (N, D)

    # --- within-query filter ---
    # Group indices by cache_dir
    groups: Dict[str, List[int]] = {}
    for i, de in enumerate(kept_entries):
        groups.setdefault(de["cache_dir"], []).append(i)

    kept_mask = torch.zeros(deriv_embs_all.shape[0], dtype=torch.bool)
    drop_log: List[dict] = []
    for qdir, idxs in groups.items():
        idxs_t = torch.tensor(idxs)
        sub_embs = deriv_embs_all[idxs_t]
        keep_local = _filter_query_pool(sub_embs, drop_frac, min_per_query)
        keep_global = idxs_t[keep_local]
        dropped_global = [j for j in idxs if j not in set(keep_global.tolist())]
        kept_mask[keep_global] = True
        drop_log.append({
            "cache_dir": qdir,
            "n_in": len(idxs),
            "n_kept": len(keep_global),
            "dropped_files": [kept_entries[j]["rel_path"] for j in dropped_global],
        })

    deriv_embs = deriv_embs_all[kept_mask]
    deriv_sources = [kept_entries[i]["rel_path"]
                     for i, m in enumerate(kept_mask.tolist()) if m]
    deriv_queries = [kept_entries[i]["query"]
                     for i, m in enumerate(kept_mask.tolist()) if m]

    # --- diagnostics ---
    if deriv_embs.shape[0] >= 2:
        sim = _pairwise_cos(deriv_embs)
        n = sim.shape[0]
        cohesion = ((sim.sum() - n) / (n * (n - 1))).item()
    else:
        cohesion = 1.0

    deriv_centroid = torch.nn.functional.normalize(
        deriv_embs.mean(dim=0, keepdim=True), p=2, dim=-1,
    )
    pkg_deriv_sep = (pkg_emb @ deriv_centroid.T).item()
    # Also: max cos from pkg to any single deriv ref (worst-case anchor overlap)
    pkg_vs_deriv_max = (pkg_emb @ deriv_embs.T).max().item() if deriv_embs.shape[0] else 0.0

    rec = {
        "pkg_embs": pkg_emb.cpu(),
        "pkg_sources": [str(pkg_path.relative_to(pdir))],
        "deriv_embs": deriv_embs.cpu(),
        "deriv_sources": deriv_sources,
        "deriv_queries": deriv_queries,
        "meta": {
            "visual_class": entry.get("visual_class"),
            "pre_filter_count": len(kept_entries),
            "survivors": int(kept_mask.sum().item()),
            "drop_frac": drop_frac,
            "min_per_query": min_per_query,
            "item_cohesion": round(cohesion, 4),
            "pkg_deriv_sep": round(pkg_deriv_sep, 4),
            "pkg_deriv_max": round(pkg_vs_deriv_max, 4),
            "drop_log": drop_log,
            "dinov3_model": "dinov3_vit7b16",
            "built_at": datetime.now().isoformat(),
        },
    }
    print(f"    survivors={rec['meta']['survivors']}  "
          f"cohesion={rec['meta']['item_cohesion']}  "
          f"pkg⟂deriv={rec['meta']['pkg_deriv_sep']:+.3f}  "
          f"pkg⟂deriv_max={rec['meta']['pkg_deriv_max']:+.3f}")
    return rec


def main():
    parser = argparse.ArgumentParser(
        description="Build DINOv3 derivative anchors for confusable items.",
    )
    parser.add_argument("--participant", required=True)
    parser.add_argument("--only", action="append", default=None,
                        help="Restrict to these instance_ids (repeatable)")
    parser.add_argument("--drop-frac", type=float, default=DEFAULT_DROP_FRAC)
    parser.add_argument("--min-per-query", type=int, default=DEFAULT_MIN_PER_QUERY)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--out", default=None,
                        help="Override output path (default: derivative_anchors/anchors.pt)")
    args = parser.parse_args()

    pdir = participant_dir(args.participant)
    profile_path = pdir / "confusable_profile.json"
    if not profile_path.exists():
        print(f"ERROR: {profile_path} missing — run 07f first")
        sys.exit(1)
    anchors_root = pdir / "derivative_anchors"
    if not anchors_root.exists():
        print(f"ERROR: {anchors_root} missing — run 07g first")
        sys.exit(1)

    profile = json.loads(profile_path.read_text())
    items = profile.get("items", {})

    targets: List[tuple] = []
    for iid, entry in items.items():
        if not entry.get("derivative_confusion_risk"):
            continue
        if args.only and iid not in args.only:
            continue
        # Must have a fetched item manifest
        if not (anchors_root / f"{iid}.json").exists():
            continue
        targets.append((iid, entry))

    if args.only:
        missing = set(args.only) - {iid for iid, _ in targets}
        if missing:
            print(f"WARN: --only ids not eligible: {sorted(missing)}")

    print(f"Participant: {args.participant}")
    print(f"Device:      {args.device}")
    print(f"dtype:       {args.dtype}")
    print(f"Targets:     {len(targets)} item(s)")
    if not targets:
        return

    device = torch.device(args.device)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]
    print("\nLoading DINOv3 ViT-7B...")
    model = load_dinov3(device, dtype=dtype)
    transform = build_transform()

    # Merge into existing anchors file if present (per-item rebuild)
    out_path = Path(args.out) if args.out else anchors_root / "anchors.pt"
    if out_path.exists():
        existing = torch.load(str(out_path), map_location="cpu", weights_only=False)
    else:
        existing = {}

    report_items: Dict[str, dict] = {}
    for i, (iid, entry) in enumerate(targets):
        print(f"\n[{i+1}/{len(targets)}] {iid}")
        rec = build_item(
            iid=iid, entry=entry, participant=args.participant,
            anchors_root=anchors_root, model=model, transform=transform,
            device=device, dtype=dtype,
            drop_frac=args.drop_frac, min_per_query=args.min_per_query,
        )
        if rec is None:
            continue
        existing[iid] = rec
        # Serializable summary (drop tensors) for the report
        report_items[iid] = {
            "visual_class": rec["meta"]["visual_class"],
            **{k: v for k, v in rec["meta"].items() if k != "drop_log"},
            "n_dropped_total": sum(
                len(d["dropped_files"]) for d in rec["meta"]["drop_log"]
            ),
        }

    torch.save(existing, str(out_path))
    print(f"\nSaved anchors: {out_path}  ({len(existing)} items in file)")

    # Build report (merge with prior if extending)
    report_path = anchors_root / "BUILD_REPORT.json"
    prior_report = {}
    if report_path.exists():
        try:
            prior_report = json.loads(report_path.read_text()).get("items", {})
        except json.JSONDecodeError:
            prior_report = {}
    prior_report.update(report_items)
    report = {
        "participant": args.participant,
        "timestamp": datetime.now().isoformat(),
        "drop_frac": args.drop_frac,
        "min_per_query": args.min_per_query,
        "dinov3_model": "dinov3_vit7b16",
        "items": prior_report,
    }
    report_path.write_text(json.dumps(report, indent=2))
    print(f"Saved report:  {report_path}")

    # Console summary of this run
    print(f"\n{'item':<55} {'n':>3} {'cohesion':>9} {'pkg_sep':>8} {'pkg_max':>8}")
    for iid, s in report_items.items():
        print(f"  {iid:<53} {s['survivors']:>3}"
              f"  {s['item_cohesion']:>7.3f}  {s['pkg_deriv_sep']:>+7.3f}"
              f"  {s['pkg_deriv_max']:>+7.3f}")


if __name__ == "__main__":
    main()

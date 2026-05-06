#!/usr/bin/env python3
"""Step 1 of the DINO package-vs-derivative filter pipeline.

For each inventory item with a product reference image, call a VLM to decide
whether the item suffers from "Package/Derivative Visual Confusion" — i.e.
whether a crop of the raw/dispensed food would look close enough to a crop of
the packaged product that DINOv2 could not tell them apart.

Input : participants/{P}/reference_images/<instance_id>/product.{webp,jpg,png}
        (every on-disk product image is profiled; visual_class pulled from
         ledger.json, optional product_name from receipt_image_mapping.json)
Output: participants/{P}/confusable_profile.json   (one entry per item)

Items flagged `derivative_confusion_risk=true` also get 1-3 generic image
search queries that describe the food's dispensed form ("blueberries in a
bowl"), which Step 2 (auto-anchor builder) will use to fetch derivative
reference images for the DINO duel.

Usage:
    python system_design/07f_confusable_profile.py --participant kailai
    python system_design/07f_confusable_profile.py --participant kailai --force
    python system_design/07f_confusable_profile.py --participant kailai --only blueberries_20260415
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

from dotenv import load_dotenv
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from utils import load_ledger, participant_dir  # noqa: E402

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

GPT_MODEL_DEFAULT = "gpt-5.4"
IMAGE_MAX_SIDE = 1024
JPEG_QUALITY = 90

PROMPT = """You are an inventory profiling system. Given a product image and \
item name, determine whether this item suffers from "Package/Derivative \
Visual Confusion."

Item Name: {item_name}

Confusion occurs when the raw food INSIDE the package dominates the \
package's visual appearance, so a crop of the food in its dispensed state \
(derivative) would look nearly identical to a crop of the packaged product. \
Examples:
- Clear plastic clamshells of berries  → loose berries in a bowl
- Mesh bags of onions, garlic, oranges → loose onions/oranges on a counter
- Transparent bags of spinach, salad, pasta → loose leaves or pasta pieces
- Produce sold loose (apples, bananas, tomatoes, carrots) — the food IS the product
- Opaque packages with a large, dominant photo of the food on the label \
  (e.g. a cereal box whose front is mostly a big photo of cereal in a bowl)

Confusion does NOT occur when the package is opaque and the dispensed food \
looks completely different from the container: milk cartons, olive oil \
bottles, yogurt tubs, canned goods, paper-wrapped butter, cardboard boxes \
without dominant food photos.

Output STRICT JSON only — no prose, no markdown, no code fences:
{{
  "is_transparent_package": <bool: true iff the package is see-through, \
mesh, open, or the product is sold loose with no package>,
  "derivative_confusion_risk": <bool: true iff a crop of the dispensed food \
could be mistaken for a crop of the packaged product>,
  "derivative_search_queries": <array of 1-3 short Google Image search \
queries describing the dispensed/derivative form (e.g. \
["blueberries in a bowl", "loose blueberries on a cutting board"]). \
Use an empty array [] when risk is false.>,
  "reason": <one short sentence explaining the call>
}}
"""


# ── Azure OpenAI client (same pattern as 07e) ──────────────────────────────

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
            "Missing Azure OpenAI credentials "
            "(AZURE_OPENAI_API_KEY[_2] / AZURE_OPENAI_ENDPOINT[_2])"
        )
    return AzureOpenAI(
        azure_endpoint=endpoint, api_key=api_key, api_version="2025-03-01-preview",
    )


# ── Image encoding ─────────────────────────────────────────────────────────

def image_to_jpeg_b64(img: Image.Image, max_side: int = IMAGE_MAX_SIDE) -> str:
    w, h = img.size
    m = max(w, h)
    if m > max_side:
        s = max_side / m
        img = img.resize((int(w * s), int(h * s)), Image.LANCZOS)
    if img.mode != "RGB":
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=JPEG_QUALITY)
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ── VLM call ───────────────────────────────────────────────────────────────

def _extract_json(text: str) -> Optional[dict]:
    if not text:
        return None
    candidates: List[str] = [text]
    for m in re.finditer(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL):
        candidates.append(m.group(1))
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        candidates.append(m.group(0))
    for c in candidates:
        try:
            obj = json.loads(c)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    return None


def _normalize(obj: dict) -> Optional[dict]:
    """Coerce VLM output into our schema. Returns None on irrecoverable shape."""
    if not isinstance(obj, dict):
        return None
    tp = bool(obj.get("is_transparent_package", False))
    risk = bool(obj.get("derivative_confusion_risk", False))

    # Accept legacy single-string field `derivative_search_query` too
    queries = obj.get("derivative_search_queries")
    if queries is None:
        single = obj.get("derivative_search_query")
        queries = [single] if isinstance(single, str) and single.strip() else []
    if not isinstance(queries, list):
        return None
    queries = [q.strip() for q in queries if isinstance(q, str) and q.strip()]
    if not risk:
        queries = []  # force consistency

    reason = obj.get("reason") or obj.get("reasoning") or ""
    if not isinstance(reason, str):
        reason = str(reason)

    return {
        "is_transparent_package": tp,
        "derivative_confusion_risk": risk,
        "derivative_search_queries": queries[:3],
        "reason": reason.strip(),
    }


def profile_item(
    client,
    item_name: str,
    image_b64: str,
    model: str,
    reasoning_effort: str = "low",
    max_retries: int = 3,
) -> Tuple[Optional[dict], dict, str]:
    content = [
        {"type": "input_text", "text": PROMPT.format(item_name=item_name)},
        {"type": "input_image",
         "image_url": f"data:image/jpeg;base64,{image_b64}",
         "detail": "high"},
    ]
    for attempt in range(max_retries):
        t0 = time.time()
        try:
            response = client.responses.create(
                model=model,
                input=[{"role": "user", "content": content}],
                reasoning={"effort": reasoning_effort},
            )
            raw = response.output_text or ""
            elapsed = round(time.time() - t0, 2)
            usage = response.usage
            parsed = _normalize(_extract_json(raw) or {})
            if parsed is None:
                if attempt < max_retries - 1:
                    print(f"    parse fail (attempt {attempt+1}), retrying", flush=True)
                    time.sleep(2)
                    continue
                return None, {
                    "error": "parse_failed",
                    "inference_time_s": elapsed,
                    "input_tokens": getattr(usage, "input_tokens", None),
                    "output_tokens": getattr(usage, "output_tokens", None),
                }, raw
            stats = {
                "model": model,
                "inference_time_s": elapsed,
                "input_tokens": getattr(usage, "input_tokens", None),
                "output_tokens": getattr(usage, "output_tokens", None),
                "attempt": attempt + 1,
            }
            return parsed, stats, raw
        except Exception as e:
            err = str(e)
            transient = any(tok in err for tok in (
                "503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED",
                "500", "INTERNAL", "timeout", "Connection",
            ))
            if transient and attempt < max_retries - 1:
                wait = 4 * (2 ** attempt)
                print(f"    transient error, retry in {wait}s: {err[:80]}", flush=True)
                time.sleep(wait)
                continue
            return None, {
                "error": err[:500],
                "inference_time_s": round(time.time() - t0, 2),
            }, ""
    return None, {"error": "max retries exceeded"}, ""


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Profile inventory items for package/derivative visual confusion."
    )
    parser.add_argument("--participant", required=True)
    parser.add_argument("--model", default=GPT_MODEL_DEFAULT)
    parser.add_argument("--reasoning", default="low",
                        choices=["minimal", "low", "medium", "high"])
    parser.add_argument("--force", action="store_true",
                        help="Reprofile items even if they already have a result")
    parser.add_argument("--only", action="append", default=None,
                        help="Restrict to these instance_ids (repeatable)")
    parser.add_argument("--dry-run", action="store_true",
                        help="List items that would be processed, do nothing else")
    args = parser.parse_args()

    pdir = participant_dir(args.participant)
    ref_root = pdir / "reference_images"
    if not ref_root.exists():
        print(f"ERROR: no reference_images dir at {ref_root}")
        sys.exit(1)

    # visual_class and product_name come from ledger and (if present) receipt mapping
    try:
        ledger = load_ledger(args.participant)
    except FileNotFoundError:
        ledger = {"items": {}}
    ledger_items = ledger.get("items", {})

    mapping_path = ref_root / "receipt_image_mapping.json"
    product_names: Dict[str, str] = {}
    if mapping_path.exists():
        try:
            mdata = json.loads(mapping_path.read_text())
            for m in mdata.get("matches", []):
                product_names[m["instance_id"]] = m.get("product_name", "")
        except json.JSONDecodeError:
            pass

    # Enumerate every reference_images/<iid>/product.* on disk
    items: List[dict] = []
    for d in sorted(ref_root.iterdir()):
        if not d.is_dir():
            continue
        iid = d.name
        product_file = None
        for f in sorted(d.iterdir()):
            if f.is_file() and f.name.startswith("product."):
                product_file = f
                break
        if product_file is None:
            continue
        vc = ledger_items.get(iid, {}).get("visual_class") or iid
        items.append({
            "instance_id": iid,
            "visual_class": vc,
            "product_name": product_names.get(iid),
            "image_file": f"{iid}/{product_file.name}",
        })

    if args.only:
        keep = set(args.only)
        items = [m for m in items if m["instance_id"] in keep]
        missing = keep - {m["instance_id"] for m in items}
        if missing:
            print(f"WARN: --only ids not found on disk: {sorted(missing)}")

    out_path = pdir / "confusable_profile.json"
    existing = {}
    if out_path.exists():
        try:
            prev = json.loads(out_path.read_text())
            existing = prev.get("items", {})
        except json.JSONDecodeError:
            existing = {}

    to_process = []
    for m in items:
        iid = m["instance_id"]
        if not args.force and iid in existing and "error" not in existing[iid]:
            continue
        to_process.append(m)

    print(f"Participant:  {args.participant}")
    print(f"Mapping items: {len(items)}")
    print(f"Already profiled: {len(items) - len(to_process)}")
    print(f"To process:   {len(to_process)}")
    print(f"Output:       {out_path}")

    if args.dry_run or not to_process:
        for m in to_process:
            print(f"  - {m['instance_id']}  ({m['visual_class']})")
        if args.dry_run or not to_process:
            return

    client = make_azure_client()
    ref_root = pdir / "reference_images"

    results = dict(existing)  # carry over prior entries
    for i, m in enumerate(to_process):
        iid = m["instance_id"]
        img_rel = m["image_file"]
        img_path = ref_root / img_rel
        header = f"[{i+1}/{len(to_process)}] {iid}  ({m['visual_class']})"
        print(f"\n{header}")
        if not img_path.exists():
            print(f"  MISSING image: {img_path}")
            results[iid] = {
                "visual_class": m["visual_class"],
                "product_name": m.get("product_name"),
                "image_file": img_rel,
                "error": "image_not_found",
                "timestamp": datetime.now().isoformat(),
            }
            continue

        try:
            img = Image.open(img_path)
            b64 = image_to_jpeg_b64(img)
        except Exception as e:
            print(f"  FAILED to load image: {e}")
            results[iid] = {
                "visual_class": m["visual_class"],
                "product_name": m.get("product_name"),
                "image_file": img_rel,
                "error": f"image_load_failed: {e}"[:200],
                "timestamp": datetime.now().isoformat(),
            }
            continue

        parsed, stats, raw = profile_item(
            client,
            item_name=m["visual_class"],
            image_b64=b64,
            model=args.model,
            reasoning_effort=args.reasoning,
        )

        entry = {
            "visual_class": m["visual_class"],
            "product_name": m.get("product_name"),
            "image_file": img_rel,
            "timestamp": datetime.now().isoformat(),
            "stats": stats,
        }
        if parsed is None:
            entry["error"] = stats.get("error", "unknown")
            entry["raw_response"] = raw
            print(f"  FAILED: {entry['error']}")
        else:
            entry.update(parsed)
            risk = "YES" if parsed["derivative_confusion_risk"] else "no"
            trans = "transparent" if parsed["is_transparent_package"] else "opaque"
            qs = ", ".join(parsed["derivative_search_queries"]) or "-"
            print(f"  confusion={risk}  pkg={trans}")
            print(f"  queries: {qs}")
            if parsed.get("reason"):
                print(f"  reason:  {parsed['reason']}")

        results[iid] = entry

        # Save incrementally so a crash doesn't lose progress
        out_path.write_text(json.dumps({
            "participant": args.participant,
            "timestamp": datetime.now().isoformat(),
            "model": args.model,
            "reasoning_effort": args.reasoning,
            "prompt_version": "v1",
            "items": results,
        }, indent=2))

    # Final summary
    n_risk = sum(
        1 for e in results.values()
        if e.get("derivative_confusion_risk") is True
    )
    n_ok = sum(
        1 for e in results.values()
        if e.get("derivative_confusion_risk") is False
    )
    n_err = sum(1 for e in results.values() if "error" in e)
    print(f"\nSummary: {len(results)} items  "
          f"confusion_risk={n_risk}  safe={n_ok}  errors={n_err}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
extract_receipt_images.py - Extract product images from saved Kroger receipt HTML.

Parses saved receipt web pages for product images (identified by
aria-label="Image of ...") and maps them to ledger items by fuzzy name matching.
Copies matched images to participants/{P}/reference_images/{instance_id}/.

Usage:
    python system_design/extract_receipt_images.py --participant kailai
    python system_design/extract_receipt_images.py --participant kailai --dry-run

The script scans all *-website/ directories under receipts/ for HTML files.
"""

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

from utils import load_ledger, participant_dir


def parse_receipt_html(html_path: Path) -> List[dict]:
    """Parse product images from Kroger receipt HTML.

    Returns list of {aria_label, product_name, image_path}.
    """
    html = html_path.read_text(errors="replace")
    assets_dir = None
    # Find the companion _files directory
    for sibling in html_path.parent.iterdir():
        if sibling.is_dir() and sibling.name.endswith("_files"):
            assets_dir = sibling
            break

    if not assets_dir:
        print(f"  WARN: No _files directory found for {html_path.name}")
        return []

    # Try two patterns:
    # 1) Old Kroger: aria-label="Image of <name>" ... src="..."
    # 2) New Kroger: <img src="..." alt="<name>" ... class="citrus-Image-img">
    patterns = [
        re.compile(r'aria-label="Image of ([^"]+)"[^>]*src="([^"]+)"'),
        re.compile(r'<img\s[^>]*src="([^"]+)"[^>]*alt="([^"]+)"[^>]*class="[^"]*citrus-Image[^"]*"'),
    ]

    products = []
    seen_names = set()

    for pattern in patterns:
        for match in pattern.finditer(html):
            groups = match.groups()
            # Pattern 1: (name, src); Pattern 2: (src, name)
            if 'aria-label' in match.group(0):
                name, src = groups[0], groups[1]
            else:
                src, name = groups[0], groups[1]

            name = name.strip()
            src = src.strip()

            # Clean up HTML entities
            name = name.replace("&amp;", "&").replace("&#39;", "'")

            if name in seen_names:
                continue
            seen_names.add(name)

            # Resolve image path
            if src.startswith("./"):
                src = src[2:]
            image_path = assets_dir.parent / src
            if not image_path.exists():
                image_path = assets_dir / Path(src).name
            if not image_path.exists():
                # Don't warn here — another pattern may find it
                continue

            products.append({
                "aria_label": name,
                "product_name": name.replace("Image of ", ""),
                "image_path": image_path,
            })

    return products


def fuzzy_match_item(product_name: str, ledger_items: Dict[str, dict]) -> Tuple[str, float]:
    """Match a product name to a ledger item by substring/keyword overlap.

    Returns (best_instance_id, score) where score is 0-1.
    """
    pn_lower = product_name.lower()
    pn_words = set(re.findall(r'\w+', pn_lower))

    best_iid = None
    best_score = 0.0

    for iid, item in ledger_items.items():
        vc = item["visual_class"].lower()
        vc_words = set(re.findall(r'\w+', vc))

        # Exact substring match
        if vc in pn_lower or pn_lower in vc:
            score = 0.9
        else:
            # Word overlap (Jaccard-ish)
            overlap = len(pn_words & vc_words)
            if overlap == 0:
                continue
            score = overlap / max(len(pn_words), len(vc_words))

        # Boost if receipt_raw also matches
        receipt_raw = item.get("receipt_raw", "").lower()
        if receipt_raw and (receipt_raw in pn_lower or pn_lower in receipt_raw):
            score = max(score, 0.95)

        if score > best_score:
            best_score = score
            best_iid = iid

    return best_iid, best_score


def get_image_extension(image_path: Path) -> str:
    """Detect actual file format and return appropriate extension."""
    header = image_path.read_bytes()[:12]
    if header[:4] == b'RIFF' and header[8:12] == b'WEBP':
        return '.webp'
    if header[:3] == b'\xff\xd8\xff':
        return '.jpg'
    if header[:8] == b'\x89PNG\r\n\x1a\n':
        return '.png'
    return '.webp'  # default for Kroger


def main():
    parser = argparse.ArgumentParser(
        description="Extract product images from saved receipt web pages"
    )
    parser.add_argument("--participant", required=True)
    parser.add_argument("--min-score", type=float, default=0.3,
                        help="Minimum fuzzy match score (default: 0.3)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show matches without copying files")
    args = parser.parse_args()

    ledger = load_ledger(args.participant)
    items = ledger["items"]
    receipts_dir = participant_dir(args.participant) / "receipts"
    ref_dir = participant_dir(args.participant) / "reference_images"

    print(f"Participant: {args.participant}")
    print(f"Ledger items: {len(items)}")
    print(f"Receipts dir: {receipts_dir}")
    print()

    # Find all saved website directories
    website_dirs = sorted(receipts_dir.glob("*-website"))
    if not website_dirs:
        print("No *-website directories found in receipts/")
        return

    # Build per-date item sets from purchase events
    events = ledger.get("events", [])
    date_items: Dict[str, Dict[str, dict]] = {}
    for ev in events:
        if ev.get("type") == "purchase":
            date = ev["time"][:8]  # "20260310-000000" -> "20260310"
            if date not in date_items:
                date_items[date] = {}
            iid = ev["item"]
            if iid in items:
                date_items[date][iid] = items[iid]

    all_products = []
    matches = []
    unmatched = []

    for wd in website_dirs:
        # Extract date from directory name (e.g. "20260310-2-website" -> "20260310")
        receipt_date = re.match(r'(\d{8})', wd.name)
        if not receipt_date:
            print(f"WARN: Cannot extract date from {wd.name}, skipping")
            continue
        receipt_date = receipt_date.group(1)

        # Get items purchased on this date
        candidate_items = date_items.get(receipt_date, {})
        if not candidate_items:
            print(f"WARN: No purchase events on {receipt_date} for {wd.name}")

        html_files = list(wd.glob("*.html"))
        for html_f in html_files:
            print(f"Parsing: {html_f.name} (receipt date: {receipt_date}, {len(candidate_items)} candidate items)")
            products = parse_receipt_html(html_f)
            print(f"  Found {len(products)} product images")
            all_products.extend(products)

            # Match only against items purchased on this receipt's date
            match_pool = candidate_items if candidate_items else items
            for prod in products:
                iid, score = fuzzy_match_item(prod["product_name"], match_pool)
                if iid and score >= args.min_score:
                    vc = match_pool[iid]["visual_class"]
                    print(f"  {prod['product_name']:<53} {vc:<33} {score:.2f}")
                    matches.append((prod, iid, score))
                else:
                    print(f"  {prod['product_name']:<53} {'--- NO MATCH ---':<33} {score:.2f}")
                    unmatched.append(prod)

    if not all_products:
        print("No product images found")
        return

    print(f"\nMatched: {len(matches)}/{len(all_products)}")
    if unmatched:
        print(f"Unmatched: {[u['product_name'] for u in unmatched]}")

    if args.dry_run:
        print("\n[DRY RUN] No files copied")
        return

    # Copy images to reference_images/
    print(f"\nCopying to {ref_dir}/")
    for prod, iid, score in matches:
        item_dir = ref_dir / iid
        item_dir.mkdir(parents=True, exist_ok=True)

        ext = get_image_extension(prod["image_path"])
        dest = item_dir / f"product{ext}"

        shutil.copy2(prod["image_path"], dest)
        print(f"  {iid}: {dest.name} ({prod['image_path'].stat().st_size // 1024}KB)")

    # Save mapping metadata
    mapping = {
        "source": "kroger_receipt_html",
        "participant": args.participant,
        "matches": [
            {
                "instance_id": iid,
                "visual_class": items[iid]["visual_class"],
                "product_name": prod["product_name"],
                "score": round(score, 3),
                "image_file": f"{iid}/product{get_image_extension(prod['image_path'])}",
            }
            for prod, iid, score in matches
        ],
    }
    mapping_path = ref_dir / "receipt_image_mapping.json"
    mapping_path.write_text(json.dumps(mapping, indent=2) + "\n")
    print(f"\nMapping saved: {mapping_path}")


if __name__ == "__main__":
    main()

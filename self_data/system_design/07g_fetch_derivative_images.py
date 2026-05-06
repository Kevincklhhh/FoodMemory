#!/usr/bin/env python3
"""Step 2 of the DINO package-vs-derivative filter pipeline.

For every item flagged `derivative_confusion_risk=true` by 07f, run each of
its `derivative_search_queries` through an image-search backend and download
the top-N results into a shared query cache. Per-item manifests reference
cached files so identical queries across different items only fetch once.

Layout:
    participants/{P}/derivative_anchors/
        ├── _query_cache/
        │   ├── <sanitized_query>/
        │   │   ├── manifest.json    # query, URLs tried, files saved/rejected
        │   │   ├── 0.jpg
        │   │   └── ...
        │   └── ...
        ├── <instance_id>.json       # per-item manifest (queries + file refs)
        └── FETCH_SUMMARY.json

Input : participants/{P}/confusable_profile.json   (from 07f)

Backends:
  - ddgs        (default, no API key, occasional scraping breakage)
  - google_cse  (needs GOOGLE_CSE_API_KEY + GOOGLE_CSE_CX in env)

Usage:
    python system_design/07g_fetch_derivative_images.py --participant kailai
    python system_design/07g_fetch_derivative_images.py --participant kailai \
        --only fresh_blueberries_20260328 --per-query 8
    python system_design/07g_fetch_derivative_images.py --participant kailai \
        --engine google_cse
    python system_design/07g_fetch_derivative_images.py --participant kailai --dry-run
"""

import argparse
import io
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv
from PIL import Image, UnidentifiedImageError

sys.path.insert(0, str(Path(__file__).parent))
from utils import participant_dir  # noqa: E402

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

# ── Defaults ───────────────────────────────────────────────────────────────

PER_QUERY_DEFAULT = 8
MIN_SHORT_SIDE = 200       # reject images whose shortest side is < this (px)
MIN_BYTES = 20_000         # reject <20 KB responses (likely thumbnails / 1×1)
REQUEST_TIMEOUT_S = 10
DOWNLOAD_MAX_SIDE = 1024   # downscale very large images on save
JPEG_QUALITY = 92
INTER_QUERY_SLEEP_S = 1.0
HTTP_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)
ENGINES = ("ddgs", "google_cse")
CACHE_DIRNAME = "_query_cache"


# ── Query normalization ────────────────────────────────────────────────────

_WS_RE = re.compile(r"\s+")
_SANITIZE_RE = re.compile(r"[^a-z0-9]+")


def normalize_query(q: str) -> str:
    """Lowercase + collapse whitespace. Used as the cache key."""
    return _WS_RE.sub(" ", q.strip().lower())


def sanitize(s: str, max_len: int = 80) -> str:
    """File-safe slug for a normalized query."""
    out = _SANITIZE_RE.sub("_", s.lower()).strip("_")
    return out[:max_len] or "q"


# ── Backends ───────────────────────────────────────────────────────────────

def _search_ddgs(query: str, n: int) -> List[dict]:
    try:
        from ddgs import DDGS  # current package name
    except ImportError:
        try:
            from duckduckgo_search import DDGS  # legacy name, still works
        except ImportError:
            raise RuntimeError("ddgs not installed. Run: pip install ddgs")
    try:
        with DDGS() as ddg:
            raw = list(ddg.images(
                query=query, region="wt-wt",
                safesearch="moderate", max_results=n,
            ))
    except Exception as e:
        print(f"      ddgs error: {e}")
        return []
    out: List[dict] = []
    for r in raw:
        img_url = r.get("image") or r.get("url")
        if not img_url:
            continue
        out.append({
            "image_url": img_url,
            "thumbnail": r.get("thumbnail"),
            "source_url": r.get("url"),
            "width": r.get("width"),
            "height": r.get("height"),
            "title": r.get("title"),
        })
    return out


def _search_google_cse(query: str, n: int) -> List[dict]:
    api_key = os.getenv("GOOGLE_CSE_API_KEY")
    cx = os.getenv("GOOGLE_CSE_CX")
    if not api_key or not cx:
        raise RuntimeError(
            "google_cse backend requires GOOGLE_CSE_API_KEY + GOOGLE_CSE_CX in env"
        )
    n = min(max(n, 1), 10)  # Google CSE max = 10 per call
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": api_key, "cx": cx, "q": query,
        "searchType": "image", "safe": "active", "num": n,
    }
    try:
        r = requests.get(url, params=params, timeout=REQUEST_TIMEOUT_S)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"      google_cse error: {e}")
        return []
    out: List[dict] = []
    for item in data.get("items", []):
        img_url = item.get("link")
        if not img_url:
            continue
        meta = item.get("image") or {}
        out.append({
            "image_url": img_url,
            "thumbnail": meta.get("thumbnailLink"),
            "source_url": meta.get("contextLink"),
            "width": meta.get("width"),
            "height": meta.get("height"),
            "title": item.get("title"),
        })
    return out


def search(engine: str, query: str, n: int) -> List[dict]:
    if engine == "ddgs":
        return _search_ddgs(query, n)
    if engine == "google_cse":
        return _search_google_cse(query, n)
    raise ValueError(f"unknown engine: {engine}")


# ── Download + validation ──────────────────────────────────────────────────

def _safe_get(url: str) -> Optional[bytes]:
    try:
        r = requests.get(
            url, timeout=REQUEST_TIMEOUT_S,
            headers={"User-Agent": HTTP_USER_AGENT},
            allow_redirects=True,
        )
    except requests.exceptions.RequestException:
        return None
    if r.status_code != 200:
        return None
    ctype = (r.headers.get("Content-Type") or "").lower()
    if not ctype.startswith("image/") \
       and "octet-stream" not in ctype and "binary" not in ctype:
        return None
    if len(r.content) < MIN_BYTES:
        return None
    return r.content


def _decode_and_save(
    blob: bytes, out_path: Path,
) -> Optional[Tuple[int, int, int]]:
    try:
        img = Image.open(io.BytesIO(blob))
        img.load()
    except (UnidentifiedImageError, OSError):
        return None
    w, h = img.size
    if min(w, h) < MIN_SHORT_SIDE:
        return None
    if img.mode != "RGB":
        img = img.convert("RGB")
    m = max(w, h)
    if m > DOWNLOAD_MAX_SIDE:
        s = DOWNLOAD_MAX_SIDE / m
        img = img.resize((int(w * s), int(h * s)), Image.LANCZOS)
        w, h = img.size
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, "JPEG", quality=JPEG_QUALITY)
    return (w, h, out_path.stat().st_size)


# ── Shared query cache ─────────────────────────────────────────────────────

def ensure_query_cache(
    query: str,
    cache_root: Path,
    engine: str,
    per_query: int,
    force: bool,
    mem_cache: Dict[str, dict],
) -> dict:
    """Fetch-and-cache one query. Returns the per-query manifest dict.

    On-disk layout: cache_root/<sanitized(normalized)>/{manifest.json, *.jpg}
    `mem_cache` is a same-process dedup (keyed by normalized query).
    """
    norm = normalize_query(query)
    if norm in mem_cache:
        return mem_cache[norm]

    dname = sanitize(norm)
    qdir = cache_root / dname
    manifest_path = qdir / "manifest.json"

    # Hit on-disk cache
    if manifest_path.exists() and not force:
        try:
            m = json.loads(manifest_path.read_text())
            files_on_disk = {p.name for p in qdir.glob("*.jpg")}
            saved = [s for s in m.get("saved", []) if s.get("file") in files_on_disk]
            if saved:
                m["saved"] = saved
                m["cache_hit"] = True
                mem_cache[norm] = m
                return m
        except json.JSONDecodeError:
            pass

    # Fresh fetch
    qdir.mkdir(parents=True, exist_ok=True)
    print(f"      search: '{query}' (engine={engine}, n={per_query})", flush=True)
    t0 = time.time()
    hits = search(engine, query, per_query)
    print(f"        got {len(hits)} hit(s) in {time.time()-t0:.1f}s")

    saved: List[dict] = []
    rejected: List[dict] = []
    for hit in hits:
        if len(saved) >= per_query:
            break
        url = hit["image_url"]
        t1 = time.time()
        blob = _safe_get(url)
        if blob is None:
            rejected.append({"url": url, "reason": "download_failed"})
            continue
        out_path = qdir / f"{len(saved)}.jpg"
        info = _decode_and_save(blob, out_path)
        if info is None:
            rejected.append({"url": url, "reason": "decode_or_too_small"})
            try:
                out_path.unlink(missing_ok=True)
            except Exception:
                pass
            continue
        w, h, nbytes = info
        saved.append({
            "file": out_path.name,
            "url": url,
            "source_url": hit.get("source_url"),
            "title": hit.get("title"),
            "width": w, "height": h, "bytes": nbytes,
            "fetch_time_s": round(time.time() - t1, 2),
        })
    print(f"        saved={len(saved)}  rejected={len(rejected)}")

    manifest = {
        "query": query,
        "normalized_query": norm,
        "dir": dname,
        "engine": engine,
        "per_query": per_query,
        "timestamp": datetime.now().isoformat(),
        "hits": len(hits),
        "saved": saved,
        "rejected": rejected,
        "cache_hit": False,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    mem_cache[norm] = manifest

    time.sleep(INTER_QUERY_SLEEP_S)
    return manifest


# ── Per-item fetch ─────────────────────────────────────────────────────────

def fetch_item(
    iid: str,
    visual_class: str,
    queries: List[str],
    anchors_root: Path,
    cache_root: Path,
    engine: str,
    per_query: int,
    force: bool,
    mem_cache: Dict[str, dict],
) -> dict:
    item_manifest = {
        "instance_id": iid,
        "visual_class": visual_class,
        "engine": engine,
        "per_query": per_query,
        "timestamp": datetime.now().isoformat(),
        "queries": [],
    }
    for qi, query in enumerate(queries):
        print(f"    query[{qi}] '{query}'", flush=True)
        qm = ensure_query_cache(
            query=query, cache_root=cache_root, engine=engine,
            per_query=per_query, force=force, mem_cache=mem_cache,
        )
        files = [
            str(Path(CACHE_DIRNAME) / qm["dir"] / s["file"])
            for s in qm.get("saved", [])
        ]
        item_manifest["queries"].append({
            "query": query,
            "cache_dir": qm["dir"],
            "cache_hit": qm.get("cache_hit", False),
            "n_saved": len(files),
            "files": files,
        })
    total = sum(q["n_saved"] for q in item_manifest["queries"])
    item_manifest["total_images"] = total

    out_path = anchors_root / f"{iid}.json"
    out_path.write_text(json.dumps(item_manifest, indent=2))
    print(f"    → total {total} image(s); manifest: {out_path.name}")
    return item_manifest


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Fetch derivative reference images for confusable items."
    )
    parser.add_argument("--participant", required=True)
    parser.add_argument("--engine", default="ddgs", choices=ENGINES)
    parser.add_argument("--per-query", type=int, default=PER_QUERY_DEFAULT,
                        help=f"Max images to save per query (default {PER_QUERY_DEFAULT})")
    parser.add_argument("--only", action="append", default=None,
                        help="Restrict to these instance_ids (repeatable)")
    parser.add_argument("--force", action="store_true",
                        help="Re-fetch even if the query is already cached")
    parser.add_argument("--dry-run", action="store_true",
                        help="List what would be fetched and exit")
    args = parser.parse_args()

    pdir = participant_dir(args.participant)
    profile_path = pdir / "confusable_profile.json"
    if not profile_path.exists():
        print(f"ERROR: {profile_path} missing — run 07f first")
        sys.exit(1)
    profile = json.loads(profile_path.read_text())
    items = profile.get("items", {})

    targets: List[Tuple[str, dict, List[str]]] = []
    for iid, entry in items.items():
        if not entry.get("derivative_confusion_risk"):
            continue
        qs = entry.get("derivative_search_queries") or []
        if not qs:
            continue
        if args.only and iid not in args.only:
            continue
        targets.append((iid, entry, qs))
    targets.sort(key=lambda t: t[0])

    anchors_root = pdir / "derivative_anchors"
    cache_root = anchors_root / CACHE_DIRNAME
    anchors_root.mkdir(parents=True, exist_ok=True)

    # Count unique queries for plan visibility
    all_queries = [q for _, _, qs in targets for q in qs]
    unique_norm = {normalize_query(q) for q in all_queries}
    print(f"Participant:    {args.participant}")
    print(f"Engine:         {args.engine}")
    print(f"Per-query cap:  {args.per_query}")
    print(f"Items:          {len(targets)} confusable")
    print(f"Query instances:{len(all_queries)}   unique(normalized): {len(unique_norm)}")
    print(f"Anchors dir:    {anchors_root}")

    if args.dry_run:
        print("\nDRY RUN — plan:")
        for iid, entry, qs in targets:
            print(f"\n  {iid}  ({entry.get('visual_class')})")
            for q in qs:
                qdir = cache_root / sanitize(normalize_query(q))
                cached = (qdir / "manifest.json").exists()
                tag = "[cached]" if cached and not args.force else "[fetch]"
                print(f"    {tag} {q}")
        return

    summary = {
        "participant": args.participant,
        "engine": args.engine,
        "per_query": args.per_query,
        "timestamp": datetime.now().isoformat(),
        "items": {},
    }
    mem_cache: Dict[str, dict] = {}
    for i, (iid, entry, qs) in enumerate(targets):
        print(f"\n[{i+1}/{len(targets)}] {iid}  ({entry.get('visual_class')})")
        try:
            m = fetch_item(
                iid=iid, visual_class=entry.get("visual_class", iid),
                queries=qs, anchors_root=anchors_root, cache_root=cache_root,
                engine=args.engine, per_query=args.per_query,
                force=args.force, mem_cache=mem_cache,
            )
        except Exception as e:
            print(f"  FAILED: {e}")
            summary["items"][iid] = {"error": str(e)[:300]}
            continue
        summary["items"][iid] = {
            "total_images": m["total_images"],
            "queries": [
                {"query": q["query"], "n_saved": q["n_saved"],
                 "cache_hit": q["cache_hit"]}
                for q in m["queries"]
            ],
        }
        (anchors_root / "FETCH_SUMMARY.json").write_text(
            json.dumps(summary, indent=2)
        )

    total_imgs = sum(v.get("total_images", 0) for v in summary["items"].values())
    n_fail = sum(1 for v in summary["items"].values() if "error" in v)
    n_cached_queries = sum(
        1 for v in summary["items"].values()
        for q in v.get("queries", []) if q.get("cache_hit")
    )
    print(f"\nDone. {len(targets)} items, {total_imgs} total images, "
          f"{n_cached_queries} cache hits, {n_fail} failures")
    print(f"Summary: {anchors_root / 'FETCH_SUMMARY.json'}")


if __name__ == "__main__":
    main()

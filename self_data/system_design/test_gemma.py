#!/usr/bin/env python3
"""Quick test: does Gemma-4-31B-IT on vLLM handle video input, frame input, or both?"""

import base64
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import requests

GEMMA_URL = "http://saltyfish.eecs.umich.edu:8000/v1/chat/completions"
GEMMA_MODEL = "google/gemma-4-31B-it"

# Use a short clip from first session
SAMPLE_VIDEO = Path(__file__).resolve().parent.parent / "participants/kailai/videos/20260310-195710/20260310-195710.mp4"

PROMPT = "Describe what you see in this image/video in one sentence."


def extract_frame(video_path: Path, t: float = 5.0) -> str:
    """Extract a single frame at time t as base64 JPEG."""
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = tmp.name
    subprocess.run(
        ["ffmpeg", "-y", "-ss", str(t), "-i", str(video_path),
         "-frames:v", "1", "-q:v", "2", tmp_path],
        capture_output=True,
    )
    with open(tmp_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    Path(tmp_path).unlink(missing_ok=True)
    return b64


def extract_short_clip(video_path: Path, start: float = 5.0, dur: float = 3.0) -> str:
    """Extract a short clip as base64 mp4."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = tmp.name
    subprocess.run(
        ["ffmpeg", "-y", "-ss", str(start), "-i", str(video_path),
         "-t", str(dur), "-c:v", "libx264", "-an", tmp_path],
        capture_output=True,
    )
    with open(tmp_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    Path(tmp_path).unlink(missing_ok=True)
    return b64


def check_server():
    """Check what models are available on the server."""
    try:
        resp = requests.get("http://saltyfish.eecs.umich.edu:8000/v1/models", timeout=10)
        resp.raise_for_status()
        models = resp.json()
        print("Available models on server:")
        for m in models.get("data", []):
            print(f"  - {m['id']}")
        return True
    except Exception as e:
        print(f"Server check failed: {e}")
        return False


def test_text_only():
    """Test 1: text-only request to confirm server is responsive."""
    print("\n=== Test 1: Text-only ===")
    t0 = time.time()
    try:
        resp = requests.post(GEMMA_URL, json={
            "model": GEMMA_MODEL,
            "messages": [{"role": "user", "content": "Hello, what model are you?"}],
            "max_tokens": 128,
            "temperature": 0.3,
        }, timeout=60)
        resp.raise_for_status()
        result = resp.json()
        text = result["choices"][0]["message"]["content"]
        usage = result.get("usage", {})
        print(f"  OK ({time.time()-t0:.1f}s) | tokens: {usage.get('prompt_tokens',0)}→{usage.get('completion_tokens',0)}")
        print(f"  Response: {text[:200]}")
        return True
    except Exception as e:
        print(f"  FAIL: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"  Body: {e.response.text[:500]}")
        return False


def test_single_frame():
    """Test 2: single image (base64 JPEG) — OpenAI image_url format."""
    print("\n=== Test 2: Single frame (image_url base64) ===")
    frame_b64 = extract_frame(SAMPLE_VIDEO)
    print(f"  Frame size: {len(frame_b64)//1024} KB base64")

    t0 = time.time()
    try:
        resp = requests.post(GEMMA_URL, json={
            "model": GEMMA_MODEL,
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}"}},
                {"type": "text", "text": PROMPT},
            ]}],
            "max_tokens": 256,
            "temperature": 0.3,
        }, timeout=120)
        resp.raise_for_status()
        result = resp.json()
        text = result["choices"][0]["message"]["content"]
        usage = result.get("usage", {})
        print(f"  OK ({time.time()-t0:.1f}s) | tokens: {usage.get('prompt_tokens',0)}→{usage.get('completion_tokens',0)}")
        print(f"  Response: {text[:300]}")
        return True
    except Exception as e:
        print(f"  FAIL: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"  Body: {e.response.text[:500]}")
        return False


def test_multi_frame():
    """Test 3: multiple frames as separate image_url entries."""
    print("\n=== Test 3: Multi-frame (3 images) ===")
    frames = [extract_frame(SAMPLE_VIDEO, t) for t in [3.0, 6.0, 9.0]]
    total_kb = sum(len(f) for f in frames) // 1024
    print(f"  3 frames, total {total_kb} KB base64")

    content = []
    for i, fb64 in enumerate(frames):
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{fb64}"}})
    content.append({"type": "text", "text": "Describe what happens across these 3 frames in one sentence."})

    t0 = time.time()
    try:
        resp = requests.post(GEMMA_URL, json={
            "model": GEMMA_MODEL,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 256,
            "temperature": 0.3,
        }, timeout=180)
        resp.raise_for_status()
        result = resp.json()
        text = result["choices"][0]["message"]["content"]
        usage = result.get("usage", {})
        print(f"  OK ({time.time()-t0:.1f}s) | tokens: {usage.get('prompt_tokens',0)}→{usage.get('completion_tokens',0)}")
        print(f"  Response: {text[:300]}")
        return True
    except Exception as e:
        print(f"  FAIL: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"  Body: {e.response.text[:500]}")
        return False


def test_video_native():
    """Test 4: native video input (Qwen-style video_url with base64 mp4)."""
    print("\n=== Test 4: Native video (video_url base64 mp4) ===")
    video_b64 = extract_short_clip(SAMPLE_VIDEO, start=3.0, dur=5.0)
    print(f"  Clip size: {len(video_b64)//1024} KB base64")

    t0 = time.time()
    try:
        resp = requests.post(GEMMA_URL, json={
            "model": GEMMA_MODEL,
            "messages": [{"role": "user", "content": [
                {"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{video_b64}"}},
                {"type": "text", "text": PROMPT},
            ]}],
            "max_tokens": 256,
            "temperature": 0.3,
        }, timeout=180)
        resp.raise_for_status()
        result = resp.json()
        text = result["choices"][0]["message"]["content"]
        usage = result.get("usage", {})
        print(f"  OK ({time.time()-t0:.1f}s) | tokens: {usage.get('prompt_tokens',0)}→{usage.get('completion_tokens',0)}")
        print(f"  Response: {text[:300]}")
        return True
    except Exception as e:
        print(f"  FAIL: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"  Body: {e.response.text[:500]}")
        return False


if __name__ == "__main__":
    print(f"Server: {GEMMA_URL}")
    print(f"Model:  {GEMMA_MODEL}")
    print(f"Video:  {SAMPLE_VIDEO}")

    if not SAMPLE_VIDEO.exists():
        print(f"ERROR: sample video not found at {SAMPLE_VIDEO}")
        sys.exit(1)

    if not check_server():
        sys.exit(1)

    results = {}
    results["text_only"] = test_text_only()
    results["single_frame"] = test_single_frame()
    results["multi_frame"] = test_multi_frame()
    results["video_native"] = test_video_native()

    print("\n=== Summary ===")
    for test, ok in results.items():
        print(f"  {test}: {'PASS' if ok else 'FAIL'}")

    # Recommendation
    if results["video_native"]:
        print("\n→ Gemma handles native video. Can use run_observer_qwen() as-is.")
    elif results["single_frame"] or results["multi_frame"]:
        print("\n→ Gemma handles frames only (not native video).")
        print("  Need a frame-based observer path (like GPT) instead of video_url.")
    else:
        print("\n→ Gemma doesn't appear to handle visual input on this server.")

"""Probe whether `gemini-3.1-pro-preview` is accepted by the Gemini batch API.

Submits a tiny one-line text-only batch job. Polls briefly. Reports the result.
Does NOT touch any video files or run any inference of consequence.

Usage: python test_batch_gemini31.py [--model gemini-3.1-pro-preview]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gemini-3.1-pro-preview",
                        help="Model id to probe (default: gemini-3.1-pro-preview)")
    parser.add_argument("--poll-seconds", type=int, default=120,
                        help="How long to poll the batch state (default: 120s)")
    args = parser.parse_args()

    # Load .env so GOOGLE_API_KEY is available without exporting it shell-side.
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

    if not os.getenv("GOOGLE_API_KEY"):
        print("ERROR: GOOGLE_API_KEY not set", file=sys.stderr)
        return 2

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

    # Step A: confirm the model exists at all (cheap, non-batch).
    try:
        m = client.models.get(model=f"models/{args.model}")
        print(f"[generate API] model is reachable: {m.name}")
    except Exception as e:
        print(f"[generate API] FAILED to reach model {args.model}: {e}")
        # Continue — batch may still accept it even if .get() doesn't list it.

    # Step B: submit a minimal batch job.
    batch_dir = Path("/tmp") / f"_probe_{args.model.replace('/', '_')}"
    batch_dir.mkdir(parents=True, exist_ok=True)
    jsonl = batch_dir / "probe.jsonl"
    jsonl.write_text(json.dumps({
        "key": "probe1",
        "request": {
            "contents": [{"role": "user", "parts": [{"text": "Reply with the single word: ok"}]}],
            "generation_config": {"temperature": 0.0, "max_output_tokens": 8},
        },
    }) + "\n")

    try:
        uploaded = client.files.upload(
            file=str(jsonl),
            config=types.UploadFileConfig(
                display_name=jsonl.stem,
                mime_type="application/jsonl",
            ),
        )
        print(f"[batch API] JSONL uploaded: {uploaded.name}")
    except Exception as e:
        print(f"[batch API] file upload FAILED: {e}")
        return 1

    try:
        job = client.batches.create(
            model=args.model,
            src=uploaded.name,
            config={"display_name": "probe_batch_avail"},
        )
        print(f"[batch API] batch created: {job.name}")
        print(f"[batch API] initial state: {job.state}")
    except Exception as e:
        print(f"[batch API] batches.create FAILED: {e}")
        return 1

    # Step C: poll briefly. We don't need it to fully succeed — going past PENDING
    # without an error already proves the model id is accepted.
    deadline = time.time() + args.poll_seconds
    last_state = None
    while time.time() < deadline:
        b = client.batches.get(name=job.name)
        s = str(b.state)
        if s != last_state:
            print(f"  state -> {s}")
            last_state = s
        if any(tok in s for tok in ("SUCCEEDED", "FAILED", "CANCELLED")):
            break
        time.sleep(10)

    final = client.batches.get(name=job.name)
    print(f"[batch API] final state after {args.poll_seconds}s: {final.state}")

    if "FAILED" in str(final.state):
        # Surface the failure detail if the SDK exposes one.
        err = getattr(final, "error", None)
        print(f"[batch API] error detail: {err}")
        return 1

    if "SUCCEEDED" in str(final.state):
        # Read the output to confirm we actually got a response.
        try:
            content_bytes = client.files.download(file=final.dest.file_name)
            for line in content_bytes.decode("utf-8").splitlines():
                if line.strip():
                    entry = json.loads(line)
                    cands = entry.get("response", {}).get("candidates", [])
                    text = "".join(p.get("text", "") for c in cands for p in c.get("content", {}).get("parts", []))
                    print(f"[batch API] response text: {text!r}")
        except Exception as e:
            print(f"[batch API] result download FAILED: {e}")

    print("[batch API] PROBE COMPLETE — model is batch-accepted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

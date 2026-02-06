#!/usr/bin/env python3
"""
test_gemini_batch.py - Gemini Batch API for VLM inventory pipeline

Builds JSONL batch requests from timeline items, uploads video clips
to the Gemini Files API, and submits batch jobs.

Usage:
    # Build + submit for all participants, LOW difficulty, hybrid prompt
    python test_gemini_batch.py --build --submit --all --difficulty LOW \
        --prompt hybrid --tag hybrid_gemini3_batch \
        --model models/gemini-3-flash-preview

    # Single participant
    python test_gemini_batch.py --build --submit --participant P01 --test 3

    # Check status / retrieve results
    python test_gemini_batch.py --status --job-name <batch_job_name>
    python test_gemini_batch.py --status --job-name <batch_job_name> --wait
"""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env", override=True)
load_dotenv(Path(__file__).parent.parent.parent / ".env", override=True)

# Paths
_SCRIPT_DIR = Path(__file__).parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
VIDEO_BASE_DIR = _PROJECT_ROOT / "data" / "HD-EPIC" / "Videos"
OUTPUT_DIR = _PROJECT_ROOT / "outputs" / "02_inventory"

DEFAULT_PADDING = 2.0
GEMINI_MODEL = "models/gemini-3-flash-preview"

SYSTEM_PROMPT = "You are a Visual Inventory Auditor analyzing cooking videos."

# ── Prompts ──────────────────────────────────────────────────

PROMPTS = {}

PROMPTS['hybrid'] = """You are a Visual Inventory Auditor for kitchen tasks.
Your goal is to meticulously track the flow of the **Target Food Item** by triangulating evidence from three visual paths (Source, Destination, Transfer) and then synthesizing a final count.

**INPUT:**
- Target Item: "{item_name}"
- Input Data: Video clip of a dispensing action.

**VISUAL REASONING STEPS (Perform this "Triangulation" mentally):**
1. **Analyze Path A (Source Subtraction):** Look at the Source container *Before* and *After*. Can you see a count difference? (e.g., 6 eggs -> 4 eggs).
2. **Analyze Path B (Destination Addition):** Look at the Destination container *Before* and *After*. Can you see new items added?
3. **Analyze Path C (Transfer Observation):** Look at the item while it is *in-transit* (in the hand/scoop). Can you count the items in the air?
4. **Synthesize:** Compare the three paths. If Source is opaque, trust Transfer. If Transfer is blurry, trust Destination.

**OUTPUT SCHEMA (Strict JSON):**
{{
  "item_name": "{item_name}",
  "quantity_category": "discrete" | "continuous" | "unknown",

  // --- PART 1: EVIDENCE LOGS (Frame Localization) ---
  // Identify the specific timestamps you used for reasoning.
  "path_source": {{
    "status": "VALID" | "INVALID_OCCLUDED" | "INVALID_OPAQUE",
    "container_description": "e.g., White egg carton",
    "timestamp_before": "MM:SS",
    "timestamp_after": "MM:SS",
    "observed_delta": <number or null>,
    "confidence": "high" | "medium" | "low"
  }},

  "path_destination": {{
    "status": "VALID" | "INVALID_OCCLUDED" | "INVALID_PILE",
    "container_description": "e.g., Black frying pan",
    "timestamp_before": "MM:SS",
    "timestamp_after": "MM:SS",
    "observed_delta": <number or null>,
    "confidence": "high" | "medium" | "low"
  }},

  "path_transfer": {{
    "status": "VALID" | "INVALID_BLURRY" | "INVALID_HIDDEN",
    "transfer_events": [
      {{ "timestamp": "MM:SS", "description": "Scoop 1 mid-air", "visible_count": 1 }}
    ],
    "total_transfer_count": <number or null>,
    "confidence": "high" | "medium" | "low"
  }},

  // --- PART 2: FINAL ESTIMATION ---
  // Synthesize the valid paths above into a final reliable number.

  // AMOUNT REMOVED:
  "numeric_count": <integer or null>,
  "amount_description": <string or null>,
  "volume_fraction": <float or null>,

  // AMOUNT REMAINING:
  "remaining_count": <integer or null>,
  "remaining_description": <string or null>,
  "remaining_fraction": <float or null>,

  "unit_type": "unit" | "scoop" | "cup" | "splash" | "pinch" | "slice",
  "confidence": "high" | "medium" | "low",

  "visual_evidence": "Source was [Status]. Dest was [Status]. Transfer was [Status]. Final count derived primarily from Path [A/B/C]."
}}

**EXAMPLES:**

*Example 1 (Discrete - Path B Destination Used):*
{{
  "item_name": "eggs",
  "quantity_category": "discrete",
  "path_source": {{
    "status": "INVALID_OCCLUDED", "container_description": "Egg carton",
    "timestamp_before": "00:02", "timestamp_after": "00:08",
    "observed_delta": null, "confidence": "low"
  }},
  "path_destination": {{
    "status": "VALID", "container_description": "Black frying pan",
    "timestamp_before": "00:03", "timestamp_after": "00:09",
    "observed_delta": 2, "confidence": "high"
  }},
  "path_transfer": {{
    "status": "INVALID_HIDDEN", "transfer_events": [],
    "total_transfer_count": null, "confidence": "low"
  }},
  "numeric_count": 2,
  "amount_description": null,
  "volume_fraction": null,
  "remaining_count": 4,
  "remaining_description": null,
  "remaining_fraction": null,
  "unit_type": "unit",
  "confidence": "high",
  "visual_evidence": "Source blocked by hand. Transfer hidden in grip. Destination clearly showed 2 yolks added. Used Path B."
}}

Return ONLY the raw JSON string. Do not use Markdown."""

PROMPTS['hybrid_no_transfer'] = """You are a Visual Inventory Auditor for kitchen tasks.
Your goal is to meticulously track the flow of the **Target Food Item** by comparing the "Before" and "After" states of the Source and Destination containers.

**INPUT:**
- Target Item: "{item_name}"
- Input Data: Video clip of a dispensing session (may contain multiple actions).

**VISUAL REASONING STEPS (Follow strictly):**
1. **Identify Anchors:** Scan the video to identify the **Source Container** (where items come from) and the **Destination** (where items go).
2. **Scan for "Global" Timeline:**
   - **Start State:** Find the clearest stable frame of the Source/Destination *before* the first interaction begins.
   - **End State:** Fast-forward to the **VERY END** of the interaction sequence.
3. **Select Best Views (The 4 Frames):**
   - Find the best **Source Before** and **Source After** frames.
   - Find the best **Dest Before** and **Dest After** frames.
   - *Criteria:* Frames must be stable (no motion blur) and unoccluded by hands.
4. **Calculate Deltas:**
   - Path A (Source): `Count_Start - Count_End`
   - Path B (Destination): `Count_End - Count_Start`

**OUTPUT SCHEMA (Strict JSON):**
{{
  "item_name": "{item_name}",
  "quantity_category": "discrete" | "continuous" | "unknown",

  "path_source": {{
    "status": "VALID" | "INVALID_OCCLUDED" | "INVALID_OPAQUE",
    "container_description": "e.g., White egg carton",
    "timestamp_before": "MM:SS",
    "timestamp_after": "MM:SS",
    "observed_delta": <number or null>,
    "confidence": "high" | "medium" | "low"
  }},

  "path_destination": {{
    "status": "VALID" | "INVALID_OCCLUDED" | "INVALID_PILE",
    "container_description": "e.g., Black frying pan",
    "timestamp_before": "MM:SS",
    "timestamp_after": "MM:SS",
    "observed_delta": <number or null>,
    "confidence": "high" | "medium" | "low"
  }},

  "numeric_count": <integer or null>,
  "amount_description": <string or null>,
  "volume_fraction": <float or null>,

  "remaining_count": <integer or null>,
  "remaining_description": <string or null>,
  "remaining_fraction": <float or null>,

  "unit_type": "unit" | "scoop" | "cup" | "splash" | "pinch" | "slice",
  "confidence": "high" | "medium" | "low",

  "visual_evidence": "Source [Status]: Start(6) -> End(4). Dest [Status]: Start(0) -> End(2). Trusting [Source/Dest] for final count."
}}

Return ONLY the raw JSON string. Do not use Markdown."""


# ── Utilities ────────────────────────────────────────────────

def compute_clip_start(segment_start: float, padding: float = DEFAULT_PADDING) -> float:
    return max(0, segment_start - padding)


def extract_clip(video_path: Path, start_ts: float, end_ts: float,
                 output_path: Path, padding: float = DEFAULT_PADDING) -> tuple[bool, float]:
    """Extract video clip with ffmpeg. Returns (success, clip_start)."""
    clip_start = compute_clip_start(start_ts, padding)
    duration = (end_ts - start_ts) + (2 * padding)
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(clip_start),
        "-i", str(video_path),
        "-t", str(duration),
        "-c:v", "libx264", "-preset", "fast", "-crf", "23", "-an",
        str(output_path)
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        return result.returncode == 0 and output_path.exists(), clip_start
    except Exception as e:
        print(f"  ERROR extracting clip: {e}")
        return False, clip_start


def upload_video(client, video_path: Path):
    """Upload video to Gemini Files API and wait until ready."""
    print(f"  Uploading {video_path.name}...", end=" ", flush=True)
    video_file = client.files.upload(file=video_path)
    while video_file.state == "PROCESSING":
        time.sleep(1)
        video_file = client.files.get(name=video_file.name)
    if video_file.state == "FAILED":
        print("FAILED")
        raise ValueError(f"Video processing failed: {video_file.name}")
    print(f"OK ({video_file.name})")
    return video_file


def find_all_participants() -> list[str]:
    """Find all participants with timeline files."""
    participants = []
    for d in sorted(OUTPUT_DIR.iterdir()):
        if d.is_dir() and d.name.startswith('P'):
            if (d / f"{d.name}_timeline_annotated.json").exists():
                participants.append(d.name)
    return participants


def load_items(participant: str, difficulty: str = None, test_n: int = None) -> list[dict]:
    """Load items from timeline, optionally filtering by difficulty."""
    timeline_path = OUTPUT_DIR / participant / f"{participant}_timeline_annotated.json"
    if not timeline_path.exists():
        return []

    with open(timeline_path) as f:
        timeline = json.load(f)

    all_items = timeline.get('items', []) if isinstance(timeline, dict) else timeline
    items = [item for item in all_items if item.get('dispensal_segments')]

    if difficulty:
        items = [item for item in items if item.get('difficulty') == difficulty]

    difficulty_order = {'LOW': 0, 'MID': 1, 'HIGH': 2, 'UNKNOWN': 3}
    items.sort(key=lambda x: difficulty_order.get(x.get('difficulty', 'UNKNOWN'), 3))

    if test_n:
        items = items[:test_n]

    return items


# ── Build ────────────────────────────────────────────────────

def build_jsonl(
    participants: list[str],
    tag: str,
    prompt_mode: str = "hybrid",
    difficulty: str = None,
    test_n: int = None,
) -> Path:
    """
    Build a combined JSONL file for Gemini batch across all participants.
    """
    from google import genai

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Set GOOGLE_API_KEY environment variable")

    client = genai.Client(api_key=api_key)

    prompt_template = PROMPTS.get(prompt_mode)
    if not prompt_template:
        raise ValueError(f"Unknown prompt mode: {prompt_mode}. Available: {list(PROMPTS.keys())}")

    jsonl_lines = []
    upload_cache = {}
    total_items = 0
    total_skipped = 0

    for participant in participants:
        items = load_items(participant, difficulty=difficulty, test_n=test_n)
        if not items:
            print(f"\n{participant}: No items found (difficulty={difficulty})")
            continue

        print(f"\n{'='*60}")
        print(f"PARTICIPANT: {participant} ({len(items)} items, difficulty={difficulty or 'ALL'})")
        print(f"{'='*60}")

        clips_dir = OUTPUT_DIR / participant / "vlm_clips"
        clips_dir.mkdir(parents=True, exist_ok=True)

        for i, item in enumerate(items):
            food_name = item.get('food_name', 'unknown')
            narr_id = item.get('narration_id', '')
            video_range = item.get('video_range', [])
            segments = item.get('dispensal_segments', [])

            print(f"\n  [{i+1}/{len(items)}] {food_name} ({len(segments)} segs)")

            prompt = prompt_template.format(item_name=food_name)
            full_prompt = f"{SYSTEM_PROMPT}\n\n{prompt}"

            for seg_idx, segment in enumerate(segments):
                start_ts = segment.get('start_timestamp', 0)
                end_ts = segment.get('end_timestamp', 0)
                video_id = segment.get('video_id') or (video_range[0] if video_range else None)
                if not video_id:
                    total_skipped += 1
                    continue

                video_path = VIDEO_BASE_DIR / participant / f"{video_id}.mp4"
                if not video_path.exists():
                    print(f"    Seg {seg_idx+1}: SKIP - video not found")
                    total_skipped += 1
                    continue

                # Extract clip
                clip_filename = f"{video_id}_seg{seg_idx}_{start_ts:.0f}_{end_ts:.0f}.mp4"
                clip_path = clips_dir / clip_filename
                clip_start = compute_clip_start(start_ts)

                if not clip_path.exists():
                    print(f"    Seg {seg_idx+1}: Extracting clip...", end=" ", flush=True)
                    success, clip_start = extract_clip(video_path, start_ts, end_ts, clip_path)
                    if not success:
                        print("FAILED")
                        total_skipped += 1
                        continue
                    print(f"OK ({clip_path.stat().st_size / 1024:.1f} KB)")
                else:
                    print(f"    Seg {seg_idx+1}: Using cached clip")

                # Upload clip (or reuse)
                clip_key = str(clip_path)
                if clip_key not in upload_cache:
                    video_file = upload_video(client, clip_path)
                    upload_cache[clip_key] = {
                        'uri': video_file.uri,
                        'name': video_file.name,
                    }
                file_info = upload_cache[clip_key]

                # Build request key
                request_key = f"{participant}|{narr_id}|{food_name}|seg{seg_idx}|{video_id}|{start_ts}|{end_ts}|{clip_start}"

                request = {
                    "key": request_key,
                    "request": {
                        "contents": [{
                            "parts": [
                                {
                                    "file_data": {
                                        "file_uri": file_info['uri'],
                                        "mime_type": "video/mp4"
                                    }
                                },
                                {
                                    "text": full_prompt
                                }
                            ],
                            "role": "user"
                        }],
                        "generation_config": {
                            "temperature": 0.3,
                            "max_output_tokens": 8192,
                        }
                    }
                }

                jsonl_lines.append(json.dumps(request))
                total_items += 1

    # Write combined JSONL
    diff_suffix = f"_{difficulty.lower()}" if difficulty else ""
    jsonl_path = OUTPUT_DIR / f"batch_{tag}{diff_suffix}.jsonl"
    with open(jsonl_path, 'w') as f:
        f.write('\n'.join(jsonl_lines) + '\n')

    # Save manifest
    manifest_path = OUTPUT_DIR / f"batch_{tag}{diff_suffix}_manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump({
            'jsonl_path': str(jsonl_path),
            'participants': participants,
            'prompt_mode': prompt_mode,
            'difficulty': difficulty,
            'num_requests': len(jsonl_lines),
            'num_clips_uploaded': len(upload_cache),
            'num_skipped': total_skipped,
            'tag': tag,
        }, f, indent=2)

    print(f"\n{'='*60}")
    print(f"JSONL written: {jsonl_path}")
    print(f"  Requests: {len(jsonl_lines)}")
    print(f"  File size: {jsonl_path.stat().st_size / 1024:.1f} KB")
    print(f"  Uploaded clips: {len(upload_cache)}")
    print(f"  Skipped: {total_skipped}")
    print(f"  Participants: {', '.join(participants)}")
    print(f"  Manifest: {manifest_path}")

    return jsonl_path


# ── Submit ───────────────────────────────────────────────────

def submit_batch(jsonl_path: str, tag: str, model: str = GEMINI_MODEL):
    """Upload JSONL and submit batch job."""
    from google import genai
    from google.genai import types

    api_key = os.getenv("GOOGLE_API_KEY")
    client = genai.Client(api_key=api_key)

    jsonl_path = Path(jsonl_path)
    if not jsonl_path.exists():
        raise FileNotFoundError(f"JSONL not found: {jsonl_path}")

    print(f"\nUploading JSONL: {jsonl_path} ({jsonl_path.stat().st_size / 1024:.1f} KB)")
    uploaded_jsonl = client.files.upload(
        file=str(jsonl_path),
        config=types.UploadFileConfig(
            display_name=f"batch-{tag}",
            mime_type="jsonl"
        )
    )
    print(f"  Uploaded as: {uploaded_jsonl.name}")

    print(f"Creating batch job (model={model})...")
    batch_job = client.batches.create(
        model=model,
        src=uploaded_jsonl.name,
        config={'display_name': f"vlm-qa-{tag}"},
    )
    print(f"  Batch job created: {batch_job.name}")
    print(f"  State: {batch_job.state}")

    return batch_job


# ── Status / Results ─────────────────────────────────────────

def check_status(job_name: str, wait: bool = False):
    """Check batch job status, optionally wait for completion."""
    from google import genai

    api_key = os.getenv("GOOGLE_API_KEY")
    client = genai.Client(api_key=api_key)

    completed_states = {
        'JOB_STATE_SUCCEEDED',
        'JOB_STATE_FAILED',
        'JOB_STATE_CANCELLED',
        'JOB_STATE_EXPIRED',
    }

    batch_job = client.batches.get(name=job_name)
    print(f"Job: {batch_job.name}")
    print(f"State: {batch_job.state}")

    if wait:
        while batch_job.state.name not in completed_states:
            print(f"  Waiting... (state={batch_job.state.name})")
            time.sleep(30)
            batch_job = client.batches.get(name=job_name)

    print(f"Final state: {batch_job.state}")

    if batch_job.state.name == 'JOB_STATE_SUCCEEDED':
        print("\nRetrieving results...")

        if batch_job.dest and batch_job.dest.file_name:
            result_file_name = batch_job.dest.file_name
            print(f"  Result file: {result_file_name}")
            file_content = client.files.download(file=result_file_name)
            decoded = file_content.decode('utf-8')

            # Parse and display summary
            results = []
            errors = 0
            for line in decoded.splitlines():
                if line.strip():
                    r = json.loads(line)
                    results.append(r)
                    if 'error' in r:
                        errors += 1

            print(f"  Got {len(results)} responses ({errors} errors)\n")

            # Group by participant
            by_participant = {}
            for r in results:
                key = r.get('key', '')
                participant = key.split('|')[0] if '|' in key else '?'
                by_participant.setdefault(participant, []).append(r)

            for p, p_results in sorted(by_participant.items()):
                ok = sum(1 for r in p_results if 'response' in r)
                err = sum(1 for r in p_results if 'error' in r)
                print(f"  {p}: {ok} OK, {err} errors")

            # Save results
            safe_name = job_name.replace('/', '_')
            output_path = OUTPUT_DIR / f"{safe_name}_results.jsonl"
            with open(output_path, 'w') as f:
                f.write(decoded)
            print(f"\n  Saved raw results to: {output_path}")
            return results

        elif batch_job.dest and batch_job.dest.inlined_responses:
            print("  Results are inline:")
            for i, resp in enumerate(batch_job.dest.inlined_responses):
                print(f"  Response {i+1}: {resp}")

    elif batch_job.state.name == 'JOB_STATE_FAILED':
        print(f"  Error: {batch_job.error}")

    return None


# ── Main ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Gemini Batch API for VLM pipeline")
    parser.add_argument('--build', action='store_true',
                        help='Build JSONL (extract clips, upload, write JSONL)')
    parser.add_argument('--submit', action='store_true',
                        help='Submit batch job from JSONL')
    parser.add_argument('--status', action='store_true',
                        help='Check batch job status')

    # Participant selection
    parser.add_argument('--participant', type=str,
                        help='Single participant (e.g., P01)')
    parser.add_argument('--all', action='store_true',
                        help='Process all participants')

    # Filtering
    parser.add_argument('--difficulty', type=str, choices=['LOW', 'MID', 'HIGH'],
                        help='Filter by difficulty level')
    parser.add_argument('--test', type=int, default=None,
                        help='Limit items per participant (for testing)')

    # Config
    parser.add_argument('--tag', type=str, default='batch_test',
                        help='Tag for output files')
    parser.add_argument('--prompt', type=str, default='hybrid',
                        choices=list(PROMPTS.keys()),
                        help=f'Prompt mode (default: hybrid)')
    parser.add_argument('--model', type=str, default=GEMINI_MODEL,
                        help=f'Gemini model (default: {GEMINI_MODEL})')

    # Status/submit args
    parser.add_argument('--jsonl', type=str, help='Path to existing JSONL file')
    parser.add_argument('--job-name', type=str, help='Batch job name to check')
    parser.add_argument('--wait', action='store_true',
                        help='Wait for job completion when checking status')

    args = parser.parse_args()

    gemini_model = args.model
    jsonl_path = args.jsonl

    if args.build:
        # Determine participants
        if args.all:
            participants = find_all_participants()
        elif args.participant:
            participants = [args.participant]
        else:
            parser.error("Need --participant or --all for --build")

        print(f"Participants: {', '.join(participants)}")
        print(f"Difficulty: {args.difficulty or 'ALL'}")
        print(f"Prompt: {args.prompt}")
        print(f"Model: {gemini_model}")
        if args.test:
            print(f"Test mode: {args.test} items per participant")

        jsonl_path = str(build_jsonl(
            participants=participants,
            tag=args.tag,
            prompt_mode=args.prompt,
            difficulty=args.difficulty,
            test_n=args.test,
        ))

    if args.submit:
        if not jsonl_path:
            raise ValueError("Need --jsonl or --build to have a JSONL file")
        batch_job = submit_batch(jsonl_path, tag=args.tag, model=gemini_model)
        print(f"\nTo check status later:")
        print(f"  python test_gemini_batch.py --status --job-name {batch_job.name}")
        print(f"  python test_gemini_batch.py --status --job-name {batch_job.name} --wait")

    if args.status:
        if not args.job_name:
            raise ValueError("Need --job-name to check status")
        check_status(args.job_name, wait=args.wait)

    if not args.build and not args.submit and not args.status:
        parser.print_help()


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
vlm_detect.py - Send session videos to VLM, detect food interaction events.

Processes all video clips in a session, sends each to a VLM (Gemini or Qwen),
and collects detected food interaction events with timestamps.

Usage:
    # Default: 30s blocks with Qwen
    python system_design/vlm_detect.py --participant kailai --session 20260310-195710 --tag qwen_block30s

    # Whole clips (no splitting)
    python system_design/vlm_detect.py --participant kailai --session 20260310-195710 --tag qwen_wholeclip --block-duration 0

    # Gemini with 60s blocks
    python system_design/vlm_detect.py --participant kailai --all --model gemini --tag gemini_block60s --block-duration 60

    # Test with first 2 clips
    python system_design/vlm_detect.py --participant kailai --session 20260310-195710 --tag test_run --test 2

Prerequisites:
    - ledger.json (from annotating/intilization.py)
    - GOOGLE_API_KEY in .env (for Gemini)
    - Qwen server running (for Qwen)
"""

import argparse
import base64
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv

# Paths
_SCRIPT_DIR = Path(__file__).parent
_SELF_DATA = _SCRIPT_DIR.parent
_KITCHEN_DIR = _SELF_DATA.parent

load_dotenv(_KITCHEN_DIR / ".env")


# =============================================================================
# PROMPT
# =============================================================================

INTERACTION_DETECT_PROMPT = """You are a Visual Interaction Auditor for kitchen tasks.
Watch the ENTIRE video segment and log EVERY instance where a person physically
interacts with a food item from the list below.

**INTERACTION TYPES:**
- **RETRIEVAL**: Bringing food from storage (fridge, cupboard, pantry) to workspace
- **ACCESS**: Opening/unwrapping the container (unscrewing lid, removing foil, cutting bag)
- **DISPENSING**: Transferring food out of container (pouring, scooping, picking, cutting)
- **RESTOCKING**: Returning food to storage or putting it down/away

**POSSIBLE FOOD ITEMS (select from this list):**
{food_items_list}

**OUTPUT:** A JSON array of interaction events, in chronological order.
Each event schema:
{{
  "timestamp_window": "MM:SS - MM:SS",
  "detected_item_name": <string from the food items list>,
  "interaction_type": "RETRIEVAL" | "ACCESS" | "DISPENSING" | "RESTOCKING",
  "confidence": "high" | "medium" | "low",
  "visual_evidence": "Brief description of what the person did."
}}

**GUIDELINES:**
- Log EVERY food interaction you observe, even brief ones.
- Use ONLY food names from the provided list. If an item isn't in the list, skip it.
- Be precise with timestamps. The MM:SS values are relative to the start of this video clip.
- Multiple interactions with the same item should each be separate entries.
- If no food interactions are observed, return an empty array [].

Return ONLY the raw JSON array. Do not use Markdown code fences."""


# =============================================================================
# GEMINI CLIENT
# =============================================================================

class GeminiClient:
    """Minimal Gemini client for video queries."""

    def __init__(self, model_name: str = "gemini-3-flash-preview"):
        from google import genai
        self.model_name = model_name
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Missing GOOGLE_API_KEY")
        self.client = genai.Client(api_key=api_key)
        self._uploaded = {}

    def upload_video(self, video_path: Path):
        key = str(video_path)
        if key in self._uploaded:
            return self._uploaded[key]

        print("(uploading)...", end=" ", flush=True)
        video_file = None
        for attempt in range(5):
            try:
                video_file = self.client.files.upload(file=str(video_path))
                break
            except Exception as e:
                wait = 2 ** attempt * 5
                if attempt < 4:
                    print(f"\n    Upload failed ({e}), retry in {wait}s...", end=" ", flush=True)
                    time.sleep(wait)
                else:
                    raise ValueError(f"Upload failed after 5 attempts: {e}")

        while video_file.state == "PROCESSING":
            time.sleep(1)
            video_file = self.client.files.get(name=video_file.name)
        if video_file.state == "FAILED":
            raise ValueError(f"Video processing failed: {video_file.name}")

        self._uploaded[key] = video_file
        return video_file

    def query(self, prompt: str, video_path: Path, temperature: float = 0.3) -> Tuple[str, Dict]:
        from google.genai import types
        stats = {}
        try:
            t0 = time.time()
            video_file = self.upload_video(video_path)
            contents = [
                types.Part.from_uri(file_uri=video_file.uri, mime_type="video/mp4"),
                prompt,
            ]
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=16384,
                ),
            )
            stats['inference_time_s'] = round(time.time() - t0, 2)
            um = response.usage_metadata
            if um:
                stats['input_tokens'] = um.prompt_token_count
                stats['output_tokens'] = um.candidates_token_count
                stats['total_tokens'] = um.total_token_count
            return response.text or "", stats
        except Exception as e:
            print(f"  ERROR: Gemini API: {e}")
            return "", stats


# =============================================================================
# QWEN CLIENT
# =============================================================================

QWEN3VL_URL = "http://localhost:8000/v1/chat/completions"
QWEN_MODEL = "Qwen/Qwen3.5-35B-A3B"

QWEN3VL_URL = "http://saltyfish.eecs.umich.edu:8000/v1/chat/completions"
QWEN_MODEL = "Qwen/Qwen3-VL-30B-A3B-Instruct"



class QwenClient:
    """Qwen VL client for video queries via local vLLM server."""

    def __init__(self, model_name: str = QWEN_MODEL):
        self.model_name = model_name

    def query(self, prompt: str, video_path: Path, temperature: float = 0.3) -> Tuple[str, Dict]:
        stats = {}
        try:
            with open(video_path, "rb") as f:
                video_b64 = base64.b64encode(f.read()).decode()

            messages = [
                {"role": "system", "content": "You are a Visual Interaction Auditor for kitchen tasks."},
                {"role": "user", "content": [
                    {"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{video_b64}"}},
                    {"type": "text", "text": prompt},
                ]},
            ]

            data = {
                "model": self.model_name,
                "messages": messages,
                "max_tokens": 16384,
                "temperature": temperature,
                "chat_template_kwargs": {"enable_thinking": False},
                "extra_body": {
                    "mm_processor_kwargs": {
                        "fps": 1,
                        "do_sample_frames": True,
                    },
                },
            }

            t0 = time.time()
            response = requests.post(
                QWEN3VL_URL, json=data, timeout=600,
                headers={"Content-Type": "application/json"},
            )
            stats['inference_time_s'] = round(time.time() - t0, 2)
            response.raise_for_status()
            result = response.json()
            usage = result.get('usage', {})
            if usage:
                stats['input_tokens'] = usage.get('prompt_tokens', 0)
                stats['output_tokens'] = usage.get('completion_tokens', 0)
                stats['total_tokens'] = usage.get('total_tokens', 0)
            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["message"]["content"], stats
            return "", stats
        except Exception as e:
            print(f"  ERROR: Qwen API: {e}")
            return "", stats


# =============================================================================
# VIDEO UTILITIES
# =============================================================================

def get_video_duration(video_path: Path) -> float:
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return float(result.stdout.strip())
    except Exception:
        return 0.0


def get_video_blocks(duration: float, block_duration: float) -> List[Tuple[int, float, float]]:
    """Split a duration into blocks. Returns [(block_idx, start, end), ...]."""
    blocks = []
    idx = 0
    start = 0.0
    while start < duration:
        end = min(start + block_duration, duration)
        if end - start >= 2.0:  # skip tiny tail blocks
            blocks.append((idx, start, end))
        idx += 1
        start += block_duration
    return blocks


def extract_video_block(video_path: Path, start: float, end: float, output_path: Path) -> bool:
    """Extract a block from a video using ffmpeg. Re-encodes at 1fps for VLM."""
    duration = end - start
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start),
        "-i", str(video_path),
        "-t", str(duration),
        "-r", "1",
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-an",
        str(output_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        return result.returncode == 0 and output_path.exists()
    except Exception:
        return False


# =============================================================================
# RESPONSE PARSING
# =============================================================================

def parse_timestamp_window(ts_str: str, clip_offset: float) -> Tuple[Optional[float], Optional[float]]:
    """Parse 'MM:SS - MM:SS' into absolute session seconds (adds clip_offset)."""
    if not ts_str:
        return None, None

    def parse_mmss(s: str) -> Optional[float]:
        s = s.strip()
        parts = s.split(':')
        try:
            if len(parts) == 2:
                return int(parts[0]) * 60 + float(parts[1])
            elif len(parts) == 3:
                return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
            else:
                return float(s)
        except (ValueError, IndexError):
            return None

    parts = ts_str.split('-')
    if len(parts) == 2:
        s = parse_mmss(parts[0])
        e = parse_mmss(parts[1])
        if s is not None:
            s += clip_offset
        if e is not None:
            e += clip_offset
        return s, e
    return None, None


def parse_vlm_response(response_text: str) -> List[Dict]:
    if not response_text or not response_text.strip():
        return []

    text = response_text.strip()

    # Try markdown code fence
    m = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
    if m:
        try:
            parsed = json.loads(m.group(1).strip())
            return parsed if isinstance(parsed, list) else [parsed]
        except json.JSONDecodeError:
            pass

    # Try JSON array
    m = re.search(r'\[[\s\S]*\]', text)
    if m:
        try:
            parsed = json.loads(m.group(0))
            return parsed if isinstance(parsed, list) else []
        except json.JSONDecodeError:
            pass

    # Try single object
    m = re.search(r'\{[\s\S]*\}', text)
    if m:
        try:
            parsed = json.loads(m.group(0))
            return [parsed] if isinstance(parsed, dict) else []
        except json.JSONDecodeError:
            pass

    return []


# =============================================================================
# DATA LOADING
# =============================================================================

def participant_dir(participant: str) -> Path:
    return _SELF_DATA / "participants" / participant


def load_food_items(participant: str) -> List[str]:
    """Get food item visual_class names from ledger.json."""
    ledger_path = participant_dir(participant) / "ledger.json"
    if not ledger_path.exists():
        raise FileNotFoundError(f"Ledger not found: {ledger_path}")
    ledger = json.loads(ledger_path.read_text())
    return sorted(set(item["visual_class"] for item in ledger["items"].values()))


def get_session_clips(participant: str, session: str) -> List[Tuple[str, Path, float]]:
    """Get sorted list of (filename, path, duration) for a session's clips."""
    session_dir = participant_dir(participant) / "videos" / session
    if not session_dir.exists():
        return []
    clips = []
    for mp4 in sorted(session_dir.glob("*.mp4")):
        dur = get_video_duration(mp4)
        if dur > 0:
            clips.append((mp4.name, mp4, dur))
    return clips


def get_sessions(participant: str) -> List[str]:
    videos_dir = participant_dir(participant) / "videos"
    if not videos_dir.exists():
        return []
    return sorted(d.name for d in videos_dir.iterdir() if d.is_dir())


# =============================================================================
# VLM INFERENCE
# =============================================================================

BLOCK_DURATION_DEFAULT = 30.0  # seconds; split clips into blocks by default


def _query_and_collect(
    vlm, model_name: str, prompt: str, video_path: Path,
    session_offset: float, label: str,
) -> Tuple[List[Dict], Dict]:
    """Query VLM on a single video/block, return (detections, stats)."""
    print(f"    Querying {model_name}...", end=" ", flush=True)
    response, stats = vlm.query(prompt, video_path)

    if not response:
        stats['num_detections'] = 0
        print("NO RESPONSE")
        return [], stats

    detections = parse_vlm_response(response)
    stats['num_detections'] = len(detections)

    in_tok = stats.get('input_tokens', '?')
    out_tok = stats.get('output_tokens', '?')
    t_s = stats.get('inference_time_s', '?')
    print(f"detected {len(detections)} events  (in={in_tok} out={out_tok} t={t_s}s)")

    resolved = []
    for det in detections:
        ts_str = det.get('timestamp_window', '')
        start_abs, end_abs = parse_timestamp_window(ts_str, session_offset)

        raw_det = {
            'detected_item_name': det.get('detected_item_name', ''),
            'interaction_type': det.get('interaction_type', ''),
            'confidence': det.get('confidence', ''),
            'visual_evidence': det.get('visual_evidence', ''),
            'timestamp_window_raw': ts_str,
            'det_start_abs': start_abs,
            'det_end_abs': end_abs,
            'source_label': label,
        }
        resolved.append(raw_det)

        item = det.get('detected_item_name', '?')
        itype = det.get('interaction_type', '?')
        print(f"      [{item}] {itype} ts={ts_str}")

    return resolved, stats


def run_vlm_session(
    participant: str,
    session: str,
    vlm,
    model_name: str,
    tag: str = "default",
    block_duration: float = BLOCK_DURATION_DEFAULT,
    test_limit: int = 0,
    delete_blocks: bool = False,
) -> Optional[Dict]:
    """Run VLM on all clips in a session, collect food interaction detections.

    If block_duration > 0, each clip is split into blocks of that length (seconds)
    and each block is sent to the VLM separately. If block_duration <= 0, whole
    clips are sent directly.
    """
    food_items = load_food_items(participant)
    food_items_str = "\n".join(f"  - {name}" for name in food_items)
    prompt = INTERACTION_DETECT_PROMPT.format(food_items_list=food_items_str)

    clips = get_session_clips(participant, session)
    if not clips:
        print(f"  No clips found for session {session}")
        return None

    if test_limit > 0:
        clips = clips[:test_limit]

    use_blocks = block_duration > 0

    print(f"\n{'='*70}")
    print(f"SESSION: {session} ({len(clips)} clips)")
    print(f"{'='*70}")
    print(f"Food items: {len(food_items)}")
    if use_blocks:
        print(f"Block duration: {block_duration:.0f}s")

    # Build clip offset map (cumulative session time)
    clip_offsets = []
    offset = 0.0
    for filename, _path, dur in clips:
        clip_offsets.append(offset)
        offset += dur
    total_duration = offset

    print(f"Total duration: {total_duration:.1f}s ({total_duration/60:.1f}min)")

    # Count total queries for progress display
    if use_blocks:
        total_queries = 0
        for _, _, dur in clips:
            total_queries += len(get_video_blocks(dur, block_duration))
    else:
        total_queries = len(clips)

    print(f"Total VLM queries: {total_queries}")

    # Prepare blocks cache directory
    output_dir = participant_dir(participant) / "outputs" / session
    output_dir.mkdir(parents=True, exist_ok=True)
    blocks_dir = output_dir / "blocks_cache"
    if use_blocks:
        blocks_dir.mkdir(exist_ok=True)

    all_detections = []
    query_stats = []
    query_idx = 0

    for i, (filename, clip_path, dur) in enumerate(clips):
        clip_offset = clip_offsets[i]
        clip_stem = Path(filename).stem

        if use_blocks:
            blocks = get_video_blocks(dur, block_duration)
            print(f"\n  Clip {i+1}/{len(clips)}: {filename} ({dur:.1f}s, {len(blocks)} blocks)")

            for block_idx, block_start, block_end in blocks:
                query_idx += 1
                session_offset = clip_offset + block_start
                label = f"{filename}[{block_start:.0f}s-{block_end:.0f}s]"
                print(f"\n    [{query_idx}/{total_queries}] {label}")

                # Extract block
                block_name = f"{clip_stem}_block{block_idx}_{block_start:.0f}s_{block_end:.0f}s.mp4"
                block_path = blocks_dir / block_name

                if not block_path.exists():
                    print("    Extracting...", end=" ", flush=True)
                    success = extract_video_block(clip_path, block_start, block_end, block_path)
                    if not success:
                        print("FAILED")
                        continue
                    size_mb = block_path.stat().st_size / (1024 * 1024)
                    print(f"OK ({size_mb:.1f} MB)")

                detections, stats = _query_and_collect(
                    vlm, model_name, prompt, block_path, session_offset, label,
                )
                stats['clip_filename'] = filename
                stats['clip_offset'] = clip_offset
                stats['clip_duration'] = dur
                stats['block_idx'] = block_idx
                stats['block_start'] = block_start
                stats['block_end'] = block_end
                query_stats.append(stats)
                all_detections.extend(detections)

                if delete_blocks and block_path.exists():
                    block_path.unlink()
        else:
            query_idx += 1
            print(f"\n  [{query_idx}/{total_queries}] {filename} ({dur:.1f}s, offset {clip_offset:.1f}s)")

            detections, stats = _query_and_collect(
                vlm, model_name, prompt, clip_path, clip_offset, filename,
            )
            stats['clip_filename'] = filename
            stats['clip_offset'] = clip_offset
            stats['clip_duration'] = dur
            query_stats.append(stats)
            all_detections.extend(detections)

    # Cleanup blocks cache
    if use_blocks and delete_blocks:
        try:
            blocks_dir.rmdir()
        except OSError:
            pass

    # Aggregate stats
    total_input = sum(s.get('input_tokens', 0) for s in query_stats)
    total_output = sum(s.get('output_tokens', 0) for s in query_stats)
    total_time = sum(s.get('inference_time_s', 0) for s in query_stats)
    n_queries = len(query_stats)
    token_stats = {
        'total_input_tokens': total_input,
        'total_output_tokens': total_output,
        'total_tokens': total_input + total_output,
        'total_inference_time_s': round(total_time, 2),
        'num_queries': n_queries,
        'avg_input_tokens_per_query': round(total_input / n_queries, 0) if n_queries else 0,
        'avg_inference_time_per_query_s': round(total_time / n_queries, 2) if n_queries else 0,
    }

    # Save results
    out_file = output_dir / f"vlm_{tag}_results.json"

    results = {
        'participant': participant,
        'session': session,
        'tag': tag,
        'method': f'vlm_{model_name}',
        'block_duration': block_duration if use_blocks else None,
        'num_food_items': len(food_items),
        'food_items': food_items,
        'num_clips': len(clips),
        'total_duration_s': round(total_duration, 2),
        'num_detections': len(all_detections),
        'token_stats': token_stats,
        'query_stats': query_stats,
        'detections': all_detections,
    }

    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary
    print(f"\n{'='*50}")
    print(f"RESULTS: {session}")
    print(f"{'='*50}")
    print(f"  Clips:            {len(clips)}")
    print(f"  VLM queries:      {n_queries}" + (f" ({int(block_duration)}s blocks)" if use_blocks else " (whole clips)"))
    print(f"  VLM detections:   {len(all_detections)}")
    print(f"\n  Token usage:")
    print(f"    Input tokens:   {total_input:,}")
    print(f"    Output tokens:  {total_output:,}")
    print(f"    Total tokens:   {total_input + total_output:,}")
    print(f"    Inference time: {total_time:.1f}s ({total_time/60:.1f}min)")
    print(f"\n  Saved: {out_file}")

    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="VLM food interaction detection for self_data")
    parser.add_argument('--participant', required=True, help='Participant ID (e.g., kailai)')
    parser.add_argument('--session', help='Session ID (e.g., 20260310-195710)')
    parser.add_argument('--all', action='store_true', help='Process all sessions')

    parser.add_argument('--tag', required=True,
                        help='Run tag for output naming (e.g., qwen_wholeclip, gemini_block30s)')
    parser.add_argument('--model', default='qwen', choices=['gemini', 'qwen'],
                        help='VLM backend (default: qwen)')
    parser.add_argument('--gemini-model', default='gemini-3-flash-preview')
    parser.add_argument('--block-duration', type=float, default=BLOCK_DURATION_DEFAULT,
                        help=f'Split clips into blocks of N seconds (default: {BLOCK_DURATION_DEFAULT} = whole clips). '
                             f'Set >0 to enable block splitting.')
    parser.add_argument('--delete-blocks', action='store_true',
                        help='Delete extracted block clips after processing')
    parser.add_argument('--test', type=int, default=0,
                        help='Test: process only first N clips per session')

    args = parser.parse_args()

    if not args.session and not args.all:
        parser.error("Need --session or --all")

    if args.all:
        sessions = get_sessions(args.participant)
    else:
        sessions = [args.session]

    print(f"Participant: {args.participant}")
    print(f"Sessions: {sessions}")

    if args.model == 'qwen':
        print(f"Initializing Qwen ({QWEN_MODEL})...")
        vlm = QwenClient()
    else:
        print(f"Initializing Gemini ({args.gemini_model})...")
        vlm = GeminiClient(model_name=args.gemini_model)

    all_results = []
    for session in sessions:
        result = run_vlm_session(
            participant=args.participant,
            session=session,
            vlm=vlm,
            model_name=args.model,
            tag=args.tag,
            block_duration=args.block_duration,
            test_limit=args.test,
            delete_blocks=args.delete_blocks,
        )
        if result:
            all_results.append(result)

    if len(all_results) > 1:
        total_dets = sum(r['num_detections'] for r in all_results)
        total_time = sum(r['token_stats']['total_inference_time_s'] for r in all_results)
        total_tokens = sum(r['token_stats']['total_tokens'] for r in all_results)
        print(f"\n{'='*70}")
        print("AGGREGATE")
        print(f"{'='*70}")
        print(f"  Sessions:         {len(all_results)}")
        print(f"  Total detections: {total_dets}")
        print(f"  Total tokens:     {total_tokens:,}")
        print(f"  Total time:       {total_time:.1f}s ({total_time/60:.1f}min)")


if __name__ == '__main__':
    main()

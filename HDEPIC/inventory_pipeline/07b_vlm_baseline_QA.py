#!/usr/bin/env python3
"""
07b_vlm_baseline_QA.py - Baseline VLM Q&A using fixed 30-second video blocks

This baseline approach:
1. Divides videos into fixed 30-second blocks
2. For each dispensal segment, finds which block(s) it overlaps with
3. Runs VLM on those blocks WITHOUT specifying the target item name
4. Outputs results per segment (may include multiple block results)

This serves as a baseline comparison to the targeted approach in 07_vlm_QA.py.

Usage:
    python 07b_vlm_baseline_QA.py --participant P03 --test 5
    python 07b_vlm_baseline_QA.py --participant P03 --model qwen
"""

import argparse
import base64
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Dict, Optional, Any, List, Tuple

import requests
from dotenv import load_dotenv
from openai import AzureOpenAI

from inventory_utils import DEFAULT_OUTPUT_DIR

# Load environment from project root
load_dotenv(Path(__file__).parent.parent.parent.parent / ".env")

# Paths
_SCRIPT_DIR = Path(__file__).parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
VIDEO_BASE_DIR = _PROJECT_ROOT / "data" / "HD-EPIC" / "Videos"

# Qwen3-VL API endpoint
QWEN3VL_URL = "http://saltyfish.eecs.umich.edu:8000/v1/chat/completions"
QWEN_MODEL = "Qwen/Qwen3-VL-30B-A3B-Instruct"

# Block duration in seconds
BLOCK_DURATION = 30.0


BASELINE_PROMPT = """You are a Visual Inventory Auditor.
Analyze the video clip to identify ANY food items being dispensed, removed, or used.

**INSTRUCTIONS:**
1. Identify all food items that are being taken out, poured, or dispensed from containers.
2. For EACH item observed, estimate the quantity removed AND the amount remaining.
3. Determine if each item is **Discrete** (countable items like eggs, slices) or **Continuous** (liquids, powders).
4. Return results for ALL food items observed in this clip.

**OUTPUT SCHEMA (Strict JSON - return an array of items):**
{{
  "items": [
    {{
      "item_name": "<detected food item name>",
      "quantity_category": "discrete" | "continuous" | "unknown",

      // AMOUNT REMOVED:
      "numeric_count": <integer or null>,
      "amount_description": <string or null>,
      "volume_fraction": <float or null>,

      // AMOUNT REMAINING:
      "remaining_count": <integer or null>,
      "remaining_description": <string or null>,
      "remaining_fraction": <float or null>,

      "unit_type": "unit" | "scoop" | "cup" | "splash" | "pinch" | "slice" | null,
      "confidence": "high" | "medium" | "low",
      "visual_evidence": "Brief description"
    }}
  ]
}}

If NO food items are being dispensed in this clip, return:
{{"items": []}}

Return ONLY the raw JSON string. Do not use Markdown (```json).
"""


class VLMClient:
    """Handles communication with Qwen VLM API"""

    def __init__(self, model_name: str = 'qwen', use_video: bool = True):
        self.model_name = model_name
        self.use_video = use_video

    def encode_video_base64(self, video_path: Path) -> str:
        """Encode video file to base64"""
        with open(video_path, "rb") as f:
            return base64.b64encode(f.read()).decode()

    def query(
        self,
        system_prompt: str,
        user_prompt: str,
        video_path: Optional[Path] = None,
        max_tokens: int = 2000,
        temperature: float = 0.3
    ) -> str:
        """Query Qwen3-VL"""
        messages = [{"role": "system", "content": system_prompt}]

        user_content = []
        if self.use_video and video_path and video_path.exists():
            video_base64 = self.encode_video_base64(video_path)
            user_content.append({
                "type": "video_url",
                "video_url": {"url": f"data:video/mp4;base64,{video_base64}"}
            })

        user_content.append({"type": "text", "text": user_prompt})
        messages.append({"role": "user", "content": user_content})

        data = {
            "model": QWEN_MODEL,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        if self.use_video and video_path and video_path.exists():
            data["extra_body"] = {
                "mm_processor_kwargs": {
                    "fps": 2,
                    "do_sample_frames": True
                }
            }

        headers = {"Content-Type": "application/json"}

        try:
            response = requests.post(QWEN3VL_URL, headers=headers, json=data, timeout=300)
            response.raise_for_status()
            result = response.json()
            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["message"]["content"]
            return ""
        except Exception as e:
            print(f"  ERROR: Qwen API Error: {e}")
            return ""


class GPT4oClient:
    """Handles communication with Azure OpenAI GPT-4o API using frame sampling"""

    def __init__(self, fps: float = 2.0, max_frames: int = 30):
        self.fps = fps
        self.max_frames = max_frames
        self.model = "gpt-4o"

        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()

        if not api_key or not endpoint:
            raise ValueError("Missing Azure OpenAI API credentials")

        self.client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version="2025-01-01-preview",
        )

    def extract_frames(self, video_path: Path) -> List[str]:
        """Extract frames from video at specified FPS"""
        import cv2

        frames_b64 = []
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            return []

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / video_fps if video_fps > 0 else 0
        expected_frames = int(duration * self.fps)

        if expected_frames > self.max_frames:
            effective_fps = self.max_frames / duration if duration > 0 else self.fps
            frame_interval = int(video_fps / effective_fps) if effective_fps > 0 else 1
        else:
            frame_interval = int(video_fps / self.fps) if self.fps > 0 and video_fps > 0 else 1

        frame_interval = max(1, frame_interval)
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                frame_b64 = base64.b64encode(buffer).decode('utf-8')
                frames_b64.append(frame_b64)

                if len(frames_b64) >= self.max_frames:
                    break

            frame_idx += 1

        cap.release()
        return frames_b64

    def query(
        self,
        system_prompt: str,
        user_prompt: str,
        video_path: Optional[Path] = None,
        max_tokens: int = 2000,
        temperature: float = 0.3
    ) -> str:
        """Query GPT-4o with frames"""
        messages = [{"role": "system", "content": system_prompt}]
        user_content = []

        if video_path and video_path.exists():
            frames = self.extract_frames(video_path)
            if frames:
                frame_info = f"[Video frames: {len(frames)} frames]\n\n"
                user_content.append({"type": "text", "text": frame_info})

                for frame_b64 in frames:
                    user_content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{frame_b64}",
                            "detail": "low"
                        }
                    })

        user_content.append({"type": "text", "text": user_prompt})
        messages.append({"role": "user", "content": user_content})

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            if response.choices and len(response.choices) > 0:
                return response.choices[0].message.content
            return ""
        except Exception as e:
            print(f"  ERROR: GPT-4o API Error: {e}")
            return ""


def get_video_duration(video_path: Path) -> float:
    """Get video duration in seconds using ffprobe"""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path)
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return float(result.stdout.strip())
    except Exception:
        return 0.0


def extract_video_block(
    video_path: Path,
    block_start: float,
    block_end: float,
    output_path: Path
) -> bool:
    """Extract a 30s video block using ffmpeg"""
    duration = block_end - block_start

    cmd = [
        "ffmpeg", "-y",
        "-ss", str(block_start),
        "-i", str(video_path),
        "-t", str(duration),
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-an",
        str(output_path)
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        return result.returncode == 0 and output_path.exists()
    except Exception as e:
        print(f"  ERROR extracting block: {e}")
        return False


def get_overlapping_blocks(
    start_time: float,
    end_time: float,
    block_duration: float = BLOCK_DURATION
) -> List[Tuple[int, float, float]]:
    """
    Get the block indices and time ranges that overlap with a given segment.

    Returns list of (block_index, block_start, block_end) tuples.
    """
    blocks = []

    # Find first block that contains start_time
    first_block = int(start_time // block_duration)
    # Find last block that contains end_time
    last_block = int(end_time // block_duration)

    for block_idx in range(first_block, last_block + 1):
        block_start = block_idx * block_duration
        block_end = (block_idx + 1) * block_duration
        blocks.append((block_idx, block_start, block_end))

    return blocks


def parse_baseline_response(response_text: str) -> List[Dict[str, Any]]:
    """
    Parse JSON array from baseline VLM response.

    Expected format: {"items": [...]}
    """
    result = None

    # Method 1: Extract JSON from markdown code block
    code_block_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response_text)
    if code_block_match:
        try:
            result = json.loads(code_block_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Method 2: Greedy match for raw JSON object
    if result is None:
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if json_match:
            try:
                result = json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass

    # Extract items array
    if result and isinstance(result, dict):
        items = result.get('items', [])
        if isinstance(items, list):
            # Normalize each item
            normalized_items = []
            for item in items:
                normalized_items.append({
                    'item_name': item.get('item_name', 'unknown'),
                    'quantity_category': item.get('quantity_category', 'unknown'),
                    'numeric_count': item.get('numeric_count'),
                    'amount_description': item.get('amount_description'),
                    'volume_fraction': item.get('volume_fraction'),
                    'remaining_count': item.get('remaining_count'),
                    'remaining_description': item.get('remaining_description'),
                    'remaining_fraction': item.get('remaining_fraction'),
                    'unit_type': item.get('unit_type'),
                    'confidence': item.get('confidence', 'low'),
                    'visual_evidence': item.get('visual_evidence', ''),
                })
            return normalized_items

    return []


def main():
    parser = argparse.ArgumentParser(
        description="Baseline VLM Q&A using fixed 30-second video blocks"
    )
    parser.add_argument('--participant', required=True, help='Participant ID (e.g., P03)')
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--test', type=int, default=0, help='Test mode: only process first N items')
    parser.add_argument('--model', default='qwen', choices=['qwen', 'gpt4o'])
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--delete-clips', action='store_true', help='Delete extracted clips after processing')
    parser.add_argument('--block-duration', type=float, default=BLOCK_DURATION, help='Block duration in seconds')

    args = parser.parse_args()
    participant = args.participant
    participant_dir = args.output_dir / participant
    block_duration = args.block_duration

    # Load timeline annotated data
    timeline_file = participant_dir / f"{participant}_timeline_annotated.json"
    if not timeline_file.exists():
        print(f"ERROR: {timeline_file.name} not found")
        return

    with open(timeline_file, 'r') as f:
        timeline_data = json.load(f)

    items = timeline_data.get('items', [])
    print(f"Loaded {len(items)} items from {timeline_file.name}")

    # Filter items with valid segments
    valid_items = [
        item for item in items
        if item.get('dispensal_segments') and len(item.get('dispensal_segments', [])) > 0
    ]
    print(f"Items with valid segments: {len(valid_items)}")

    if args.test > 0:
        valid_items = valid_items[:args.test]
        print(f"TEST MODE: Processing only first {len(valid_items)} items")

    # Initialize VLM client
    print(f"\nInitializing {args.model} VLM client...")
    if args.model == 'gpt4o':
        vlm = GPT4oClient(fps=2.0)
    else:
        vlm = VLMClient(model_name=args.model, use_video=True)

    # Create clips directory
    clips_dir = participant_dir / "vlm_baseline_clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    # Cache for block results (to avoid re-querying same blocks)
    block_cache: Dict[str, List[Dict]] = {}  # key: "video_id_blockN" -> detections

    print(f"\n{'='*70}")
    print(f"BASELINE: {block_duration}s BLOCKS (no item name in prompt)")
    print(f"{'='*70}")

    results = []

    for i, item in enumerate(valid_items):
        food_name = item.get('food_name', 'unknown')
        narr_id = item.get('narration_id', '')
        difficulty = item.get('difficulty', 'UNKNOWN')
        segments = item.get('dispensal_segments', [])

        print(f"\n[{i+1}/{len(valid_items)}] {food_name}")
        print(f"  Narration ID: {narr_id}")
        print(f"  Ground truth: {item.get('total_count')} {item.get('count_unit', '')}")
        print(f"  Segments: {len(segments)}")

        segment_results = []

        for seg_idx, segment in enumerate(segments):
            start_ts = segment.get('start_timestamp', 0)
            end_ts = segment.get('end_timestamp', 0)
            gt_count = segment.get('count')
            gt_unit = segment.get('count_unit')
            video_id = segment.get('video_id')

            if not video_id:
                print(f"  Segment {seg_idx+1}: SKIP - no video ID")
                continue

            video_path = VIDEO_BASE_DIR / participant / f"{video_id}.mp4"
            if not video_path.exists():
                print(f"  Segment {seg_idx+1}: SKIP - video not found")
                continue

            # Get overlapping blocks
            blocks = get_overlapping_blocks(start_ts, end_ts, block_duration)
            print(f"  Segment {seg_idx+1}: {start_ts:.1f}s-{end_ts:.1f}s -> blocks {[b[0] for b in blocks]}")

            # Query VLM for each block (use cache if available)
            block_detections = []

            for block_idx, block_start, block_end in blocks:
                cache_key = f"{video_id}_block{block_idx}"

                if cache_key in block_cache:
                    print(f"    Block {block_idx} ({block_start:.0f}s-{block_end:.0f}s): cached")
                    block_detections.append({
                        'block_idx': block_idx,
                        'block_start': block_start,
                        'block_end': block_end,
                        'detections': block_cache[cache_key],
                        'from_cache': True
                    })
                    continue

                # Extract block clip
                clip_filename = f"{video_id}_block{block_idx}_{block_start:.0f}s.mp4"
                clip_path = clips_dir / clip_filename

                if not clip_path.exists():
                    print(f"    Block {block_idx} ({block_start:.0f}s-{block_end:.0f}s): extracting...", end=" ", flush=True)

                    # Get video duration to avoid extracting beyond end
                    video_duration = get_video_duration(video_path)
                    actual_end = min(block_end, video_duration)

                    success = extract_video_block(video_path, block_start, actual_end, clip_path)
                    if not success:
                        print("FAILED")
                        continue
                    print(f"OK ({clip_path.stat().st_size / 1024:.1f} KB)")

                # Query VLM
                print(f"    Querying {args.model} for block {block_idx}...", end=" ", flush=True)
                response = vlm.query(
                    system_prompt="You are a Visual Inventory Auditor analyzing cooking videos.",
                    user_prompt=BASELINE_PROMPT,
                    video_path=clip_path
                )

                if not response:
                    print("NO RESPONSE")
                    block_cache[cache_key] = []
                    block_detections.append({
                        'block_idx': block_idx,
                        'block_start': block_start,
                        'block_end': block_end,
                        'detections': [],
                        'from_cache': False
                    })
                    continue

                # Parse response
                detections = parse_baseline_response(response)
                block_cache[cache_key] = detections

                print(f"found {len(detections)} items")
                if args.verbose and detections:
                    for det in detections:
                        print(f"      - {det.get('item_name')}: {det.get('numeric_count') or det.get('amount_description')}")

                block_detections.append({
                    'block_idx': block_idx,
                    'block_start': block_start,
                    'block_end': block_end,
                    'detections': detections,
                    'from_cache': False
                })

                # Clean up if requested
                if args.delete_clips and clip_path.exists():
                    clip_path.unlink()

            # Collect all detections from blocks (no matching - just attach all)
            all_detections = []
            for bd in block_detections:
                all_detections.extend(bd.get('detections', []))

            # Print summary of detections
            if all_detections:
                print(f"    Detected {len(all_detections)} items: {[d.get('item_name') for d in all_detections]}")
            else:
                print(f"    No items detected")

            segment_results.append({
                'segment_idx': seg_idx,
                'video_id': video_id,
                'start_timestamp': start_ts,
                'end_timestamp': end_ts,
                'ground_truth_count': gt_count,
                'ground_truth_unit': gt_unit,
                'ground_truth_food_name': food_name,
                'blocks_queried': [
                    {
                        'block_idx': bd['block_idx'],
                        'block_start': bd['block_start'],
                        'block_end': bd['block_end'],
                        'num_detections': len(bd['detections']),
                        'detections': bd['detections'],
                        'from_cache': bd['from_cache']
                    }
                    for bd in block_detections
                ],
                'all_detections': all_detections,
            })

        # Aggregate item results
        item_result = {
            'narration_id': narr_id,
            'food_name': food_name,
            'difficulty': difficulty,
            'total_ground_truth': item.get('total_count'),
            'total_ground_truth_unit': item.get('count_unit'),
            'num_segments': len(segments),
            'segments': segment_results,
        }

        # Count total detections across all segments
        total_detections = sum(len(s.get('all_detections', [])) for s in segment_results)
        item_result['total_detections'] = total_detections

        results.append(item_result)

    # Clean up
    if args.delete_clips:
        try:
            clips_dir.rmdir()
        except OSError:
            pass

    # Save results
    print(f"\n{'='*70}")
    print(f"SAVING RESULTS")
    print(f"{'='*70}")

    output_file = participant_dir / f"{participant}_vlm_baseline_{args.model}_results.json"
    output_data = {
        'participant': participant,
        'model': args.model,
        'method': 'baseline',
        'block_duration': block_duration,
        'total_items': len(valid_items),
        'items': results
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)

    print(f"  Saved: {output_file.name}")

    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")

    total_segments = sum(len(r.get('segments', [])) for r in results)
    total_detections = sum(r.get('total_detections', 0) for r in results)
    segments_with_detections = sum(
        1 for r in results
        for s in r.get('segments', [])
        if len(s.get('all_detections', [])) > 0
    )

    print(f"Total items: {len(results)}")
    print(f"Total segments: {total_segments}")
    print(f"Segments with detections: {segments_with_detections} ({100*segments_with_detections/max(1,total_segments):.1f}%)")
    print(f"Total detections: {total_detections}")

    # Results table
    print(f"\n{'='*70}")
    print(f"RESULTS TABLE")
    print(f"{'='*70}")
    print(f"{'GT Food':<30} {'GT Count':<10} {'#Seg':<6} {'#Det':<6} {'Detected Items':<40}")
    print("-" * 95)

    for r in results:
        food = (r.get('food_name') or '')[:29]
        gt = r.get('total_ground_truth')
        gt_str = str(gt) if gt is not None else '-'
        num_seg = len(r.get('segments', []))
        num_det = r.get('total_detections', 0)

        # Collect unique detected item names
        detected_names = set()
        for seg in r.get('segments', []):
            for det in seg.get('all_detections', []):
                detected_names.add(det.get('item_name', 'unknown'))

        detected_str = ', '.join(sorted(detected_names))[:39] if detected_names else '-'

        print(f"{food:<30} {gt_str:<10} {num_seg:<6} {num_det:<6} {detected_str:<40}")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
07_vlm_QA.py - VLM Q&A Engine for Quantity Estimation

Tests VLM capability for estimating quantity change from dispensing actions.
Uses timeline_annotated.json to extract video clips and query VLM.

Prerequisites:
    Run 06_timeline_aggregation.py first to generate timeline_annotated.json

Usage:
    # Test on first 5 items
    python 07_vlm_QA.py --participant P03 --test 5

    # Process all items with ground truth
    python 07_vlm_QA.py --participant P03

    # Specify model
    python 07_vlm_QA.py --participant P03 --model qwen

Inputs:
    {participant}_timeline_annotated.json
    Video files from data/HD-EPIC/Videos/{participant}/

Outputs:
    {participant}_vlm_qa_results.json
"""

import argparse
import base64
import json
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any

import requests

from inventory_utils import DEFAULT_OUTPUT_DIR

# Paths
_SCRIPT_DIR = Path(__file__).parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
VIDEO_BASE_DIR = _PROJECT_ROOT / "data" / "HD-EPIC" / "Videos"

# Qwen3-VL API endpoint
QWEN3VL_URL = "http://saltyfish.eecs.umich.edu:8000/v1/chat/completions"
QWEN_MODEL = "Qwen/Qwen3-VL-30B-A3B-Instruct"


QUANTITY_ESTIMATION_PROMPT = """You are a Visual Inventory Auditor.
Your task is to analyze the video clip and estimate how much of the Target Food Item was removed, used, or dispensed by the user.

**INPUT:**
- **Target Food Item:** "{food_name}"
- **Video Clip:** (Attached)

**INSTRUCTIONS:**
1. **Focus ONLY on the Target Food Item.** Ignore all other ingredients.
2. **Determine the Quantity Type:**
   - **Countable:** Can you distinctly count the items? (e.g., eggs, carrots, scoops, slices).
   - **Uncountable:** Is it a fluid/powder pour? (e.g., milk, oil, flour).
3. **Estimate the Delta (Amount Used):**
   - If **Countable**: Count the exact number of units transferred.
   - If **Uncountable**: Estimate the amount based on:
     - *Container Volume:* (e.g., "About 1/4 of the jar")
     - *Standard Units:* (e.g., "About 1 cup", "A small splash")
     - *Action Duration:* (e.g., "A long 5-second pour" vs "A quick dash")

**OUTPUT FORMAT (JSON):**
{{
  "item_name": "{food_name}",
  "quantity_type": "count" | "volume_estimate" | "unknown",
  "count": <integer or null if uncountable>,
  "count_unit": "<unit name like 'eggs', 'slices', 'scoops', or null>",
  "confidence": "high" | "medium" | "low",
  "reasoning": "Describe the specific visual cues (e.g., 'I saw the user pick up 3 distinct eggs', 'The milk level dropped by half')."
}}

**EXAMPLES:**
- **Input:** Eggs
- **Output:** {{ "quantity_type": "count", "count": 2, "count_unit": "eggs", "confidence": "high", "reasoning": "User picked one egg, placed it in bowl, then picked a second egg." }}

- **Input:** Milk
- **Output:** {{ "quantity_type": "volume_estimate", "count": null, "count_unit": null, "confidence": "medium", "reasoning": "User poured a steady stream for 3 seconds into a small mug, filling it halfway." }}
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


def extract_video_clip(
    video_path: Path,
    start_time: float,
    end_time: float,
    output_path: Path,
    padding: float = 2.0
) -> bool:
    """
    Extract a video clip using ffmpeg.

    Args:
        video_path: Source video file
        start_time: Start timestamp in seconds
        end_time: End timestamp in seconds
        output_path: Output clip path
        padding: Extra seconds to add before/after the segment

    Returns:
        True if extraction successful
    """
    # Add padding
    start_padded = max(0, start_time - padding)
    duration = (end_time - start_time) + (2 * padding)

    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start_padded),
        "-i", str(video_path),
        "-t", str(duration),
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-an",  # No audio
        str(output_path)
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120
        )
        return result.returncode == 0 and output_path.exists()
    except Exception as e:
        print(f"  ERROR extracting clip: {e}")
        return False


def parse_vlm_response(response_text: str) -> Dict[str, Any]:
    """Parse JSON from VLM response."""
    # Try to find JSON in response
    json_match = re.search(r'\{[\s\S]*?\}', response_text)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass

    # Fallback: try to extract count from text
    count_match = re.search(r'(\d+)\s*(eggs?|slices?|pieces?|scoops?|potatoes?|onions?)', response_text.lower())
    if count_match:
        return {
            "count": int(count_match.group(1)),
            "count_unit": count_match.group(2),
            "confidence": "low",
            "reasoning": "Extracted from unstructured response"
        }

    return {
        "count": None,
        "count_unit": None,
        "confidence": "low",
        "reasoning": f"Could not parse response: {response_text[:200]}"
    }


def evaluate_result(predicted: Dict, ground_truth: Dict) -> Dict[str, Any]:
    """Compare predicted count with ground truth."""
    gt_count = ground_truth.get('total_count')
    pred_count = predicted.get('count')

    result = {
        'predicted_count': pred_count,
        'ground_truth_count': gt_count,
        'predicted_unit': predicted.get('count_unit'),
        'ground_truth_unit': ground_truth.get('count_unit'),
        'confidence': predicted.get('confidence'),
        'reasoning': predicted.get('reasoning'),
    }

    if gt_count is None and pred_count is None:
        result['match'] = 'both_uncountable'
        result['error'] = 0
    elif gt_count is None:
        result['match'] = 'gt_uncountable'
        result['error'] = None
    elif pred_count is None:
        result['match'] = 'pred_uncountable'
        result['error'] = None
    else:
        result['error'] = pred_count - gt_count
        result['abs_error'] = abs(pred_count - gt_count)
        if pred_count == gt_count:
            result['match'] = 'exact'
        elif abs(pred_count - gt_count) <= 1:
            result['match'] = 'close'
        else:
            result['match'] = 'wrong'

    return result


def main():
    parser = argparse.ArgumentParser(
        description="VLM Q&A Engine for Quantity Estimation"
    )
    parser.add_argument(
        '--participant',
        required=True,
        help='Participant ID (e.g., P03)'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help='Output directory'
    )
    parser.add_argument(
        '--test',
        type=int,
        default=0,
        help='Test mode: only process first N items (0 = all)'
    )
    parser.add_argument(
        '--model',
        default='qwen',
        choices=['qwen'],
        help='VLM model to use'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )
    parser.add_argument(
        '--keep-clips',
        action='store_true',
        help='Keep extracted video clips (default: delete after processing)'
    )

    args = parser.parse_args()
    participant = args.participant
    participant_dir = args.output_dir / participant

    # Load timeline annotated data
    timeline_file = participant_dir / f"{participant}_timeline_annotated.json"
    if not timeline_file.exists():
        print(f"ERROR: {timeline_file.name} not found")
        print("       Run 06_timeline_aggregation.py first")
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

    # Apply test limit
    if args.test > 0:
        valid_items = valid_items[:args.test]
        print(f"TEST MODE: Processing only first {len(valid_items)} items")

    # Sort by difficulty for processing
    difficulty_order = {'LOW': 0, 'MID': 1, 'HIGH': 2, 'UNKNOWN': 3}
    valid_items.sort(key=lambda x: difficulty_order.get(x.get('difficulty', 'UNKNOWN'), 3))

    # Initialize VLM client
    print(f"\nInitializing {args.model} VLM client...")
    vlm = VLMClient(model_name=args.model, use_video=True)

    # Create temp directory for clips
    clips_dir = participant_dir / "vlm_clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    # Process items
    print(f"\n{'='*70}")
    print(f"PROCESSING ITEMS (sorted by difficulty)")
    print(f"{'='*70}")

    results = []
    current_difficulty = None

    for i, item in enumerate(valid_items):
        food_name = item.get('food_name', 'unknown')
        narr_id = item.get('narration_id', '')
        difficulty = item.get('difficulty', 'UNKNOWN')
        video_range = item.get('video_range', [])
        segments = item.get('dispensal_segments', [])

        # Print difficulty header
        if difficulty != current_difficulty:
            current_difficulty = difficulty
            print(f"\n{'='*70}")
            print(f"DIFFICULTY: {difficulty}")
            print(f"{'='*70}")

        print(f"\n[{i+1}/{len(valid_items)}] {food_name}")
        print(f"  Narration ID: {narr_id}")
        print(f"  Ground truth: {item.get('total_count')} {item.get('count_unit', '')}")
        print(f"  Segments: {len(segments)}")

        # Process each segment
        segment_results = []
        for seg_idx, segment in enumerate(segments):
            start_ts = segment.get('start_timestamp', 0)
            end_ts = segment.get('end_timestamp', 0)
            gt_count = segment.get('count')
            gt_unit = segment.get('count_unit')

            # Get video_id from segment (with fallback to video_range[0])
            video_id = segment.get('video_id') or (video_range[0] if video_range else None)

            print(f"  Segment {seg_idx+1}: [{video_id}] {start_ts:.1f}s - {end_ts:.1f}s (gt: {gt_count} {gt_unit or ''})")

            # Find video file
            if not video_id:
                print(f"    SKIP: No video ID")
                continue

            video_path = VIDEO_BASE_DIR / participant / f"{video_id}.mp4"
            if not video_path.exists():
                print(f"    SKIP: Video not found: {video_path}")
                continue

            # Extract clip (include video_id in filename to avoid collisions)
            clip_filename = f"{video_id}_seg{seg_idx}_{start_ts:.0f}_{end_ts:.0f}.mp4"
            clip_path = clips_dir / clip_filename

            if not clip_path.exists():
                print(f"    Extracting clip...", end=" ", flush=True)
                success = extract_video_clip(video_path, start_ts, end_ts, clip_path)
                if not success:
                    print("FAILED")
                    continue
                print(f"OK ({clip_path.stat().st_size / 1024:.1f} KB)")
            else:
                print(f"    Using cached clip")

            # Query VLM
            print(f"    Querying {args.model}...", end=" ", flush=True)
            prompt = QUANTITY_ESTIMATION_PROMPT.format(food_name=food_name)
            response = vlm.query(
                system_prompt="You are a video analysis assistant.",
                user_prompt=prompt,
                video_path=clip_path
            )

            if not response:
                print("NO RESPONSE")
                segment_results.append({
                    'segment_idx': seg_idx,
                    'start_timestamp': start_ts,
                    'end_timestamp': end_ts,
                    'error': 'No VLM response'
                })
                continue

            # Parse response
            parsed = parse_vlm_response(response)
            pred_count = parsed.get('count')
            pred_unit = parsed.get('count_unit')

            print(f"predicted: {pred_count} {pred_unit or ''}")

            if args.verbose:
                print(f"    Reasoning: {parsed.get('reasoning', '')[:100]}...")

            # Evaluate
            evaluation = evaluate_result(
                parsed,
                {'total_count': gt_count, 'count_unit': gt_unit}
            )

            segment_results.append({
                'segment_idx': seg_idx,
                'video_id': video_id,
                'start_timestamp': start_ts,
                'end_timestamp': end_ts,
                'ground_truth_count': gt_count,
                'ground_truth_unit': gt_unit,
                'predicted_count': pred_count,
                'predicted_unit': pred_unit,
                'confidence': parsed.get('confidence'),
                'reasoning': parsed.get('reasoning'),
                'match': evaluation.get('match'),
                'error': evaluation.get('error'),
                'clip_path': str(clip_path) if args.keep_clips else None
            })

            # Clean up clip if not keeping
            if not args.keep_clips and clip_path.exists():
                clip_path.unlink()

        # Aggregate item results
        item_result = {
            'narration_id': narr_id,
            'food_name': food_name,
            'difficulty': difficulty,
            'video_range': video_range,
            'total_ground_truth': item.get('total_count'),
            'total_ground_truth_unit': item.get('count_unit'),
            'num_segments': len(segments),
            'segments': segment_results
        }

        # Calculate total predicted
        total_predicted = sum(
            s.get('predicted_count', 0) or 0
            for s in segment_results
            if s.get('predicted_count') is not None
        )
        item_result['total_predicted'] = total_predicted if total_predicted > 0 else None

        results.append(item_result)

    # Clean up clips directory if empty
    if not args.keep_clips:
        try:
            clips_dir.rmdir()
        except OSError:
            pass  # Directory not empty

    # Save results
    print(f"\n{'='*70}")
    print(f"SAVING RESULTS")
    print(f"{'='*70}")

    output_file = participant_dir / f"{participant}_vlm_qa_results.json"
    output_data = {
        'participant': participant,
        'model': args.model,
        'total_items': len(valid_items),
        'items': results
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)

    print(f"  Saved: {output_file.name}")

    # Summary by difficulty
    print(f"\n{'='*70}")
    print(f"SUMMARY BY DIFFICULTY")
    print(f"{'='*70}")

    for diff in ['LOW', 'MID', 'HIGH']:
        diff_items = [r for r in results if r.get('difficulty') == diff]
        if not diff_items:
            continue

        total_segments = sum(len(r.get('segments', [])) for r in diff_items)
        exact_matches = sum(
            1 for r in diff_items
            for s in r.get('segments', [])
            if s.get('match') == 'exact'
        )
        close_matches = sum(
            1 for r in diff_items
            for s in r.get('segments', [])
            if s.get('match') == 'close'
        )

        print(f"\n{diff}:")
        print(f"  Items: {len(diff_items)}")
        print(f"  Segments: {total_segments}")
        print(f"  Exact matches: {exact_matches} ({100*exact_matches/max(1,total_segments):.1f}%)")
        print(f"  Close matches (+/-1): {close_matches} ({100*close_matches/max(1,total_segments):.1f}%)")

    # Results table
    print(f"\n{'='*70}")
    print(f"RESULTS TABLE")
    print(f"{'='*70}")
    print(f"{'Food':<25} {'Diff':<6} {'GT':<8} {'Pred':<8} {'Match':<10}")
    print("-" * 60)

    for r in results:
        food = (r.get('food_name') or '')[:24]
        diff = r.get('difficulty', '?')[:5]
        gt = r.get('total_ground_truth')
        gt_str = str(gt) if gt is not None else '-'
        pred = r.get('total_predicted')
        pred_str = str(pred) if pred is not None else '-'

        # Determine overall match
        if gt is None and pred is None:
            match = 'uncountable'
        elif gt is None or pred is None:
            match = 'n/a'
        elif gt == pred:
            match = 'EXACT'
        elif abs(gt - pred) <= 1:
            match = 'close'
        else:
            match = f'off by {pred - gt}'

        print(f"{food:<25} {diff:<6} {gt_str:<8} {pred_str:<8} {match:<10}")


if __name__ == '__main__':
    main()

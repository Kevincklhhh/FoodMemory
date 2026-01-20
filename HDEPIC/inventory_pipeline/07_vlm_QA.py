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
import os
import re
import subprocess
from pathlib import Path
from typing import Dict, Optional, Any, List

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


ACTION_ESTIMATION_PROMPT = """You are a Visual Inventory Auditor.
Analyze the video clip to estimate the quantity of the Target Food Item removed.

**INPUT:**
- Target Item: "{item_name}"

**INSTRUCTIONS:**
1. Focus ONLY on the Target Item.
2. Determine if the action is **Discrete** (countable items like eggs, distinct scoops) or **Continuous** (pouring liquid, approximate piles).
3. Provide the estimate in the strictly defined JSON format below.

**OUTPUT SCHEMA (Strict JSON):**
{{
  "item_name": "{item_name}",
  "quantity_category": "discrete" | "continuous" | "unknown",
  
  // IF DISCRETE (Countable):
  "numeric_count": <integer or null>,  // e.g., 1, 2, 5. Null if continuous.
  
  // IF CONTINUOUS (Fluids/Piles):
  "amount_description": <string>,      // e.g., "about half a cup", "a small splash"
  "volume_fraction": <float or null>,  // Estimate 0.0 to 1.0 of container size if visible. Null if unknown.
  
  "unit_type": "unit" | "scoop" | "cup" | "splash" | "pinch" | "slice",
  "confidence": "high" | "medium" | "low",
  "visual_evidence": "Brief description of visual proof."
}}

**EXAMPLES:**

*Example 1 (Discrete):*
{{
  "item_name": "eggs",
  "quantity_category": "discrete",
  "numeric_count": 2,
  "amount_description": null,
  "volume_fraction": null,
  "unit_type": "unit",
  "confidence": "high",
  "visual_evidence": "User picked two distinct eggs from the carton."
}}

*Example 2 (Continuous):*
{{
  "item_name": "milk",
  "quantity_category": "continuous",
  "numeric_count": null,
  "amount_description": "approx 1/2 cup",
  "volume_fraction": 0.1,
  "unit_type": "cup",
  "confidence": "medium",
  "visual_evidence": "Steady pour for 2 seconds into a small mug."
}}

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
        self.max_frames = max_frames  # Azure OpenAI limit is 50, we use 30 for safety
        self.model = "gpt-4o"

        # Use Azure OpenAI endpoint
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()

        if not api_key or not endpoint:
            raise ValueError("Missing Azure OpenAI API credentials. Set AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT")

        self.client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version="2025-01-01-preview",
        )

    def extract_frames(self, video_path: Path) -> List[str]:
        """
        Extract frames from video at specified FPS and return as base64 encoded images.
        Limits to max_frames by adjusting the sampling interval if needed.
        """
        import cv2

        frames_b64 = []
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            print(f"    ERROR: Could not open video {video_path}")
            return []

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Calculate how many frames we'd get at target FPS
        duration = total_frames / video_fps if video_fps > 0 else 0
        expected_frames = int(duration * self.fps)

        # Adjust frame interval to stay within max_frames
        if expected_frames > self.max_frames:
            # Need to sample less frequently
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
                # Encode frame as JPEG
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                frame_b64 = base64.b64encode(buffer).decode('utf-8')
                frames_b64.append(frame_b64)

                # Stop if we've reached max frames
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
        """Query GPT-4o with frames extracted from video"""
        messages = [{"role": "system", "content": system_prompt}]

        user_content = []

        # Extract frames from video and add as images
        if video_path and video_path.exists():
            frames = self.extract_frames(video_path)
            if frames:
                # Add frame count info to prompt
                frame_info = f"[Video frames: {len(frames)} frames at {self.fps} FPS]\n\n"
                user_content.append({"type": "text", "text": frame_info})

                # Add each frame as an image
                for frame_b64 in frames:
                    user_content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{frame_b64}",
                            "detail": "low"  # Use low detail to reduce tokens
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
    """
    Parse JSON from VLM response.

    Expected schema:
    {
        "quantity_category": "discrete" | "continuous" | "unknown",
        "numeric_count": int or null,        # for discrete items
        "amount_description": str or null,   # for continuous items (e.g., "half a cup")
        "volume_fraction": float or null,    # 0.0-1.0 of container
        "unit_type": str,                    # "unit", "scoop", "cup", etc.
        "confidence": str,
        "visual_evidence": str
    }
    """
    result = None

    # Method 1: Extract JSON from markdown code block (```json ... ```)
    code_block_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response_text)
    if code_block_match:
        try:
            result = json.loads(code_block_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Method 2: Greedy match for raw JSON object (handles nested structures)
    if result is None:
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if json_match:
            try:
                result = json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass

    # Normalize to internal format
    if result:
        normalized = {
            'quantity_category': result.get('quantity_category', 'unknown'),
            'numeric_count': result.get('numeric_count'),
            'amount_description': result.get('amount_description'),
            'volume_fraction': result.get('volume_fraction'),
            'unit_type': result.get('unit_type'),
            'confidence': result.get('confidence', 'low'),
            'visual_evidence': result.get('visual_evidence', ''),
        }
        return normalized

    # Method 3: Regex fallback for plain text like "3 eggs"
    count_match = re.search(r'(\d+)\s*(eggs?|slices?|pieces?|scoops?|potatoes?|onions?|carrots?)', response_text.lower())
    if count_match:
        return {
            'quantity_category': 'discrete',
            'numeric_count': int(count_match.group(1)),
            'amount_description': None,
            'volume_fraction': None,
            'unit_type': count_match.group(2),
            'confidence': 'low',
            'visual_evidence': 'Extracted from unstructured response'
        }

    return {
        'quantity_category': 'unknown',
        'numeric_count': None,
        'amount_description': None,
        'volume_fraction': None,
        'unit_type': None,
        'confidence': 'low',
        'visual_evidence': f"Could not parse response: {response_text[:200]}"
    }


def evaluate_result(predicted: Dict, ground_truth: Dict) -> Dict[str, Any]:
    """
    Compare predicted result with ground truth.

    Handles both discrete (countable) and continuous (volume/weight) items.
    """
    gt_count = ground_truth.get('total_count')
    pred_category = predicted.get('quantity_category', 'unknown')
    pred_count = predicted.get('numeric_count')
    pred_amount = predicted.get('amount_description')

    result = {
        'quantity_category': pred_category,
        'predicted_count': pred_count,
        'predicted_amount': pred_amount,
        'ground_truth_count': gt_count,
        'predicted_unit': predicted.get('unit_type'),
        'ground_truth_unit': ground_truth.get('count_unit'),
        'confidence': predicted.get('confidence'),
        'visual_evidence': predicted.get('visual_evidence'),
    }

    # Evaluate based on category
    if pred_category == 'discrete' and pred_count is not None:
        # Discrete: compare numeric counts
        if gt_count is None:
            result['match'] = 'gt_uncountable'
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
    elif pred_category == 'continuous':
        # Continuous: can't directly compare, mark as estimated
        result['match'] = 'continuous_estimate'
        result['error'] = None
    else:
        # Unknown or failed to parse
        if gt_count is None:
            result['match'] = 'both_unknown'
        else:
            result['match'] = 'pred_unknown'
        result['error'] = None

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
        choices=['qwen', 'gpt4o'],
        help='VLM model to use (qwen: video input, gpt4o: frame sampling at 2fps)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )
    parser.add_argument(
        '--delete-clips',
        action='store_true',
        help='Delete extracted video clips after processing (default: keep clips)'
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
    if args.model == 'gpt4o':
        vlm = GPT4oClient(fps=2.0)
    else:
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
            prompt = ACTION_ESTIMATION_PROMPT.format(item_name=food_name)
            response = vlm.query(
                system_prompt="You are a Visual Inventory Auditor analyzing cooking videos.",
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
            pred_category = parsed.get('quantity_category', 'unknown')
            pred_count = parsed.get('numeric_count')
            pred_amount = parsed.get('amount_description')
            pred_unit = parsed.get('unit_type')

            # Display result based on category
            if pred_category == 'discrete' and pred_count is not None:
                print(f"predicted: {pred_count} {pred_unit or 'units'}")
            elif pred_category == 'continuous' and pred_amount:
                print(f"predicted: {pred_amount} (continuous)")
            else:
                print(f"predicted: unknown")

            if args.verbose:
                print(f"    Evidence: {parsed.get('visual_evidence', '')[:100]}...")

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
                'quantity_category': pred_category,
                'predicted_count': pred_count,
                'predicted_amount': pred_amount,
                'predicted_unit': pred_unit,
                'confidence': parsed.get('confidence'),
                'visual_evidence': parsed.get('visual_evidence'),
                'match': evaluation.get('match'),
                'error': evaluation.get('error'),
                'clip_path': str(clip_path) if not args.delete_clips else None
            })

            # Clean up clip if requested
            if args.delete_clips and clip_path.exists():
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
            'segments': segment_results,
            # Recipe amount (for visualization only, separate from dispensal ground truth)
            'recipe_amount': item.get('matched_ingredient_weight'),
        }

        # Calculate total predicted
        total_predicted = sum(
            s.get('predicted_count', 0) or 0
            for s in segment_results
            if s.get('predicted_count') is not None
        )
        item_result['total_predicted'] = total_predicted if total_predicted > 0 else None

        results.append(item_result)

    # Clean up clips directory if empty and deletion was requested
    if args.delete_clips:
        try:
            clips_dir.rmdir()
        except OSError:
            pass  # Directory not empty

    # Save results
    print(f"\n{'='*70}")
    print(f"SAVING RESULTS")
    print(f"{'='*70}")

    output_file = participant_dir / f"{participant}_vlm_qa_{args.model}_results.json"
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
        continuous_estimates = sum(
            1 for r in diff_items
            for s in r.get('segments', [])
            if s.get('match') == 'continuous_estimate'
        )

        print(f"\n{diff}:")
        print(f"  Items: {len(diff_items)}")
        print(f"  Segments: {total_segments}")
        print(f"  Exact matches: {exact_matches} ({100*exact_matches/max(1,total_segments):.1f}%)")
        print(f"  Close matches (+/-1): {close_matches} ({100*close_matches/max(1,total_segments):.1f}%)")
        if continuous_estimates > 0:
            print(f"  Continuous estimates: {continuous_estimates}")

    # Results table
    print(f"\n{'='*70}")
    print(f"RESULTS TABLE")
    print(f"{'='*70}")
    print(f"{'Food':<25} {'Diff':<6} {'GT':<8} {'Predicted':<20} {'Match':<12}")
    print("-" * 75)

    for r in results:
        food = (r.get('food_name') or '')[:24]
        diff = r.get('difficulty', '?')[:5]
        gt = r.get('total_ground_truth')
        gt_str = str(gt) if gt is not None else '-'

        # Get prediction info from segments
        segments = r.get('segments', [])
        if segments:
            first_seg = segments[0]
            cat = first_seg.get('quantity_category', 'unknown')
            if cat == 'discrete':
                pred = r.get('total_predicted')
                pred_str = str(pred) if pred is not None else '-'
            elif cat == 'continuous':
                pred_str = first_seg.get('predicted_amount', 'continuous')[:18]
            else:
                pred_str = 'unknown'
        else:
            pred_str = '-'

        # Determine overall match
        pred_count = r.get('total_predicted')
        if segments and segments[0].get('quantity_category') == 'continuous':
            match = 'continuous'
        elif gt is None and pred_count is None:
            match = 'uncountable'
        elif gt is None or pred_count is None:
            match = 'n/a'
        elif gt == pred_count:
            match = 'EXACT'
        elif abs(gt - pred_count) <= 1:
            match = 'close'
        else:
            match = f'off by {pred_count - gt}'

        print(f"{food:<25} {diff:<6} {gt_str:<8} {pred_str:<20} {match:<12}")


if __name__ == '__main__':
    main()

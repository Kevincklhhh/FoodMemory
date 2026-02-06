#!/usr/bin/env python3
"""
07d_vlm_frame_only.py - Frame-Only VLM Counting (No Video Clip)

Uses pre-selected evidence frames from a previous 07c run (hybrid/multipath)
to query VLM with exactly 4 labeled images: source_before, source_after,
dest_before, dest_after. This isolates the VLM's counting ability from
its frame-selection ability.

Usage:
    python 07d_vlm_frame_only.py --source-tag hybrid_gemini3_batch_low --tag frame_only_test --participant P03
    python 07d_vlm_frame_only.py --source-tag hybrid_gemini3_batch_low --tag frame_only_v1 --all --model gpt5
    python 07d_vlm_frame_only.py --source-tag hybrid_gemini3_batch_low --tag frame_only_v1 --all --skip-existing

Inputs:
    {participant}_vlm_qa_{source_tag}_results.json  (evidence frames from 07c)
    Video files from data/HD-EPIC/Videos/{participant}/

Outputs:
    {participant}_vlm_qa_{tag}_results.json  (visualizer-compatible)
"""

import argparse
import base64
import json
import os
import re
import time
from pathlib import Path
from typing import Dict, Optional, Any, List

import cv2
import requests

from dotenv import load_dotenv
from openai import AzureOpenAI
from google import genai
from google.genai import types

from inventory_utils import DEFAULT_OUTPUT_DIR

# Load environment variables
load_dotenv(Path(__file__).parent.parent.parent / ".env")
load_dotenv(Path(__file__).parent.parent / ".env", override=True)

# Paths
_SCRIPT_DIR = Path(__file__).parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
VIDEO_BASE_DIR = _PROJECT_ROOT / "data" / "HD-EPIC" / "Videos"

# Qwen3-VL API endpoint
QWEN3VL_URL = "http://saltyfish.eecs.umich.edu:8000/v1/chat/completions"
QWEN_MODEL = "Qwen/Qwen3-VL-30B-A3B-Instruct"

# Evidence frame roles we need (4 images)
REQUIRED_ROLES = ['source_before', 'source_after', 'dest_before', 'dest_after']

# ============================================================
# PROMPT
# ============================================================

FRAME_ONLY_PROMPT = """You are a Visual Inventory Auditor.
I have extracted 4 specific "Evidence Frames" from a cooking video to help you audit the usage of a specific food item.

**INPUT CONTEXT:**
- **Target Item:** "{item_name}"
- **Image 1 (Source Start):** The source container *before* any action.
- **Image 2 (Source End):** The source container *after* the action is complete.
- **Image 3 (Dest Start):** The destination *before* items arrive.
- **Image 4 (Dest End):** The destination *after* items have settled.

**VISUAL REASONING STEPS (Follow this "Triangulation" logic):**

1.  **Analyze Path A (Source Subtraction):**
    * Compare **Image 1** vs. **Image 2**.
    * Can you see inside the container? Is it the same container?
    * Attempt to count items in both. Calculate: `Start_Count - End_Count`.
    * *Failure Check:* If the container is opaque (e.g., flour bag) or the view is blocked by a hand/lid, mark this path as `INVALID`.

2.  **Analyze Path B (Destination Addition):**
    * Compare **Image 3** vs. **Image 4**.
    * Can you see the new items added?
    * Attempt to count the new items. Calculate: `End_Count - Start_Count`.
    * *Failure Check:* If the items are submerged, piled up, or hidden, mark this path as `INVALID`.

3.  **Synthesize & Resolve:**
    * Compare the deltas from Path A and Path B.
    * **Priority Rule:**
        * If **Source** is clear and countable (e.g., eggs in carton), it is usually most reliable.
        * If **Source** is opaque, you MUST rely on **Destination**.
        * If **Destination** is a messy pile, you MUST rely on **Source**.
        * If both are valid but disagree, choose the one with the clearer, unobstructed view.

**OUTPUT SCHEMA (Strict JSON):**
{{
  "item_name": "{item_name}",
  "quantity_category": "discrete" | "continuous" | "unknown",

  // --- PART 1: EVIDENCE LOGS (Analysis of the Image Pairs) ---
  "path_source": {{
    "status": "VALID" | "INVALID_OCCLUDED" | "INVALID_OPAQUE" | "INVALID_BLURRY",
    "container_description": "Describe what you see in Img 1 & 2 (e.g. 'Clear view of egg carton')",
    "observed_start_count": <number or null>,
    "observed_end_count": <number or null>,
    "calculated_delta": <number or null>,
    "confidence": "high" | "medium" | "low"
  }},

  "path_destination": {{
    "status": "VALID" | "INVALID_OCCLUDED" | "INVALID_PILE" | "INVALID_BLURRY",
    "container_description": "Describe what you see in Img 3 & 4 (e.g. 'Frying pan on stove')",
    "observed_start_count": <number or null>,
    "observed_end_count": <number or null>,
    "calculated_delta": <number or null>,
    "confidence": "high" | "medium" | "low"
  }},

  // --- PART 2: FINAL ESTIMATION ---
  // AMOUNT REMOVED (The usage):
  "numeric_count": <integer or null>,      // Final estimated usage count
  "amount_description": <string or null>,  // Use for continuous items (e.g. "approx 1/2 cup")
  "volume_fraction": <float or null>,      // Fraction of container removed (0.0-1.0)

  // AMOUNT REMAINING (In Source):
  "remaining_count": <integer or null>,       // Estimated items left in Source (from Img 2)
  "remaining_description": <string or null>,  // Description (if continuous)
  "remaining_fraction": <float or null>,      // Fraction remaining (0.0-1.0)

  "unit_type": "unit" | "scoop" | "cup" | "splash" | "pinch" | "slice",
  "confidence": "high" | "medium" | "low",

  "visual_evidence": "Summarize decision. e.g. 'Source view was blocked (Invalid). Dest view clearly showed 2 distinct items added. Using Path B count.'"
}}

**EXAMPLES:**

*Example 1 (Discrete - Source Clear):*
{{
  "item_name": "eggs",
  "quantity_category": "discrete",
  "path_source": {{
    "status": "VALID", "container_description": "White egg carton, clearly visible",
    "observed_start_count": 6, "observed_end_count": 4, "calculated_delta": 2, "confidence": "high"
  }},
  "path_destination": {{
    "status": "INVALID_OCCLUDED", "container_description": "Pan partially blocked by hand",
    "observed_start_count": null, "observed_end_count": null, "calculated_delta": null, "confidence": "low"
  }},
  "numeric_count": 2, "amount_description": null, "volume_fraction": null,
  "remaining_count": 4, "remaining_description": null, "remaining_fraction": null,
  "unit_type": "unit", "confidence": "high",
  "visual_evidence": "Source carton clearly showed 6->4 eggs. Dest blocked by hand. Trusting Source Path."
}}

Return ONLY the raw JSON string. Do not use Markdown.
"""


# ============================================================
# VLM CLIENTS (reused from 07c)
# ============================================================

class QwenClient:
    """Handles communication with Qwen VLM API (image input)."""

    def query(
        self,
        system_prompt: str,
        user_prompt: str,
        frames_b64: List[str],
        max_tokens: int = 2000,
        temperature: float = 0.3,
    ) -> str:
        messages = [{"role": "system", "content": system_prompt}]
        user_content = []

        for fb64 in frames_b64:
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{fb64}"}
            })

        user_content.append({"type": "text", "text": user_prompt})
        messages.append({"role": "user", "content": user_content})

        data = {
            "model": QWEN_MODEL,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        try:
            response = requests.post(QWEN3VL_URL, json=data, timeout=300,
                                     headers={"Content-Type": "application/json"})
            response.raise_for_status()
            result = response.json()
            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["message"]["content"]
            return ""
        except Exception as e:
            print(f"  ERROR: Qwen API Error: {e}")
            return ""


class GPT5Client:
    """Handles communication with Azure OpenAI GPT-5.2 API (image input)."""

    def __init__(self, reasoning_effort: str = "high"):
        self.model = "gpt-5.2"
        self.reasoning_effort = reasoning_effort

        api_key = os.getenv("AZURE_OPENAI_API_KEY_2")
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT_2", "").strip()
        if not api_key or not endpoint:
            raise ValueError("Missing Azure OpenAI API credentials for GPT-5.2.")

        self.client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version="2025-03-01-preview",
        )

    def query(
        self,
        system_prompt: str,
        user_prompt: str,
        frames_b64: List[str],
        max_tokens: int = 16000,
        temperature: float = 0.3,
    ) -> str:
        messages = [{"role": "system", "content": system_prompt}]
        user_content = []

        for fb64 in frames_b64:
            user_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{fb64}",
                    "detail": "high"
                }
            })

        user_content.append({"type": "text", "text": user_prompt})
        messages.append({"role": "user", "content": user_content})

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                reasoning_effort=self.reasoning_effort,
            )
            if response.choices and len(response.choices) > 0:
                return response.choices[0].message.content
            return ""
        except Exception as e:
            print(f"  ERROR: GPT-5.2 API Error: {e}")
            return ""


class GeminiClient:
    """Handles communication with Google Gemini API (image input)."""

    def __init__(self, model_name: str = "gemini-3-flash-preview"):
        self.model_name = model_name

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Missing GOOGLE_API_KEY.")

        self.client = genai.Client(api_key=api_key)

    def query(
        self,
        system_prompt: str,
        user_prompt: str,
        frames_b64: List[str],
        max_tokens: int = 8192,
        temperature: float = 0.3,
    ) -> str:
        try:
            contents = []
            for fb64 in frames_b64:
                contents.append(types.Part.from_bytes(
                    data=base64.b64decode(fb64),
                    mime_type="image/jpeg"
                ))

            full_prompt = f"{system_prompt}\n\n{user_prompt}"
            contents.append(full_prompt)

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                )
            )

            if response.text:
                return response.text
            return ""
        except Exception as e:
            print(f"  ERROR: Gemini API Error: {e}")
            return ""


# ============================================================
# FRAME EXTRACTION
# ============================================================

def extract_frame_at_timestamp(video_path: Path, timestamp_sec: float) -> Optional[str]:
    """
    Extract a single frame from video at the given absolute timestamp.

    Returns:
        Base64-encoded JPEG string, or None on failure.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    # Seek to timestamp
    cap.set(cv2.CAP_PROP_POS_MSEC, timestamp_sec * 1000.0)
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        return None

    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buffer).decode('utf-8')


def get_evidence_frame_map(evidence_frames: List[Dict]) -> Dict[str, Dict]:
    """
    Build a role -> frame_info map from evidence_frames list.
    Only keeps the first occurrence of each required role.
    """
    frame_map = {}
    for ef in evidence_frames:
        role = ef.get('role', '')
        if role in REQUIRED_ROLES and role not in frame_map:
            frame_map[role] = ef
    return frame_map


# ============================================================
# RESPONSE PARSING
# ============================================================

def _extract_json(response_text: str) -> Optional[Dict]:
    """Extract JSON object from VLM response text."""
    code_block_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response_text)
    if code_block_match:
        try:
            return json.loads(code_block_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    json_match = re.search(r'\{[\s\S]*\}', response_text)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass

    return None


def parse_frame_only_response(response_text: str) -> Optional[Dict[str, Any]]:
    """
    Parse frame-only JSON from VLM response.

    Returns a normalized dict compatible with the visualizer, or None on failure.
    """
    result = _extract_json(response_text)
    if not result:
        return None

    path_source = result.get('path_source')
    path_dest = result.get('path_destination')
    if not path_source or not path_dest:
        return None

    parsed = {
        'item_name': result.get('item_name', ''),
        'quantity_category': result.get('quantity_category', 'unknown'),

        # Path evidence (richer than hybrid: includes start/end counts)
        'path_source': {
            'status': path_source.get('status', 'UNKNOWN'),
            'container_description': path_source.get('container_description', ''),
            'observed_start_count': path_source.get('observed_start_count'),
            'observed_end_count': path_source.get('observed_end_count'),
            'observed_delta': path_source.get('calculated_delta'),
            'confidence': path_source.get('confidence', 'low'),
        },
        'path_destination': {
            'status': path_dest.get('status', 'UNKNOWN'),
            'container_description': path_dest.get('container_description', ''),
            'observed_start_count': path_dest.get('observed_start_count'),
            'observed_end_count': path_dest.get('observed_end_count'),
            'observed_delta': path_dest.get('calculated_delta'),
            'confidence': path_dest.get('confidence', 'low'),
        },

        # QA fields
        'numeric_count': result.get('numeric_count'),
        'amount_description': result.get('amount_description'),
        'volume_fraction': result.get('volume_fraction'),
        'remaining_count': result.get('remaining_count'),
        'remaining_description': result.get('remaining_description'),
        'remaining_fraction': result.get('remaining_fraction'),
        'unit_type': result.get('unit_type'),
        'confidence': result.get('confidence', 'low'),
        'visual_evidence': result.get('visual_evidence', ''),
    }

    return parsed


# ============================================================
# EVALUATION (same as 07c)
# ============================================================

def evaluate_result(predicted: Dict, ground_truth: Dict) -> Dict[str, Any]:
    """Compare predicted result with ground truth."""
    gt_count = ground_truth.get('total_count')
    pred_category = predicted.get('quantity_category', 'unknown')
    pred_count = predicted.get('numeric_count')

    result = {
        'predicted_count': pred_count,
        'predicted_amount': predicted.get('amount_description'),
        'predicted_unit': predicted.get('unit_type'),
        'ground_truth_count': gt_count,
        'ground_truth_unit': ground_truth.get('count_unit'),
    }

    if pred_category == 'unknown' and pred_count is not None:
        pred_category = 'discrete'

    if pred_category == 'discrete' and pred_count is not None:
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
        result['match'] = 'continuous_estimate'
        result['error'] = None
    else:
        if gt_count is None:
            result['match'] = 'both_unknown'
        else:
            result['match'] = 'pred_unknown'
        result['error'] = None

    return result


# ============================================================
# BUILD VISUALIZER-COMPATIBLE EVIDENCE FRAMES
# ============================================================

def build_evidence_frames_from_source(source_frame_map: Dict[str, Dict]) -> List[Dict]:
    """
    Build evidence_frames list from the source results' frame map,
    preserving the absolute timestamps from the original 07c run.
    """
    frames = []
    for role in REQUIRED_ROLES:
        ef = source_frame_map.get(role)
        if ef:
            frames.append({
                'role': role,
                'absolute_timestamp': ef.get('absolute_timestamp'),
                'timestamp_raw': ef.get('timestamp_raw', ''),
                'timestamp_seconds': ef.get('timestamp_seconds'),
                'description': ef.get('description', ''),
                'visibility_status': ef.get('visibility_status', 'unknown'),
                'container_description': ef.get('container_description', ''),
            })
    return frames


# ============================================================
# PROCESSING
# ============================================================

def find_participants_with_source(output_dir: Path, source_tag: str) -> List[str]:
    """Find all participants that have source results files."""
    participants = []
    for participant_dir in sorted(output_dir.iterdir()):
        if not participant_dir.is_dir():
            continue
        participant = participant_dir.name
        source_file = participant_dir / f"{participant}_vlm_qa_{source_tag}_results.json"
        if source_file.exists():
            participants.append(participant)
    return participants


def process_participant(
    participant: str,
    output_dir: Path,
    source_tag: str,
    tag: str,
    model: str,
    test_limit: int = 0,
    verbose: bool = False,
    skip_existing: bool = False,
    reasoning_effort: str = "high",
) -> Optional[Path]:
    """Process a single participant using pre-selected evidence frames."""
    participant_dir = output_dir / participant

    # Load source results (from 07c)
    source_file = participant_dir / f"{participant}_vlm_qa_{source_tag}_results.json"
    if not source_file.exists():
        print(f"SKIP {participant}: source file {source_file.name} not found")
        return None

    output_file = participant_dir / f"{participant}_vlm_qa_{tag}_results.json"
    if skip_existing and output_file.exists():
        print(f"SKIP {participant}: {output_file.name} already exists")
        return output_file

    with open(source_file, 'r') as f:
        source_data = json.load(f)

    source_items = source_data.get('items', [])
    print(f"\n{'='*70}")
    print(f"PARTICIPANT: {participant}")
    print(f"{'='*70}")
    print(f"Source: {source_file.name} ({len(source_items)} items)")

    if test_limit > 0:
        source_items = source_items[:test_limit]
        print(f"TEST MODE: Processing only first {len(source_items)} items")

    # Initialize VLM client
    print(f"Initializing {model} VLM client...")
    if model == 'gpt5':
        vlm = GPT5Client(reasoning_effort=reasoning_effort)
    elif model == 'gemini':
        vlm = GeminiClient()
    else:
        vlm = QwenClient()

    # Process items
    results = []
    total_segs = 0
    total_ok = 0
    total_skip = 0

    for i, source_item in enumerate(source_items):
        narration_id = source_item.get('narration_id', '')
        food_name = source_item.get('food_name', 'unknown')
        difficulty = source_item.get('difficulty', 'UNKNOWN')
        video_range = source_item.get('video_range', [])
        source_segments = source_item.get('segments', [])

        print(f"\n[{i+1}/{len(source_items)}] {food_name} ({difficulty})")
        print(f"  Narration ID: {narration_id}")
        print(f"  Source segments: {len(source_segments)}")

        segment_results = []
        for seg_idx, source_seg in enumerate(source_segments):
            total_segs += 1
            video_id = source_seg.get('video_id')
            start_ts = source_seg.get('start_timestamp', 0)
            end_ts = source_seg.get('end_timestamp', 0)
            segment_id = source_seg.get('segment_id', '')
            gt_count = source_seg.get('ground_truth_count')
            gt_unit = source_seg.get('ground_truth_unit')

            # Get evidence frames from source
            evidence_frames = source_seg.get('evidence_frames', [])
            frame_map = get_evidence_frame_map(evidence_frames)

            missing_roles = [r for r in REQUIRED_ROLES if r not in frame_map]
            if missing_roles:
                print(f"    Seg {seg_idx}: SKIP - missing roles: {missing_roles}")
                total_skip += 1
                segment_results.append({
                    'segment_id': segment_id,
                    'segment_idx': seg_idx,
                    'video_id': video_id,
                    'start_timestamp': start_ts,
                    'end_timestamp': end_ts,
                    'ground_truth_count': gt_count,
                    'ground_truth_unit': gt_unit,
                    'error': f'Missing evidence frame roles: {missing_roles}',
                })
                continue

            if not video_id:
                print(f"    Seg {seg_idx}: SKIP - no video_id")
                total_skip += 1
                continue

            # Find video file
            video_path = VIDEO_BASE_DIR / participant / f"{video_id}.mp4"
            if not video_path.exists():
                print(f"    Seg {seg_idx}: SKIP - video not found: {video_path.name}")
                total_skip += 1
                segment_results.append({
                    'segment_id': segment_id,
                    'segment_idx': seg_idx,
                    'video_id': video_id,
                    'start_timestamp': start_ts,
                    'end_timestamp': end_ts,
                    'ground_truth_count': gt_count,
                    'ground_truth_unit': gt_unit,
                    'error': f'Video not found: {video_path.name}',
                })
                continue

            # Extract 4 frames from video at the pre-selected timestamps
            print(f"    Seg {seg_idx}: Extracting 4 evidence frames...", end=" ", flush=True)
            frames_b64 = []
            frame_labels = []
            extraction_ok = True

            for role in REQUIRED_ROLES:
                ef = frame_map[role]
                abs_ts = ef.get('absolute_timestamp')
                if abs_ts is None:
                    print(f"NO TIMESTAMP for {role}")
                    extraction_ok = False
                    break

                frame_b64 = extract_frame_at_timestamp(video_path, abs_ts)
                if frame_b64 is None:
                    print(f"FAILED to extract {role} @ {abs_ts:.1f}s")
                    extraction_ok = False
                    break

                frames_b64.append(frame_b64)
                frame_labels.append(f"{role} @ {abs_ts:.1f}s")

            if not extraction_ok or len(frames_b64) != 4:
                print("EXTRACTION FAILED")
                total_skip += 1
                segment_results.append({
                    'segment_id': segment_id,
                    'segment_idx': seg_idx,
                    'video_id': video_id,
                    'start_timestamp': start_ts,
                    'end_timestamp': end_ts,
                    'ground_truth_count': gt_count,
                    'ground_truth_unit': gt_unit,
                    'error': 'Failed to extract evidence frames from video',
                })
                continue

            print(f"OK ({', '.join(frame_labels)})")

            # Build prompt
            item_name = source_seg.get('item_name', food_name)
            prompt = FRAME_ONLY_PROMPT.format(item_name=item_name)

            # Query VLM
            print(f"    Querying {model}...", end=" ", flush=True)
            response = vlm.query(
                system_prompt="You are a Visual Inventory Auditor analyzing cooking images.",
                user_prompt=prompt,
                frames_b64=frames_b64,
            )

            if not response:
                print("NO RESPONSE")
                segment_results.append({
                    'segment_id': segment_id,
                    'segment_idx': seg_idx,
                    'video_id': video_id,
                    'start_timestamp': start_ts,
                    'end_timestamp': end_ts,
                    'ground_truth_count': gt_count,
                    'ground_truth_unit': gt_unit,
                    'error': 'No VLM response',
                })
                continue

            # Parse response
            parsed = parse_frame_only_response(response)
            if not parsed:
                print("PARSE FAILED")
                if verbose:
                    print(f"      Raw: {response[:300]}...")
                segment_results.append({
                    'segment_id': segment_id,
                    'segment_idx': seg_idx,
                    'video_id': video_id,
                    'start_timestamp': start_ts,
                    'end_timestamp': end_ts,
                    'ground_truth_count': gt_count,
                    'ground_truth_unit': gt_unit,
                    'error': 'Failed to parse frame-only response',
                    'raw_vlm_response': response,
                })
                continue

            total_ok += 1

            # Build evidence_frames for visualizer (reuse source timestamps)
            vis_evidence_frames = build_evidence_frames_from_source(frame_map)

            # Update evidence frame descriptions from VLM response
            for vef in vis_evidence_frames:
                role = vef['role']
                if role.startswith('source'):
                    vef['container_description'] = parsed['path_source'].get('container_description', '')
                    vef['visibility_status'] = (
                        'clear_view' if parsed['path_source']['status'] == 'VALID' else 'occluded'
                    )
                elif role.startswith('dest'):
                    vef['container_description'] = parsed['path_destination'].get('container_description', '')
                    vef['visibility_status'] = (
                        'clear_view' if parsed['path_destination']['status'] == 'VALID' else 'occluded'
                    )

            # Evaluate against ground truth
            eval_result = {}
            if gt_count is not None:
                eval_result = evaluate_result(parsed, {
                    'total_count': gt_count,
                    'count_unit': gt_unit,
                })

            pred_count = parsed.get('numeric_count')
            match_str = eval_result.get('match', '')
            print(f"OK  pred={pred_count}  gt={gt_count}  [{match_str}]")
            print(f"      src={parsed['path_source']['status']}  "
                  f"dst={parsed['path_destination']['status']}  "
                  f"conf={parsed.get('confidence')}")

            # Build segment result (visualizer-compatible)
            seg_result = {
                'segment_id': segment_id,
                'segment_idx': seg_idx,
                'video_id': video_id,
                'start_timestamp': start_ts,
                'end_timestamp': end_ts,
                'clip_start': source_seg.get('clip_start'),
                'item_name': parsed.get('item_name', ''),
                'evidence_frames': vis_evidence_frames,
                'raw_vlm_response': response,
                # QA fields
                'quantity_category': parsed.get('quantity_category'),
                'predicted_count': parsed.get('numeric_count'),
                'predicted_amount': parsed.get('amount_description'),
                'predicted_unit': parsed.get('unit_type'),
                'remaining_count': parsed.get('remaining_count'),
                'remaining_description': parsed.get('remaining_description'),
                'remaining_fraction': parsed.get('remaining_fraction'),
                'confidence': parsed.get('confidence'),
                'visual_evidence': parsed.get('visual_evidence'),
                # Ground truth (carried from source)
                'ground_truth_count': gt_count,
                'ground_truth_unit': gt_unit,
                # Paths (for visualizer)
                'paths': {
                    'source': parsed['path_source'],
                    'destination': parsed['path_destination'],
                },
            }

            # Add evaluation fields
            if eval_result:
                seg_result['match'] = eval_result.get('match')
                seg_result['error'] = eval_result.get('error')
                seg_result['abs_error'] = eval_result.get('abs_error')

            segment_results.append(seg_result)

        # Aggregate item result
        total_predicted = sum(
            s.get('predicted_count', 0) or 0
            for s in segment_results
            if s.get('predicted_count') is not None
        )

        item_result = {
            'narration_id': narration_id,
            'food_name': food_name,
            'difficulty': difficulty,
            'video_range': video_range,
            'total_ground_truth': source_item.get('total_ground_truth'),
            'total_ground_truth_unit': source_item.get('total_ground_truth_unit'),
            'num_segments': len(source_segments),
            'segments': segment_results,
            'total_predicted': total_predicted if total_predicted > 0 else None,
            'recipe_amount': source_item.get('recipe_amount'),
        }

        results.append(item_result)

    # Save results
    print(f"\n{'='*70}")
    print(f"SAVING RESULTS")
    print(f"{'='*70}")

    output_data = {
        'participant': participant,
        'model': model,
        'tag': tag,
        'source_tag': source_tag,
        'prompt_mode': 'frame_only',
        'task': 'frame_only_counting',
        'low_only': source_data.get('low_only', False),
        'total_items': len(source_items),
        'items': results,
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)

    print(f"  Saved: {output_file.name}")
    print(f"  Total segments: {total_segs}")
    print(f"  Processed OK: {total_ok}")
    print(f"  Skipped: {total_skip}")

    # Summary table
    print(f"\n{'Food':<30} {'Diff':<6} {'Segs':<5} {'Pred':<8} {'GT':<8}")
    print("-" * 60)

    for r in results:
        food = (r.get('food_name') or '')[:29]
        diff = (r.get('difficulty') or '?')[:5]
        n_segs = len(r.get('segments', []))
        pred = r.get('total_predicted', '?')
        gt = r.get('total_ground_truth', '?')
        print(f"{food:<30} {diff:<6} {n_segs:<5} {str(pred):<8} {str(gt):<8}")

    return output_file


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Frame-Only VLM Counting using pre-selected evidence frames"
    )
    parser.add_argument('--participant', help='Participant ID (e.g., P03)')
    parser.add_argument('--all', action='store_true', help='Process all participants')
    parser.add_argument('--source-tag', required=True,
                        help='Tag of source results (from 07c) to get evidence frames')
    parser.add_argument('--tag', required=True,
                        help='Tag for this run (used in output filename)')
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--test', type=int, default=0,
                        help='Process first N items only')
    parser.add_argument(
        '--model', default='qwen',
        choices=['qwen', 'gpt5', 'gemini'],
        help='VLM model to use for counting'
    )
    parser.add_argument('--reasoning', default='medium',
                        choices=['low', 'medium', 'high'],
                        help='Reasoning effort for GPT-5')
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--skip-existing', action='store_true',
                        help='Skip if result file exists')

    args = parser.parse_args()

    if not args.participant and not args.all:
        parser.error("Either --participant or --all must be specified")

    if args.all:
        participants = find_participants_with_source(args.output_dir, args.source_tag)
        if not participants:
            print(f"No participants with source tag '{args.source_tag}' found in {args.output_dir}")
            return 1
        print(f"Found {len(participants)} participants: {', '.join(participants)}")
    else:
        participants = [args.participant]

    result_files = []
    for p in participants:
        result_file = process_participant(
            participant=p,
            output_dir=args.output_dir,
            source_tag=args.source_tag,
            tag=args.tag,
            model=args.model,
            test_limit=args.test,
            verbose=args.verbose,
            skip_existing=args.skip_existing,
            reasoning_effort=args.reasoning,
        )
        if result_file:
            result_files.append(result_file)

    print(f"\n{'='*70}")
    print("COMPLETE")
    print(f"{'='*70}")
    print(f"Processed {len(result_files)} participant(s)")
    for f in result_files:
        print(f"  - {f}")

    return 0


if __name__ == '__main__':
    exit(main())

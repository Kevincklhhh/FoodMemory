#!/usr/bin/env python3
"""
07b_vlm_baseline_QA.py - Baseline VLM Q&A using fixed 30-second video blocks

This baseline approach:
1. Divides videos into fixed 30-second blocks
2. For each dispensal segment, finds which block(s) it overlaps with
3. Runs VLM on those blocks using the same prompt as blind mode (with food items list)
4. Compares detected item against ground truth using semantic similarity

This serves as a baseline comparison to the targeted approach in 07_vlm_QA.py.

Usage:
    python 07b_vlm_baseline_QA.py --participant P03 --tag baseline_v1 --test 5
    python 07b_vlm_baseline_QA.py --all --tag baseline_v1
"""

import argparse
import base64
import json
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Any, List, Tuple

import requests
from dotenv import load_dotenv
from openai import AzureOpenAI
from sentence_transformers import SentenceTransformer

from inventory_utils import DEFAULT_OUTPUT_DIR, generate_segment_id

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

# Semantic similarity
_SENTENCE_MODEL = None
SEMANTIC_SIMILARITY_THRESHOLD = 0.8


def get_sentence_model():
    """Lazy load the sentence transformer model."""
    global _SENTENCE_MODEL
    if _SENTENCE_MODEL is None:
        print("Loading SentenceTransformer model...")
        _SENTENCE_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
    return _SENTENCE_MODEL


def compute_semantic_similarity(text1: str, text2: str) -> float:
    """Compute cosine similarity between two texts using SentenceTransformer."""
    model = get_sentence_model()
    embeddings = model.encode([text1, text2], convert_to_tensor=True)
    similarity = float(embeddings[0] @ embeddings[1])
    return similarity


def check_item_match(detected: str, expected: str, threshold: float = SEMANTIC_SIMILARITY_THRESHOLD) -> Tuple[bool, float]:
    """Check if detected item matches expected item using semantic similarity."""
    if not detected or not expected:
        return False, 0.0

    detected_norm = detected.lower().strip()
    expected_norm = expected.lower().strip()

    if detected_norm == expected_norm:
        return True, 1.0

    similarity = compute_semantic_similarity(detected_norm, expected_norm)
    return similarity >= threshold, similarity


# Same prompt as blind mode in 07_vlm_QA.py
BASELINE_PROMPT = """You are a Visual Inventory Auditor for kitchen tasks.
Your goal is to identify which food item is being dispensed and meticulously track its flow from source to destination.

**POSSIBLE FOOD ITEMS (select from this list):**
{food_items_list}

**INPUT:**
- Input Video: Egocentric view of a dispensing action.

**VISUAL REASONING STEPS (Follow these mentally before answering):**
1. **Identify the Food Item:** Which item from the list above is being taken from a container/package? Use the EXACT name from the list.
2. **Identify Source State:** Look at the container *before* the action starts. Is it full? Can you count the items?
3. **Analyze the Transfer:** Watch the user's hand.
   - What exactly is currently being held or moved?
   - Is the view obstructed by the hand? If so, look at the *destination* to verify what was put down.
4. **Identify End State:** Look at the container *after* the hand leaves. What is left?

**OUTPUT TASK:**
Estimate strictly:
1. **Item Detected:** Which food item from the list is being dispensed.
2. **Quantity Removed:** The amount transferred out.
3. **Quantity Remaining:** The amount left in the original source container.

**OUTPUT SCHEMA (Strict JSON):**
{{
  "detected_item_name": <string or null>,  // Which food item from the list was dispensed? Use EXACT name from list. null if none/unclear
  "quantity_category": "discrete" | "continuous" | "unknown",

  // AMOUNT REMOVED:
  "numeric_count": <integer or null>,      // Items removed (if discrete). e.g., 2
  "amount_description": <string or null>,  // Description (if continuous). e.g., "about half a cup"
  "volume_fraction": <float or null>,      // Fraction of container removed (0.0-1.0)

  // AMOUNT REMAINING:
  "remaining_count": <integer or null>,       // Items left (if discrete). e.g., 4
  "remaining_description": <string or null>,  // Description (if continuous). e.g., "about 3/4 full"
  "remaining_fraction": <float or null>,      // Fraction remaining (0.0-1.0)

  "unit_type": "unit" | "scoop" | "cup" | "splash" | "pinch" | "slice",
  "confidence": "high" | "medium" | "low",
  "visual_evidence": "Describe the Start State, the Action, and the End State."
}}

**EXAMPLES:**

*Example 1 (Discrete - Countable):*
{{
  "detected_item_name": "eggs",
  "quantity_category": "discrete",
  "numeric_count": 2,
  "amount_description": null,
  "volume_fraction": null,
  "remaining_count": 4,
  "remaining_description": null,
  "remaining_fraction": null,
  "unit_type": "unit",
  "confidence": "high",
  "visual_evidence": "Carton started with 6 eggs. Hand blocked view during pick up. After hand left, 4 eggs remaining in carton."
}}

*Example 2 (Continuous):*
{{
  "detected_item_name": "milk",
  "quantity_category": "continuous",
  "numeric_count": null,
  "amount_description": "about half a cup",
  "volume_fraction": 0.1,
  "remaining_count": null,
  "remaining_description": "about 3/4 of carton",
  "remaining_fraction": 0.75,
  "unit_type": "cup",
  "confidence": "medium",
  "visual_evidence": "Steady pour for 2 seconds. Carton appears mostly full after pouring."
}}

Return ONLY the raw JSON string. Do not use Markdown.
"""

# Text-only aggregation prompt for segments with multiple blocks
AGGREGATION_PROMPT = """You are a Visual Inventory Auditor. You have received detection results from multiple 30-second video blocks that span a single food dispensing event.

**POSSIBLE FOOD ITEMS (select from this list):**
{food_items_list}

**BLOCK DETECTION RESULTS:**
{block_results}

**YOUR TASK:**
Analyze these block results and determine the FINAL aggregated answer for this dispensing event.
- If multiple blocks detected the same item, combine the counts if appropriate
- If blocks detected different items, determine which is most likely correct based on confidence
- If a block detected "null" or nothing, it may not have contained the actual dispensing action

**OUTPUT SCHEMA (Strict JSON):**
{{
  "detected_item_name": <string or null>,  // Final item from the list. null if no clear detection
  "quantity_category": "discrete" | "continuous" | "unknown",
  "numeric_count": <integer or null>,  // Final aggregated count
  "amount_description": <string or null>,
  "unit_type": "unit" | "scoop" | "cup" | "splash" | "pinch" | "slice",
  "confidence": "high" | "medium" | "low",
  "aggregation_reasoning": "Explain how you combined/chose from the block results."
}}

Return ONLY the raw JSON string. Do not use Markdown.
"""

# Detection prompt for full-scan mode: first check if any dispensal is happening
DETECTION_PROMPT = """You are a Visual Inventory Auditor for kitchen tasks.
Your goal is to detect if a FOOD DISPENSAL ACTION is happening in this video segment.

**POSSIBLE FOOD ITEMS TO DETECT (select from this list only):**
{food_items_list}

**WHAT COUNTS AS A DISPENSAL ACTION:**
A dispensal is the moment when quantity leaves a Source Container. Look for:
- **Pouring/Scooping:** Liquid or granular items being poured or scooped from container to destination
- **Dispensing Cuts:** Cutting a portion OFF the main block (e.g., slicing butter, cutting cheese)
- **Scanning/Taking:** Taking discrete units from a multipack (e.g., taking eggs from carton, grabbing items from bag)

**WHAT IS NOT A DISPENSAL:**
- Just opening a container without removing anything
- Moving a closed container
- Handling food that's already been dispensed (cooking, stirring)
- Food that's not on the items list

**VISUAL CUES TO LOOK FOR:**
1. Hand reaching INTO a container/package
2. Items being lifted/poured OUT of a source
3. Quantity moving from one location to another
4. Container state changing (becoming emptier)

**OUTPUT SCHEMA (Strict JSON):**
{{
  "dispensal_detected": true | false,
  "detected_item_name": <string or null>,  // Which item from the list? null if no dispensal
  "action_type": "pouring" | "scooping" | "cutting" | "taking" | "none",
  "confidence": "high" | "medium" | "low",
  "evidence": "Brief description of what you observed."
}}

**EXAMPLES:**

*Example 1 (Dispensal detected):*
{{
  "dispensal_detected": true,
  "detected_item_name": "eggs",
  "action_type": "taking",
  "confidence": "high",
  "evidence": "Hand reaches into egg carton, removes 2 eggs, places on counter."
}}

*Example 2 (No dispensal):*
{{
  "dispensal_detected": false,
  "detected_item_name": null,
  "action_type": "none",
  "confidence": "high",
  "evidence": "Person is stirring pot on stove, no items being taken from containers."
}}

*Example 3 (Uncertain):*
{{
  "dispensal_detected": false,
  "detected_item_name": null,
  "action_type": "none",
  "confidence": "low",
  "evidence": "Hand partially visible near cabinet, unclear if anything is being removed."
}}

Return ONLY the raw JSON string. Do not use Markdown.
"""


def parse_detection_response(response_text: str) -> Dict[str, Any]:
    """Parse JSON from detection VLM response."""
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

    if result:
        return {
            'dispensal_detected': result.get('dispensal_detected', False),
            'detected_item_name': result.get('detected_item_name'),
            'action_type': result.get('action_type', 'none'),
            'confidence': result.get('confidence', 'low'),
            'evidence': result.get('evidence', ''),
        }

    return {
        'dispensal_detected': False,
        'detected_item_name': None,
        'action_type': 'none',
        'confidence': 'low',
        'evidence': f"Could not parse response: {response_text[:200]}"
    }


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

    def text_query(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.3
    ) -> str:
        """Text-only query (no video) to Qwen"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        data = {
            "model": QWEN_MODEL,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        headers = {"Content-Type": "application/json"}

        try:
            response = requests.post(QWEN3VL_URL, headers=headers, json=data, timeout=120)
            response.raise_for_status()
            result = response.json()
            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["message"]["content"]
            return ""
        except Exception as e:
            print(f"  ERROR: Qwen text query Error: {e}")
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

    def text_query(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.3
    ) -> str:
        """Text-only query (no frames) to GPT-4o"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

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
            print(f"  ERROR: GPT-4o text query Error: {e}")
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
    """Extract a video block using ffmpeg"""
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
    """Get the block indices and time ranges that overlap with a given segment."""
    blocks = []
    first_block = int(start_time // block_duration)
    last_block = int(end_time // block_duration)

    for block_idx in range(first_block, last_block + 1):
        block_start = block_idx * block_duration
        block_end = (block_idx + 1) * block_duration
        blocks.append((block_idx, block_start, block_end))

    return blocks


def parse_vlm_response(response_text: str) -> Dict[str, Any]:
    """
    Parse JSON from VLM response.
    Same as 07_vlm_QA.py to ensure consistent output format.
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

    if result:
        return {
            'detected_item_name': result.get('detected_item_name') or result.get('item_name'),
            'quantity_category': result.get('quantity_category', 'unknown'),
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

    return {
        'detected_item_name': None,
        'quantity_category': 'unknown',
        'numeric_count': None,
        'amount_description': None,
        'volume_fraction': None,
        'remaining_count': None,
        'remaining_description': None,
        'remaining_fraction': None,
        'unit_type': None,
        'confidence': 'low',
        'visual_evidence': f"Could not parse response: {response_text[:200]}"
    }


def format_block_results_for_aggregation(block_results: List[Dict]) -> str:
    """Format block results as text for the aggregation prompt."""
    lines = []
    for br in block_results:
        parsed = br.get('parsed')
        if not parsed:
            lines.append(f"Block {br['block_idx']} ({br['block_start']:.0f}s-{br['block_end']:.0f}s): No detection")
            continue

        detected = parsed.get('detected_item_name') or "null"
        count = parsed.get('numeric_count')
        category = parsed.get('quantity_category', 'unknown')
        confidence = parsed.get('confidence', 'low')
        evidence = parsed.get('visual_evidence', '')[:200]  # Truncate long evidence

        count_str = str(count) if count is not None else "null"
        lines.append(
            f"Block {br['block_idx']} ({br['block_start']:.0f}s-{br['block_end']:.0f}s):\n"
            f"  - Item: {detected}\n"
            f"  - Count: {count_str} ({category})\n"
            f"  - Confidence: {confidence}\n"
            f"  - Evidence: {evidence}"
        )

    return "\n\n".join(lines)


def parse_aggregation_response(response_text: str) -> Dict[str, Any]:
    """Parse JSON from aggregation VLM response."""
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

    if result:
        return {
            'detected_item_name': result.get('detected_item_name'),
            'quantity_category': result.get('quantity_category', 'unknown'),
            'numeric_count': result.get('numeric_count'),
            'amount_description': result.get('amount_description'),
            'unit_type': result.get('unit_type'),
            'confidence': result.get('confidence', 'low'),
            'aggregation_reasoning': result.get('aggregation_reasoning', ''),
        }

    return {
        'detected_item_name': None,
        'quantity_category': 'unknown',
        'numeric_count': None,
        'amount_description': None,
        'unit_type': None,
        'confidence': 'low',
        'aggregation_reasoning': f"Could not parse response: {response_text[:200]}"
    }


def evaluate_result(predicted: Dict, ground_truth: Dict, item_match: bool = None) -> Dict[str, Any]:
    """
    Compare predicted result with ground truth.

    Args:
        predicted: VLM prediction dict
        ground_truth: Ground truth dict with 'total_count'
        item_match: Whether detected item matches expected item.
                   If False, match is 'item_mismatch' regardless of count.
                   If None, item match check is skipped.
    """
    gt_count = ground_truth.get('total_count')
    pred_category = predicted.get('quantity_category', 'unknown')
    pred_count = predicted.get('numeric_count')

    result = {
        'quantity_category': pred_category,
        'predicted_count': pred_count,
        'ground_truth_count': gt_count,
    }

    # First check: item match
    # If item doesn't match, mark as item_mismatch regardless of count
    if item_match is False:
        result['match'] = 'item_mismatch'
        if gt_count is not None and pred_count is not None:
            result['error'] = pred_count - gt_count
            result['abs_error'] = abs(pred_count - gt_count)
        else:
            result['error'] = None
        return result

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
        result['match'] = 'pred_unknown' if gt_count is not None else 'both_unknown'
        result['error'] = None

    return result


def find_participants_with_timeline(output_dir: Path) -> List[str]:
    """Find all participants that have annotated timeline files."""
    participants = []
    for participant_dir in sorted(output_dir.iterdir()):
        if not participant_dir.is_dir():
            continue
        participant = participant_dir.name
        timeline_file = participant_dir / f"{participant}_timeline_annotated.json"
        if timeline_file.exists():
            participants.append(participant)
    return participants


def process_participant(
    participant: str,
    output_dir: Path,
    tag: str,
    model: str,
    block_duration: float = BLOCK_DURATION,
    low_only: bool = False,
    test_limit: int = 0,
    verbose: bool = False,
    delete_clips: bool = False,
    skip_existing: bool = False,
) -> Optional[Path]:
    """Process a single participant using baseline block approach."""
    participant_dir = output_dir / participant

    # Check if result file already exists
    output_file = participant_dir / f"{participant}_vlm_baseline_{tag}_results.json"
    if skip_existing and output_file.exists():
        print(f"SKIP {participant}: {output_file.name} already exists")
        return output_file

    # Load timeline annotated data
    timeline_file = participant_dir / f"{participant}_timeline_annotated.json"
    if not timeline_file.exists():
        print(f"SKIP {participant}: {timeline_file.name} not found")
        return None

    with open(timeline_file, 'r') as f:
        timeline_data = json.load(f)

    items = timeline_data.get('items', [])
    print(f"\n{'='*70}")
    print(f"PARTICIPANT: {participant} (BASELINE)")
    print(f"{'='*70}")
    print(f"Loaded {len(items)} items from {timeline_file.name}")

    # Collect unique food items for prompt
    all_food_items = sorted(set(item.get('food_name', '') for item in items if item.get('food_name')))
    print(f"Food items list: {len(all_food_items)} unique items")

    # Filter items with valid segments
    valid_items = [
        item for item in items
        if item.get('dispensal_segments') and len(item.get('dispensal_segments', [])) > 0
    ]
    print(f"Items with valid segments: {len(valid_items)}")

    # Filter by difficulty if requested
    if low_only:
        valid_items = [
            item for item in valid_items
            if item.get('difficulty', '').upper() == 'LOW'
        ]
        print(f"LOW difficulty items: {len(valid_items)}")

    if not valid_items:
        print(f"No items to process for {participant}")
        return None

    # Apply test limit
    if test_limit > 0:
        valid_items = valid_items[:test_limit]
        print(f"TEST MODE: Processing only first {len(valid_items)} items")

    # Initialize VLM client
    print(f"\nInitializing {model} VLM client...")
    if model == 'gpt4o':
        vlm = GPT4oClient(fps=2.0)
    else:
        vlm = VLMClient(model_name=model, use_video=True)

    # Create clips directory
    clips_dir = participant_dir / "vlm_baseline_clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    # Cache for block results
    block_cache: Dict[str, Dict] = {}

    # Format food items list for prompt
    food_items_str = "\n".join(f"  - {name}" for name in all_food_items)

    print(f"\n{'='*70}")
    print(f"BASELINE: {block_duration}s BLOCKS")
    print(f"{'='*70}")

    results = []

    for i, item in enumerate(valid_items):
        food_name = item.get('food_name', 'unknown')
        narr_id = item.get('narration_id', '')
        difficulty = item.get('difficulty', 'UNKNOWN')
        video_range = item.get('video_range', [])
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
            video_id = segment.get('video_id') or (video_range[0] if video_range else None)

            # Get or generate segment_id
            seg_id = segment.get('segment_id')
            if not seg_id and video_id:
                seg_id = generate_segment_id(narr_id, video_id, start_ts, end_ts)

            if not video_id:
                print(f"    Segment {seg_idx+1}: SKIP - no video ID")
                continue

            video_path = VIDEO_BASE_DIR / participant / f"{video_id}.mp4"
            if not video_path.exists():
                print(f"    Segment {seg_idx+1}: SKIP - video not found")
                continue

            # Get overlapping blocks
            blocks = get_overlapping_blocks(start_ts, end_ts, block_duration)
            print(f"    Segment {seg_idx+1}: {start_ts:.1f}s-{end_ts:.1f}s -> blocks {[b[0] for b in blocks]}")

            # Query VLM for each block, aggregate results
            block_results = []

            for block_idx, block_start, block_end in blocks:
                cache_key = f"{video_id}_block{block_idx}"

                if cache_key in block_cache:
                    print(f"      Block {block_idx}: cached")
                    block_results.append(block_cache[cache_key])
                    continue

                # Extract block clip
                clip_filename = f"{video_id}_block{block_idx}_{block_start:.0f}s.mp4"
                clip_path = clips_dir / clip_filename

                if not clip_path.exists():
                    print(f"      Block {block_idx} ({block_start:.0f}s-{block_end:.0f}s): extracting...", end=" ", flush=True)
                    video_duration = get_video_duration(video_path)
                    actual_end = min(block_end, video_duration)

                    success = extract_video_block(video_path, block_start, actual_end, clip_path)
                    if not success:
                        print("FAILED")
                        continue
                    print(f"OK ({clip_path.stat().st_size / 1024:.1f} KB)")

                # Query VLM
                print(f"      Querying {model} for block {block_idx}...", end=" ", flush=True)
                prompt = BASELINE_PROMPT.format(food_items_list=food_items_str)
                response = vlm.query(
                    system_prompt="You are a Visual Inventory Auditor analyzing cooking videos.",
                    user_prompt=prompt,
                    video_path=clip_path
                )

                if not response:
                    print("NO RESPONSE")
                    block_cache[cache_key] = {'block_idx': block_idx, 'parsed': None}
                    continue

                # Parse response
                parsed = parse_vlm_response(response)
                detected_item = parsed.get('detected_item_name')
                pred_count = parsed.get('numeric_count')

                block_result = {
                    'block_idx': block_idx,
                    'block_start': block_start,
                    'block_end': block_end,
                    'parsed': parsed,
                }
                block_cache[cache_key] = block_result
                block_results.append(block_result)

                detected_str = f"[{detected_item}]" if detected_item else "none"
                print(f"detected: {detected_str} count: {pred_count}")

                if delete_clips and clip_path.exists():
                    clip_path.unlink()

            # Filter block results to those with valid detections
            valid_block_results = [
                br for br in block_results
                if br.get('parsed') and br['parsed'].get('detected_item_name')
            ]

            final_result = None
            aggregation_used = False

            # Decision: use aggregation only if 2+ blocks have valid detections
            if len(valid_block_results) >= 2:
                # Use text-only VLM aggregation
                print(f"      Aggregating {len(valid_block_results)} block results via VLM...", end=" ", flush=True)

                block_results_text = format_block_results_for_aggregation(block_results)
                agg_prompt = AGGREGATION_PROMPT.format(
                    food_items_list=food_items_str,
                    block_results=block_results_text
                )

                agg_response = vlm.text_query(
                    system_prompt="You are a Visual Inventory Auditor aggregating detection results.",
                    user_prompt=agg_prompt
                )

                if agg_response:
                    agg_parsed = parse_aggregation_response(agg_response)
                    detected = agg_parsed.get('detected_item_name')
                    if detected:
                        match, similarity = check_item_match(detected, food_name)
                        final_result = {
                            'parsed': agg_parsed,
                            'item_match': match,
                            'item_similarity': similarity,
                            'aggregation_used': True,
                            'num_blocks_aggregated': len(valid_block_results),
                        }
                        aggregation_used = True
                        print(f"[{detected}] count={agg_parsed.get('numeric_count')}")
                    else:
                        print("no item detected")
                else:
                    print("NO RESPONSE")

            # Fallback: use best single block result
            if final_result is None:
                best_similarity = 0.0
                for br in valid_block_results:
                    parsed = br.get('parsed')
                    detected = parsed.get('detected_item_name')
                    match, similarity = check_item_match(detected, food_name)

                    if similarity > best_similarity:
                        best_similarity = similarity
                        final_result = {
                            'parsed': parsed,
                            'block_idx': br['block_idx'],
                            'item_match': match,
                            'item_similarity': similarity,
                            'aggregation_used': False,
                        }

            # Build segment result
            if final_result:
                parsed = final_result['parsed']
                # Pass item_match - requires both item AND count match for 'exact'
                evaluation = evaluate_result(
                    parsed,
                    {'total_count': gt_count},
                    item_match=final_result.get('item_match')
                )

                segment_result = {
                    'segment_id': seg_id,
                    'segment_idx': seg_idx,
                    'video_id': video_id,
                    'start_timestamp': start_ts,
                    'end_timestamp': end_ts,
                    'ground_truth_count': gt_count,
                    'ground_truth_unit': gt_unit,
                    'quantity_category': parsed.get('quantity_category'),
                    'predicted_count': parsed.get('numeric_count'),
                    'predicted_amount': parsed.get('amount_description'),
                    'predicted_unit': parsed.get('unit_type'),
                    'detected_item_name': parsed.get('detected_item_name'),
                    'item_match': final_result['item_match'],
                    'item_similarity': final_result['item_similarity'],
                    'confidence': parsed.get('confidence'),
                    'match': evaluation.get('match'),
                    'error': evaluation.get('error'),
                    'blocks_queried': [b[0] for b in blocks],
                    'aggregation_used': final_result.get('aggregation_used', False),
                }

                # Add aggregation-specific or single-block-specific fields
                if final_result.get('aggregation_used'):
                    segment_result['aggregation_reasoning'] = parsed.get('aggregation_reasoning', '')
                    segment_result['num_blocks_aggregated'] = final_result.get('num_blocks_aggregated')
                else:
                    segment_result['best_block_idx'] = final_result.get('block_idx')
                    segment_result['remaining_count'] = parsed.get('remaining_count')
                    segment_result['remaining_description'] = parsed.get('remaining_description')
                    segment_result['remaining_fraction'] = parsed.get('remaining_fraction')
                    segment_result['visual_evidence'] = parsed.get('visual_evidence')

                segment_results.append(segment_result)

                agg_str = " (aggregated)" if aggregation_used else ""
                print(f"    -> Final{agg_str}: [{parsed.get('detected_item_name')}] sim={final_result['item_similarity']:.2f} count={parsed.get('numeric_count')}")
            else:
                segment_results.append({
                    'segment_id': seg_id,
                    'segment_idx': seg_idx,
                    'video_id': video_id,
                    'start_timestamp': start_ts,
                    'end_timestamp': end_ts,
                    'ground_truth_count': gt_count,
                    'ground_truth_unit': gt_unit,
                    'error': 'No valid detection from blocks',
                    'blocks_queried': [b[0] for b in blocks],
                })
                print(f"    -> No detection")

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

    # Clean up
    if delete_clips:
        try:
            clips_dir.rmdir()
        except OSError:
            pass

    # Save results
    print(f"\nSAVING RESULTS")

    output_data = {
        'participant': participant,
        'model': model,
        'tag': tag,
        'method': 'baseline',
        'block_duration': block_duration,
        'low_only': low_only,
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
    exact_matches = sum(
        1 for r in results
        for s in r.get('segments', [])
        if s.get('match') == 'exact'
    )
    item_matches = sum(
        1 for r in results
        for s in r.get('segments', [])
        if s.get('item_match') is True
    )
    aggregated_segments = sum(
        1 for r in results
        for s in r.get('segments', [])
        if s.get('aggregation_used') is True
    )

    print(f"Total items: {len(results)}")
    print(f"Total segments: {total_segments}")
    print(f"Segments using aggregation: {aggregated_segments} ({100*aggregated_segments/max(1,total_segments):.1f}%)")
    print(f"Exact count matches: {exact_matches} ({100*exact_matches/max(1,total_segments):.1f}%)")
    print(f"Item detection matches: {item_matches} ({100*item_matches/max(1,total_segments):.1f}%)")

    return output_file


# =============================================================================
# FULL-SCAN MODE: Process ALL blocks in video to detect false positives
# =============================================================================

def process_participant_full_scan(
    participant: str,
    output_dir: Path,
    tag: str,
    model: str,
    block_duration: float = BLOCK_DURATION,
    verbose: bool = False,
    delete_clips: bool = False,
    skip_existing: bool = False,
    max_videos: int = 0,
) -> Optional[Path]:
    """
    Full-scan mode: Process ALL blocks in all videos for a participant.

    This mode runs VLM on every block to detect:
    - True positives: Dispensals detected in blocks overlapping with ground truth
    - False positives: Dispensals detected in blocks with no ground truth
    """
    participant_dir = output_dir / participant

    # Check if result file already exists
    output_file = participant_dir / f"{participant}_vlm_fullscan_{tag}_results.json"
    if skip_existing and output_file.exists():
        print(f"SKIP {participant}: {output_file.name} already exists")
        return output_file

    # Load timeline annotated data
    timeline_file = participant_dir / f"{participant}_timeline_annotated.json"
    if not timeline_file.exists():
        print(f"SKIP {participant}: {timeline_file.name} not found")
        return None

    with open(timeline_file, 'r') as f:
        timeline_data = json.load(f)

    items = timeline_data.get('items', [])
    print(f"\n{'='*70}")
    print(f"PARTICIPANT: {participant} (FULL-SCAN)")
    print(f"{'='*70}")
    print(f"Loaded {len(items)} items from {timeline_file.name}")

    # Collect unique food items for prompt
    all_food_items = sorted(set(item.get('food_name', '') for item in items if item.get('food_name')))
    food_items_str = '\n'.join(f"- {item}" for item in all_food_items)
    print(f"Food items list: {len(all_food_items)} unique items")

    # Build a map of all known dispensal time ranges per video
    # key: video_id, value: list of (start, end, food_name, narration_id)
    dispensal_ranges = {}
    for item in items:
        for seg in item.get('dispensal_segments', []):
            video_id = seg.get('video_id')
            if not video_id:
                continue
            if video_id not in dispensal_ranges:
                dispensal_ranges[video_id] = []
            dispensal_ranges[video_id].append({
                'start': seg.get('start_timestamp'),
                'end': seg.get('end_timestamp'),
                'food_name': item.get('food_name'),
                'narration_id': item.get('narration_id'),
                'difficulty': item.get('difficulty'),
            })

    print(f"Videos with dispensal data: {len(dispensal_ranges)}")

    # Initialize VLM client
    print(f"\nInitializing {model} VLM client...")
    if model == 'gpt4o':
        vlm = GPT4oClient(fps=2.0)
    else:
        vlm = VLMClient(model_name=model, use_video=True)

    # Prepare clips directory
    clips_dir = participant_dir / "clips_fullscan"
    clips_dir.mkdir(exist_ok=True)

    # Find all videos for this participant
    participant_video_dir = VIDEO_BASE_DIR / participant
    if not participant_video_dir.exists():
        print(f"SKIP {participant}: video directory not found")
        return None

    # Get unique video IDs from dispensal data
    all_video_ids = set(dispensal_ranges.keys())

    # Also scan the video directory for any videos
    for video_file in participant_video_dir.glob("*.mp4"):
        all_video_ids.add(video_file.stem)

    # Apply max_videos limit
    video_ids_to_process = sorted(all_video_ids)
    if max_videos > 0:
        video_ids_to_process = video_ids_to_process[:max_videos]
        print(f"Videos to scan: {len(video_ids_to_process)} (limited from {len(all_video_ids)})")
    else:
        print(f"Videos to scan: {len(video_ids_to_process)}")

    # Process each video
    all_blocks = []

    for video_id in video_ids_to_process:
        video_path = participant_video_dir / f"{video_id}.mp4"
        if not video_path.exists():
            continue

        video_duration = get_video_duration(video_path)
        if video_duration <= 0:
            continue

        num_blocks = int(video_duration // block_duration) + 1
        print(f"\n--- {video_id}: {video_duration:.0f}s = {num_blocks} blocks ---")

        # Get dispensal ranges for this video
        video_dispensals = dispensal_ranges.get(video_id, [])

        for block_idx in range(num_blocks):
            block_start = block_idx * block_duration
            block_end = min((block_idx + 1) * block_duration, video_duration)

            # Check if this block overlaps with any known dispensal
            overlapping_dispensals = []
            for d in video_dispensals:
                if d['start'] < block_end and d['end'] > block_start:
                    overlapping_dispensals.append(d)

            is_dispensal_block = len(overlapping_dispensals) > 0

            # Extract block clip
            clip_filename = f"{video_id}_block{block_idx}_{block_start:.0f}s.mp4"
            clip_path = clips_dir / clip_filename

            if not clip_path.exists():
                if verbose:
                    print(f"  Block {block_idx} ({block_start:.0f}s-{block_end:.0f}s): extracting...", end=" ", flush=True)
                success = extract_video_block(video_path, block_start, block_end, clip_path)
                if not success:
                    if verbose:
                        print("FAILED")
                    continue
                if verbose:
                    print(f"OK ({clip_path.stat().st_size / 1024:.1f} KB)")

            # Run detection prompt
            if verbose:
                marker = "[GT]" if is_dispensal_block else "[BG]"
                print(f"  {marker} Block {block_idx}: querying...", end=" ", flush=True)

            detection_prompt = DETECTION_PROMPT.format(food_items_list=food_items_str)
            response = vlm.query(
                system_prompt="You are a Visual Inventory Auditor detecting food dispensal actions.",
                user_prompt=detection_prompt,
                video_path=clip_path
            )

            if not response:
                if verbose:
                    print("NO RESPONSE")
                block_result = {
                    'video_id': video_id,
                    'block_idx': block_idx,
                    'block_start': block_start,
                    'block_end': block_end,
                    'is_dispensal_block': is_dispensal_block,
                    'overlapping_dispensals': overlapping_dispensals,
                    'detection': None,
                    'error': 'No VLM response',
                }
            else:
                detection = parse_detection_response(response)

                if verbose:
                    if detection['dispensal_detected']:
                        print(f"DETECTED: {detection['detected_item_name']} ({detection['action_type']})")
                    else:
                        print(f"none")

                block_result = {
                    'video_id': video_id,
                    'block_idx': block_idx,
                    'block_start': block_start,
                    'block_end': block_end,
                    'is_dispensal_block': is_dispensal_block,
                    'overlapping_dispensals': overlapping_dispensals,
                    'detection': detection,
                }

                # Check if detection matches ground truth (for dispensal blocks)
                if detection['dispensal_detected'] and overlapping_dispensals:
                    detected_item = detection.get('detected_item_name')
                    best_match = None
                    best_similarity = 0.0
                    for d in overlapping_dispensals:
                        match, sim = check_item_match(detected_item, d['food_name'])
                        if sim > best_similarity:
                            best_similarity = sim
                            best_match = d['food_name'] if match else None
                    block_result['item_match'] = best_match is not None
                    block_result['item_similarity'] = best_similarity
                    block_result['matched_food'] = best_match

            all_blocks.append(block_result)

            if delete_clips and clip_path.exists():
                clip_path.unlink()

    # Clean up
    if delete_clips:
        try:
            clips_dir.rmdir()
        except OSError:
            pass

    # Compute summary statistics
    total_blocks = len(all_blocks)
    gt_blocks = sum(1 for b in all_blocks if b['is_dispensal_block'])
    bg_blocks = total_blocks - gt_blocks

    # Detection counts
    detected_in_gt = sum(1 for b in all_blocks if b['is_dispensal_block'] and b.get('detection', {}).get('dispensal_detected'))
    detected_in_bg = sum(1 for b in all_blocks if not b['is_dispensal_block'] and b.get('detection', {}).get('dispensal_detected'))

    # Item match (in GT blocks with detection)
    item_match_in_gt = sum(1 for b in all_blocks if b['is_dispensal_block'] and b.get('item_match'))

    summary = {
        'total_blocks': total_blocks,
        'gt_blocks': gt_blocks,
        'bg_blocks': bg_blocks,
        'detected_in_gt': detected_in_gt,
        'detected_in_bg': detected_in_bg,
        'true_positive_rate': detected_in_gt / gt_blocks if gt_blocks > 0 else None,
        'false_positive_rate': detected_in_bg / bg_blocks if bg_blocks > 0 else None,
        'item_match_in_gt': item_match_in_gt,
        'item_match_rate': item_match_in_gt / detected_in_gt if detected_in_gt > 0 else None,
    }

    # Save results
    print(f"\nSAVING RESULTS")

    output_data = {
        'participant': participant,
        'model': model,
        'tag': tag,
        'method': 'full_scan',
        'block_duration': block_duration,
        'summary': summary,
        'blocks': all_blocks
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)

    print(f"  Saved: {output_file.name}")

    # Print summary
    print(f"\n{'='*70}")
    print(f"FULL-SCAN SUMMARY")
    print(f"{'='*70}")
    print(f"Total blocks scanned: {total_blocks}")
    print(f"  GT blocks (with dispensal): {gt_blocks}")
    print(f"  BG blocks (background): {bg_blocks}")
    print(f"\nDetections:")
    print(f"  In GT blocks: {detected_in_gt}/{gt_blocks} ({100*detected_in_gt/max(1,gt_blocks):.1f}%) - True Positive Rate")
    print(f"  In BG blocks: {detected_in_bg}/{bg_blocks} ({100*detected_in_bg/max(1,bg_blocks):.1f}%) - False Positive Rate")
    if detected_in_gt > 0:
        print(f"  Item match (in GT): {item_match_in_gt}/{detected_in_gt} ({100*item_match_in_gt/detected_in_gt:.1f}%)")

    return output_file


# =============================================================================
# EVALUATION REPORT GENERATION
# =============================================================================

def evaluate_segment_for_report(seg: dict, food_name: str) -> dict:
    """
    Evaluate a single segment for the evaluation report.

    Returns dict with evaluation results.
    """
    gt = seg.get('ground_truth_count')
    pred = seg.get('predicted_count')
    gt_unit = seg.get('ground_truth_unit', '')
    pred_unit = seg.get('predicted_unit', '')
    detected_item = seg.get('detected_item_name')
    item_match = seg.get('item_match')
    item_similarity = seg.get('item_similarity')
    quantity_category = seg.get('quantity_category', '')
    match_status = seg.get('match', '')

    result = {
        'segment_id': seg.get('segment_id'),
        'segment_idx': seg.get('segment_idx'),
        'ground_truth': gt,
        'predicted': pred,
        'gt_unit': gt_unit,
        'pred_unit': pred_unit,
        'detected_item': detected_item,
        'expected_item': food_name,
        'match_status': match_status,
        'aggregation_used': seg.get('aggregation_used', False),
    }

    # Count accuracy
    if gt is not None and pred is not None:
        result['count_correct'] = (pred == gt)
        result['count_close'] = abs(pred - gt) <= 1
        result['absolute_error'] = abs(pred - gt)
    else:
        result['count_correct'] = None
        result['count_close'] = None
        result['absolute_error'] = None

    # Item match
    if item_match is not None:
        result['item_match'] = item_match
        result['item_similarity'] = item_similarity
    elif detected_item:
        result['item_match'] = detected_item.lower().strip() == food_name.lower().strip()
        result['item_similarity'] = 1.0 if result['item_match'] else 0.0
    else:
        result['item_match'] = None
        result['item_similarity'] = None

    # Category mismatch: VLM says continuous but GT is discrete
    result['category_mismatch'] = (
        quantity_category == 'continuous' and gt is not None
    )

    # Prediction missing (VLM returned null count for discrete GT)
    result['prediction_missing'] = (gt is not None and pred is None)

    return result


def compute_baseline_metrics(all_segments: list) -> dict:
    """Compute aggregate metrics from all segments."""

    # Filter to countable segments
    countable = [s for s in all_segments if s['count_correct'] is not None]

    # Basic count metrics
    n_total = len(countable)
    n_correct = sum(1 for s in countable if s['count_correct'])
    n_close = sum(1 for s in countable if s['count_close'])
    total_abs_error = sum(s['absolute_error'] for s in countable)

    # Item detection metrics - all segments
    n_with_item_info = sum(1 for s in all_segments if s['item_match'] is not None)
    n_item_match = sum(1 for s in all_segments if s['item_match'] is True)
    n_item_mismatch = sum(1 for s in all_segments if s['item_match'] is False)

    n_category_mismatch = sum(1 for s in all_segments if s['category_mismatch'])
    n_prediction_missing = sum(1 for s in all_segments if s['prediction_missing'])

    # Aggregation stats
    n_aggregated = sum(1 for s in all_segments if s.get('aggregation_used'))

    # Correlation analysis: count accuracy conditioned on item match/mismatch
    item_match_correct = sum(1 for s in countable if s['item_match'] is True and s['count_correct'])
    item_match_total = sum(1 for s in countable if s['item_match'] is True)

    item_mismatch_correct = sum(1 for s in countable if s['item_match'] is False and s['count_correct'])
    item_mismatch_total = sum(1 for s in countable if s['item_match'] is False)

    # Aggregation correlation
    agg_correct = sum(1 for s in countable if s.get('aggregation_used') and s['count_correct'])
    agg_total = sum(1 for s in countable if s.get('aggregation_used'))
    non_agg_correct = sum(1 for s in countable if not s.get('aggregation_used') and s['count_correct'])
    non_agg_total = sum(1 for s in countable if not s.get('aggregation_used'))

    return {
        # Count accuracy
        'n_segments_total': len(all_segments),
        'n_segments_countable': n_total,
        'n_count_correct': n_correct,
        'n_count_close': n_close,
        'count_accuracy': n_correct / n_total if n_total > 0 else None,
        'count_close_rate': n_close / n_total if n_total > 0 else None,
        'mean_absolute_error': total_abs_error / n_total if n_total > 0 else None,

        # Item detection
        'n_with_item_detection': n_with_item_info,
        'n_item_match': n_item_match,
        'n_item_mismatch': n_item_mismatch,
        'item_match_rate': n_item_match / n_with_item_info if n_with_item_info > 0 else None,
        'item_mismatch_rate': n_item_mismatch / n_with_item_info if n_with_item_info > 0 else None,

        # Category mismatch
        'n_category_mismatch': n_category_mismatch,
        'category_mismatch_rate': n_category_mismatch / len(all_segments) if len(all_segments) > 0 else None,

        # Prediction missing
        'n_prediction_missing': n_prediction_missing,
        'prediction_missing_rate': n_prediction_missing / len(all_segments) if len(all_segments) > 0 else None,

        # Aggregation stats
        'n_aggregated': n_aggregated,
        'aggregation_rate': n_aggregated / len(all_segments) if len(all_segments) > 0 else None,

        # Correlation: count accuracy given item match/mismatch
        'count_accuracy_given_item_match': item_match_correct / item_match_total if item_match_total > 0 else None,
        'count_accuracy_given_item_mismatch': item_mismatch_correct / item_mismatch_total if item_mismatch_total > 0 else None,
        'n_item_match_countable': item_match_total,
        'n_item_mismatch_countable': item_mismatch_total,

        # Correlation: count accuracy given aggregation
        'count_accuracy_given_aggregation': agg_correct / agg_total if agg_total > 0 else None,
        'count_accuracy_given_no_aggregation': non_agg_correct / non_agg_total if non_agg_total > 0 else None,
        'n_aggregated_countable': agg_total,
        'n_non_aggregated_countable': non_agg_total,
    }


def generate_eval_report(
    result_files: List[Path],
    output_dir: Path,
    tag: str,
    low_only: bool = False
) -> Optional[dict]:
    """Generate evaluation report from result files."""

    if not result_files:
        return None

    all_segments = []
    per_participant = []

    for result_file in result_files:
        with open(result_file, 'r') as f:
            data = json.load(f)

        participant = data.get('participant')
        items = data.get('items', [])

        # Collect segment evaluations
        participant_segments = []
        for item in items:
            difficulty = item.get('difficulty', '').upper()
            if low_only and difficulty != 'LOW':
                continue

            food_name = item.get('food_name', '')
            for seg in item.get('segments', []):
                seg_eval = evaluate_segment_for_report(seg, food_name)
                seg_eval['difficulty'] = difficulty
                seg_eval['participant'] = participant
                participant_segments.append(seg_eval)
                all_segments.append(seg_eval)

        # Per-participant metrics
        p_metrics = compute_baseline_metrics(participant_segments)
        per_participant.append({
            'participant': participant,
            'model': data.get('model'),
            'tag': data.get('tag'),
            'method': data.get('method', 'baseline'),
            **p_metrics
        })

    # Aggregate metrics
    aggregate = compute_baseline_metrics(all_segments)

    report = {
        'generated_at': datetime.now().isoformat(),
        'tag': tag,
        'method': 'baseline',
        'filter': 'LOW difficulty only' if low_only else 'All difficulties',
        'aggregate': aggregate,
        'per_participant': per_participant,
    }

    # Save report
    difficulty_suffix = '' if low_only else '_all'
    report_path = output_dir / f"vlm_baseline_{tag}{difficulty_suffix}_eval_report.json"

    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\nSaved evaluation report: {report_path}")
    return report


def print_eval_report(report: dict):
    """Print human-readable summary of the evaluation report."""
    agg = report['aggregate']

    print("\n" + "=" * 80)
    print("BASELINE VLM EVALUATION REPORT")
    print("=" * 80)
    print(f"Tag: {report.get('tag')}")
    print(f"Filter: {report.get('filter')}")
    print(f"Participants: {len(report['per_participant'])}")

    # Overall metrics
    print("\n" + "-" * 80)
    print("AGGREGATE METRICS")
    print("-" * 80)

    print(f"\n{'Metric':<45} {'Value':>12} {'Count':>15}")
    print("-" * 72)

    # Count accuracy
    acc = f"{agg['count_accuracy']:.1%}" if agg['count_accuracy'] is not None else "N/A"
    print(f"{'Count Accuracy (exact)':<45} {acc:>12} {agg['n_count_correct']:>10}/{agg['n_segments_countable']}")

    close = f"{agg['count_close_rate']:.1%}" if agg['count_close_rate'] is not None else "N/A"
    print(f"{'Count Close Rate (±1)':<45} {close:>12} {agg['n_count_close']:>10}/{agg['n_segments_countable']}")

    mae = f"{agg['mean_absolute_error']:.2f}" if agg['mean_absolute_error'] is not None else "N/A"
    print(f"{'Mean Absolute Error':<45} {mae:>12}")

    # Item detection metrics
    print("\n--- Item Detection ---")

    item_rate = f"{agg['item_match_rate']:.1%}" if agg['item_match_rate'] is not None else "N/A"
    print(f"{'Item Detection Match Rate':<45} {item_rate:>12} {agg['n_item_match']:>10}/{agg['n_with_item_detection']}")

    item_mis = f"{agg['item_mismatch_rate']:.1%}" if agg['item_mismatch_rate'] is not None else "N/A"
    print(f"{'Item Detection Mismatch Rate':<45} {item_mis:>12} {agg['n_item_mismatch']:>10}/{agg['n_with_item_detection']}")

    cat_mis = f"{agg['category_mismatch_rate']:.1%}" if agg['category_mismatch_rate'] is not None else "N/A"
    print(f"{'Category Mismatch Rate (cont. vs disc.)':<45} {cat_mis:>12} {agg['n_category_mismatch']:>10}/{agg['n_segments_total']}")

    pred_mis = f"{agg['prediction_missing_rate']:.1%}" if agg['prediction_missing_rate'] is not None else "N/A"
    print(f"{'Prediction Missing Rate':<45} {pred_mis:>12} {agg['n_prediction_missing']:>10}/{agg['n_segments_total']}")

    # Aggregation stats
    print("\n--- Aggregation ---")

    agg_rate = f"{agg['aggregation_rate']:.1%}" if agg['aggregation_rate'] is not None else "N/A"
    print(f"{'Segments Using Aggregation':<45} {agg_rate:>12} {agg['n_aggregated']:>10}/{agg['n_segments_total']}")

    agg_acc = f"{agg['count_accuracy_given_aggregation']:.1%}" if agg['count_accuracy_given_aggregation'] is not None else "N/A"
    print(f"{'Count Accuracy | Aggregation Used':<45} {agg_acc:>12} (n={agg['n_aggregated_countable']})")

    non_agg_acc = f"{agg['count_accuracy_given_no_aggregation']:.1%}" if agg['count_accuracy_given_no_aggregation'] is not None else "N/A"
    print(f"{'Count Accuracy | No Aggregation':<45} {non_agg_acc:>12} (n={agg['n_non_aggregated_countable']})")

    # Correlation analysis
    print("\n--- Correlation: Count Accuracy by Condition ---")

    acc_item_match = f"{agg['count_accuracy_given_item_match']:.1%}" if agg['count_accuracy_given_item_match'] is not None else "N/A"
    print(f"{'Count Accuracy | Item Match':<45} {acc_item_match:>12} (n={agg['n_item_match_countable']})")

    acc_item_mis = f"{agg['count_accuracy_given_item_mismatch']:.1%}" if agg['count_accuracy_given_item_mismatch'] is not None else "N/A"
    print(f"{'Count Accuracy | Item Mismatch':<45} {acc_item_mis:>12} (n={agg['n_item_mismatch_countable']})")

    # Per-participant table
    print("\n" + "-" * 80)
    print("PER-PARTICIPANT SUMMARY")
    print("-" * 80)
    print(f"\n{'Part':<6} {'Segs':>6} {'Count':>8} {'Item':>8} {'Cat':>6} {'Agg':>6} {'MAE':>6}")
    print(f"{'':6} {'':>6} {'Acc':>8} {'Match':>8} {'Mis':>6} {'Used':>6} {'':>6}")
    print("-" * 72)

    for p in report['per_participant']:
        acc = f"{p['count_accuracy']:.0%}" if p['count_accuracy'] is not None else "N/A"
        item = f"{p['item_match_rate']:.0%}" if p['item_match_rate'] is not None else "N/A"
        cat = f"{p['n_category_mismatch']}"
        agg_n = f"{p['n_aggregated']}"
        mae = f"{p['mean_absolute_error']:.2f}" if p['mean_absolute_error'] is not None else "N/A"
        print(f"{p['participant']:<6} {p['n_segments_total']:>6} {acc:>8} {item:>8} {cat:>6} {agg_n:>6} {mae:>6}")

    print("=" * 80)


def find_baseline_results(output_dir: Path, tag: str) -> List[Path]:
    """Find all baseline VLM result files in output directory."""
    pattern = f"*_vlm_baseline_{tag}_results.json"
    results = []

    for participant_dir in sorted(output_dir.iterdir()):
        if not participant_dir.is_dir() or not participant_dir.name.startswith('P'):
            continue
        for f in participant_dir.glob(pattern):
            if '_eval_report' not in f.name:
                results.append(f)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Baseline VLM Q&A using fixed 30-second video blocks"
    )
    parser.add_argument('--participant', help='Participant ID (e.g., P03). Use --all for all.')
    parser.add_argument('--all', action='store_true', help='Process all participants')
    parser.add_argument('--tag', required=True, help='Tag for this run (used in output filename)')
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--test', type=int, default=0, help='Test mode: only process first N items')
    parser.add_argument('--model', default='qwen', choices=['qwen', 'gpt4o'])
    parser.add_argument('--low-only', action='store_true', help='Only process LOW difficulty items')
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--delete-clips', action='store_true', help='Delete clips after processing')
    parser.add_argument('--skip-existing', action='store_true', help='Skip if result file exists')
    parser.add_argument('--block-duration', type=float, default=BLOCK_DURATION, help='Block duration in seconds')
    parser.add_argument('--eval-only', action='store_true',
                       help='Only generate evaluation report from existing results (no VLM processing)')
    parser.add_argument('--full-scan', action='store_true',
                       help='Full-scan mode: process ALL blocks in videos to measure false positive rate')
    parser.add_argument('--max-videos', type=int, default=0,
                       help='Limit number of videos to process (for testing full-scan mode)')

    args = parser.parse_args()

    # Eval-only mode: just generate report from existing files
    if args.eval_only:
        result_files = find_baseline_results(args.output_dir, args.tag)
        if not result_files:
            print(f"No baseline results found with tag '{args.tag}'")
            return 1
        print(f"Found {len(result_files)} result files for tag '{args.tag}'")
        report = generate_eval_report(
            result_files=result_files,
            output_dir=args.output_dir,
            tag=args.tag,
            low_only=args.low_only
        )
        if report:
            print_eval_report(report)
        return 0

    # Validate arguments for processing mode
    if not args.participant and not args.all:
        parser.error("Either --participant or --all must be specified")

    # Determine participants to process
    if args.all:
        participants = find_participants_with_timeline(args.output_dir)
        if not participants:
            print(f"No participants with annotated timelines found")
            return 1
        print(f"Found {len(participants)} participants: {', '.join(participants)}")
    else:
        participants = [args.participant]

    # Process each participant
    result_files = []
    for participant in participants:
        if args.full_scan:
            # Full-scan mode: process ALL blocks in videos
            result_file = process_participant_full_scan(
                participant=participant,
                output_dir=args.output_dir,
                tag=args.tag,
                model=args.model,
                block_duration=args.block_duration,
                verbose=args.verbose,
                delete_clips=args.delete_clips,
                skip_existing=args.skip_existing,
                max_videos=args.max_videos,
            )
        else:
            # Normal baseline mode: process only dispensal segments
            result_file = process_participant(
                participant=participant,
                output_dir=args.output_dir,
                tag=args.tag,
                model=args.model,
                block_duration=args.block_duration,
                low_only=args.low_only,
                test_limit=args.test,
                verbose=args.verbose,
                delete_clips=args.delete_clips,
                skip_existing=args.skip_existing,
            )
        if result_file:
            result_files.append(result_file)

    print(f"\n{'='*70}")
    print("COMPLETE")
    print(f"{'='*70}")
    print(f"Processed {len(result_files)} participant(s)")

    # Generate evaluation report if we have results (only for normal baseline mode)
    if result_files and not args.full_scan:
        report = generate_eval_report(
            result_files=result_files,
            output_dir=args.output_dir,
            tag=args.tag,
            low_only=args.low_only
        )
        if report:
            print_eval_report(report)

    return 0


if __name__ == '__main__':
    exit(main())

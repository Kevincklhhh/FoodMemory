#!/usr/bin/env python3
"""
07_vlm_QA.py - VLM Q&A Engine for Quantity Estimation

Tests VLM capability for estimating quantity change from dispensing actions.
Uses timeline_annotated.json to extract video clips and query VLM.

Prerequisites:
    Run 06_timeline_aggregation.py first to generate timeline_annotated.json

Usage:
    # Process single participant
    python 07_vlm_QA.py --participant P03 --tag qwen_v1

    # Process all participants with annotated timelines
    python 07_vlm_QA.py --all --tag qwen_v1

    # Process only LOW difficulty items
    python 07_vlm_QA.py --all --low-only --tag qwen_low

    # Test mode (first N items)
    python 07_vlm_QA.py --participant P03 --test 5 --tag test

    # Skip participants that already have results for this tag
    python 07_vlm_QA.py --all --tag qwen_v1 --skip-existing

Inputs:
    {participant}_timeline_annotated.json
    Video files from data/HD-EPIC/Videos/{participant}/

Outputs:
    {participant}_vlm_qa_{tag}_results.json
    vlm_qa_{tag}_count_eval_report.json (aggregate report)
"""

import argparse
import base64
import json
import os
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Any, List, Tuple

import requests

from dotenv import load_dotenv
from openai import AzureOpenAI
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types

from inventory_utils import DEFAULT_OUTPUT_DIR, generate_segment_id

# Default frames subdirectory under participant output dir
FRAMES_SUBDIR = "hands23_detection"

# Semantic similarity model (lazy loaded)
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
    """
    Check if detected item matches expected item using semantic similarity.

    Returns:
        Tuple of (is_match, similarity_score)
    """
    if not detected or not expected:
        return False, 0.0

    # Normalize strings
    detected_norm = detected.lower().strip()
    expected_norm = expected.lower().strip()

    # Exact match
    if detected_norm == expected_norm:
        return True, 1.0

    # Semantic similarity
    similarity = compute_semantic_similarity(detected_norm, expected_norm)
    return similarity >= threshold, similarity

# Load environment from kitchen root
load_dotenv(Path(__file__).parent.parent.parent / ".env")

# Paths
_SCRIPT_DIR = Path(__file__).parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
VIDEO_BASE_DIR = _PROJECT_ROOT / "data" / "HD-EPIC" / "Videos"

# Qwen3-VL API endpoint
QWEN3VL_URL = "http://saltyfish.eecs.umich.edu:8000/v1/chat/completions"
QWEN_MODEL = "Qwen/Qwen3-VL-30B-A3B-Instruct"


# ACTION_ESTIMATION_PROMPT = """You are a Visual Inventory Auditor.
# Analyze the video clip to estimate the quantity of the Target Food Item removed AND how much remains.

# **INPUT:**
# - Target Item: "{item_name}"

# **INSTRUCTIONS:**
# 1. Focus ONLY on the Target Item.
# 2. Determine if the action is **Discrete** (countable items like eggs, distinct scoops) or **Continuous** (pouring liquid, approximate piles).
# 3. Estimate BOTH the amount removed AND the amount remaining in the container/source.
# 4. Provide the estimate in the strictly defined JSON format below.

# **OUTPUT SCHEMA (Strict JSON):**
# {{
#   "item_name": "{item_name}",
#   "quantity_category": "discrete" | "continuous" | "unknown",

#   // AMOUNT REMOVED:
#   // IF DISCRETE (Countable):
#   "numeric_count": <integer or null>,  // e.g., 1, 2, 5. Null if continuous.
#   // IF CONTINUOUS (Fluids/Piles):
#   "amount_description": <string>,      // e.g., "about half a cup", "a small splash"
#   "volume_fraction": <float or null>,  // Fraction removed (0.0 to 1.0 of container). Null if unknown.

#   // AMOUNT REMAINING:
#   // IF DISCRETE (Countable):
#   "remaining_count": <integer or null>,  // How many items remain. Null if continuous or not visible.
#   // IF CONTINUOUS (Fluids/Piles):
#   "remaining_description": <string or null>,  // e.g., "about 3/4 full", "nearly empty"
#   "remaining_fraction": <float or null>,  // Fraction remaining (0.0 to 1.0 of container). Null if unknown.

#   "unit_type": "unit" | "scoop" | "cup" | "splash" | "pinch" | "slice",
#   "confidence": "high" | "medium" | "low",
#   "visual_evidence": "Brief description of visual proof for both removed and remaining."
# }}

# **EXAMPLES:**

# *Example 1 (Discrete):*
# {{
#   "item_name": "eggs",
#   "quantity_category": "discrete",
#   "numeric_count": 2,
#   "amount_description": null,
#   "volume_fraction": null,
#   "remaining_count": 4,
#   "remaining_description": null,
#   "remaining_fraction": null,
#   "unit_type": "unit",
#   "confidence": "high",
#   "visual_evidence": "User picked two eggs from a carton that had 6 eggs, leaving 4 remaining."
# }}

# *Example 2 (Continuous):*
# {{
#   "item_name": "milk",
#   "quantity_category": "continuous",
#   "numeric_count": null,
#   "amount_description": "approx 1/2 cup",
#   "volume_fraction": 0.1,
#   "remaining_count": null,
#   "remaining_description": "about 3/4 of carton",
#   "remaining_fraction": 0.75,
#   "unit_type": "cup",
#   "confidence": "medium",
#   "visual_evidence": "Steady pour for 2 seconds. Carton appears mostly full after pouring."
# }}

# Return ONLY the raw JSON string. Do not use Markdown (```json).
# """

ACTION_ESTIMATION_PROMPT = """You are a Visual Inventory Auditor for kitchen tasks.
Your goal is to meticulously track the flow of the **Target Food Item** from its source to its destination.

**INPUT:**
- Target Item: "{item_name}"
- Input Video: Egocentric view of a dispensing action.

**VISUAL REASONING STEPS (Follow these mentally before answering):**
1. **Identify Source State:** Look at the container *before* the action starts. Is it full? Can you count the items?
2. **Analyze the Transfer:** Watch the user's hand.
   - What exactly is currently being held or moved?
   - Is the view obstructed by the hand? If so, look at the *destination* to verify what was put down.
3. **Identify End State:** Look at the container *after* the hand leaves. What is left?

**OUTPUT TASK:**
Estimate strictly:
1. **Quantity Removed:** The amount transferred out.
2. **Quantity Remaining:** The amount left in the original source container.

**OUTPUT SCHEMA (Strict JSON):**
{{
  "item_name": "{item_name}",
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
  "item_name": "eggs",
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
  "item_name": "milk",
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

ACTION_ESTIMATION_PROMPT_BLIND = """You are a Visual Inventory Auditor for kitchen tasks.
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

TWO_FRAME_ESTIMATION_PROMPT = """You are a Visual Inventory Auditor.
Your task is to estimate food inventory changes by comparing two specific snapshots in time.

**POSSIBLE FOOD ITEMS:**
{food_items_list}

**INPUT DATA:**
- **Image 1 (Context/Source):** The moment the user reaches for the item. Shows the "Before" state of the source container.
- **Image 2 (Destination/Result):** The moment the item is placed or used. Shows the "After" state of the destination container (e.g., pan, bowl).

**VISUAL REASONING STEPS (Perform this mental check):**
1. **Identify the Item:** Look at Image 1. Which item from the list is the user interacting with?
2. **Analyze Image 1 (The Source):** - Can you count the total items in the source container *before* the action? (Let's call this `Start_Count`).
3. **Analyze Image 2 (The Destination):**
   - Count the *new* items that have appeared in the destination container. (This is your `Quantity Removed`).
4. **Deduce Remaining:** - If you saw the `Start_Count` in Image 1, calculate: `Remaining = Start_Count - Quantity_Removed`.
   - If Image 1 was occluded or unclear, look for visual clues in Image 2 (e.g., did they put the rest of the package down?). Otherwise, mark Remaining as "unknown".

**OUTPUT SCHEMA (Strict JSON):**
{{
  "detected_item_name": <string or null>,
  "quantity_category": "discrete" | "continuous" | "unknown",

  "reasoning_trace": {{
    "image_1_observation": "What did you see in the source container?",
    "image_2_observation": "What did you see added to the destination?",
    "calculation_logic": "Explain how you derived the numbers (e.g., 'Saw 6 eggs in Img1, saw 2 eggs in Img2, so 6-2=4 remaining').",
    "visual_description": "Brief summary of both images and the action."
  }},

  "removed": {{
    "count": <integer or null>,
    "description": <string or null>
  }},

  "remaining": {{
    "count": <integer or null>,
    "description": <string or null>
  }},

  "confidence": "high" | "medium" | "low"
}}

**EXAMPLES:**

*Example 1 (Discrete - Deductive Counting):*
{{
  "detected_item_name": "eggs",
  "quantity_category": "discrete",
  "reasoning_trace": {{
    "image_1_observation": "Saw a carton with 6 eggs. User's hand is reaching for them.",
    "image_2_observation": "Saw a frying pan with 2 distinct yolks cracked into it.",
    "calculation_logic": "Source started with 6. Destination shows 2 used. Therefore, 6 - 2 = 4 remaining.",
    "visual_description": "Carton with 6 eggs, then pan with 2 cracked eggs."
  }},
  "removed": {{"count": 2, "description": "2 eggs cracked"}},
  "remaining": {{"count": 4, "description": "4 eggs estimated in carton"}},
  "confidence": "high"
}}

*Example 2 (Continuous - Visual Estimate):*
{{
  "detected_item_name": "milk",
  "quantity_category": "continuous",
  "reasoning_trace": {{
    "image_1_observation": "Saw a full gallon of milk.",
    "image_2_observation": "Saw a bowl with a small splash of white liquid, maybe 1/4 cup.",
    "calculation_logic": "Only a small amount visible in destination. Source likely still mostly full.",
    "visual_description": "Full gallon of milk, then bowl with small splash."
  }},
  "removed": {{"count": null, "description": "approx 0.25 cup"}},
  "remaining": {{"count": null, "description": "almost full gallon"}},
  "confidence": "medium"
}}

Return ONLY the raw JSON string.
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
        frames_b64: Optional[List[str]] = None,
        max_tokens: int = 2000,
        temperature: float = 0.3
    ) -> str:
        """Query Qwen3-VL with video or pre-selected frames."""
        messages = [{"role": "system", "content": system_prompt}]

        user_content = []
        use_video_input = False

        if frames_b64:
            # Send pre-selected frames as images
            frame_info = f"[Selected frames: {len(frames_b64)} frames]\n\n"
            user_content.append({"type": "text", "text": frame_info})
            for fb64 in frames_b64:
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{fb64}"}
                })
        elif self.use_video and video_path and video_path.exists():
            video_base64 = self.encode_video_base64(video_path)
            user_content.append({
                "type": "video_url",
                "video_url": {"url": f"data:video/mp4;base64,{video_base64}"}
            })
            use_video_input = True

        user_content.append({"type": "text", "text": user_prompt})
        messages.append({"role": "user", "content": user_content})

        data = {
            "model": QWEN_MODEL,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        if use_video_input:
            data["extra_body"] = {
                "mm_processor_kwargs": {
                    "fps": 3,
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


class GPT5Client:
    """Handles communication with Azure OpenAI GPT-5.2 API with reasoning using frame sampling"""

    def __init__(self, fps: float = 2.0, max_frames: int = 30, reasoning_effort: str = "high"):
        self.fps = fps
        self.max_frames = max_frames
        self.model = "gpt-5.2"
        self.reasoning_effort = reasoning_effort

        # Use Azure OpenAI endpoint 2 for gpt-5.2
        api_key = os.getenv("AZURE_OPENAI_API_KEY_2")
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT_2", "").strip()

        if not api_key or not endpoint:
            raise ValueError("Missing Azure OpenAI API credentials for GPT-5.2. Set AZURE_OPENAI_API_KEY_2 and AZURE_OPENAI_ENDPOINT_2")

        self.client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version="2025-03-01-preview",
        )

    def extract_frames(self, video_path: Path) -> List[str]:
        """Extract frames from video at specified FPS and return as base64 encoded images."""
        import cv2

        frames_b64 = []
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            print(f"    ERROR: Could not open video {video_path}")
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
        frames_b64: Optional[List[str]] = None,
        max_tokens: int = 16000,
        temperature: float = 0.3
    ) -> str:
        """Query GPT-5.2 with reasoning using frames extracted from video or pre-selected frames."""

        messages = [{"role": "system", "content": system_prompt}]
        user_content = []

        if frames_b64:
            # Use pre-selected frames
            frame_info = f"[Selected frames: {len(frames_b64)} frames]\n\n"
            user_content.append({"type": "text", "text": frame_info})
            for fb64 in frames_b64:
                user_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{fb64}",
                        "detail": "low"
                    }
                })
        elif video_path and video_path.exists():
            frames = self.extract_frames(video_path)
            if frames:
                frame_info = f"[Video frames: {len(frames)} frames at {self.fps} FPS]\n\n"
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
            # GPT-5.2 with reasoning - don't set max_tokens, let model decide
            num_frames = len([c for c in user_content if c.get("type") == "image_url"])
            print(f"({num_frames} frames)...", end=" ", flush=True)
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                reasoning_effort=self.reasoning_effort,
            )
            if response.choices and len(response.choices) > 0:
                return response.choices[0].message.content
            return ""
        except Exception as e:
            print(f"ERROR: GPT-5.2 API Error: {e}")
            return ""


class Gemini25Client:
    """Handles communication with Google Gemini 2.5 API using video input"""

    def __init__(self, model_name: str = "gemini-3.0-flash"):
        self.model_name = model_name

        # Load API key from environment
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Missing GOOGLE_API_KEY. Set it in .env file.")

        # Initialize the Gemini client
        self.client = genai.Client(api_key=api_key)
        self._uploaded_files = {}  # Cache for uploaded files

    def upload_video(self, video_path: Path) -> types.File:
        """Upload video to Gemini Files API and wait for processing."""
        video_key = str(video_path)

        # Check cache first
        if video_key in self._uploaded_files:
            return self._uploaded_files[video_key]

        # Upload the file
        print(f"(uploading)...", end=" ", flush=True)
        video_file = self.client.files.upload(file=video_path)

        # Wait for processing to complete
        while video_file.state == "PROCESSING":
            time.sleep(1)
            video_file = self.client.files.get(name=video_file.name)

        if video_file.state == "FAILED":
            raise ValueError(f"Video processing failed: {video_file.name}")

        # Cache the uploaded file
        self._uploaded_files[video_key] = video_file
        return video_file

    def query(
        self,
        system_prompt: str,
        user_prompt: str,
        video_path: Optional[Path] = None,
        frames_b64: Optional[List[str]] = None,
        max_tokens: int = 2000,
        temperature: float = 0.3
    ) -> str:
        """Query Gemini 2.5 Flash with video or pre-selected frames"""
        try:
            contents = []

            if frames_b64:
                # Send pre-selected frames as inline images
                for fb64 in frames_b64:
                    contents.append(types.Part.from_bytes(
                        data=base64.b64decode(fb64),
                        mime_type="image/jpeg"
                    ))
            elif video_path and video_path.exists():
                video_file = self.upload_video(video_path)
                contents.append(types.Part.from_uri(
                    file_uri=video_file.uri,
                    mime_type="video/mp4"
                ))

            # Add the combined prompt (system + user)
            full_prompt = f"{system_prompt}\n\n{user_prompt}"
            contents.append(full_prompt)

            # Make the request
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
            print(f"ERROR: Gemini API Error: {e}")
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


def load_frame_select(frame_select_path: Path) -> Dict[str, Any]:
    """
    Load a frame selection JSON file.

    Expected schema:
    {
      "participant": "P03",
      "frames_dir": "hands23_detection",   // subdirectory under participant output dir
      "selections": [
        {
          "segment_id": "seg_2bac7566",
          "frame_timestamps": [716, 718, 720, 725]   // integer seconds matching tXXX.00s filenames
        }
      ]
    }

    Returns:
        Dict with 'frames_dir' and 'by_segment' (segment_id -> list of timestamps)
    """
    with open(frame_select_path, 'r') as f:
        data = json.load(f)

    by_segment = {}
    for sel in data.get('selections', []):
        seg_id = sel.get('segment_id')
        if seg_id:
            by_segment[seg_id] = sel.get('frame_timestamps', [])

    return {
        'participant': data.get('participant'),
        'frames_dir': data.get('frames_dir', FRAMES_SUBDIR),
        'by_segment': by_segment,
    }


def load_frames_for_segment(
    frames_base_dir: Path,
    video_id: str,
    timestamps: List[int],
) -> List[str]:
    """
    Load specific pre-extracted frames as base64 encoded JPEGs.

    Args:
        frames_base_dir: Base directory containing per-video frame dirs
                         (e.g., outputs/02_inventory/P03/hands23_detection)
        video_id: Video session ID (e.g., P03-20240216-185832)
        timestamps: List of integer timestamps in seconds to load

    Returns:
        List of base64 encoded JPEG strings (in timestamp order)
    """
    frames_dir = frames_base_dir / video_id / "frames"
    if not frames_dir.exists():
        print(f"      WARNING: Frames dir not found: {frames_dir}")
        return []

    frames_b64 = []
    for ts in sorted(timestamps):
        # Match frame filename pattern: frame_XXXXXXXX_tY.00s.jpg
        pattern = f"*_t{ts}.00s.jpg"
        matches = list(frames_dir.glob(pattern))
        if not matches:
            # Try float format (e.g., t716.00s)
            pattern = f"*_t{ts:.2f}s.jpg"
            matches = list(frames_dir.glob(pattern))
        if not matches:
            print(f"      WARNING: No frame found for t={ts}s in {frames_dir}")
            continue

        frame_path = matches[0]
        with open(frame_path, 'rb') as f:
            frame_b64 = base64.b64encode(f.read()).decode('utf-8')
        frames_b64.append(frame_b64)

    return frames_b64


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

    # Normalize to internal format (handles both old and new prompt formats)
    if result:
        # Check if new format (has 'removed' and 'remaining' nested objects)
        removed = result.get('removed', {})
        remaining = result.get('remaining', {})
        reasoning = result.get('reasoning_trace', {})

        # Extract numeric_count: new format uses removed.count, old uses numeric_count
        numeric_count = removed.get('count') if removed else result.get('numeric_count')

        # Extract amount_description: new format uses removed.description
        amount_desc = removed.get('description') if removed else result.get('amount_description')

        # Extract volume_fraction: new format uses removed.volume_fraction
        vol_frac = removed.get('volume_fraction') if removed else result.get('volume_fraction')

        # Extract remaining fields: new format uses remaining.count, etc.
        rem_count = remaining.get('count') if remaining else result.get('remaining_count')
        rem_desc = remaining.get('description') if remaining else result.get('remaining_description')
        rem_frac = remaining.get('volume_fraction') if remaining else result.get('remaining_fraction')

        # Extract visual_evidence: new format uses reasoning_trace.visual_description
        visual_ev = reasoning.get('visual_description') if reasoning else result.get('visual_evidence', '')

        normalized = {
            'quantity_category': result.get('quantity_category', 'unknown'),
            'numeric_count': numeric_count,
            'amount_description': amount_desc,
            'volume_fraction': vol_frac,
            # Remaining quantity fields
            'remaining_count': rem_count,
            'remaining_description': rem_desc,
            'remaining_fraction': rem_frac,
            'unit_type': result.get('unit_type'),
            'confidence': result.get('confidence', 'low'),
            'visual_evidence': visual_ev or '',
            # Blind mode: detected item name
            'detected_item_name': result.get('detected_item_name') or result.get('item_name'),
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
            'remaining_count': None,
            'remaining_description': None,
            'remaining_fraction': None,
            'unit_type': count_match.group(2),
            'confidence': 'low',
            'visual_evidence': 'Extracted from unstructured response'
        }

    return {
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


def evaluate_result(predicted: Dict, ground_truth: Dict, item_match: bool = None) -> Dict[str, Any]:
    """
    Compare predicted result with ground truth.

    Handles both discrete (countable) and continuous (volume/weight) items.

    Args:
        predicted: VLM prediction dict
        ground_truth: Ground truth dict with 'total_count' and 'count_unit'
        item_match: Whether detected item matches expected item (blind mode).
                   If False, match is 'item_mismatch' regardless of count.
                   If None, item match check is skipped (non-blind mode).
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

    # First check: item match (blind mode only)
    # If item doesn't match, mark as item_mismatch regardless of count
    if item_match is False:
        result['match'] = 'item_mismatch'
        # Still compute error for analysis purposes
        if gt_count is not None and pred_count is not None:
            result['error'] = pred_count - gt_count
            result['abs_error'] = abs(pred_count - gt_count)
        else:
            result['error'] = None
        return result

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
    low_only: bool = False,
    test_limit: int = 0,
    verbose: bool = False,
    delete_clips: bool = False,
    skip_existing: bool = False,
    fps: float = 2.0,
    max_frames: int = 30,
    reasoning_effort: str = "high",
    blind_mode: bool = False,
    frame_select: Optional[Dict[str, Any]] = None,
) -> Optional[Path]:
    """
    Process a single participant and return the output file path.

    Returns:
        Path to results file if successful, None otherwise
    """
    participant_dir = output_dir / participant

    # Check if result file already exists
    output_file = participant_dir / f"{participant}_vlm_qa_{tag}_results.json"
    if skip_existing and output_file.exists():
        print(f"SKIP {participant}: {output_file.name} already exists (use without --skip-existing to overwrite)")
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
    print(f"PARTICIPANT: {participant}")
    print(f"{'='*70}")
    print(f"Loaded {len(items)} items from {timeline_file.name}")

    # Collect unique food items for blind mode
    all_food_items = sorted(set(item.get('food_name', '') for item in items if item.get('food_name')))
    if blind_mode:
        print(f"Food items for blind mode: {len(all_food_items)} unique items")

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

    # Sort by difficulty for processing
    difficulty_order = {'LOW': 0, 'MID': 1, 'HIGH': 2, 'UNKNOWN': 3}
    valid_items.sort(key=lambda x: difficulty_order.get(x.get('difficulty', 'UNKNOWN'), 3))

    # Initialize VLM client
    print(f"\nInitializing {model} VLM client...")
    if model == 'gpt4o':
        vlm = GPT4oClient(fps=fps, max_frames=max_frames)
    elif model == 'gpt5':
        vlm = GPT5Client(fps=fps, max_frames=max_frames, reasoning_effort=reasoning_effort)
    elif model == 'gemini':
        vlm = Gemini25Client()
    else:
        vlm = VLMClient(model_name=model, use_video=True)

    # Create temp directory for clips
    clips_dir = participant_dir / "vlm_clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    # Process items
    print(f"\nPROCESSING ITEMS (sorted by difficulty)")

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

            # Get or generate segment_id
            seg_id = segment.get('segment_id')
            if not seg_id and video_id:
                seg_id = generate_segment_id(narr_id, video_id, start_ts, end_ts)

            print(f"    Segment {seg_idx+1}: [{video_id}] {start_ts:.1f}s - {end_ts:.1f}s (gt: {gt_count} {gt_unit or ''})")

            # Check if this segment has a frame selection
            selected_frames_b64 = None
            if frame_select and seg_id in frame_select['by_segment']:
                frame_timestamps = frame_select['by_segment'][seg_id]
                frames_base = participant_dir / frame_select['frames_dir']
                print(f"      Loading {len(frame_timestamps)} selected frames (t={frame_timestamps})...", end=" ", flush=True)
                selected_frames_b64 = load_frames_for_segment(frames_base, video_id, frame_timestamps)
                if not selected_frames_b64:
                    print("FAILED - no frames loaded")
                    continue
                print(f"OK ({len(selected_frames_b64)} frames)")
            elif frame_select:
                # frame_select mode active but this segment has no selection -> skip
                print(f"      SKIP: No frame selection for {seg_id}")
                continue

            # Find video file (only needed when not using frame selection)
            clip_path = None
            if selected_frames_b64 is None:
                if not video_id:
                    print(f"      SKIP: No video ID")
                    continue

                video_path = VIDEO_BASE_DIR / participant / f"{video_id}.mp4"
                if not video_path.exists():
                    print(f"      SKIP: Video not found: {video_path}")
                    continue

                # Extract clip (include video_id in filename to avoid collisions)
                clip_filename = f"{video_id}_seg{seg_idx}_{start_ts:.0f}_{end_ts:.0f}.mp4"
                clip_path = clips_dir / clip_filename

                if not clip_path.exists():
                    print(f"      Extracting clip...", end=" ", flush=True)
                    success = extract_video_clip(video_path, start_ts, end_ts, clip_path)
                    if not success:
                        print("FAILED")
                        continue
                    print(f"OK ({clip_path.stat().st_size / 1024:.1f} KB)")
                else:
                    print(f"      Using cached clip")

            # Query VLM
            print(f"      Querying {model}...", end=" ", flush=True)
            if selected_frames_b64:
                # Frame-select mode: use two-frame prompt (always blind)
                food_items_str = "\n".join(f"  - {name}" for name in all_food_items)
                prompt = TWO_FRAME_ESTIMATION_PROMPT.format(food_items_list=food_items_str)
            elif blind_mode:
                food_items_str = "\n".join(f"  - {name}" for name in all_food_items)
                prompt = ACTION_ESTIMATION_PROMPT_BLIND.format(food_items_list=food_items_str)
            else:
                prompt = ACTION_ESTIMATION_PROMPT.format(item_name=food_name)
            response = vlm.query(
                system_prompt="You are a Visual Inventory Auditor analyzing cooking videos.",
                user_prompt=prompt,
                video_path=clip_path,
                frames_b64=selected_frames_b64,
            )

            if not response:
                print("NO RESPONSE")
                segment_results.append({
                    'segment_id': seg_id,
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
            detected_item = parsed.get('detected_item_name')

            # Display result based on category
            remaining_count = parsed.get('remaining_count')
            remaining_desc = parsed.get('remaining_description')
            remaining_frac = parsed.get('remaining_fraction')

            # Show detected item in blind mode
            detected_str = f" [{detected_item}]" if (blind_mode and detected_item) else ""
            if pred_category == 'discrete' and pred_count is not None:
                remain_str = f", remaining: {remaining_count}" if remaining_count is not None else ""
                print(f"predicted:{detected_str} {pred_count} {pred_unit or 'units'}{remain_str}")
            elif pred_category == 'continuous' and pred_amount:
                remain_str = f", remaining: {remaining_desc or f'{remaining_frac:.0%}' if remaining_frac else ''}" if (remaining_desc or remaining_frac) else ""
                print(f"predicted:{detected_str} {pred_amount} (continuous){remain_str}")
            else:
                print(f"predicted:{detected_str} unknown")

            if verbose:
                print(f"      Evidence: {parsed.get('visual_evidence', '')[:100]}...")

            # Check item match in blind mode using semantic similarity
            item_match = None
            item_similarity = None
            if blind_mode and detected_item:
                item_match, item_similarity = check_item_match(detected_item, food_name)

            seg_result = {
                'segment_id': seg_id,
                'segment_idx': seg_idx,
                'quantity_category': pred_category,
                'predicted_count': pred_count,
                'predicted_amount': pred_amount,
                'predicted_unit': pred_unit,
                # Blind mode: detected item and match
                'detected_item_name': detected_item,
                'item_match': item_match,
                'item_similarity': item_similarity,
                # Remaining quantity fields
                'remaining_count': parsed.get('remaining_count'),
                'remaining_description': parsed.get('remaining_description'),
                'remaining_fraction': parsed.get('remaining_fraction'),
                'confidence': parsed.get('confidence'),
                'visual_evidence': parsed.get('visual_evidence'),
                'clip_path': str(clip_path) if (clip_path and not delete_clips) else None,
                'raw_vlm_response': response,
            }
            # Record which frames were used (frame-select mode)
            if frame_select and seg_id in frame_select['by_segment']:
                seg_result['frame_timestamps'] = frame_select['by_segment'][seg_id]
                seg_result['num_frames'] = len(selected_frames_b64) if selected_frames_b64 else 0

            segment_results.append(seg_result)

            # Clean up clip if requested
            if delete_clips and clip_path and clip_path.exists():
                clip_path.unlink()

        # Aggregate item results
        item_result = {
            'narration_id': narr_id,
            'num_segments': len(segments),
            'segments': segment_results,
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
    if delete_clips:
        try:
            clips_dir.rmdir()
        except OSError:
            pass  # Directory not empty

    # Save results
    print(f"\nSAVING RESULTS")

    output_data = {
        'participant': participant,
        'model': model,
        'tag': tag,
        'low_only': low_only,
        'blind_mode': blind_mode,
        'frame_select_mode': frame_select is not None,
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
    print(f"{'Food':<25} {'Diff':<6} {'GT':<8} {'Predicted':<15} {'Remaining':<15} {'Match':<10}")
    print("-" * 85)

    for r in results:
        food = (r.get('food_name') or '')[:24]
        diff = r.get('difficulty', '?')[:5]
        gt = r.get('total_ground_truth')
        gt_str = str(gt) if gt is not None else '-'

        # Get prediction info from segments
        segments = r.get('segments', [])
        remaining_str = '-'
        if segments:
            first_seg = segments[0]
            cat = first_seg.get('quantity_category', 'unknown')
            if cat == 'discrete':
                pred = r.get('total_predicted')
                pred_str = str(pred) if pred is not None else '-'
                # Get remaining count (from last segment as it's most recent state)
                last_seg = segments[-1]
                rem_count = last_seg.get('remaining_count')
                remaining_str = str(rem_count) if rem_count is not None else '-'
            elif cat == 'continuous':
                pred_str = first_seg.get('predicted_amount', 'continuous')[:13]
                # Get remaining description/fraction
                last_seg = segments[-1]
                rem_desc = last_seg.get('remaining_description')
                rem_frac = last_seg.get('remaining_fraction')
                if rem_desc:
                    remaining_str = rem_desc[:13]
                elif rem_frac is not None:
                    remaining_str = f"{rem_frac:.0%}"
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
            match = f'off {pred_count - gt:+d}'

        print(f"{food:<25} {diff:<6} {gt_str:<8} {pred_str:<15} {remaining_str:<15} {match:<10}")

    return output_file


def load_timeline_segment_lookup(output_dir: Path, participants: List[str]) -> Dict[str, Dict]:
    """
    Load timeline_annotated data for participants and build segment lookup.

    Returns:
        Dict mapping segment_id -> {
            'participant': str,
            'narration_id': str,
            'food_name': str,
            'difficulty': str,
            'video_range': list,
            'total_count': int or None,
            'count_unit': str or None,
            'segment': {...}  # full segment data
        }
    """
    segment_lookup = {}

    for participant in participants:
        timeline_file = output_dir / participant / f"{participant}_timeline_annotated.json"
        if not timeline_file.exists():
            print(f"  WARNING: {timeline_file.name} not found")
            continue

        with open(timeline_file, 'r') as f:
            timeline_data = json.load(f)

        for item in timeline_data.get('items', []):
            narration_id = item.get('narration_id', '')
            for seg in item.get('dispensal_segments', []):
                seg_id = seg.get('segment_id')
                if not seg_id:
                    # Generate segment_id if missing (for migration)
                    video_id = seg.get('video_id', '')
                    start_ts = seg.get('start_timestamp', 0)
                    end_ts = seg.get('end_timestamp', 0)
                    seg_id = generate_segment_id(narration_id, video_id, start_ts, end_ts)
                    seg['segment_id'] = seg_id

                segment_lookup[seg_id] = {
                    'participant': participant,
                    'narration_id': narration_id,
                    'food_name': item.get('food_name', 'unknown'),
                    'difficulty': item.get('difficulty', 'UNKNOWN'),
                    'video_range': item.get('video_range', []),
                    'total_count': item.get('total_count'),
                    'count_unit': item.get('count_unit'),
                    'matched_ingredient_weight': item.get('matched_ingredient_weight'),
                    'segment': seg
                }

    return segment_lookup


def evaluate_failure_cases_output(results_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate failure cases VLM results and generate evaluation report.

    Unlike participant evaluation which filters by LOW difficulty,
    failure cases are already curated so we evaluate all included segments.

    Args:
        results_data: The full results data structure from process_failure_cases

    Returns:
        Evaluation report dict with metrics and details
    """
    items = results_data.get('items', [])

    # Collect segment-level evaluations
    all_segments = []
    item_summaries = []
    by_difficulty = {'LOW': [], 'MID': [], 'HIGH': [], 'UNKNOWN': []}

    for item in items:
        difficulty = item.get('difficulty', 'UNKNOWN').upper()
        segments = item.get('segments', [])

        item_correct = 0
        item_total = 0
        item_abs_errors = []
        seg_details = []

        for seg in segments:
            gt = seg.get('ground_truth_count')
            pred = seg.get('predicted_count')

            # Skip if either is null
            if gt is None or pred is None:
                seg_details.append({
                    'segment_id': seg.get('segment_id'),
                    'case_id': seg.get('case_id'),
                    'ground_truth': gt,
                    'predicted': pred,
                    'is_correct': None,
                    'absolute_error': None,
                    'skipped': True,
                    'notes': seg.get('notes', '')
                })
                continue

            is_correct = (pred == gt)
            abs_error = abs(pred - gt)

            seg_detail = {
                'segment_id': seg.get('segment_id'),
                'case_id': seg.get('case_id'),
                'ground_truth': gt,
                'predicted': pred,
                'unit': seg.get('ground_truth_unit'),
                'is_correct': is_correct,
                'absolute_error': abs_error,
                'skipped': False,
                'notes': seg.get('notes', '')
            }
            seg_details.append(seg_detail)

            segment_record = {
                'is_correct': is_correct,
                'absolute_error': abs_error,
                'difficulty': difficulty
            }
            all_segments.append(segment_record)
            by_difficulty[difficulty].append(segment_record)

            item_total += 1
            if is_correct:
                item_correct += 1
            item_abs_errors.append(abs_error)

        if item_total > 0 or seg_details:
            item_summaries.append({
                'food_name': item.get('food_name'),
                'narration_id': item.get('narration_id'),
                'participant': item.get('participant'),
                'difficulty': difficulty,
                'num_segments': len(segments),
                'num_evaluated': item_total,
                'num_correct': item_correct,
                'accuracy': item_correct / item_total if item_total > 0 else None,
                'mean_abs_error': sum(item_abs_errors) / len(item_abs_errors) if item_abs_errors else None,
                'segments': seg_details
            })

    # Aggregate metrics
    n_total = len(all_segments)
    n_correct = sum(1 for s in all_segments if s['is_correct'])
    total_abs_error = sum(s['absolute_error'] for s in all_segments)

    # By-difficulty breakdown
    difficulty_metrics = {}
    for diff, segs in by_difficulty.items():
        if not segs:
            continue
        n = len(segs)
        correct = sum(1 for s in segs if s['is_correct'])
        abs_err = sum(s['absolute_error'] for s in segs)
        difficulty_metrics[diff] = {
            'n_segments': n,
            'n_correct': correct,
            'mean_accuracy': correct / n if n > 0 else None,
            'mean_absolute_error': abs_err / n if n > 0 else None
        }

    return {
        'generated_at': datetime.now().isoformat(),
        'source': results_data.get('source_file', 'unknown'),
        'vlm_tag': results_data.get('vlm_tag'),
        'vlm_model': results_data.get('vlm_model'),
        'aggregate': {
            'n_items': len(item_summaries),
            'n_segments': n_total,
            'n_correct': n_correct,
            'mean_accuracy': n_correct / n_total if n_total > 0 else None,
            'mean_absolute_error': total_abs_error / n_total if n_total > 0 else None
        },
        'by_difficulty': difficulty_metrics,
        'items': item_summaries
    }


def process_failure_cases(
    failure_cases_file: Path,
    output_dir: Path,
    tag: str,
    model: str,
    verbose: bool = False,
    delete_clips: bool = False,
    fps: float = 2.0,
    max_frames: int = 30,
    reasoning_effort: str = "high",
    blind_mode: bool = False,
) -> Optional[Path]:
    """
    Process a failure_cases JSON file and generate versioned output.

    Args:
        failure_cases_file: Path to failure_cases_{name}.json
        output_dir: Output directory (outputs/02_inventory)
        tag: Tag for output file versioning
        model: VLM model to use
        verbose: Verbose output
        delete_clips: Delete clips after processing
        blind_mode: If True, use prompt without item name

    Returns:
        Path to output file if successful
    """
    # Load failure cases file
    if not failure_cases_file.exists():
        # Try looking in output_dir/failure_cases/ first, then output_dir
        failure_cases_file_fc = output_dir / "failure_cases" / failure_cases_file.name
        failure_cases_file_root = output_dir / failure_cases_file.name
        if failure_cases_file_fc.exists():
            failure_cases_file = failure_cases_file_fc
        elif failure_cases_file_root.exists():
            failure_cases_file = failure_cases_file_root
        if not failure_cases_file.exists():
            print(f"ERROR: Failure cases file not found: {failure_cases_file}")
            return None

    with open(failure_cases_file, 'r') as f:
        failure_data = json.load(f)

    schema_version = failure_data.get('schema_version', 1)
    name = failure_data.get('name', 'unknown')

    # Handle v2 format (reference-based)
    if schema_version == 2:
        cases = failure_data.get('cases', [])
        included_cases = [c for c in cases if c.get('include', True)]

        print(f"\n{'='*70}")
        print(f"FAILURE CASES (v2): {failure_cases_file.name}")
        print(f"{'='*70}")
        print(f"Total cases: {len(cases)}")
        print(f"Included cases: {len(included_cases)}")

        if not included_cases:
            print("No cases to process (all have include=false)")
            return None

        # Collect unique participants
        participants = list(set(c.get('participant', '') for c in included_cases))
        print(f"Participants: {', '.join(participants)}")

        # Load timeline data to resolve segment references
        print(f"\nLoading timeline data...")
        segment_lookup = load_timeline_segment_lookup(output_dir, participants)
        print(f"Loaded {len(segment_lookup)} segments from timeline data")

        # Convert v2 cases to v1-style items for processing
        items = []
        for case in included_cases:
            seg_id = case.get('segment_id')
            if not seg_id or seg_id not in segment_lookup:
                print(f"  WARNING: segment_id {seg_id} not found in timeline data")
                continue

            seg_data = segment_lookup[seg_id]
            segment = seg_data['segment']

            # Build v1-style item with single segment
            item = {
                'narration_id': seg_data['narration_id'],
                'food_name': seg_data['food_name'],
                'difficulty': seg_data['difficulty'],
                'video_range': seg_data['video_range'],
                'total_ground_truth': seg_data['total_count'],
                'total_ground_truth_unit': seg_data['count_unit'],
                'participant': seg_data['participant'],
                'recipe_amount': seg_data.get('matched_ingredient_weight'),
                'segments': [{
                    'segment_id': seg_id,
                    'segment_idx': 0,
                    'video_id': segment.get('video_id'),
                    'start_timestamp': segment.get('start_timestamp'),
                    'end_timestamp': segment.get('end_timestamp'),
                    'ground_truth_count': segment.get('count'),
                    'ground_truth_unit': segment.get('count_unit'),
                    'case_id': case.get('case_id'),
                    'include': case.get('include', True),
                    'priority': case.get('priority', 0),
                    'notes': case.get('notes', ''),
                    'tags': case.get('tags', []),
                }]
            }
            items.append(item)

        total_segments = len(items)
        included_segments = sum(
            1 for item in items
            for seg in item.get('segments', [])
            if seg.get('include', True)
        )
    else:
        # v1 format (nested items->segments)
        items = failure_data.get('items', [])

        # Count total segments and included segments (include is per-segment now)
        total_segments = sum(len(item.get('segments', [])) for item in items)
        included_segments = sum(
            sum(1 for seg in item.get('segments', []) if seg.get('include', True))
            for item in items
        )

    print(f"\n{'='*70}")
    print(f"FAILURE CASES: {failure_cases_file.name}")
    print(f"{'='*70}")
    print(f"Total items: {len(items)}")
    print(f"Total segments: {total_segments}")
    print(f"Included segments: {included_segments}")

    if included_segments == 0:
        print("No segments to process (all have include=false)")
        return None

    # Collect all food items for blind mode (from all involved participants)
    all_food_items = []
    if blind_mode:
        # Get unique participants from items
        item_participants = list(set(item.get('participant', '') for item in items if item.get('participant')))
        print(f"\nCollecting food items for blind mode from {len(item_participants)} participant(s)...")
        for p in item_participants:
            timeline_file = output_dir / p / f"{p}_timeline_annotated.json"
            if timeline_file.exists():
                with open(timeline_file, 'r') as f:
                    p_data = json.load(f)
                p_items = [i.get('food_name', '') for i in p_data.get('items', []) if i.get('food_name')]
                all_food_items.extend(p_items)
        all_food_items = sorted(set(all_food_items))
        print(f"Food items for blind mode: {len(all_food_items)} unique items")

    # Initialize VLM client
    print(f"\nInitializing {model} VLM client...")
    if model == 'gpt4o':
        vlm = GPT4oClient(fps=fps, max_frames=max_frames)
    elif model == 'gpt5':
        vlm = GPT5Client(fps=fps, max_frames=max_frames, reasoning_effort=reasoning_effort)
    elif model == 'gemini':
        vlm = Gemini25Client()
    else:
        vlm = VLMClient(model_name=model, use_video=True)

    # Create clips directory
    clips_dir = output_dir / "failure_case_clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    # Process items
    print(f"\nPROCESSING FAILURE CASES")
    results = []

    for i, item in enumerate(items):
        food_name = item.get('food_name', 'unknown')
        participant = item.get('participant', 'unknown')
        narr_id = item.get('narration_id', '')
        difficulty = item.get('difficulty', 'UNKNOWN')

        # Get segments to process (include is per-segment)
        all_segments = item.get('segments', [])
        # Filter to included segments and sort by priority
        segments_to_process = [
            seg for seg in all_segments
            if seg.get('include', True)
        ]
        segments_to_process.sort(key=lambda s: (s.get('priority', 999), -abs(s.get('error', 0) or 0)))

        if not segments_to_process:
            continue  # Skip items with no included segments

        print(f"\n[{i+1}/{len(items)}] {food_name}")
        print(f"  Participant: {participant}, Difficulty: {difficulty}")
        print(f"  Segments to process: {len(segments_to_process)}/{len(all_segments)}")

        # Process each segment
        segment_results = []
        for seg_idx, segment in enumerate(segments_to_process):
            start_ts = segment.get('start_timestamp', 0)
            end_ts = segment.get('end_timestamp', 0)
            gt_count = segment.get('ground_truth_count')
            gt_unit = segment.get('ground_truth_unit')
            video_id = segment.get('video_id')
            case_id = segment.get('case_id', f'FC{seg_idx+1:03d}')
            seg_notes = segment.get('notes', '')
            seg_priority = segment.get('priority', 0)
            seg_tags = segment.get('tags', [])

            # Get or generate segment_id
            seg_id = segment.get('segment_id')
            if not seg_id and video_id:
                seg_id = generate_segment_id(narr_id, video_id, start_ts, end_ts)

            if not video_id:
                print(f"    [{case_id}] SKIP - No video_id")
                continue

            notes_str = f" | {seg_notes}" if seg_notes else ""
            print(f"    [{case_id}] [{video_id}] {start_ts:.1f}s-{end_ts:.1f}s (gt: {gt_count} {gt_unit or ''}){notes_str}")

            # Find video file
            video_path = VIDEO_BASE_DIR / participant / f"{video_id}.mp4"
            if not video_path.exists():
                print(f"      SKIP: Video not found: {video_path}")
                continue

            # Extract clip
            clip_filename = f"{case_id}_{video_id}_seg{seg_idx}_{start_ts:.0f}_{end_ts:.0f}.mp4"
            clip_path = clips_dir / clip_filename

            if not clip_path.exists():
                print(f"      Extracting clip...", end=" ", flush=True)
                success = extract_video_clip(video_path, start_ts, end_ts, clip_path)
                if not success:
                    print("FAILED")
                    continue
                print(f"OK ({clip_path.stat().st_size / 1024:.1f} KB)")
            else:
                print(f"      Using cached clip")

            # Query VLM
            print(f"      Querying {model}...", end=" ", flush=True)
            if blind_mode:
                # Format food items as bullet list for the prompt
                food_items_str = "\n".join(f"  - {name}" for name in all_food_items)
                prompt = ACTION_ESTIMATION_PROMPT_BLIND.format(food_items_list=food_items_str)
            else:
                prompt = ACTION_ESTIMATION_PROMPT.format(item_name=food_name)
            response = vlm.query(
                system_prompt="You are a Visual Inventory Auditor analyzing cooking videos.",
                user_prompt=prompt,
                video_path=clip_path
            )

            if not response:
                print("NO RESPONSE")
                segment_results.append({
                    'segment_id': seg_id,
                    'case_id': case_id,
                    'segment_idx': segment.get('segment_idx', seg_idx),
                    'video_id': video_id,
                    'start_timestamp': start_ts,
                    'end_timestamp': end_ts,
                    'error': 'No VLM response',
                    'include': True,
                    'priority': seg_priority,
                    'notes': seg_notes,
                    'tags': seg_tags,
                })
                continue

            # Parse response
            parsed = parse_vlm_response(response)
            pred_category = parsed.get('quantity_category', 'unknown')
            pred_count = parsed.get('numeric_count')
            pred_amount = parsed.get('amount_description')
            pred_unit = parsed.get('unit_type')
            detected_item = parsed.get('detected_item_name')

            remaining_count = parsed.get('remaining_count')
            remaining_desc = parsed.get('remaining_description')
            remaining_frac = parsed.get('remaining_fraction')

            # Show detected item in blind mode
            detected_str = f" [{detected_item}]" if (blind_mode and detected_item) else ""
            if pred_category == 'discrete' and pred_count is not None:
                remain_str = f", remaining: {remaining_count}" if remaining_count is not None else ""
                print(f"predicted:{detected_str} {pred_count} {pred_unit or 'units'}{remain_str}")
            elif pred_category == 'continuous' and pred_amount:
                remain_str = f", remaining: {remaining_desc or f'{remaining_frac:.0%}' if remaining_frac else ''}" if (remaining_desc or remaining_frac) else ""
                print(f"predicted:{detected_str} {pred_amount} (continuous){remain_str}")
            else:
                print(f"predicted:{detected_str} unknown")

            if verbose:
                print(f"      Evidence: {parsed.get('visual_evidence', '')[:100]}...")

            # Check item match in blind mode using semantic similarity
            item_match = None
            item_similarity = None
            if blind_mode and detected_item:
                item_match, item_similarity = check_item_match(detected_item, food_name)

            segment_results.append({
                'segment_id': seg_id,
                'case_id': case_id,
                'segment_idx': segment.get('segment_idx', seg_idx),  # preserve original segment_idx
                'quantity_category': pred_category,
                'predicted_count': pred_count,
                'predicted_amount': pred_amount,
                'predicted_unit': pred_unit,
                # Blind mode: detected item and match
                'detected_item_name': detected_item,
                'item_match': item_match,
                'item_similarity': item_similarity,
                'remaining_count': parsed.get('remaining_count'),
                'remaining_description': parsed.get('remaining_description'),
                'remaining_fraction': parsed.get('remaining_fraction'),
                'confidence': parsed.get('confidence'),
                'visual_evidence': parsed.get('visual_evidence'),
                'clip_path': str(clip_path) if not delete_clips else None,
                'include': True,  # Preserve for next iteration
                'priority': seg_priority,
                'notes': seg_notes,
                'tags': seg_tags,
            })

            # Clean up clip if requested
            if delete_clips and clip_path.exists():
                clip_path.unlink()

        # Build item result (preserve original fields + new results)
        item_result = item.copy()
        item_result['segments'] = segment_results
        item_result['rerun_tag'] = tag

        # Calculate total predicted for rerun
        total_predicted = sum(
            s.get('predicted_count', 0) or 0
            for s in segment_results
            if s.get('predicted_count') is not None
        )
        item_result['total_predicted'] = total_predicted if total_predicted > 0 else None

        results.append(item_result)

    # Clean up clips directory if empty and deletion was requested
    if delete_clips:
        try:
            clips_dir.rmdir()
        except OSError:
            pass

    # Build output data (v1 format with full results, not v2 references)
    output_data = {
        'name': name,
        'schema_version': 1,  # Full data format, not reference-based
        'source_file': failure_cases_file.name,
        'vlm_tag': tag,
        'vlm_model': model,
        'blind_mode': blind_mode,
        'processed_at': datetime.now().isoformat(),
        'total_items': len(results),
        'total_segments': sum(len(r.get('segments', [])) for r in results),
        'items': results,
    }

    # Output filename includes the tag for differentiation
    fc_dir = output_dir / "failure_cases"
    fc_dir.mkdir(parents=True, exist_ok=True)
    output_file = fc_dir / f"failure_cases_{name}_{tag}_results.json"

    # Save results
    print(f"\nSAVING RESULTS")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)

    print(f"  Saved: {output_file.name}")

    # Generate evaluation report
    print(f"\nGENERATING EVALUATION REPORT")
    eval_report = evaluate_failure_cases_output(output_data)

    eval_reports_dir = output_dir / "eval_reports"
    eval_reports_dir.mkdir(parents=True, exist_ok=True)
    eval_report_file = eval_reports_dir / f"failure_cases_{name}_{tag}_eval_report.json"
    with open(eval_report_file, 'w', encoding='utf-8') as f:
        json.dump(eval_report, f, indent=2)

    print(f"  Saved: {eval_report_file.name}")

    # Summary
    print(f"\n{'='*70}")
    print(f"EVALUATION SUMMARY")
    print(f"{'='*70}")

    agg = eval_report['aggregate']
    print(f"Items processed: {agg['n_items']}")
    print(f"Segments evaluated: {agg['n_segments']}")
    print(f"Correct predictions: {agg['n_correct']}")
    if agg['mean_accuracy'] is not None:
        print(f"Mean Accuracy: {agg['mean_accuracy']:.1%}")
    if agg['mean_absolute_error'] is not None:
        print(f"Mean Absolute Error: {agg['mean_absolute_error']:.2f}")

    # By difficulty breakdown
    if eval_report.get('by_difficulty'):
        print(f"\nBy Difficulty:")
        print(f"  {'Difficulty':<10} {'Segments':>10} {'Correct':>10} {'Accuracy':>12} {'MAE':>10}")
        print(f"  {'-'*52}")
        for diff in ['LOW', 'MID', 'HIGH']:
            metrics = eval_report['by_difficulty'].get(diff)
            if not metrics:
                continue
            acc = f"{metrics['mean_accuracy']:.1%}" if metrics['mean_accuracy'] is not None else "N/A"
            mae = f"{metrics['mean_absolute_error']:.2f}" if metrics['mean_absolute_error'] is not None else "N/A"
            print(f"  {diff:<10} {metrics['n_segments']:>10} {metrics['n_correct']:>10} {acc:>12} {mae:>10}")

    # Match type summary (from original segments data)
    total_segments = sum(len(r.get('segments', [])) for r in results)
    exact_matches = sum(
        1 for r in results
        for s in r.get('segments', [])
        if s.get('match') == 'exact'
    )
    close_matches = sum(
        1 for r in results
        for s in r.get('segments', [])
        if s.get('match') == 'close'
    )
    wrong_matches = sum(
        1 for r in results
        for s in r.get('segments', [])
        if s.get('match') == 'wrong'
    )

    print(f"\nMatch Types:")
    print(f"  Exact matches: {exact_matches} ({100*exact_matches/max(1,total_segments):.1f}%)")
    print(f"  Close matches (+/-1): {close_matches} ({100*close_matches/max(1,total_segments):.1f}%)")
    print(f"  Wrong matches: {wrong_matches} ({100*wrong_matches/max(1,total_segments):.1f}%)")

    return output_file


def main():
    parser = argparse.ArgumentParser(
        description="VLM Q&A Engine for Quantity Estimation"
    )
    parser.add_argument(
        '--participant',
        help='Participant ID (e.g., P03). Use --all for all participants.'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Process all participants with annotated timelines'
    )
    parser.add_argument(
        '--tag',
        required=True,
        help='Tag for this VLM run (e.g., "qwen_v1", "baseline"). Used in output filenames.'
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
        help='Test mode: only process first N items per participant (0 = all)'
    )
    parser.add_argument(
        '--model',
        default='qwen',
        choices=['qwen', 'gpt4o', 'gpt5', 'gemini'],
        help='VLM model to use (qwen: video input, gpt4o: frame sampling at 2fps, gpt5: GPT-5.2 with reasoning, gemini: Gemini 2.5 Flash)'
    )
    parser.add_argument(
        '--fps',
        type=float,
        default=3.0,
        help='Frames per second for frame-based models (gpt4o, gpt5)'
    )
    parser.add_argument(
        '--max-frames',
        type=int,
        default=40,
        help='Maximum frames to extract per clip (default 30, Azure limit is 50)'
    )
    parser.add_argument(
        '--reasoning',
        default='medium',
        choices=['low', 'medium', 'high'],
        help='Reasoning effort for GPT-5 model'
    )
    parser.add_argument(
        '--low-only',
        action='store_true',
        help='Only process LOW difficulty items'
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
    parser.add_argument(
        '--no-eval',
        action='store_true',
        help='Skip automatic evaluation report generation'
    )
    parser.add_argument(
        '--skip-existing',
        action='store_true',
        help='Skip participant if result file for this tag already exists'
    )
    parser.add_argument(
        '--failure-cases',
        type=str,
        help='Process a failure_cases JSON file instead of timeline_annotated. '
             'Output will be versioned (e.g., failure_cases_{name}_v2.json)'
    )
    parser.add_argument(
        '--blind',
        action='store_true',
        help='Blind mode: do not provide item name to VLM (tests detection capability)'
    )
    parser.add_argument(
        '--frame-select',
        type=str,
        help='Path to frame selection JSON file. When provided, only segments listed '
             'in the file are processed, using the specified frames instead of video clips. '
             'See load_frame_select() docstring for JSON schema.'
    )

    args = parser.parse_args()

    # Load frame selection if provided
    frame_select_data = None
    if args.frame_select:
        frame_select_path = Path(args.frame_select)
        if not frame_select_path.exists():
            print(f"ERROR: Frame selection file not found: {frame_select_path}")
            return 1
        frame_select_data = load_frame_select(frame_select_path)
        n_selections = len(frame_select_data['by_segment'])
        print(f"Loaded frame selections: {n_selections} segments from {frame_select_path.name}")
        print(f"  Frames dir: {frame_select_data['frames_dir']}")

    # Handle failure-cases mode
    if args.failure_cases:
        failure_cases_file = Path(args.failure_cases)
        result_file = process_failure_cases(
            failure_cases_file=failure_cases_file,
            output_dir=args.output_dir,
            tag=args.tag,
            model=args.model,
            verbose=args.verbose,
            delete_clips=args.delete_clips,
            fps=args.fps,
            max_frames=args.max_frames,
            reasoning_effort=args.reasoning,
            blind_mode=args.blind,
        )
        if result_file:
            print(f"\n{'='*70}")
            print("COMPLETE")
            print(f"{'='*70}")
            print(f"Output: {result_file}")
            return 0
        return 1

    # Validate arguments
    if not args.participant and not args.all:
        parser.error("Either --participant or --all must be specified")

    # Determine participants to process
    if args.all:
        participants = find_participants_with_timeline(args.output_dir)
        if not participants:
            print(f"No participants with annotated timelines found in {args.output_dir}")
            return 1
        print(f"Found {len(participants)} participants with annotated timelines: {', '.join(participants)}")
    else:
        participants = [args.participant]

    # Process each participant
    result_files = []
    for participant in participants:
        result_file = process_participant(
            participant=participant,
            output_dir=args.output_dir,
            tag=args.tag,
            model=args.model,
            low_only=args.low_only,
            test_limit=args.test,
            verbose=args.verbose,
            delete_clips=args.delete_clips,
            skip_existing=args.skip_existing,
            fps=args.fps,
            max_frames=args.max_frames,
            reasoning_effort=args.reasoning,
            blind_mode=args.blind,
            frame_select=frame_select_data,
        )
        if result_file:
            result_files.append(result_file)

    # Generate evaluation report
    if result_files and not args.no_eval:
        print(f"\n{'='*70}")
        print("GENERATING EVALUATION REPORT")
        print(f"{'='*70}")

        import importlib
        evaluate_mod = importlib.import_module("08_evaluate_vlm_count")
        evaluate_all = evaluate_mod.evaluate_all
        eval_reports_dir = args.output_dir / "eval_reports"
        eval_reports_dir.mkdir(parents=True, exist_ok=True)
        report_path = eval_reports_dir / f"vlm_qa_{args.tag}_count_eval_report.json"
        report = evaluate_all(args.output_dir, model=args.tag, report_path=report_path)

        if report:
            agg = report['aggregate']
            print(f"\nAggregate Results (LOW difficulty):")
            print(f"  Participants: {agg['n_participants']}")
            print(f"  Segments: {agg['n_segments']}")
            print(f"  Correct: {agg['n_correct']}")
            if agg['mean_accuracy'] is not None:
                print(f"  Mean Accuracy: {agg['mean_accuracy']:.1%}")
            if agg['mean_absolute_error'] is not None:
                print(f"  Mean Absolute Error: {agg['mean_absolute_error']:.2f}")

    print(f"\n{'='*70}")
    print("COMPLETE")
    print(f"{'='*70}")
    print(f"Processed {len(result_files)} participant(s)")
    if result_files:
        for f in result_files:
            print(f"  - {f}")

    return 0


if __name__ == '__main__':
    exit(main())

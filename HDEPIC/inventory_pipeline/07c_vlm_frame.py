#!/usr/bin/env python3
"""
07c_vlm_frame.py - VLM Keyframe Selection for Dispensal Events

Tests VLM capability for finding the most informative "evidence frames"
(source state and destination state) for each dispensal event.

Uses timeline_annotated.json to extract video clips, queries VLM with
KEYFRAME_SELECTION_PROMPT, and converts VLM-output timestamps back to
absolute video timestamps.

Timestamp Alignment:
    - Video-based models (Qwen, Gemini): VLM sees a clip starting at
      clip_start = max(0, segment_start - padding). VLM outputs MM:SS
      relative to clip start. absolute_time = clip_start + parsed_seconds.
    - Frame-based models (GPT-5.2): Frames are extracted from the clip at
      a given FPS. Each frame's timestamp relative to clip is tracked and
      included in the prompt so the VLM can reference them. The same
      clip_start offset converts to absolute time.

Usage:
    python 07c_vlm_frame.py --participant P03 --tag frame_v1
    python 07c_vlm_frame.py --all --tag frame_v1 --model gpt5
    python 07c_vlm_frame.py --participant P03 --test 5 --tag frame_test

Inputs:
    {participant}_timeline_annotated.json
    Video files from data/HD-EPIC/Videos/{participant}/

Outputs:
    {participant}_vlm_qa_{tag}_results.json   (multipath mode, evaluation-compatible)
    {participant}_vlm_frame_{tag}_results.json (keyframe mode)
"""

import argparse
import base64
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Dict, Optional, Any, List, Tuple

import requests

from dotenv import load_dotenv
from openai import AzureOpenAI
from google import genai
from google.genai import types

from inventory_utils import DEFAULT_OUTPUT_DIR, generate_segment_id

# Load environment variables (NeuroTrace root + kitchen root)
load_dotenv(Path(__file__).parent.parent.parent / ".env")
load_dotenv(Path(__file__).parent.parent / ".env", override=True)

# Paths
_SCRIPT_DIR = Path(__file__).parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
VIDEO_BASE_DIR = _PROJECT_ROOT / "data" / "HD-EPIC" / "Videos"

# Qwen3-VL API endpoint
QWEN3VL_URL = "http://saltyfish.eecs.umich.edu:8000/v1/chat/completions"
QWEN_MODEL = "Qwen/Qwen3-VL-30B-A3B-Instruct"

# Default clip padding in seconds
DEFAULT_PADDING = 2.0


# ============================================================
# PROMPT
# ============================================================

KEYFRAME_SELECTION_PROMPT = """You are a Visual Inventory Auditor.
Your goal is to find the two most informative "Evidence Frames" in this video clip that allow you to calculate the change in inventory.

**THE SCENARIO:**
A user is transferring a food item from a Source (e.g., fridge, shelf, bag) to a Destination (e.g., pan, bowl, counter).

**YOUR TASK:**
1.  **Select Timestamps:** Identify the exact timestamps for the "Source State" and "Destination State".
2.  **Define Strategy:** Explain *how* these specific frames will allow you to calculate the count.

**CRITERIA FOR A "GOOD" FRAME:**
* **In-Frame:** The relevant object (container or items) must be within the camera's field of view (not cut off by the edge).
* **Unoccluded:** The relevant items must not be hidden behind the user's hand or other objects.
* **Note:** Motion blur is acceptable as long as the quantity is discernible.

**OUTPUT SCHEMA (Strict JSON):**
{{
  "item_name": "string",

  // THE STRATEGY
  "counting_strategy": "Explain how you will derive the count using these two frames. Example: 'Source view is blocked, so I selected a Destination frame showing the 2 eggs in the pan to count directly.'",

  // FRAME SELECTION
  "source_frame": {{
    "timestamp": "MM:SS",
    "description": "Describe the view. Is the container full? Open?",
    "visibility_status": "clear_view" | "partially_in_frame" | "occluded"
  }},
  "destination_frame": {{
    "timestamp": "MM:SS",
    "description": "Describe the view. Can you see the transferred items?",
    "visibility_status": "clear_view" | "partially_in_frame" | "occluded"
  }}
}}
"""

MULTI_PATH_SELECTION_PROMPT = """You are a Forensic Inventory Auditor.
Your goal is to determine the exact quantity of food moved by finding evidence across three distinct "Visual Paths."

**THE SCENARIO:**
A user moves items from a **Source** to a **Destination**.
The view may be blocked, occluded, or messy. You must find *any* clear frames that prove the count.

**INSTRUCTIONS:**
Scan the video and attempt to complete the three "Evidence Logs" below.
If a path is blocked (e.g., "Source is opaque"), mark it as "INVALID".

**1. SOURCE PATH (The "Subtraction" Method)**
   - Find a clear frame of the Source *Before* the hand enters.
   - Find a clear frame of the Source *After* the hand leaves.
   - *Goal:* Calculate (Start_Count - End_Count).

**2. DESTINATION PATH (The "Addition" Method)**
   - Find a clear frame of the Destination *Before* the item arrives.
   - Find a clear frame of the Destination *After* the item settles.
   - *Goal:* Calculate (End_Count - Start_Count).

**3. TRANSFER PATH (The "In-Flight" Method)**
   - Find the specific moments where the item is *in the air* or *in the hand*.
   - If the action is fragmented (e.g., user takes 2 scoops), find a timestamp for EACH transfer.
   - *Goal:* Count the items visible in the hand/scoop.

**OUTPUT SCHEMA (Strict JSON):**
{{
  "item_name": "string",

  "path_source": {{
    "status": "VALID" | "INVALID_OCCLUDED" | "INVALID_OPAQUE",
    "timestamp_before": "MM:SS",
    "timestamp_after": "MM:SS",
    "observed_delta": <number or null>,
    "confidence": "high" | "medium" | "low"
  }},

  "path_destination": {{
    "status": "VALID" | "INVALID_OCCLUDED" | "INVALID_PILE",
    "timestamp_before": "MM:SS",
    "timestamp_after": "MM:SS",
    "observed_delta": <number or null>,
    "confidence": "high" | "medium" | "low"
  }},

  "path_transfer": {{
    "status": "VALID" | "INVALID_BLURRY" | "INVALID_HIDDEN",
    "transfer_events": [
      {{ "timestamp": "MM:SS", "description": "Scoop 1 in mid-air", "visible_count": 1 }},
      {{ "timestamp": "MM:SS", "description": "Scoop 2 in mid-air", "visible_count": 1 }}
    ],
    "total_transfer_count": <number or null>,
    "confidence": "high" | "medium" | "low"
  }},

  "final_synthesis": {{
    "best_path_selected": "source" | "destination" | "transfer",
    "final_count_estimate": <number>,
    "reasoning": "Source was opaque. Destination was a pile. Transfer view clearly showed 2 distinct scoops."
  }}
}}
"""

KEYFRAME_SELECTION_PROMPT_FRAMES = """You are a Visual Inventory Auditor.
Your goal is to find the two most informative "Evidence Frames" from the provided images that allow you to calculate the change in inventory.

**THE SCENARIO:**
A user is transferring a food item from a Source (e.g., fridge, shelf, bag) to a Destination (e.g., pan, bowl, counter).

**YOUR TASK:**
1.  **Select Frames:** Choose the best "Source State" and "Destination State" frames from the images shown.
2.  **Define Strategy:** Explain *how* these specific frames will allow you to calculate the count.

**FRAME TIMING REFERENCE:**
You are viewing {num_frames} frames extracted from a video clip. Each frame corresponds to:
{frame_timing}

Use the timestamps from the reference above in your output.

**CRITERIA FOR A "GOOD" FRAME:**
* **In-Frame:** The relevant object (container or items) must be within the camera's field of view (not cut off by the edge).
* **Unoccluded:** The relevant items must not be hidden behind the user's hand or other objects.
* **Note:** Motion blur is acceptable as long as the quantity is discernible.

**OUTPUT SCHEMA (Strict JSON):**
{{{{
  "item_name": "string",

  // THE STRATEGY
  "counting_strategy": "Explain how you will derive the count using these two frames.",

  // FRAME SELECTION
  "source_frame": {{{{
    "timestamp": "MM:SS",
    "description": "Describe the view. Is the container full? Open?",
    "visibility_status": "clear_view" | "partially_in_frame" | "occluded"
  }}}},
  "destination_frame": {{{{
    "timestamp": "MM:SS",
    "description": "Describe the view. Can you see the transferred items?",
    "visibility_status": "clear_view" | "partially_in_frame" | "occluded"
  }}}}
}}}}
"""

# MULTI_PATH_SELECTION_PROMPT_FRAMES = """You are a Forensic Inventory Auditor.
# Your goal is to determine the exact quantity of food moved by finding evidence across three distinct "Visual Paths."

# **THE SCENARIO:**
# A user moves items from a **Source** to a **Destination**.
# The view may be blocked, occluded, or messy. You must find *any* clear frames that prove the count.

# **FRAME TIMING REFERENCE:**
# You are viewing {num_frames} frames extracted from a video clip. Each frame corresponds to:
# {frame_timing}

# Use the timestamps from the reference above in your output.

# **INSTRUCTIONS:**
# Scan the frames and attempt to complete the three "Evidence Logs" below.
# If a path is blocked (e.g., "Source is opaque"), mark it as "INVALID".

# **1. SOURCE PATH (The "Subtraction" Method)**
#    - Find a clear frame of the Source *Before* the hand enters.
#    - Find a clear frame of the Source *After* the hand leaves.
#    - *Goal:* Calculate (Start_Count - End_Count).

# **2. DESTINATION PATH (The "Addition" Method)**
#    - Find a clear frame of the Destination *Before* the item arrives.
#    - Find a clear frame of the Destination *After* the item settles.
#    - *Goal:* Calculate (End_Count - Start_Count).

# **3. TRANSFER PATH (The "In-Flight" Method)**
#    - Find the specific moments where the item is *in the air* or *in the hand*.
#    - If the action is fragmented (e.g., user takes 2 scoops), find a timestamp for EACH transfer.
#    - *Goal:* Count the items visible in the hand/scoop.

# **OUTPUT SCHEMA (Strict JSON):**
# {{{{
#   "item_name": "string",

#   "path_source": {{{{
#     "status": "VALID" | "INVALID_OCCLUDED" | "INVALID_OPAQUE",
#     "timestamp_before": "MM:SS",
#     "timestamp_after": "MM:SS",
#     "observed_delta": <number or null>,
#     "confidence": "high" | "medium" | "low"
#   }}}},

#   "path_destination": {{{{
#     "status": "VALID" | "INVALID_OCCLUDED" | "INVALID_PILE",
#     "timestamp_before": "MM:SS",
#     "timestamp_after": "MM:SS",
#     "observed_delta": <number or null>,
#     "confidence": "high" | "medium" | "low"
#   }}}},

#   "path_transfer": {{{{
#     "status": "VALID" | "INVALID_BLURRY" | "INVALID_HIDDEN",
#     "transfer_events": [
#       {{{{ "timestamp": "MM:SS", "description": "Scoop 1 in mid-air", "visible_count": 1 }}}},
#       {{{{ "timestamp": "MM:SS", "description": "Scoop 2 in mid-air", "visible_count": 1 }}}}
#     ],
#     "total_transfer_count": <number or null>,
#     "confidence": "high" | "medium" | "low"
#   }}}},

#   "final_synthesis": {{{{
#     "best_path_selected": "source" | "destination" | "transfer",
#     "final_count_estimate": <number>,
#     "reasoning": "Source was opaque. Destination was a pile. Transfer view clearly showed 2 distinct scoops."
#   }}}}
# }}}}
# """
MULTI_PATH_HYBRID_PROMPT = """You are a Visual Inventory Auditor for kitchen tasks.
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
    "timestamp_before": "MM:SS",
    "timestamp_after": "MM:SS",
    "observed_delta": <number or null>,
    "confidence": "high" | "medium" | "low"
  }},

  "path_destination": {{
    "status": "VALID" | "INVALID_OCCLUDED" | "INVALID_PILE",
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

  // --- PART 2: FINAL ESTIMATION (Baseline Compatible) ---
  // Synthesize the valid paths above into a final reliable number.

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
  
  // Summarize which path you trusted for the final answer
  "visual_evidence": "Source was [Status]. Dest was [Status]. Transfer was [Status]. Final count derived primarily from Path [A/B/C]."
}}

**EXAMPLES:**

*Example 1 (Discrete - Path B Destination Used):*
{{
  "item_name": "eggs",
  "quantity_category": "discrete",
  "path_source": {{
    "status": "INVALID_OCCLUDED", "timestamp_before": "00:02", "timestamp_after": "00:08", 
    "observed_delta": null, "confidence": "low"
  }},
  "path_destination": {{
    "status": "VALID", "timestamp_before": "00:03", "timestamp_after": "00:09", 
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

Return ONLY the raw JSON string. Do not use Markdown.
"""

MULTI_PATH_HYBRID_NO_TRANSFER_PROMPT = """You are a Visual Inventory Auditor for kitchen tasks.
Your goal is to meticulously track the flow of the **Target Food Item** by comparing the "Before" and "After" states of the Source and Destination containers.

**INPUT:**
- Target Item: "{item_name}"
- Input Data: Video clip of a dispensing session (may contain multiple actions).

**VISUAL REASONING STEPS (Perform this "Triangulation" mentally):**

1. **Analyze Path A (Source Subtraction):** Look at the Source container *Before* and *After*. Can you see a count difference? (e.g., 6 eggs -> 4 eggs).

2. **Analyze Path B (Destination Addition):** Look at the Destination container *Before* and *After*. Can you see new items added?

3. **Synthesize:** Compare the two paths. If Source is opaque, trust Destination. If Destination is a pile, trust Source.

**OUTPUT SCHEMA (Strict JSON):**
{{
  "item_name": "{item_name}",
  "quantity_category": "discrete" | "continuous" | "unknown",

  // --- PART 1: EVIDENCE LOGS (4 Keyframes) ---
  // Identify the specific timestamps for the "Global Start" and "Global End" states.
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

  // --- PART 2: FINAL ESTIMATION ---
  // Synthesize the valid paths into a final number.

  // AMOUNT REMOVED (The usage):
  "numeric_count": <integer or null>,      // e.g., 2
  "amount_description": <string or null>,  // e.g., "about half a cup"
  "volume_fraction": <float or null>,      // e.g., 0.5

  // AMOUNT REMAINING (In Source):
  "remaining_count": <integer or null>,       // e.g., 4
  "remaining_description": <string or null>,  // e.g., "about 3/4 full"
  "remaining_fraction": <float or null>,      // e.g., 0.75

  "unit_type": "unit" | "scoop" | "cup" | "splash" | "pinch" | "slice",
  "confidence": "high" | "medium" | "low",

  "visual_evidence": "Source [Status]: Start(6) -> End(4). Dest [Status]: Start(0) -> End(2). Trusting [Source/Dest] for final count."
}}

**EXAMPLES:**

*Example 1 (Multiple Items - Discrete):*
{{
  "item_name": "eggs",
  "quantity_category": "discrete",
  "path_source": {{
    "status": "VALID", "container_description": "Carton",
    "timestamp_before": "00:02", "timestamp_after": "00:15",
    "observed_delta": 2, "confidence": "high"
  }},
  "path_destination": {{
    "status": "INVALID_OCCLUDED", "container_description": "Pan",
    "timestamp_before": "00:03", "timestamp_after": "00:16",
    "observed_delta": null, "confidence": "low"
  }},
  "numeric_count": 2,
  "amount_description": null,
  "remaining_count": 4,
  "unit_type": "unit",
  "confidence": "high",
  "visual_evidence": "Source clearly shows 6 eggs at start and 4 at end. Dest view blocked by hand. Trusting Source Path."
}}

Return ONLY the raw JSON string. Do not use Markdown.
"""

# ============================================================
# TIMESTAMP UTILITIES
# ============================================================

def parse_mmss(timestamp_str: str) -> Optional[float]:
    """
    Parse timestamp string to seconds.

    Supports formats: MM:SS, M:SS, HH:MM:SS, SS, MM:SS.ff

    Returns:
        Seconds as float, or None if parsing fails.
    """
    if not timestamp_str or not isinstance(timestamp_str, str):
        return None

    timestamp_str = timestamp_str.strip()

    # Try HH:MM:SS (Gemini sometimes outputs this)
    m = re.match(r'^(\d{1,2}):(\d{2}):(\d{2}(?:\.\d+)?)$', timestamp_str)
    if m:
        hours = int(m.group(1))
        minutes = int(m.group(2))
        seconds = float(m.group(3))
        return hours * 3600.0 + minutes * 60.0 + seconds

    # Try MM:SS or M:SS (with optional fractional seconds)
    m = re.match(r'^(\d{1,2}):(\d{2}(?:\.\d+)?)$', timestamp_str)
    if m:
        minutes = int(m.group(1))
        seconds = float(m.group(2))
        return minutes * 60.0 + seconds

    # Try plain seconds
    try:
        return float(timestamp_str)
    except ValueError:
        return None


def format_mmss(seconds: float) -> str:
    """Format seconds as MM:SS."""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


def compute_clip_start(segment_start: float, padding: float = DEFAULT_PADDING) -> float:
    """Compute actual clip start time in original video (with padding)."""
    return max(0, segment_start - padding)


def clip_relative_to_absolute(
    timestamp_in_clip: float,
    clip_start: float,
) -> float:
    """Convert a clip-relative timestamp to absolute video timestamp."""
    return clip_start + timestamp_in_clip


# ============================================================
# FRAME EXTRACTION WITH TIMESTAMPS
# ============================================================

def extract_frames_with_timestamps(
    video_path: Path,
    fps: float = 2.0,
    max_frames: int = 30,
) -> Tuple[List[str], List[float]]:
    """
    Extract frames from video at specified FPS and return both
    base64 encoded images and their timestamps (seconds from clip start).

    Args:
        video_path: Path to video clip
        fps: Target frames per second
        max_frames: Maximum number of frames

    Returns:
        Tuple of (frames_b64, timestamps_in_clip)
    """
    import cv2

    frames_b64 = []
    timestamps = []
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"    ERROR: Could not open video {video_path}")
        return [], []

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / video_fps if video_fps > 0 else 0

    expected_frames = int(duration * fps)
    if expected_frames > max_frames:
        effective_fps = max_frames / duration if duration > 0 else fps
        frame_interval = int(video_fps / effective_fps) if effective_fps > 0 else 1
    else:
        frame_interval = int(video_fps / fps) if fps > 0 and video_fps > 0 else 1

    frame_interval = max(1, frame_interval)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            ts_in_clip = frame_idx / video_fps if video_fps > 0 else 0
            timestamps.append(ts_in_clip)

            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            frame_b64 = base64.b64encode(buffer).decode('utf-8')
            frames_b64.append(frame_b64)

            if len(frames_b64) >= max_frames:
                break

        frame_idx += 1

    cap.release()
    return frames_b64, timestamps


def build_frame_timing_text(timestamps: List[float]) -> str:
    """Build human-readable frame timing reference for the prompt."""
    lines = []
    for i, ts in enumerate(timestamps):
        lines.append(f"  Frame {i+1}: {format_mmss(ts)}")
    return "\n".join(lines)


# ============================================================
# VLM CLIENTS (kept from 07_vlm_QA.py)
# ============================================================

class VLMClient:
    """Handles communication with Qwen VLM API (video input)"""

    def __init__(self, model_name: str = 'qwen', use_video: bool = True):
        self.model_name = model_name
        self.use_video = use_video

    def encode_video_base64(self, video_path: Path) -> str:
        with open(video_path, "rb") as f:
            return base64.b64encode(f.read()).decode()

    def query(
        self,
        system_prompt: str,
        user_prompt: str,
        video_path: Optional[Path] = None,
        frames_b64: Optional[List[str]] = None,
        max_tokens: int = 2000,
        temperature: float = 0.3,
    ) -> str:
        messages = [{"role": "system", "content": system_prompt}]
        user_content = []
        use_video_input = False

        if frames_b64:
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
                    "fps": 1,
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


class GPT5Client:
    """Handles communication with Azure OpenAI GPT-5.2 API (frame input)"""

    def __init__(self, fps: float = 2.0, max_frames: int = 30, reasoning_effort: str = "high"):
        self.fps = fps
        self.max_frames = max_frames
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
        video_path: Optional[Path] = None,
        frames_b64: Optional[List[str]] = None,
        max_tokens: int = 16000,
        temperature: float = 0.3,
    ) -> str:
        messages = [{"role": "system", "content": system_prompt}]
        user_content = []

        if frames_b64:
            frame_info = f"[{len(frames_b64)} frames]\n\n"
            user_content.append({"type": "text", "text": frame_info})
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
    """Handles communication with Google Gemini API (video input)"""

    def __init__(self, model_name: str = "gemini-3-flash-preview"):
        self.model_name = model_name

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Missing GOOGLE_API_KEY.")

        self.client = genai.Client(api_key=api_key)
        self._uploaded_files = {}

    def upload_video(self, video_path: Path) -> types.File:
        video_key = str(video_path)
        if video_key in self._uploaded_files:
            return self._uploaded_files[video_key]

        print(f"(uploading)...", end=" ", flush=True)
        video_file = self.client.files.upload(file=video_path)

        while video_file.state == "PROCESSING":
            time.sleep(1)
            video_file = self.client.files.get(name=video_file.name)

        if video_file.state == "FAILED":
            raise ValueError(f"Video processing failed: {video_file.name}")

        self._uploaded_files[video_key] = video_file
        return video_file

    def query(  # noqa: ARG002
        self,
        system_prompt: str,
        user_prompt: str,
        video_path: Optional[Path] = None,
        frames_b64: Optional[List[str]] = None,
        max_tokens: int = 8192,
        temperature: float = 0.3,
    ) -> str:
        try:
            contents = []

            if frames_b64:
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
            print(f"ERROR: Gemini API Error: {e}")
            return ""


# ============================================================
# CLIP EXTRACTION
# ============================================================

def extract_video_clip(
    video_path: Path,
    start_time: float,
    end_time: float,
    output_path: Path,
    padding: float = DEFAULT_PADDING,
) -> Tuple[bool, float]:
    """
    Extract a video clip using ffmpeg.

    Returns:
        Tuple of (success, clip_start_absolute) where clip_start_absolute
        is the actual start time of the clip in the original video.
    """
    clip_start = compute_clip_start(start_time, padding)
    duration = (end_time - start_time) + (2 * padding)

    cmd = [
        "ffmpeg", "-y",
        "-ss", str(clip_start),
        "-i", str(video_path),
        "-t", str(duration),
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-an",
        str(output_path)
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120
        )
        success = result.returncode == 0 and output_path.exists()
        return success, clip_start
    except Exception as e:
        print(f"  ERROR extracting clip: {e}")
        return False, clip_start


# ============================================================
# RESPONSE PARSING
# ============================================================

def parse_keyframe_response(response_text: str) -> Optional[Dict[str, Any]]:
    """
    Parse keyframe selection JSON from VLM response.

    Expected schema:
    {
        "item_name": str,
        "counting_strategy": str,
        "source_frame": {
            "timestamp": "MM:SS",
            "description": str,
            "visibility_status": str
        },
        "destination_frame": {
            "timestamp": "MM:SS",
            "description": str,
            "visibility_status": str
        }
    }

    Returns:
        Parsed dict or None if parsing fails.
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

    if not result:
        return None

    # Validate required fields
    source = result.get('source_frame')
    dest = result.get('destination_frame')

    if not source or not dest:
        return None

    return {
        'item_name': result.get('item_name', ''),
        'counting_strategy': result.get('counting_strategy', ''),
        'source_frame': {
            'timestamp_raw': source.get('timestamp', ''),
            'description': source.get('description', ''),
            'visibility_status': source.get('visibility_status', 'unknown'),
        },
        'destination_frame': {
            'timestamp_raw': dest.get('timestamp', ''),
            'description': dest.get('description', ''),
            'visibility_status': dest.get('visibility_status', 'unknown'),
        },
    }


def resolve_frame_timestamps(
    parsed: Dict[str, Any],
    clip_start: float,
    frame_timestamps: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """
    Convert VLM-output timestamps to absolute video timestamps.

    For video-based models: parse MM:SS directly as clip-relative time.
    For frame-based models: snap to nearest frame timestamp if available,
    then convert via clip_start offset.

    Args:
        parsed: Output from parse_keyframe_response
        clip_start: Absolute start of clip in original video (seconds)
        frame_timestamps: If provided, list of timestamps (seconds from clip
                         start) for each extracted frame. Used to snap VLM
                         output to actual frame times for frame-based models.

    Returns:
        Same dict with added timestamp_seconds and absolute_timestamp fields.
    """
    for key in ('source_frame', 'destination_frame'):
        frame_data = parsed[key]
        raw_ts = frame_data['timestamp_raw']
        ts_in_clip = parse_mmss(raw_ts)

        if ts_in_clip is None:
            frame_data['timestamp_seconds'] = None
            frame_data['absolute_timestamp'] = None
            continue

        # For frame-based models, snap to nearest actual frame timestamp
        if frame_timestamps:
            closest_idx = min(
                range(len(frame_timestamps)),
                key=lambda i: abs(frame_timestamps[i] - ts_in_clip)
            )
            ts_in_clip = frame_timestamps[closest_idx]
            frame_data['snapped_to_frame'] = closest_idx + 1  # 1-indexed

        frame_data['timestamp_seconds'] = round(ts_in_clip, 2)
        frame_data['absolute_timestamp'] = round(
            clip_relative_to_absolute(ts_in_clip, clip_start), 2
        )

    return parsed


def _extract_json(response_text: str) -> Optional[Dict]:
    """Extract JSON object from VLM response text."""
    # Method 1: Extract JSON from markdown code block
    code_block_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response_text)
    if code_block_match:
        try:
            return json.loads(code_block_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Method 2: Greedy match for raw JSON object
    json_match = re.search(r'\{[\s\S]*\}', response_text)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass

    return None


def _parse_path_evidence(result: Dict, require_transfer: bool = True) -> Optional[Dict[str, Any]]:
    """
    Extract path evidence (source/destination/transfer) from parsed JSON.

    Args:
        result: Parsed JSON dict from VLM response.
        require_transfer: If True, path_transfer must be present.
            If False, a missing path_transfer is stored as None.

    Returns dict with path_source, path_destination, path_transfer keys,
    or None if required path fields are missing.
    """
    path_source = result.get('path_source')
    path_dest = result.get('path_destination')
    path_transfer = result.get('path_transfer')

    if not path_source or not path_dest:
        return None
    if require_transfer and not path_transfer:
        return None

    parsed = {
        'path_source': {
            'status': path_source.get('status', 'UNKNOWN'),
            'timestamp_before_raw': path_source.get('timestamp_before', ''),
            'timestamp_after_raw': path_source.get('timestamp_after', ''),
            'observed_delta': path_source.get('observed_delta'),
            'confidence': path_source.get('confidence', 'low'),
            'container_description': path_source.get('container_description', ''),
        },
        'path_destination': {
            'status': path_dest.get('status', 'UNKNOWN'),
            'timestamp_before_raw': path_dest.get('timestamp_before', ''),
            'timestamp_after_raw': path_dest.get('timestamp_after', ''),
            'observed_delta': path_dest.get('observed_delta'),
            'confidence': path_dest.get('confidence', 'low'),
            'container_description': path_dest.get('container_description', ''),
        },
        'path_transfer': None,
    }

    if path_transfer:
        parsed['path_transfer'] = {
            'status': path_transfer.get('status', 'UNKNOWN'),
            'transfer_events': [
                {
                    'timestamp_raw': evt.get('timestamp', ''),
                    'description': evt.get('description', ''),
                    'visible_count': evt.get('visible_count'),
                }
                for evt in (path_transfer.get('transfer_events') or [])
            ],
            'total_transfer_count': path_transfer.get('total_transfer_count'),
            'confidence': path_transfer.get('confidence', 'low'),
        }

    return parsed


def _parse_qa_fields(result: Dict) -> Dict[str, Any]:
    """
    Extract QA-compatible fields from parsed JSON.

    Matches the normalized output of parse_vlm_response() in 07_vlm_QA.py,
    handling both flat format and nested removed/remaining format.
    """
    removed = result.get('removed', {})
    remaining = result.get('remaining', {})
    reasoning = result.get('reasoning_trace', {})

    numeric_count = removed.get('count') if removed else result.get('numeric_count')
    amount_desc = removed.get('description') if removed else result.get('amount_description')
    vol_frac = removed.get('volume_fraction') if removed else result.get('volume_fraction')

    rem_count = remaining.get('count') if remaining else result.get('remaining_count')
    rem_desc = remaining.get('description') if remaining else result.get('remaining_description')
    rem_frac = remaining.get('volume_fraction') if remaining else result.get('remaining_fraction')

    visual_ev = reasoning.get('visual_description') if reasoning else result.get('visual_evidence', '')

    return {
        'quantity_category': result.get('quantity_category', 'unknown'),
        'numeric_count': numeric_count,
        'amount_description': amount_desc,
        'volume_fraction': vol_frac,
        'remaining_count': rem_count,
        'remaining_description': rem_desc,
        'remaining_fraction': rem_frac,
        'unit_type': result.get('unit_type'),
        'confidence': result.get('confidence', 'low'),
        'visual_evidence': visual_ev or '',
    }


def parse_multipath_response(response_text: str, require_transfer: bool = True) -> Optional[Dict[str, Any]]:
    """
    Parse multipath/hybrid JSON from VLM response.

    Handles two output formats:
    - Legacy multipath: has final_synthesis block, no QA fields
    - Hybrid: has flat QA fields (numeric_count, etc.), no final_synthesis

    Args:
        response_text: Raw VLM response string.
        require_transfer: If True, path_transfer must be present.
            Set to False for hybrid_no_transfer mode.

    Returns:
        Parsed dict with path evidence, QA fields, and optional
        final_synthesis, or None if parsing fails.
    """
    result = _extract_json(response_text)
    if not result:
        return None

    # Extract path evidence
    paths = _parse_path_evidence(result, require_transfer=require_transfer)
    if not paths:
        return None

    # Extract QA-compatible fields
    qa = _parse_qa_fields(result)

    # Check for legacy final_synthesis block
    synthesis = result.get('final_synthesis')

    # Build predicted count: prefer QA numeric_count, fall back to synthesis
    pred_count = qa['numeric_count']
    if pred_count is None and synthesis:
        pred_count = synthesis.get('final_count_estimate')
    qa['numeric_count'] = pred_count

    parsed = {
        'item_name': result.get('item_name', ''),
        **paths,
        **qa,
    }

    # Include final_synthesis if present (legacy format)
    if synthesis:
        parsed['final_synthesis'] = {
            'best_path_selected': synthesis.get('best_path_selected', ''),
            'final_count_estimate': synthesis.get('final_count_estimate'),
            'reasoning': synthesis.get('reasoning', ''),
        }

    return parsed


def _resolve_single_timestamp(
    raw_ts: str,
    clip_start: float,
    frame_timestamps: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """Resolve a single MM:SS string to absolute timestamp."""
    ts_in_clip = parse_mmss(raw_ts)
    info: Dict[str, Any] = {'timestamp_raw': raw_ts}
    if ts_in_clip is None:
        info['timestamp_seconds'] = None
        info['absolute_timestamp'] = None
        return info

    if frame_timestamps:
        closest_idx = min(
            range(len(frame_timestamps)),
            key=lambda i: abs(frame_timestamps[i] - ts_in_clip)
        )
        ts_in_clip = frame_timestamps[closest_idx]
        info['snapped_to_frame'] = closest_idx + 1

    info['timestamp_seconds'] = round(ts_in_clip, 2)
    info['absolute_timestamp'] = round(clip_relative_to_absolute(ts_in_clip, clip_start), 2)
    return info


def resolve_multipath_timestamps(
    parsed: Dict[str, Any],
    clip_start: float,
    frame_timestamps: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """
    Resolve all timestamps in a multipath response to absolute video timestamps.
    """
    # Source path
    ps = parsed['path_source']
    for field in ('before', 'after'):
        raw_key = f'timestamp_{field}_raw'
        resolved = _resolve_single_timestamp(ps[raw_key], clip_start, frame_timestamps)
        ps[f'timestamp_{field}_seconds'] = resolved.get('timestamp_seconds')
        ps[f'absolute_timestamp_{field}'] = resolved.get('absolute_timestamp')
        if 'snapped_to_frame' in resolved:
            ps[f'snapped_to_frame_{field}'] = resolved['snapped_to_frame']

    # Destination path
    pd = parsed['path_destination']
    for field in ('before', 'after'):
        raw_key = f'timestamp_{field}_raw'
        resolved = _resolve_single_timestamp(pd[raw_key], clip_start, frame_timestamps)
        pd[f'timestamp_{field}_seconds'] = resolved.get('timestamp_seconds')
        pd[f'absolute_timestamp_{field}'] = resolved.get('absolute_timestamp')
        if 'snapped_to_frame' in resolved:
            pd[f'snapped_to_frame_{field}'] = resolved['snapped_to_frame']

    # Transfer path events (may be None for hybrid_no_transfer)
    pt = parsed.get('path_transfer')
    if pt:
        for evt in pt.get('transfer_events', []):
            resolved = _resolve_single_timestamp(evt['timestamp_raw'], clip_start, frame_timestamps)
            evt['timestamp_seconds'] = resolved.get('timestamp_seconds')
            evt['absolute_timestamp'] = resolved.get('absolute_timestamp')
            if 'snapped_to_frame' in resolved:
                evt['snapped_to_frame'] = resolved['snapped_to_frame']

    return parsed


def evaluate_result(predicted: Dict, ground_truth: Dict) -> Dict[str, Any]:
    """
    Compare predicted result with ground truth.

    Matches the evaluation logic in 07_vlm_QA.py evaluate_result().
    Handles discrete, continuous, and unknown categories.

    Args:
        predicted: Parsed VLM prediction (with numeric_count, quantity_category, etc.)
        ground_truth: Dict with 'total_count' and 'count_unit'

    Returns:
        Dict with predicted_count, ground_truth_count, match, error, and
        all QA-compatible fields.
    """
    gt_count = ground_truth.get('total_count')
    pred_category = predicted.get('quantity_category', 'unknown')
    pred_count = predicted.get('numeric_count')
    pred_amount = predicted.get('amount_description')

    result = {
        'quantity_category': pred_category,
        'predicted_count': pred_count,
        'predicted_amount': pred_amount,
        'predicted_unit': predicted.get('unit_type'),
        'ground_truth_count': gt_count,
        'ground_truth_unit': ground_truth.get('count_unit'),
        'remaining_count': predicted.get('remaining_count'),
        'remaining_description': predicted.get('remaining_description'),
        'remaining_fraction': predicted.get('remaining_fraction'),
        'confidence': predicted.get('confidence'),
        'visual_evidence': predicted.get('visual_evidence'),
    }

    # Treat unknown category with a numeric count as discrete
    if pred_category == 'unknown' and pred_count is not None:
        pred_category = 'discrete'
        result['quantity_category'] = 'discrete'

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


def build_evidence_frames(
    parsed: Dict[str, Any],
    prompt_mode: str,
) -> List[Dict[str, Any]]:
    """
    Build a unified evidence_frames[] list from either keyframe or multipath results.

    For keyframe: produces 2 entries (source, destination).
    For multipath: flattens all resolved timestamps with role labels.
    """
    frames = []

    if prompt_mode == 'keyframe':
        src = parsed.get('source_frame', {})
        dst = parsed.get('destination_frame', {})
        frames.append({
            'role': 'source',
            'timestamp_raw': src.get('timestamp_raw', ''),
            'timestamp_seconds': src.get('timestamp_seconds'),
            'absolute_timestamp': src.get('absolute_timestamp'),
            'description': src.get('description', ''),
            'visibility_status': src.get('visibility_status', 'unknown'),
        })
        frames.append({
            'role': 'destination',
            'timestamp_raw': dst.get('timestamp_raw', ''),
            'timestamp_seconds': dst.get('timestamp_seconds'),
            'absolute_timestamp': dst.get('absolute_timestamp'),
            'description': dst.get('description', ''),
            'visibility_status': dst.get('visibility_status', 'unknown'),
        })

    elif prompt_mode in ('multipath', 'hybrid', 'hybrid_no_transfer'):
        ps = parsed.get('path_source', {})
        pd = parsed.get('path_destination', {})
        pt = parsed.get('path_transfer') or {}

        # Source before/after
        if ps.get('absolute_timestamp_before') is not None:
            frame_entry = {
                'role': 'source_before',
                'timestamp_raw': ps.get('timestamp_before_raw', ''),
                'timestamp_seconds': ps.get('timestamp_before_seconds'),
                'absolute_timestamp': ps.get('absolute_timestamp_before'),
                'description': f"Source before (delta={ps.get('observed_delta')})",
                'visibility_status': 'clear_view' if ps.get('status') == 'VALID' else 'occluded',
            }
            if ps.get('container_description'):
                frame_entry['container_description'] = ps['container_description']
            if ps.get('snapped_to_frame_before') is not None:
                frame_entry['snapped_to_frame'] = ps['snapped_to_frame_before']
            frames.append(frame_entry)
        if ps.get('absolute_timestamp_after') is not None:
            frame_entry = {
                'role': 'source_after',
                'timestamp_raw': ps.get('timestamp_after_raw', ''),
                'timestamp_seconds': ps.get('timestamp_after_seconds'),
                'absolute_timestamp': ps.get('absolute_timestamp_after'),
                'description': f"Source after (delta={ps.get('observed_delta')})",
                'visibility_status': 'clear_view' if ps.get('status') == 'VALID' else 'occluded',
            }
            if ps.get('container_description'):
                frame_entry['container_description'] = ps['container_description']
            if ps.get('snapped_to_frame_after') is not None:
                frame_entry['snapped_to_frame'] = ps['snapped_to_frame_after']
            frames.append(frame_entry)

        # Destination before/after
        if pd.get('absolute_timestamp_before') is not None:
            frame_entry = {
                'role': 'dest_before',
                'timestamp_raw': pd.get('timestamp_before_raw', ''),
                'timestamp_seconds': pd.get('timestamp_before_seconds'),
                'absolute_timestamp': pd.get('absolute_timestamp_before'),
                'description': f"Destination before (delta={pd.get('observed_delta')})",
                'visibility_status': 'clear_view' if pd.get('status') == 'VALID' else 'occluded',
            }
            if pd.get('container_description'):
                frame_entry['container_description'] = pd['container_description']
            if pd.get('snapped_to_frame_before') is not None:
                frame_entry['snapped_to_frame'] = pd['snapped_to_frame_before']
            frames.append(frame_entry)
        if pd.get('absolute_timestamp_after') is not None:
            frame_entry = {
                'role': 'dest_after',
                'timestamp_raw': pd.get('timestamp_after_raw', ''),
                'timestamp_seconds': pd.get('timestamp_after_seconds'),
                'absolute_timestamp': pd.get('absolute_timestamp_after'),
                'description': f"Destination after (delta={pd.get('observed_delta')})",
                'visibility_status': 'clear_view' if pd.get('status') == 'VALID' else 'occluded',
            }
            if pd.get('container_description'):
                frame_entry['container_description'] = pd['container_description']
            if pd.get('snapped_to_frame_after') is not None:
                frame_entry['snapped_to_frame'] = pd['snapped_to_frame_after']
            frames.append(frame_entry)

        # Transfer events
        for evt in pt.get('transfer_events', []):
            if evt.get('absolute_timestamp') is not None:
                frames.append({
                    'role': 'transfer',
                    'timestamp_raw': evt.get('timestamp_raw', ''),
                    'timestamp_seconds': evt.get('timestamp_seconds'),
                    'absolute_timestamp': evt.get('absolute_timestamp'),
                    'description': evt.get('description', ''),
                    'visibility_status': 'clear_view' if pt.get('status') == 'VALID' else 'occluded',
                    'visible_count': evt.get('visible_count'),
                })

    return frames


# ============================================================
# PARTICIPANT PROCESSING
# ============================================================

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


def is_frame_based_model(model: str) -> bool:
    """Check if model uses frame input rather than video."""
    return model in ('gpt4o', 'gpt5')


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
    padding: float = DEFAULT_PADDING,
    prompt_mode: str = "keyframe",
) -> Optional[Path]:
    """Process a single participant for keyframe/multipath selection."""
    participant_dir = output_dir / participant

    if prompt_mode in ('multipath', 'hybrid', 'hybrid_no_transfer'):
        output_file = participant_dir / f"{participant}_vlm_qa_{tag}_results.json"
    else:
        output_file = participant_dir / f"{participant}_vlm_frame_{tag}_results.json"
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
    print(f"PARTICIPANT: {participant}")
    print(f"{'='*70}")
    print(f"Loaded {len(items)} items from {timeline_file.name}")

    # Filter items with valid segments
    valid_items = [
        item for item in items
        if item.get('dispensal_segments') and len(item.get('dispensal_segments', [])) > 0
    ]
    print(f"Items with valid segments: {len(valid_items)}")

    if low_only:
        valid_items = [
            item for item in valid_items
            if item.get('difficulty', '').upper() == 'LOW'
        ]
        print(f"LOW difficulty items: {len(valid_items)}")

    if not valid_items:
        print(f"No items to process for {participant}")
        return None

    if test_limit > 0:
        valid_items = valid_items[:test_limit]
        print(f"TEST MODE: Processing only first {len(valid_items)} items")

    # Sort by difficulty
    difficulty_order = {'LOW': 0, 'MID': 1, 'HIGH': 2, 'UNKNOWN': 3}
    valid_items.sort(key=lambda x: difficulty_order.get(x.get('difficulty', 'UNKNOWN'), 3))

    # Initialize VLM client
    print(f"\nInitializing {model} VLM client...")
    if model == 'gpt5':
        vlm = GPT5Client(fps=fps, max_frames=max_frames, reasoning_effort=reasoning_effort)
    elif model == 'gemini':
        vlm = Gemini25Client()
    else:
        vlm = VLMClient(model_name=model, use_video=True)

    use_frames = is_frame_based_model(model)

    # Create temp directory for clips
    clips_dir = participant_dir / "vlm_clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    # Process items
    print(f"\nPROCESSING ITEMS (model={model}, use_frames={use_frames})")

    results = []
    current_difficulty = None

    for i, item in enumerate(valid_items):
        food_name = item.get('food_name', 'unknown')
        narr_id = item.get('narration_id', '')
        difficulty = item.get('difficulty', 'UNKNOWN')
        video_range = item.get('video_range', [])
        segments = item.get('dispensal_segments', [])

        if difficulty != current_difficulty:
            current_difficulty = difficulty
            print(f"\n--- {difficulty} ---")

        print(f"\n[{i+1}/{len(valid_items)}] {food_name}")
        print(f"  Narration ID: {narr_id}")
        print(f"  Segments: {len(segments)}")

        segment_results = []
        for seg_idx, segment in enumerate(segments):
            start_ts = segment.get('start_timestamp', 0)
            end_ts = segment.get('end_timestamp', 0)
            video_id = segment.get('video_id') or (video_range[0] if video_range else None)

            seg_id = segment.get('segment_id')
            if not seg_id and video_id:
                seg_id = generate_segment_id(narr_id, video_id, start_ts, end_ts)

            if not video_id:
                print(f"    Seg {seg_idx+1}: SKIP - No video ID")
                continue

            print(f"    Seg {seg_idx+1}: [{video_id}] {start_ts:.1f}s - {end_ts:.1f}s")

            # Find video file
            video_path = VIDEO_BASE_DIR / participant / f"{video_id}.mp4"
            if not video_path.exists():
                print(f"      SKIP: Video not found: {video_path}")
                continue

            # Extract clip
            clip_filename = f"{video_id}_seg{seg_idx}_{start_ts:.0f}_{end_ts:.0f}.mp4"
            clip_path = clips_dir / clip_filename
            clip_start = compute_clip_start(start_ts, padding)

            if not clip_path.exists():
                print(f"      Extracting clip...", end=" ", flush=True)
                success, clip_start = extract_video_clip(
                    video_path, start_ts, end_ts, clip_path, padding=padding
                )
                if not success:
                    print("FAILED")
                    continue
                print(f"OK ({clip_path.stat().st_size / 1024:.1f} KB)")
            else:
                print(f"      Using cached clip")

            # Build prompt and extract frames if needed
            frame_timestamps = None
            frames_b64 = None

            if use_frames:
                # Frame-based model: extract frames with timestamps
                print(f"      Extracting frames at {fps} FPS...", end=" ", flush=True)
                frames_b64, frame_timestamps = extract_frames_with_timestamps(
                    clip_path, fps=fps, max_frames=max_frames
                )
                if not frames_b64:
                    print("FAILED - no frames extracted")
                    continue
                print(f"OK ({len(frames_b64)} frames)")

                # Build prompt with frame timing reference
                frame_timing = build_frame_timing_text(frame_timestamps)
                frame_timing_block = (
                    f"\n\n**FRAME TIMING REFERENCE:**\n"
                    f"You are viewing {len(frames_b64)} still frames (NOT a video). "
                    f"Each frame corresponds to:\n{frame_timing}\n"
                    f"Use ONLY timestamps from the list above in your output."
                )
                if prompt_mode == 'hybrid':
                    # No frame-specific variant yet; use video prompt with frames
                    prompt = MULTI_PATH_HYBRID_PROMPT.format(item_name=food_name) + frame_timing_block
                elif prompt_mode == 'hybrid_no_transfer':
                    prompt = MULTI_PATH_HYBRID_NO_TRANSFER_PROMPT.format(item_name=food_name) + frame_timing_block
                elif prompt_mode == 'multipath':
                    prompt = MULTI_PATH_SELECTION_PROMPT + frame_timing_block
                else:
                    prompt = KEYFRAME_SELECTION_PROMPT_FRAMES.format(
                        num_frames=len(frames_b64),
                        frame_timing=frame_timing,
                    )
            else:
                # Video-based model: use standard prompt
                if prompt_mode == 'hybrid':
                    prompt = MULTI_PATH_HYBRID_PROMPT.format(item_name=food_name)
                elif prompt_mode == 'hybrid_no_transfer':
                    prompt = MULTI_PATH_HYBRID_NO_TRANSFER_PROMPT.format(item_name=food_name)
                elif prompt_mode == 'multipath':
                    prompt = MULTI_PATH_SELECTION_PROMPT
                else:
                    prompt = KEYFRAME_SELECTION_PROMPT

            # Query VLM
            if use_frames and frame_timestamps:
                print(f"      Frame timing sent to VLM ({len(frame_timestamps)} frames):")
                for fi, ft in enumerate(frame_timestamps):
                    print(f"        Frame {fi+1}: {format_mmss(ft)} ({ft:.2f}s clip-relative, {ft + clip_start:.2f}s absolute)")
            print(f"      Querying {model}...", end=" ", flush=True)
            response = vlm.query(
                system_prompt="You are a Visual Inventory Auditor analyzing cooking videos.",
                user_prompt=prompt,
                video_path=clip_path if not use_frames else None,
                frames_b64=frames_b64,
            )

            if not response:
                print("NO RESPONSE")
                segment_results.append({
                    'segment_id': seg_id,
                    'segment_idx': seg_idx,
                    'video_id': video_id,
                    'start_timestamp': start_ts,
                    'end_timestamp': end_ts,
                    'clip_start': clip_start,
                    'error': 'No VLM response',
                })
                continue

            # Parse response based on prompt mode
            if prompt_mode in ('multipath', 'hybrid', 'hybrid_no_transfer'):
                parsed = parse_multipath_response(
                    response,
                    require_transfer=(prompt_mode != 'hybrid_no_transfer'),
                )
            else:
                parsed = parse_keyframe_response(response)

            # Log raw timestamps from VLM response
            if parsed and use_frames:
                if prompt_mode in ('multipath', 'hybrid', 'hybrid_no_transfer'):
                    ps_raw = parsed.get('path_source', {})
                    pd_raw = parsed.get('path_destination', {})
                    last_frame_ts = frame_timestamps[-1] if frame_timestamps else None
                    print(f"      VLM raw timestamps (last frame={format_mmss(last_frame_ts) if last_frame_ts is not None else '?'}):")
                    print(f"        Source:  before={ps_raw.get('timestamp_before_raw')}  after={ps_raw.get('timestamp_after_raw')}")
                    print(f"        Dest:    before={pd_raw.get('timestamp_before_raw')}  after={pd_raw.get('timestamp_after_raw')}")

            if not parsed:
                print("PARSE FAILED")
                if verbose:
                    print(f"      Raw: {response[:200]}...")
                segment_results.append({
                    'segment_id': seg_id,
                    'segment_idx': seg_idx,
                    'video_id': video_id,
                    'start_timestamp': start_ts,
                    'end_timestamp': end_ts,
                    'clip_start': clip_start,
                    'error': f'Failed to parse {prompt_mode} response',
                    'raw_vlm_response': response,
                })
                continue

            # Resolve timestamps: clip-relative -> absolute video time
            if prompt_mode in ('multipath', 'hybrid', 'hybrid_no_transfer'):
                parsed = resolve_multipath_timestamps(
                    parsed,
                    clip_start=clip_start,
                    frame_timestamps=frame_timestamps,
                )
            else:
                parsed = resolve_frame_timestamps(
                    parsed,
                    clip_start=clip_start,
                    frame_timestamps=frame_timestamps,
                )

            # Build unified evidence_frames
            evidence_frames = build_evidence_frames(parsed, prompt_mode)

            # Print summary
            print(f"OK")
            if prompt_mode == 'keyframe':
                src = parsed['source_frame']
                dst = parsed['destination_frame']
                src_abs = src.get('absolute_timestamp')
                dst_abs = dst.get('absolute_timestamp')
                print(f"      Source:  {src['timestamp_raw']} -> {format_mmss(src_abs) if src_abs else '?'}  "
                      f"(abs: {src_abs:.1f}s)  [{src['visibility_status']}]" if src_abs else
                      f"      Source:  {src['timestamp_raw']} -> FAILED  [{src['visibility_status']}]")
                print(f"      Dest:    {dst['timestamp_raw']} -> {format_mmss(dst_abs) if dst_abs else '?'}  "
                      f"(abs: {dst_abs:.1f}s)  [{dst['visibility_status']}]" if dst_abs else
                      f"      Dest:    {dst['timestamp_raw']} -> FAILED  [{dst['visibility_status']}]")
                print(f"      Strategy: {parsed.get('counting_strategy', '')[:80]}")
            else:
                xfer_status = parsed['path_transfer']['status'] if parsed.get('path_transfer') else 'N/A'
                print(f"      Paths: src={parsed['path_source']['status']} "
                      f"dst={parsed['path_destination']['status']} "
                      f"xfer={xfer_status}")
                pred_count = parsed.get('numeric_count')
                print(f"      Predicted: {pred_count}  "
                      f"Category: {parsed.get('quantity_category', '?')}")
                print(f"      Evidence frames: {len(evidence_frames)}")

            seg_result = {
                'segment_id': seg_id,
                'segment_idx': seg_idx,
                'clip_start': clip_start,
                'item_name': parsed.get('item_name', ''),
                'evidence_frames': evidence_frames,
                'raw_vlm_response': response,
            }

            # For multipath: add prediction fields (GT comes from timeline_annotated at eval time)
            if prompt_mode in ('multipath', 'hybrid', 'hybrid_no_transfer'):
                seg_result.update({
                    'quantity_category': parsed.get('quantity_category'),
                    'predicted_count': parsed.get('numeric_count'),
                    'predicted_amount': parsed.get('amount_description'),
                    'predicted_unit': parsed.get('unit_type'),
                    'remaining_count': parsed.get('remaining_count'),
                    'remaining_description': parsed.get('remaining_description'),
                    'remaining_fraction': parsed.get('remaining_fraction'),
                    'confidence': parsed.get('confidence'),
                    'visual_evidence': parsed.get('visual_evidence'),
                })

            # Add prompt-mode-specific fields
            if prompt_mode == 'keyframe':
                seg_result['counting_strategy'] = parsed.get('counting_strategy', '')
                seg_result['source_frame'] = parsed['source_frame']
                seg_result['destination_frame'] = parsed['destination_frame']
            else:
                paths_dict = {
                    'source': parsed['path_source'],
                    'destination': parsed['path_destination'],
                }
                if parsed.get('path_transfer') is not None:
                    paths_dict['transfer'] = parsed['path_transfer']
                seg_result['paths'] = paths_dict
                # Include final_synthesis if present (legacy multipath format)
                if parsed.get('final_synthesis'):
                    seg_result['final_synthesis'] = parsed['final_synthesis']

            # Record frame info for frame-based models
            if frame_timestamps:
                seg_result['num_frames'] = len(frames_b64)
                seg_result['frame_fps'] = fps

            segment_results.append(seg_result)

            # Clean up clip if requested
            if delete_clips and clip_path.exists():
                clip_path.unlink()

        # Aggregate item result
        item_result = {
            'narration_id': narr_id,
            'num_segments': len(segments),
            'segments': segment_results,
        }

        # For multipath: add item-level prediction summary
        if prompt_mode in ('multipath', 'hybrid', 'hybrid_no_transfer'):
            total_predicted = sum(
                s.get('predicted_count', 0) or 0
                for s in segment_results
                if s.get('predicted_count') is not None
            )
            item_result['total_predicted'] = total_predicted if total_predicted > 0 else None
        results.append(item_result)

    # Clean up clips directory if empty
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
        'prompt_mode': prompt_mode,
        'task': 'keyframe_selection' if prompt_mode == 'keyframe' else 'multipath_selection' if prompt_mode == 'multipath' else 'hybrid_no_transfer_selection' if prompt_mode == 'hybrid_no_transfer' else 'hybrid_selection',
        'low_only': low_only,
        'padding': padding,
        'fps': fps if use_frames else None,
        'max_frames': max_frames if use_frames else None,
        'total_items': len(valid_items),
        'items': results,
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)

    print(f"  Saved: {output_file.name}")

    # Summary table
    print(f"\n{'='*70}")
    print(f"RESULTS TABLE")
    print(f"{'='*70}")
    print(f"{'Food':<25} {'Diff':<6} {'Segs':<5} {'Src OK':<8} {'Dst OK':<8}")
    print("-" * 55)

    for r in results:
        food = (r.get('food_name') or '')[:24]
        diff = r.get('difficulty', '?')[:5]
        segs = r.get('segments', [])
        n_segs = len(segs)
        src_ok = sum(1 for s in segs if s.get('source_frame', {}).get('absolute_timestamp') is not None)
        dst_ok = sum(1 for s in segs if s.get('destination_frame', {}).get('absolute_timestamp') is not None)

        print(f"{food:<25} {diff:<6} {n_segs:<5} {src_ok}/{n_segs:<6} {dst_ok}/{n_segs:<6}")

    return output_file


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="VLM Keyframe Selection for Dispensal Events"
    )
    parser.add_argument('--participant', help='Participant ID (e.g., P03)')
    parser.add_argument('--all', action='store_true', help='Process all participants')
    parser.add_argument('--tag', required=True, help='Tag for this run (used in output filename)')
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--test', type=int, default=0, help='Process first N items only')
    parser.add_argument(
        '--model', default='qwen',
        choices=['qwen', 'gpt5', 'gemini'],
        help='VLM model (qwen/gemini: video input, gpt5: frame sampling)'
    )
    parser.add_argument('--fps', type=float, default=1.0,
                        help='FPS for frame extraction (frame-based models)')
    parser.add_argument('--max-frames', type=int, default=40,
                        help='Max frames per clip (frame-based models)')
    parser.add_argument('--reasoning', default='medium',
                        choices=['low', 'medium', 'high'],
                        help='Reasoning effort for GPT-5')
    parser.add_argument('--padding', type=float, default=DEFAULT_PADDING,
                        help='Padding in seconds around segment for clip extraction')
    parser.add_argument('--prompt', default='keyframe',
                        choices=['keyframe', 'multipath', 'hybrid', 'hybrid_no_transfer'],
                        help='Prompt mode: keyframe (2-frame), multipath (3-path forensic), hybrid (3-path + QA fields), or hybrid_no_transfer (2-path source+dest + QA fields)')
    parser.add_argument('--low-only', action='store_true',
                        help='Only process LOW difficulty items')
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--delete-clips', action='store_true',
                        help='Delete clips after processing')
    parser.add_argument('--skip-existing', action='store_true',
                        help='Skip if result file exists')

    args = parser.parse_args()

    if not args.participant and not args.all:
        parser.error("Either --participant or --all must be specified")

    if args.all:
        participants = find_participants_with_timeline(args.output_dir)
        if not participants:
            print(f"No participants with annotated timelines found in {args.output_dir}")
            return 1
        print(f"Found {len(participants)} participants: {', '.join(participants)}")
    else:
        participants = [args.participant]

    result_files = []
    for p in participants:
        result_file = process_participant(
            participant=p,
            output_dir=args.output_dir,
            tag=args.tag,
            model=args.model,
            #low_only=args.low_only,
            low_only=True,
            test_limit=args.test,
            verbose=args.verbose,
            delete_clips=args.delete_clips,
            skip_existing=args.skip_existing,
            fps=args.fps,
            max_frames=args.max_frames,
            reasoning_effort=args.reasoning,
            padding=args.padding,
            prompt_mode=args.prompt,
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

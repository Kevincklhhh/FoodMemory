#!/usr/bin/env python3
"""
VLM Perception Module (VLM Call 1)
Context-free food detection using Vision-Language Model.
"""

import base64
import requests
import json
import re
from pathlib import Path
from typing import List, Dict, Optional


class VLMPerception:
    """
    Handles context-free VLM calls for initial food detection.
    This is VLM Call 1 - identifies food items without memory context.
    """

    def __init__(self, api_url: str = "http://localhost:8000/v1/chat/completions",
                 model: str = "Qwen3-VL-30B", fps: int = 2, logger=None):
        """
        Initialize VLM Perception module.

        Args:
            api_url: URL of the vLLM API endpoint
            model: Model name to use
            fps: Frames per second for video sampling
            logger: VLMLogger instance for logging
        """
        self.api_url = api_url
        self.model = model
        self.fps = fps
        self.logger = logger

    def encode_video(self, video_path: str) -> str:
        """Encode video file to base64."""
        with open(video_path, "rb") as f:
            return base64.b64encode(f.read()).decode()

    def detect_food_items(self, video_path: str, video_name: str = "", clip_index: int = 0) -> Optional[Dict]:
        """
        Detect food items in a video clip without any memory context.

        Args:
            video_path: Path to video clip file
            video_name: Parent video name (for logging)
            clip_index: Clip index (for logging)

        Returns:
            Dictionary with detected food names, or None on error
            Format: {"food_names": ["milk", "chicken", ...]}
        """
        print(f"\n[VLM Call 1] Detecting food items in: {Path(video_path).name}")

        if not video_name:
            video_name = Path(video_path).stem

        # Encode video
        video_base64 = self.encode_video(video_path)

        # Construct prompt for context-free detection
        prompt = """You are a food tracker. Analyze this egocentric video clip and identify all food items the user *directly interacts with*.

IMPORTANT RULES:
- Only identify food items that the user physically touches, grasps, or manipulates
- Ignore food in the background that is not being interacted with
- Focus on direct physical contact with food items
- Be specific about the food item (e.g., "whole milk" not just "milk", "raw chicken breast" not just "chicken")

Output ONLY a JSON object in this exact format:
{
  "food_names": ["specific food item 1", "specific food item 2", ...]
}

If no food interactions are detected, return:
{
  "food_names": []
}
"""

        # Prepare API request
        data = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video_url",
                            "video_url": {
                                "url": f"data:video/mp4;base64,{video_base64}"
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ],
            "max_tokens": 300,
            "temperature": 0.3,
            "extra_body": {
                "mm_processor_kwargs": {
                    "fps": self.fps,
                    "do_sample_frames": True
                }
            }
        }

        try:
            # Make API call
            response = requests.post(
                self.api_url,
                headers={"Content-Type": "application/json"},
                json=data,
                timeout=180
            )
            response.raise_for_status()

            result = response.json()

            # Extract content
            if "choices" in result and len(result["choices"]) > 0:
                content = result["choices"][0]["message"]["content"]
                print(f"  Raw response: {content[:200]}...")

                # Parse JSON from response
                parsed = self._parse_json_response(content)

                if parsed and "food_names" in parsed:
                    food_names = parsed["food_names"]
                    print(f"  ✓ Detected {len(food_names)} food items: {food_names}")

                    # Log VLM Call 1
                    if self.logger:
                        self.logger.log_vlm_call_1(
                            clip_path=video_path,
                            prompt=prompt,
                            detected_food=food_names,
                            raw_response=content,
                            video_name=video_name,
                            clip_index=clip_index
                        )

                    return parsed
                else:
                    print(f"  ⚠ Could not parse food_names from response")
                    return {"food_names": []}

            else:
                print(f"  ⚠ Unexpected response format")
                return {"food_names": []}

        except requests.exceptions.RequestException as e:
            print(f"  ✗ API Error: {e}")
            return None
        except Exception as e:
            print(f"  ✗ Error: {e}")
            return None

    def _parse_json_response(self, content: str) -> Optional[Dict]:
        """
        Parse JSON from VLM response, handling markdown code blocks.

        Args:
            content: Raw response content

        Returns:
            Parsed JSON dictionary or None
        """
        # Try to extract JSON from markdown code blocks
        json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find JSON object directly
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                json_str = content

        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return None


if __name__ == "__main__":
    # Test the VLMPerception module
    print("Testing VLMPerception module...\n")

    vlm = VLMPerception()

    # Test with a clip (if exists)
    test_clip = "test_clips/20251029-193737_clip_000.mp4"

    if Path(test_clip).exists():
        print(f"Testing detection on: {test_clip}\n")
        result = vlm.detect_food_items(test_clip)

        if result:
            print(f"\n{'='*60}")
            print("Detection Result:")
            print(json.dumps(result, indent=2))
            print('='*60)
    else:
        print(f"Test clip not found: {test_clip}")
        print("Run video_processor.py first to generate test clips")

    print("\n✓ VLMPerception module test complete")

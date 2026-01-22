#!/usr/bin/env python3
"""
VLM Update Module (VLM Call 2)
Context-aware food interaction analysis with memory context.
Handles both acquisition (un-bagging) and standard interaction scenarios.
"""

import base64
import requests
import json
import re
from pathlib import Path
from typing import List, Dict, Optional


class VLMUpdate:
    """
    Handles context-aware VLM calls for food interaction analysis.
    This is VLM Call 2 - uses memory context and receipt expectations.
    """

    def __init__(self, api_url: str = "http://localhost:8000/v1/chat/completions",
                 model: str = "Qwen3-VL-30B", fps: int = 2, logger=None):
        """
        Initialize VLM Update module.

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

    def analyze_interactions(self, video_path: str, context_block: str,
                            expected_items: Optional[List[str]] = None,
                            is_unbagging: bool = False,
                            video_name: str = "", clip_index: int = 0) -> Optional[List[Dict]]:
        """
        Analyze food interactions with memory context.

        Args:
            video_path: Path to video clip file
            context_block: Context from MemoryRetriever
            expected_items: List of expected items (for un-bagging videos)
            is_unbagging: Whether this is an un-bagging video
            video_name: Parent video name (for logging)
            clip_index: Clip index (for logging)

        Returns:
            List of update commands, or None on error
            Format: [
                {'command': 'CREATE', 'data': {...}},
                {'command': 'UPDATE', 'food_id': '...', 'data': {...}}
            ]
        """
        print(f"\n[VLM Call 2] Analyzing interactions in: {Path(video_path).name}")
        if is_unbagging:
            print(f"  Mode: UN-BAGGING (expecting {len(expected_items or [])} items)")
        else:
            print(f"  Mode: STANDARD INTERACTION")

        if not video_name:
            video_name = Path(video_path).stem

        # Encode video
        video_base64 = self.encode_video(video_path)

        # Construct appropriate prompt
        if is_unbagging:
            prompt = self._create_acquisition_prompt(context_block, expected_items)
        else:
            prompt = self._create_standard_prompt(context_block)

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
            "max_tokens": 1000,
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

                if parsed and "commands" in parsed:
                    commands = parsed["commands"]
                    print(f"  ✓ Generated {len(commands)} update commands")

                    # Log VLM Call 2
                    if self.logger:
                        self.logger.log_vlm_call_2(
                            clip_path=video_path,
                            prompt=prompt,
                            context_block=context_block,
                            commands=commands,
                            raw_response=content,
                            video_name=video_name,
                            clip_index=clip_index,
                            is_unbagging=is_unbagging,
                            expected_items=expected_items
                        )

                    return commands
                else:
                    print(f"  ⚠ Could not parse commands from response")
                    return []

            else:
                print(f"  ⚠ Unexpected response format")
                return []

        except requests.exceptions.RequestException as e:
            print(f"  ✗ API Error: {e}")
            return None
        except Exception as e:
            print(f"  ✗ Error: {e}")
            return None

    def _create_acquisition_prompt(self, context_block: str,
                                   expected_items: List[str]) -> str:
        """Create prompt for un-bagging (acquisition) scenario."""
        items_list = "\n".join(f"  - {item}" for item in expected_items)

        prompt = f"""You are a food memory manager. The user is UN-BAGGING GROCERIES in this video.

EXPECTED ITEMS FROM RECEIPT:
{items_list}

{context_block}

TASK:
Analyze the video and identify each food item being un-bagged. For each item:
1. Check if it matches an expected item from the receipt
2. Check if it matches any existing item in the memory context
3. Decide whether to CREATE a new item or UPDATE an existing one
4. Extract visual embedding (use a placeholder for now: [0.0, 0.0, 0.0])

Output a JSON object with a list of commands:

{{
  "commands": [
    {{
      "command": "CREATE",
      "data": {{
        "primary_label": "Kroger Whole Milk Gallon",
        "description": "Gallon of whole milk being un-bagged",
        "visual_embedding": [0.0, 0.0, 0.0],
        "location": "counter",
        "quantity": "1 gallon",
        "action": "purchased and un-bagged"
      }}
    }},
    {{
      "command": "UPDATE",
      "food_id": "existing_item_id_from_context",
      "data": {{
        "location": "counter",
        "quantity": "updated quantity",
        "action": "retrieved from bag"
      }}
    }}
  ]
}}

RULES:
- CREATE new items for items not in memory
- UPDATE existing items if they match context
- Set initial location to "counter" or "table" (where un-bagging happens)
- Be specific with labels (match receipt items)
- All items should have an action describing what's happening
"""
        return prompt

    def _create_standard_prompt(self, context_block: str) -> str:
        """Create prompt for standard interaction scenario."""
        prompt = f"""You are a food memory manager. Analyze this video clip showing food interactions.

{context_block}

TASK:
1. Match the food items in the video to their IDs from the memory context above
2. Identify the interaction (e.g., "retrieve from fridge", "place in fridge", "cook", "eat", "store leftover")
3. Determine the new state (location, quantity)
4. Output commands to update the memory

Output a JSON object with a list of commands:

{{
  "commands": [
    {{
      "command": "UPDATE",
      "food_id": "milk_abc123",
      "data": {{
        "location": "counter",
        "quantity": "half-full",
        "action": "retrieve from fridge"
      }}
    }},
    {{
      "command": "CREATE",
      "data": {{
        "primary_label": "Leftover Pasta",
        "description": "Container of leftover pasta",
        "visual_embedding": [0.0, 0.0, 0.0],
        "location": "counter",
        "quantity": "1 container",
        "action": "created from cooking"
      }}
    }}
  ]
}}

RULES:
- UPDATE existing items from context when you see them
- CREATE new items only if they don't match anything in context
- Be specific about actions (what is the user doing with the food?)
- Update location and quantity based on what you observe
- If no interactions detected, return empty commands list: {{"commands": []}}
"""
        return prompt

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
    # Test the VLMUpdate module
    print("Testing VLMUpdate module...\n")

    vlm = VLMUpdate()

    # Test with a clip (if exists)
    test_clip = "test_clips/20251029-193737_clip_000.mp4"

    if Path(test_clip).exists():
        # Test context
        test_context = """CONTEXT: The food memory contains the following potentially relevant items:

1. Kroger Whole Milk Gallon (ID: milk_test123)
   Description: Gallon of whole milk
   Current Location: fridge
   Current Quantity: full
   Purchased: 2025-10-28T10:00:00
"""

        # Test un-bagging scenario
        test_expected_items = [
            "Fresh Bunch of Organic Bananas",
            "Garlic",
            "Raw Shrimp",
            "Whole Milk Gallon",
            "Firm Tofu"
        ]

        print("Testing UN-BAGGING scenario:\n")
        commands = vlm.analyze_interactions(
            test_clip,
            test_context,
            expected_items=test_expected_items,
            is_unbagging=True
        )

        if commands:
            print(f"\n{'='*60}")
            print("Generated Commands:")
            print(json.dumps({"commands": commands}, indent=2))
            print('='*60)
    else:
        print(f"Test clip not found: {test_clip}")
        print("Run video_processor.py first to generate test clips")

    print("\n✓ VLMUpdate module test complete")

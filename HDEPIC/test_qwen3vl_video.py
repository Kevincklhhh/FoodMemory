#!/usr/bin/env python3
import base64
import requests
import json
import re
from pathlib import Path

# Read and encode the video
video_path = "/Users/kailaicui/Downloads/kitchen_videos/20251021-221844.MOV"
with open(video_path, "rb") as f:
    video_base64 = base64.b64encode(f.read()).decode()

# Prepare the request for video analysis
url = "http://saltyfish.eecs.umich.edu:8000/v1/chat/completions"
#url = "http://141.212.114.81:8000/v1/chat/completions"
headers = {"Content-Type": "application/json"}


prompt = """This is an egocentric video from a first-person perspective.

Please analyze this video and find ONE frame where a food item has a clear, unobstructed, and complete view during user interaction.

Locate the food item with the clearest, most complete view and report the result in JSON format like this:
{
  "timestamp": <time in seconds>,
  "label": "<food item name>",
  "visibility_quality": "<description of why the view is clear>",
}

- Only provide ONE detection
- Choose a frame where the food item is clearly being interacted with AND has a complete, unobstructed view
- The food item should be fully visible without being blocked by hands, containers, or other objects

"""

data = {
    "model": "Qwen/Qwen3-VL-30B-A3B-Instruct",
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
    "max_tokens": 500,
    "temperature": 0.3,  # Lower temperature for more consistent structured output
    # Configure video sampling rate (FPS)
    "extra_body": {
        "mm_processor_kwargs": {
            "fps": 1,  # Sample 2 frames per second
            "do_sample_frames": True
        }
    }
}


try:
    response = requests.post(url, headers=headers, json=data, timeout=180)
    response.raise_for_status()

    result = response.json()

    # Extract and print the response
    if "choices" in result and len(result["choices"]) > 0:
        content = result["choices"][0]["message"]["content"]
        print("Unexpected response format:")
        print(json.dumps(result, indent=2))

except requests.exceptions.RequestException as e:
    print(f"Error: {e}")
    if hasattr(e, 'response') and e.response is not None:
        print(f"Response status: {e.response.status_code}")
        print(f"Response body: {e.response.text}")

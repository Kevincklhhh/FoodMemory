#!/usr/bin/env python3
import base64
import requests
import json
import cv2
import re
from pathlib import Path

# Read and encode the video
video_path = "/home/kailaic/NeuroTrace/kitchen/home_videos/184_1761161289.mp4"
video_path = "/home/kailaic/NeuroTrace/kitchen/home_videos/185_1761161310.mp4"
with open(video_path, "rb") as f:
    video_base64 = base64.b64encode(f.read()).decode()

# Prepare the request for video analysis
url = "http://localhost:8000/v1/chat/completions"
headers = {"Content-Type": "application/json"}

# Structured prompt for bounding box detection (following Qwen3-VL format)
# prompt = """This is an egocentric video from a first-person perspective.

# Please analyze this video and find ONE frame where a food item has a clear, unobstructed, and complete view during user interaction.

# Locate the food item with the clearest, most complete view and report the result in JSON format like this:
# {
#   "timestamp": <time in seconds>,
#   "bbox_2d": [x1, y1, x2, y2],
#   "label": "<food item name>",
#   "visibility_quality": "<description of why the view is clear>",
# }

# IMPORTANT:
# - Coordinates must be in the range 0-1000 (Qwen3-VL coordinate system)
# - bbox_2d format: [x1, y1, x2, y2] where x1,y1 is top-left and x2,y2 is bottom-right
# - Only provide ONE detection
# - Choose a frame where the food item is clearly being interacted with AND has a complete, unobstructed view
# - The food item should be fully visible without being blocked by hands, containers, or other objects

# """
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
    "model": "Qwen3-VL-30B",
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
            "fps": 8,  # Sample 2 frames per second
            "do_sample_frames": True
        }
    }
}

print("Sending video to vLLM server for analysis...")
print(f"Video: {video_path}")
print(f"Video sampling: {data['extra_body']['mm_processor_kwargs']['fps']} FPS")
print(f"Task: Detect food item with clear, unobstructed view and bounding box\n")

try:
    response = requests.post(url, headers=headers, json=data, timeout=180)
    response.raise_for_status()

    result = response.json()

    # Extract and print the response
    if "choices" in result and len(result["choices"]) > 0:
        content = result["choices"][0]["message"]["content"]
        print("=" * 80)
        print("MODEL RESPONSE - UNOBSTRUCTED FOOD DETECTION:")
        print("=" * 80)
        print(content)
        print("=" * 80)
        print(f"\nTokens used: {result.get('usage', {})}")

        # # Try to parse JSON from response
        # try:
        #     # Extract JSON from markdown code blocks if present
        #     json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
        #     if json_match:
        #         json_str = json_match.group(1)
        #     else:
        #         # Try to find JSON object directly
        #         json_match = re.search(r'\{.*\}', content, re.DOTALL)
        #         if json_match:
        #             json_str = json_match.group(0)
        #         else:
        #             json_str = content

        #     detection = json.loads(json_str)

        #     print("\n" + "=" * 80)
        #     print("PARSED DETECTION:")
        #     print("=" * 80)
        #     print(f"Timestamp: {detection['timestamp']} seconds")
        #     print(f"Food Item: {detection['label']}")
        #     print(f"Visibility Quality: {detection['visibility_quality']}")
        #     print(f"Bounding Box (0-1000): {detection['bbox_2d']}")
        #     print(f"Action: {detection['action']}")

        #     # Visualize the detection
        #     print("\n" + "=" * 80)
        #     print("VISUALIZING DETECTION...")
        #     print("=" * 80)

        #     # Open video
        #     cap = cv2.VideoCapture(video_path)
        #     if not cap.isOpened():
        #         print(f"Error: Could not open video {video_path}")
        #     else:
        #         # Get video properties
        #         fps = cap.get(cv2.CAP_PROP_FPS)
        #         width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        #         height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        #         total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        #         duration = total_frames / fps

        #         print(f"Video properties: {width}x{height} @ {fps:.2f} FPS")
        #         print(f"Video duration: {duration:.2f}s ({total_frames} frames)")

        #         # Calculate frame number from timestamp
        #         timestamp = float(detection['timestamp'])

        #         # Validate and cap timestamp to video duration
        #         if timestamp > duration:
        #             print(f"Warning: Timestamp {timestamp:.2f}s exceeds video duration {duration:.2f}s")
        #             print(f"Capping to maximum duration...")
        #             timestamp = duration - 0.5  # Use a frame near the end

        #         frame_number = int(timestamp * fps)
        #         frame_number = min(frame_number, total_frames - 1)  # Ensure within bounds

        #         print(f"Extracting frame {frame_number} at {timestamp:.2f}s...")

        #         # Seek to the frame
        #         cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        #         ret, frame = cap.read()

        #         if ret:
        #             # Convert Qwen3-VL coordinates (0-1000) to pixel coordinates
        #             # bbox_2d format: [x1, y1, x2, y2]
        #             bbox = detection['bbox_2d']
        #             x1 = int(bbox[0] / 1000 * width)
        #             y1 = int(bbox[1] / 1000 * height)
        #             x2 = int(bbox[2] / 1000 * width)
        #             y2 = int(bbox[3] / 1000 * height)

        #             # Ensure coordinates are in correct order
        #             if x1 > x2:
        #                 x1, x2 = x2, x1
        #             if y1 > y2:
        #                 y1, y2 = y2, y1

        #             print(f"Bounding box (pixels): ({x1}, {y1}) to ({x2}, {y2})")

        #             # Draw bounding box
        #             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

        #             # Add label
        #             label = f"{detection['label']} (clear view)"
        #             label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

        #             # Draw label background
        #             cv2.rectangle(frame,
        #                         (x1, y1 - label_size[1] - 10),
        #                         (x1 + label_size[0], y1),
        #                         (0, 255, 0), -1)

        #             # Draw label text
        #             cv2.putText(frame, label,
        #                       (x1, y1 - 5),
        #                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        #             # Add timestamp info
        #             timestamp_label = f"Time: {timestamp:.2f}s | Frame: {frame_number}"
        #             cv2.putText(frame, timestamp_label,
        #                       (10, 30),
        #                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        #             # Save the annotated frame
        #             output_dir = Path(video_path).parent
        #             output_path = output_dir / f"unobstructed_food_detection_frame_{frame_number}.jpg"
        #             cv2.imwrite(str(output_path), frame)

        #             print(f"\n✓ Annotated frame saved to: {output_path}")
        #             print(f"  - Food item: {detection['label']}")
        #             print(f"  - Visibility quality: {detection['visibility_quality']}")
        #             print(f"  - Location: ({x1}, {y1}) to ({x2}, {y2})")
        #         else:
        #             print(f"Error: Could not read frame {frame_number}")

        #         cap.release()

    #     except json.JSONDecodeError as e:
    #         print(f"\nWarning: Could not parse JSON from response: {e}")
    #         print("Response may not be in expected format. Please check the output above.")
    #     except KeyError as e:
    #         print(f"\nWarning: Missing expected key in response: {e}")
    #         print("Response structure may be different than expected.")
    #     except Exception as e:
    #         print(f"\nError during visualization: {e}")
    # else:
        print("Unexpected response format:")
        print(json.dumps(result, indent=2))

except requests.exceptions.RequestException as e:
    print(f"Error: {e}")
    if hasattr(e, 'response') and e.response is not None:
        print(f"Response status: {e.response.status_code}")
        print(f"Response body: {e.response.text}")

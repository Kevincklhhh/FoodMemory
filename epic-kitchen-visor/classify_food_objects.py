#!/usr/bin/env python3
"""
Classify objects from P01 videos as food or food-containing items using Qwen3-VL LLM.

This script:
1. Loads all unique objects from P01 annotations
2. Sends each object name to Qwen3-VL LLM for classification (text-only)
3. Saves food objects to JSON and text files

Uses the same Qwen3-VL API as test_qwen3vl_video.py
"""

import json
import csv
from pathlib import Path
from typing import List, Dict, Set
import requests
import time


def load_annotations():
    """Load mask_info.json and assoc_info.json"""
    mask_path = Path('hd-epic-annotations/scene-and-object-movements/mask_info.json')
    assoc_path = Path('hd-epic-annotations/scene-and-object-movements/assoc_info.json')

    with open(mask_path, 'r') as f:
        mask_info = json.load(f)

    with open(assoc_path, 'r') as f:
        assoc_info = json.load(f)

    return mask_info, assoc_info


def extract_unique_objects(assoc_info) -> List[Dict[str, str]]:
    """Extract all unique objects from P01 videos"""
    unique_objects = {}

    # Filter P01 videos
    p01_videos = sorted([vid for vid in assoc_info.keys() if vid.startswith('P01')])

    print(f"Found {len(p01_videos)} P01 videos")

    for video_id in p01_videos:
        video_objects = assoc_info[video_id]

        for object_id, object_data in video_objects.items():
            if object_id not in unique_objects:
                unique_objects[object_id] = {
                    'object_id': object_id,
                    'object_name': object_data['name'],
                    'first_seen_video': video_id
                }

    return list(unique_objects.values())


def query_llm_qwen(object_name: str, model: str = "Qwen/Qwen3-VL-30B-A3B-Instruct") -> Dict[str, any]:
    """
    Query Qwen3-VL LLM to determine if object is food or contains food.

    Args:
        object_name: Name of the object to classify
        model: Qwen model to use

    Returns:
        Dict with 'is_food' (bool) and 'reasoning' (str)
    """
    prompt = f"""You are classifying kitchen objects. Determine if the following object is food or contains food.

Object: "{object_name}"

Consider:
- Is it edible food? (e.g., "apple", "bread", "cheese")
- Does it contain food? (e.g., "bottle of olive oil", "jar of jam", "milk carton")
- Food ingredients count as food (e.g., "salt", "sugar", "flour")
- Empty containers are NOT food (e.g., "empty bottle", "bowl")

Respond in this exact format:
DECISION: [YES or NO]
REASONING: [Brief explanation in one sentence]

Examples:
Object: "orange"
DECISION: YES
REASONING: An orange is a fruit and is edible food.

Object: "bottle of olive oil"
DECISION: YES
REASONING: Contains olive oil which is a food ingredient.

Object: "knife"
DECISION: NO
REASONING: A knife is a utensil, not food.

Now classify:
Object: "{object_name}"
"""

    url = "http://saltyfish.eecs.umich.edu:8000/v1/chat/completions"
    headers = {"Content-Type": "application/json"}

    data = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ],
        "max_tokens": 500,
        "temperature": 0.3  # Lower temperature for more consistent structured output
    }

    try:
        response = requests.post(url, headers=headers, json=data, timeout=60)
        response.raise_for_status()

        result = response.json()

        # Extract the response from OpenAI-compatible format
        if "choices" in result and len(result["choices"]) > 0:
            text = result["choices"][0]["message"]["content"].strip()

            # Parse the response
            is_food = 'DECISION: YES' in text.upper()

            # Extract reasoning
            reasoning = ""
            if 'REASONING:' in text:
                reasoning = text.split('REASONING:')[1].strip().split('\n')[0]

            return {
                'is_food': is_food,
                'reasoning': reasoning,
                'raw_response': text
            }
        else:
            print(f"Unexpected response format: {result}")
            return {
                'is_food': False,
                'reasoning': "Unexpected API response format",
                'raw_response': str(result)
            }

    except requests.exceptions.RequestException as e:
        print(f"Error querying LLM: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response status: {e.response.status_code}")
            print(f"Response body: {e.response.text}")
        return {
            'is_food': False,
            'reasoning': f"Error: {str(e)}",
            'raw_response': ""
        }
    except Exception as e:
        print(f"Unexpected error: {e}")
        return {
            'is_food': False,
            'reasoning': f"Error: {str(e)}",
            'raw_response': ""
        }


def classify_objects(objects: List[Dict[str, str]], model: str = "Qwen/Qwen3-VL-30B-A3B-Instruct") -> List[Dict]:
    """
    Classify all objects using LLM.

    Args:
        objects: List of object dictionaries with object_id and object_name
        model: Qwen model to use

    Returns:
        List of food objects with classification results
    """
    food_objects = []

    print(f"\nClassifying {len(objects)} unique objects...")
    print("=" * 80)

    for idx, obj in enumerate(objects, 1):
        object_name = obj['object_name']
        object_id = obj['object_id']

        print(f"\n[{idx}/{len(objects)}] Classifying: {object_name}")

        # Query LLM
        result = query_llm_qwen(object_name, model)

        if result['is_food']:
            food_obj = {
                'object_id': object_id,
                'object_name': object_name,
                'first_seen_video': obj['first_seen_video'],
                'reasoning': result['reasoning'],
                'raw_response': result['raw_response']
            }
            food_objects.append(food_obj)
            print(f"  ✓ FOOD: {result['reasoning']}")
        else:
            print(f"  ✗ NOT FOOD: {result['reasoning']}")

        # Small delay to avoid overwhelming the API
        time.sleep(0.5)  # Conservative delay for remote API

    return food_objects


def save_food_objects(food_objects: List[Dict],
                      detailed_file: str = 'food_objects_detailed.json',
                      names_file: str = 'food_objects_names.txt'):
    """
    Save food objects to files.

    Args:
        food_objects: List of food object dictionaries
        detailed_file: JSON file with full details
        names_file: Text file with just names
    """
    # Save detailed JSON
    with open(detailed_file, 'w') as f:
        json.dump(food_objects, f, indent=2)
    print(f"\n✓ Saved detailed results to {detailed_file}")

    # Save just names to text file
    with open(names_file, 'w') as f:
        for obj in food_objects:
            f.write(obj['object_name'] + '\n')
    print(f"✓ Saved food names to {names_file}")

    # Also save as CSV for easier viewing
    csv_file = detailed_file.replace('.json', '.csv')
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['object_id', 'object_name', 'first_seen_video', 'reasoning'])
        writer.writeheader()
        for obj in food_objects:
            writer.writerow({
                'object_id': obj['object_id'],
                'object_name': obj['object_name'],
                'first_seen_video': obj['first_seen_video'],
                'reasoning': obj['reasoning']
            })
    print(f"✓ Saved detailed results to {csv_file}")


def print_summary(food_objects: List[Dict], total_objects: int):
    """Print classification summary"""
    print("\n" + "=" * 80)
    print("CLASSIFICATION SUMMARY")
    print("=" * 80)
    print(f"Total unique objects: {total_objects}")
    print(f"Food objects found: {len(food_objects)}")
    print(f"Non-food objects: {total_objects - len(food_objects)}")
    print(f"Food percentage: {len(food_objects)/total_objects*100:.1f}%")
    print("=" * 80)

    print("\nFOOD OBJECTS FOUND:")
    print("-" * 80)
    for obj in food_objects:
        print(f"  • {obj['object_name']}")
        print(f"    └─ {obj['reasoning']}")


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Classify P01 objects as food or food-containing using LLM"
    )
    parser.add_argument(
        '--model',
        default='Qwen/Qwen3-VL-30B-A3B-Instruct',
        help='Qwen model to use (default: Qwen/Qwen3-VL-30B-A3B-Instruct)'
    )
    parser.add_argument(
        '--output-json',
        default='food_objects_detailed.json',
        help='Output JSON file with detailed results'
    )
    parser.add_argument(
        '--output-txt',
        default='food_objects_names.txt',
        help='Output text file with just object names'
    )

    args = parser.parse_args()

    print("Loading annotations...")
    mask_info, assoc_info = load_annotations()

    print("Extracting unique objects...")
    unique_objects = extract_unique_objects(assoc_info)
    print(f"Found {len(unique_objects)} unique objects")

    # Classify objects
    food_objects = classify_objects(unique_objects, model=args.model)

    # Save results
    save_food_objects(food_objects, args.output_json, args.output_txt)

    # Print summary
    print_summary(food_objects, len(unique_objects))

    print("\n✓ Done!")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Step 1: Classify HD-EPIC noun classes as food or food-containing items using LLM.

This script:
1. Loads HD-EPIC noun classes from HD_EPIC_noun_classes.csv
2. Sends each noun key to Qwen3-VL LLM for classification (text-only)
3. Saves food nouns to JSON and text files

Similar to epic-kitchen-visor/classify_food_objects.py but for HD-EPIC noun classes.
"""

import json
import csv
from pathlib import Path
from typing import List, Dict
import requests
import time


def load_hdepic_noun_classes(csv_path: str) -> List[Dict[str, any]]:
    """Load HD-EPIC noun classes from CSV.

    Returns:
        List of dicts with id, key, instances, category
    """
    noun_classes = []

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            noun_classes.append({
                'class_id': int(row['id']),
                'key': row['key'],
                'instances': row['instances'],
                'category': row['category']
            })

    return noun_classes


def query_llm_qwen(noun_key: str, model: str = "Qwen/Qwen3-VL-30B-A3B-Instruct") -> Dict[str, any]:
    """
    Query Qwen3-VL LLM to determine if noun is food or contains food.

    Args:
        noun_key: Name of the noun to classify
        model: Qwen model to use

    Returns:
        Dict with 'is_food' (bool) and 'reasoning' (str)
    """
    prompt = f"""You are classifying kitchen objects. Determine if the following object is food or contains food.

Object: "{noun_key}"

Consider:
- Is it edible food? (e.g., "apple", "bread", "cheese", "milk", "orange")
- Does it contain food? (e.g., "bottle of olive oil", "jar of jam", "milk carton")
- Food ingredients count as food (e.g., "salt", "sugar", "flour", "spices")
- Beverages count as food (e.g., "coffee", "tea", "juice", "wine", "beer")
- WATER is NOT considered food (plain water, tap water, cold water, hot water, etc.)
- Empty containers are NOT food (e.g., "empty bottle", "bowl", "plate")
- Utensils and tools are NOT food (e.g., "knife", "spoon", "pan")

Respond in this exact format:
DECISION: [YES or NO]
REASONING: [Brief explanation in one sentence]

Examples:
Object: "orange"
DECISION: YES
REASONING: An orange is a fruit and is edible food.

Object: "milk"
DECISION: YES
REASONING: Milk is a beverage and food ingredient.

Object: "bottle of olive oil"
DECISION: YES
REASONING: Contains olive oil which is a food ingredient.

Object: "knife"
DECISION: NO
REASONING: A knife is a utensil, not food.

Object: "pan"
DECISION: NO
REASONING: A pan is cookware, not food.

Object: "water"
DECISION: NO
REASONING: Water is excluded from food analysis per instructions.

Now classify:
Object: "{noun_key}"
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


def classify_nouns(noun_classes: List[Dict], model: str = "Qwen/Qwen3-VL-30B-A3B-Instruct") -> List[Dict]:
    """
    Classify all noun classes using LLM.

    Args:
        noun_classes: List of noun class dictionaries
        model: Qwen model to use

    Returns:
        List of food nouns with classification results
    """
    food_nouns = []

    print(f"\nClassifying {len(noun_classes)} noun classes...")
    print("=" * 80)

    for idx, noun in enumerate(noun_classes, 1):
        noun_key = noun['key']
        class_id = noun['class_id']

        print(f"\n[{idx}/{len(noun_classes)}] Classifying: {noun_key} (class_id: {class_id})")

        # Hardcoded exclusion: water is never considered food
        if noun_key.lower() == 'water':
            print(f"  → EXCLUDED (water is not food per configuration)")
            continue

        # Query LLM
        result = query_llm_qwen(noun_key, model)

        if result['is_food']:
            food_noun = {
                'class_id': class_id,
                'noun_key': noun_key,
                'instances': noun['instances'],
                'category': noun['category'],
                'reasoning': result['reasoning'],
                'raw_response': result['raw_response']
            }
            food_nouns.append(food_noun)
            print(f"  ✓ FOOD: {result['reasoning']}")
        else:
            print(f"  ✗ NOT FOOD: {result['reasoning']}")

        # Small delay to avoid overwhelming the API
        time.sleep(0.5)

    return food_nouns


def save_food_nouns(food_nouns: List[Dict],
                    detailed_file: str = 'hdepic_food_nouns_detailed.json',
                    names_file: str = 'hdepic_food_nouns_names.txt'):
    """
    Save food nouns to files.

    Args:
        food_nouns: List of food noun dictionaries
        detailed_file: JSON file with full details
        names_file: Text file with just names
    """
    # Save detailed JSON
    with open(detailed_file, 'w') as f:
        json.dump(food_nouns, f, indent=2)
    print(f"\n✓ Saved detailed results to {detailed_file}")

    # Save just names to text file
    with open(names_file, 'w') as f:
        for noun in food_nouns:
            f.write(noun['noun_key'] + '\n')
    print(f"✓ Saved food names to {names_file}")

    # Also save as CSV for easier viewing
    csv_file = detailed_file.replace('.json', '.csv')
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['class_id', 'noun_key', 'category', 'reasoning'])
        writer.writeheader()
        for noun in food_nouns:
            writer.writerow({
                'class_id': noun['class_id'],
                'noun_key': noun['noun_key'],
                'category': noun['category'],
                'reasoning': noun['reasoning']
            })
    print(f"✓ Saved detailed results to {csv_file}")


def print_summary(food_nouns: List[Dict], total_nouns: int):
    """Print classification summary"""
    print("\n" + "=" * 80)
    print("CLASSIFICATION SUMMARY")
    print("=" * 80)
    print(f"Total noun classes: {total_nouns}")
    print(f"Food nouns found: {len(food_nouns)}")
    print(f"Non-food nouns: {total_nouns - len(food_nouns)}")
    print(f"Food percentage: {len(food_nouns)/total_nouns*100:.1f}%")
    print("=" * 80)

    print("\nFOOD NOUNS FOUND:")
    print("-" * 80)
    for noun in food_nouns:
        print(f"  • {noun['noun_key']} [class_id: {noun['class_id']}]")
        print(f"    └─ {noun['reasoning']}")


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Classify HD-EPIC noun classes as food using LLM (Step 1)"
    )
    parser.add_argument(
        '--noun-classes',
        default='hd-epic-annotations/narrations-and-action-segments/HD_EPIC_noun_classes.csv',
        help='Path to HD_EPIC_noun_classes.csv'
    )
    parser.add_argument(
        '--model',
        default='Qwen/Qwen3-VL-30B-A3B-Instruct',
        help='Qwen model to use'
    )
    parser.add_argument(
        '--output-json',
        default='hdepic_food_nouns_detailed.json',
        help='Output JSON file with detailed results'
    )
    parser.add_argument(
        '--output-txt',
        default='hdepic_food_nouns_names.txt',
        help='Output text file with just noun names'
    )

    args = parser.parse_args()

    print("=" * 80)
    print("STEP 1: Classify HD-EPIC Food Nouns using LLM")
    print("=" * 80)

    print("\nLoading HD-EPIC noun classes...")
    noun_classes = load_hdepic_noun_classes(args.noun_classes)
    print(f"✓ Loaded {len(noun_classes)} noun classes")

    # Classify nouns
    food_nouns = classify_nouns(noun_classes, model=args.model)

    # Save results
    save_food_nouns(food_nouns, args.output_json, args.output_txt)

    # Print summary
    print_summary(food_nouns, len(noun_classes))

    print("\n✓ Done! Next step: Run 2_extract_hdepic_food_items.py")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Classify EPIC-KITCHENS-100 noun classes as food or food-containing items using Qwen3-VL LLM.

Adapted from classify_food_objects.py to work with EPIC_100_noun_classes_v2.csv

This script:
1. Loads noun classes from EPIC_100_noun_classes_v2.csv
2. Extracts all object instances/variations
3. Classifies each using Qwen3-VL LLM
4. Saves food items to JSON and text files
"""

import json
import csv
from pathlib import Path
from typing import List, Dict, Set
import requests
import time


def load_epic_noun_classes(csv_path: str = '/home/kailaic/NeuroTrace/kitchen/epic-kitchen-visor/EPIC_100_noun_classes_v2.csv') -> List[Dict]:
    """
    Load EPIC-100 noun classes from CSV.

    Returns:
        List of dictionaries with class_id, key, instances, and category
    """
    noun_classes = []

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            class_id = int(row['id'])
            key = row['key']
            instances = eval(row['instances'])  # Convert string list to actual list
            category = row['category']

            noun_classes.append({
                'class_id': class_id,
                'key': key,
                'instances': instances,
                'category': category
            })

    return noun_classes


def extract_unique_nouns(noun_classes: List[Dict]) -> List[Dict]:
    """
    Extract all unique noun instances from EPIC-100 classes.

    For each noun class, we'll classify the main 'key' rather than all variations
    to avoid redundancy.

    Returns:
        List of dictionaries with noun info
    """
    unique_nouns = []

    for noun_class in noun_classes:
        # Use the main key as the representative noun
        unique_nouns.append({
            'class_id': noun_class['class_id'],
            'noun_name': noun_class['key'],
            'category': noun_class['category'],
            'instance_count': len(noun_class['instances']),
            'sample_instances': noun_class['instances'][:5]  # Keep first 5 for reference
        })

    return unique_nouns


def query_llm_qwen(noun_name: str, category: str = None, model: str = "Qwen/Qwen3-VL-30B-A3B-Instruct") -> Dict[str, any]:
    """
    Query Qwen3-VL LLM to determine if noun is food or contains food.

    Args:
        noun_name: Name of the noun to classify (e.g., "plate", "apple", "oil")
        category: EPIC-100 category for additional context
        model: Qwen model to use

    Returns:
        Dict with 'is_food' (bool) and 'reasoning' (str)
    """
    category_context = f"\nEPIC-100 Category: {category}" if category else ""

    prompt = f"""You are classifying kitchen objects. Determine if the following object is food or contains food.

Object: "{noun_name}"{category_context}

Consider:
- Is it edible food? (e.g., "apple", "bread", "cheese")
- Does it contain food? (e.g., "oil", "milk", "sauce")
- Food ingredients count as food (e.g., "salt", "sugar", "flour")
- Empty containers are NOT food (e.g., "plate", "bowl", "jar" without contents)
- Utensils and appliances are NOT food (e.g., "knife", "oven", "pan")

Respond in this exact format:
DECISION: [YES or NO]
REASONING: [Brief explanation in one sentence]

Examples:
Object: "orange"
DECISION: YES
REASONING: An orange is a fruit and is edible food.

Object: "oil"
DECISION: YES
REASONING: Oil is a cooking ingredient and food product.

Object: "plate"
DECISION: NO
REASONING: A plate is a dish for serving food, not food itself.

Now classify:
Object: "{noun_name}"{category_context}
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
        "temperature": 0.3
    }

    try:
        response = requests.post(url, headers=headers, json=data, timeout=60)
        response.raise_for_status()

        result = response.json()

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


def classify_nouns(nouns: List[Dict], model: str = "Qwen/Qwen3-VL-30B-A3B-Instruct") -> List[Dict]:
    """
    Classify all nouns using LLM.

    Args:
        nouns: List of noun dictionaries
        model: Qwen model to use

    Returns:
        List of food nouns with classification results
    """
    food_nouns = []

    print(f"\nClassifying {len(nouns)} noun classes...")
    print("=" * 80)

    for idx, noun in enumerate(nouns, 1):
        noun_name = noun['noun_name']
        class_id = noun['class_id']
        category = noun['category']

        print(f"\n[{idx}/{len(nouns)}] Classifying: {noun_name} (class {class_id}, category: {category})")

        # Query LLM
        result = query_llm_qwen(noun_name, category, model)

        if result['is_food']:
            food_noun = {
                'class_id': class_id,
                'noun_name': noun_name,
                'category': category,
                'instance_count': noun['instance_count'],
                'sample_instances': noun['sample_instances'],
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
                    output_dir: str = '/home/kailaic/NeuroTrace/kitchen/epic-kitchen-visor',
                    prefix: str = 'epic_food_nouns'):
    """
    Save food nouns to files.

    Args:
        food_nouns: List of food noun dictionaries
        output_dir: Directory to save files
        prefix: File name prefix
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save detailed JSON
    json_file = output_dir / f'{prefix}_detailed.json'
    with open(json_file, 'w') as f:
        json.dump(food_nouns, f, indent=2)
    print(f"\n✓ Saved detailed results to {json_file}")

    # Save just names to text file
    txt_file = output_dir / f'{prefix}_names.txt'
    with open(txt_file, 'w') as f:
        for noun in sorted(food_nouns, key=lambda x: x['noun_name']):
            f.write(noun['noun_name'] + '\n')
    print(f"✓ Saved food names to {txt_file}")

    # Save as CSV for easier viewing
    csv_file = output_dir / f'{prefix}_detailed.csv'
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['class_id', 'noun_name', 'category', 'instance_count', 'reasoning'])
        writer.writeheader()
        for noun in sorted(food_nouns, key=lambda x: x['class_id']):
            writer.writerow({
                'class_id': noun['class_id'],
                'noun_name': noun['noun_name'],
                'category': noun['category'],
                'instance_count': noun['instance_count'],
                'reasoning': noun['reasoning']
            })
    print(f"✓ Saved detailed results to {csv_file}")

    # Save category breakdown
    category_file = output_dir / f'{prefix}_by_category.txt'
    with open(category_file, 'w') as f:
        from collections import defaultdict
        by_category = defaultdict(list)
        for noun in food_nouns:
            by_category[noun['category']].append(noun['noun_name'])

        f.write("FOOD ITEMS BY CATEGORY\n")
        f.write("=" * 80 + "\n\n")
        for category in sorted(by_category.keys()):
            f.write(f"{category.upper()}:\n")
            for name in sorted(by_category[category]):
                f.write(f"  - {name}\n")
            f.write("\n")
    print(f"✓ Saved category breakdown to {category_file}")


def print_summary(food_nouns: List[Dict], total_nouns: int):
    """Print classification summary"""
    print("\n" + "=" * 80)
    print("CLASSIFICATION SUMMARY")
    print("=" * 80)
    print(f"Total noun classes: {total_nouns}")
    print(f"Food noun classes found: {len(food_nouns)}")
    print(f"Non-food noun classes: {total_nouns - len(food_nouns)}")
    print(f"Food percentage: {len(food_nouns)/total_nouns*100:.1f}%")
    print("=" * 80)

    # Group by category
    from collections import defaultdict
    by_category = defaultdict(list)
    for noun in food_nouns:
        by_category[noun['category']].append(noun['noun_name'])

    print("\nFOOD ITEMS BY CATEGORY:")
    print("-" * 80)
    for category in sorted(by_category.keys()):
        print(f"\n{category.upper()} ({len(by_category[category])} items):")
        for name in sorted(by_category[category]):
            f_noun = next(n for n in food_nouns if n['noun_name'] == name)
            print(f"  • {name}")
            print(f"    └─ {f_noun['reasoning']}")


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Classify EPIC-100 noun classes as food using LLM"
    )
    parser.add_argument(
        '--csv',
        default='/home/kailaic/NeuroTrace/kitchen/epic-kitchen-visor/EPIC_100_noun_classes_v2.csv',
        help='Path to EPIC_100_noun_classes_v2.csv'
    )
    parser.add_argument(
        '--model',
        default='Qwen/Qwen3-VL-30B-A3B-Instruct',
        help='Qwen model to use'
    )
    parser.add_argument(
        '--output-dir',
        default='/home/kailaic/NeuroTrace/kitchen/epic-kitchen-visor',
        help='Output directory for results'
    )

    args = parser.parse_args()

    print("Loading EPIC-100 noun classes...")
    noun_classes = load_epic_noun_classes(args.csv)
    print(f"Loaded {len(noun_classes)} noun classes")

    print("\nExtracting unique nouns...")
    unique_nouns = extract_unique_nouns(noun_classes)
    print(f"Prepared {len(unique_nouns)} nouns for classification")

    # Classify nouns
    food_nouns = classify_nouns(unique_nouns, model=args.model)

    # Save results
    save_food_nouns(food_nouns, output_dir=args.output_dir)

    # Print summary
    print_summary(food_nouns, len(unique_nouns))

    print("\n✓ Done!")


if __name__ == '__main__':
    main()

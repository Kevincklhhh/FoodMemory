#!/usr/bin/env python3
"""
Normalize food names from unique_food_items.json using LLM

This script processes the complex food names (e.g., "carrot1", "bag of carrots",
"another carrot") and reduces them to basic food names (e.g., "carrot") using
an LLM for semantic understanding.
"""

import json
import requests
import argparse
from pathlib import Path
from typing import List, Dict
import time


def call_llm(food_names: List[str], url: str = "http://localhost:8000/v1/chat/completions") -> Dict[str, str]:
    """
    Call LLM to normalize a batch of food names.

    Args:
        food_names: List of food names to normalize
        url: LLM API endpoint

    Returns:
        Dictionary mapping original names to normalized names
    """
    # Create prompt with examples and instructions
    prompt = f"""You are a food name normalization assistant. Given a list of food item names that may include:
- Quantity indicators (e.g., "carrot1", "carrot2", "another carrot")
- State descriptions (e.g., "pitted date", "peeled carrot", "cut slice")
- Container descriptions (e.g., "bag of flour", "bottle of milk", "jar of nutella")
- Portion descriptions (e.g., "slice of butter", "piece of chicken", "half onion")
- Packaging descriptions (e.g., "packet of biscuits", "box of chicken stock")

Extract and return ONLY the basic food item name. Remove all:
- Numbers and counters (1, 2, first, second, another, one, two, etc.)
- Quantities (handful, few, pile, etc.)
- Containers (bag, bottle, jar, can, box, packet, container, mesh, etc.)
- States (pitted, peeled, cut, opened, remaining, etc.)
- Portions (slice, piece, half, cube, end, etc.)
- Descriptors that are not essential to the food type (new, thick, wide, etc.)

Keep:
- Essential food modifiers that define the type (e.g., "olive oil" not just "oil", "chicken stock" not just "stock")
- Specific food varieties (e.g., "balsamic vinegar", "parmesan cheese")

Examples:
- "carrot1" → "carrot"
- "another carrot" → "carrot"
- "bag of carrots" → "carrot"
- "carrot piece" → "carrot"
- "pitted date3" → "date"
- "bottle of olive oil" → "olive oil"
- "jar of nutella" → "nutella"
- "can of condensed milk" → "condensed milk"
- "box of chicken stock cubes" → "chicken stock cube"
- "handful of dates" → "date"
- "butter slice" → "butter"
- "one slice of butter" → "butter"
- "orange1 half" → "orange"
- "chicken piece1" → "chicken"
- "packaging of meat" → "meat"
- "packet of biscuits" → "biscuit"

Now normalize these food names. Return ONLY a JSON object mapping each original name to its normalized form:

Food names to normalize:
{json.dumps(food_names, indent=2)}

Return format (valid JSON only, no other text):
{{
  "original_name_1": "normalized_name_1",
  "original_name_2": "normalized_name_2",
  ...
}}"""

    headers = {"Content-Type": "application/json"}
    data = {
        "model": "Qwen3-VL-30B",
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
        "max_tokens": 2000,
        "temperature": 0.1,  # Low temperature for consistent normalization
    }

    try:
        response = requests.post(url, headers=headers, json=data, timeout=60)
        response.raise_for_status()

        result = response.json()
        content = result['choices'][0]['message']['content']

        # Extract JSON from response (handle markdown code blocks)
        content = content.strip()
        if content.startswith('```json'):
            content = content[7:]
        elif content.startswith('```'):
            content = content[3:]
        if content.endswith('```'):
            content = content[:-3]
        content = content.strip()

        # Parse JSON response
        normalized_mapping = json.loads(content)
        return normalized_mapping

    except Exception as e:
        print(f"Error calling LLM: {e}")
        return {}


def normalize_food_names(
    input_file: str = "food_extracted_frames/unique_food_items.json",
    output_file: str = "food_extracted_frames/normalized_food_names.json",
    llm_url: str = "http://localhost:8000/v1/chat/completions",
    batch_size: int = 50,
    verbose: bool = True
):
    """
    Normalize all food names using LLM in batches.

    Args:
        input_file: Path to unique_food_items.json
        output_file: Path to save normalized results
        llm_url: LLM API endpoint
        batch_size: Number of items to process per LLM call
        verbose: Print progress
    """
    # Load unique food items
    if verbose:
        print(f"Loading food items from {input_file}...")

    with open(input_file, 'r') as f:
        food_items = json.load(f)

    if verbose:
        print(f"  Found {len(food_items)} unique food items")
        print(f"\nNormalizing food names in batches of {batch_size}...")

    # Prepare data structures
    all_normalized = {}
    food_names = [item['name'] for item in food_items]

    # Process in batches
    total_batches = (len(food_names) + batch_size - 1) // batch_size

    for i in range(0, len(food_names), batch_size):
        batch_num = i // batch_size + 1
        batch = food_names[i:i + batch_size]

        if verbose:
            print(f"\n[Batch {batch_num}/{total_batches}] Processing {len(batch)} items...")

        # Call LLM
        normalized_batch = call_llm(batch, llm_url)

        if normalized_batch:
            all_normalized.update(normalized_batch)

            if verbose:
                # Show some examples from this batch
                examples = list(normalized_batch.items())[:5]
                for orig, norm in examples:
                    print(f"  {orig} → {norm}")
                if len(normalized_batch) > 5:
                    print(f"  ... and {len(normalized_batch) - 5} more")
        else:
            if verbose:
                print(f"  ✗ Failed to process batch {batch_num}")

        # Rate limiting
        if i + batch_size < len(food_names):
            time.sleep(1)

    # Create output with counts
    normalized_with_counts = {}
    for item in food_items:
        original_name = item['name']
        normalized_name = all_normalized.get(original_name, original_name)

        if normalized_name not in normalized_with_counts:
            normalized_with_counts[normalized_name] = {
                'normalized_name': normalized_name,
                'total_instances': 0,
                'original_names': []
            }

        normalized_with_counts[normalized_name]['total_instances'] += item['count']
        normalized_with_counts[normalized_name]['original_names'].append({
            'name': original_name,
            'count': item['count']
        })

    # Sort by total instances
    normalized_list = sorted(
        normalized_with_counts.values(),
        key=lambda x: x['total_instances'],
        reverse=True
    )

    # Prepare output
    output_data = {
        'total_unique_normalized': len(normalized_list),
        'total_original': len(food_items),
        'normalized_foods': normalized_list,
        'mapping': all_normalized
    }

    # Save results
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    if verbose:
        print(f"\n{'='*80}")
        print("NORMALIZATION COMPLETE")
        print('='*80)
        print(f"Original unique items: {len(food_items)}")
        print(f"Normalized unique items: {len(normalized_list)}")
        print(f"Reduction: {len(food_items) - len(normalized_list)} items consolidated")
        print(f"\nOutput saved: {output_path.resolve()}")

        # Show top normalized foods
        print(f"\nTop 20 foods by instance count:")
        for i, item in enumerate(normalized_list[:20], 1):
            orig_count = len(item['original_names'])
            print(f"{i:2d}. {item['normalized_name']:30s} "
                  f"({item['total_instances']:3d} instances from {orig_count} variants)")

    return output_data


def main():
    parser = argparse.ArgumentParser(
        description='Normalize food names using LLM'
    )
    parser.add_argument('--input', '-i',
                       default='food_extracted_frames/unique_food_items.json',
                       help='Input file (default: food_extracted_frames/unique_food_items.json)')
    parser.add_argument('--output', '-o',
                       default='food_extracted_frames/normalized_food_names.json',
                       help='Output file (default: food_extracted_frames/normalized_food_names.json)')
    parser.add_argument('--url', '-u',
                       default='http://localhost:8000/v1/chat/completions',
                       help='LLM API endpoint (default: http://localhost:8000/v1/chat/completions)')
    parser.add_argument('--batch-size', '-b', type=int, default=50,
                       help='Batch size for LLM calls (default: 50)')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Suppress verbose output')

    args = parser.parse_args()

    normalize_food_names(
        input_file=args.input,
        output_file=args.output,
        llm_url=args.url,
        batch_size=args.batch_size,
        verbose=not args.quiet
    )


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Deduplicate food items from food_objects_names.txt using semantic analysis.
Reduces variations like "first orange", "second orange", "orange" to just "orange".
"""

import re
from collections import defaultdict
from typing import List, Set, Dict


def read_food_items(file_path: str) -> List[str]:
    """Read food items from the file, skipping empty lines."""
    items = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = line.strip()
            if item:  # Skip empty items
                items.append(item)
    return items


def extract_core_food_item(item: str) -> str:
    """
    Extract the core food item by removing:
    - Ordinal numbers (first, second, third, etc.)
    - Quantities (one, two, three, handful, few, etc.)
    - Positional descriptors (left, right, top, bottom, etc.)
    - States (half, piece, slice, end, etc.)
    - Container descriptors (bag of, pack of, bottle of, etc.)
    """
    # Convert to lowercase
    item_lower = item.lower()

    # Remove ordinal numbers and quantities
    ordinals = r'\b(first|second|third|fourth|fifth|one|two|three|few|some|another|other|new|remaining|rest of)\s+'
    item_lower = re.sub(ordinals, '', item_lower)

    # Remove positional descriptors
    positions = r'\b(left|right|top|bottom|wide|thick|broken)\s+(half|end|piece|side)\s+of\s+'
    item_lower = re.sub(positions, '', item_lower)

    # Remove state descriptors before "of"
    states = r'\b(half|piece|slice|end|pieces|halves|slices|handful|pile|clove|cube|head)\s+of\s+'
    item_lower = re.sub(states, '', item_lower)

    # Remove container prefixes
    containers = r'^(bag|pack|packet|package|box|bottle|jar|tin|can|container|plastic\s+\w+|mesh|tube|stack|tray)\s+(of|containing)\s+'
    item_lower = re.sub(containers, '', item_lower)

    # Remove leading container adjectives
    item_lower = re.sub(r'^(plastic|glass|blue|green)\s+(bag|container|box|package|tube)\s+of\s+', '', item_lower)

    # Remove standalone state words at the beginning
    item_lower = re.sub(r'^(pitted|peeled|opened|cut|minced)\s+', '', item_lower)

    # Remove trailing numbers (carrot1, carrot10, etc.)
    item_lower = re.sub(r'\d+$', '', item_lower)

    # Remove multiple spaces
    item_lower = re.sub(r'\s+', ' ', item_lower).strip()

    return item_lower


def normalize_food_name(core_item: str) -> str:
    """
    Normalize food names to canonical forms.
    Handles plurals and common variations.
    """
    # Plural to singular mappings
    plurals = {
        'oranges': 'orange',
        'onions': 'onion',
        'carrots': 'carrot',
        'grapes': 'grapes',  # Keep as plural
        'tomatoes': 'tomato',
        'bagels': 'bagel',
        'dates': 'date',
        'biscuits': 'biscuit',
        'herbs': 'herbs',  # Keep as plural
        'crisps': 'crisps',  # Keep as plural
        'cloves': 'clove',
        'cubes': 'cube',
        'pies': 'pie',
    }

    # Check for exact plural matches
    for plural, singular in plurals.items():
        if core_item == plural or core_item.endswith(' ' + plural):
            core_item = core_item.replace(plural, singular)

    # Normalize compound items
    normalizations = {
        'olive oil': 'olive oil',
        'vegetable oil': 'vegetable oil',
        'balsemic vinegar': 'balsamic vinegar',
        'soya sauce': 'soy sauce',
        'condensed milk': 'condensed milk',
        'chicken stock': 'chicken stock',
        'stock cube': 'stock cube',
        'black pepper': 'black pepper',
        'cocoa powder': 'cocoa',
    }

    for variant, normalized in normalizations.items():
        if variant in core_item:
            return normalized

    return core_item


def categorize_items(items: List[str]) -> Dict[str, Set[str]]:
    """Categorize items by their core food name."""
    categories = defaultdict(set)

    for item in items:
        core = extract_core_food_item(item)
        normalized = normalize_food_name(core)
        categories[normalized].add(item)

    return categories


def main():
    # Read input file
    input_file = 'food_objects_names.txt'
    print(f"Reading food items from {input_file}...")
    items = read_food_items(input_file)
    print(f"Found {len(items)} total items")

    # Categorize and deduplicate
    print("\nDeduplicating items...")
    categories = categorize_items(items)

    # Sort unique items
    unique_items = sorted(categories.keys())

    # Write output
    output_file = 'food_objects_unique.txt'
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, item in enumerate(unique_items, 1):
            f.write(f"{i:3d}. {item}\n")

    print(f"\nReduced {len(items)} items to {len(unique_items)} unique food items")
    print(f"Output written to {output_file}")

    # Print detailed mapping if requested
    print("\n" + "="*80)
    print("DEDUPLICATION MAPPING:")
    print("="*80)

    for unique_item in unique_items:
        variations = categories[unique_item]
        if len(variations) > 1:
            print(f"\n'{unique_item}' ({len(variations)} variations):")
            for var in sorted(variations)[:5]:  # Show first 5 variations
                print(f"  - {var}")
            if len(variations) > 5:
                print(f"  ... and {len(variations) - 5} more")

    print("\n" + "="*80)
    print("\nUnique food items:")
    print("="*80)
    for i, item in enumerate(unique_items, 1):
        print(f"{i:3d}. {item}")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Entity Extraction from Narration CSV
Leverages pre-parsed nouns to identify food items and locations.
"""

import ast
from typing import Dict, List, Optional, Tuple


# Food-related keywords for identifying food entities
FOOD_KEYWORDS = {
    'milk', 'bottle', 'cheese', 'bread', 'butter', 'egg', 'eggs',
    'meat', 'chicken', 'fish', 'vegetable', 'fruit', 'apple', 'orange',
    'banana', 'tomato', 'potato', 'onion', 'garlic', 'carrot',
    'lettuce', 'cucumber', 'pepper', 'rice', 'pasta', 'noodles',
    'yogurt', 'cream', 'juice', 'water', 'soda', 'beer', 'wine',
    'coffee', 'tea', 'sugar', 'salt', 'flour', 'oil',
    'sauce', 'ketchup', 'mayonnaise', 'mustard',
    'cereal', 'snack', 'cookie', 'chocolate', 'cake',
    'soup', 'salad', 'sandwich', 'pizza', 'burger',
    'food', 'meal', 'dish', 'ingredient', 'capsule'
}

# Container keywords (these are food nodes when they hold food)
CONTAINER_KEYWORDS = {
    'bottle', 'jar', 'container', 'box', 'bag', 'package',
    'carton', 'can', 'tin', 'packet', 'mug', 'cup', 'glass',
    'bowl', 'plate', 'dish', 'pot', 'pan', 'tray', 'capsule'
}

# Location/zone keywords
LOCATION_KEYWORDS = {
    'fridge', 'refrigerator', 'freezer', 'cupboard', 'cabinet',
    'drawer', 'shelf', 'counter', 'countertop', 'table',
    'pantry', 'storage', 'sink', 'stove', 'oven', 'microwave',
    'dishwasher', 'rack', 'hook'
}

# Tool/appliance keywords (NOT food nodes)
TOOL_KEYWORDS = {
    'knife', 'spoon', 'fork', 'scissors', 'opener', 'peeler',
    'grater', 'whisk', 'spatula', 'tong', 'ladle',
    'machine', 'frother', 'blender', 'mixer', 'processor',
    'button', 'handle', 'lid', 'cover', 'wrap', 'foil',
    'towel', 'cloth', 'sponge', 'brush', 'tap', 'faucet'
}


def parse_nouns(nouns_str: str) -> List[str]:
    """
    Parse the nouns column from CSV (stored as string representation of list).

    Args:
        nouns_str: String like "['milk bottle', 'handle']"

    Returns:
        List of noun phrases
    """
    try:
        return ast.literal_eval(nouns_str)
    except:
        return []


def is_food_noun(noun: str) -> bool:
    """
    Determine if a noun phrase likely refers to food.

    Args:
        noun: Noun phrase (e.g., "milk bottle", "block of cheese")

    Returns:
        True if likely a food item
    """
    noun_lower = noun.lower()

    # Exclude if it's clearly a tool
    if any(tool in noun_lower for tool in TOOL_KEYWORDS):
        return False

    # Exclude if it's clearly a location (even though it might be in CONTAINER_KEYWORDS)
    if any(loc in noun_lower for loc in LOCATION_KEYWORDS):
        return False

    # Include if it contains a food keyword
    if any(food in noun_lower for food in FOOD_KEYWORDS):
        return True

    # Include if it's a container (assume it holds food)
    if any(container in noun_lower for container in CONTAINER_KEYWORDS):
        return True

    return False


def is_location_noun(noun: str) -> bool:
    """
    Determine if a noun phrase refers to a location/zone.

    Args:
        noun: Noun phrase

    Returns:
        True if likely a location
    """
    noun_lower = noun.lower()
    return any(loc in noun_lower for loc in LOCATION_KEYWORDS)


def normalize_location(location_phrase: str) -> str:
    """
    Normalize location phrase to simple zone name.

    Examples:
        "lower shelf of door of fridge" -> "fridge"
        "top drawer" -> "drawer"
        "kitchen counter" -> "counter"

    Args:
        location_phrase: Raw location noun phrase

    Returns:
        Normalized single-level location name
    """
    location_lower = location_phrase.lower()

    # Priority order: check specific locations first
    for location_keyword in LOCATION_KEYWORDS:
        if location_keyword in location_lower:
            # Special handling for multi-word locations
            if location_keyword == 'refrigerator':
                return 'fridge'
            elif location_keyword == 'countertop':
                return 'counter'
            elif location_keyword == 'cabinet':
                return 'cupboard'
            else:
                return location_keyword

    return location_phrase  # Return as-is if no match


def extract_entities(narration_row: Dict) -> Dict[str, Optional[str]]:
    """
    Extract food and location entities from a CSV narration row.

    Args:
        narration_row: Dictionary with keys: narration, nouns, verbs, etc.

    Returns:
        Dictionary with:
            - food_entity: Name of food item (or None)
            - location_entity: Normalized location name (or None)
            - all_food_entities: List of all food items found
            - all_locations: List of all locations found
    """
    nouns_str = narration_row.get('nouns', '[]')
    nouns = parse_nouns(nouns_str)

    food_entities = []
    location_entities = []

    for noun in nouns:
        if is_food_noun(noun):
            food_entities.append(noun)
        if is_location_noun(noun):
            normalized = normalize_location(noun)
            if normalized not in location_entities:
                location_entities.append(normalized)

    return {
        'food_entity': food_entities[0] if food_entities else None,
        'location_entity': location_entities[0] if location_entities else None,
        'all_food_entities': food_entities,
        'all_locations': location_entities
    }


def get_primary_action(narration_row: Dict) -> Optional[str]:
    """
    Extract the primary action verb from the narration row.

    Args:
        narration_row: Dictionary with 'verbs' or 'main_actions' field

    Returns:
        Primary action verb or None
    """
    # Try main_actions first (already prioritized)
    main_actions_str = narration_row.get('main_actions', '[]')
    try:
        main_actions = ast.literal_eval(main_actions_str)
        if main_actions and isinstance(main_actions[0], tuple):
            # Format is [('pick up', 'milk bottle')]
            return main_actions[0][0]
    except:
        pass

    # Fallback to first verb
    verbs_str = narration_row.get('verbs', '[]')
    try:
        verbs = ast.literal_eval(verbs_str)
        if verbs:
            return verbs[0]
    except:
        pass

    return None


def extract_narration_info(narration_row: Dict) -> Dict:
    """
    Extract all relevant information from a narration CSV row.

    Args:
        narration_row: CSV row as dictionary

    Returns:
        Dictionary with extracted information ready for KG update
    """
    entities = extract_entities(narration_row)
    primary_action = get_primary_action(narration_row)

    return {
        'narration_id': narration_row.get('unique_narration_id', ''),
        'video_id': narration_row.get('video_id', ''),
        'start_time': float(narration_row.get('start_timestamp', 0)),
        'end_time': float(narration_row.get('end_timestamp', 0)),
        'narration': narration_row.get('narration', ''),
        'food_entity': entities['food_entity'],
        'location_entity': entities['location_entity'],
        'all_food_entities': entities['all_food_entities'],
        'all_locations': entities['all_locations'],
        'primary_action': primary_action
    }

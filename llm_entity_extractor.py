#!/usr/bin/env python3
"""
LLM-based Entity Extraction
Uses LLM to extract food items and locations from narrations with better accuracy.
"""

import json
from typing import Dict, Optional, List
from ollama import Client


ENTITY_EXTRACTION_PROMPT = """You are a food tracking assistant. Given a narration of a kitchen action, extract:
1. The primary FOOD item being interacted with (if any)
2. The LOCATION where the action takes place (if any)

Rules:
- FOOD: Only consumable items or containers holding consumable items (e.g., "milk bottle", "cheese", "mug with coffee") Return only singular form (return 'orange' if narration mentions 'oranges')
- NOT FOOD: Tools, appliances, buttons, handles, lids by themselves (e.g., "knife", "coffee machine", "button", "lid")
- LOCATION: Storage areas or surfaces (e.g., "fridge", "counter", "shelf")
- Return null if no food/location is involved

Examples:

Input: "Pick up milk bottle from fridge"
Output: {{"food": "milk bottle", "location": "fridge", "reasoning": "Milk bottle is a container with consumable content, fridge is storage location"}}

Input: "Slide the milk frother on the counter"
Output: {{"food": null, "location": "counter", "reasoning": "Milk frother is a tool, not food"}}

Input: "Press the power button on the coffee machine"
Output: {{"food": null, "location": null, "reasoning": "Button and machine are tools, no food involved"}}

Input: "Open the milk bottle lid"
Output: {{"food": "milk bottle", "location": null, "reasoning": "Action involves milk bottle (food), lid is part of the container"}}

Input: "Place mug under coffee machine nozzle"
Output: {{"food": "mug", "location": null, "reasoning": "Mug will hold coffee (consumable), so it's tracked as food container"}}

Input: "Tilt the milk bottle to pour milk into mug"
Output: {{"food": "milk bottle", "location": null, "reasoning": "Primary action is on milk bottle; mug is secondary"}}

Now extract entities from this narration:
"{narration}"

Respond with JSON only."""


def extract_entities_with_llm(
    client: Client,
    model: str,
    narration: str,
    nouns: List[str] = None,
    max_retries: int = 2
) -> Dict[str, Optional[str]]:
    """
    Extract food and location entities using LLM.

    Args:
        client: Ollama client
        model: Model name
        narration: The narration text
        nouns: Optional list of pre-parsed nouns (for reference)
        max_retries: Number of retry attempts

    Returns:
        Dictionary with:
            - food_entity: Name of food item (or None)
            - location_entity: Name of location (or None)
            - reasoning: LLM's reasoning (optional)
    """
    prompt = ENTITY_EXTRACTION_PROMPT.format(narration=narration)

    for attempt in range(max_retries):
        try:
            resp = client.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                options={
                    "num_gpu": -1,
                    "num_thread": 8,
                    "temperature": 0.1,
                }
            )

            response_text = resp["message"]["content"]

            # Parse JSON response
            # Handle markdown code blocks
            if "```json" in response_text:
                start = response_text.find("```json") + 7
                end = response_text.find("```", start)
                json_text = response_text[start:end].strip()
            elif "```" in response_text:
                start = response_text.find("```") + 3
                end = response_text.find("```", start)
                json_text = response_text[start:end].strip()
            else:
                json_text = response_text.strip()

            result = json.loads(json_text)

            return {
                'food_entity': result.get('food'),
                'location_entity': result.get('location'),
                'reasoning': result.get('reasoning', ''),
                'all_food_entities': [result['food']] if result.get('food') else [],
                'all_locations': [result['location']] if result.get('location') else []
            }

        except json.JSONDecodeError as e:
            if attempt == max_retries - 1:
                print(f"  Warning: Failed to parse LLM entity extraction: {e}")
                print(f"    Response: {response_text[:200]}")
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"  Warning: Error in LLM entity extraction: {e}")

    # Fallback: return empty result
    return {
        'food_entity': None,
        'location_entity': None,
        'reasoning': 'LLM extraction failed',
        'all_food_entities': [],
        'all_locations': []
    }


def extract_narration_info_with_llm(
    client: Client,
    model: str,
    narration_row: Dict
) -> Dict:
    """
    Extract all relevant information from a narration CSV row using LLM.

    Args:
        client: Ollama client
        model: Model name
        narration_row: CSV row as dictionary

    Returns:
        Dictionary with extracted information ready for KG update
    """
    # Get basic info
    narration = narration_row.get('narration', '')

    # Parse nouns for reference (optional)
    import ast
    nouns_str = narration_row.get('nouns', '[]')
    try:
        nouns = ast.literal_eval(nouns_str)
    except:
        nouns = []

    # Extract entities using LLM
    entities = extract_entities_with_llm(client, model, narration, nouns)

    # Get primary action from CSV
    main_actions_str = narration_row.get('main_actions', '[]')
    primary_action = None
    try:
        main_actions = ast.literal_eval(main_actions_str)
        if main_actions and isinstance(main_actions[0], tuple):
            primary_action = main_actions[0][0]
    except:
        # Fallback to first verb
        verbs_str = narration_row.get('verbs', '[]')
        try:
            verbs = ast.literal_eval(verbs_str)
            if verbs:
                primary_action = verbs[0]
        except:
            pass

    return {
        'narration_id': narration_row.get('unique_narration_id', ''),
        'video_id': narration_row.get('video_id', ''),
        'start_time': float(narration_row.get('start_timestamp', 0)),
        'end_time': float(narration_row.get('end_timestamp', 0)),
        'narration': narration,
        'food_entity': entities['food_entity'],
        'location_entity': entities['location_entity'],
        'all_food_entities': entities['all_food_entities'],
        'all_locations': entities['all_locations'],
        'primary_action': primary_action,
        'llm_reasoning': entities.get('reasoning', '')
    }

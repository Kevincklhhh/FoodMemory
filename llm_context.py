#!/usr/bin/env python3
"""
LLM Context Builder
Prepares prompts for LLM with current KG state + new narration.
"""

import json
from typing import Dict, List, Optional, Any


def build_kg_update_prompt(
    narration_info: Dict,
    existing_food: Optional[Dict] = None,
    kg: Optional[Dict] = None
) -> List[Dict[str, str]]:
    """
    Build a prompt for the LLM to generate KG update JSON.

    Args:
        narration_info: Extracted narration information from entity_extractor
        existing_food: Current state of food node if it exists
        kg: Full knowledge graph (for zone lookups)

    Returns:
        List of message dicts for OpenAI chat completion API
    """
    # System prompt - conversational with few-shot examples
    system_prompt = """You help track food in a kitchen by responding with JSON."""

    # Build context about current KG state
    context_parts = []

    if existing_food:
        food_location = "in hand (null)"
        if existing_food.get("location") and kg:
            zone_id = existing_food["location"]
            if zone_id in kg.get("zones", {}):
                zone_name = kg["zones"][zone_id]["name"]
                food_location = f"{zone_name} ({zone_id})"

        context_parts.append(f"""CURRENT FOOD NODE:
- Food ID: {existing_food['food_id']}
- Name: {existing_food['name']}
- State: {existing_food.get('state', 'unknown')}
- Location: {food_location}
- Quantity: {existing_food.get('quantity', 'unknown')}
- Previous interactions: {len(existing_food.get('interaction_history', []))}""")

        # Add recent interaction context
        if existing_food.get('interaction_history'):
            recent = existing_food['interaction_history'][-3:]
            context_parts.append("\nRECENT INTERACTIONS:")
            for interaction in recent:
                context_parts.append(
                    f"  - {interaction['action']} at {interaction['start_time']:.1f}s"
                )
    else:
        context_parts.append("NO EXISTING FOOD NODE FOUND - This appears to be a new food item")

    # Build information about new narration
    narration_parts = [
        f"\nNEW ACTION TO PROCESS:",
        f"- Time: {narration_info['start_time']:.2f}s to {narration_info['end_time']:.2f}s",
        f"- Narration: \"{narration_info['narration']}\""
    ]

    # Add few-shot examples
    examples = """
Example 1 - Creating new food (first time seeing this item):
Input: "Pick up milk bottle from fridge at 21.4-22.0s"
Current: NO EXISTING FOOD NODE FOUND
Output: {"update_type": "CREATE_NEW", "new_food_info": {"name": "milk bottle"}, "updates": {"location": null}, "history_entry": {"start_time": 21.4, "end_time": 22.0, "action": "pick up", "narration_text": "pick up milk bottle", "location_context": "zone_fridge_1"}}

Example 2 - Updating existing food (append interaction):
Input: "Place milk bottle in fridge at 37.0-38.0s"
Current: food_milk_bottle_1 is in hand
Output: {"update_type": "UPDATE_EXISTING", "target_food_id": "food_milk_bottle_1", "updates": {"location": "zone_fridge_1"}, "history_entry": {"start_time": 37.0, "end_time": 38.0, "action": "place", "narration_text": "place milk bottle", "location_context": "zone_fridge_1"}}

Example 3 - Updating existing food (just append interaction, no property changes):
Input: "Turn the milk bottle to read the label at 45.0-46.0s"
Current: food_milk_bottle_1 is in hand
Output: {"update_type": "UPDATE_EXISTING", "target_food_id": "food_milk_bottle_1", "updates": {}, "history_entry": {"start_time": 45.0, "end_time": 46.0, "action": "turn", "narration_text": "turn the milk bottle", "location_context": null}}"""

    # Combine into user message
    user_message = examples + "\n\n" + "\n".join(context_parts + narration_parts)

    # Add simple instruction
    user_message += "\n\nRespond with JSON only."

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]

    return messages


def parse_llm_response(response_text: str) -> Optional[Dict]:
    """
    Parse LLM response to extract JSON update command.

    Args:
        response_text: Raw text response from LLM

    Returns:
        Parsed JSON dict or None if parsing fails
    """
    # Try to find JSON in the response
    try:
        # Sometimes LLM wraps JSON in markdown code blocks
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

        # Parse JSON
        update_command = json.loads(json_text)
        return update_command

    except json.JSONDecodeError as e:
        print(f"Error parsing LLM response as JSON: {e}")
        print(f"Response text: {response_text[:200]}")
        return None


def validate_update_command(command: Dict) -> tuple[bool, Optional[str]]:
    """
    Validate the structure of an update command.

    Args:
        command: Parsed JSON update command

    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check required fields
    if "update_type" not in command:
        return False, "Missing 'update_type' field"

    # Normalize to uppercase
    command["update_type"] = command["update_type"].upper()

    # Only two valid types
    if command["update_type"] not in ["CREATE_NEW", "UPDATE_EXISTING"]:
        return False, f"Invalid update_type: {command['update_type']}. Must be CREATE_NEW or UPDATE_EXISTING"

    if "history_entry" not in command:
        return False, "Missing 'history_entry' field"

    # Validate history_entry structure
    history = command["history_entry"]
    required_history_fields = ["start_time", "end_time", "action", "narration_text", "location_context"]
    for field in required_history_fields:
        if field not in history:
            return False, f"Missing '{field}' in history_entry"

    # Validate based on update type
    if command["update_type"] == "CREATE_NEW":
        if "new_food_info" not in command:
            return False, "CREATE_NEW requires 'new_food_info' field"
        if "name" not in command["new_food_info"]:
            return False, "new_food_info missing 'name' field"
        if not command["new_food_info"]["name"]:
            return False, "new_food_info 'name' cannot be None or empty"
    elif command["update_type"] == "UPDATE_EXISTING":
        if "target_food_id" not in command or not command["target_food_id"]:
            return False, "UPDATE_EXISTING requires 'target_food_id'"

    return True, None


def format_prompt_for_display(messages: List[Dict[str, str]]) -> str:
    """
    Format prompt messages for human-readable display.

    Args:
        messages: List of message dicts

    Returns:
        Formatted string
    """
    lines = []
    for msg in messages:
        role = msg["role"].upper()
        content = msg["content"]
        lines.append(f"\n{'=' * 60}")
        lines.append(f"{role}:")
        lines.append(f"{'=' * 60}")
        lines.append(content)

    return "\n".join(lines)

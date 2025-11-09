#!/usr/bin/env python3
"""
Knowledge Graph Storage and Management
Provides data structures and functions for managing food and zone nodes.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime


def create_empty_kg() -> Dict[str, Any]:
    """Create an empty knowledge graph structure."""
    return {
        "zones": {},
        "foods": {},
        "metadata": {
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "version": "1.0"
        }
    }


def load_kg(json_path: str) -> Dict[str, Any]:
    """
    Load knowledge graph from JSON file.

    Args:
        json_path: Path to JSON file

    Returns:
        Knowledge graph dictionary
    """
    path = Path(json_path)
    if path.exists():
        with open(path, 'r') as f:
            kg = json.load(f)
        print(f"Loaded KG from {json_path}")
        print(f"  - {len(kg.get('zones', {}))} zones")
        print(f"  - {len(kg.get('foods', {}))} food items")
        return kg
    else:
        print(f"No existing KG found at {json_path}, creating new one")
        return create_empty_kg()


def save_kg(kg: Dict[str, Any], json_path: str) -> None:
    """
    Save knowledge graph to JSON file.

    Args:
        kg: Knowledge graph dictionary
        json_path: Path to save JSON file
    """
    # Update metadata
    kg["metadata"]["last_updated"] = datetime.now().isoformat()

    path = Path(json_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'w') as f:
        json.dump(kg, f, indent=2)

    print(f"Saved KG to {json_path}")


def get_or_create_zone(kg: Dict[str, Any], zone_name: str, zone_type: str = "Storage") -> str:
    """
    Get existing zone ID or create new zone.

    Args:
        kg: Knowledge graph
        zone_name: Human-readable zone name (e.g., "fridge", "counter")
        zone_type: Type of zone (Storage, PreparationSurface, Appliance)

    Returns:
        zone_id
    """
    # Check if zone already exists
    for zone_id, zone_data in kg["zones"].items():
        if zone_data["name"].lower() == zone_name.lower():
            return zone_id

    # Create new zone
    zone_id = f"zone_{zone_name.lower().replace(' ', '_')}_{len(kg['zones']) + 1}"
    kg["zones"][zone_id] = {
        "zone_id": zone_id,
        "name": zone_name,
        "type": zone_type
    }

    print(f"Created new zone: {zone_id} ({zone_name})")
    return zone_id


def find_food(kg: Dict[str, Any],
              name_pattern: Optional[str] = None,
              location: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Find food nodes matching criteria.

    Args:
        kg: Knowledge graph
        name_pattern: Partial name to match (case-insensitive)
        location: Zone ID or zone name to match

    Returns:
        List of matching food node dictionaries
    """
    matches = []

    for food_id, food_data in kg["foods"].items():
        # Check name match
        if name_pattern:
            if name_pattern.lower() not in food_data["name"].lower():
                continue

        # Check location match
        if location:
            food_location = food_data.get("location")
            if food_location is None:
                continue

            # Match by zone_id or zone name
            if location.lower() in food_location.lower():
                pass  # Match by zone_id
            elif food_location in kg["zones"]:
                zone_name = kg["zones"][food_location]["name"]
                if location.lower() not in zone_name.lower():
                    continue
            else:
                continue

        matches.append(food_data)

    return matches


def add_food_node(kg: Dict[str, Any],
                  name: str,
                  state: str = "unknown",
                  quantity: str = "unknown",
                  location: Optional[str] = None,
                  first_seen_time: float = 0.0) -> str:
    """
    Add a new food node to the knowledge graph.

    Args:
        kg: Knowledge graph
        name: Food name
        state: Food state (raw, chopped, cooked, opened, etc.)
        quantity: Amount description
        location: Zone ID where food is located
        first_seen_time: Timestamp of first interaction

    Returns:
        food_id of the created node
    """
    # Generate unique food_id
    base_name = name.lower().replace(' ', '_')
    food_id = f"food_{base_name}_{len(kg['foods']) + 1}"

    kg["foods"][food_id] = {
        "food_id": food_id,
        "name": name,
        "state": state,
        "first_seen_time": first_seen_time,
        "quantity": quantity,
        "location": location,
        "interaction_history": []
    }

    print(f"Created new food node: {food_id} ({name})")
    return food_id


def update_food_node(kg: Dict[str, Any],
                     food_id: str,
                     updates: Dict[str, Any]) -> bool:
    """
    Update properties of an existing food node.

    Args:
        kg: Knowledge graph
        food_id: ID of food to update
        updates: Dictionary of properties to update

    Returns:
        True if successful, False if food_id not found
    """
    if food_id not in kg["foods"]:
        print(f"Error: Food ID {food_id} not found")
        return False

    for key, value in updates.items():
        if key != "food_id":  # Don't allow changing the ID
            kg["foods"][food_id][key] = value

    return True


def add_interaction(kg: Dict[str, Any],
                   food_id: str,
                   start_time: float,
                   end_time: float,
                   action: str,
                   narration_text: str,
                   location_context: str,
                   video_id: str = None) -> bool:
    """
    Add an interaction to a food node's history.

    Args:
        kg: Knowledge graph
        food_id: ID of food
        start_time: Start timestamp of interaction
        end_time: End timestamp of interaction
        action: Action verb (pick up, place, chop, etc.)
        narration_text: Full narration text
        location_context: Zone ID where action occurred
        video_id: Video identifier for multi-video KG tracking

    Returns:
        True if successful, False if food_id not found
    """
    if food_id not in kg["foods"]:
        print(f"Error: Food ID {food_id} not found")
        return False

    interaction_entry = {
        "start_time": start_time,
        "end_time": end_time,
        "action": action,
        "narration_text": narration_text,
        "location_context": location_context,
        "video_id": video_id
    }

    kg["foods"][food_id]["interaction_history"].append(interaction_entry)
    return True


def get_food_summary(kg: Dict[str, Any]) -> str:
    """
    Generate a human-readable summary of all food in the KG.

    Args:
        kg: Knowledge graph

    Returns:
        Formatted string summary
    """
    if not kg["foods"]:
        return "No food items in knowledge graph"

    lines = ["Current Food Inventory:"]
    lines.append("=" * 60)

    for food_id, food_data in kg["foods"].items():
        location_name = "in hand"
        if food_data.get("location"):
            zone_id = food_data["location"]
            if zone_id in kg["zones"]:
                location_name = kg["zones"][zone_id]["name"]

        lines.append(f"\n{food_data['name'].upper()}")
        lines.append(f"  ID: {food_id}")
        lines.append(f"  State: {food_data.get('state', 'unknown')}")
        lines.append(f"  Quantity: {food_data.get('quantity', 'unknown')}")
        lines.append(f"  Location: {location_name}")
        lines.append(f"  First seen: {food_data.get('first_seen_time', 0)}s")
        lines.append(f"  Interactions: {len(food_data.get('interaction_history', []))}")

        # Show recent interactions
        if food_data.get("interaction_history"):
            lines.append("  Recent actions:")
            for interaction in food_data["interaction_history"][-3:]:
                lines.append(f"    - {interaction['action']} ({interaction['start_time']:.1f}s)")

    return "\n".join(lines)

#!/usr/bin/env python3
"""
Food Memory Storage Module
Manages the persistent storage of food items and their interaction history.
"""

import json
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime


class FoodItem:
    """Represents a single food item in memory."""

    def __init__(self, food_id: str, primary_label: str, description: str,
                 visual_embedding: List[float], current_location: str,
                 current_quantity: str):
        self.food_id = food_id
        self.primary_label = primary_label
        self.description = description
        self.visual_embedding = visual_embedding
        self.current_location = current_location
        self.current_quantity = current_quantity
        self.interaction_history: List[Dict[str, Any]] = []

    def add_interaction(self, timestamp: str, action: str, **kwargs):
        """Add an interaction to the history."""
        interaction = {
            "timestamp": timestamp,
            "action": action,
            **kwargs
        }
        self.interaction_history.append(interaction)

    def update_state(self, location: Optional[str] = None,
                     quantity: Optional[str] = None):
        """Update current state of the food item."""
        if location is not None:
            self.current_location = location
        if quantity is not None:
            self.current_quantity = quantity

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "food_id": self.food_id,
            "primary_label": self.primary_label,
            "description": self.description,
            "visual_embedding": self.visual_embedding,
            "current_location": self.current_location,
            "current_quantity": self.current_quantity,
            "interaction_history": self.interaction_history
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'FoodItem':
        """Create FoodItem from dictionary."""
        item = cls(
            food_id=data["food_id"],
            primary_label=data["primary_label"],
            description=data["description"],
            visual_embedding=data["visual_embedding"],
            current_location=data["current_location"],
            current_quantity=data["current_quantity"]
        )
        item.interaction_history = data.get("interaction_history", [])
        return item


class FoodMemory:
    """Main memory storage for all food items."""

    def __init__(self, storage_path: str = "food_memory.json"):
        self.storage_path = Path(storage_path)
        self.items: Dict[str, FoodItem] = {}
        self.load()

    def load(self):
        """Load memory from disk if it exists."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                    self.items = {
                        food_id: FoodItem.from_dict(item_data)
                        for food_id, item_data in data.get("items", {}).items()
                    }
                print(f"Loaded {len(self.items)} items from {self.storage_path}")
            except Exception as e:
                print(f"Error loading memory: {e}")
                self.items = {}
        else:
            print(f"No existing memory found at {self.storage_path}, starting fresh")
            self.items = {}

    def save(self):
        """Save memory to disk."""
        try:
            data = {
                "items": {
                    food_id: item.to_dict()
                    for food_id, item in self.items.items()
                }
            }
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"Saved {len(self.items)} items to {self.storage_path}")
        except Exception as e:
            print(f"Error saving memory: {e}")

    def create_item(self, primary_label: str, description: str,
                    visual_embedding: List[float], location: str,
                    quantity: str) -> str:
        """Create a new food item and return its ID."""
        food_id = f"{primary_label.lower().replace(' ', '_')}_{uuid.uuid4().hex[:8]}"

        item = FoodItem(
            food_id=food_id,
            primary_label=primary_label,
            description=description,
            visual_embedding=visual_embedding,
            current_location=location,
            current_quantity=quantity
        )

        self.items[food_id] = item
        print(f"Created new item: {food_id} - {primary_label}")
        return food_id

    def get_item(self, food_id: str) -> Optional[FoodItem]:
        """Get a food item by ID."""
        return self.items.get(food_id)

    def update_item(self, food_id: str, location: Optional[str] = None,
                    quantity: Optional[str] = None,
                    interaction: Optional[Dict] = None):
        """Update an existing food item."""
        item = self.get_item(food_id)
        if not item:
            print(f"Warning: Item {food_id} not found")
            return False

        item.update_state(location=location, quantity=quantity)

        if interaction:
            item.add_interaction(**interaction)

        print(f"Updated item: {food_id}")
        return True

    def search_by_label(self, label: str, top_k: int = 5) -> List[FoodItem]:
        """
        Search for items by label similarity.
        Simple text matching for now (can be enhanced with fuzzy matching).
        """
        label_lower = label.lower()
        matches = []

        for item in self.items.values():
            # Simple substring matching
            if label_lower in item.primary_label.lower():
                matches.append((item, 1.0))  # Exact substring match
            elif any(word in item.primary_label.lower()
                    for word in label_lower.split()):
                matches.append((item, 0.5))  # Partial word match

        # Sort by score (higher is better)
        matches.sort(key=lambda x: x[1], reverse=True)

        return [item for item, score in matches[:top_k]]

    def get_items_by_location(self, location: str) -> List[FoodItem]:
        """Get all items at a specific location."""
        return [item for item in self.items.values()
                if item.current_location.lower() == location.lower()]

    def get_recent_interactions(self, hours: int = 24) -> List[FoodItem]:
        """Get items with recent interactions."""
        # For now, return all items (can be enhanced with timestamp filtering)
        return list(self.items.values())

    def to_dict(self) -> Dict:
        """Convert entire memory to dictionary."""
        return {
            "items": {
                food_id: item.to_dict()
                for food_id, item in self.items.items()
            }
        }


if __name__ == "__main__":
    # Test the FoodMemory module
    print("Testing FoodMemory module...")

    memory = FoodMemory("test_food_memory.json")

    # Create a test item
    food_id = memory.create_item(
        primary_label="Milk 2%",
        description="Carton of 2% Milk",
        visual_embedding=[0.1, 0.2, 0.3],
        location="fridge",
        quantity="full"
    )

    # Add interaction
    memory.update_item(
        food_id,
        interaction={
            "timestamp": "2025-10-29T20:00:00",
            "action": "retrieve from fridge"
        }
    )

    # Save
    memory.save()

    # Test search
    results = memory.search_by_label("milk")
    print(f"\nSearch results for 'milk': {len(results)} items found")
    for item in results:
        print(f"  - {item.primary_label} ({item.food_id})")

    print("\n✓ FoodMemory module test complete")

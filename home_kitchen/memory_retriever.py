#!/usr/bin/env python3
"""
Memory Retriever Module
Retrieves relevant food items from memory with context-aware filtering.
OPTIMIZATION: Smart filtering to reduce context size.
"""

from typing import List, Dict
from food_memory import FoodMemory, FoodItem
from datetime import datetime, timedelta


class MemoryRetriever:
    """
    Retrieves relevant food items from memory based on detected food names.
    Uses context-aware filtering to prioritize recent and relevant items.
    """

    def __init__(self, memory: FoodMemory):
        """
        Initialize MemoryRetriever.

        Args:
            memory: FoodMemory instance to query
        """
        self.memory = memory

    def retrieve_context(self, detected_food_names: List[str],
                        max_items_per_food: int = 3) -> str:
        """
        Retrieve relevant context for detected food names.

        OPTIMIZATION: Context-aware filtering
        - Pre-filter candidates using text similarity
        - Prioritize recent interactions
        - Limit results per food name to reduce context size

        Args:
            detected_food_names: List of food names from VLM Call 1
            max_items_per_food: Maximum matching items per food name

        Returns:
            Human-readable context block string
        """
        if not detected_food_names:
            return "CONTEXT: No food items detected in previous analysis."

        print(f"\n[Memory Retrieval] Searching for context:")
        print(f"  Detected food names: {detected_food_names}")

        context_items = []

        for food_name in detected_food_names:
            # Search for matching items in memory
            matches = self.memory.search_by_label(food_name, top_k=max_items_per_food)

            if matches:
                print(f"  Found {len(matches)} match(es) for '{food_name}'")
                context_items.extend(matches)
            else:
                print(f"  No matches found for '{food_name}' (likely new item)")

        # Remove duplicates (in case multiple food names matched same item)
        unique_items = list({item.food_id: item for item in context_items}.values())

        # Sort by most recent interaction (prioritize recently used items)
        unique_items.sort(
            key=lambda x: x.interaction_history[-1]["timestamp"] if x.interaction_history else "",
            reverse=True
        )

        print(f"  Total unique items in context: {len(unique_items)}")

        # Format context block
        if not unique_items:
            return "CONTEXT: No matching items found in memory. All detected items are likely new."

        context_block = self._format_context_block(unique_items)
        return context_block

    def _format_context_block(self, items: List[FoodItem]) -> str:
        """
        Format food items into a human-readable context block.

        Args:
            items: List of FoodItem objects

        Returns:
            Formatted context string
        """
        lines = ["CONTEXT: The food memory contains the following potentially relevant items:\n"]

        for i, item in enumerate(items, 1):
            lines.append(f"{i}. {item.primary_label} (ID: {item.food_id})")
            lines.append(f"   Description: {item.description}")
            lines.append(f"   Current Location: {item.current_location}")
            lines.append(f"   Current Quantity: {item.current_quantity}")

            # Add last interaction if available
            if item.interaction_history:
                last_interaction = item.interaction_history[-1]
                lines.append(
                    f"   Last Interaction: {last_interaction['action']} "
                    f"at {last_interaction['timestamp']}"
                )

            lines.append("")  # Blank line between items

        return "\n".join(lines)

    def get_location_summary(self) -> str:
        """
        Get a summary of items by location.

        Returns:
            Summary string of items grouped by location
        """
        # Group items by location
        location_map: Dict[str, List[str]] = {}

        for item in self.memory.items.values():
            loc = item.current_location
            if loc not in location_map:
                location_map[loc] = []
            location_map[loc].append(item.primary_label)

        # Format summary
        lines = ["Location Summary:"]
        for location, items in sorted(location_map.items()):
            lines.append(f"  {location}: {', '.join(items)}")

        return "\n".join(lines)


if __name__ == "__main__":
    # Test the MemoryRetriever module
    print("Testing MemoryRetriever module...\n")

    # Create test memory
    from food_memory import FoodMemory

    memory = FoodMemory("test_food_memory.json")

    # Add some test items if memory is empty
    if not memory.items:
        print("Creating test memory items...\n")

        memory.create_item(
            primary_label="Kroger Whole Milk Gallon",
            description="Gallon of whole milk",
            visual_embedding=[0.1, 0.2, 0.3],
            location="fridge",
            quantity="full"
        )

        memory.create_item(
            primary_label="Raw Chicken Breast",
            description="Package of raw chicken breast",
            visual_embedding=[0.2, 0.3, 0.4],
            location="fridge",
            quantity="1 package"
        )

        memory.save()

    # Test retrieval
    retriever = MemoryRetriever(memory)

    test_detected_names = ["milk", "chicken"]
    context = retriever.retrieve_context(test_detected_names)

    print("\n" + "="*60)
    print("Retrieved Context:")
    print("="*60)
    print(context)
    print("="*60)

    # Test location summary
    print("\n" + retriever.get_location_summary())

    print("\n✓ MemoryRetriever module test complete")

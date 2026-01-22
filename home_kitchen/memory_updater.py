#!/usr/bin/env python3
"""
Memory Updater Module
Executes update commands from VLM Call 2 on the FoodMemory.
"""

from typing import List, Dict
from datetime import datetime
from food_memory import FoodMemory


class MemoryUpdater:
    """
    Executes update commands on the FoodMemory.
    Handles both CREATE and UPDATE operations.
    """

    def __init__(self, memory: FoodMemory, logger=None):
        """
        Initialize MemoryUpdater.

        Args:
            memory: FoodMemory instance to update
            logger: VLMLogger instance for logging
        """
        self.memory = memory
        self.logger = logger

    def execute_commands(self, commands: List[Dict], video_timestamp: str,
                        video_name: str = "", clip_index: int = 0) -> Dict[str, int]:
        """
        Execute a list of update commands.

        Args:
            commands: List of command dictionaries from VLM Call 2
            video_timestamp: Timestamp of the video being processed
            video_name: Parent video name (for logging)
            clip_index: Clip index (for logging)

        Returns:
            Statistics: {"created": N, "updated": M, "failed": K}
        """
        stats = {"created": 0, "updated": 0, "failed": 0}

        print(f"\n[Memory Updater] Executing {len(commands)} commands")

        for i, cmd in enumerate(commands, 1):
            print(f"  Command {i}/{len(commands)}: {cmd.get('command', 'UNKNOWN')}")

            try:
                if cmd["command"] == "CREATE":
                    self._execute_create(cmd["data"], video_timestamp)
                    stats["created"] += 1

                elif cmd["command"] == "UPDATE":
                    success = self._execute_update(cmd["food_id"], cmd["data"], video_timestamp)
                    if success:
                        stats["updated"] += 1
                    else:
                        stats["failed"] += 1

                else:
                    print(f"    ⚠ Unknown command: {cmd.get('command')}")
                    stats["failed"] += 1

            except KeyError as e:
                print(f"    ✗ Missing required field: {e}")
                stats["failed"] += 1
            except Exception as e:
                print(f"    ✗ Error executing command: {e}")
                stats["failed"] += 1

        print(f"  Stats: {stats['created']} created, {stats['updated']} updated, {stats['failed']} failed")

        # Log memory update
        if self.logger:
            self.logger.log_memory_update(
                commands=commands,
                stats=stats,
                video_name=video_name,
                clip_index=clip_index,
                video_timestamp=video_timestamp
            )

        return stats

    def _execute_create(self, data: Dict, video_timestamp: str):
        """
        Execute a CREATE command.

        Args:
            data: Data dictionary with item properties
            video_timestamp: Timestamp of the video
        """
        # Create the item
        food_id = self.memory.create_item(
            primary_label=data["primary_label"],
            description=data.get("description", ""),
            visual_embedding=data.get("visual_embedding", [0.0, 0.0, 0.0]),
            location=data.get("location", "unknown"),
            quantity=data.get("quantity", "unknown")
        )

        # Add initial interaction
        action = data.get("action", "created")
        self.memory.update_item(
            food_id,
            interaction={
                "timestamp": video_timestamp,
                "action": action
            }
        )

        print(f"    ✓ Created: {data['primary_label']} (ID: {food_id})")

    def _execute_update(self, food_id: str, data: Dict, video_timestamp: str) -> bool:
        """
        Execute an UPDATE command.

        Args:
            food_id: ID of the item to update
            data: Data dictionary with updates
            video_timestamp: Timestamp of the video

        Returns:
            True if successful, False otherwise
        """
        # Check if item exists
        item = self.memory.get_item(food_id)
        if not item:
            print(f"    ⚠ Item not found: {food_id}")
            return False

        # Prepare updates
        location = data.get("location")
        quantity = data.get("quantity")
        action = data.get("action", "interaction")

        # Build interaction record
        interaction = {
            "timestamp": video_timestamp,
            "action": action
        }

        # Add any extra fields from data
        for key, value in data.items():
            if key not in ["location", "quantity", "action"]:
                interaction[key] = value

        # Execute update
        success = self.memory.update_item(
            food_id,
            location=location,
            quantity=quantity,
            interaction=interaction
        )

        if success:
            print(f"    ✓ Updated: {item.primary_label} (ID: {food_id})")
            print(f"      Action: {action}")
            if location:
                print(f"      New location: {location}")
            if quantity:
                print(f"      New quantity: {quantity}")

        return success


if __name__ == "__main__":
    # Test the MemoryUpdater module
    print("Testing MemoryUpdater module...\n")

    from food_memory import FoodMemory

    # Create test memory
    memory = FoodMemory("test_food_memory.json")

    # Create updater
    updater = MemoryUpdater(memory)

    # Test commands
    test_commands = [
        {
            "command": "CREATE",
            "data": {
                "primary_label": "Fresh Bananas",
                "description": "Bunch of fresh organic bananas",
                "visual_embedding": [0.1, 0.2, 0.3],
                "location": "counter",
                "quantity": "1 bunch",
                "action": "purchased and un-bagged"
            }
        },
        {
            "command": "UPDATE",
            "food_id": "milk_test123",
            "data": {
                "location": "counter",
                "quantity": "half-full",
                "action": "retrieve from fridge"
            }
        }
    ]

    # Execute commands
    video_timestamp = "2025-10-29T20:00:00"
    stats = updater.execute_commands(test_commands, video_timestamp)

    print(f"\n{'='*60}")
    print(f"Execution Stats: {stats}")
    print('='*60)

    # Save memory
    memory.save()

    print("\n✓ MemoryUpdater module test complete")

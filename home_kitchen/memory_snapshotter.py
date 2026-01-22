#!/usr/bin/env python3
"""
Memory Snapshotter Module
Saves snapshots of the FoodMemory after processing each video.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Optional
from food_memory import FoodMemory


class MemorySnapshotter:
    """
    Creates and manages snapshots of FoodMemory state.
    Saves a snapshot after each video is processed.
    """

    def __init__(self, snapshot_dir: str = "snapshots"):
        """
        Initialize MemorySnapshotter.

        Args:
            snapshot_dir: Directory to store snapshot files
        """
        self.snapshot_dir = Path(snapshot_dir)
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)

    def save_snapshot(self, memory: FoodMemory, video_name: str,
                     video_timestamp: Optional[str] = None) -> str:
        """
        Save a snapshot of the current memory state.

        Args:
            memory: FoodMemory instance to snapshot
            video_name: Name of the video that was just processed
            video_timestamp: Optional timestamp from video filename

        Returns:
            Path to the snapshot file
        """
        # Extract timestamp from video name if not provided
        if not video_timestamp:
            video_file = Path(video_name)
            video_timestamp = video_file.stem  # e.g., "20251029-193737"

        # Create snapshot filename
        snapshot_name = f"snapshot_{video_timestamp}.json"
        snapshot_path = self.snapshot_dir / snapshot_name

        # Create snapshot data with metadata
        snapshot_data = {
            "metadata": {
                "video_name": video_name,
                "video_timestamp": video_timestamp,
                "snapshot_time": datetime.now().isoformat(),
                "total_items": len(memory.items)
            },
            "memory": memory.to_dict()
        }

        # Save snapshot
        try:
            with open(snapshot_path, 'w') as f:
                json.dump(snapshot_data, f, indent=2)

            print(f"\n[Snapshot] Saved memory state to: {snapshot_path.name}")
            print(f"  Total items: {len(memory.items)}")
            print(f"  Video: {video_name}")

            return str(snapshot_path)

        except Exception as e:
            print(f"[Snapshot] Error saving snapshot: {e}")
            return ""

    def load_snapshot(self, snapshot_path: str) -> Optional[FoodMemory]:
        """
        Load a memory snapshot from file.

        Args:
            snapshot_path: Path to snapshot file

        Returns:
            FoodMemory instance loaded from snapshot, or None on error
        """
        path = Path(snapshot_path)

        if not path.exists():
            print(f"[Snapshot] Snapshot not found: {snapshot_path}")
            return None

        try:
            with open(path, 'r') as f:
                snapshot_data = json.load(f)

            # Create a temporary memory instance
            memory = FoodMemory(storage_path=":memory:")  # In-memory only

            # Load items from snapshot
            memory_data = snapshot_data.get("memory", {})
            items_data = memory_data.get("items", {})

            from food_memory import FoodItem
            memory.items = {
                food_id: FoodItem.from_dict(item_data)
                for food_id, item_data in items_data.items()
            }

            # Print metadata
            metadata = snapshot_data.get("metadata", {})
            print(f"\n[Snapshot] Loaded snapshot: {path.name}")
            print(f"  Video: {metadata.get('video_name', 'unknown')}")
            print(f"  Items: {len(memory.items)}")
            print(f"  Snapshot time: {metadata.get('snapshot_time', 'unknown')}")

            return memory

        except Exception as e:
            print(f"[Snapshot] Error loading snapshot: {e}")
            return None

    def get_latest_snapshot(self) -> Optional[str]:
        """
        Find the most recent snapshot file.

        Returns:
            Path to latest snapshot, or None if no snapshots exist
        """
        snapshots = sorted(self.snapshot_dir.glob("snapshot_*.json"))

        if snapshots:
            latest = snapshots[-1]
            print(f"[Snapshot] Found latest snapshot: {latest.name}")
            return str(latest)

        print(f"[Snapshot] No snapshots found in {self.snapshot_dir}")
        return None

    def list_snapshots(self) -> list:
        """
        List all available snapshots.

        Returns:
            List of snapshot file paths
        """
        snapshots = sorted(self.snapshot_dir.glob("snapshot_*.json"))
        return [str(s) for s in snapshots]


if __name__ == "__main__":
    # Test the MemorySnapshotter module
    print("Testing MemorySnapshotter module...\n")

    from food_memory import FoodMemory

    # Create test memory
    memory = FoodMemory("test_food_memory.json")

    # Add a test item if empty
    if not memory.items:
        memory.create_item(
            primary_label="Test Milk",
            description="Test milk item",
            visual_embedding=[0.1, 0.2, 0.3],
            location="fridge",
            quantity="full"
        )

    # Create snapshotter
    snapshotter = MemorySnapshotter("test_snapshots")

    # Save snapshot
    snapshot_path = snapshotter.save_snapshot(
        memory,
        video_name="20251029-193737.MOV",
        video_timestamp="20251029-193737"
    )

    print(f"\n{'='*60}")
    print(f"Snapshot saved to: {snapshot_path}")
    print('='*60)

    # List snapshots
    print("\nAvailable snapshots:")
    for snap in snapshotter.list_snapshots():
        print(f"  - {Path(snap).name}")

    # Test loading
    print("\nTesting snapshot load...")
    loaded_memory = snapshotter.load_snapshot(snapshot_path)
    if loaded_memory:
        print(f"✓ Successfully loaded {len(loaded_memory.items)} items")

    # Test get latest
    print("\nTesting latest snapshot retrieval...")
    latest = snapshotter.get_latest_snapshot()
    if latest:
        print(f"✓ Latest snapshot: {Path(latest).name}")

    print("\n✓ MemorySnapshotter module test complete")

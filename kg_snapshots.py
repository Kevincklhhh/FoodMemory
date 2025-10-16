#!/usr/bin/env python3
"""
Knowledge Graph Snapshot Management
Saves KG state after each narration for evaluation purposes.
"""

import json
import copy
from pathlib import Path
from typing import Dict, Any


class KGSnapshotManager:
    """Manages KG snapshots for temporal evaluation."""

    def __init__(self, snapshots_dir: str = "kg_snapshots"):
        """
        Initialize snapshot manager.

        Args:
            snapshots_dir: Directory to store snapshots
        """
        self.snapshots_dir = Path(snapshots_dir)
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)

        # Metadata file tracks all snapshots
        self.metadata_file = self.snapshots_dir / "snapshots_metadata.jsonl"

    def save_snapshot(
        self,
        kg: Dict[str, Any],
        narration_id: str,
        video_id: str,
        start_time: float,
        end_time: float,
        narration_text: str,
        success: bool,
        reason: str = None
    ) -> str:
        """
        Save a KG snapshot after processing a narration.

        Args:
            kg: Current knowledge graph state
            narration_id: Unique narration identifier (e.g., "P01-20240202-110250-1")
            video_id: Video identifier
            start_time: Narration start time
            end_time: Narration end time
            narration_text: Full narration text
            success: Whether the update was successful
            reason: Optional reason for failure (if success=False)

        Returns:
            Path to saved snapshot file
        """
        # Create deep copy to avoid mutation
        kg_snapshot = copy.deepcopy(kg)

        # Add snapshot metadata
        snapshot_info = {
            "narration_id": narration_id,
            "video_id": video_id,
            "start_time": start_time,
            "end_time": end_time,
            "narration_text": narration_text,
            "update_success": success,
            "snapshot_metadata": {
                "num_foods": len(kg_snapshot.get("foods", {})),
                "num_zones": len(kg_snapshot.get("zones", {})),
                "total_interactions": sum(
                    len(food.get("interaction_history", []))
                    for food in kg_snapshot.get("foods", {}).values()
                )
            }
        }

        if reason:
            snapshot_info["failure_reason"] = reason

        # Save snapshot to individual file
        snapshot_filename = f"snapshot_{narration_id}.json"
        snapshot_path = self.snapshots_dir / snapshot_filename

        full_snapshot = {
            "snapshot_info": snapshot_info,
            "kg_state": kg_snapshot
        }

        with open(snapshot_path, 'w') as f:
            json.dump(full_snapshot, f, indent=2)

        # Append to metadata log
        metadata_entry = {
            "narration_id": narration_id,
            "snapshot_file": snapshot_filename,
            "video_id": video_id,
            "start_time": start_time,
            "end_time": end_time,
            "narration_text": narration_text,
            "success": success,
            "num_foods": snapshot_info["snapshot_metadata"]["num_foods"],
            "num_zones": snapshot_info["snapshot_metadata"]["num_zones"],
            "total_interactions": snapshot_info["snapshot_metadata"]["total_interactions"]
        }

        with open(self.metadata_file, 'a') as f:
            f.write(json.dumps(metadata_entry) + '\n')

        return str(snapshot_path)

    def load_snapshot(self, narration_id: str) -> Dict[str, Any]:
        """
        Load a specific snapshot by narration ID.

        Args:
            narration_id: Narration identifier

        Returns:
            Full snapshot dict with snapshot_info and kg_state
        """
        snapshot_filename = f"snapshot_{narration_id}.json"
        snapshot_path = self.snapshots_dir / snapshot_filename

        if not snapshot_path.exists():
            raise FileNotFoundError(f"Snapshot not found: {snapshot_path}")

        with open(snapshot_path, 'r') as f:
            return json.load(f)

    def get_kg_at_time(self, video_id: str, timestamp: float) -> Dict[str, Any]:
        """
        Get KG state at a specific timestamp.

        Args:
            video_id: Video identifier
            timestamp: Time in seconds

        Returns:
            KG state at that time (most recent snapshot before timestamp)
        """
        # Read metadata to find relevant snapshot
        if not self.metadata_file.exists():
            return None

        latest_snapshot = None
        latest_time = -1

        with open(self.metadata_file, 'r') as f:
            for line in f:
                metadata = json.loads(line)
                if metadata['video_id'] == video_id and metadata['start_time'] <= timestamp:
                    if metadata['start_time'] > latest_time:
                        latest_time = metadata['start_time']
                        latest_snapshot = metadata['narration_id']

        if latest_snapshot:
            snapshot = self.load_snapshot(latest_snapshot)
            return snapshot['kg_state']

        return None

    def list_snapshots(self, video_id: str = None) -> list:
        """
        List all snapshots, optionally filtered by video.

        Args:
            video_id: Optional video ID to filter by

        Returns:
            List of snapshot metadata dicts
        """
        if not self.metadata_file.exists():
            return []

        snapshots = []
        with open(self.metadata_file, 'r') as f:
            for line in f:
                metadata = json.loads(line)
                if video_id is None or metadata['video_id'] == video_id:
                    snapshots.append(metadata)

        return snapshots

    def get_summary_stats(self) -> Dict[str, Any]:
        """
        Get summary statistics across all snapshots.

        Returns:
            Dict with summary statistics
        """
        if not self.metadata_file.exists():
            return {
                "total_snapshots": 0,
                "videos": [],
                "max_foods": 0,
                "max_zones": 0,
                "max_interactions": 0
            }

        videos = set()
        max_foods = 0
        max_zones = 0
        max_interactions = 0
        total_snapshots = 0

        with open(self.metadata_file, 'r') as f:
            for line in f:
                metadata = json.loads(line)
                videos.add(metadata['video_id'])
                max_foods = max(max_foods, metadata['num_foods'])
                max_zones = max(max_zones, metadata['num_zones'])
                max_interactions = max(max_interactions, metadata['total_interactions'])
                total_snapshots += 1

        return {
            "total_snapshots": total_snapshots,
            "videos": sorted(list(videos)),
            "max_foods": max_foods,
            "max_zones": max_zones,
            "max_interactions": max_interactions
        }


def main():
    """CLI for snapshot management."""
    import argparse

    parser = argparse.ArgumentParser(description='Manage KG snapshots')
    parser.add_argument('--snapshots-dir', default='kg_snapshots',
                        help='Snapshots directory')
    parser.add_argument('--list', action='store_true',
                        help='List all snapshots')
    parser.add_argument('--stats', action='store_true',
                        help='Show summary statistics')
    parser.add_argument('--video-id', help='Filter by video ID')
    parser.add_argument('--load', help='Load specific snapshot by narration ID')
    parser.add_argument('--at-time', type=float,
                        help='Get KG at specific timestamp (requires --video-id)')

    args = parser.parse_args()

    manager = KGSnapshotManager(args.snapshots_dir)

    if args.stats:
        stats = manager.get_summary_stats()
        print("Snapshot Statistics:")
        print(f"  Total snapshots: {stats['total_snapshots']}")
        print(f"  Videos: {', '.join(stats['videos'])}")
        print(f"  Max foods tracked: {stats['max_foods']}")
        print(f"  Max zones: {stats['max_zones']}")
        print(f"  Max interactions: {stats['max_interactions']}")

    elif args.list:
        snapshots = manager.list_snapshots(args.video_id)
        print(f"Found {len(snapshots)} snapshots:")
        for s in snapshots:
            print(f"  {s['narration_id']}: {s['start_time']:.1f}s - "
                  f"{s['num_foods']} foods, {s['num_zones']} zones "
                  f"({'✓' if s['success'] else '✗'})")

    elif args.load:
        snapshot = manager.load_snapshot(args.load)
        print(json.dumps(snapshot, indent=2))

    elif args.at_time is not None:
        if not args.video_id:
            print("Error: --at-time requires --video-id")
            return

        kg = manager.get_kg_at_time(args.video_id, args.at_time)
        if kg:
            print(f"KG state at {args.at_time}s:")
            print(json.dumps(kg, indent=2))
        else:
            print(f"No snapshot found for {args.video_id} at {args.at_time}s")


if __name__ == "__main__":
    main()

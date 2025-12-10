#!/usr/bin/env python3
"""
Semantic Group Utilities

Shared utilities for semantic group-based processing of HDEPIC narrations.
Provides common functions used across pipelines:
- Loading semantic groupings from JSON
- Enriching groups with CSV narration data
- Finding videos with semantic groupings
"""

import json
import csv
from pathlib import Path
from typing import List, Dict, Optional, Set
from dataclasses import dataclass, field


@dataclass
class SemanticGroup:
    """Represents a semantically-grouped set of narrations.

    Each semantic group corresponds to a coherent action (e.g., "take a mug from the cupboard")
    composed of multiple individual narration steps.
    """
    group_id: int               # Sequential index within video (0, 1, 2, ...)
    video_id: str               # e.g., "P01-20240202-110250"
    query: str                  # Merged action description
    start_time: float           # Start timestamp in seconds
    end_time: float             # End timestamp in seconds
    merged_ids: List[str]       # List of unique_narration_id values
    narrations: List[Dict] = field(default_factory=list)  # Full narration records from CSV
    has_food_action: Optional[bool] = None  # VLM classification result

    @property
    def duration(self) -> float:
        """Duration of the semantic group in seconds."""
        return self.end_time - self.start_time

    @property
    def num_narrations(self) -> int:
        """Number of narrations in this group."""
        return len(self.narrations)

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'group_id': self.group_id,
            'video_id': self.video_id,
            'query': self.query,
            'group_start_time': self.start_time,
            'group_end_time': self.end_time,
            'duration': self.duration,
            'merged_ids': self.merged_ids,
            'num_narrations': self.num_narrations,
            'narrations': self.narrations,
            'has_food_action': self.has_food_action,
        }

    def get_narration_text(self) -> str:
        """Get formatted narration text for VLM prompts.

        Returns narrations sorted by timestamp with line numbers.
        """
        lines = []
        sorted_narrations = sorted(self.narrations, key=lambda n: n.get('start_timestamp', 0))
        for i, n in enumerate(sorted_narrations, 1):
            start = n.get('start_timestamp', 0)
            text = n.get('narration', '').strip()
            lines.append(f"{i}. [{start:.2f}s] {text}")
        return "\n".join(lines)


def find_videos_with_groupings(grouping_dir: Path) -> Set[str]:
    """Find all video IDs that have semantic grouping files.

    Args:
        grouping_dir: Path to narration_grouping directory

    Returns:
        Set of video IDs (e.g., {"P01-20240202-110250", "P01-20240203-093333"})
    """
    video_ids = set()
    grouping_dir = Path(grouping_dir)

    for f in grouping_dir.glob("*_anonymized.json"):
        video_id = f.stem.replace("_anonymized", "")
        video_ids.add(video_id)

    return video_ids


def load_semantic_groupings(grouping_dir: Path, video_id: str) -> Optional[List[Dict]]:
    """Load raw semantic groupings for a video.

    Args:
        grouping_dir: Path to narration_grouping directory
        video_id: e.g., "P01-20240202-110250"

    Returns:
        List of group dicts with keys: query, start, end, merged_id
        None if no grouping file exists (video should be skipped)
    """
    grouping_dir = Path(grouping_dir)
    grouping_file = grouping_dir / f"{video_id}_anonymized.json"

    if not grouping_file.exists():
        return None

    with open(grouping_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_narrations_for_video(csv_path: Path, video_id: str) -> List[Dict]:
    """Load narrations for a specific video from CSV.

    Args:
        csv_path: Path to participant_P01_narrations.csv
        video_id: e.g., "P01-20240202-110250"

    Returns:
        List of narration dicts with all CSV fields
    """
    csv_path = Path(csv_path)
    narrations = []

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['video_id'] == video_id:
                narration = {
                    'unique_narration_id': row['unique_narration_id'],
                    'participant_id': row['participant_id'],
                    'video_id': row['video_id'],
                    'narration': row['narration'],
                    'start_timestamp': float(row['start_timestamp']),
                    'end_timestamp': float(row['end_timestamp']),
                    'narration_timestamp': float(row['narration_timestamp']),
                }

                # Parse list fields safely
                try:
                    narration['nouns'] = eval(row['nouns']) if row.get('nouns') else []
                except:
                    narration['nouns'] = []

                try:
                    narration['verbs'] = eval(row['verbs']) if row.get('verbs') else []
                except:
                    narration['verbs'] = []

                try:
                    narration['hands'] = eval(row['hands']) if row.get('hands') else []
                except:
                    narration['hands'] = []

                narrations.append(narration)

    return narrations


def load_all_narrations(csv_path: Path) -> Dict[str, Dict]:
    """Load all narrations from CSV into a lookup dict.

    Args:
        csv_path: Path to participant_P01_narrations.csv

    Returns:
        Dict mapping unique_narration_id to narration dict
    """
    csv_path = Path(csv_path)
    narrations = {}

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            narration = {
                'unique_narration_id': row['unique_narration_id'],
                'participant_id': row['participant_id'],
                'video_id': row['video_id'],
                'narration': row['narration'],
                'start_timestamp': float(row['start_timestamp']),
                'end_timestamp': float(row['end_timestamp']),
                'narration_timestamp': float(row['narration_timestamp']),
            }

            # Parse list fields safely
            try:
                narration['nouns'] = eval(row['nouns']) if row.get('nouns') else []
            except:
                narration['nouns'] = []

            try:
                narration['verbs'] = eval(row['verbs']) if row.get('verbs') else []
            except:
                narration['verbs'] = []

            try:
                narration['hands'] = eval(row['hands']) if row.get('hands') else []
            except:
                narration['hands'] = []

            narrations[narration['unique_narration_id']] = narration

    return narrations


def enrich_groups_with_narrations(
    raw_groups: List[Dict],
    narrations: List[Dict],
    video_id: str
) -> List[SemanticGroup]:
    """Convert raw groupings to SemanticGroup objects with enriched narration data.

    Args:
        raw_groups: Raw groups from JSON (query, start, end, merged_id)
        narrations: Full narration records from CSV for this video
        video_id: The video ID

    Returns:
        List of SemanticGroup objects with enriched narrations
    """
    # Build lookup by unique_narration_id
    narration_lookup = {n['unique_narration_id']: n for n in narrations}

    enriched = []
    for i, group in enumerate(raw_groups):
        # Look up each merged narration
        group_narrations = []
        for narr_id in group.get('merged_id', []):
            if narr_id in narration_lookup:
                group_narrations.append(narration_lookup[narr_id])

        sg = SemanticGroup(
            group_id=i,
            video_id=video_id,
            query=group.get('query', ''),
            start_time=float(group.get('start', 0)),
            end_time=float(group.get('end', 0)),
            merged_ids=group.get('merged_id', []),
            narrations=group_narrations,
            has_food_action=None,
        )
        enriched.append(sg)

    return enriched


def enrich_groups_with_narration_lookup(
    raw_groups: List[Dict],
    narration_lookup: Dict[str, Dict],
    video_id: str
) -> List[SemanticGroup]:
    """Convert raw groupings to SemanticGroup objects using pre-built lookup.

    More efficient when processing multiple videos with same narration lookup.

    Args:
        raw_groups: Raw groups from JSON (query, start, end, merged_id)
        narration_lookup: Dict mapping unique_narration_id to narration dict
        video_id: The video ID

    Returns:
        List of SemanticGroup objects with enriched narrations
    """
    enriched = []
    for i, group in enumerate(raw_groups):
        group_narrations = []
        for narr_id in group.get('merged_id', []):
            if narr_id in narration_lookup:
                group_narrations.append(narration_lookup[narr_id])

        sg = SemanticGroup(
            group_id=i,
            video_id=video_id,
            query=group.get('query', ''),
            start_time=float(group.get('start', 0)),
            end_time=float(group.get('end', 0)),
            merged_ids=group.get('merged_id', []),
            narrations=group_narrations,
            has_food_action=None,
        )
        enriched.append(sg)

    return enriched


def load_and_enrich_video_groups(
    grouping_dir: Path,
    csv_path: Path,
    video_id: str
) -> Optional[List[SemanticGroup]]:
    """Convenience function to load and enrich groups for a single video.

    Args:
        grouping_dir: Path to narration_grouping directory
        csv_path: Path to participant_P01_narrations.csv
        video_id: e.g., "P01-20240202-110250"

    Returns:
        List of enriched SemanticGroup objects, or None if no groupings exist
    """
    # Load raw groupings
    raw_groups = load_semantic_groupings(grouping_dir, video_id)
    if raw_groups is None:
        return None

    # Load narrations for this video
    narrations = load_narrations_for_video(csv_path, video_id)

    # Enrich and return
    return enrich_groups_with_narrations(raw_groups, narrations, video_id)


# CLI for testing
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test semantic group utilities")
    parser.add_argument('--grouping-dir', type=str, default='../narration_grouping',
                       help='Path to narration_grouping directory')
    parser.add_argument('--csv', type=str, default='../participant_P01_narrations.csv',
                       help='Path to narrations CSV')
    parser.add_argument('--video-id', type=str, default='P01-20240202-110250',
                       help='Video ID to test')
    args = parser.parse_args()

    grouping_dir = Path(args.grouping_dir)
    csv_path = Path(args.csv)

    print("=" * 60)
    print("Testing Semantic Group Utilities")
    print("=" * 60)

    # Test find_videos_with_groupings
    print("\n1. Finding videos with groupings...")
    videos = find_videos_with_groupings(grouping_dir)
    print(f"   Found {len(videos)} videos with semantic groupings")
    for v in sorted(list(videos))[:5]:
        print(f"   - {v}")
    if len(videos) > 5:
        print(f"   ... and {len(videos) - 5} more")

    # Test load_and_enrich_video_groups
    print(f"\n2. Loading groups for {args.video_id}...")
    groups = load_and_enrich_video_groups(grouping_dir, csv_path, args.video_id)

    if groups is None:
        print(f"   ERROR: No groupings found for {args.video_id}")
    else:
        print(f"   Loaded {len(groups)} semantic groups")

        # Show first 3 groups
        print("\n3. Sample groups:")
        for g in groups[:3]:
            print(f"\n   Group {g.group_id}: \"{g.query}\"")
            print(f"   Time: {g.start_time:.1f}s - {g.end_time:.1f}s (duration: {g.duration:.1f}s)")
            print(f"   Narrations: {g.num_narrations}")
            if g.narrations:
                print(f"   First narration: {g.narrations[0]['narration'][:60]}...")

        # Statistics
        print("\n4. Statistics:")
        durations = [g.duration for g in groups]
        narr_counts = [g.num_narrations for g in groups]
        print(f"   Total groups: {len(groups)}")
        print(f"   Duration range: {min(durations):.1f}s - {max(durations):.1f}s")
        print(f"   Avg duration: {sum(durations)/len(durations):.1f}s")
        print(f"   Narrations per group: {min(narr_counts)} - {max(narr_counts)} (avg: {sum(narr_counts)/len(narr_counts):.1f})")

        # Check for groups < 10s
        short_groups = [g for g in groups if g.duration < 10]
        print(f"\n   Groups with duration < 10s: {len(short_groups)} ({100*len(short_groups)/len(groups):.1f}%)")

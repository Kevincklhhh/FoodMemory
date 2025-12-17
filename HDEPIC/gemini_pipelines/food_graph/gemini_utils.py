#!/usr/bin/env python3
"""
Gemini Utils - Utilities for processing Gemini pre-annotations.

Provides:
- NarrationLookup: Load and index narrations from CSV for timestamp lookups
- VideoClipExtractor: Extract video clips on-demand for events
"""

import csv
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import Counter, defaultdict


class NarrationLookup:
    """Load and index narrations from CSV for timestamp lookups."""

    def __init__(self, csv_path: Path):
        """
        Initialize narration lookup from CSV file.

        Args:
            csv_path: Path to participant narrations CSV file
        """
        self.csv_path = Path(csv_path)
        self.narrations: Dict[str, Dict] = {}
        self._load_csv()

    def _load_csv(self):
        """Load CSV into dict indexed by unique_narration_id."""
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Narrations CSV not found: {self.csv_path}")

        with open(self.csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                narr_id = row['unique_narration_id']
                self.narrations[narr_id] = {
                    'unique_narration_id': narr_id,
                    'participant_id': row.get('participant_id', ''),
                    'video_id': row.get('video_id', ''),
                    'narration': row.get('narration', '').strip(),
                    'start_timestamp': float(row.get('start_timestamp', 0)),
                    'end_timestamp': float(row.get('end_timestamp', 0)),
                    'narration_timestamp': float(row.get('narration_timestamp', 0)),
                }

        print(f"[NarrationLookup] Loaded {len(self.narrations)} narrations from {self.csv_path.name}")

    def get_narration(self, narration_id: str) -> Optional[Dict]:
        """
        Get narration metadata by ID.

        Args:
            narration_id: Unique narration ID (e.g., "P01-20240202-161948-6")

        Returns:
            Dict with narration metadata or None if not found
        """
        return self.narrations.get(narration_id)

    def get_video_id(self, narration_id: str) -> Optional[str]:
        """Extract video_id from narration."""
        narr = self.get_narration(narration_id)
        return narr['video_id'] if narr else None

    def get_timestamp_range(self, narration_ids: List[str]) -> Tuple[Optional[str], float, float]:
        """
        Get video_id and time range spanning all narration IDs.

        If narration IDs span multiple videos (rare), returns the video
        containing the MAJORITY of narrations.

        Args:
            narration_ids: List of narration IDs

        Returns:
            Tuple of (video_id, start_time, end_time) or (None, 0, 0) if no valid narrations
        """
        if not narration_ids:
            return None, 0.0, 0.0

        video_counts: Counter = Counter()
        timestamps_by_video: Dict[str, List[Tuple[float, float]]] = defaultdict(list)

        for narr_id in narration_ids:
            narr = self.get_narration(narr_id)
            if narr:
                vid = narr['video_id']
                video_counts[vid] += 1
                timestamps_by_video[vid].append(
                    (narr['start_timestamp'], narr['end_timestamp'])
                )

        if not video_counts:
            return None, 0.0, 0.0

        # Use majority video
        primary_video = video_counts.most_common(1)[0][0]
        times = timestamps_by_video[primary_video]

        start_time = min(t[0] for t in times)
        end_time = max(t[1] for t in times)

        # Warn if narrations span multiple videos
        if len(video_counts) > 1:
            print(f"  WARNING: Narrations span {len(video_counts)} videos, using {primary_video}")

        return primary_video, start_time, end_time


class VideoClipExtractor:
    """Extract video clips on-demand for events."""

    def __init__(
        self,
        video_dir: Path,
        cache_dir: Path,
        fps: int = 2,
        default_buffer: float = 2.0
    ):
        """
        Initialize video clip extractor.

        Args:
            video_dir: Directory containing source video files
            cache_dir: Directory to cache extracted clips
            fps: Output FPS for extracted clips (default: 2)
            default_buffer: Default buffer time in seconds before/after timestamps
        """
        self.video_dir = Path(video_dir)
        self.cache_dir = Path(cache_dir)
        self.fps = fps
        self.default_buffer = default_buffer

        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_video_path(self, video_id: str) -> Path:
        """Get path to source video file."""
        return self.video_dir / f"{video_id}.mp4"

    def extract_clip(
        self,
        video_id: str,
        start_time: float,
        end_time: float,
        clip_name: str,
        buffer_before: Optional[float] = None,
        buffer_after: Optional[float] = None
    ) -> Optional[Path]:
        """
        Extract a video clip with optional time buffers.

        Args:
            video_id: Source video ID
            start_time: Start timestamp in seconds
            end_time: End timestamp in seconds
            clip_name: Name for the output clip (without extension)
            buffer_before: Buffer seconds before start (default: self.default_buffer)
            buffer_after: Buffer seconds after end (default: self.default_buffer)

        Returns:
            Path to extracted clip or None if extraction failed
        """
        video_path = self.get_video_path(video_id)
        if not video_path.exists():
            print(f"  ERROR: Video not found: {video_path}")
            return None

        # Apply buffers
        buf_before = buffer_before if buffer_before is not None else self.default_buffer
        buf_after = buffer_after if buffer_after is not None else self.default_buffer

        actual_start = max(0, start_time - buf_before)
        actual_end = end_time + buf_after
        duration = actual_end - actual_start

        # Output path
        output_path = self.cache_dir / f"{clip_name}.mp4"

        # Skip if already cached
        if output_path.exists():
            return output_path

        # Extract using ffmpeg
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(actual_start),
            "-i", str(video_path),
            "-t", str(duration),
            "-vf", f"fps={self.fps}",
            "-c:v", "libx264",
            "-crf", "28",
            "-preset", "fast",
            "-an",  # No audio
            "-movflags", "+faststart",
            str(output_path)
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if result.returncode == 0:
                return output_path
            else:
                print(f"  ERROR: ffmpeg failed: {result.stderr[:200]}")
                return None
        except subprocess.TimeoutExpired:
            print(f"  ERROR: ffmpeg timeout for clip {clip_name}")
            return None
        except Exception as e:
            print(f"  ERROR: Failed to extract clip: {e}")
            return None

    def extract_clip_for_narration(
        self,
        narration_lookup: 'NarrationLookup',
        narration_id: str,
        clip_name: str,
        buffer_before: Optional[float] = None,
        buffer_after: Optional[float] = None
    ) -> Optional[Path]:
        """
        Extract a clip centered around a single narration.

        Args:
            narration_lookup: NarrationLookup instance
            narration_id: Narration ID to extract around
            clip_name: Name for the output clip
            buffer_before: Buffer before start timestamp
            buffer_after: Buffer after end timestamp

        Returns:
            Path to extracted clip or None if failed
        """
        narr = narration_lookup.get_narration(narration_id)
        if not narr:
            print(f"  ERROR: Narration not found: {narration_id}")
            return None

        return self.extract_clip(
            video_id=narr['video_id'],
            start_time=narr['start_timestamp'],
            end_time=narr['end_timestamp'],
            clip_name=clip_name,
            buffer_before=buffer_before,
            buffer_after=buffer_after
        )

    def extract_clip_for_event(
        self,
        narration_lookup: 'NarrationLookup',
        narration_ids: List[str],
        clip_name: str,
        buffer_before: Optional[float] = None,
        buffer_after: Optional[float] = None
    ) -> Optional[Path]:
        """
        Extract a clip spanning multiple narrations (for state change events).

        Args:
            narration_lookup: NarrationLookup instance
            narration_ids: List of narration IDs in the event
            clip_name: Name for the output clip
            buffer_before: Buffer before first narration start
            buffer_after: Buffer after last narration end

        Returns:
            Path to extracted clip or None if failed
        """
        video_id, start_time, end_time = narration_lookup.get_timestamp_range(narration_ids)
        if not video_id:
            print(f"  ERROR: Could not resolve timestamps for narration IDs")
            return None

        return self.extract_clip(
            video_id=video_id,
            start_time=start_time,
            end_time=end_time,
            clip_name=clip_name,
            buffer_before=buffer_before,
            buffer_after=buffer_after
        )

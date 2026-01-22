#!/usr/bin/env python3
"""
Video Processor Module
Splits videos into 30-second clips for VLM processing.
"""

import cv2
import subprocess
from pathlib import Path
from typing import List, Tuple, Optional
from datetime import datetime, timedelta


class VideoClip:
    """Represents a 30-second video clip."""

    def __init__(self, video_path: str, start_time: float, end_time: float,
                 clip_index: int, parent_timestamp: str):
        self.video_path = video_path
        self.start_time = start_time
        self.end_time = end_time
        self.clip_index = clip_index
        self.parent_timestamp = parent_timestamp  # From parent video filename

    def __repr__(self):
        return (f"VideoClip(clip_{self.clip_index}, "
                f"{self.start_time:.1f}s-{self.end_time:.1f}s)")


class VideoProcessor:
    """Processes videos into 30-second clips."""

    def __init__(self, clip_duration: int = 30):
        """
        Initialize VideoProcessor.

        Args:
            clip_duration: Duration of each clip in seconds (default: 30)
        """
        self.clip_duration = clip_duration

    def get_video_info(self, video_path: str) -> Tuple[float, float]:
        """
        Get video duration and FPS.

        Args:
            video_path: Path to video file

        Returns:
            Tuple of (duration_in_seconds, fps)
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0

        cap.release()

        return duration, fps

    def extract_timestamp_from_filename(self, video_path: str) -> str:
        """
        Extract timestamp from video filename.

        Args:
            video_path: Path to video file (e.g., "20251029-193737.MOV")

        Returns:
            ISO timestamp string
        """
        video_file = Path(video_path)
        filename = video_file.stem  # e.g., "20251029-193737"

        try:
            # Parse yyyymmdd-hhmmss format
            dt = datetime.strptime(filename, "%Y%m%d-%H%M%S")
            return dt.isoformat()
        except ValueError:
            print(f"Warning: Could not parse timestamp from {filename}")
            return datetime.now().isoformat()

    def split_into_clips(self, video_path: str) -> List[VideoClip]:
        """
        Split video into 30-second clips.

        Args:
            video_path: Path to video file

        Returns:
            List of VideoClip objects
        """
        video_path = str(Path(video_path).resolve())
        print(f"\nProcessing video: {video_path}")

        # Get video info
        duration, fps = self.get_video_info(video_path)
        print(f"  Duration: {duration:.1f}s, FPS: {fps:.2f}")

        # Extract parent timestamp
        parent_timestamp = self.extract_timestamp_from_filename(video_path)

        # Create clips
        clips = []
        clip_index = 0
        start_time = 0.0

        while start_time < duration:
            end_time = min(start_time + self.clip_duration, duration)

            clip = VideoClip(
                video_path=video_path,
                start_time=start_time,
                end_time=end_time,
                clip_index=clip_index,
                parent_timestamp=parent_timestamp
            )
            clips.append(clip)

            start_time = end_time
            clip_index += 1

        print(f"  Created {len(clips)} clips of {self.clip_duration}s each")
        return clips

    def extract_clip_to_file(self, clip: VideoClip, output_dir: str) -> str:
        """
        Extract a clip to a separate video file using ffmpeg.

        Args:
            clip: VideoClip object
            output_dir: Directory to save clip

        Returns:
            Path to extracted clip file
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate output filename
        video_name = Path(clip.video_path).stem
        output_name = f"{video_name}_clip_{clip.clip_index:03d}.mp4"
        output_path = output_dir / output_name

        # Skip if already exists
        if output_path.exists():
            print(f"  Clip already exists: {output_path.name}")
            return str(output_path)

        # Extract clip using ffmpeg with re-encoding to ensure frame consistency
        duration = clip.end_time - clip.start_time
        cmd = [
            'ffmpeg',
            '-ss', str(clip.start_time),  # Seek before input for accuracy
            '-i', clip.video_path,
            '-t', str(duration),
            '-c:v', 'libx264',  # Re-encode video
            '-preset', 'medium',
            '-crf', '23',  # Quality setting
            '-r', '30',  # Fixed frame rate at 30 FPS
            '-c:a', 'aac',  # Re-encode audio
            '-b:a', '128k',
            '-y',  # Overwrite if exists
            str(output_path)
        ]

        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True
            )
            print(f"  Extracted: {output_path.name}")
            return str(output_path)
        except subprocess.CalledProcessError as e:
            print(f"  Error extracting clip: {e}")
            print(f"  stderr: {e.stderr.decode()[:200]}")
            raise


if __name__ == "__main__":
    # Test the VideoProcessor module
    print("Testing VideoProcessor module...\n")

    processor = VideoProcessor(clip_duration=30)

    # Test with an existing video
    test_video = "20251029-193737.MOV"

    if Path(test_video).exists():
        # Get video info
        duration, fps = processor.get_video_info(test_video)
        print(f"Video info: {duration:.1f}s @ {fps:.2f} FPS")

        # Split into clips
        clips = processor.split_into_clips(test_video)

        print(f"\nGenerated clips:")
        for clip in clips:
            print(f"  {clip}")

        # Test extracting first clip
        if clips:
            print(f"\nExtracting first clip to 'test_clips/'...")
            clip_path = processor.extract_clip_to_file(clips[0], "test_clips")
            print(f"✓ Clip saved to: {clip_path}")
    else:
        print(f"Test video not found: {test_video}")

    print("\n✓ VideoProcessor module test complete")

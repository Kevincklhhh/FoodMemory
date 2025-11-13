#!/usr/bin/env python3
"""
Extract frames from video at specified FPS with timestamp tracking.

The script extracts frames from a video and saves them with filenames that
encode the timestamp, allowing for easy tracing back to the original video time.
"""

import cv2
import os
import argparse
import json
from pathlib import Path


def extract_frames(video_path, output_dir, fps=1.0, video_id=None):
    """
    Extract frames from video at specified FPS.

    Args:
        video_path: Path to input video file
        output_dir: Directory to save extracted frames
        fps: Frames per second to extract (default: 1.0)
        video_id: Optional video ID for naming (extracted from filename if not provided)

    Returns:
        dict: Metadata about extraction including frame count and timestamps
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Extract video_id from filename if not provided
    if video_id is None:
        video_id = Path(video_path).stem

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    # Get video properties
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / video_fps

    print(f"Video: {video_path}")
    print(f"Video FPS: {video_fps:.2f}")
    print(f"Total frames: {total_frames}")
    print(f"Duration: {duration:.2f} seconds")
    print(f"Extracting at {fps} FPS")

    # Calculate frame interval
    frame_interval = int(video_fps / fps)

    # Metadata to store
    metadata = {
        "video_path": str(video_path),
        "video_id": video_id,
        "video_fps": video_fps,
        "extraction_fps": fps,
        "total_video_frames": total_frames,
        "duration_seconds": duration,
        "frames": []
    }

    frame_count = 0
    extracted_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Extract frame at specified interval
        if frame_count % frame_interval == 0:
            # Calculate timestamp
            timestamp = frame_count / video_fps

            # Create filename with timestamp: video_id_frame_XXXXX_ts_XX.XX.jpg
            # Format: frame number (5 digits), timestamp (2 decimal places)
            frame_filename = f"{video_id}_frame_{frame_count:05d}_ts_{timestamp:.2f}.jpg"
            frame_path = os.path.join(output_dir, frame_filename)

            # Save frame
            cv2.imwrite(frame_path, frame)

            # Store metadata
            metadata["frames"].append({
                "filename": frame_filename,
                "frame_number": frame_count,
                "timestamp": timestamp
            })

            extracted_count += 1

            if extracted_count % 10 == 0:
                print(f"Extracted {extracted_count} frames (timestamp: {timestamp:.2f}s)")

        frame_count += 1

    cap.release()

    print(f"\nExtraction complete!")
    print(f"Total extracted: {extracted_count} frames")
    print(f"Saved to: {output_dir}")

    # Save metadata
    metadata_path = os.path.join(output_dir, "frame_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to: {metadata_path}")

    return metadata


def main():
    parser = argparse.ArgumentParser(
        description="Extract frames from video with timestamp tracking"
    )
    parser.add_argument(
        "--video_path",
        required=True,
        help="Path to input video file"
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory to save extracted frames"
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=1.0,
        help="Frames per second to extract (default: 1.0)"
    )
    parser.add_argument(
        "--video_id",
        default=None,
        help="Video ID for naming (default: extracted from filename)"
    )

    args = parser.parse_args()

    extract_frames(
        video_path=args.video_path,
        output_dir=args.output_dir,
        fps=args.fps,
        video_id=args.video_id
    )


if __name__ == "__main__":
    main()

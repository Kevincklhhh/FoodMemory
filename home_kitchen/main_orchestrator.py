#!/usr/bin/env python3
"""
Main Orchestrator
Coordinates the entire food memory tracking pipeline.
"""

import argparse
from pathlib import Path
from typing import List, Tuple
from datetime import datetime

# Import all modules
from food_memory import FoodMemory
from receipt_parser import ReceiptParser
from video_processor import VideoProcessor, VideoClip
from vlm_perception import VLMPerception
from memory_retriever import MemoryRetriever
from vlm_update import VLMUpdate
from memory_updater import MemoryUpdater
from memory_snapshotter import MemorySnapshotter
from vlm_logger import VLMLogger


class MainOrchestrator:
    """
    Main orchestrator that coordinates the entire pipeline.
    """

    def __init__(self, input_dir: str, memory_path: str = "food_memory.json",
                 snapshot_dir: str = "snapshots", clip_dir: str = "clips",
                 log_dir: str = "vlm_logs",
                 vlm_api_url: str = "http://localhost:8000/v1/chat/completions",
                 vlm_model: str = "Qwen3-VL-30B", vlm_fps: int = 2):
        """
        Initialize Main Orchestrator.

        Args:
            input_dir: Directory containing video and receipt files
            memory_path: Path to persistent memory file
            snapshot_dir: Directory for memory snapshots
            clip_dir: Directory for extracted video clips
            log_dir: Directory for VLM logs
            vlm_api_url: VLM API endpoint
            vlm_model: VLM model name
            vlm_fps: Frames per second for VLM video processing
        """
        self.input_dir = Path(input_dir)
        self.clip_dir = Path(clip_dir)

        # Initialize logger
        self.logger = VLMLogger(log_dir)

        # Initialize components
        self.memory = FoodMemory(memory_path)
        self.video_processor = VideoProcessor(clip_duration=30)
        self.vlm_perception = VLMPerception(vlm_api_url, vlm_model, vlm_fps, logger=self.logger)
        self.vlm_update = VLMUpdate(vlm_api_url, vlm_model, vlm_fps, logger=self.logger)
        self.retriever = MemoryRetriever(self.memory)
        self.updater = MemoryUpdater(self.memory, logger=self.logger)
        self.snapshotter = MemorySnapshotter(snapshot_dir)

        print(f"\n{'='*80}")
        print("FOOD MEMORY TRACKER - Main Orchestrator")
        print('='*80)
        print(f"Input directory: {self.input_dir}")
        print(f"Memory file: {memory_path}")
        print(f"Snapshot directory: {snapshot_dir}")
        print(f"VLM Model: {vlm_model} @ {vlm_api_url}")
        print('='*80)

    def find_and_sort_files(self) -> List[Tuple[str, str]]:
        """
        Find all video files and their associated receipts, sorted chronologically.

        Returns:
            List of tuples: (video_path, receipt_path_or_none)
        """
        print("\n[1] Finding and sorting video files...")

        # Find all video files
        video_extensions = ['.mp4', '.MP4', '.mov', '.MOV', '.avi', '.AVI']
        video_files = []

        for ext in video_extensions:
            video_files.extend(self.input_dir.glob(f"*{ext}"))

        # Sort by filename (assumes yyyymmdd-hhmmss format)
        video_files.sort(key=lambda x: x.stem)

        print(f"  Found {len(video_files)} video files")

        # Match with receipts
        files_with_receipts = []
        for video in video_files:
            receipt = ReceiptParser.find_receipt_for_video(str(video))
            files_with_receipts.append((str(video), receipt))

            if receipt:
                print(f"    {video.name} -> {Path(receipt).name}")
            else:
                print(f"    {video.name} -> (no receipt)")

        return files_with_receipts

    def process_video(self, video_path: str, receipt_path: str = None):
        """
        Process a single video through the entire pipeline.

        Args:
            video_path: Path to video file
            receipt_path: Optional path to receipt file
        """
        video_name = Path(video_path).name
        print(f"\n{'='*80}")
        print(f"PROCESSING VIDEO: {video_name}")
        print('='*80)

        # Check if this is an un-bagging video
        is_unbagging = receipt_path is not None
        expected_items = None

        if is_unbagging:
            print(f"[UN-BAGGING MODE] Receipt found: {Path(receipt_path).name}")
            expected_items = ReceiptParser.parse(receipt_path)
            if expected_items:
                print(f"  Expected items: {len(expected_items)}")
                for item in expected_items:
                    print(f"    - {item}")

        # Step 1: Split video into 30-second clips
        print(f"\n[2] Splitting video into clips...")
        clips = self.video_processor.split_into_clips(video_path)

        # Step 2: Process each clip
        for clip_idx, clip in enumerate(clips, 1):
            print(f"\n{'='*80}")
            print(f"CLIP {clip_idx}/{len(clips)}: {clip}")
            print('='*80)

            # Extract clip to file (for VLM processing)
            clip_path = self.video_processor.extract_clip_to_file(clip, str(self.clip_dir))

            # Step 2a: VLM Call 1 - Context-free detection
            detection_result = self.vlm_perception.detect_food_items(
                clip_path,
                video_name=video_name,
                clip_index=clip_idx
            )

            if detection_result is None:
                print("  ⚠ VLM Call 1 failed, skipping clip")
                continue

            detected_names = detection_result.get("food_names", [])

            if not detected_names and not is_unbagging:
                print("  No food items detected, skipping clip")
                continue

            # Step 2b: Retrieve context from memory
            context_block = self.retriever.retrieve_context(detected_names)

            # Step 2c: VLM Call 2 - Context-aware analysis
            commands = self.vlm_update.analyze_interactions(
                clip_path,
                context_block,
                expected_items=expected_items if clip_idx == 1 and is_unbagging else None,
                is_unbagging=(clip_idx == 1 and is_unbagging),
                video_name=video_name,
                clip_index=clip_idx
            )

            if commands is None:
                print("  ⚠ VLM Call 2 failed, skipping clip")
                continue

            if not commands:
                print("  No update commands generated")
                continue

            # Step 2d: Execute commands
            video_timestamp = clip.parent_timestamp
            self.updater.execute_commands(
                commands,
                video_timestamp,
                video_name=video_name,
                clip_index=clip_idx
            )

        # Step 3: Save memory state
        self.memory.save()

        # Step 4: Create snapshot
        video_timestamp = self.video_processor.extract_timestamp_from_filename(video_path)
        self.snapshotter.save_snapshot(self.memory, video_name, video_timestamp)

        # Step 5: Create summary report for this video
        self.logger.create_summary_report(video_name)

        print(f"\n{'='*80}")
        print(f"✓ COMPLETED: {video_name}")
        print('='*80)

    def run(self):
        """
        Run the entire pipeline on all videos.
        """
        # Find and sort files
        files = self.find_and_sort_files()

        if not files:
            print("\n⚠ No video files found in input directory")
            return

        print(f"\n{'='*80}")
        print(f"STARTING PIPELINE: {len(files)} videos to process")
        print('='*80)

        # Process each video
        for idx, (video_path, receipt_path) in enumerate(files, 1):
            print(f"\n\n[VIDEO {idx}/{len(files)}]")
            try:
                self.process_video(video_path, receipt_path)
            except Exception as e:
                print(f"\n✗ ERROR processing {Path(video_path).name}: {e}")
                print("  Continuing to next video...")
                continue

        # Organize logs into timestamped directory
        run_dir = self.logger.save_session_log()

        # Final summary
        print(f"\n\n{'='*80}")
        print("PIPELINE COMPLETE")
        print('='*80)
        print(f"Total videos processed: {len(files)}")
        print(f"Total items in memory: {len(self.memory.items)}")
        print(f"\nMemory saved to: {self.memory.storage_path}")
        print(f"Snapshots saved to: {self.snapshotter.snapshot_dir}")
        print(f"VLM logs organized in: {run_dir}")
        print('='*80)


def main():
    parser = argparse.ArgumentParser(
        description="Food Memory Tracker - Main Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all videos in current directory
  python main_orchestrator.py

  # Process videos in specific directory
  python main_orchestrator.py --input-dir /path/to/videos

  # Use custom VLM endpoint
  python main_orchestrator.py --vlm-url http://192.168.1.100:8000/v1/chat/completions

  # Use different model and sampling rate
  python main_orchestrator.py --vlm-model Qwen3-VL-30B --vlm-fps 4
        """
    )

    parser.add_argument(
        '--input-dir',
        default='.',
        help='Directory containing video and receipt files (default: current directory)'
    )

    parser.add_argument(
        '--memory-file',
        default='food_memory.json',
        help='Path to persistent memory file (default: food_memory.json)'
    )

    parser.add_argument(
        '--snapshot-dir',
        default='snapshots',
        help='Directory for memory snapshots (default: snapshots)'
    )

    parser.add_argument(
        '--clip-dir',
        default='clips',
        help='Directory for extracted video clips (default: clips)'
    )

    parser.add_argument(
        '--log-dir',
        default='vlm_logs',
        help='Directory for VLM logs (default: vlm_logs)'
    )

    parser.add_argument(
        '--vlm-url',
        default='http://localhost:8000/v1/chat/completions',
        help='VLM API endpoint URL (default: http://localhost:8000/v1/chat/completions)'
    )

    parser.add_argument(
        '--vlm-model',
        default='Qwen3-VL-30B',
        help='VLM model name (default: Qwen3-VL-30B)'
    )

    parser.add_argument(
        '--vlm-fps',
        type=int,
        default=2,
        help='Video sampling rate in FPS for VLM (default: 2)'
    )

    args = parser.parse_args()

    # Create and run orchestrator
    orchestrator = MainOrchestrator(
        input_dir=args.input_dir,
        memory_path=args.memory_file,
        snapshot_dir=args.snapshot_dir,
        clip_dir=args.clip_dir,
        log_dir=args.log_dir,
        vlm_api_url=args.vlm_url,
        vlm_model=args.vlm_model,
        vlm_fps=args.vlm_fps
    )

    orchestrator.run()


if __name__ == "__main__":
    main()

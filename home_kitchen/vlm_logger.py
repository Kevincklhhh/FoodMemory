#!/usr/bin/env python3
"""
VLM Logger Module
Documents all VLM runs including prompts, clips, and outputs.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any


class VLMLogger:
    """
    Logs all VLM interactions for documentation and debugging.
    Creates detailed logs for each VLM call including prompts and outputs.
    """

    def __init__(self, log_dir: str = "vlm_logs"):
        """
        Initialize VLM Logger.

        Args:
            log_dir: Directory to store VLM logs
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Running log for current session
        self.session_log = []
        self.session_start_time = datetime.now().isoformat()

    def log_vlm_call_1(self, clip_path: str, prompt: str,
                       detected_food: List[str], raw_response: str,
                       video_name: str, clip_index: int) -> str:
        """
        Log VLM Call 1 (context-free detection).

        Args:
            clip_path: Path to video clip
            prompt: Prompt sent to VLM
            detected_food: List of detected food names
            raw_response: Raw VLM response
            video_name: Parent video name
            clip_index: Clip index

        Returns:
            Empty string (no individual file created)
        """
        log_entry = {
            "vlm_call": "VLM_CALL_1_DETECTION",
            "timestamp": datetime.now().isoformat(),
            "video_name": video_name,
            "clip_index": clip_index,
            "clip_path": clip_path,
            "prompt": prompt,
            "raw_response": raw_response,
            "parsed_output": {
                "food_names": detected_food
            }
        }

        self.session_log.append(log_entry)
        print(f"  [Logger] Logged VLM Call 1 (will be saved in summary)")
        return ""

    def log_vlm_call_2(self, clip_path: str, prompt: str, context_block: str,
                       commands: List[Dict], raw_response: str,
                       video_name: str, clip_index: int,
                       is_unbagging: bool = False,
                       expected_items: Optional[List[str]] = None) -> str:
        """
        Log VLM Call 2 (context-aware update).

        Args:
            clip_path: Path to video clip
            prompt: Prompt sent to VLM
            context_block: Memory context provided
            commands: Generated update commands
            raw_response: Raw VLM response
            video_name: Parent video name
            clip_index: Clip index
            is_unbagging: Whether this was un-bagging mode
            expected_items: Expected items from receipt (if un-bagging)

        Returns:
            Empty string (no individual file created)
        """
        log_entry = {
            "vlm_call": "VLM_CALL_2_UPDATE",
            "timestamp": datetime.now().isoformat(),
            "video_name": video_name,
            "clip_index": clip_index,
            "clip_path": clip_path,
            "mode": "UN-BAGGING" if is_unbagging else "STANDARD_INTERACTION",
            "expected_items": expected_items if is_unbagging else None,
            "context_provided": context_block,
            "prompt": prompt,
            "raw_response": raw_response,
            "parsed_output": {
                "commands": commands,
                "num_creates": sum(1 for c in commands if c.get('command') == 'CREATE'),
                "num_updates": sum(1 for c in commands if c.get('command') == 'UPDATE')
            }
        }

        self.session_log.append(log_entry)
        print(f"  [Logger] Logged VLM Call 2 (will be saved in summary)")
        return ""

    def log_memory_update(self, commands: List[Dict], stats: Dict[str, int],
                         video_name: str, clip_index: int,
                         video_timestamp: str) -> str:
        """
        Log memory update execution results.

        Args:
            commands: Commands that were executed
            stats: Execution statistics (created, updated, failed)
            video_name: Parent video name
            clip_index: Clip index
            video_timestamp: Video timestamp

        Returns:
            Empty string (no individual file created)
        """
        log_entry = {
            "event": "MEMORY_UPDATE",
            "timestamp": datetime.now().isoformat(),
            "video_name": video_name,
            "video_timestamp": video_timestamp,
            "clip_index": clip_index,
            "commands": commands,
            "execution_stats": stats
        }

        self.session_log.append(log_entry)
        print(f"  [Logger] Logged memory update (will be saved in summary)")
        return ""

    def save_session_log(self) -> str:
        """
        Organize all generated log files into a timestamped directory.
        Moves all summary txt files from log_dir into a subdirectory named by run timestamp.

        Returns:
            Path to timestamped log directory
        """
        import shutil

        # Create timestamped directory name
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        run_dir_name = f"run_{timestamp}"
        run_dir_path = self.log_dir / run_dir_name
        run_dir_path.mkdir(parents=True, exist_ok=True)

        # Move all summary txt files into the timestamped directory
        moved_count = 0
        for txt_file in self.log_dir.glob("summary_*.txt"):
            if txt_file.is_file():
                dest_path = run_dir_path / txt_file.name
                shutil.move(str(txt_file), str(dest_path))
                moved_count += 1

        # Calculate session statistics
        total_vlm_calls = len(self.session_log)
        vlm_call_1_count = sum(1 for e in self.session_log if e.get('vlm_call') == 'VLM_CALL_1_DETECTION')
        vlm_call_2_count = sum(1 for e in self.session_log if e.get('vlm_call') == 'VLM_CALL_2_UPDATE')
        memory_update_count = sum(1 for e in self.session_log if e.get('event') == 'MEMORY_UPDATE')

        print(f"\n{'='*80}")
        print(f"[Logger] Session Complete - Logs organized into: {run_dir_name}")
        print('='*80)
        print(f"  Run directory: {run_dir_path}")
        print(f"  Summary files moved: {moved_count}")
        print(f"  Total log entries: {total_vlm_calls}")
        print(f"    - VLM Call 1 (Detection): {vlm_call_1_count}")
        print(f"    - VLM Call 2 (Update): {vlm_call_2_count}")
        print(f"    - Memory Updates: {memory_update_count}")
        print('='*80)

        return str(run_dir_path)

    def create_summary_report(self, video_name: str) -> str:
        """
        Create a comprehensive human-readable summary report for a video.
        Includes all VLM calls, memory updates, prompts, and responses.

        Args:
            video_name: Name of the video to summarize

        Returns:
            Path to summary report
        """
        # Filter logs for this video
        video_logs = [log for log in self.session_log if log.get('video_name') == video_name]

        if not video_logs:
            return ""

        report_lines = [
            "="*80,
            f"VLM INTERACTION REPORT: {video_name}",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "="*80,
            ""
        ]

        # Group by clip
        clips = {}
        for log in video_logs:
            clip_idx = log.get('clip_index', 0)
            if clip_idx not in clips:
                clips[clip_idx] = []
            clips[clip_idx].append(log)

        # Report each clip
        for clip_idx in sorted(clips.keys()):
            clip_logs = clips[clip_idx]

            report_lines.append(f"\n{'='*80}")
            report_lines.append(f"CLIP {clip_idx}")
            report_lines.append('='*80)

            # Find VLM Call 1
            call1 = next((l for l in clip_logs if l.get('vlm_call') == 'VLM_CALL_1_DETECTION'), None)
            if call1:
                report_lines.append("\n" + "-"*80)
                report_lines.append("[VLM CALL 1: Context-Free Detection]")
                report_lines.append("-"*80)
                report_lines.append(f"Clip: {Path(call1['clip_path']).name}")
                report_lines.append(f"Timestamp: {call1['timestamp']}")

                report_lines.append(f"\nPrompt:")
                report_lines.append(call1['prompt'])

                report_lines.append(f"\nRaw VLM Response:")
                report_lines.append(call1['raw_response'])

                report_lines.append(f"\nDetected Food Items ({len(call1['parsed_output']['food_names'])}):")
                if call1['parsed_output']['food_names']:
                    for food in call1['parsed_output']['food_names']:
                        report_lines.append(f"  • {food}")
                else:
                    report_lines.append("  (No food items detected)")

            # Find VLM Call 2
            call2 = next((l for l in clip_logs if l.get('vlm_call') == 'VLM_CALL_2_UPDATE'), None)
            if call2:
                report_lines.append(f"\n" + "-"*80)
                report_lines.append(f"[VLM CALL 2: Context-Aware Update]")
                report_lines.append("-"*80)
                report_lines.append(f"Mode: {call2['mode']}")
                report_lines.append(f"Timestamp: {call2['timestamp']}")

                if call2.get('expected_items'):
                    report_lines.append(f"\nExpected Items from Receipt:")
                    for item in call2['expected_items']:
                        report_lines.append(f"  • {item}")

                report_lines.append(f"\nMemory Context Provided:")
                report_lines.append(call2['context_provided'])

                report_lines.append(f"\nPrompt:")
                report_lines.append(call2['prompt'])

                report_lines.append(f"\nRaw VLM Response:")
                report_lines.append(call2['raw_response'])

                report_lines.append(f"\nGenerated Commands ({len(call2['parsed_output']['commands'])}):")
                if call2['parsed_output']['commands']:
                    for idx, cmd in enumerate(call2['parsed_output']['commands'], 1):
                        if cmd.get('command') == 'CREATE':
                            data = cmd.get('data', {})
                            report_lines.append(f"  {idx}. CREATE")
                            report_lines.append(f"     Label: {data.get('primary_label', 'Unknown')}")
                            report_lines.append(f"     Description: {data.get('description', 'N/A')}")
                            report_lines.append(f"     Location: {data.get('location', 'unknown')}")
                            report_lines.append(f"     Quantity: {data.get('quantity', 'unknown')}")
                            report_lines.append(f"     Action: {data.get('action', 'N/A')}")
                        elif cmd.get('command') == 'UPDATE':
                            data = cmd.get('data', {})
                            report_lines.append(f"  {idx}. UPDATE")
                            report_lines.append(f"     Food ID: {cmd.get('food_id', 'Unknown')}")
                            report_lines.append(f"     Location: {data.get('location', 'no change')}")
                            report_lines.append(f"     Quantity: {data.get('quantity', 'no change')}")
                            report_lines.append(f"     Action: {data.get('action', 'N/A')}")
                else:
                    report_lines.append("  (No commands generated)")

                report_lines.append(f"\nCommand Summary:")
                report_lines.append(f"  Creates: {call2['parsed_output']['num_creates']}")
                report_lines.append(f"  Updates: {call2['parsed_output']['num_updates']}")

            # Find Memory Update
            mem_update = next((l for l in clip_logs if l.get('event') == 'MEMORY_UPDATE'), None)
            if mem_update:
                report_lines.append(f"\n" + "-"*80)
                report_lines.append(f"[Memory Update Execution]")
                report_lines.append("-"*80)
                report_lines.append(f"Timestamp: {mem_update['timestamp']}")
                report_lines.append(f"Video Timestamp: {mem_update['video_timestamp']}")

                stats = mem_update['execution_stats']
                report_lines.append(f"\nExecution Results:")
                report_lines.append(f"  ✓ Created: {stats.get('created', 0)} items")
                report_lines.append(f"  ✓ Updated: {stats.get('updated', 0)} items")
                report_lines.append(f"  ✗ Failed: {stats.get('failed', 0)} items")

        # Add overall summary
        report_lines.append(f"\n{'='*80}")
        report_lines.append("OVERALL SUMMARY")
        report_lines.append('='*80)

        total_call1 = sum(1 for log in video_logs if log.get('vlm_call') == 'VLM_CALL_1_DETECTION')
        total_call2 = sum(1 for log in video_logs if log.get('vlm_call') == 'VLM_CALL_2_UPDATE')
        total_mem_updates = sum(1 for log in video_logs if log.get('event') == 'MEMORY_UPDATE')

        report_lines.append(f"Total Clips Processed: {len(clips)}")
        report_lines.append(f"Total VLM Call 1 (Detection): {total_call1}")
        report_lines.append(f"Total VLM Call 2 (Update): {total_call2}")
        report_lines.append(f"Total Memory Updates: {total_mem_updates}")

        # Sum up all memory updates
        total_created = sum(log['execution_stats'].get('created', 0)
                           for log in video_logs if log.get('event') == 'MEMORY_UPDATE')
        total_updated = sum(log['execution_stats'].get('updated', 0)
                           for log in video_logs if log.get('event') == 'MEMORY_UPDATE')
        total_failed = sum(log['execution_stats'].get('failed', 0)
                          for log in video_logs if log.get('event') == 'MEMORY_UPDATE')

        report_lines.append(f"\nTotal Items Created: {total_created}")
        report_lines.append(f"Total Items Updated: {total_updated}")
        report_lines.append(f"Total Failed Operations: {total_failed}")

        report_lines.append(f"\n{'='*80}\n")

        # Save report
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        report_filename = f"summary_{timestamp}_{video_name.replace('.', '_')}.txt"
        report_path = self.log_dir / report_filename

        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))

        print(f"  [Logger] Created summary report: {report_filename}")
        return str(report_path)


if __name__ == "__main__":
    # Test the VLMLogger module
    print("Testing VLMLogger module...\n")

    logger = VLMLogger("test_vlm_logs")

    # Test VLM Call 1 logging
    logger.log_vlm_call_1(
        clip_path="test_clip_000.mp4",
        prompt="Test prompt for detection",
        detected_food=["milk", "chicken"],
        raw_response='{"food_names": ["milk", "chicken"]}',
        video_name="test_video.MOV",
        clip_index=0
    )

    # Test VLM Call 2 logging
    logger.log_vlm_call_2(
        clip_path="test_clip_000.mp4",
        prompt="Test prompt for update",
        context_block="Test context",
        commands=[{"command": "CREATE", "data": {"primary_label": "Test Milk"}}],
        raw_response='{"commands": [...]}',
        video_name="test_video.MOV",
        clip_index=0,
        is_unbagging=True,
        expected_items=["milk", "eggs"]
    )

    # Test memory update logging
    logger.log_memory_update(
        commands=[{"command": "CREATE"}],
        stats={"created": 1, "updated": 0, "failed": 0},
        video_name="test_video.MOV",
        clip_index=0,
        video_timestamp="2025-10-29T19:37:37"
    )

    # Create summary report
    logger.create_summary_report("test_video.MOV")

    # Save session log
    logger.save_session_log()

    print("\n✓ VLMLogger module test complete")

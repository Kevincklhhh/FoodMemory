#!/usr/bin/env python3
"""
Add grounding data to existing annotation tasks.

This script loads previously created annotation tasks and runs grounding models
to pre-populate masks for food objects.
"""

import json
import sys
from pathlib import Path
from typing import List, Dict
import argparse

# Import grounding utilities
from grounding_utils import run_grounding_on_block


def add_grounding_to_tasks(
    tasks_file: Path,
    output_dir: Path,
    hands23_config: Path,
    hands23_weights: Path
):
    """Add grounding data to annotation tasks

    Args:
        tasks_file: Path to annotation tasks JSON
        output_dir: Base output directory (where assets are stored)
        hands23_config: Path to Hands23 config file
        hands23_weights: Path to Hands23 model weights
    """
    print("=" * 80)
    print("ADDING GROUNDING DATA TO ANNOTATION TASKS")
    print("=" * 80)
    print(f"Tasks file: {tasks_file}")
    print(f"Output directory: {output_dir}")
    print(f"Hands23 config: {hands23_config}")
    print(f"Hands23 weights: {hands23_weights}")
    print()

    # Load tasks
    print("[Step 1] Loading annotation tasks...")
    with open(tasks_file, 'r') as f:
        tasks = json.load(f)
    print(f"✓ Loaded {len(tasks)} annotation tasks")

    # Process each task
    print("\n[Step 2] Running grounding on each task block...")
    total_masks_generated = 0

    for i, task in enumerate(tasks):
        print(f"\nProcessing task {i+1}/{len(tasks)}: {task['task_id']}")
        print(f"  Food nouns: {', '.join(task['target_food_nouns'])}")
        print(f"  Frames: {len(task['assets']['before_frames'])} before, {len(task['assets']['after_frames'])} after")

        # Run grounding
        grounding_data = run_grounding_on_block(
            task,
            output_dir,
            str(hands23_config),
            str(hands23_weights)
        )

        # Update task with grounding data
        task['auto_grounding_data'] = grounding_data

        # Count masks generated
        masks_in_task = 0
        for noun_data in grounding_data.values():
            masks_in_task += len(noun_data['before_frames']) + len(noun_data['after_frames'])

        total_masks_generated += masks_in_task
        print(f"  ✓ Generated {masks_in_task} masks")

    # Save updated tasks
    print(f"\n[Step 3] Saving updated tasks...")
    with open(tasks_file, 'w') as f:
        json.dump(tasks, f, indent=2)

    print(f"✓ Saved updated tasks to {tasks_file}")

    # Print summary
    print("\n" + "=" * 80)
    print("GROUNDING COMPLETE - SUMMARY")
    print("=" * 80)
    print(f"Total tasks processed: {len(tasks)}")
    print(f"Total masks generated: {total_masks_generated}")
    print(f"Average masks per task: {total_masks_generated / len(tasks):.1f}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Add grounding data to annotation tasks"
    )
    parser.add_argument(
        '--tasks-file',
        default=Path('../../outputs/state_change_annotation/P01-20240203-121517_annotation_tasks.json'),
        type=Path,
        help='Path to annotation tasks JSON file'
    )
    parser.add_argument(
        '--output-dir',
        default=Path('../../outputs/state_change_annotation'),
        type=Path,
        help='Base output directory (where assets are stored)'
    )
    parser.add_argument(
        '--hands23-config',
        type=Path,
        default=Path('../../models/hands23_detector/faster_rcnn_X_101_32x8d_FPN_3x_Hands23.yaml'),
        help='Path to Hands23 config file'
    )
    parser.add_argument(
        '--hands23-weights',
        type=Path,
        default=Path('../../models/hands23_detector/model_weights/model_hands23.pth'),
        help='Path to Hands23 model weights'
    )

    args = parser.parse_args()

    # Validate paths
    if not args.tasks_file.exists():
        print(f"Error: Tasks file not found: {args.tasks_file}")
        sys.exit(1)

    if not args.hands23_config.exists():
        print(f"Error: Hands23 config not found: {args.hands23_config}")
        sys.exit(1)

    if not args.hands23_weights.exists():
        print(f"Error: Hands23 weights not found: {args.hands23_weights}")
        sys.exit(1)

    # Run grounding
    add_grounding_to_tasks(
        args.tasks_file,
        args.output_dir,
        args.hands23_config,
        args.hands23_weights
    )


if __name__ == '__main__':
    main()

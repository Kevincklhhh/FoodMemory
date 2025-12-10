#!/usr/bin/env python3
"""
Group All P01 Narrations into 30-Second Blocks and Classify Food-Related Actions

This script processes all narrations from participant P01, groups them into 30-second
temporal blocks, and uses GPT-4o to classify whether each block contains food-related
actions (including interactions with containers that have food inside).

Pipeline:
1. Load all narrations from participant_P01_narrations.csv
2. Group narrations by video_id
3. Within each video, group narrations into 30-second blocks based on timestamps
4. For each block, call GPT-4o to classify food-related actions
5. Store classification results under HDEPIC/outputs/food_classification/

Output Format:
- Per-video files: {video_id}_food_blocks.json
- Aggregated results: all_p01_food_blocks.json
"""

import json
import csv
import sys
from pathlib import Path
from typing import List, Dict
from collections import defaultdict

# Add llm-api to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'llm-api'))
from openai_api import OpenAIAPI  # type: ignore


class NarrationBlock:
    """Represents a 30-second temporal block of narrations"""

    def __init__(self, block_id: int, start_time: float, end_time: float):
        self.block_id = block_id
        self.start_time = start_time
        self.end_time = end_time
        self.narrations = []

    def add_narration(self, narration: Dict):
        """Add a narration to this block"""
        self.narrations.append(narration)

    def get_narration_text(self) -> str:
        """Get concatenated narration text for LLM classification"""
        return "\n".join([
            f"{i+1}. [{n['start_timestamp']:.2f}s - {n['end_timestamp']:.2f}s] {n['narration'].strip()}"
            for i, n in enumerate(self.narrations)
        ])

    def to_dict(self) -> Dict:
        """Convert block to dictionary representation"""
        return {
            'block_id': self.block_id,
            'block_start_time': self.start_time,
            'block_end_time': self.end_time,
            'duration': self.end_time - self.start_time,
            'num_narrations': len(self.narrations),
            'narrations': self.narrations,
            'has_food_action': None,  # To be filled by GPT-4o
        }


def load_all_p01_narrations(csv_path: str) -> Dict[str, List[Dict]]:
    """Load all P01 narrations from CSV and group by video_id

    Args:
        csv_path: Path to participant_P01_narrations.csv

    Returns:
        Dictionary mapping video_id -> list of narration dictionaries
    """
    video_narrations = defaultdict(list)

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        for row in reader:
            video_id = row['video_id']

            # Parse narration data
            narration = {
                'unique_narration_id': row['unique_narration_id'],
                'participant_id': row['participant_id'],
                'video_id': video_id,
                'narration': row['narration'],
                'start_timestamp': float(row['start_timestamp']),
                'end_timestamp': float(row['end_timestamp']),
                'nouns': eval(row['nouns']) if row['nouns'] else [],
                'verbs': eval(row['verbs']) if row['verbs'] else [],
                'hands': eval(row['hands']) if row['hands'] else [],
                'narration_timestamp': float(row['narration_timestamp']),
            }

            video_narrations[video_id].append(narration)

    return dict(video_narrations)


def group_narrations_into_blocks(
    narrations: List[Dict],
    block_duration: float = 30.0
) -> List[NarrationBlock]:
    """Group narrations into temporal blocks of fixed duration

    Strategy:
    - Divide the video timeline into fixed 30-second windows
    - Assign each narration to a block based on its start_timestamp
    - Block boundaries: [0-30s], [30-60s], [60-90s], etc.

    Args:
        narrations: List of narration dictionaries
        block_duration: Block duration in seconds (default: 30.0)

    Returns:
        List of NarrationBlock objects
    """
    if not narrations:
        return []

    # Sort narrations by start timestamp
    sorted_narrations = sorted(narrations, key=lambda x: x['start_timestamp'])

    # Determine video duration
    max_timestamp = max(n['end_timestamp'] for n in sorted_narrations)

    # Create blocks for the entire video timeline
    num_blocks = int(max_timestamp / block_duration) + 1
    blocks = []

    for i in range(num_blocks):
        block_start = i * block_duration
        block_end = (i + 1) * block_duration
        block = NarrationBlock(
            block_id=i,
            start_time=block_start,
            end_time=block_end
        )
        blocks.append(block)

    # Assign narrations to blocks based on start_timestamp
    for narration in sorted_narrations:
        block_idx = int(narration['start_timestamp'] / block_duration)
        if block_idx < len(blocks):
            blocks[block_idx].add_narration(narration)

    # Filter out empty blocks
    non_empty_blocks = [b for b in blocks if len(b.narrations) > 0]

    # Renumber block IDs sequentially
    for i, block in enumerate(non_empty_blocks):
        block.block_id = i

    return non_empty_blocks


def classify_food_block_with_gpt4o(
    block: NarrationBlock,
    api: OpenAIAPI,
    verbose: bool = False
) -> Dict:
    """Use GPT-4o to classify if block contains food-related actions

    Prompt Strategy:
    - Ask GPT-4o to identify if narrations involve food or food containers
    - Include actions like: preparing, cooking, eating, storing, opening/closing containers with food
    - Return structured JSON with classification and reasoning

    Args:
        block: NarrationBlock to classify
        api: OpenAIAPI instance
        verbose: Print debug information

    Returns:
        Classification result dictionary
    """
    narration_text = block.get_narration_text()

    prompt = f"""You are analyzing narrations from an egocentric kitchen video to identify food-related actions.

NARRATIONS (Block {block.block_id}: {block.start_time:.1f}s - {block.end_time:.1f}s):
{narration_text}

TASK:
Classify whether this block contains any food-related actions, including:
1. Direct food interactions: preparing, cooking, cutting, mixing, eating, drinking food
2. Food container interactions: opening/closing containers that contain food (bottles, jars, packages, fridge, cupboards with food items)
3. Food utensil usage: using utensils/appliances for food (knife for cutting food, pan for cooking, etc.)

IMPORTANT:
- Empty containers or containers without food do NOT count as food-related
- General object manipulation (opening empty cupboard, moving non-food items) does NOT count
- Focus on actions where food or food-containing items are directly involved

Respond in JSON format:
{{
    "has_food_action": true/false
}}"""

    messages = [
        {
            "role": "system",
            "content": "You are an expert at analyzing egocentric kitchen video narrations to identify food-related activities."
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    if verbose:
        print(f"\n{'='*80}")
        print(f"Block {block.block_id}: Calling GPT-4o for classification...")
        print(f"{'='*80}")

    try:
        completion = api.chat_completion(messages, max_tokens=500)
        response_text = completion.choices[0].message.content

        if verbose:
            print(f"Response: {response_text}")

        # Parse JSON response
        # Handle potential markdown code blocks
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()

        classification = json.loads(response_text)

        return classification

    except Exception as e:
        print(f"ERROR classifying block {block.block_id}: {e}")
        return {
            "has_food_action": False
        }


def print_video_summary(video_id: str, blocks: List[NarrationBlock]):
    """Print summary statistics for a video"""
    print(f"\n{'='*80}")
    print(f"VIDEO: {video_id}")
    print(f"{'='*80}")
    print(f"Total blocks: {len(blocks)}")
    print(f"Total narrations: {sum(len(b.narrations) for b in blocks)}")

    # Count food blocks
    food_blocks = [b for b in blocks if b.to_dict().get('has_food_action', False)]
    print(f"Food-related blocks: {len(food_blocks)}")

    print(f"\nBlock details:")
    for block in blocks[:5]:  # Show first 5 blocks
        print(f"  Block {block.block_id}: {block.start_time:.1f}s-{block.end_time:.1f}s, "
              f"{len(block.narrations)} narrations")

    if len(blocks) > 5:
        print(f"  ... and {len(blocks) - 5} more blocks")


def main():
    """Main pipeline execution"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Group all P01 narrations into 30-second blocks and classify food-related actions"
    )
    parser.add_argument(
        '--input',
        default='../participant_P01_narrations.csv',
        help='Path to participant_P01_narrations.csv'
    )
    parser.add_argument(
        '--output-dir',
        default='../outputs/food_classification',
        help='Output directory for classification results'
    )
    parser.add_argument(
        '--block-duration',
        type=float,
        default=30.0,
        help='Block duration in seconds (default: 30.0)'
    )
    parser.add_argument(
        '--model',
        default='gpt-4o',
        choices=['gpt-4o', 'o4-mini', 'gpt-4.1-mini'],
        help='GPT model to use for classification'
    )
    parser.add_argument(
        '--video-id',
        default=None,
        help='Process only a specific video ID (for testing)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print verbose output'
    )

    args = parser.parse_args()

    # Setup paths
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("P01 NARRATION FOOD CLASSIFICATION PIPELINE")
    print("="*80)
    print(f"Input: {input_path}")
    print(f"Output: {output_dir}")
    print(f"Block duration: {args.block_duration}s")
    print(f"Model: {args.model}")

    # Initialize GPT-4o API
    print(f"\n[Setup] Initializing {args.model} API...")
    api = OpenAIAPI(deployment=args.model)
    print("✓ API initialized")

    # Step 1: Load all P01 narrations
    print("\n[Step 1] Loading P01 narrations from CSV...")
    video_narrations = load_all_p01_narrations(str(input_path))
    print(f"✓ Loaded narrations for {len(video_narrations)} videos")
    print(f"  Total narrations: {sum(len(narrs) for narrs in video_narrations.values())}")

    # Filter to specific video if requested
    if args.video_id:
        if args.video_id not in video_narrations:
            print(f"ERROR: Video ID '{args.video_id}' not found in data")
            return
        video_narrations = {args.video_id: video_narrations[args.video_id]}
        print(f"  Filtering to video: {args.video_id}")

    # Step 2: Process each video
    all_results = {}

    for video_idx, (video_id, narrations) in enumerate(video_narrations.items(), 1):
        print(f"\n{'='*80}")
        print(f"[Step 2.{video_idx}] Processing video: {video_id}")
        print(f"{'='*80}")

        # Group narrations into blocks
        print(f"  Grouping {len(narrations)} narrations into {args.block_duration}s blocks...")
        blocks = group_narrations_into_blocks(narrations, args.block_duration)
        print(f"  ✓ Created {len(blocks)} non-empty blocks")

        # Classify each block with GPT-4o
        print(f"  Classifying blocks with {args.model}...")
        video_results = []

        for block in blocks:
            if args.verbose:
                print(f"\n  Processing block {block.block_id}/{len(blocks)}...")

            classification = classify_food_block_with_gpt4o(block, api, args.verbose)

            # Create result dictionary
            block_dict = block.to_dict()
            block_dict['has_food_action'] = classification['has_food_action']

            video_results.append(block_dict)

            # Print progress
            if not args.verbose and (block.block_id + 1) % 10 == 0:
                food_count = sum(1 for r in video_results if r['has_food_action'])
                print(f"    Progress: {block.block_id + 1}/{len(blocks)} blocks "
                      f"({food_count} food-related)")

        # Summary for this video
        food_blocks = [r for r in video_results if r['has_food_action']]
        print(f"\n  ✓ Completed classification for {video_id}")
        print(f"    Total blocks: {len(video_results)}")
        print(f"    Food-related blocks: {len(food_blocks)} ({len(food_blocks)/len(video_results)*100:.1f}%)")

        # Save per-video results
        video_output_file = output_dir / f"{video_id}_food_blocks.json"
        with open(video_output_file, 'w') as f:
            json.dump({
                'video_id': video_id,
                'block_duration': args.block_duration,
                'total_blocks': len(video_results),
                'food_blocks_count': len(food_blocks),
                'blocks': video_results
            }, f, indent=2)
        print(f"    Saved to: {video_output_file}")

        all_results[video_id] = {
            'total_blocks': len(video_results),
            'food_blocks_count': len(food_blocks),
            'blocks': video_results
        }

    # Step 3: Save aggregated results
    print(f"\n{'='*80}")
    print("[Step 3] Saving aggregated results...")
    print(f"{'='*80}")

    aggregated_output = output_dir / "all_p01_food_blocks.json"
    with open(aggregated_output, 'w') as f:
        json.dump({
            'participant_id': 'P01',
            'block_duration': args.block_duration,
            'model_used': args.model,
            'total_videos': len(all_results),
            'videos': all_results
        }, f, indent=2)

    print(f"✓ Saved aggregated results to: {aggregated_output}")

    # Final summary
    total_blocks = sum(v['total_blocks'] for v in all_results.values())
    total_food_blocks = sum(v['food_blocks_count'] for v in all_results.values())

    print(f"\n{'='*80}")
    print("PIPELINE COMPLETE - SUMMARY")
    print(f"{'='*80}")
    print(f"Total videos processed: {len(all_results)}")
    print(f"Total blocks created: {total_blocks}")
    print(f"Total food-related blocks: {total_food_blocks} ({total_food_blocks/total_blocks*100:.1f}%)")
    print(f"\nResults saved to: {output_dir}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()

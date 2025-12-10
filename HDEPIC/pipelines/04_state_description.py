#!/usr/bin/env python3
"""
Test State Description Generation

This script tests the intermediate "State Description" step that converts raw narrations
into natural language descriptions of food state changes.

The VLM acts as a semantic interpreter:
- Filters non-food actions (SKIP)
- Verifies actions against video evidence (CONTEXT)
- Infers state changes in natural language (STATE)

Usage:
    python test_state_description.py --video-id P01-20240202-110250
    python test_state_description.py --video-id P01-20240202-110250 --block-id 5
    python test_state_description.py --video-id P01-20240202-110250 --verbose
"""

import json
import sys
import argparse
import requests
import base64
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / 'food_graph'))

from food_graph.vlm_prompts import (
    STATE_DESCRIPTION_PROMPT,
    build_state_description_prompt,
    parse_state_description_response,
)

# Qwen3-VL API endpoint
QWEN3VL_URL = "http://saltyfish.eecs.umich.edu:8000/v1/chat/completions"
QWEN_MODEL = "Qwen/Qwen3-VL-30B-A3B-Instruct"


class VLMClient:
    """Handles communication with Qwen VLM API"""

    def __init__(self):
        print(f"[VLMClient] Using Qwen3-VL at {QWEN3VL_URL}")

    def encode_video_base64(self, video_path: Path) -> str:
        """Encode video file to base64"""
        with open(video_path, "rb") as f:
            return base64.b64encode(f.read()).decode()

    def query(
        self,
        system_prompt: str,
        user_prompt: str,
        video_path: Optional[Path] = None,
        max_tokens: int = 3000,
        temperature: float = 0.3
    ) -> str:
        """Query Qwen3-VL with optional video"""
        messages = [{"role": "system", "content": system_prompt}]

        user_content = []
        if video_path and video_path.exists():
            video_base64 = self.encode_video_base64(video_path)
            user_content.append({
                "type": "video_url",
                "video_url": {"url": f"data:video/mp4;base64,{video_base64}"}
            })

        user_content.append({"type": "text", "text": user_prompt})
        messages.append({"role": "user", "content": user_content})

        data = {
            "model": QWEN_MODEL,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        if video_path and video_path.exists():
            data["extra_body"] = {
                "mm_processor_kwargs": {
                    "fps": 1,
                    "do_sample_frames": True
                }
            }

        headers = {"Content-Type": "application/json"}

        try:
            response = requests.post(QWEN3VL_URL, headers=headers, json=data, timeout=180)
            response.raise_for_status()
            result = response.json()
            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["message"]["content"]
            return ""
        except Exception as e:
            print(f"  ERROR: Qwen API Error: {e}")
            return ""


def load_food_blocks(video_id: str, blocks_dir: Path) -> List[Dict]:
    """Load food blocks for a video"""
    blocks_file = blocks_dir / f"{video_id}_food_blocks.json"
    if not blocks_file.exists():
        print(f"ERROR: Blocks file not found: {blocks_file}")
        return []

    with open(blocks_file, 'r') as f:
        data = json.load(f)

    # Filter to food blocks only
    blocks = [b for b in data.get('blocks', []) if b.get('has_food_action', False)]
    blocks.sort(key=lambda x: x['block_id'])
    return blocks


def get_video_clip_path(video_id: str, block_id: int, clips_dir: Path) -> Optional[Path]:
    """Get path to video clip for a block"""
    clip_path = clips_dir / video_id / f"block_{block_id:03d}.mp4"
    if clip_path.exists():
        return clip_path
    return None


def format_text_output(video_id: str, block: Dict, descriptions: List[Dict]) -> str:
    """Format descriptions as human-readable text"""
    lines = [
        f"=== BLOCK {block['block_id']} ({block['block_start_time']:.1f}s - {block['block_end_time']:.1f}s) ===",
        ""
    ]

    for desc in descriptions:
        ts = desc.get('timestamp', 0)
        narration = desc.get('original_narration', '')
        lines.append(f"[{ts:.2f}s] \"{narration}\"")

        if desc.get('skip'):
            reason = desc.get('skip_reason', 'Non-food action')
            lines.append(f"-> SKIP ({reason})")
        else:
            context = desc.get('context')
            state = desc.get('state_description')
            if context:
                lines.append(f"-> CONTEXT: {context}")
            if state:
                lines.append(f"-> STATE: {state}")

        lines.append("")

    return "\n".join(lines)


def process_block(
    vlm_client: VLMClient,
    video_id: str,
    block: Dict,
    clips_dir: Path,
    verbose: bool = False
) -> Dict:
    """Process a single block and generate state descriptions"""
    block_id = block['block_id']
    narrations = block.get('narrations', [])

    if verbose:
        print(f"\n  Block {block_id}: {len(narrations)} narrations")

    # Get video clip
    video_path = get_video_clip_path(video_id, block_id, clips_dir)
    if video_path:
        if verbose:
            print(f"    Video: {video_path.name}")
    else:
        if verbose:
            print(f"    Video: (not found)")

    # Build prompt
    user_prompt = build_state_description_prompt(narrations, block['block_start_time'])

    if verbose:
        print(f"    Querying VLM...")

    # Query VLM
    response = vlm_client.query(
        STATE_DESCRIPTION_PROMPT,
        user_prompt,
        video_path=video_path,
        max_tokens=3000
    )

    # Parse response
    descriptions = parse_state_description_response(response, narrations)

    # Count stats
    skip_count = sum(1 for d in descriptions if d.get('skip'))
    state_count = sum(1 for d in descriptions if d.get('state_description'))

    if verbose:
        print(f"    Results: {skip_count} skipped, {state_count} with state descriptions")

    return {
        'video_id': video_id,
        'block_id': block_id,
        'block_start_time': block['block_start_time'],
        'block_end_time': block['block_end_time'],
        'video_clip': str(video_path) if video_path else None,
        'raw_response': response,
        'descriptions': descriptions,
        'stats': {
            'total_narrations': len(narrations),
            'skipped': skip_count,
            'with_state': state_count
        }
    }


def main():
    parser = argparse.ArgumentParser(
        description="Test State Description Generation with VLM"
    )
    parser.add_argument(
        '--video-id',
        required=True,
        help='Video ID to process (e.g., P01-20240202-110250)'
    )
    parser.add_argument(
        '--block-id',
        type=int,
        default=None,
        help='Process only a specific block ID'
    )
    parser.add_argument(
        '--blocks-dir',
        default='../outputs/food_classification',
        help='Directory containing food_blocks.json files'
    )
    parser.add_argument(
        '--clips-dir',
        default='../outputs/food_clips',
        help='Directory containing video clips'
    )
    parser.add_argument(
        '--output-dir',
        default='../outputs/food_classification/state_descriptions',
        help='Output directory for state descriptions'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print verbose output'
    )

    args = parser.parse_args()

    # Setup paths
    blocks_dir = Path(args.blocks_dir)
    clips_dir = Path(args.clips_dir)
    output_dir = Path(args.output_dir) / args.video_id
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("STATE DESCRIPTION GENERATION TEST")
    print("=" * 70)
    print(f"Video: {args.video_id}")
    print(f"Blocks Dir: {blocks_dir}")
    print(f"Clips Dir: {clips_dir}")
    print(f"Output Dir: {output_dir}")

    # Initialize VLM client
    print("\n[Setup] Initializing VLM client...")
    vlm_client = VLMClient()

    # Load food blocks
    print(f"\n[Step 1] Loading food blocks for {args.video_id}...")
    blocks = load_food_blocks(args.video_id, blocks_dir)

    if not blocks:
        print("ERROR: No food blocks found")
        return

    # Filter to specific block if requested
    if args.block_id is not None:
        blocks = [b for b in blocks if b['block_id'] == args.block_id]
        if not blocks:
            print(f"ERROR: Block {args.block_id} not found")
            return

    print(f"Found {len(blocks)} food blocks to process")

    # Check video clips
    clips_available = sum(
        1 for b in blocks
        if get_video_clip_path(args.video_id, b['block_id'], clips_dir)
    )
    print(f"Video clips available: {clips_available}/{len(blocks)}")

    # Process blocks
    print(f"\n[Step 2] Processing blocks...")
    all_results = []
    total_stats = {'total_narrations': 0, 'skipped': 0, 'with_state': 0}

    for i, block in enumerate(blocks, 1):
        block_id = block['block_id']
        print(f"\n  [{i}/{len(blocks)}] Block {block_id}...")

        result = process_block(
            vlm_client, args.video_id, block, clips_dir, args.verbose
        )
        all_results.append(result)

        # Aggregate stats
        total_stats['total_narrations'] += result['stats']['total_narrations']
        total_stats['skipped'] += result['stats']['skipped']
        total_stats['with_state'] += result['stats']['with_state']

        # Save per-block outputs
        json_file = output_dir / f"block_{block_id:03d}_descriptions.json"
        txt_file = output_dir / f"block_{block_id:03d}_descriptions.txt"

        with open(json_file, 'w') as f:
            json.dump(result, f, indent=2)

        text_output = format_text_output(args.video_id, block, result['descriptions'])
        with open(txt_file, 'w') as f:
            f.write(text_output)

        print(f"    Saved: {json_file.name}, {txt_file.name}")

    # Save summary
    print(f"\n[Step 3] Saving summary...")
    summary = {
        'video_id': args.video_id,
        'processed_at': datetime.now().isoformat(),
        'blocks_processed': len(all_results),
        'total_stats': total_stats,
        'blocks': [
            {
                'block_id': r['block_id'],
                'stats': r['stats']
            }
            for r in all_results
        ]
    }

    summary_file = output_dir / 'summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved: {summary_file}")

    # Print final summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print("=" * 70)
    print(f"Blocks processed: {len(all_results)}")
    print(f"Total narrations: {total_stats['total_narrations']}")
    print(f"  - Skipped (non-food): {total_stats['skipped']}")
    print(f"  - With state descriptions: {total_stats['with_state']}")
    print(f"\nOutputs saved to: {output_dir}")
    print("=" * 70)


if __name__ == '__main__':
    main()
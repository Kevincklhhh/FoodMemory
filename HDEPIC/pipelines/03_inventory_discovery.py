#!/usr/bin/env python3
"""
Inventory Discovery (Pass 1) - Identify First Appearances of Food Items

This script processes narration blocks chronologically to identify NEW food items
that enter the tracking scope for the first time. It maintains an accumulated
inventory and uses GPT-4o to semantically filter out:
- Items already in inventory (including synonyms)
- State changes of existing items (slice from bread)
- Units from collective containers (egg from carton)

Pipeline:
1. Load food_blocks.json for each video
2. Filter to has_food_action: true blocks
3. Process blocks chronologically with accumulated inventory
4. Call GPT-4o per block to identify NEW food arrivals
5. Save per-video arrival events + aggregated summary

Output: HDEPIC/outputs/inventory_discovery/{video_id}_arrivals.json

NOTE: Inventory persists across videos processed in chronological order.
Items discovered in earlier videos won't be re-discovered in later ones.
"""

import json
import sys
import requests
from pathlib import Path
from typing import List, Dict, Set, Tuple
from dataclasses import dataclass, asdict

# Add llm-api to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'llm-api'))
try:
    from openai_api import OpenAIAPI  # type: ignore
except ImportError:
    OpenAIAPI = None

# Qwen3-VL API endpoint
QWEN3VL_URL = "http://saltyfish.eecs.umich.edu:8000/v1/chat/completions"
QWEN_MODEL = "Qwen/Qwen3-VL-30B-A3B-Instruct"

# Default paths (relative to this script's location in pipelines/)
_SCRIPT_DIR = Path(__file__).parent
_PROJECT_ROOT = _SCRIPT_DIR.parent


class VLMClient:
    """Handles communication with VLM APIs (Qwen and GPT-4o) for text-only queries"""

    def __init__(self, model_name: str = 'qwen'):
        self.model_name = model_name
        self.openai_api = None

        if model_name == 'gpt-4o':
            if OpenAIAPI is None:
                raise ImportError("OpenAIAPI not found. Cannot use gpt-4o.")
            print(f"[VLMClient] Initializing GPT-4o API...")
            self.openai_api = OpenAIAPI(deployment='gpt-4o')
        else:
            print(f"[VLMClient] Using Qwen3-VL at {QWEN3VL_URL}")

    def query(self, system_prompt: str, user_prompt: str, max_tokens: int = 500) -> str:
        """Send text-only query to VLM"""
        if self.model_name == 'gpt-4o':
            return self._query_openai(system_prompt, user_prompt, max_tokens)
        else:
            return self._query_qwen(system_prompt, user_prompt, max_tokens)

    def _query_openai(self, system_prompt: str, user_prompt: str, max_tokens: int) -> str:
        """Query GPT-4o"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        try:
            completion = self.openai_api.chat_completion(messages, max_tokens=max_tokens)
            return completion.choices[0].message.content
        except Exception as e:
            print(f"  ✗ OpenAI API Error: {e}")
            return ""

    def _query_qwen(self, system_prompt: str, user_prompt: str, max_tokens: int) -> str:
        """Query Qwen3-VL (text-only)"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [{"type": "text", "text": user_prompt}]}
        ]

        data = {
            "model": QWEN_MODEL,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.3,
        }

        headers = {"Content-Type": "application/json"}

        try:
            response = requests.post(QWEN3VL_URL, headers=headers, json=data, timeout=60)
            response.raise_for_status()
            result = response.json()
            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["message"]["content"]
            return ""
        except Exception as e:
            print(f"  ✗ Qwen API Error: {e}")
            return ""


@dataclass
class NewArrivalEvent:
    """Represents a new food item entering the scene"""
    timestamp: float          # Time of the block where it appears
    semantic_name: str        # Canonical name (e.g., "butter", "chicken")
    trigger_text: str         # The narration text that triggered detection
    detected_form: str        # How it appeared (e.g., "in plastic container")
    block_id: int             # Block where it was detected
    video_id: str = ""        # Video where it was discovered


def format_block_narrations(block: Dict) -> str:
    """Format all narrations in a block for the LLM prompt"""
    lines = []
    for i, narration in enumerate(block['narrations'], 1):
        text = narration['narration'].strip()
        start = narration['start_timestamp']
        end = narration['end_timestamp']
        lines.append(f"{i}. [{start:.2f}s - {end:.2f}s] {text}")
    return "\n".join(lines)


def format_inventory(inventory: Set[str]) -> str:
    """Format current inventory for the LLM prompt"""
    if not inventory:
        return "Empty - no food items tracked yet"
    return ", ".join(sorted(inventory))


def classify_block_arrivals(
    block: Dict,
    inventory: Set[str],
    vlm_client: VLMClient,
    verbose: bool = False
) -> List[Dict]:
    """Use VLM to identify NEW food arrivals in a block

    Args:
        block: Block dictionary with narrations
        inventory: Current accumulated food inventory
        vlm_client: VLMClient instance (Qwen or GPT-4o)
        verbose: Print debug output

    Returns:
        List of new arrival dictionaries with semantic_name and detected_form
    """
    block_text = format_block_narrations(block)
    inventory_str = format_inventory(inventory)

    system_prompt = """You are an "Ingredient Spotter" for a cooking dataset. Your job is to identify NEW food items that first appear in the narrations.

IMPORTANT - PRESERVE EXACT NAMES:
- Use the EXACT name as it appears in the narration (e.g., "mesh of oranges" NOT "orange")
- Include containers/packaging as part of the name (e.g., "milk bottle", "carton of eggs", "bag of flour")
- This is critical because the downstream VLM needs to track splits (oranges from mesh) vs state changes

DECISION RULES:
1. IGNORE items already in Current Inventory (including synonyms)
2. IGNORE state changes of existing items (slice from bread already in inventory)
3. IGNORE units extracted from collective containers already in inventory (egg from "carton of eggs")
4. IGNORE non-food items (utensils, appliances, surfaces)
5. ACCEPT only genuinely NEW food items entering the scene from fridge, pantry, or off-camera

EXAMPLES:
- Inventory has "loaf of bread" → "cut a slice" → IGNORE (slice comes from loaf)
- Inventory has "carton of eggs" → "crack an egg" → IGNORE (egg from carton)
- Inventory empty → "get the mesh of oranges" → ACCEPT "mesh of oranges" (NOT just "oranges")
- Inventory empty → "grab the milk bottle" → ACCEPT "milk bottle" (NOT just "milk")
- Inventory has "milk bottle" → "pour the milk" → IGNORE (already tracked)
- Inventory empty → "pick up the bowl" → IGNORE (bowl is not food)"""

    user_prompt = f"""**CURRENT FOOD INVENTORY:**
{inventory_str}

**NARRATIONS (Block {block['block_id']}: {block['block_start_time']:.1f}s - {block['block_end_time']:.1f}s):**
{block_text}

**TASK:**
Identify NEW food items that appear for the first time in these narrations.
IMPORTANT: Preserve the EXACT name as mentioned (e.g., "mesh of oranges", "milk bottle", "bag of flour").
Return a JSON array. If no new food items, return empty array [].

Format: [{{"semantic_name": "exact_food_name_from_narration", "detected_form": "brief_description"}}]

Examples:
- [{{"semantic_name": "mesh of oranges", "detected_form": "from fridge"}}]
- [{{"semantic_name": "milk bottle", "detected_form": "from fridge door"}}]
- [{{"semantic_name": "bag of flour", "detected_form": "from pantry"}}]
- []"""

    if verbose:
        print(f"\n{'='*60}")
        print(f"Block {block['block_id']}: Classifying arrivals...")
        print(f"Current inventory: {inventory_str}")
        print(f"{'='*60}")

    try:
        response_text = vlm_client.query(system_prompt, user_prompt, max_tokens=500)

        if verbose:
            print(f"Response: {response_text}")

        # Parse JSON response (handle markdown code blocks)
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()

        arrivals = json.loads(response_text)

        # Validate response structure
        if not isinstance(arrivals, list):
            arrivals = []

        return arrivals

    except Exception as e:
        print(f"ERROR classifying block {block['block_id']}: {e}")
        return []


def process_video(
    food_blocks_path: Path,
    vlm_client: VLMClient,
    initial_inventory: Set[str] = None,
    verbose: bool = False
) -> Tuple[Dict, Set[str]]:
    """Process a single video's food blocks to discover inventory arrivals

    Args:
        food_blocks_path: Path to {video_id}_food_blocks.json
        vlm_client: VLMClient instance (Qwen or GPT-4o)
        initial_inventory: Inventory carried over from previous videos
        verbose: Print debug output

    Returns:
        Tuple of (result dict, updated inventory set)
    """
    # Load food blocks
    with open(food_blocks_path, 'r') as f:
        data = json.load(f)

    video_id = data['video_id']
    blocks = data['blocks']

    # Filter to food blocks only and sort by block_id
    food_blocks = [b for b in blocks if b.get('has_food_action', False)]
    food_blocks = sorted(food_blocks, key=lambda x: x['block_id'])

    # Start with inherited inventory or empty set
    global_inventory: Set[str] = set(initial_inventory) if initial_inventory else set()
    initial_size = len(global_inventory)

    print(f"\nProcessing {video_id}: {len(food_blocks)} food blocks "
          f"(starting inventory: {initial_size} items)")

    arrivals: List[NewArrivalEvent] = []

    for block in food_blocks:
        # Classify new arrivals in this block
        new_items = classify_block_arrivals(block, global_inventory, vlm_client, verbose)

        # Create arrival events and update inventory
        for item in new_items:
            semantic_name = item.get('semantic_name', '').lower().strip()
            detected_form = item.get('detected_form', '')

            if not semantic_name:
                continue

            # Skip if somehow already in inventory (LLM should have filtered)
            if semantic_name in global_inventory:
                continue

            # Find the first narration mentioning this item for trigger_text
            trigger_text = ""
            trigger_timestamp = block['block_start_time']
            for narration in block['narrations']:
                narr_text = narration['narration'].lower()
                if semantic_name in narr_text or any(
                    word in narr_text for word in semantic_name.split()
                ):
                    trigger_text = narration['narration'].strip()
                    trigger_timestamp = narration['start_timestamp']
                    break

            if not trigger_text:
                # Fallback to first narration in block
                trigger_text = block['narrations'][0]['narration'].strip()

            event = NewArrivalEvent(
                timestamp=trigger_timestamp,
                semantic_name=semantic_name,
                trigger_text=trigger_text,
                detected_form=detected_form,
                block_id=block['block_id'],
                video_id=video_id
            )
            arrivals.append(event)
            global_inventory.add(semantic_name)

            if verbose:
                print(f"  NEW: {semantic_name} ({detected_form}) @ {trigger_timestamp:.2f}s")

        # Progress update
        if not verbose and (block['block_id'] + 1) % 10 == 0:
            print(f"  Progress: {block['block_id'] + 1}/{len(food_blocks)} blocks, "
                  f"{len(arrivals)} arrivals")

    print(f"  Completed: {len(arrivals)} new arrivals in this video, "
          f"cumulative inventory: {len(global_inventory)} items")

    result = {
        'video_id': video_id,
        'total_blocks_processed': len(food_blocks),
        'total_arrivals': len(arrivals),
        'inherited_inventory_size': initial_size,
        'arrivals': [asdict(a) for a in arrivals],
        'video_inventory': sorted([a.semantic_name for a in arrivals]),  # Items discovered in THIS video
        'cumulative_inventory': sorted(list(global_inventory))  # All items up to this video
    }

    return result, global_inventory


def get_video_range(input_dir: Path, start_video: str = None, end_video: str = None) -> List[Path]:
    """Get list of video files within the specified range (inclusive).

    Videos are sorted chronologically by ID (format: P01-YYYYMMDD-HHMMSS).
    """
    all_files = sorted(input_dir.glob("P01-*_food_blocks.json"))

    if not start_video and not end_video:
        return all_files

    # Find start and end indices
    video_ids = [f.stem.replace("_food_blocks", "") for f in all_files]

    start_idx = 0
    end_idx = len(all_files)

    if start_video:
        if start_video in video_ids:
            start_idx = video_ids.index(start_video)
        else:
            print(f"WARNING: Start video {start_video} not found, starting from beginning")

    if end_video:
        if end_video in video_ids:
            end_idx = video_ids.index(end_video) + 1  # inclusive
        else:
            print(f"WARNING: End video {end_video} not found, processing to end")

    return all_files[start_idx:end_idx]


def main():
    """Main pipeline execution"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Inventory Discovery - Identify first appearances of food items"
    )
    parser.add_argument(
        '--input-dir',
        default=str(_PROJECT_ROOT / "outputs" / "food_classification"),
        help='Directory containing food_blocks.json files'
    )
    parser.add_argument(
        '--output-dir',
        default=str(_PROJECT_ROOT / "outputs" / "inventory_discovery_global"),
        help='Output directory for arrival results (ignored if --local is set)'
    )
    parser.add_argument(
        '--local',
        action='store_true',
        help='Local mode: output to inventory_discovery_local/{start}_to_{end}/'
    )
    parser.add_argument(
        '--start-video',
        default=None,
        help='Start video ID for range (e.g., P01-20240202-110250)'
    )
    parser.add_argument(
        '--end-video',
        default=None,
        help='End video ID for range (e.g., P01-20240203-121517)'
    )
    parser.add_argument(
        '--model',
        default='qwen',
        choices=['qwen', 'gpt-4o'],
        help='VLM model to use (default: qwen)'
    )
    parser.add_argument(
        '--video-id',
        default=None,
        help='Process only a specific video ID (shorthand for --start-video X --end-video X)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print verbose output'
    )

    args = parser.parse_args()

    # Handle --video-id as shorthand for single video range
    if args.video_id:
        args.start_video = args.video_id
        args.end_video = args.video_id

    # Setup paths
    input_dir = Path(args.input_dir)

    # Determine output directory
    if args.local:
        if not args.start_video or not args.end_video:
            print("ERROR: --local requires both --start-video and --end-video")
            return
        # Create local output directory named after video range
        range_name = f"{args.start_video}_to_{args.end_video}"
        output_dir = _PROJECT_ROOT / "outputs" / "inventory_discovery_local" / range_name
    else:
        output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("INVENTORY DISCOVERY (Pass 1)")
    print("="*70)
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Mode: {'LOCAL' if args.local else 'GLOBAL'}")
    if args.start_video or args.end_video:
        print(f"Range: {args.start_video or '(start)'} -> {args.end_video or '(end)'}")
    print(f"Model: {args.model}")

    # Initialize VLM client
    print(f"\n[Setup] Initializing {args.model}...")
    vlm_client = VLMClient(model_name=args.model)
    print("VLM client initialized")

    # Find video files in range
    food_blocks_files = get_video_range(input_dir, args.start_video, args.end_video)

    if not food_blocks_files:
        print("ERROR: No video files found in specified range")
        return

    print(f"\n[Step 1] Found {len(food_blocks_files)} video files to process")
    print("Processing in chronological order with PERSISTENT inventory")

    # Process each video with persistent inventory
    all_results = {}
    total_arrivals = 0
    persistent_inventory: Set[str] = set()  # Carries across all videos

    for i, food_blocks_path in enumerate(food_blocks_files, 1):
        print(f"\n{'='*70}")
        print(f"[Step 2.{i}] Processing: {food_blocks_path.stem}")
        print("="*70)

        # Pass current inventory to video processing
        result, persistent_inventory = process_video(
            food_blocks_path, vlm_client, persistent_inventory, args.verbose
        )

        # Save per-video results
        video_id = result['video_id']
        output_file = output_dir / f"{video_id}_arrivals.json"
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"  Saved to: {output_file}")

        all_results[video_id] = result
        total_arrivals += result['total_arrivals']

    # Save aggregated results
    print(f"\n{'='*70}")
    print("[Step 3] Saving aggregated results...")
    print("="*70)

    aggregated = {
        'participant_id': 'P01',
        'model_used': args.model,
        'total_videos': len(all_results),
        'total_arrivals': total_arrivals,
        'final_inventory': sorted(list(persistent_inventory)),
        'final_inventory_size': len(persistent_inventory),
        'videos': all_results
    }

    aggregated_file = output_dir / "all_p01_arrivals.json"
    with open(aggregated_file, 'w') as f:
        json.dump(aggregated, f, indent=2)
    print(f"Saved aggregated results to: {aggregated_file}")

    # Final summary
    print(f"\n{'='*70}")
    print("PIPELINE COMPLETE - SUMMARY")
    print("="*70)
    print(f"Videos processed: {len(all_results)}")
    print(f"Total unique food arrivals: {total_arrivals}")
    print(f"Final inventory size: {len(persistent_inventory)} unique items")

    # Show top videos by arrivals
    sorted_videos = sorted(
        all_results.items(),
        key=lambda x: x[1]['total_arrivals'],
        reverse=True
    )[:5]
    print(f"\nTop 5 videos by new arrivals:")
    for vid, res in sorted_videos:
        print(f"  {vid}: {res['total_arrivals']} arrivals")

    # Show final inventory
    print(f"\nFinal cumulative inventory ({len(persistent_inventory)} items):")
    inv_list = sorted(list(persistent_inventory))
    for i in range(0, len(inv_list), 8):
        print(f"  {', '.join(inv_list[i:i+8])}")

    print(f"\nResults saved to: {output_dir}")
    print("="*70)


if __name__ == '__main__':
    main()

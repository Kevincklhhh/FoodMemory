#!/usr/bin/env python3
"""
Inventory Discovery (Semantic Groups) - Identify First Appearances of Food Items

This script processes semantic groups chronologically to identify NEW food items
that enter the tracking scope for the first time. It maintains an accumulated
inventory and uses VLM to semantically filter out:
- Items already in inventory (including synonyms)
- State changes of existing items (slice from bread)
- Units from collective containers (egg from carton)

Differences from block-based version:
- Loads semantic groups from narration_grouping/*.json
- Processes ALL groups (no has_food_action filter)
- Handles timestamp mapping for padded video clips

Pipeline:
1. Load semantic groups for each video via semantic_utils
2. Load clip manifest for timestamp mapping
3. Process groups chronologically with accumulated inventory
4. Call VLM per group to identify NEW food arrivals
5. Save per-video arrival events + aggregated summary

Output: HDEPIC/outputs/inventory_discovery/{video_id}_arrivals.json
"""

import json
import sys
import requests
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass, asdict

from semantic_utils import (
    SemanticGroup,
    find_videos_with_groupings,
    load_and_enrich_video_groups,
)

# Add llm-api to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'llm-api'))
try:
    from openai_api import OpenAIAPI  # type: ignore
except ImportError:
    OpenAIAPI = None

# Qwen3-VL API endpoint
QWEN3VL_URL = "http://saltyfish.eecs.umich.edu:8000/v1/chat/completions"
QWEN_MODEL = "Qwen/Qwen3-VL-30B-A3B-Instruct"

# Default paths
DEFAULT_GROUPING_DIR = Path("/home/kailaic/NeuroTrace/kitchen/HDEPIC/narration_grouping")
DEFAULT_CSV_PATH = Path("/home/kailaic/NeuroTrace/kitchen/HDEPIC/participant_P01_narrations.csv")
DEFAULT_CLIPS_DIR = Path("/home/kailaic/NeuroTrace/kitchen/HDEPIC/outputs/food_clips")
DEFAULT_OUTPUT_DIR = Path("/home/kailaic/NeuroTrace/kitchen/HDEPIC/outputs/inventory_discovery")

# Global variable for log directory (set in main)
VLM_LOG_DIR: Optional[Path] = None


def log_vlm_call(video_id: str, group_id: int, system_prompt: str, user_prompt: str, response: str):
    """Log VLM input and output to files."""
    if VLM_LOG_DIR is None:
        return

    log_dir = VLM_LOG_DIR / video_id
    log_dir.mkdir(parents=True, exist_ok=True)

    # Log input (system + user prompt)
    input_file = log_dir / f"group_{group_id:03d}_input.txt"
    with open(input_file, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("SYSTEM PROMPT:\n")
        f.write("=" * 60 + "\n")
        f.write(system_prompt)
        f.write("\n\n")
        f.write("=" * 60 + "\n")
        f.write("USER PROMPT:\n")
        f.write("=" * 60 + "\n")
        f.write(user_prompt)

    # Log output (VLM response)
    output_file = log_dir / f"group_{group_id:03d}_output.txt"
    with open(output_file, 'w') as f:
        f.write(response)


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
    timestamp: float          # Original video timestamp where it appears
    clip_timestamp: float     # Clip-relative timestamp
    semantic_name: str        # Canonical name (e.g., "butter", "chicken")
    trigger_text: str         # The narration text that triggered detection
    detected_form: str        # How it appeared (e.g., "in plastic container")
    group_id: int             # Group where it was detected
    video_id: str = ""        # Video where it was discovered


def load_clip_manifest(clips_dir: Path, video_id: str) -> Optional[Dict[int, Dict]]:
    """Load manifest mapping group_id to clip info.

    Args:
        clips_dir: Path to clips directory
        video_id: Video ID

    Returns:
        Dict mapping group_id to clip info, or None if manifest doesn't exist
    """
    manifest_path = clips_dir / video_id / "manifest.json"
    if not manifest_path.exists():
        return None

    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    # Build lookup by group_id
    return {c['group_id']: c for c in manifest.get('clips', [])}


def format_group_narrations(group: SemanticGroup, clip_info: Optional[Dict] = None) -> str:
    """Format all narrations in a group for the LLM prompt.

    Args:
        group: SemanticGroup with narrations
        clip_info: Optional clip info for timestamp adjustment

    Returns:
        Formatted narration text with timestamps
    """
    clip_start = clip_info.get('clip_start', group.start_time) if clip_info else group.start_time

    lines = []
    for i, narration in enumerate(group.narrations, 1):
        text = narration['narration'].strip()
        original_ts = narration['start_timestamp']
        # Adjust to clip-relative timestamp
        clip_ts = original_ts - clip_start
        lines.append(f"{i}. [{clip_ts:.2f}s] {text}")
    return "\n".join(lines)


def format_inventory(inventory: Set[str]) -> str:
    """Format current inventory for the LLM prompt"""
    if not inventory:
        return "Empty - no food items tracked yet"
    return ", ".join(sorted(inventory))


def classify_group_arrivals(
    group: SemanticGroup,
    clip_info: Optional[Dict],
    inventory: Set[str],
    vlm_client: VLMClient,
    video_id: str = "",
    verbose: bool = False
) -> List[Dict]:
    """Use VLM to identify NEW food arrivals in a semantic group.

    Args:
        group: SemanticGroup to process
        clip_info: Clip info for timestamp mapping (or None)
        inventory: Current accumulated food inventory
        vlm_client: VLMClient instance (Qwen or GPT-4o)
        verbose: Print debug output

    Returns:
        List of new arrival dictionaries with semantic_name and detected_form
    """
    group_text = format_group_narrations(group, clip_info)
    inventory_str = format_inventory(inventory)

    # Get clip time info for prompt
    if clip_info:
        clip_start = clip_info['clip_start']
        clip_end = clip_info['clip_end']
        time_info = f"Clip: {clip_start:.1f}s - {clip_end:.1f}s (timestamps below are clip-relative)"
    else:
        time_info = f"Time: {group.start_time:.1f}s - {group.end_time:.1f}s"

    system_prompt = """You are an "Ingredient Spotter." Your STRICT goal is to identify the first appearance of EDIBLE food items.

CORE DEFINITION: VALID ROOT ENTITY
A Valid Root is a consumable food ingredient (e.g., "milk", "flour", "orange") or its direct packaging (e.g., "bottle of milk", "bag of flour").

⛔ EXCLUSION LIST (ABSOLUTELY IGNORE THESE):
1. INFRASTRUCTURE: Fridge, oven, microwave, sink, drawer, cupboard, boiler, counter.
2. TOOLS & UTENSILS: Knife, spoon, fork, spatula, grater, stirrer, sponge, towel.
3. CONTAINERS (Empty): Bowl, plate, mug, glass, pan, pot, bin, bag (unless it explicitly contains food).
4. APPLIANCES: Coffee machine, juicer, food processor, frother, toaster.

STRICT FILTERING RULES:
1. CHECK INVENTORY: If the item (or its parent) is already in "Current Inventory," IGNORE IT.
   - *Example:* Inventory has "milk bottle" -> Ignore "milk" or "milk cap".
2. CHECK EDIBILITY: If you cannot eat/drink it (or its contents), IGNORE IT.
   - *Example:* "Grab the towel" -> IGNORE (Not edible).
   - *Example:* "Get the milk bottle" -> ACCEPT (Contains edible milk).

OUTPUT FORMAT:
Return a JSON array of NEW food roots.
"""

    user_prompt = f"""**CURRENT INVENTORY (Already Known):**
{inventory_str}

**NARRATION BLOCK:**
{group_text}

**TASK:**
Scan the narration block for NEW, EDIBLE Root Entities not listed in Current Inventory.
Apply the "Exclusion List" strictly.

Return JSON:
[
  {{
    "semantic_name": "exact name",
    "trigger_text": "full sentence",
    "trigger_line": line_number,
    "category": "food" // Force classification to ensure it's not a tool
  }}
]
"""

    if verbose:
        print(f"\n{'='*60}")
        print(f"Group {group.group_id}: \"{group.query}\"")
        print(f"Current inventory: {inventory_str}")
        print(f"{'='*60}")

    try:
        response_text = vlm_client.query(system_prompt, user_prompt, max_tokens=500)

        # Log VLM call
        log_vlm_call(video_id, group.group_id, system_prompt, user_prompt, response_text)

        if verbose:
            print(f"Response: {response_text}")

        # Parse JSON response (handle markdown code blocks and extra text)
        parse_text = response_text

        # Handle markdown code blocks
        if "```json" in parse_text:
            parse_text = parse_text.split("```json")[1].split("```")[0].strip()
        elif "```" in parse_text:
            parse_text = parse_text.split("```")[1].split("```")[0].strip()

        # Extract just the JSON array - find matching brackets
        # VLM sometimes adds explanatory text after the JSON
        if parse_text.strip().startswith('['):
            bracket_count = 0
            end_idx = 0
            for i, char in enumerate(parse_text):
                if char == '[':
                    bracket_count += 1
                elif char == ']':
                    bracket_count -= 1
                    if bracket_count == 0:
                        end_idx = i + 1
                        break
            if end_idx > 0:
                parse_text = parse_text[:end_idx]

        arrivals = json.loads(parse_text)

        # Validate response structure
        if not isinstance(arrivals, list):
            arrivals = []

        return arrivals

    except Exception as e:
        print(f"ERROR classifying group {group.group_id}: {e}")
        return []


def process_video(
    video_id: str,
    grouping_dir: Path,
    csv_path: Path,
    clips_dir: Path,
    vlm_client: VLMClient,
    initial_inventory: Set[str] = None,
    verbose: bool = False,
    dry_run: bool = False
) -> Tuple[Dict, Set[str]]:
    """Process a single video's semantic groups to discover inventory arrivals.

    Args:
        video_id: Video ID to process
        grouping_dir: Path to narration_grouping directory
        csv_path: Path to narrations CSV
        clips_dir: Path to extracted clips (for timestamp mapping)
        vlm_client: VLMClient instance
        initial_inventory: Inventory carried over from previous videos
        verbose: Print debug output
        dry_run: If True, don't call VLM

    Returns:
        Tuple of (result dict, updated inventory set)
    """
    # Load semantic groups
    groups = load_and_enrich_video_groups(grouping_dir, csv_path, video_id)
    if groups is None:
        print(f"  SKIP: No semantic groupings found for {video_id}")
        return {'video_id': video_id, 'error': 'no_groupings', 'arrivals': []}, set(initial_inventory or [])

    # Load clip manifest for timestamp mapping
    clip_manifest = load_clip_manifest(clips_dir, video_id)
    if clip_manifest:
        print(f"  Loaded clip manifest: {len(clip_manifest)} clips")
    else:
        print(f"  Warning: No clip manifest found, using original timestamps")

    # Start with inherited inventory or empty set
    global_inventory: Set[str] = set(initial_inventory) if initial_inventory else set()
    initial_size = len(global_inventory)

    print(f"\nProcessing {video_id}: {len(groups)} semantic groups "
          f"(starting inventory: {initial_size} items)")

    arrivals: List[NewArrivalEvent] = []

    for group in groups:
        # Get clip info for this group
        clip_info = clip_manifest.get(group.group_id) if clip_manifest else None

        if dry_run:
            print(f"  G{group.group_id:02d}: \"{group.query[:40]}...\" [DRY-RUN]")
            continue

        # Classify new arrivals in this group
        new_items = classify_group_arrivals(group, clip_info, global_inventory, vlm_client, video_id, verbose)

        # Create arrival events and update inventory
        for item in new_items:
            semantic_name = item.get('semantic_name', '').lower().strip()
            detected_form = item.get('detected_form', '')
            trigger_text = item.get('trigger_text', '').strip()  # From VLM response
            trigger_line = item.get('trigger_line', 1)  # 1-indexed line number

            if not semantic_name:
                continue

            # Skip if somehow already in inventory (LLM should have filtered)
            if semantic_name in global_inventory:
                continue

            # Get clip_start for timestamp calculation
            clip_start = clip_info.get('clip_start', group.start_time) if clip_info else group.start_time

            # Get trigger_timestamp from VLM-provided trigger_line
            trigger_timestamp = group.start_time
            if trigger_line and 1 <= trigger_line <= len(group.narrations):
                narration = group.narrations[trigger_line - 1]  # Convert to 0-indexed
                trigger_timestamp = narration['start_timestamp']
                # If VLM didn't provide trigger_text, use the narration text
                if not trigger_text:
                    trigger_text = narration['narration'].strip()
            elif not trigger_text and group.narrations:
                # Fallback to first narration in group
                trigger_text = group.narrations[0]['narration'].strip()
                trigger_timestamp = group.narrations[0]['start_timestamp']

            # Calculate clip-relative timestamp
            clip_timestamp = trigger_timestamp - clip_start

            event = NewArrivalEvent(
                timestamp=trigger_timestamp,
                clip_timestamp=clip_timestamp,
                semantic_name=semantic_name,
                trigger_text=trigger_text,
                detected_form=detected_form,
                group_id=group.group_id,
                video_id=video_id
            )
            arrivals.append(event)
            global_inventory.add(semantic_name)

            if verbose:
                print(f"  NEW: {semantic_name} ({detected_form}) @ {trigger_timestamp:.2f}s (clip: {clip_timestamp:.2f}s)")

        # Progress update
        if not verbose and (group.group_id + 1) % 10 == 0:
            print(f"  Progress: {group.group_id + 1}/{len(groups)} groups, "
                  f"{len(arrivals)} arrivals")

    print(f"  Completed: {len(arrivals)} new arrivals in this video, "
          f"cumulative inventory: {len(global_inventory)} items")

    result = {
        'video_id': video_id,
        'total_groups_processed': len(groups),
        'total_arrivals': len(arrivals),
        'inherited_inventory_size': initial_size,
        'arrivals': [asdict(a) for a in arrivals],
        'video_inventory': sorted([a.semantic_name for a in arrivals]),
        'cumulative_inventory': sorted(list(global_inventory))
    }

    return result, global_inventory


def main():
    """Main pipeline execution"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Inventory Discovery (Semantic Groups) - Identify first appearances of food items"
    )
    parser.add_argument(
        '--grouping-dir',
        type=Path,
        default=DEFAULT_GROUPING_DIR,
        help='Directory containing semantic grouping JSON files'
    )
    parser.add_argument(
        '--csv',
        type=Path,
        default=DEFAULT_CSV_PATH,
        help='Path to narrations CSV file'
    )
    parser.add_argument(
        '--clips-dir',
        type=Path,
        default=DEFAULT_CLIPS_DIR,
        help='Directory containing extracted clips (for timestamp mapping)'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help='Output directory for arrival results'
    )
    parser.add_argument(
        '--video-id',
        default=None,
        help='Process only a specific video ID'
    )
    parser.add_argument(
        '--start-video',
        default=None,
        help='Start video ID for range (e.g., P01-20240202-110250)'
    )
    parser.add_argument(
        '--end-video',
        default=None,
        help='End video ID for range'
    )
    parser.add_argument(
        '--model',
        default='qwen',
        choices=['qwen', 'gpt-4o'],
        help='VLM model to use (default: qwen)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print verbose output'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show groups without calling VLM'
    )

    args = parser.parse_args()

    # Handle --video-id as shorthand for single video range
    if args.video_id:
        args.start_video = args.video_id
        args.end_video = args.video_id

    print("="*70)
    print("INVENTORY DISCOVERY (Semantic Groups)")
    print("="*70)
    print(f"Grouping dir: {args.grouping_dir}")
    print(f"Clips dir:    {args.clips_dir}")
    print(f"Output dir:   {args.output_dir}")
    print(f"Model:        {args.model}")
    if args.dry_run:
        print("MODE:         DRY-RUN (no VLM calls)")

    # Find videos with groupings
    all_video_ids = sorted(find_videos_with_groupings(args.grouping_dir))

    if not all_video_ids:
        print("ERROR: No videos with semantic groupings found")
        return

    # Filter to range if specified
    if args.start_video or args.end_video:
        start_idx = 0
        end_idx = len(all_video_ids)

        if args.start_video and args.start_video in all_video_ids:
            start_idx = all_video_ids.index(args.start_video)
        if args.end_video and args.end_video in all_video_ids:
            end_idx = all_video_ids.index(args.end_video) + 1

        video_ids = all_video_ids[start_idx:end_idx]
        print(f"Range: {args.start_video or '(start)'} -> {args.end_video or '(end)'}")
    else:
        video_ids = all_video_ids

    print(f"\nVideos to process: {len(video_ids)}")

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Set up VLM logging directory
    global VLM_LOG_DIR
    VLM_LOG_DIR = args.output_dir / "vlm_logs"
    VLM_LOG_DIR.mkdir(parents=True, exist_ok=True)
    print(f"VLM logs:     {VLM_LOG_DIR}")

    # Initialize VLM client (unless dry run)
    vlm_client = None
    if not args.dry_run:
        print(f"\n[Setup] Initializing {args.model}...")
        vlm_client = VLMClient(model_name=args.model)
        print("VLM client initialized")

    # Process each video with persistent inventory
    all_results = {}
    total_arrivals = 0
    persistent_inventory: Set[str] = set()

    for i, video_id in enumerate(video_ids, 1):
        print(f"\n{'='*70}")
        print(f"[{i}/{len(video_ids)}] Processing: {video_id}")
        print("="*70)

        result, persistent_inventory = process_video(
            video_id=video_id,
            grouping_dir=args.grouping_dir,
            csv_path=args.csv,
            clips_dir=args.clips_dir,
            vlm_client=vlm_client,
            initial_inventory=persistent_inventory,
            verbose=args.verbose,
            dry_run=args.dry_run
        )

        # Save per-video results (unless dry run)
        if not args.dry_run:
            output_file = args.output_dir / f"{video_id}_arrivals.json"
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"  Saved to: {output_file}")

        all_results[video_id] = result
        total_arrivals += result.get('total_arrivals', 0)

    # Save aggregated results (unless dry run)
    if not args.dry_run:
        print(f"\n{'='*70}")
        print("Saving aggregated results...")
        print("="*70)

        aggregated = {
            'participant_id': 'P01',
            'model_used': args.model,
            'grouping_type': 'semantic',
            'total_videos': len(all_results),
            'total_arrivals': total_arrivals,
            'final_inventory': sorted(list(persistent_inventory)),
            'final_inventory_size': len(persistent_inventory),
            'videos': all_results
        }

        aggregated_file = args.output_dir / "all_p01_arrivals.json"
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

    if persistent_inventory:
        print(f"\nFinal cumulative inventory ({len(persistent_inventory)} items):")
        inv_list = sorted(list(persistent_inventory))
        for i in range(0, len(inv_list), 8):
            print(f"  {', '.join(inv_list[i:i+8])}")

    print(f"\nResults saved to: {args.output_dir}")
    print("="*70)


if __name__ == '__main__':
    main()

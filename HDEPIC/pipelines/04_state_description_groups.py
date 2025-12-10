#!/usr/bin/env python3
"""
State Description Generation (Semantic Groups)

This script generates state descriptions for semantic groups, providing context
via padded timestamps:
- PREVIOUS: Narrations in the padding before the group
- CURRENT: Narrations within the semantic group (focus for state changes)
- AFTER: Narrations in the padding after the group

The VLM uses PREVIOUS/AFTER for context but only outputs state changes for CURRENT.

Usage:
    python 04_state_description_groups.py --video-id P01-20240202-110250
    python 04_state_description_groups.py --video-id P01-20240202-110250 --group-id 5
    python 04_state_description_groups.py --video-id P01-20240202-110250 --verbose
"""

import json
import sys
import argparse
import requests
import base64
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / 'food_graph'))

from semantic_utils import (
    SemanticGroup,
    find_videos_with_groupings,
    load_and_enrich_video_groups,
    load_narrations_for_video,
)

from food_graph.vlm_prompts import (
    STATE_DESCRIPTION_PROMPT,
    parse_state_description_response,
)

# Qwen3-VL API endpoint
QWEN3VL_URL = "http://saltyfish.eecs.umich.edu:8000/v1/chat/completions"
QWEN_MODEL = "Qwen/Qwen3-VL-30B-A3B-Instruct"

# Default paths (relative to this script's location in pipelines/)
_SCRIPT_DIR = Path(__file__).parent
_PROJECT_ROOT = _SCRIPT_DIR.parent

DEFAULT_GROUPING_DIR = _PROJECT_ROOT / "narration_grouping"
DEFAULT_CSV_PATH = _PROJECT_ROOT / "P01" / "participant_P01_narrations.csv"
DEFAULT_CLIPS_DIR = _PROJECT_ROOT / "outputs" / "food_clips"
DEFAULT_OUTPUT_DIR = _PROJECT_ROOT / "outputs" / "state_descriptions"

# Global variable for log directory
VLM_LOG_DIR: Optional[Path] = None


def log_vlm_call(video_id: str, group_id: int, system_prompt: str, user_prompt: str, response: str):
    """Log VLM input and output to files."""
    if VLM_LOG_DIR is None:
        return

    log_dir = VLM_LOG_DIR / video_id
    log_dir.mkdir(parents=True, exist_ok=True)

    # Log input
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

    # Log output
    output_file = log_dir / f"group_{group_id:03d}_output.txt"
    with open(output_file, 'w') as f:
        f.write(response)


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


def load_clip_manifest(clips_dir: Path, video_id: str) -> Optional[Dict[int, Dict]]:
    """Load manifest mapping group_id to clip info."""
    manifest_path = clips_dir / video_id / "manifest.json"
    if not manifest_path.exists():
        return None

    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    return {c['group_id']: c for c in manifest.get('clips', [])}


def get_narrations_in_range(
    all_narrations: List[Dict],
    start_time: float,
    end_time: float,
    include_start: bool = True,
    include_end: bool = True
) -> List[Dict]:
    """Get narrations within a timestamp range based on start_timestamp."""
    result = []
    for narr in all_narrations:
        ts = narr.get('start_timestamp', 0)

        # Check boundaries
        if include_start:
            if ts < start_time:
                continue
        else:
            if ts <= start_time:
                continue

        if include_end:
            if ts > end_time:
                continue
        else:
            if ts >= end_time:
                continue

        result.append(narr)

    return sorted(result, key=lambda x: x.get('start_timestamp', 0))


def format_narrations_section(
    narrations: List[Dict],
    clip_start: float,
    section_name: str
) -> str:
    """Format narrations for a section with clip-relative timestamps."""
    if not narrations:
        return f"**{section_name}**\n(none)\n"

    lines = [f"**{section_name}**"]
    for narr in narrations:
        abs_ts = narr.get('start_timestamp', 0)
        clip_ts = abs_ts - clip_start
        text = narr.get('narration', '').strip()
        lines.append(f"[{clip_ts:.2f}s] {text}")

    return "\n".join(lines) + "\n"


def build_context_prompt(
    all_narrations: List[Dict],
    clip_info: Dict,
    group: SemanticGroup
) -> str:
    """Build prompt with PREVIOUS / CURRENT / AFTER sections."""
    clip_start = clip_info['clip_start']
    original_start = clip_info['original_start']
    original_end = clip_info['original_end']
    clip_end = clip_info['clip_end']

    # Previous: [clip_start, original_start)
    previous = get_narrations_in_range(
        all_narrations, clip_start, original_start,
        include_start=True, include_end=False
    )

    # Current: use group's narrations (already enriched)
    current = group.narrations

    # After: (original_end, clip_end]
    after = get_narrations_in_range(
        all_narrations, original_end, clip_end,
        include_start=False, include_end=True
    )

    # Build prompt sections
    sections = []

    # Previous context
    if previous:
        sections.append(format_narrations_section(previous, clip_start, "CONTEXT: PREVIOUS NARRATIONS (before current action)"))

    # Current narrations (the focus)
    current_lines = ["**CURRENT NARRATIONS (analyze these for state changes)**"]
    for i, narr in enumerate(current, 1):
        abs_ts = narr.get('start_timestamp', 0)
        clip_ts = abs_ts - clip_start
        text = narr.get('narration', '').strip()
        current_lines.append(f"{i}. [{clip_ts:.2f}s] {text}")
    sections.append("\n".join(current_lines) + "\n")

    # After context
    if after:
        sections.append(format_narrations_section(after, clip_start, "CONTEXT: AFTER NARRATIONS (after current action)"))

    # Task instruction
    sections.append("""**TASK:**
For each narration in CURRENT NARRATIONS above, apply the 3-step process:
1. RELEVANCE FILTER - Is this a food-related action?
2. VISUAL VERIFICATION - What does the video show?
3. INFER STATE CHANGE - Describe the outcome.

Use PREVIOUS/AFTER narrations for context only. Output analysis ONLY for CURRENT narrations.
Output your analysis for EACH CURRENT line in the format: [timestamp] "narration" -> SKIP/CONTEXT/STATE""")

    return "\n".join(sections)


def format_text_output(video_id: str, group: SemanticGroup, clip_info: Dict, descriptions: List[Dict]) -> str:
    """Format descriptions as human-readable text."""
    lines = [
        f"=== GROUP {group.group_id}: \"{group.query}\" ===",
        f"Time: {clip_info['original_start']:.1f}s - {clip_info['original_end']:.1f}s",
        f"Clip: {clip_info['clip_start']:.1f}s - {clip_info['clip_end']:.1f}s (padded: {clip_info['padded']})",
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


def process_group(
    vlm_client: VLMClient,
    video_id: str,
    group: SemanticGroup,
    clip_info: Dict,
    all_narrations: List[Dict],
    clips_dir: Path,
    verbose: bool = False
) -> Dict:
    """Process a single semantic group and generate state descriptions."""
    group_id = group.group_id
    narrations = group.narrations

    if verbose:
        print(f"\n  Group {group_id}: \"{group.query[:50]}...\"")
        print(f"    Narrations: {len(narrations)}")

    # Get video clip path
    video_path = clips_dir / video_id / clip_info['clip_path']
    if video_path.exists():
        if verbose:
            print(f"    Video: {video_path.name}")
    else:
        video_path = None
        if verbose:
            print(f"    Video: (not found)")

    # Build prompt with context
    user_prompt = build_context_prompt(all_narrations, clip_info, group)

    if verbose:
        print(f"    Querying VLM...")

    # Query VLM
    response = vlm_client.query(
        STATE_DESCRIPTION_PROMPT,
        user_prompt,
        video_path=video_path,
        max_tokens=3000
    )

    # Log VLM call
    log_vlm_call(video_id, group_id, STATE_DESCRIPTION_PROMPT, user_prompt, response)

    # Parse response
    descriptions = parse_state_description_response(response, narrations)

    # Count stats
    skip_count = sum(1 for d in descriptions if d.get('skip'))
    state_count = sum(1 for d in descriptions if d.get('state_description'))

    if verbose:
        print(f"    Results: {skip_count} skipped, {state_count} with state descriptions")

    return {
        'video_id': video_id,
        'group_id': group_id,
        'query': group.query,
        'original_start': clip_info['original_start'],
        'original_end': clip_info['original_end'],
        'clip_start': clip_info['clip_start'],
        'clip_end': clip_info['clip_end'],
        'padded': clip_info['padded'],
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
        description="State Description Generation (Semantic Groups)"
    )
    parser.add_argument(
        '--video-id',
        required=True,
        help='Video ID to process (e.g., P01-20240202-110250)'
    )
    parser.add_argument(
        '--group-id',
        type=int,
        default=None,
        help='Process only a specific group ID'
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
        help='Directory containing video clips'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help='Output directory for state descriptions'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print verbose output'
    )

    args = parser.parse_args()

    # Setup output directories
    output_dir = args.output_dir / args.video_id
    output_dir.mkdir(parents=True, exist_ok=True)

    global VLM_LOG_DIR
    VLM_LOG_DIR = args.output_dir / "vlm_logs"
    VLM_LOG_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("STATE DESCRIPTION GENERATION (Semantic Groups)")
    print("=" * 70)
    print(f"Video:       {args.video_id}")
    print(f"Grouping:    {args.grouping_dir}")
    print(f"Clips:       {args.clips_dir}")
    print(f"Output:      {output_dir}")
    print(f"VLM logs:    {VLM_LOG_DIR}")

    # Initialize VLM client
    print("\n[Setup] Initializing VLM client...")
    vlm_client = VLMClient()

    # Load semantic groups
    print(f"\n[Step 1] Loading semantic groups for {args.video_id}...")
    groups = load_and_enrich_video_groups(args.grouping_dir, args.csv, args.video_id)

    if groups is None:
        print("ERROR: No semantic groups found")
        return

    # Load all narrations for context
    print(f"[Step 2] Loading all narrations for context...")
    all_narrations = load_narrations_for_video(args.csv, args.video_id)
    print(f"  Loaded {len(all_narrations)} narrations")

    # Load clip manifest
    print(f"[Step 3] Loading clip manifest...")
    clip_manifest = load_clip_manifest(args.clips_dir, args.video_id)
    if clip_manifest is None:
        print("ERROR: Clip manifest not found")
        return
    print(f"  Loaded {len(clip_manifest)} clips")

    # Filter to specific group if requested
    if args.group_id is not None:
        groups = [g for g in groups if g.group_id == args.group_id]
        if not groups:
            print(f"ERROR: Group {args.group_id} not found")
            return

    print(f"\n[Step 4] Processing {len(groups)} groups...")

    # Process groups
    all_results = []
    total_stats = {'total_narrations': 0, 'skipped': 0, 'with_state': 0}

    for i, group in enumerate(groups, 1):
        group_id = group.group_id

        if group_id not in clip_manifest:
            print(f"  [{i}/{len(groups)}] Group {group_id}: SKIP (no clip info)")
            continue

        clip_info = clip_manifest[group_id]
        print(f"  [{i}/{len(groups)}] Group {group_id}: \"{group.query[:40]}...\"")

        result = process_group(
            vlm_client, args.video_id, group, clip_info,
            all_narrations, args.clips_dir, args.verbose
        )
        all_results.append(result)

        # Aggregate stats
        total_stats['total_narrations'] += result['stats']['total_narrations']
        total_stats['skipped'] += result['stats']['skipped']
        total_stats['with_state'] += result['stats']['with_state']

        # Save per-group outputs
        json_file = output_dir / f"group_{group_id:03d}_descriptions.json"
        txt_file = output_dir / f"group_{group_id:03d}_descriptions.txt"

        with open(json_file, 'w') as f:
            json.dump(result, f, indent=2)

        text_output = format_text_output(args.video_id, group, clip_info, result['descriptions'])
        with open(txt_file, 'w') as f:
            f.write(text_output)

        if args.verbose:
            print(f"    Saved: {json_file.name}, {txt_file.name}")

    # Save summary
    print(f"\n[Step 5] Saving summary...")
    summary = {
        'video_id': args.video_id,
        'processed_at': datetime.now().isoformat(),
        'groups_processed': len(all_results),
        'total_stats': total_stats,
        'groups': [
            {
                'group_id': r['group_id'],
                'query': r['query'],
                'stats': r['stats']
            }
            for r in all_results
        ]
    }

    summary_file = output_dir / 'summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary saved: {summary_file}")

    # Print final summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print("=" * 70)
    print(f"Groups processed: {len(all_results)}")
    print(f"Total narrations: {total_stats['total_narrations']}")
    print(f"  - Skipped (non-food): {total_stats['skipped']}")
    print(f"  - With state descriptions: {total_stats['with_state']}")
    print(f"\nOutputs saved to: {output_dir}")
    print("=" * 70)


if __name__ == '__main__':
    main()

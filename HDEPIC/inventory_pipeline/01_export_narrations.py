#!/usr/bin/env python3
"""
Export Narrations Tool

Filter food-related narrations using GPT-5.2 and export for downstream processing.
Supports all participants (P01-P09) from HD_EPIC_Narrations.pkl.

Commands:

    list - Show available participants and videos
        python 01_export_narrations.py list
        python 01_export_narrations.py list --participant P02

    filter - Filter food-related narrations using GPT-5.2
        python 01_export_narrations.py filter --video-id P01-20240202-110250
        python 01_export_narrations.py filter --participant P02 --all
        python 01_export_narrations.py filter --video-id P01-20240202-110250 --skip-api

    extract - Extract filtered lines from existing response file (legacy)
        python 01_export_narrations.py extract --video-id P01-20240202-110250

    inventory - Export filtered food narrations for inventory discovery
        python 01_export_narrations.py inventory --participant P01 --video-id P01-20240202-110250

Output structure:
    outputs/01_filter/{participant}/
        filter_{video_id}.txt         - Input narrations (indexed)
        filter_{video_id}_response.txt - JSON response with narration_ids
        filter_{video_id}_raw.txt     - Raw GPT response (for debugging)
        filtered_{video_id}.txt       - Filtered narrations (for human review)
"""

import csv
import json
import argparse
import os
import re
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Set, Tuple

from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

# Default paths (relative to this script's location in annotation_pipeline/)
_SCRIPT_DIR = Path(__file__).parent
_PROJECT_ROOT = _SCRIPT_DIR.parent

DEFAULT_PICKLE_PATH = _PROJECT_ROOT / "data" / "hd-epic-annotations" / "narrations-and-action-segments" / "HD_EPIC_Narrations.pkl"
DEFAULT_OUTPUT_DIR = _PROJECT_ROOT / "outputs" / "01_filter"

# Cache for pickle data (loaded once)
_PICKLE_CACHE = None


# =============================================================================
# GPT-5.2 Filter Prompt
# =============================================================================

FILTER_PROMPT = """You are a "Kitchen Activity Segmenter."

Your goal is to process a raw narration log and produce a structured list of **FOOD-ONLY** activity blocks.

INPUT:
A list of narration IDs and narrations.

### PART 1: THE FILTER (Relevance Check)
Scan the log and identify lines that are relevant. Use this "Chain of Custody" logic:

1. 🟢 **KEEP (DEFINITELY FOOD):**
   - Explicit mentions of ingredients (milk, flour, chicken).
   - Explicit mentions of eating, tasting, or disposing of food.

2. 🟡 **KEEP (ACTIVE CONTAINER CHAINS):**
   - **Look Forward (Setup):** Interactions with empty containers/tools (bowls, pans, blenders) ONLY IF they are immediately followed by food entering them.
   - **Look Backward (Transport):** Movements of containers ONLY IF they currently hold food (based on previous lines).
   - **Rule:** If a container holds food, ANY interaction with it (moving, covering, uncovering) is relevant.

3. 🔴 **DISCARD (DISCONNECTED NOISE):**
   - **Infrastructure:** Opening cupboards/drawers that don't lead to retrieving food.
   - **Isolated Maintenance:** Cleaning, washing, or arranging items that never touch food in this sequence.
   - **Passive Actions:** Movements or observations that have no effect on food state.

### PART 2: THE GROUPER (Segmentation)
Group the "KEEP" lines into cohesive semantic blocks.
- **Cluster:** Combine consecutive relevant lines into a single Event (e.g., "Making Coffee", "Chopping Salad").
- **Gaps:** If there is a long period (e.g., >30s) of "DISCARD" lines between actions, split them into separate blocks.

### OUTPUT FORMAT
Return a JSON list of segments.
**CRITICAL:** If a time range contains NO relevant food actions, DO NOT create a block. Skip it entirely.

[
  {
    "label": "Pouring Milk",
    "relevant_lines": [15, 16, 17, 18]
  },
  ...
]"""


# =============================================================================
# GPT Client
# =============================================================================

class GPTClient:
    """Wrapper for Azure OpenAI API with Responses API support."""

    def __init__(self, model: str = "gpt-5.2"):
        self.model = model

        api_key = os.getenv("AZURE_OPENAI_API_KEY_2")
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT_2", "").strip()
        api_version = "2025-03-01-preview"  # Responses API

        if not api_key or not endpoint:
            raise ValueError(f"Missing API credentials. Set AZURE_OPENAI_API_KEY_2 and AZURE_OPENAI_ENDPOINT_2")

        self.client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=api_version,
        )


# =============================================================================
# GPT Filter Functions
# =============================================================================

def filter_with_gpt(
    narrations: List[Dict],
    reasoning_effort: str = "high",
    verbose: bool = False
) -> Tuple[str, Set[str]]:
    """
    Call GPT-5.2 to filter food-related narrations.

    Args:
        narrations: List of narration dicts with 'unique_narration_id' and 'narration'
        reasoning_effort: 'low', 'medium', or 'high'
        verbose: Print progress info

    Returns:
        (raw_response_text, set of narration_ids that passed filter)
    """
    client = GPTClient()

    # Format input: line_number | narration_id | narration
    # Use 0-indexed line numbers for the model
    input_lines = []
    for idx, n in enumerate(narrations):
        input_lines.append(f"{idx} | {n['unique_narration_id']} | {n['narration']}")
    input_text = "\n".join(input_lines)

    if verbose:
        print(f"Calling GPT-5.2 with {len(narrations)} narrations...")
        print(f"  Reasoning effort: {reasoning_effort}")

    response = client.client.responses.create(
        model=client.model,
        reasoning={"effort": reasoning_effort},
        instructions=FILTER_PROMPT,
        input=input_text,
        max_output_tokens=16384,
    )

    raw_response = response.output_text

    if verbose:
        print(f"  Input tokens: {response.usage.input_tokens}")
        print(f"  Output tokens: {response.usage.output_tokens}")
        if hasattr(response.usage, 'output_tokens_details'):
            print(f"  Reasoning tokens: {response.usage.output_tokens_details.reasoning_tokens}")

    # Parse response and convert to narration IDs
    filtered_ids = parse_filter_response(raw_response, narrations)

    return raw_response, filtered_ids


def parse_filter_response(response_text: str, narrations: List[Dict]) -> Set[str]:
    """
    Parse GPT filter response and convert line indices to narration_ids.

    Expected format:
    [{"label": "...", "relevant_lines": [0, 1, 2, ...]}, ...]

    Returns set of narration_ids that passed the filter.
    """
    # Try to find JSON in markdown code block
    code_block_match = re.search(r'```(?:json)?\s*(\[[\s\S]*?\])\s*```', response_text)
    if code_block_match:
        try:
            data = json.loads(code_block_match.group(1))
        except json.JSONDecodeError:
            data = None
    else:
        # Try to find raw JSON array
        json_match = re.search(r'\[[\s\S]*\]', response_text)
        if json_match:
            try:
                data = json.loads(json_match.group(0))
            except json.JSONDecodeError:
                data = None
        else:
            data = None

    if data is None:
        print(f"WARNING: Could not parse JSON from response")
        return set()

    # Extract all line indices and convert to narration IDs
    narration_ids = set()

    for segment in data:
        if 'relevant_lines' in segment:
            for line_idx in segment['relevant_lines']:
                if 0 <= line_idx < len(narrations):
                    narration_ids.add(narrations[line_idx]['unique_narration_id'])
                else:
                    print(f"WARNING: Line index {line_idx} out of range (0-{len(narrations)-1})")

    return narration_ids


# =============================================================================
# Narration Loading Functions (CSV + Pickle support)
# =============================================================================

def load_pickle_data() -> Dict:
    """Load and cache the HD_EPIC_Narrations.pkl file."""
    global _PICKLE_CACHE
    if _PICKLE_CACHE is None:
        if not DEFAULT_PICKLE_PATH.exists():
            raise FileNotFoundError(f"Pickle file not found: {DEFAULT_PICKLE_PATH}")
        with open(DEFAULT_PICKLE_PATH, 'rb') as f:
            _PICKLE_CACHE = pickle.load(f)
    return _PICKLE_CACHE


def get_participant_csv_path(participant: str) -> Optional[Path]:
    """Get CSV path for a participant if it exists."""
    csv_path = _PROJECT_ROOT / participant / f"participant_{participant}_narrations.csv"
    return csv_path if csv_path.exists() else None


def load_narrations_for_video(video_id: str, participant: str = None) -> List[Dict]:
    """
    Load all narrations for a specific video.

    Tries participant CSV first, falls back to pickle file.
    """
    # Infer participant from video_id if not provided
    if participant is None:
        participant = video_id.split('-')[0]

    # Try CSV first
    csv_path = get_participant_csv_path(participant)
    if csv_path:
        return _load_narrations_from_csv(csv_path, video_id)

    # Fall back to pickle
    return _load_narrations_from_pickle(video_id)


def _load_narrations_from_csv(csv_path: Path, video_id: str) -> List[Dict]:
    """Load narrations from CSV file."""
    narrations = []

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['video_id'] == video_id:
                narrations.append({
                    'unique_narration_id': row['unique_narration_id'],
                    'narration': row['narration'].strip(),
                    'start_timestamp': float(row['start_timestamp']),
                })

    # Sort by timestamp
    narrations.sort(key=lambda x: x['start_timestamp'])
    return narrations


def _load_narrations_from_pickle(video_id: str) -> List[Dict]:
    """Load narrations from pickle file."""
    df = load_pickle_data()

    # Filter to this video
    video_df = df[df['video_id'] == video_id]

    narrations = []
    for _, row in video_df.iterrows():
        narrations.append({
            'unique_narration_id': row['unique_narration_id'],
            'narration': row['narration'].strip(),
            'start_timestamp': float(row['start_timestamp']),
        })

    # Sort by timestamp
    narrations.sort(key=lambda x: x['start_timestamp'])
    return narrations


def get_all_video_ids(participant: str = None) -> List[str]:
    """
    Get list of all unique video IDs.

    If participant is specified, returns only videos for that participant.
    """
    # Try CSV first
    if participant:
        csv_path = get_participant_csv_path(participant)
        if csv_path:
            return _get_video_ids_from_csv(csv_path)

    # Fall back to pickle
    return _get_video_ids_from_pickle(participant)


def _get_video_ids_from_csv(csv_path: Path) -> List[str]:
    """Get video IDs from CSV file."""
    video_ids = set()

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            video_ids.add(row['video_id'])

    return sorted(video_ids)


def _get_video_ids_from_pickle(participant: str = None) -> List[str]:
    """Get video IDs from pickle file."""
    df = load_pickle_data()

    if participant:
        df = df[df['video_id'].str.startswith(participant)]

    return sorted(df['video_id'].unique().tolist())


def export_for_filter(video_id: str, csv_path: Path, output_dir: Path) -> Path:
    """
    Export single video narrations for food filtering.

    Output format: narration_id | narration_text
    Also creates an empty response file for user to paste Gemini output.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load narrations
    narrations = load_narrations_for_video(csv_path, video_id)

    if not narrations:
        print(f"ERROR: No narrations found for video {video_id}")
        return None

    # Write input TXT
    output_file = output_dir / f"filter_{video_id}.txt"

    with open(output_file, 'w', encoding='utf-8') as f:
        for narr in narrations:
            line = f"{narr['unique_narration_id']} | {narr['narration']}\n"
            f.write(line)

    # Create empty response file for user to paste Gemini output
    response_file = output_dir / f"filter_{video_id}_response.txt"
    response_file.touch()

    return output_file


def load_filter_response(response_file: Path) -> Optional[Set[str]]:
    """
    Load relevant narration IDs from filter response.

    Expected format (narration IDs):
    [
        {"label": "...", "narration_ids": ["P01-20240203-150506-1", "P01-20240203-150506-3", ...]},
        ...
    ]

    Legacy format (line numbers - deprecated):
    [
        {"label": "...", "relevant_lines": [1, 2, 3, ...]},
        ...
    ]

    Returns set of narration IDs (strings).
    """
    if not response_file.exists():
        return None

    with open(response_file, 'r', encoding='utf-8') as f:
        content = f.read().strip()

    if not content:
        return None

    try:
        data = json.loads(content)
        if isinstance(data, list):
            if len(data) > 0 and isinstance(data[0], dict):
                ids = set()
                for group in data:
                    # New format: narration_ids (preferred)
                    if 'narration_ids' in group:
                        ids.update(group['narration_ids'])
                    # Legacy format: relevant_lines (line numbers)
                    elif 'relevant_lines' in group:
                        # Convert line numbers to strings for compatibility
                        # Caller will need to handle this
                        for ln in group['relevant_lines']:
                            ids.add(f"LINE:{ln}")
                return ids
            else:
                # Plain list format
                return set(str(x) for x in data)
    except json.JSONDecodeError:
        print(f"WARNING: Could not parse JSON from {response_file}")
        return None

    return None


def extract_filtered_lines(video_id: str, input_dir: Path, output_dir: Path, csv_path: Path = None) -> Optional[Path]:
    """
    Extract filtered lines from filter input based on response.

    Reads:
        - filter_{video_id}.txt (all narrations)
        - filter_{video_id}_response.txt (JSON with narration_ids)
        - CSV file (fallback for narration IDs not in filter file)

    Writes:
        - filtered_{video_id}.txt (only relevant lines, same format)
    """
    input_file = input_dir / f"filter_{video_id}.txt"
    response_file = input_dir / f"filter_{video_id}_response.txt"

    if not response_file.exists():
        print(f"ERROR: Response file not found: {response_file}")
        return None

    # Load relevant IDs
    relevant_ids = load_filter_response(response_file)

    if relevant_ids is None:
        print(f"ERROR: Could not parse response file: {response_file}")
        return None

    if not relevant_ids:
        print(f"WARNING: No relevant IDs found in response")
        return None

    # Build narration_id -> line mapping from filter file (if exists)
    id_to_line = {}
    if input_file.exists():
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if ' | ' in line:
                    narr_id = line.split(' | ')[0].strip()
                    id_to_line[narr_id] = line

    # Also load from pickle for fallback
    if csv_path is None:
        try:
            df = load_pickle_data()
            for _, row in df.iterrows():
                narr_id = row['unique_narration_id']
                if narr_id not in id_to_line:
                    id_to_line[narr_id] = f"{narr_id} | {row['narration'].strip()}"
        except FileNotFoundError:
            pass  # No pickle available
    elif csv_path.exists():
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                narr_id = row['unique_narration_id']
                if narr_id not in id_to_line:
                    id_to_line[narr_id] = f"{narr_id} | {row['narration'].strip()}"

    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"filtered_{video_id}.txt"

    extracted_count = 0
    warnings = 0

    with open(output_file, 'w', encoding='utf-8') as f:
        for narr_id in sorted(relevant_ids):
            if narr_id in id_to_line:
                f.write(id_to_line[narr_id] + '\n')
                extracted_count += 1
            else:
                if warnings < 5:
                    print(f"  WARNING: Narration ID not found: {narr_id}")
                warnings += 1
        if warnings > 5:
            print(f"  ... and {warnings - 5} more warnings")

    print(f"Extracted {extracted_count} lines")
    return output_file


def export_for_inventory(
    video_ids: List[str],
    participant: str,
    input_dir: Path,
    output_dir: Path
) -> Optional[Path]:
    """
    Export filtered food narrations for inventory discovery.

    Uses Task 1 response files to get filtered narration IDs,
    then looks up full narration text from CSV or pickle.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    all_narrations = []
    videos_processed = []
    videos_skipped = []

    # Input dir is per-participant
    participant_input_dir = input_dir / participant

    for video_id in video_ids:
        # Load filter response for this video
        response_file = participant_input_dir / f"filter_{video_id}_response.txt"
        filtered_ids = load_filter_response(response_file)

        if filtered_ids is None:
            videos_skipped.append(video_id)
            continue

        # Load narrations
        narrations = load_narrations_for_video(video_id, participant)

        # Filter to only food-related narrations
        for narr in narrations:
            if narr['unique_narration_id'] in filtered_ids:
                narr['video_id'] = video_id
                all_narrations.append(narr)

        videos_processed.append(video_id)

    if not all_narrations:
        print("ERROR: No filtered narrations found")
        return None

    # Determine output filename
    if len(video_ids) == 1:
        output_file = output_dir / f"inventory_{video_ids[0]}.txt"
    else:
        output_file = output_dir / f"inventory_{video_ids[0]}_to_{video_ids[-1]}.txt"

    # Write to TXT with video headers
    with open(output_file, 'w', encoding='utf-8') as f:
        current_video = None
        for narr in all_narrations:
            if narr['video_id'] != current_video:
                if current_video is not None:
                    f.write("\n")
                current_video = narr['video_id']
                f.write(f"=== VIDEO: {current_video} ===\n")

            line = f"{narr['unique_narration_id']} | {narr['narration']}\n"
            f.write(line)

    return output_file, videos_processed, videos_skipped, len(all_narrations)


def main():
    parser = argparse.ArgumentParser(
        description="Export narrations to TXT for Gemini web processing"
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Filter command
    filter_parser = subparsers.add_parser(
        'filter',
        help='Filter food-related narrations using GPT-5.2'
    )
    filter_parser.add_argument(
        '--video-id',
        default=None,
        help='Video ID to filter (e.g., P01-20240202-110250)'
    )
    filter_parser.add_argument(
        '--participant',
        default=None,
        help='Participant ID (P01-P09). Inferred from video-id if not specified.'
    )
    filter_parser.add_argument(
        '--all',
        action='store_true',
        help='Filter all videos for the participant'
    )
    filter_parser.add_argument(
        '--output-dir',
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help='Base output directory for TXT files'
    )
    filter_parser.add_argument(
        '--reasoning-effort',
        default='high',
        choices=['low', 'medium', 'high'],
        help='GPT-5.2 reasoning effort level (default: high)'
    )
    filter_parser.add_argument(
        '--skip-api',
        action='store_true',
        help='Skip GPT API call, use existing response file'
    )
    filter_parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Print detailed progress'
    )

    # Extract command
    extract_parser = subparsers.add_parser(
        'extract',
        help='Extract filtered lines based on Gemini response'
    )
    extract_parser.add_argument(
        '--video-id',
        default=None,
        help='Video ID to extract (e.g., P01-20240202-110250)'
    )
    extract_parser.add_argument(
        '--all',
        action='store_true',
        help='Process all non-empty filter responses'
    )
    extract_parser.add_argument(
        '--input-dir',
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help='Directory containing filter and response files'
    )
    extract_parser.add_argument(
        '--output-dir',
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help='Output directory for filtered file'
    )

    # Inventory command
    inventory_parser = subparsers.add_parser(
        'inventory',
        help='Export filtered food narrations for inventory discovery'
    )
    inventory_parser.add_argument(
        '--video-id',
        default=None,
        help='Single video ID to export'
    )
    inventory_parser.add_argument(
        '--start-video',
        default=None,
        help='Start video ID for range export'
    )
    inventory_parser.add_argument(
        '--end-video',
        default=None,
        help='End video ID for range export'
    )
    inventory_parser.add_argument(
        '--participant',
        default='P01',
        help='Participant ID (P01-P09)'
    )
    inventory_parser.add_argument(
        '--input-dir',
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help='Directory containing filter response files'
    )
    inventory_parser.add_argument(
        '--output-dir',
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help='Output directory for TXT files'
    )

    # List command (helper)
    list_parser = subparsers.add_parser(
        'list',
        help='List all available video IDs'
    )
    list_parser.add_argument(
        '--participant',
        default=None,
        help='Participant ID to list videos for (P01-P09). Lists all if not specified.'
    )

    args = parser.parse_args()

    if args.command == 'filter':
        # Determine participant
        if args.participant:
            participant = args.participant
        elif args.video_id:
            participant = args.video_id.split('-')[0]
        else:
            print("ERROR: Specify --video-id or --participant with --all")
            return

        # Get videos to process
        if args.all:
            video_ids = get_all_video_ids(participant)
            print(f"Filtering all {len(video_ids)} videos for participant {participant}")
        elif args.video_id:
            video_ids = [args.video_id]
        else:
            print("ERROR: Specify --video-id or --all")
            return

        # Output directory (per-participant)
        output_dir = args.output_dir / participant
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Participant: {participant}")
        print(f"Output: {output_dir}")
        print()

        # Process each video
        total_filtered = 0
        total_narrations = 0

        for video_id in video_ids:
            print(f"Processing {video_id}...")

            # Load narrations
            narrations = load_narrations_for_video(video_id, participant)

            if not narrations:
                print(f"  WARNING: No narrations found for video {video_id}")
                continue

            response_file = output_dir / f"filter_{video_id}_response.txt"

            if args.skip_api:
                # Use existing response file
                if not response_file.exists() or response_file.stat().st_size == 0:
                    print(f"  SKIPPED: No existing response file")
                    continue

                print(f"  Using existing response file")
                with open(response_file, 'r', encoding='utf-8') as f:
                    raw_response = f.read()
                filtered_ids = parse_filter_response(raw_response, narrations)
            else:
                # Call GPT-5.2 API
                print(f"  Calling GPT-5.2 (reasoning_effort={args.reasoning_effort})...")
                raw_response, filtered_ids = filter_with_gpt(
                    narrations,
                    reasoning_effort=args.reasoning_effort,
                    verbose=args.verbose
                )

                # Save raw response for debugging/audit
                raw_response_file = output_dir / f"filter_{video_id}_raw.txt"
                with open(raw_response_file, 'w', encoding='utf-8') as f:
                    f.write(raw_response)

            # Export the input TXT (for reference)
            input_file = output_dir / f"filter_{video_id}.txt"
            with open(input_file, 'w', encoding='utf-8') as f:
                for idx, narr in enumerate(narrations):
                    f.write(f"{idx} | {narr['unique_narration_id']} | {narr['narration']}\n")

            # Save filtered output (for human review)
            filtered_output = output_dir / f"filtered_{video_id}.txt"
            with open(filtered_output, 'w', encoding='utf-8') as f:
                for narr in narrations:
                    if narr['unique_narration_id'] in filtered_ids:
                        f.write(f"{narr['unique_narration_id']} | {narr['narration']}\n")

            # Save response in narration_ids format (for downstream compatibility)
            response_with_ids = [{"label": "filtered", "narration_ids": sorted(list(filtered_ids))}]
            with open(response_file, 'w', encoding='utf-8') as f:
                json.dump(response_with_ids, f, indent=2)

            pct = len(filtered_ids) / len(narrations) * 100 if narrations else 0
            print(f"  Filtered {len(filtered_ids)} / {len(narrations)} ({pct:.1f}%)")

            total_filtered += len(filtered_ids)
            total_narrations += len(narrations)

        # Summary
        if len(video_ids) > 1:
            print(f"\n{'='*60}")
            pct = total_filtered / total_narrations * 100 if total_narrations else 0
            print(f"Total: {total_filtered} / {total_narrations} narrations ({pct:.1f}%)")

    elif args.command == 'extract':
        if args.all:
            # Find all non-empty response files
            import re
            response_files = list(args.input_dir.glob("filter_*_response.txt"))
            video_ids = []
            for rf in response_files:
                if rf.stat().st_size > 0:
                    match = re.search(r'filter_(.+)_response\.txt', rf.name)
                    if match:
                        video_ids.append(match.group(1))

            video_ids.sort()
            print(f"Processing {len(video_ids)} videos with non-empty filter responses...")
            print(f"Input: {args.input_dir}")
            print(f"Output: {args.output_dir}\n")

            success = 0
            failed = 0
            for vid in video_ids:
                print(f"  {vid}...", end=" ", flush=True)
                output_file = extract_filtered_lines(vid, args.input_dir, args.output_dir)
                if output_file:
                    success += 1
                else:
                    failed += 1

            print(f"\nDone: {success} succeeded, {failed} failed")

        elif args.video_id:
            print(f"Extracting filtered lines for video: {args.video_id}")
            print(f"Input: {args.input_dir}")
            print(f"Output: {args.output_dir}")

            output_file = extract_filtered_lines(args.video_id, args.input_dir, args.output_dir)

            if output_file:
                print(f"\nFiltered narrations written to:")
                print(f"  {output_file}")

        else:
            print("ERROR: Specify --video-id or --all")

    elif args.command == 'inventory':
        # Determine video IDs to process
        participant = args.participant
        all_video_ids = get_all_video_ids(participant)

        if args.video_id:
            # Single video
            video_ids = [args.video_id]
        elif args.start_video and args.end_video:
            # Video range
            try:
                start_idx = all_video_ids.index(args.start_video)
                end_idx = all_video_ids.index(args.end_video)
                video_ids = all_video_ids[start_idx:end_idx + 1]
            except ValueError as e:
                print(f"ERROR: Video not found - {e}")
                return
        else:
            print("ERROR: Specify --video-id OR --start-video and --end-video")
            return

        print(f"Exporting filtered narrations for inventory discovery")
        print(f"Participant: {participant}")
        print(f"Videos: {len(video_ids)} ({video_ids[0]} to {video_ids[-1]})")
        print(f"Input (filter responses): {args.input_dir}/{participant}")
        print(f"Output: {args.output_dir}")

        result = export_for_inventory(video_ids, participant, args.input_dir, args.output_dir)

        if result:
            output_file, processed, skipped, total_narrations = result

            print(f"\nProcessed {len(processed)} videos, skipped {len(skipped)}")
            if skipped:
                print(f"  Skipped (no filter response): {', '.join(skipped)}")

            print(f"\nExported {total_narrations} filtered narrations to:")
            print(f"  {output_file}")
            print(f"\nPaste contents into Gemini web interface for inventory discovery.")

    elif args.command == 'list':
        if args.participant:
            video_ids = get_all_video_ids(args.participant)
            print(f"Found {len(video_ids)} videos for {args.participant}:\n")
        else:
            # List all participants with counts
            print("Available participants:\n")
            df = load_pickle_data()
            for p in ['P01', 'P02', 'P03', 'P04', 'P05', 'P06', 'P07', 'P08', 'P09']:
                p_df = df[df['video_id'].str.startswith(p)]
                if len(p_df) > 0:
                    videos = p_df['video_id'].nunique()
                    print(f"  {p}: {len(p_df):,} narrations across {videos} videos")
            print("\nUse --participant P01 to list specific videos")
            return

        for vid in video_ids:
            print(f"  {vid}")

    else:
        parser.print_help()


if __name__ == '__main__':
    main()

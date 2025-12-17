#!/usr/bin/env python3
"""
Filter Narrations with GPT-5

Uses GPT-5 to segment kitchen activity narrations and identify food-related segments.

Usage:
    python filter_narrations_gpt5.py --video-id P01-20240202-110250
    python filter_narrations_gpt5.py --all  # Process all filter_*.txt files without responses

Input:
    gemini_outputs/gemini_input/filter_{video_id}.txt

Output:
    gemini_outputs/gemini_input/filter_{video_id}_response.txt
"""

import sys
import json
import argparse
from pathlib import Path

# Add llm-api to path
_SCRIPT_DIR = Path(__file__).parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent  # FoodMemory/
sys.path.insert(0, str(_PROJECT_ROOT / "llm-api"))

from openai import AzureOpenAI
from dotenv import load_dotenv
import os

# Load environment variables
env_path = _PROJECT_ROOT / '.env'
load_dotenv(dotenv_path=env_path)

# Default paths
DEFAULT_INPUT_DIR = _SCRIPT_DIR.parent / "gemini_outputs" / "gemini_input"

# The prompt for GPT-5
SEGMENTER_PROMPT = '''You are a "Kitchen Activity Segmenter."
Your goal is to process a raw narration log and produce a structured list of **FOOD-ONLY** activity blocks.

INPUT:
A list of narrtion IDs and narrations.

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
- **Isolated Maintenance:** Cleaning, washing, or arranging items that doensn't contain food.
- **Passive Actions:** Movements or observations that have no effect on food state.

### PART 2: THE GROUPER (Segmentation)
Group the "KEEP" lines into cohesive semantic blocks.
- **Cluster:** Combine consecutive relevant lines into a single Event (e.g., "Making Coffee", "Chopping Salad").
- **Gaps:** If there is a long period (e.g., >30s) of "DISCARD" lines between actions, split them into separate blocks.

### OUTPUT FORMAT
Return a JSON list of segments.
**CRITICAL:** If a time range contains NO relevant food actions, DO NOT create a block. Skip it entirely.

[{
 "label": "Pouring Milk",
"relevant_lines": [15, 16, 17, 18] // The specific indices that passed the filter},...
]
'''


def get_client():
    """Initialize Azure OpenAI client for GPT-5."""
    api_key = os.getenv("AZURE_OPENAI_API_KEY_2")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT_2")

    if not api_key or not endpoint:
        raise ValueError(
            "Missing Azure OpenAI credentials. "
            "Set AZURE_OPENAI_API_KEY_2 and AZURE_OPENAI_ENDPOINT_2 in .env"
        )

    return AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version="2025-03-01-preview",
    )


def filter_narrations(
    video_id: str,
    input_dir: Path,
    model: str = "gpt-5",
    reasoning_effort: str = "low",
    verbosity: str = "low",
    force: bool = False
) -> Path:
    """
    Filter narrations for a single video using GPT-5.

    Args:
        video_id: Video identifier (e.g., P01-20240202-110250)
        input_dir: Directory containing filter_*.txt files
        model: Model to use (gpt-5, gpt-5.1, etc.)
        reasoning_effort: "low", "medium", or "high"
        verbosity: "low", "medium", or "high"
        force: Overwrite existing response file

    Returns:
        Path to the output response file
    """
    input_file = input_dir / f"filter_{video_id}.txt"
    output_file = input_dir / f"filter_{video_id}_response.txt"

    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    if output_file.exists() and not force:
        print(f"  Skipping {video_id} - response already exists (use --force to overwrite)")
        return output_file

    # Read narrations
    with open(input_file, 'r', encoding='utf-8') as f:
        narrations = f.read()

    print(f"  Input: {len(narrations.splitlines())} lines")

    # Build full input
    full_input = f"{SEGMENTER_PROMPT}\n\n# Narrations:\n{narrations}"

    # Call GPT-5
    client = get_client()
    print(f"  Calling {model}...")

    result = client.responses.create(
        model=model,
        input=full_input,
        reasoning={"effort": reasoning_effort},
        text={"verbosity": verbosity},
    )

    response_text = result.output_text

    # Save response
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(response_text)


    print(f"  Saved: {output_file.name}")
    return output_file


def find_videos_without_response(input_dir: Path) -> list:
    """Find all filter_*.txt files that don't have corresponding response files."""
    videos = []
    for f in input_dir.glob("filter_P*.txt"):
        if "_response" not in f.name:
            video_id = f.stem.replace("filter_", "")
            response_file = input_dir / f"filter_{video_id}_response.txt"
            if not response_file.exists():
                videos.append(video_id)
    return sorted(videos)


def main():
    parser = argparse.ArgumentParser(
        description="Filter kitchen narrations using GPT-5"
    )
    parser.add_argument(
        '--video-id',
        help='Video ID to process (e.g., P01-20240202-110250)'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Process all filter_*.txt files without responses'
    )
    parser.add_argument(
        '--input-dir',
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help='Directory containing filter_*.txt files'
    )
    parser.add_argument(
        '--model',
        default='gpt-5',
        choices=['gpt-5', 'gpt-5.1', 'gpt-5-chat'],
        help='Model to use (default: gpt-5)'
    )
    parser.add_argument(
        '--reasoning',
        default='high',
        choices=['low', 'medium', 'high'],
        help='Reasoning effort level (default: low)'
    )
    parser.add_argument(
        '--verbosity',
        default='low',
        choices=['low', 'medium', 'high'],
        help='Output verbosity level (default: low)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Overwrite existing response files'
    )

    args = parser.parse_args()

    if not args.video_id and not args.all:
        print("ERROR: Must specify --video-id or --all")
        parser.print_help()
        return

    print("=" * 60)
    print("FILTER NARRATIONS WITH GPT-5")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Input dir: {args.input_dir}")

    if args.all:
        video_ids = find_videos_without_response(args.input_dir)
        if not video_ids:
            print("\nNo videos found without responses.")
            return
        print(f"\nFound {len(video_ids)} videos to process")
    else:
        video_ids = [args.video_id]

    for video_id in video_ids:
        print(f"\n[{video_id}]")
        try:
            filter_narrations(
                video_id=video_id,
                input_dir=args.input_dir,
                model=args.model,
                reasoning_effort=args.reasoning,
                verbosity=args.verbosity,
                force=args.force
            )
        except Exception as e:
            print(f"  ERROR: {e}")

    print(f"\n{'=' * 60}")
    print("DONE")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Sequential Knowledge Graph Update Pipeline using Ollama
Processes narrations one-by-one, querying KG state before each LLM call.
"""

import pandas as pd
import argparse
import json
import time
from pathlib import Path
from typing import Dict, Optional, List
from ollama import Client

# Import our custom modules
from kg_storage import (
    load_kg, save_kg, find_food, add_food_node, update_food_node,
    add_interaction, get_or_create_zone, get_food_summary
)
from entity_extractor import extract_narration_info
from llm_context import (
    build_kg_update_prompt, parse_llm_response,
    validate_update_command
)
from kg_update_executor import execute_kg_update
from kg_snapshots import KGSnapshotManager


def call_ollama_for_kg_update(
    client: Client,
    model: str,
    narration_info: Dict,
    existing_food: Optional[Dict],
    kg: Dict,
    max_retries: int = 3,
    verbose: bool = False
) -> Optional[Dict]:
    """
    Call Ollama LLM to generate KG update command.

    Args:
        client: Ollama client
        model: Model name
        narration_info: Extracted narration information
        existing_food: Current state of food node if it exists
        kg: Full knowledge graph
        max_retries: Number of retry attempts
        verbose: Print detailed error information

    Returns:
        Parsed update command dict or None if failed
    """
    # Build prompt with current KG context
    messages = build_kg_update_prompt(narration_info, existing_food, kg)

    # Extract user message content (messages[1] is user message)
    prompt = messages[1]["content"]

    if verbose:
        print(f"\n  --- LLM CONTEXT ---")
        print(f"  System: {messages[0]['content'][:100]}...")
        print(f"  User prompt (first 500 chars):\n{prompt[:500]}...")

    last_error = None
    last_response = None

    for attempt in range(max_retries):
        try:
            # Call Ollama with simplified message format
            resp = client.chat(
                model=model,
                messages=[
                    {"role": "system", "content": messages[0]["content"]},
                    {"role": "user", "content": prompt}
                ],
                options={
                    "num_gpu": -1,  # Use all available GPUs
                    "num_thread": 8,  # Optimize threading
                    "temperature": 0.1,  # Lower temperature for consistency
                }
            )

            response_text = resp["message"]["content"]
            last_response = response_text

            if verbose:
                print(f"\n  --- LLM RESPONSE ---")
                print(f"  {response_text[:800]}...")

            # Parse and validate response
            update_command = parse_llm_response(response_text)
            if update_command:
                is_valid, error_msg = validate_update_command(update_command)
                if is_valid:
                    if verbose:
                        print(f"\n  --- PARSED UPDATE COMMAND ---")
                        print(f"  Update type: {update_command['update_type']}")
                        print(f"  Target food: {update_command.get('target_food_id', 'None')}")
                        if update_command.get('new_food_info'):
                            print(f"  New food: {update_command['new_food_info']}")
                        if update_command.get('updates'):
                            print(f"  Updates: {update_command['updates']}")
                    return update_command
                else:
                    last_error = f"Invalid update command: {error_msg}"
                    if verbose:
                        print(f"  Warning: Invalid update command (attempt {attempt + 1}): {error_msg}")
                        print(f"    Response: {response_text[:200]}")
            else:
                last_error = "Failed to parse LLM response"
                if verbose:
                    print(f"  Warning: Failed to parse LLM response (attempt {attempt + 1})")
                    print(f"    Response: {response_text[:200]}")

        except Exception as e:
            last_error = str(e)
            if verbose:
                print(f"  Warning: Error calling Ollama (attempt {attempt + 1}): {e}")
                import traceback
                traceback.print_exc()

        if attempt < max_retries - 1:
            time.sleep(1.0)  # Brief pause before retry

    # Log final failure details (always print this, not just in verbose mode)
    print(f"  ✗ All {max_retries} attempts failed. Last error: {last_error}")
    if verbose and last_response:
        print(f"    Last response: {last_response[:300]}")

    return None


def process_narration_sequential(
    row: Dict,
    kg: Dict,
    client: Client,
    model: str,
    snapshot_mgr: Optional[KGSnapshotManager] = None,
    verbose: bool = False,
    use_llm_extraction: bool = False
) -> bool:
    """
    Process a single narration row sequentially:
    1. Query current KG state
    2. Build prompt with context
    3. Call Ollama LLM
    4. Update KG
    5. Save snapshot

    Args:
        row: CSV row as dictionary
        kg: Knowledge graph (will be updated in-place)
        client: Ollama client
        model: Model name
        snapshot_mgr: Optional snapshot manager for saving KG states
        verbose: Print detailed information
        use_llm_extraction: Use LLM for entity extraction instead of keywords

    Returns:
        True if processed successfully, False otherwise
    """
    # Step 1: Extract entities and narration info
    if use_llm_extraction:
        from llm_entity_extractor import extract_narration_info_with_llm
        narration_info = extract_narration_info_with_llm(client, model, row)
        if verbose and narration_info.get('llm_reasoning'):
            print(f"  LLM extraction: {narration_info['llm_reasoning'][:80]}...")
    else:
        narration_info = extract_narration_info(row)

    if verbose:
        print(f"\n{'=' * 80}")
        print(f"Processing: {narration_info['narration_id']}")
        print(f"  Time: {narration_info['start_time']:.1f}s - {narration_info['end_time']:.1f}s")
        print(f"  Narration: {narration_info['narration'][:80]}...")
        print(f"  Food: {narration_info['food_entity']}")
        print(f"  Location: {narration_info['location_entity']}")
        print(f"  Action: {narration_info['primary_action']}")

    # Check if this narration involves food
    if not narration_info['food_entity']:
        if verbose:
            print("  → Skipping: No food entity found")

        # Save snapshot even for skipped narrations
        if snapshot_mgr:
            snapshot_mgr.save_snapshot(
                kg=kg,
                narration_id=narration_info['narration_id'],
                video_id=narration_info['video_id'],
                start_time=narration_info['start_time'],
                end_time=narration_info['end_time'],
                narration_text=narration_info['narration'],
                success=False,
                reason="No food entity found"
            )

        return False

    # Step 2: Query KG for matching food (CRITICAL: query current state!)
    food_name = narration_info['food_entity']

    if verbose:
        print(f"\n  --- KG RETRIEVAL ---")
        print(f"  Searching for: '{food_name}'")

    # Find all foods matching the name pattern (ignoring location/zone)
    matching_foods = find_food(kg, name_pattern=food_name)

    existing_food = None
    if matching_foods:
        if verbose:
            print(f"  Found {len(matching_foods)} matching food(s):")
            for i, food in enumerate(matching_foods):
                last_time = food['interaction_history'][-1]['end_time'] if food.get('interaction_history') else 0
                print(f"    {i+1}. {food['food_id']} - name: '{food['name']}', last interaction: {last_time:.1f}s")

        # Sort by most recent interaction
        matching_foods.sort(
            key=lambda f: f['interaction_history'][-1]['end_time'] if f.get('interaction_history') else 0,
            reverse=True
        )
        existing_food = matching_foods[0]

        if verbose:
            print(f"  → Selected most recent: {existing_food['food_id']} (location: {existing_food.get('location', 'unknown')})")
    else:
        if verbose:
            print(f"  → No existing food found")

    # Step 3: Call Ollama LLM with current KG context
    update_command = call_ollama_for_kg_update(
        client, model, narration_info, existing_food, kg, verbose=verbose
    )

    if not update_command:
        print(f"  Error: Failed to get valid update command from LLM")

        # Save snapshot for failed LLM call
        if snapshot_mgr:
            snapshot_mgr.save_snapshot(
                kg=kg,
                narration_id=narration_info['narration_id'],
                video_id=narration_info['video_id'],
                start_time=narration_info['start_time'],
                end_time=narration_info['end_time'],
                narration_text=narration_info['narration'],
                success=False,
                reason="LLM call failed"
            )

        return False

    if verbose:
        print(f"  → Update type: {update_command['update_type']}")

    # Step 4: Execute KG update
    success = execute_kg_update(kg, update_command, narration_info, verbose)

    # Step 5: Save snapshot AFTER update
    if snapshot_mgr:
        snapshot_mgr.save_snapshot(
            kg=kg,
            narration_id=narration_info['narration_id'],
            video_id=narration_info['video_id'],
            start_time=narration_info['start_time'],
            end_time=narration_info['end_time'],
            narration_text=narration_info['narration'],
            success=success
        )

    return success


def main():
    parser = argparse.ArgumentParser(description='Sequential KG pipeline using Ollama')
    parser.add_argument('--csv', '-c', required=True,
                        help='Path to narration CSV file')
    parser.add_argument('--kg', '-k', default='food_kg_sequential.json',
                        help='Path to knowledge graph JSON file (default: food_kg_sequential.json)')
    parser.add_argument('--snapshots', default='kg_snapshots',
                        help='Directory for KG snapshots (default: kg_snapshots)')
    parser.add_argument('--model', '-m', default='gpt-oss:120b',
                        help='Ollama model name (default: gpt-oss:120b)')
    parser.add_argument('--host', default='http://localhost:11434',
                        help='Ollama host URL (default: http://localhost:11434)')
    parser.add_argument('--limit', '-l', type=int,
                        help='Limit number of rows to process (for testing)')
    parser.add_argument('--start', '-s', type=int, default=0,
                        help='Start row index (default: 0)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Print detailed processing information')
    parser.add_argument('--save-interval', type=int, default=10,
                        help='Save KG every N rows (default: 10)')
    parser.add_argument('--entity-extraction', choices=['keyword', 'llm'], default='llm',
                        help='Entity extraction method: keyword (fast) or llm (accurate, slower) (default: llm)')

    args = parser.parse_args()

    use_llm_extraction = (args.entity_extraction == 'llm')

    # Load CSV
    print(f"Loading narration CSV from {args.csv}")
    df = pd.read_csv(args.csv)
    print(f"  Total rows: {len(df)}")

    # Apply filtering
    if args.limit:
        df = df.iloc[args.start:args.start + args.limit]
        print(f"  Processing rows {args.start} to {args.start + len(df)}")
    elif args.start > 0:
        df = df.iloc[args.start:]
        print(f"  Processing from row {args.start}")

    # Load or create KG
    print(f"\nLoading knowledge graph from {args.kg}")
    kg = load_kg(args.kg)
    print(f"  Current foods: {len(kg.get('foods', {}))}")
    print(f"  Current zones: {len(kg.get('zones', {}))}")

    # Initialize snapshot manager
    snapshot_mgr = KGSnapshotManager(args.snapshots)
    print(f"\nSnapshot directory: {args.snapshots}")

    # Initialize Ollama client
    print(f"\nInitializing Ollama client...")
    print(f"  Host: {args.host}")
    print(f"  Model: {args.model}")
    print(f"  Entity extraction: {args.entity_extraction} {'(slower, more accurate)' if use_llm_extraction else '(faster)'}")

    try:
        client = Client(host=args.host)
        # Test connection
        client.list()
        print(f"  ✓ Connection successful")
    except Exception as e:
        print(f"  Error: Failed to connect to Ollama: {e}")
        print(f"  Make sure Ollama is running on {args.host}")
        return

    # Process each row SEQUENTIALLY
    print(f"\n{'=' * 80}")
    print("STARTING SEQUENTIAL PROCESSING")
    print(f"{'=' * 80}\n")

    processed_count = 0
    success_count = 0
    start_time = time.time()

    for idx, row in df.iterrows():
        row_dict = row.to_dict()

        # Process this narration with current KG state
        success = process_narration_sequential(
            row_dict, kg, client, args.model, snapshot_mgr, args.verbose, use_llm_extraction
        )

        processed_count += 1
        if success:
            success_count += 1

        # Periodic save and progress report
        if processed_count % args.save_interval == 0:
            save_kg(kg, args.kg)
            elapsed = time.time() - start_time
            rate = processed_count / elapsed if elapsed > 0 else 0
            eta = (len(df) - processed_count) / rate if rate > 0 else 0

            print(f"\nProgress: {processed_count}/{len(df)} rows "
                  f"({success_count} successful, {processed_count - success_count} skipped/failed)")
            print(f"  Rate: {rate:.2f} rows/sec")
            print(f"  ETA: {eta/60:.1f} minutes")
            print(f"  Foods in KG: {len(kg['foods'])}")

    # Final save
    save_kg(kg, args.kg)

    # Summary
    elapsed = time.time() - start_time
    print(f"\n{'=' * 80}")
    print(f"PROCESSING COMPLETE")
    print(f"{'=' * 80}")
    print(f"Total rows processed: {processed_count}")
    print(f"Successful updates: {success_count}")
    print(f"Skipped/Failed: {processed_count - success_count}")
    print(f"Time elapsed: {elapsed:.1f}s ({processed_count/elapsed:.2f} rows/sec)")
    print(f"\n{get_food_summary(kg)}")
    print(f"\nKnowledge graph saved to: {args.kg}")
    print(f"Snapshots saved to: {args.snapshots}/")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
VLM-Based Food State Tracking for Annotation Benchmark Creation

This script processes annotation task blocks by:
1. Loading the block's video clip
2. Sending to Qwen3-VL with:
   - Current food memory state
   - State taxonomy schema
   - Block narrations
3. Receiving updated food memory with state changes
4. Populating the annotation task with VLM predictions

This creates the benchmark dataset with VLM pre-populated states.
"""

import base64
import requests
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional
import argparse


# Qwen3-VL API endpoint
QWEN3VL_URL = "http://saltyfish.eecs.umich.edu:8000/v1/chat/completions"


def load_state_taxonomy(taxonomy_path: Path) -> Dict:
    """Load the food state taxonomy schema"""
    with open(taxonomy_path, 'r') as f:
        return json.load(f)


def load_annotation_tasks(tasks_file: Path) -> List[Dict]:
    """Load annotation tasks JSON"""
    with open(tasks_file, 'r') as f:
        return json.load(f)


def encode_video_base64(video_path: Path) -> str:
    """Encode video file to base64"""
    with open(video_path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def generate_instance_id(food_memory: Dict, food_noun: str, semantic_descriptor: str) -> str:
    """
    Generate unique instance ID with format: {food_noun}_{semantic_descriptor}_{counter}

    Args:
        food_memory: Current memory dict
        food_noun: Base food noun (e.g., "flour")
        semantic_descriptor: VLM-provided descriptor (e.g., "in_bowl", "from_bag")

    Returns:
        Unique instance ID (e.g., "flour_in_bowl_002")
    """
    # Extract all existing counters for this food_noun
    existing_counters = []
    prefix = f"{food_noun}_"

    for instance_id in food_memory.keys():
        if instance_id.startswith(prefix):
            # Extract counter from instance_id (last segment if numeric)
            parts = instance_id.rsplit('_', 1)
            if len(parts) == 2 and parts[1].isdigit():
                existing_counters.append(int(parts[1]))

    # Find next available counter
    next_counter = max(existing_counters, default=0) + 1

    # Format with zero-padding (001, 002, etc.)
    # Dynamic padding if counter exceeds 999
    digits = max(3, len(str(next_counter)))
    instance_id = f"{food_noun}_{semantic_descriptor}_{next_counter:0{digits}d}"

    return instance_id


def retrieve_relevant_instances(food_memory: Dict, target_food_nouns: List[str]) -> Dict:
    """
    Simple retrieval: Pull ALL instances for target food nouns.
    VLM decides which instance to update or if new instance needed.

    Args:
        food_memory: Current food memory dict
        target_food_nouns: List of food nouns in current block

    Returns:
        Filtered dict of relevant instances
    """
    relevant_instances = {}

    for instance_id, instance_data in food_memory.items():
        food_noun = instance_data.get('food_noun')

        # Pull all instances of target foods
        if food_noun in target_food_nouns:
            # Skip consumed instances (lifecycle rule)
            quantity = instance_data.get('state', {}).get('consumption_state', {}).get('quantity')
            if quantity != 'consumed':
                relevant_instances[instance_id] = instance_data

    return relevant_instances


def format_instance_memory(memory: Dict, target_foods: List[str]) -> str:
    """
    Format instances grouped by food noun for readability in VLM prompt

    Args:
        memory: Filtered instances dict
        target_foods: List of target food nouns

    Returns:
        Formatted string for VLM prompt
    """
    output = []

    # Group instances by food noun
    by_food = {}
    for instance_id, data in memory.items():
        food_noun = data.get('food_noun')
        if food_noun not in by_food:
            by_food[food_noun] = []
        by_food[food_noun].append((instance_id, data))

    # Format each food's instances
    for food_noun in target_foods:
        if food_noun not in by_food:
            continue

        instances = by_food[food_noun]
        output.append(f"\nFor {food_noun} ({len(instances)} active instance{'s' if len(instances) > 1 else ''}):")

        for instance_id, data in instances:
            state = data.get('state', {})
            history = data.get('interaction_history', [])
            last_interaction = history[-1] if history else None

            # Compact state summary
            container = state.get('container_state', {}).get('container_type', 'unknown')
            location = state.get('location_state', {}).get('location_type', 'unknown')
            quantity = state.get('consumption_state', {}).get('quantity', 'unknown')
            form = state.get('preparation_state', {}).get('form_state', 'unknown')

            output.append(f'- Instance ID: "{instance_id}"')
            output.append(f'  State: container={container}, location={location}, quantity={quantity}, form={form}')

            if last_interaction:
                output.append(
                    f'  Last: [Block {last_interaction.get("block_id", "?")}, '
                    f'{last_interaction.get("timestamp", 0):.2f}s] '
                    f'"{last_interaction.get("narration", "")}" ({last_interaction.get("event_type", "unknown")})'
                )

            if 'parent_instance' in data:
                output.append(f'  Parent: {data["parent_instance"]}')

    return '\n'.join(output) if output else "No active instances"


def create_vlm_prompt(
    narrations: List[str],
    food_nouns: List[str],
    current_memory: Dict,
    state_schema: Dict
) -> str:
    """Create the prompt for VLM state tracking

    Args:
        narrations: List of narration texts in this block
        food_nouns: List of target food nouns in this block
        current_memory: Current food memory state
        state_schema: State taxonomy schema

    Returns:
        Formatted prompt string
    """
    # Retrieve all instances for target food items
    relevant_memory = retrieve_relevant_instances(current_memory, food_nouns)

    # Format memory for VLM readability
    memory_text = format_instance_memory(relevant_memory, food_nouns)

    # Generate state category descriptions dynamically from schema
    state_descriptions = []
    for category, fields in state_schema.items():
        for field_name, values in fields.items():
            # Get first few example values
            examples = ', '.join(values[:3])
            if len(values) > 3:
                examples += ", ..."

            # Create human-readable description
            if category == "container_state":
                desc = f"Container: What is the food in? (values: {examples})"
            elif category == "preparation_state":
                desc = f"Preparation: What form is the food? (values: {examples})"
            elif category == "consumption_state":
                desc = f"Quantity: How much remains? (values: {examples})"
            elif category == "location_state":
                desc = f"Location: Where is the food? (values: {examples})"
            else:
                desc = f"{category}: {field_name} (values: {examples})"

            state_descriptions.append(desc)

    state_descriptions_text = '\n   - '.join(state_descriptions)

    # Generate output format structure dynamically from schema
    output_structure = {}
    for category, fields in state_schema.items():
        output_structure[category] = {field: "..." for field in fields.keys()}

    # Build example output format based on whether we have existing instances
    if relevant_memory:
        # Show example with both existing and new instances
        example_existing_id = list(relevant_memory.keys())[0] if relevant_memory else "flour_from_bag_001"
        output_format_json = json.dumps({
            "food_memory": {
                example_existing_id: {
                    "food_noun": "flour",
                    "instance_id": example_existing_id,
                    "state": output_structure
                },
                "NEW_INSTANCE": {
                    "food_noun": "flour",
                    "semantic_name": "in_bowl",
                    "state": output_structure,
                    "parent_instance": example_existing_id
                }
            }
        }, indent=2)
    else:
        # First block: only show NEW_INSTANCE examples (multiple instances)
        output_format_json = json.dumps({
            "food_memory": {
                "NEW_INSTANCE": {
                    "food_noun": "lemon",
                    "semantic_name": "in_fridge",
                    "state": output_structure
                },
                "NEW_INSTANCE_2": {
                    "food_noun": "meat",
                    "semantic_name": "minced",
                    "state": output_structure
                }
            }
        }, indent=2)

    prompt = f"""You are analyzing an egocentric cooking video to track food item states.

**TASK**: Determine the current state of each food item at the END of this video clip.

**APPROACH**:
- PRIMARY SOURCE: Read the narration text carefully to understand what actions occurred
- SECONDARY SOURCE: Watch the video to VALIDATE the narrations and resolve any ambiguities
- Focus on what the narrations describe, using video as confirmation

**NARRATIONS** (what happened in this clip):
{chr(10).join(f"- {narr}" for narr in narrations)}

**TARGET FOOD ITEMS**: {', '.join(food_nouns)}

**CURRENT FOOD MEMORY** (all active instances for target items):
{memory_text}

**STATE TAXONOMY SCHEMA**:
```json
{json.dumps(state_schema, indent=2)}
```

**INSTRUCTIONS FOR INSTANCE MANAGEMENT**:

1. **Matching existing instances**:
   - Read the narration carefully
   - Determine which existing instance(s) the narration refers to
   - Update the state of matched instance(s) by using their full instance_id
   - Append to interaction_history (system will add automatically)

2. **Creating new instances**:
   - If narration describes a NEW physical instance (split, new preparation):
     - Use keys: "NEW_INSTANCE", "NEW_INSTANCE_2", "NEW_INSTANCE_3", etc.
     - Provide semantic_name from narration context (system appends counter)
     - Examples of semantic names: "in_bowl", "from_bag", "minced", "sliced", "on_plate"
   - For splits (pour, transfer, scoop):
     - Update source instance (reduce quantity if partial)
     - Create NEW_INSTANCE with semantic_name and parent_instance field

3. **Semantic name guidelines**:
   - Use snake_case format
   - Keep concise (1-3 words): descriptor only, not food noun
   - Descriptors can be:
     * Container-based: "in_bowl", "on_plate", "in_cup"
     * Preparation: "minced", "sliced", "chopped", "mixed"
     * Source: "from_bag", "from_fridge", "from_package"
     * Location: "on_counter", "in_oven"
   - System creates full instance_id as: {{food_noun}}_{{semantic_name}}_{{counter}}

4. **Consumption**:
   - If food is fully consumed/discarded, set quantity="consumed"
   - Instance will be removed from memory automatically

5. **State tracking**:
   - Use only values from the state taxonomy schema (exact matches required)
   - Focus on observable state changes:
     - {state_descriptions_text}

**OUTPUT FORMAT** (strict JSON):
```json
{output_format_json}
```

**IMPORTANT**:
- For existing instances, use their full instance_id as key
- For new instances, use keys "NEW_INSTANCE", "NEW_INSTANCE_2", "NEW_INSTANCE_3", etc.
- Each new instance must have semantic_name field
- System handles counter assignment to prevent duplicates
- VLM has full autonomy to decide: update existing OR create new
- Base your reasoning primarily on the narration text
- Return ONLY the JSON, no additional text

Return ONLY the JSON, no additional text.
"""
    return prompt


def query_qwen3vl(
    video_path: Path,
    prompt: str,
    fps: int = 1,
    max_tokens: int = 2000,
    temperature: float = 0.3
) -> Dict:
    """Query Qwen3-VL with video and prompt

    Args:
        video_path: Path to video clip
        prompt: Text prompt
        fps: Frames per second to sample
        max_tokens: Maximum response tokens
        temperature: Sampling temperature

    Returns:
        Parsed JSON response from VLM
    """
    print(f"  Encoding video: {video_path.name}")
    video_base64 = encode_video_base64(video_path)

    print(f"  Querying Qwen3-VL (fps={fps}, max_tokens={max_tokens})...")

    data = {
        "model": "Qwen/Qwen3-VL-30B-A3B-Instruct",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video_url",
                        "video_url": {
                            "url": f"data:video/mp4;base64,{video_base64}"
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "extra_body": {
            "mm_processor_kwargs": {
                "fps": fps,
                "do_sample_frames": True
            }
        }
    }

    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(QWEN3VL_URL, headers=headers, json=data, timeout=300)
        response.raise_for_status()

        result = response.json()

        if "choices" in result and len(result["choices"]) > 0:
            content = result["choices"][0]["message"]["content"]
            print(f"  ✓ Received response ({len(content)} chars)")

            # Try to parse JSON from response
            # Remove markdown code blocks if present
            content = content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

            parsed = json.loads(content)
            return parsed

        else:
            print(f"  ✗ Unexpected response format")
            print(json.dumps(result, indent=2))
            return {}

    except requests.exceptions.RequestException as e:
        print(f"  ✗ API Error: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"  Response status: {e.response.status_code}")
            print(f"  Response body: {e.response.text[:500]}")
        return {}
    except json.JSONDecodeError as e:
        print(f"  ✗ JSON parsing error: {e}")
        print(f"  Raw content: {content[:500]}")
        return {}


def process_annotation_task(
    task: Dict,
    output_dir: Path,
    food_memory: Dict,
    state_schema: Dict,
    fps: int = 1
) -> Dict:
    """Process a single annotation task with VLM

    Args:
        task: Annotation task dictionary
        output_dir: Base output directory
        food_memory: Current food memory state (updated in-place)
        state_schema: State taxonomy schema
        fps: Video sampling rate

    Returns:
        Updated food memory
    """
    task_id = task['task_id']
    print(f"\nProcessing task: {task_id}")

    # Get video clip path
    clip_path = output_dir / task['assets']['clip_path']
    if not clip_path.exists():
        print(f"  ✗ Video clip not found: {clip_path}")
        return food_memory

    # Get narrations and food nouns
    narrations = task['narrations_in_block']
    food_nouns = task['target_food_nouns']
    block_start = task['block_start_time']
    block_end = task['block_end_time']

    print(f"  Narrations: {len(narrations)}")
    print(f"  Food items: {', '.join(food_nouns)}")
    print(f"  Time: {block_start:.2f}s - {block_end:.2f}s")

    # Create prompt
    prompt = create_vlm_prompt(narrations, food_nouns, food_memory, state_schema)

    # Log VLM input (save to file for debugging)
    log_dir = output_dir / "vlm_logs"
    log_dir.mkdir(exist_ok=True)

    prompt_log_file = log_dir / f"{task_id}_prompt.txt"
    with open(prompt_log_file, 'w') as f:
        f.write(prompt)
    print(f"  📝 Saved VLM prompt to: {prompt_log_file.name}")

    # Query VLM
    vlm_response = query_qwen3vl(clip_path, prompt, fps=fps)

    if not vlm_response or 'food_memory' not in vlm_response:
        print(f"  ✗ No valid response from VLM")
        return food_memory

    # Log VLM output (save to file for debugging)
    response_log_file = log_dir / f"{task_id}_response.json"
    with open(response_log_file, 'w') as f:
        json.dump(vlm_response, f, indent=2)
    print(f"  📝 Saved VLM response to: {response_log_file.name}")

    # Process VLM response, handling new instance creation
    vlm_food_memory = vlm_response['food_memory']

    # Log what VLM returned
    print(f"  📊 VLM returned {len(vlm_food_memory)} instances:")
    for key in vlm_food_memory.keys():
        print(f"     - {key}")

    # Extract block_id from task_id (e.g., "P01-20240203-121517_block_003" -> 3)
    block_id_str = task_id.split('_block_')[-1]
    current_block_id = int(block_id_str) if block_id_str.isdigit() else 0

    for key, instance_data in vlm_food_memory.items():
        food_noun = instance_data.get('food_noun')

        if not food_noun:
            print(f"  ✗ Warning: Instance missing food_noun field, skipping")
            continue

        # Check if this is a new instance (key starts with "NEW_INSTANCE")
        if key.startswith("NEW_INSTANCE") or 'semantic_name' in instance_data:
            # Generate unique instance_id
            semantic_name = instance_data.get('semantic_name', 'instance')
            # Clean semantic_name (remove spaces, make snake_case)
            semantic_name = semantic_name.strip().lower().replace(' ', '_').replace('-', '_')

            instance_id = generate_instance_id(food_memory, food_noun, semantic_name)
            instance_data['instance_id'] = instance_id

            print(f"  + Created instance: {instance_id}")
            event_type = 'created_by_split' if 'parent_instance' in instance_data else 'created'
        else:
            # Existing instance being updated
            instance_id = key
            instance_data['instance_id'] = instance_id

            print(f"  ↻ Updated instance: {instance_id}")
            event_type = 'updated'

        # Add interaction history
        if 'interaction_history' not in instance_data:
            instance_data['interaction_history'] = []

        # Get most relevant narration (last one in block)
        relevant_narration = narrations[-1] if narrations else "No narration"

        instance_data['interaction_history'].append({
            'block_id': current_block_id,
            'timestamp': block_end,
            'narration': relevant_narration,
            'event_type': event_type
        })

        # Store semantic_name for reference if not already present
        if 'semantic_name' not in instance_data and instance_id:
            # Extract semantic name from instance_id
            # Format: {food_noun}_{semantic_descriptor}_{counter}
            parts = instance_id.rsplit('_', 1)
            if len(parts) == 2 and parts[1].isdigit():
                # Remove food_noun prefix and counter suffix
                semantic_part = instance_id.replace(f"{food_noun}_", "")
                semantic_name = semantic_part.rsplit('_', 1)[0]
                instance_data['semantic_name'] = semantic_name

        # Update global food memory (instance-based)
        food_memory[instance_id] = instance_data

        # Update task's state table with instance data
        if food_noun in task['ground_truth_state_table']:
            # Ensure instances dict exists
            if 'instances' not in task['ground_truth_state_table'][food_noun]:
                task['ground_truth_state_table'][food_noun]['instances'] = {}

            # Store per-instance annotations
            task['ground_truth_state_table'][food_noun]['instances'][instance_id] = {
                'vlm_state_after': instance_data.get('state'),
                'vlm_reasoning': instance_data.get('reasoning', ''),
                'semantic_name': instance_data.get('semantic_name'),
                'block_time_range': {'start': block_start, 'end': block_end},
                'annotator_state_after': None
            }

    # Remove consumed instances from active memory
    consumed_instances = [
        iid for iid, idata in food_memory.items()
        if idata.get('state', {}).get('consumption_state', {}).get('quantity') == 'consumed'
    ]
    for iid in consumed_instances:
        print(f"  - Removed consumed instance: {iid}")
        del food_memory[iid]

    return food_memory


def main():
    """Main VLM state tracking pipeline"""
    parser = argparse.ArgumentParser(
        description="VLM-based food state tracking for annotation benchmark"
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
        help='Base output directory (where video clips are stored)'
    )
    parser.add_argument(
        '--taxonomy-file',
        type=Path,
        default=Path('food_state_taxonomy.json'),
        help='Path to food state taxonomy JSON'
    )
    parser.add_argument(
        '--fps',
        type=int,
        default=1,
        help='Video sampling rate (frames per second) for VLM'
    )
    parser.add_argument(
        '--start-task',
        type=int,
        default=0,
        help='Start from this task index (for resuming)'
    )

    args = parser.parse_args()

    print("=" * 80)
    print("VLM FOOD STATE TRACKING")
    print("=" * 80)
    print(f"Tasks file: {args.tasks_file}")
    print(f"Output directory: {args.output_dir}")
    print(f"Taxonomy: {args.taxonomy_file}")
    print(f"Video sampling: {args.fps} fps")
    print()

    # Load taxonomy
    print("[Step 1] Loading state taxonomy...")
    taxonomy = load_state_taxonomy(args.taxonomy_file)
    state_schema = taxonomy['state_schema']
    print(f"✓ Loaded taxonomy with {len(state_schema)} state categories")

    # Load annotation tasks
    print("\n[Step 2] Loading annotation tasks...")
    tasks = load_annotation_tasks(args.tasks_file)
    print(f"✓ Loaded {len(tasks)} annotation tasks")

    # Initialize food memory
    food_memory = {}

    # Process each task
    print("\n[Step 3] Processing tasks with VLM...")
    successful = 0
    failed = 0

    for i in range(args.start_task, len(tasks)):
        task = tasks[i]

        try:
            food_memory = process_annotation_task(
                task,
                args.output_dir,
                food_memory,
                state_schema,
                args.fps
            )
            successful += 1

        except Exception as e:
            print(f"  ✗ Error processing task {i}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    # Save updated tasks (main output - contains all data)
    print(f"\n[Step 4] Saving results...")
    with open(args.tasks_file, 'w') as f:
        json.dump(tasks, f, indent=2)
    print(f"✓ Saved updated tasks to {args.tasks_file}")
    print(f"  This file contains: blocks, frames, clips, grounding, VLM states")

    # Print summary
    print("\n" + "=" * 80)
    print("VLM STATE TRACKING COMPLETE")
    print("=" * 80)
    print(f"Tasks processed: {successful}/{len(tasks)}")
    print(f"Failed: {failed}")
    print(f"Food items in memory: {len(food_memory)}")
    print(f"Food items: {', '.join(sorted(food_memory.keys()))}")
    print("=" * 80)


if __name__ == '__main__':
    main()

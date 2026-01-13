"""
VLM Prompts for Spatio-Temporal Food Graph

This module provides prompt construction and response parsing for:
1. Food initialization: Determine initial state from trigger_text
2. Transaction inference: Infer transactions from block narrations
"""

import json
import logging
from typing import Dict, List, Any, Optional

from .data_structures import BlockGraph, FoodNode, NodeStatus

logger = logging.getLogger(__name__)


# ============================================================================
# Food Initialization Prompt
# ============================================================================

FOOD_INIT_SYSTEM_PROMPT = """You are initializing food items for a cooking video tracker.

INPUT:
1. ARRIVALS: List of food items with trigger narration and timestamp
2. VIDEO CLIP: The video for this block - use it to verify and count items

STATE TAXONOMY:
- form_state: whole | prepared_ingredient | cooking_in_progress | cooked_dish | leftover
- quantity: full | partial | nearly_empty
- count: integer (for countable items like eggs, slices) or null

GUIDELINES:
1. For each arrival, look at the specified timestamp in the video
2. Count visible items at that moment (e.g., oranges in a mesh bag)
3. Use the video to verify the form_state (whole, sliced, etc.)

Return JSON:
[
  {
    "food_noun": "milk",
    "state": {"form_state": "whole", "quantity": "full", "count": null}
  }
]"""


def build_food_init_prompt(arrivals: List[Dict], block_start_time: float = 0.0) -> str:
    """
    Build prompt to initialize food items with state.

    Args:
        arrivals: List of inventory arrivals with semantic_name, trigger_text, timestamp
        block_start_time: Start time of the block (for calculating clip-relative timestamps)

    Returns:
        User prompt string
    """
    lines = ["ARRIVALS TO INITIALIZE:"]
    lines.append("(Look at the specified timestamp in the attached video to count items)\n")

    for i, arrival in enumerate(arrivals, 1):
        semantic_name = arrival.get('semantic_name', 'unknown')
        trigger_text = arrival.get('trigger_text', '')
        timestamp = arrival.get('timestamp', 0)
        clip_time = timestamp - block_start_time

        lines.append(f"{i}. Food: {semantic_name}")
        lines.append(f"   Trigger: \"{trigger_text}\"")
        lines.append(f"   Video timestamp: [{clip_time:.1f}s] in attached clip")

    lines.append("""
For each item, return:
- food_noun: the canonical name
- state: {form_state, quantity, count}

Use the video to COUNT items (e.g., how many oranges in the mesh bag?).
""")

    return "\n".join(lines)


def parse_food_init_response(response_text: str) -> List[Dict]:
    """
    Parse VLM response for food initialization.

    Returns list of dicts with food_noun, state.
    """
    # Handle "JSON:" marker if present
    text = response_text.strip()
    if "JSON:" in text:
        text = text.split("JSON:")[1].strip()

    # Handle markdown code blocks
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()

    try:
        result = json.loads(text)
        if not isinstance(result, list):
            result = [result]
        return result
    except json.JSONDecodeError as e:
        logger.error(f"ERROR parsing food init response: {e}")
        logger.error(f"Full response text:\n{response_text}")
        return []


# ============================================================================
# Transaction Inference Prompt
# ============================================================================

TRANSACTION_SYSTEM_PROMPT = """You are a food state tracker. Analyze narrations and video to identify food state changes.

LOCATION RULES (CONTAINER-ONLY MODEL):
1. TRACKED (Container): Any object with walls/volume (pan, bowl, cup, jar, blender).
   - System auto-numbers as container_id (e.g., "pan" → pan_001).
   - Food INSIDE containers is subject to mixing when stirred.

2. UNTRACKED (Surface): Any flat open surface (counter, table, stove, cutting_board, shelf).
   - Set `to_location: null` for food on surfaces.
   - Food ON SURFACES remains distinct (no automatic merging).

For NEW containers not in the list, just name it (e.g., "bowl") - system auto-numbers.

RULES:
1. Use food IDs from the 'Active Food Items' table for existing items.
2. If an item is created via SPLIT, refer to it by its Name (e.g., "orange half").
3. If an item is consumed/merged, STOP using its ID.

TRANSACTION TYPES:
- TRANSFER: Move food to a location.
  {type: "transfer", food_id: "...", to_location: "..."}

- SPLIT: Divide food.
  * Partial (source remains): {type: "split", subtype: "partial", parent_id: "...", children: [{food_noun: "...", quantity: "...", count: N}]}
  * Complete (source consumed): {type: "split", subtype: "complete", parent_id: "...", children: [{food_noun: "...", ...}, ...]}

- MERGE: Absorb one food into another (subject dies, target survives).
  {type: "merge", subject_id: "...", target_id: "..."}

- CONSUME: Food eaten or discarded.
  {type: "consume", food_id: "...", consume_type: "eaten"|"discarded"}

- UPDATE: Change state or identity in place.
  {type: "update", food_id: "...", state_changes: {food_noun?: "...", form_state?: "...", quantity?: "..."}}
  * Use food_noun when the item transforms into something new:
    - Orange being juiced → food_noun: "orange_juice"
    - Bread being toasted → food_noun: "toast"
    - Egg being fried → food_noun: "fried_egg"

VALID VALUES:
- form_state: whole | prepared_ingredient | cooking_in_progress | cooked_dish | leftover
"""

def build_active_food_table(graph: BlockGraph) -> str:
    """Build markdown table of active food items with their locations (V3 container-only model)"""
    lines = ["| Instance ID | Food | Form | Quantity | Location |",
             "|-------------|------|------|----------|----------|"]

    active_nodes = graph.get_active_food_nodes()
    for instance_id, node in sorted(active_nodes.items()):
        # V3: location is either container_id or None (on surface)
        location = node.location or graph.get_container_for_food(instance_id)
        location_str = location if location else "(on surface)"
        count_str = f" (x{node.state.count})" if node.state.count else ""
        lines.append(
            f"| {instance_id} | {node.food_noun} | {node.state.form_state}{count_str} | {node.state.quantity} | {location_str} |"
        )

    if len(lines) == 2:
        lines.append("| (no active food items) | - | - | - | - |")

    return "\n".join(lines)


def build_containers_table(graph: BlockGraph) -> str:
    """Build markdown table of known containers (DEPRECATED - use build_known_locations_table)"""
    return build_known_locations_table(graph)


def build_known_locations_table(graph: BlockGraph) -> str:
    """Build markdown table of containers and surface items (V3 container-only model)"""
    from collections import defaultdict

    # Collect what's in each container vs on surfaces
    container_contents = defaultdict(list)
    surface_items = []

    for instance_id, node in graph.get_active_food_nodes().items():
        location = node.location or graph.get_container_for_food(instance_id)
        if location:
            container_contents[location].append(instance_id)
        else:
            surface_items.append(instance_id)

    lines = ["**CONTAINERS:**",
             "| Container | Contents |",
             "|-----------|----------|"]

    for location in sorted(container_contents.keys()):
        contents = container_contents[location]
        contents_str = ", ".join(contents) if contents else "(empty)"
        lines.append(f"| {location} | {contents_str} |")

    if len(container_contents) == 0:
        lines.append("| (no containers in use) | - |")

    # Add surface items section
    lines.append("")
    lines.append("**FOOD ON SURFACES (location=null, stays distinct):**")
    if surface_items:
        lines.append(", ".join(surface_items))
    else:
        lines.append("(none)")

    return "\n".join(lines)


def format_narrations(narrations: List[Dict], block_start_time: float = 0.0) -> str:
    """Format narrations for the prompt with timestamps relative to block start"""
    lines = []
    for i, narr in enumerate(narrations, 1):
        text = narr.get('narration', '')
        start = narr.get('start_timestamp', 0) - block_start_time
        end = narr.get('end_timestamp', 0) - block_start_time
        lines.append(f"{i}. [{start:.2f}s - {end:.2f}s] {text}")
    return "\n".join(lines)


def build_transaction_prompt(
    graph: BlockGraph,
    block: Dict,
    newly_created_food_ids: List[str] = None
) -> str:
    """
    Build prompt for transaction inference (V2 Location model).

    Args:
        graph: Current block graph with active nodes
        block: Block metadata including narrations
        newly_created_food_ids: IDs of food items just created in this block

    Returns:
        User prompt string
    """
    food_table = build_active_food_table(graph)
    locations_table = build_known_locations_table(graph)
    block_start_time = block.get('block_start_time', 0)
    narrations = format_narrations(block.get('narrations', []), block_start_time)

    # Note about newly created items
    new_items_note = ""
    if newly_created_food_ids:
        new_items_note = f"\nNOTE: These food items were just created from inventory arrivals: {', '.join(newly_created_food_ids)}\n"

    prompt = f"""**BLOCK INFO:**
Video: {block.get('video_id', 'unknown')}
Block: {block.get('block_id', 0)}
Time: {block.get('block_start_time', 0):.1f}s - {block.get('block_end_time', 0):.1f}s
{new_items_note}
**ACTIVE FOOD ITEMS:**
{food_table}

**KNOWN LOCATIONS:**
{locations_table}

**NARRATIONS:**
{narrations}

**TASK:**
Analyze the narrations to identify food state changes or movements.
1. Identify any transactions (Transfer, Split, Merge, Consume, Update).
2. Use existing location IDs from "KNOWN LOCATIONS" when applicable.
3. For NEW containers, use the container type (e.g., "bowl") - system will auto-number it.

Return JSON:
```json
{{
  "transactions": [
    {{"type": "transfer", "food_id": "egg_001", "to_location": "pan_001"}},
    {{"type": "update", "food_id": "egg_001", "state_changes": {{"form_state": "cooking_in_progress"}}}}
  ]
}}
```"""

    return prompt


def parse_vlm_response(response_text: str) -> Dict[str, Any]:
    """
    Parse VLM response for transactions.

    Returns dict with new_containers and transactions arrays.
    """
    # Handle "JSON:" marker if present
    text = response_text.strip()
    if "JSON:" in text:
        text = text.split("JSON:")[1].strip()

    # Handle markdown code blocks
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()

    try:
        result = json.loads(text)
        if not isinstance(result, dict):
            return {"new_containers": [], "transactions": []}

        # Ensure required fields exist
        if "new_containers" not in result:
            result["new_containers"] = []
        if "transactions" not in result:
            result["transactions"] = []

        return result

    except json.JSONDecodeError as e:
        logger.error(f"ERROR parsing VLM response: {e}")
        logger.error(f"Full response text:\n{response_text}")
        return {"new_containers": [], "transactions": [], "parse_error": str(e)}


# ============================================================================
# State Description Prompt (Intermediate Step)
# ============================================================================

STATE_DESCRIPTION_PROMPT = """You are an expert Food State Analyst.
Your goal is to convert raw narrations of kitchen activities into precise DESCRIPTIONS of food state changes.

INPUT:
1. A block of narrations.
2. A video clip.

YOUR PROCESS:

STEP 1: RELEVANCE FILTER (Food-Centric)
- Is this about FOOD or a CONTAINER holding food?
- Ignore empty tools, appliances setup, or general cleaning.
- If irrelevant, output "SKIP".

STEP 2: VISUAL VERIFICATION & CLASSIFICATION
Watch the video and narration to determine which category of change occurred. Use the corresponding PHRASING pattern:

A. SEPARATION / POURING
   - Reality: A portion is removed, liquid is poured, or food is divided. The source often remains.
   - Required Phrasing: "A portion of [Food] was separated/poured from [Source] into [Target/Location]."
   - Example: "A portion of milk was poured from the jug into the glass." (Not "Milk moved to glass").

B. MOVEMENT / RELOCATION
   - Reality: The distinct object moves entirely. It is the SAME object, just elsewhere.
   - Required Phrasing: "The [Food] was moved/transferred from [Source] to [Destination]."
   - Example: "The bowl of salad was moved to the table."

C. COMBINATION
   - Reality: One food enters another and loses its separate identity (mixing, dissolving).
   - Required Phrasing: "The [Subject Food] was mixed into/added to the [Target Food]."

D. TRANSFORMATION
   - Reality: The food changes form, identity, or state (cooking, peeling, juicing).
   - Required Phrasing: "The [Food] was transformed/processed into [New Form/Name]."

OUTPUT FORMAT:
[Timestamp] "Original Text"
-> CONTEXT: (Visual proof)
-> STATE: (The precise sentence using the phrasing above)
"""


def build_state_description_prompt(narrations: List[Dict], block_start_time: float = 0.0) -> str:
    """
    Build prompt for state description generation.

    Args:
        narrations: List of narration dicts with narration, start_timestamp, end_timestamp
        block_start_time: Start time of the block for relative timestamps

    Returns:
        User prompt string
    """
    lines = ["**NARRATIONS TO ANALYZE:**\n"]

    for i, narr in enumerate(narrations, 1):
        text = narr.get('narration', '')
        start = narr.get('start_timestamp', 0)
        lines.append(f"{i}. [{start:.2f}s] {text}")

    lines.append("\n**TASK:**")
    lines.append("For each narration line above, apply the 3-step process:")
    lines.append("1. RELEVANCE FILTER - Is this a food-related action?")
    lines.append("2. VISUAL VERIFICATION - What does the video show?")
    lines.append("3. INFER STATE CHANGE - Describe the outcome.")
    lines.append("\nOutput your analysis for EACH line in the format shown in the examples.")

    return "\n".join(lines)


def parse_state_description_response(response_text: str, narrations: List[Dict]) -> List[Dict]:
    """
    Parse VLM response for state descriptions.

    Handles multiple formats:
    - "1. [21.31s] ..." (numbered)
    - "[21.31s] ..." (timestamp only)

    Returns list of dicts with:
    - timestamp: float
    - original_narration: str
    - skip: bool
    - skip_reason: str (if skip=True)
    - context: str (if skip=False)
    - state_description: str (if skip=False)
    """
    results = []
    lines = response_text.strip().split('\n')

    current_entry = None

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Check for new entry - supports multiple timestamp formats:
        # - "[21.31s]" or "[7.44s]" (seconds)
        # - "[0:00]" or "[1:23]" (MM:SS)
        # Pattern: line starts with optional number prefix then [timestamp]
        has_timestamp = '[' in line and ']' in line

        # Check if this is a new entry line (has timestamp near start)
        is_new_entry = False
        timestamp = 0.0
        if has_timestamp:
            bracket_pos = line.index('[')
            # It's a new entry if bracket is near start (with optional number prefix)
            if bracket_pos < 10:  # Allow for "1. " or "12. " prefix
                # Try to extract timestamp in various formats
                bracket_end = line.index(']', bracket_pos)
                ts_content = line[bracket_pos + 1:bracket_end]

                # Format 1: "21.31s" or "7.44s" (seconds with 's' suffix)
                if ts_content.endswith('s'):
                    try:
                        timestamp = float(ts_content[:-1])
                        is_new_entry = True
                    except ValueError:
                        pass
                # Format 2: "0:00" or "1:23" (MM:SS)
                elif ':' in ts_content:
                    try:
                        parts = ts_content.split(':')
                        if len(parts) == 2:
                            mins, secs = float(parts[0]), float(parts[1])
                            timestamp = mins * 60 + secs
                            is_new_entry = True
                    except ValueError:
                        pass
                # Format 3: Just a number like "21.31"
                else:
                    try:
                        timestamp = float(ts_content)
                        is_new_entry = True
                    except ValueError:
                        pass

        if is_new_entry:
            # Save previous entry if exists
            if current_entry:
                results.append(current_entry)

            # Extract content after timestamp
            bracket_end = line.index(']')
            content_after_ts = line[bracket_end + 1:].strip()

            # Check for inline format: "narration -> CONTEXT: ... -> STATE: ..."
            narration = ""
            inline_context = None
            inline_state = None

            if '-> CONTEXT:' in content_after_ts or '-> STATE:' in content_after_ts:
                # Inline format - parse all parts from the same line
                parts = content_after_ts

                # Extract narration (everything before first "->")
                if '->' in parts:
                    narration = parts.split('->')[0].strip()
                    # Remove quotes if present
                    if narration.startswith('"') and '"' in narration[1:]:
                        narration = narration[1:narration.index('"', 1)]

                # Extract CONTEXT if present
                if 'CONTEXT:' in parts:
                    context_start = parts.index('CONTEXT:') + 8
                    context_end = parts.index('->', context_start) if '->' in parts[context_start:] else len(parts)
                    inline_context = parts[context_start:context_end].strip()

                # Extract STATE if present
                if 'STATE:' in parts:
                    state_start = parts.index('STATE:') + 6
                    inline_state = parts[state_start:].strip()

            elif '"' in content_after_ts:
                # Quoted format: "narration text"
                quote_start = content_after_ts.index('"') + 1
                quote_end = content_after_ts.rindex('"') if content_after_ts.count('"') > 1 else len(content_after_ts)
                narration = content_after_ts[quote_start:quote_end]
            else:
                # Unquoted format
                narration = content_after_ts

            current_entry = {
                'timestamp': timestamp,
                'original_narration': narration,
                'skip': False,
                'skip_reason': None,
                'context': inline_context,
                'state_description': inline_state
            }

        # Check for SKIP
        elif '-> SKIP' in line or '->SKIP' in line or 'SKIP' in line.upper():
            if current_entry:
                current_entry['skip'] = True
                # Extract reason if in parentheses
                if '(' in line and ')' in line:
                    reason_start = line.index('(') + 1
                    reason_end = line.index(')')
                    current_entry['skip_reason'] = line[reason_start:reason_end]
                else:
                    current_entry['skip_reason'] = "Non-food action"

        # Check for CONTEXT
        elif 'CONTEXT:' in line:
            if current_entry:
                context = line.split('CONTEXT:')[1].strip()
                current_entry['context'] = context

        # Check for STATE
        elif 'STATE:' in line:
            if current_entry:
                state = line.split('STATE:')[1].strip()
                current_entry['state_description'] = state

    # Don't forget last entry
    if current_entry:
        results.append(current_entry)

    # If parsing failed, create basic entries from narrations
    if not results and narrations:
        for narr in narrations:
            results.append({
                'timestamp': narr.get('start_timestamp', 0),
                'original_narration': narr.get('narration', ''),
                'skip': False,
                'skip_reason': None,
                'context': None,
                'state_description': None,
                'parse_error': True
            })

    return results


# ============================================================================
# Transaction Inference from State Descriptions (Two-Stage Pipeline)
# ============================================================================

TRANSACTION_FROM_DESCRIPTIONS_PROMPT = """You are a food state tracker. Convert state descriptions into formal graph transactions.

INPUT:
1. STATE DESCRIPTIONS: Pre-analyzed descriptions with timestamps (relative to clip start).
   NOTE: These may include non-food items (tools, appliances). ONLY create transactions for items in the Active Food Items table.
2. VIDEO CLIP: The actual video for this block - use it to verify and ground your transactions.

LOCATION RULES (CONTAINER-ONLY MODEL):
1. TRACKED (Container): Any object with walls/volume (pan, bowl, cup, jar, blender, plate).
   - System auto-numbers as container_id (e.g., "pan" → pan_001).
   - Food INSIDE containers is subject to mixing when stirred.
   - Examples: pan, pot, bowl, cup, mug, jar, bottle, blender, plate, bag

2. UNTRACKED (Surface): Any flat open surface (counter, table, stove, cutting_board, shelf).
   - Set `to_location: null` for food placed on surfaces.
   - Food ON SURFACES remains distinct (no automatic merging).
   - Examples: counter, table, stove, cutting_board, shelf, fridge

For NEW containers, just name it (e.g., "bowl") - system will auto-number.

TRANSACTION TYPES (each requires a "reasoning" field):
- TRANSFER: Move food to a location.
  {type: "transfer", food_id: "...", to_location: "...", reasoning: "..."}

- SPLIT: Divide food.
  * Partial: {type: "split", subtype: "partial", parent_id: "...", children: [...], reasoning: "..."}
  * Complete: {type: "split", subtype: "complete", parent_id: "...", children: [...], reasoning: "..."}

- MERGE: Absorb one food into another (subject dies, target survives).
  {type: "merge", subject_id: "...", target_id: "...", reasoning: "..."}

- CONSUME: Food eaten or discarded.
  {type: "consume", food_id: "...", consume_type: "eaten"|"discarded", reasoning: "..."}

- UPDATE: Change state or identity in place.
  {type: "update", food_id: "...", state_changes: {...}, reasoning: "..."}
  * Use food_noun when the item transforms into something new:
    - Orange being juiced → food_noun: "orange_juice"
    - Bread being toasted → food_noun: "toast"
    - Egg being fried → food_noun: "fried_egg"

RULES:
1. ONLY create transactions for food items listed in the Active Food Items table.
2. IGNORE descriptions about non-food items (tools, appliances, empty containers).
3. Use food IDs from the Active Food Items table for existing items.
4. For items created via SPLIT in this block, use the food_noun (e.g., "butter").
5. One transaction per distinct state change.
6. Each transaction MUST include a "reasoning" field explaining the visual evidence.
7. Return empty transactions array if no food state changes occur in the descriptions.

VALID VALUES:
- form_state: whole | prepared_ingredient | cooking_in_progress | cooked_dish | leftover
"""


def format_state_descriptions(descriptions: List[Dict], block_start_time: float = 0.0) -> str:
    """Format state descriptions for the transaction prompt.

    Only includes non-skipped descriptions with state changes.
    Context is omitted - VLM should verify against the video clip directly.
    Timestamps are relative to clip start (not absolute video time).
    """
    lines = []
    for desc in descriptions:
        if desc.get('skip'):
            continue

        ts = desc.get('timestamp', 0) - block_start_time
        state = desc.get('state_description', '')

        if state:
            lines.append(f"[{ts:.1f}s] {state}")

    if not lines:
        return "(No food-related state changes in this block)"

    return "\n".join(lines)


def build_transaction_prompt_from_descriptions(
    graph: BlockGraph,
    block: Dict,
    descriptions: List[Dict],
    newly_created_food_ids: List[str] = None
) -> str:
    """
    Build prompt for transaction inference using pre-analyzed state descriptions (V2 Location model).

    This is the second stage of the two-stage VLM pipeline:
    Stage 1: Narrations + Video → State Descriptions (natural language)
    Stage 2: State Descriptions → Transactions (structured JSON)

    Args:
        graph: Current block graph with active nodes
        block: Block metadata
        descriptions: Pre-analyzed state descriptions from stage 1
        newly_created_food_ids: IDs of food items just created in this block

    Returns:
        User prompt string
    """
    food_table = build_active_food_table(graph)
    locations_table = build_known_locations_table(graph)
    block_start_time = block.get('block_start_time', 0)
    state_desc_text = format_state_descriptions(descriptions, block_start_time)

    # Note about newly created items
    new_items_note = ""
    if newly_created_food_ids:
        new_items_note = f"\nNOTE: These food items were just created from inventory arrivals: {', '.join(newly_created_food_ids)}\n"

    # Calculate clip duration
    clip_duration = block.get('block_end_time', 0) - block_start_time

    prompt = f"""**BLOCK INFO:**
Video: {block.get('video_id', 'unknown')}
Block: {block.get('block_id', 0)}
Clip duration: {clip_duration:.1f}s (timestamps below are relative to clip start)
{new_items_note}
**ACTIVE FOOD ITEMS:**
{food_table}

**KNOWN LOCATIONS:**
{locations_table}

**STATE DESCRIPTIONS:**
(Timestamps are relative to clip start. May include non-food items - ignore those.)
{state_desc_text}

**VIDEO CLIP:** (attached - use to verify state changes)

**TASK:**
Convert state descriptions about FOOD ITEMS into formal transactions.
1. ONLY process descriptions that match items in the Active Food Items table.
2. SKIP descriptions about tools, appliances, or non-food items.
3. For NEW containers, use the container type (e.g., "bowl") - system will auto-number it.
4. Include source_text: The exact quote from the input description that triggered this.

Return JSON:
```json
{{
  "transactions": [
    {{"type": "transfer", "food_id": "milk_001", "to_location": "frother_001", "reasoning": "Video shows milk bottle being placed into frother"}}
  ]
}}
```"""

    return prompt


def load_state_descriptions(video_id: str, block_id: int, base_dir: str = "../outputs/food_classification/state_descriptions") -> Optional[List[Dict]]:
    """
    Load pre-computed state descriptions for a block.

    Args:
        video_id: Video identifier
        block_id: Block number
        base_dir: Base directory for state descriptions

    Returns:
        List of description dicts, or None if not found
    """
    import os
    desc_file = os.path.join(base_dir, video_id, f"block_{block_id:03d}_descriptions.json")

    if not os.path.exists(desc_file):
        return None

    try:
        with open(desc_file, 'r') as f:
            data = json.load(f)
        return data.get('descriptions', [])
    except Exception as e:
        logger.warning(f"Failed to load state descriptions from {desc_file}: {e}")
        return None


# ============================================================================
# Gemini Pre-Annotation Transaction Inference
# ============================================================================

GEMINI_TRANSACTION_SYSTEM_PROMPT = """You are converting Gemini state descriptions into formal graph transactions.

INPUT:
1. STATE DESCRIPTION: A pre-analyzed description of what happened to food.
2. VIDEO CLIP: Visual evidence for the event.
3. ACTIVE FOOD ITEMS: Current inventory state with instance IDs.

YOUR TASK:
Map the state description to one or more formal transactions that update the food graph.

LOCATION RULES (PHYSICS-BASED):

1. TRACKED LOCATIONS (Containers):
   - **Definition:** Any object that physically encloses food with walls or volume.
   - **Rule:** If food is INSIDE it, generate a `container_id`. Food here is subject to mixing/physics.
   - **Examples:** pan, pot, bowl, cup, mug, jar, bottle, blender, plate, bag, tupperware, ramekin, tray.

2. UNTRACKED LOCATIONS (Surfaces):
   - **Definition:** Any open surface where food rests *on top*.
   - **Rule:** Set `to_location: null` (or `destination: null`). Food here remains distinct (no automatic merging).
   - **Examples:** counter, table, stove_top, cutting_board, shelf, fridge_shelf, sink, drainer.

TRANSACTION TYPES:

1. TRANSFER - Move an existing food item to a new location
   - Use ONLY if the *whole* item moves.
   {"type": "transfer", "food_id": "butter_001", "to_location": "pan_001", "reasoning": "..."}

2. SPLIT - Divide food or dispense a portion (USE "destination" FOR IMMEDIATE PLACEMENT)

   Partial (Pouring/Grinding/Slicing - source remains):
   {
     "type": "split",
     "subtype": "partial",
     "parent_id": "jar_of_pepper_001",
     "children": [
       {"food_noun": "ground_pepper", "quantity": "partial", "destination": "pan_001"}
     ],
     "reasoning": "Grinding pepper creates a portion that lands in the pan."
   }

   Complete (Cutting/Processing - source is consumed):
   {
     "type": "split",
     "subtype": "complete",
     "parent_id": "orange_001",
     "children": [
       {"food_noun": "orange_half", "quantity": "full", "count": 2, "destination": "cutting_board"}
     ],
     "reasoning": "Orange cut into two halves."
   }

3. MERGE - Combine two foods (subject is absorbed into target)
   - Use when adding ingredients to a mixture.
   {"type": "merge", "subject_id": "ground_pepper_001", "target_id": "sauce_001", "reasoning": "..."}

4. CONSUME - Food eaten or discarded
   {"type": "consume", "food_id": "orange_peel_001", "consume_type": "discarded", "reasoning": "..."}

5. UPDATE - Change food state in place
   {"type": "update", "food_id": "butter_001", "state_changes": {"form_state": "cooking_in_progress"}, "reasoning": "..."}

   Identity transformation (Rename):
   {"type": "update", "food_id": "butter_001", "state_changes": {"food_noun": "melted_butter"}, "reasoning": "..."}

**CRITICAL: THE "DISPENSING" PATTERN**
When food is poured, sprinkled, ground, or sliced INTO something:
1. Use SPLIT (partial) because the source container/block remains.
2. Use the "destination" field in children to specify where the portion goes.
3. DO NOT generate a separate TRANSFER for the portion - it's redundant.

RULES:
1. Use food IDs from the Active Food Items table for existing items.
2. For items created via SPLIT, they are auto-assigned IDs (butter_slice_001, etc.).
3. Each transaction MUST include a "reasoning" field.
4. For NEW containers, just use the noun (e.g., "bowl") - system will auto-number.
5. Return empty transactions array if no actionable food state change occurred.

VALID VALUES:
- form_state: whole | prepared_ingredient | cooking_in_progress | cooked_dish | leftover
- quantity: full | partial | nearly_empty
- consume_type: eaten | discarded

Return JSON:
{"transactions": [...]}
"""


def build_gemini_transaction_prompt(
    graph: BlockGraph,
    event: Dict,
    newly_created_food_ids: List[str] = None
) -> str:
    """
    Build prompt for transaction inference from Gemini state description.

    Args:
        graph: Current block graph with active food nodes
        event: State change event dict with:
            - event_id
            - timestamp_start, timestamp_end
            - primary_action
            - visual_evidence
            - state_description
        newly_created_food_ids: IDs of food items just created from inventory

    Returns:
        User prompt string for VLM
    """
    food_table = build_active_food_table(graph)
    locations_table = build_known_locations_table(graph)

    # Note about newly created items
    new_items_note = ""
    if newly_created_food_ids:
        new_items_note = f"\nNOTE: These items were just created from inventory: {', '.join(newly_created_food_ids)}\n"

    prompt = f"""**EVENT INFO:**
Event ID: {event.get('event_id', 0)}
Time in video: {event.get('timestamp_start', 0):.1f}s - {event.get('timestamp_end', 0):.1f}s
Primary Action: {event.get('primary_action', 'unknown')}
{new_items_note}
**STATE DESCRIPTION (from Gemini):**
{event.get('state_description', '(no description)')}

**VISUAL EVIDENCE (from Gemini):**
{event.get('visual_evidence', '(no visual evidence)')}

**ACTIVE FOOD ITEMS:**
{food_table}

**KNOWN LOCATIONS:**
{locations_table}

**VIDEO CLIP:** (attached - verify the state description against what you see)

**TASK:**
Convert the state description into formal transactions.
- Match food references to items in the Active Food Items table.
- If the description mentions a new container (e.g., "pan"), just use that name.
- Provide "reasoning" for each transaction explaining how you interpreted the description.

Return JSON:
```json
{{"transactions": [...]}}
```"""

    return prompt


# ============================================================================
# Prompt/Response Logging
# ============================================================================

def create_log_entry(
    video_id: str,
    block_id: int,
    prompt_type: str,
    system_prompt: str,
    user_prompt: str,
    response: str,
    parsed: Any
) -> Dict[str, Any]:
    """Create a structured log entry for VLM interaction"""
    return {
        "video_id": video_id,
        "block_id": block_id,
        "prompt_type": prompt_type,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "raw_response": response,
        "parsed_response": parsed
    }

#!/usr/bin/env python3
"""
Shared utilities for inventory pipeline scripts.

Contains:
- GPTClient for API calls
- Path helpers for output files
- Recipe/narration loading utilities
- Common prompts
"""

import hashlib
import os
import json
import pickle
from pathlib import Path
from typing import List, Dict, Optional
from collections import Counter

from dotenv import load_dotenv
from openai import AzureOpenAI

# Load environment variables
load_dotenv(Path(__file__).parent.parent.parent.parent / ".env")

_SCRIPT_DIR = Path(__file__).parent
_PROJECT_ROOT = _SCRIPT_DIR.parent

# Default paths
DEFAULT_OUTPUT_DIR = _PROJECT_ROOT / "outputs" / "02_inventory"
RECIPES_FILE = _PROJECT_ROOT / "data" / "hd-epic-annotations" / "high-level" / "complete_recipes.json"
DEFAULT_PICKLE_PATH = _PROJECT_ROOT / "data" / "hd-epic-annotations" / "narrations-and-action-segments" / "HD_EPIC_Narrations.pkl"

# Cache for pickle data
_PICKLE_CACHE = None


# ============================================================
# SEGMENT ID GENERATION
# ============================================================

def generate_segment_id(narration_id: str, video_id: str, start_ts: float, end_ts: float) -> str:
    """
    Generate a stable segment ID using hash of immutable properties.

    Args:
        narration_id: Parent item's narration ID (e.g., 'P03-20240216-185832-1024')
        video_id: Video ID for this segment (e.g., 'P03-20240216-185832')
        start_ts: Start timestamp in seconds
        end_ts: End timestamp in seconds

    Returns:
        Segment ID like 'seg_a3b8c92f' (8-char hex hash)
    """
    # Use 2 decimal places for timestamps to ensure stability
    key = f"{narration_id}:{video_id}:{start_ts:.2f}:{end_ts:.2f}"
    hash_hex = hashlib.sha256(key.encode()).hexdigest()[:8]
    return f"seg_{hash_hex}"


def add_segment_ids_to_item(item: Dict) -> Dict:
    """
    Add segment_id to each dispensal_segment in an item.

    Args:
        item: Item dict with narration_id and dispensal_segments

    Returns:
        Item dict with segment_ids added to segments
    """
    narration_id = item.get('narration_id', '')
    segments = item.get('dispensal_segments', [])

    for seg in segments:
        video_id = seg.get('video_id', '')
        start_ts = seg.get('start_timestamp', 0)
        end_ts = seg.get('end_timestamp', 0)
        seg['segment_id'] = generate_segment_id(narration_id, video_id, start_ts, end_ts)

    return item


class GPTClient:
    """Wrapper for Azure OpenAI API with support for multiple endpoints and Responses API."""

    def __init__(self, model: str = "gpt-4o", use_reasoning: bool = False):
        self.model = model
        self.use_reasoning = use_reasoning

        # Use endpoint 2 for gpt-5/gpt-5.2/o3, endpoint 1 for others
        if model in ["gpt-5", "gpt-5.2", "gpt-5-chat", "o3-pro", "o3"]:
            api_key = os.getenv("AZURE_OPENAI_API_KEY_2")
            endpoint = os.getenv("AZURE_OPENAI_ENDPOINT_2", "").strip()
            api_version = "2025-03-01-preview" if use_reasoning else "2025-01-01-preview"
        else:
            api_key = os.getenv("AZURE_OPENAI_API_KEY")
            endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()
            api_version = "2025-01-01-preview"

        if not api_key or not endpoint:
            raise ValueError(f"Missing API credentials for {model}")

        self.client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=api_version,
        )

    def chat_completion(self, messages: List[Dict], max_tokens: int = 16384):
        """Standard chat completion API."""
        return self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_completion_tokens=max_tokens,
        )

    def responses_create(
        self,
        instructions: str,
        input_text: str,
        reasoning_effort: str = "medium",
        max_tokens: int = 16384
    ) -> str:
        """Use Responses API with reasoning enabled."""
        response = self.client.responses.create(
            model=self.model,
            reasoning={"effort": reasoning_effort},
            instructions=instructions,
            input=input_text,
            max_output_tokens=max_tokens,
        )
        return response.output_text, response.usage


# ============================================================
# PROMPTS
# ============================================================

INVENTORY_DISCOVERY_PROMPT = """
You are an "Ingredient Spotter" for a cooking video.

Your task is to analyze the entire narration log and identify the FIRST APPEARANCE of every distinct food item, excluding water.

CORE DEFINITION: ROOT ENTITY

A "Root Entity" is a physical food object that enters the user's workspace from an external source (fridge, cupboard, pantry, off-camera).

YOUR GOAL:

Create a chronological list of *only* the new food items arriving in the scene.

⛔ EXCLUSION RULES (CRITICAL):

1. IGNORE TOOLS & APPLIANCES: Do not list knives, spoons, bowls, pans, juicers, coffee machines, ovens, or fridges.

2. IGNORE RE-APPEARANCES: If "milk" is listed at 5.0s, do not list "milk" again at 20.0s when it is poured.

3. IGNORE DERIVATIVES: If "orange" is listed, do not list "orange peel", "orange juice", or "orange half" later. These are children of the root.

4. IGNORE EMPTY CONTAINERS: Do not list "mug" or "glass" unless they are pre-filled with food.

INPUT FORMAT:

A list of: [Narration ID], [Narration Text]

OUTPUT FORMAT:

Return a JSON list of objects:

[
  {
    "narration_id": "P01-20240202-161948-269",
    "food_name": "milk bottle",
    "source_action": "picked up from fridge door"
  },
  ...
]

TASK:

Identify the Valid Root Entities (Food/Ingredients) entering the scene.

Apply the exclusion rules strictly. No tools. No appliances. No duplicate mentions.
"""

INVENTORY_LIFECYCLE_PROMPT = """You are an "Inventory Auditor."
Your task is to trace the COMPLETE LIFECYCLE of **ONE specific item** through the narration log.

INPUTS:
1. TARGET ITEM: "{item_name}"
2. NARRATION LOG: Chronological list of all user actions.

YOUR GOAL:
Scan the log and extract ONLY the logistic events for the Target Item.
Ignore all other food items. Ignore cooking/processing steps.

### THE LIFECYCLE STAGES (Look for these):

1. **RETRIEVAL (Start of Cycle):**
   - Bringing the item from a storage zone (Fridge, Cupboard, Pantry) to the workspace.
   - *Example:* "Pick up {item_name} from fridge."

2. **ACCESS (Prep for Dispensing):**
   - Opening the container physically.
   - *Example:* "Unscrew cap of {item_name}", "Remove lid", "Unwrap foil."

3. **DISPENSING (Quantity Reduction):**
   - The moment quantity leaves the Source Container.
   **REQUIRED FIELD: "method" (Select one):**
   - **"pour"**: Gravity-fed liquids or loose solids (milk, cereal, rice).
   - **"scoop"**: Using a spoon/cup to remove powders or pastes (flour, peanut butter).
   - **"pick_unit"**: Removing a distinct whole item (egg from carton, slice from bread bag).
   - **"cut_portion"**: Cutting a piece OFF a larger solid block (butter, cheese) to use.
   - **"shake"**: Sprinkling from a shaker (salt, spices).
   - **"squeeze"**: Forcing semi-solids out of a tube/bottle (ketchup, honey).
   *Example:* "User pours milk" -> `{{"stage": "DISPENSING", "method": "pour"}}`
   *Example:* "User takes an egg" -> `{{"stage": "DISPENSING", "method": "pick_unit"}}`

4. **RESTOCKING (End of Cycle):**
   - Returning the Source Container to a storage zone.
   - *Example:* "Place {item_name} back in fridge."
   - *Note:* If the item is EMPTY/TRASHED, record that as the end of lifecycle.

### DISCARD (Noise Filter):
- **Culinary Processing:** Chopping/dicing the portion *after* it has been dispensed (e.g., "Chop the butter slice").
- **Cooking:** Boiling, frying, baking.
- **Other Items:** Actions related to other food items.

### OUTPUT FORMAT (JSON):
Return a list of Lifecycle Events for this item.
If the item is never touched, return an empty list.

[
  {{
    "narration_id": "P01-...",
    "stage": "RETRIEVAL",
    "action": "User retrieves {item_name} from fridge."
  }},
  {{
    "narration_id": "P01-...",
    "stage": "ACCESS",
    "action": "User unwraps the {item_name}."
  }},
  {{
    "narration_id": "P01-...",
    "stage": "DISPENSING",
    "action": "User pours {item_name} into the bowl.",
    "method": "pour"
  }},
  {{
    "narration_id": "P01-...",
    "stage": "RESTOCKING",
    "action": "User puts {item_name} back in the fridge."
  }}
]
"""

INGREDIENT_MAPPING_PROMPT = """You are a culinary ingredient matcher.

TASK: For each observed inventory item, find if there's a matching ground-truth ingredient to get its amount.

RECIPE: {recipe_name}

OBSERVED INVENTORY ITEMS (extracted from video):
{inventory_list}

GROUND-TRUTH INGREDIENTS (from recipe, with amounts):
{ingredients_list}

INSTRUCTIONS:
1. For each inventory item, check if it matches a ground-truth ingredient.
2. Use semantic matching - "parmesan cheese" matches "parmigiano", "pasta box" matches "penne pasta".
3. If a match is found, assign the ingredient's amount and amount_unit to the inventory item.
4. If no match, set amount and amount_unit to null.

OUTPUT FORMAT (JSON):
Return a list with ALL inventory items, enriched with amount info where available:

[
  {{
    "inventory_item": "pasta box",
    "matched_ingredient": "penne pasta",
    "amount": "125",
    "amount_unit": "g"
  }},
  {{
    "inventory_item": "water",
    "matched_ingredient": null,
    "amount": null,
    "amount_unit": null
  }}
]
"""


# ============================================================
# PATH HELPERS
# ============================================================

def get_output_path(output_dir: Path, recipe_id: str, file_type: str, capture_suffix: str = "") -> Path:
    """
    Generate output path with participant subdirectory and recipe-first naming.

    Args:
        output_dir: Base output directory
        recipe_id: Recipe ID (e.g., 'P03_R01')
        file_type: 'inventory', 'lifecycle', or 'edit'
        capture_suffix: Optional capture suffix (e.g., '_C0')

    Returns:
        Path like: output_dir/P03/R01_inventory.json or output_dir/P03/R01_C0_lifecycle.json
    """
    parts = recipe_id.split("_")
    participant = parts[0]
    recipe_num = parts[1] if len(parts) > 1 else recipe_id

    if capture_suffix:
        cap = capture_suffix.lstrip("_")
        filename = f"{recipe_num}_{cap}_{file_type}.json"
    else:
        filename = f"{recipe_num}_{file_type}.json"

    participant_dir = output_dir / participant
    participant_dir.mkdir(parents=True, exist_ok=True)

    return participant_dir / filename


def get_edit_path(output_dir: Path, recipe_id: str, capture_suffix: str = "") -> Path:
    """Generate path for editable inventory file (user verification step)."""
    return get_output_path(output_dir, recipe_id, "edit", capture_suffix)


def get_video_report_path(output_dir: Path, video_id: str) -> Path:
    """Generate path for per-video report file."""
    participant = video_id.split("-")[0]
    report_dir = output_dir / participant / "video_reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    return report_dir / f"{video_id}_report.json"


# ============================================================
# RECIPE/NARRATION LOADING
# ============================================================

def load_recipes(recipes_file: Path = None) -> Dict:
    """Load recipe definitions from complete_recipes.json."""
    if recipes_file is None:
        recipes_file = RECIPES_FILE
    if not recipes_file.exists():
        raise FileNotFoundError(f"Recipes file not found: {recipes_file}")
    with open(recipes_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_recipe_captures(recipes: Dict, recipe_id: str) -> List[Dict]:
    """Get list of captures for a recipe, each with videos and ingredients."""
    if recipe_id not in recipes:
        return []

    recipe = recipes[recipe_id]
    captures = []

    for i, capture in enumerate(recipe.get('captures', [])):
        videos = capture.get('videos', [])
        ingredients = []
        for ing_id, ing in capture.get('ingredients', {}).items():
            ingredients.append({
                "ingredient_id": ing_id,
                "name": ing.get("name", ""),
                "amount": str(ing.get("amount", "N/A")),
                "amount_unit": ing.get("amount_unit", "N/A")
            })
        captures.append({
            "index": i,
            "videos": videos,
            "ingredients": ingredients
        })

    return captures


def load_pickle_data():
    """Load and cache the HD_EPIC_Narrations.pkl file."""
    global _PICKLE_CACHE
    if _PICKLE_CACHE is None:
        if not DEFAULT_PICKLE_PATH.exists():
            raise FileNotFoundError(f"Pickle file not found: {DEFAULT_PICKLE_PATH}")
        with open(DEFAULT_PICKLE_PATH, 'rb') as f:
            _PICKLE_CACHE = pickle.load(f)
    return _PICKLE_CACHE


def load_raw_narrations_for_videos(video_ids: List[str]) -> Optional[str]:
    """Load and concatenate RAW narrations (unfiltered) from pickle file."""
    df = load_pickle_data()
    all_narrations = []

    for video_id in video_ids:
        video_df = df[df['video_id'] == video_id].copy()

        if video_df.empty:
            print(f"  WARNING: No narrations found for video: {video_id}")
            continue

        video_df = video_df.sort_values('start_timestamp')

        lines = []
        for _, row in video_df.iterrows():
            lines.append(f"{row['unique_narration_id']} | {row['narration'].strip()}")

        if lines:
            content = "\n".join(lines)
            all_narrations.append(f"=== VIDEO: {video_id} ===\n{content}")

    if not all_narrations:
        return None

    return "\n\n".join(all_narrations)


def extract_json_from_response(response_text: str) -> Optional[List[Dict]]:
    """Extract JSON array from LLM response."""
    import re

    # Try to find JSON in markdown code block
    code_block_match = re.search(r'```(?:json)?\s*(\[[\s\S]*?\])\s*```', response_text)
    if code_block_match:
        try:
            return json.loads(code_block_match.group(1))
        except json.JSONDecodeError:
            pass

    # Try to find raw JSON array
    json_match = re.search(r'\[[\s\S]*\]', response_text)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass

    return None


# ============================================================
# API CALL WRAPPERS
# ============================================================

def run_inventory_discovery(
    api: GPTClient,
    narrations: str,
    verbose: bool = False,
    use_reasoning: bool = False,
    reasoning_effort: str = "medium"
) -> Optional[List[Dict]]:
    """Run inventory discovery to find root food entities."""

    print(f"Calling {api.model} for inventory discovery" +
          (f" (reasoning={reasoning_effort})..." if use_reasoning else "..."))

    try:
        if use_reasoning:
            response_text, usage = api.responses_create(
                instructions=INVENTORY_DISCOVERY_PROMPT,
                input_text=f"=== NARRATION LOG ===\n\n{narrations}",
                reasoning_effort=reasoning_effort
            )
            if verbose:
                print(f"\n--- Usage ---")
                print(f"  Input tokens: {usage.input_tokens}")
                print(f"  Output tokens: {usage.output_tokens}")
                print(f"  Reasoning tokens: {usage.output_tokens_details.reasoning_tokens}")
        else:
            messages = [
                {"role": "system", "content": INVENTORY_DISCOVERY_PROMPT},
                {"role": "user", "content": f"=== NARRATION LOG ===\n\n{narrations}"}
            ]
            response = api.chat_completion(messages)
            response_text = response.choices[0].message.content

        if verbose:
            print(f"\n--- Raw Response ---\n{response_text}\n---")

        inventory = extract_json_from_response(response_text)

        if inventory is None:
            print("ERROR: Could not parse JSON from response")
            print(f"Response: {response_text[:500]}...")
            return None

        return inventory

    except Exception as e:
        print(f"ERROR: API call failed: {e}")
        return None


def run_ingredient_mapping(
    api: GPTClient,
    inventory: List[Dict],
    ingredients: List[Dict],
    recipe_name: str,
    verbose: bool = False,
    use_reasoning: bool = False,
    reasoning_effort: str = "medium"
) -> Optional[List[Dict]]:
    """Map inventory items to ground-truth ingredients to get amounts."""

    inventory_list = "\n".join([
        f"- {item.get('food_name', 'unknown')}"
        for item in inventory
    ])

    ingredients_list = "\n".join([
        f"- {ing['name']}: {ing['amount']} {ing['amount_unit']}"
        for ing in ingredients
    ])

    prompt = INGREDIENT_MAPPING_PROMPT.format(
        recipe_name=recipe_name,
        inventory_list=inventory_list,
        ingredients_list=ingredients_list
    )

    print(f"Calling {api.model} for ingredient mapping...")

    try:
        if use_reasoning:
            response_text, usage = api.responses_create(
                instructions=prompt,
                input_text="Map the inventory items to ingredients.",
                reasoning_effort=reasoning_effort
            )
            if verbose:
                print(f"  Tokens: {usage.input_tokens} in, {usage.output_tokens} out")
        else:
            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": "Map the inventory items to ingredients."}
            ]
            response = api.chat_completion(messages)
            response_text = response.choices[0].message.content

        if verbose:
            print(f"\n--- Raw Response ---\n{response_text}\n---")

        mappings = extract_json_from_response(response_text)

        if mappings is None:
            print("ERROR: Could not parse JSON from response")
            return None

        return mappings

    except Exception as e:
        print(f"ERROR: API call failed: {e}")
        return None


def run_lifecycle_for_item(
    api: GPTClient,
    item_name: str,
    narrations: str,
    verbose: bool = False,
    use_reasoning: bool = False,
    reasoning_effort: str = "medium"
) -> Optional[List[Dict]]:
    """Run lifecycle event tracking for a single inventory item."""

    prompt = INVENTORY_LIFECYCLE_PROMPT.format(item_name=item_name)
    user_content = f"=== NARRATION_LOG ===\n\n{narrations}"

    try:
        if use_reasoning:
            response_text, usage = api.responses_create(
                instructions=prompt,
                input_text=user_content,
                reasoning_effort=reasoning_effort
            )
            if verbose:
                print(f"    Tokens: {usage.input_tokens} in, {usage.output_tokens} out, {usage.output_tokens_details.reasoning_tokens} reasoning")
        else:
            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_content}
            ]
            response = api.chat_completion(messages)
            response_text = response.choices[0].message.content

        if verbose:
            print(f"\n--- Raw Response ---\n{response_text}\n---")

        events = extract_json_from_response(response_text)

        if events is None:
            print(f"    WARNING: Could not parse JSON from response")
            return None

        for evt in events:
            evt['item_name'] = item_name

        return events

    except Exception as e:
        print(f"    ERROR: API call failed: {e}")
        return None


def run_lifecycle_events(
    api: GPTClient,
    inventory: List[Dict],
    narrations: str,
    verbose: bool = False,
    use_reasoning: bool = False,
    reasoning_effort: str = "medium"
) -> Dict[str, List[Dict]]:
    """Run lifecycle event tracking for each inventory item separately."""

    all_events = {}

    for i, item in enumerate(inventory):
        item_name = item.get('food_name', 'unknown')
        print(f"  [{i+1}/{len(inventory)}] Processing: {item_name}...", end=" ", flush=True)

        events = run_lifecycle_for_item(
            api, item_name, narrations, verbose,
            use_reasoning, reasoning_effort
        )

        if events:
            all_events[item_name] = events
            print(f"found {len(events)} events")
        else:
            all_events[item_name] = []
            print("no events found")

    return all_events


# ============================================================
# VIDEO REPORT GENERATION
# ============================================================

def build_video_to_recipes_map(recipes: Dict, participant: str = None) -> Dict[str, List[Dict]]:
    """Build a mapping from video_id to list of recipes that use that video."""
    video_map = {}

    for recipe_id, recipe in recipes.items():
        if participant and not recipe_id.startswith(f"{participant}_"):
            continue

        recipe_name = recipe.get('name', 'Unknown')
        for cap_idx, capture in enumerate(recipe.get('captures', [])):
            for video_id in capture.get('videos', []):
                if video_id not in video_map:
                    video_map[video_id] = []
                video_map[video_id].append({
                    "recipe_id": recipe_id,
                    "recipe_name": recipe_name,
                    "capture_index": cap_idx
                })

    return video_map


def generate_video_reports(
    output_dir: Path,
    recipes: Dict,
    participant: str,
    verbose: bool = False
) -> None:
    """Generate per-video reports showing which recipes use each video and potential duplicates."""
    print(f"\n{'='*60}")
    print("GENERATING VIDEO REPORTS")
    print(f"{'='*60}")

    video_map = build_video_to_recipes_map(recipes, participant)

    participant_dir = output_dir / participant
    edit_files = list(participant_dir.glob("R*_edit.json")) + list(participant_dir.glob("R*_C*_edit.json"))

    recipe_inventory = {}
    for edit_file in edit_files:
        with open(edit_file, 'r') as f:
            data = json.load(f)
            recipe_id = data.get('recipe_id')
            cap_idx = data.get('capture_index', 0)
            suffix = f"_C{cap_idx}" if len(recipes.get(recipe_id, {}).get('captures', [])) > 1 else ""
            key = f"{recipe_id}{suffix}"
            recipe_inventory[key] = {
                "recipe_id": recipe_id,
                "recipe_name": data.get('recipe_name', 'Unknown'),
                "capture_index": cap_idx,
                "videos": data.get('videos', []),
                "inventory": data.get('inventory', [])
            }

    for video_id, recipe_list in sorted(video_map.items()):
        report = {
            "video_id": video_id,
            "recipes_in_video": recipe_list,
            "inventory_by_recipe": {},
            "potential_duplicates": []
        }

        all_items = []
        for recipe_info in recipe_list:
            recipe_id = recipe_info['recipe_id']
            cap_idx = recipe_info['capture_index']
            suffix = f"_C{cap_idx}" if len(recipes.get(recipe_id, {}).get('captures', [])) > 1 else ""
            key = f"{recipe_id}{suffix}"

            if key in recipe_inventory:
                inv_data = recipe_inventory[key]
                if video_id in inv_data.get('videos', []):
                    items = []
                    for item in inv_data.get('inventory', []):
                        narr_id = item.get('narration_id', '')
                        if narr_id.startswith(video_id):
                            items.append({
                                "food_name": item.get('food_name'),
                                "narration_id": item.get('narration_id'),
                                "matched_ingredient": item.get('matched_ingredient'),
                                "include": item.get('include', True)
                            })
                            all_items.append((item.get('food_name', '').lower(), recipe_id))

                    if items:
                        report["inventory_by_recipe"][recipe_id] = items

        food_counts = Counter(name for name, _ in all_items)
        duplicates = {}
        for name, count in food_counts.items():
            if count > 1:
                recipes_with_item = list(set(rid for n, rid in all_items if n == name))
                duplicates[name] = recipes_with_item

        if duplicates:
            report["potential_duplicates"] = [
                {"item": name, "found_in_recipes": rids}
                for name, rids in sorted(duplicates.items())
            ]

        report_path = get_video_report_path(output_dir, video_id)
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        if verbose or report["potential_duplicates"]:
            dup_count = len(report["potential_duplicates"])
            print(f"  {video_id}: {len(recipe_list)} recipes, {dup_count} potential duplicates")

    print(f"\n  Generated {len(video_map)} video reports")


def group_captures_by_video_range(recipes: Dict, recipe_ids: List[str]) -> Dict[tuple, List[Dict]]:
    """
    Group recipe-captures by their unique video range.

    Returns a dict mapping video_range (tuple of video IDs) to list of recipe-capture info.
    This allows running inventory discovery once per unique video range.
    """
    video_range_to_captures = {}

    for recipe_id in recipe_ids:
        if recipe_id not in recipes:
            continue

        recipe = recipes[recipe_id]
        recipe_name = recipe.get('name', 'Unknown')
        captures = get_recipe_captures(recipes, recipe_id)

        for cap_idx, capture in enumerate(captures):
            videos = capture['videos']
            video_range = tuple(sorted(videos))  # Sorted tuple for consistent key

            if video_range not in video_range_to_captures:
                video_range_to_captures[video_range] = []

            suffix = f"_C{cap_idx}" if len(captures) > 1 else ""
            video_range_to_captures[video_range].append({
                "recipe_id": recipe_id,
                "recipe_name": recipe_name,
                "capture_index": cap_idx,
                "suffix": suffix,
                "videos": videos,
                "ingredients": capture['ingredients']
            })

    return video_range_to_captures


def build_optimized_discovery_plan(recipes: Dict, recipe_ids: List[str]) -> Dict:
    """
    Build an optimized discovery plan that handles both exact matches AND subset relationships.

    Returns a dict with:
        - 'discovery_ranges': list of video ranges that need actual GPT calls
        - 'range_to_captures': maps each unique range to its recipe-captures
        - 'subset_map': maps subset ranges to their superset range (for filtering)
        - 'stats': optimization statistics
    """
    # First, group by exact video range
    video_range_to_captures = group_captures_by_video_range(recipes, recipe_ids)
    unique_ranges = list(video_range_to_captures.keys())

    # Find subset relationships: for each range, find its minimal superset
    subset_map = {}  # maps subset_range -> superset_range
    for r1 in unique_ranges:
        s1 = set(r1)
        best_superset = None
        best_size = float('inf')
        for r2 in unique_ranges:
            if r1 == r2:
                continue
            s2 = set(r2)
            if s1 < s2 and len(s2) < best_size:  # r1 is proper subset of r2
                best_superset = r2
                best_size = len(s2)
        if best_superset:
            subset_map[r1] = best_superset

    # Ranges that need discovery calls = unique_ranges - subsets
    discovery_ranges = [r for r in unique_ranges if r not in subset_map]

    # Calculate stats
    total_captures = sum(len(caps) for caps in video_range_to_captures.values())

    return {
        'discovery_ranges': discovery_ranges,
        'range_to_captures': video_range_to_captures,
        'subset_map': subset_map,
        'stats': {
            'total_captures': total_captures,
            'unique_ranges': len(unique_ranges),
            'discovery_calls': len(discovery_ranges),
            'exact_match_savings': total_captures - len(unique_ranges),
            'subset_savings': len(subset_map),
            'total_savings': total_captures - len(discovery_ranges)
        }
    }


def filter_inventory_by_videos(inventory: List[Dict], video_ids: List[str]) -> List[Dict]:
    """
    Filter inventory items to only those from specific videos.

    Used to derive subset results from superset discovery.
    """
    video_prefixes = tuple(video_ids)
    filtered = []
    for item in inventory:
        narration_id = item.get('narration_id', '')
        # Check if narration_id starts with any of the video IDs
        for vid in video_ids:
            if narration_id.startswith(vid):
                filtered.append(item)
                break
    return filtered


def list_recipes(recipes: Dict, output_dir: Path = None):
    """List available recipes with their captures and edit file status."""
    print("\nAvailable Recipes:")
    print("=" * 80)

    for recipe_id, recipe in sorted(recipes.items()):
        captures = get_recipe_captures(recipes, recipe_id)
        if not captures:
            continue

        total_videos = sum(len(c['videos']) for c in captures)

        # Check for edit files if output_dir provided
        if output_dir:
            edit_count = 0
            for cap in captures:
                suffix = f"_C{cap['index']}" if len(captures) > 1 else ""
                edit_file = get_edit_path(output_dir, recipe_id, suffix)
                if edit_file.exists():
                    edit_count += 1
            status = "✓" if edit_count == len(captures) else f"({edit_count}/{len(captures)})"
        else:
            status = ""

        print(f"\n{recipe_id}: {recipe.get('name', 'Unknown')} {status}")
        print(f"  Captures: {len(captures)}" + (" (run per-capture!)" if len(captures) > 1 else ""))

        for cap in captures:
            print(f"    [{cap['index']}] Videos: {', '.join(cap['videos'])}")

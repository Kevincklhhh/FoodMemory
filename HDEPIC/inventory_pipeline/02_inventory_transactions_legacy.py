#!/usr/bin/env python3
"""
Inventory Lifecycle Pipeline (Two-Step Workflow)

Recipe-based pipeline using GPT-5.2 to:
1. Discover inventory items (root food entities) from narrations
2. Track lifecycle events (RETRIEVAL, ACCESS, DISPENSING, RETURN) for each item

=== TWO-STEP WORKFLOW ===

STEP 1: Discovery (outputs editable files for user verification)
    python 02_inventory_transactions.py --stage discovery --participant P03

    Outputs:
    - {participant}/{recipe}_edit.json    : Editable inventory (user verifies/filters)
    - {participant}/video_reports/{video}_report.json : Per-video report showing
        which recipes use this video and potential duplicate items

STEP 2: Lifecycle (processes user-verified inventory)
    python 02_inventory_transactions.py --stage lifecycle --participant P03

    Reads: {recipe}_edit.json (user-verified, items with "include": false are skipped)
    Outputs:
    - {participant}/{recipe}_inventory.json : Final verified inventory
    - {participant}/{recipe}_lifecycle.json : Lifecycle events for each item

=== LEGACY USAGE (runs both stages without user verification) ===

    # Run full pipeline for a recipe
    python 02_inventory_transactions.py --recipe-id P01_R03 --raw --stage both

    # List available recipes
    python 02_inventory_transactions.py --list-recipes
"""

import os
import json
import argparse
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
DEFAULT_FILTER_DIR = _PROJECT_ROOT / "outputs" / "01_narrations" / "filtered"
DEFAULT_OUTPUT_DIR = _PROJECT_ROOT / "outputs" / "02_inventory" / "lifecycle"
DEFAULT_VIDEO_OUTPUT_DIR = _PROJECT_ROOT / "outputs" / "02_inventory" / "video_lifecycle"
RECIPES_FILE = _PROJECT_ROOT / "data" / "hd-epic-annotations" / "high-level" / "complete_recipes.json"
DEFAULT_PICKLE_PATH = _PROJECT_ROOT / "data" / "hd-epic-annotations" / "narrations-and-action-segments" / "HD_EPIC_Narrations.pkl"

# Cache for pickle data
_PICKLE_CACHE = None


class GPTClient:
    """Wrapper for Azure OpenAI API with support for multiple endpoints and Responses API."""

    def __init__(self, model: str = "gpt-4o", use_reasoning: bool = False):
        self.model = model
        self.use_reasoning = use_reasoning

        # Use endpoint 2 for gpt-5/gpt-5.2/o3, endpoint 1 for others
        if model in ["gpt-5", "gpt-5.2", "gpt-5-chat", "o3-pro", "o3"]:
            api_key = os.getenv("AZURE_OPENAI_API_KEY_2")
            endpoint = os.getenv("AZURE_OPENAI_ENDPOINT_2", "").strip()
            # Use newer API version for Responses API with reasoning
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


# Prompts
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


def load_recipes(recipes_file: Path) -> Dict:
    """Load recipe definitions from complete_recipes.json."""
    if not recipes_file.exists():
        raise FileNotFoundError(f"Recipes file not found: {recipes_file}")

    with open(recipes_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_output_path(output_dir: Path, recipe_id: str, file_type: str, capture_suffix: str = "") -> Path:
    """
    Generate output path with participant subdirectory and recipe-first naming.

    Args:
        output_dir: Base output directory
        recipe_id: Recipe ID (e.g., 'P03_R01')
        file_type: 'inventory' or 'lifecycle'
        capture_suffix: Optional capture suffix (e.g., '_C0')

    Returns:
        Path like: output_dir/P03/R01_inventory.json or output_dir/P03/R01_C0_lifecycle.json
    """
    # Parse recipe_id: "P03_R01" -> participant="P03", recipe_num="R01"
    parts = recipe_id.split("_")
    participant = parts[0]
    recipe_num = parts[1] if len(parts) > 1 else recipe_id

    # Build filename: R01_inventory.json or R01_C0_lifecycle.json
    if capture_suffix:
        # Remove leading underscore if present for cleaner naming
        cap = capture_suffix.lstrip("_")
        filename = f"{recipe_num}_{cap}_{file_type}.json"
    else:
        filename = f"{recipe_num}_{file_type}.json"

    # Create participant subdirectory
    participant_dir = output_dir / participant
    participant_dir.mkdir(parents=True, exist_ok=True)

    return participant_dir / filename


def get_edit_path(output_dir: Path, recipe_id: str, capture_suffix: str = "") -> Path:
    """
    Generate path for editable inventory file (user verification step).

    Returns: output_dir/P03/R01_edit.json or output_dir/P03/R01_C0_edit.json
    """
    return get_output_path(output_dir, recipe_id, "edit", capture_suffix)


def get_video_report_path(output_dir: Path, video_id: str) -> Path:
    """
    Generate path for per-video report file.

    Returns: output_dir/{participant}/video_reports/{video_id}_report.json
    """
    # Extract participant from video_id (e.g., "P03-20240217-131219" -> "P03")
    participant = video_id.split("-")[0]

    report_dir = output_dir / participant / "video_reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    return report_dir / f"{video_id}_report.json"


def build_video_to_recipes_map(recipes: Dict, participant: str = None) -> Dict[str, List[Dict]]:
    """
    Build a mapping from video_id to list of recipes that use that video.

    Returns: {video_id: [{recipe_id, recipe_name, capture_index}, ...]}
    """
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
    """
    Generate per-video reports showing which recipes use each video
    and what inventory items were found (with potential duplicates).
    """
    print(f"\n{'='*60}")
    print("GENERATING VIDEO REPORTS")
    print(f"{'='*60}")

    # Build video -> recipes mapping
    video_map = build_video_to_recipes_map(recipes, participant)

    # Collect all inventory items from edit files
    participant_dir = output_dir / participant
    edit_files = list(participant_dir.glob("R*_edit.json")) + list(participant_dir.glob("R*_C*_edit.json"))

    # Build recipe -> inventory mapping
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

    # Generate report for each video
    for video_id, recipe_list in sorted(video_map.items()):
        report = {
            "video_id": video_id,
            "recipes_in_video": recipe_list,
            "inventory_by_recipe": {},
            "potential_duplicates": []
        }

        # Collect inventory items for this video from each recipe
        all_items = []  # (food_name, recipe_id)
        for recipe_info in recipe_list:
            recipe_id = recipe_info['recipe_id']
            cap_idx = recipe_info['capture_index']
            suffix = f"_C{cap_idx}" if len(recipes.get(recipe_id, {}).get('captures', [])) > 1 else ""
            key = f"{recipe_id}{suffix}"

            if key in recipe_inventory:
                inv_data = recipe_inventory[key]
                # Only include items from videos that match
                if video_id in inv_data.get('videos', []):
                    items = []
                    for item in inv_data.get('inventory', []):
                        # Check if item's narration belongs to this video
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

        # Find potential duplicates (same food name across multiple recipes)
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

        # Save report
        report_path = get_video_report_path(output_dir, video_id)
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        if verbose or report["potential_duplicates"]:
            dup_count = len(report["potential_duplicates"])
            print(f"  {video_id}: {len(recipe_list)} recipes, {dup_count} potential duplicates")

    print(f"\n  Generated {len(video_map)} video reports")


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


def get_recipe_videos(recipes: Dict, recipe_id: str, capture_index: int = None) -> List[str]:
    """Get list of video IDs for a recipe (optionally for a specific capture)."""
    if recipe_id not in recipes:
        return []

    recipe = recipes[recipe_id]
    captures = recipe.get('captures', [])

    if capture_index is not None:
        if 0 <= capture_index < len(captures):
            return captures[capture_index].get('videos', [])
        return []

    # Return all videos if no capture specified
    videos = []
    for capture in captures:
        videos.extend(capture.get('videos', []))
    return videos


def get_recipe_ingredients(recipes: Dict, recipe_id: str, capture_index: int = None) -> List[Dict]:
    """Get list of ingredients for a recipe (optionally for a specific capture)."""
    if recipe_id not in recipes:
        return []

    recipe = recipes[recipe_id]
    captures = recipe.get('captures', [])
    ingredients = []

    if capture_index is not None:
        if 0 <= capture_index < len(captures):
            capture = captures[capture_index]
            for ing_id, ing in capture.get('ingredients', {}).items():
                ingredients.append({
                    "ingredient_id": ing_id,
                    "name": ing.get("name", ""),
                    "amount": str(ing.get("amount", "N/A")),
                    "amount_unit": ing.get("amount_unit", "N/A")
                })
    else:
        for capture in captures:
            for ing_id, ing in capture.get('ingredients', {}).items():
                ingredients.append({
                    "ingredient_id": ing_id,
                    "name": ing.get("name", ""),
                    "amount": str(ing.get("amount", "N/A")),
                    "amount_unit": ing.get("amount_unit", "N/A")
                })

    return ingredients


def get_all_unique_videos(recipes: Dict, participant: str = None) -> List[str]:
    """
    Extract all unique video IDs from all recipes.

    Args:
        recipes: Recipe dictionary from complete_recipes.json
        participant: Optional participant filter (e.g., 'P01')

    Returns:
        Sorted list of unique video IDs
    """
    video_set = set()

    for recipe_id, recipe in recipes.items():
        # Filter by participant if specified
        if participant and not recipe_id.startswith(f"{participant}_"):
            continue

        for capture in recipe.get('captures', []):
            for video_id in capture.get('videos', []):
                video_set.add(video_id)

    return sorted(video_set)


def process_single_video(
    api: 'GPTClient',
    video_id: str,
    output_dir: Path,
    use_raw: bool = True,
    use_reasoning: bool = False,
    reasoning_effort: str = "high",
    verbose: bool = False
) -> Optional[Dict]:
    """
    Process a single video: inventory discovery + lifecycle events.

    Args:
        api: GPT client instance
        video_id: Video ID to process
        output_dir: Directory to save output
        use_raw: Whether to use raw narrations from pickle
        use_reasoning: Whether to use reasoning mode
        reasoning_effort: Reasoning effort level
        verbose: Whether to print verbose output

    Returns:
        Video result dict or None if failed
    """
    print(f"\n{'='*60}")
    print(f"Processing video: {video_id}")
    print(f"{'='*60}")

    # Load narrations for this video
    if use_raw:
        narrations = load_raw_narrations_for_videos([video_id])
    else:
        # Would need filter_dir passed in for filtered mode
        print("  ERROR: Filtered mode not supported for video processing")
        return None

    if narrations is None:
        print(f"  ERROR: No narrations found for video {video_id}")
        return None

    line_count = len(narrations.strip().split('\n'))
    print(f"  Loaded {line_count} narrations")

    # Stage 1: Inventory Discovery
    print(f"\n  --- INVENTORY DISCOVERY ---")
    inventory = run_inventory_discovery(
        api, narrations, verbose,
        use_reasoning=use_reasoning,
        reasoning_effort=reasoning_effort
    )

    if not inventory:
        print(f"  WARNING: No inventory items found for {video_id}")
        inventory = []
    else:
        print(f"  Found {len(inventory)} inventory items")

    # Stage 2: Lifecycle Events (for each item)
    print(f"\n  --- LIFECYCLE EVENTS ---")
    lifecycle_events = {}

    if inventory:
        lifecycle_events = run_lifecycle_events(
            api, inventory, narrations, verbose,
            use_reasoning=use_reasoning,
            reasoning_effort=reasoning_effort
        )

    total_events = sum(len(evts) for evts in lifecycle_events.values())
    print(f"  Found {total_events} lifecycle events across {len(lifecycle_events)} items")

    # Build result
    result = {
        "video_id": video_id,
        "narration_count": line_count,
        "inventory": inventory,
        "lifecycle_events": lifecycle_events
    }

    # Save to file
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{video_id}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    print(f"\n  Saved to: {output_file}")

    return result


def aggregate_recipe_from_videos(
    recipe_id: str,
    recipe: Dict,
    capture_index: int,
    video_output_dir: Path,
    output_dir: Path,
    api: 'GPTClient' = None,
    use_reasoning: bool = True,
    reasoning_effort: str = "high",
    verbose: bool = False
) -> Optional[Dict]:
    """
    Aggregate video outputs into recipe-level inventory and lifecycle.

    Args:
        recipe_id: Recipe ID (e.g., 'P01_R03')
        recipe: Recipe definition from complete_recipes.json
        capture_index: Which capture to aggregate
        video_output_dir: Directory containing video JSON files
        output_dir: Directory to save recipe outputs
        api: Optional GPT client for ingredient mapping
        use_reasoning: Whether to use reasoning for mapping
        reasoning_effort: Reasoning effort level
        verbose: Verbose output

    Returns:
        Aggregated result dict or None if failed
    """
    recipe_name = recipe.get('name', 'Unknown')
    captures = recipe.get('captures', [])

    if capture_index >= len(captures):
        print(f"  ERROR: Invalid capture index {capture_index}")
        return None

    capture = captures[capture_index]
    video_ids = capture.get('videos', [])
    ingredients = []
    for ing_id, ing in capture.get('ingredients', {}).items():
        ingredients.append({
            "ingredient_id": ing_id,
            "name": ing.get("name", ""),
            "amount": str(ing.get("amount", "N/A")),
            "amount_unit": ing.get("amount_unit", "N/A")
        })

    suffix = f"_C{capture_index}" if len(captures) > 1 else ""

    print(f"\n  Aggregating {recipe_id}{suffix}: {', '.join(video_ids)}")

    # Load video outputs
    merged_inventory = []
    merged_lifecycle = {}
    seen_narration_ids = set()

    for video_id in video_ids:
        video_file = video_output_dir / f"{video_id}.json"
        if not video_file.exists():
            print(f"    WARNING: Video output not found: {video_file}")
            continue

        with open(video_file, 'r') as f:
            video_data = json.load(f)

        # Merge inventory (dedupe by narration_id)
        for item in video_data.get('inventory', []):
            narr_id = item.get('narration_id')
            if narr_id and narr_id not in seen_narration_ids:
                merged_inventory.append(item)
                seen_narration_ids.add(narr_id)

        # Merge lifecycle events
        for item_name, events in video_data.get('lifecycle_events', {}).items():
            if item_name not in merged_lifecycle:
                merged_lifecycle[item_name] = []
            for evt in events:
                narr_id = evt.get('narration_id')
                if narr_id and narr_id not in seen_narration_ids:
                    merged_lifecycle[item_name].append(evt)
                    seen_narration_ids.add(narr_id)

    print(f"    Merged: {len(merged_inventory)} inventory items, {sum(len(e) for e in merged_lifecycle.values())} events")

    # Ingredient mapping (if API client provided)
    if api and merged_inventory and ingredients:
        print(f"    Mapping to {len(ingredients)} recipe ingredients...")
        mappings = run_ingredient_mapping(
            api, merged_inventory, ingredients, recipe_name,
            verbose=verbose, use_reasoning=use_reasoning,
            reasoning_effort=reasoning_effort
        )
        if mappings:
            mapping_dict = {m['inventory_item']: m for m in mappings}
            for item in merged_inventory:
                food_name = item.get('food_name', '')
                if food_name in mapping_dict:
                    m = mapping_dict[food_name]
                    item['matched_ingredient'] = m.get('matched_ingredient')
                    item['amount'] = m.get('amount')
                    item['amount_unit'] = m.get('amount_unit')
            matched_count = sum(1 for item in merged_inventory if item.get('matched_ingredient'))
            print(f"    Matched: {matched_count}/{len(merged_inventory)} items with amounts")

    # Save inventory file
    output_dir.mkdir(parents=True, exist_ok=True)

    inventory_result = {
        "recipe_id": recipe_id,
        "recipe_name": recipe_name,
        "capture_index": capture_index,
        "videos": video_ids,
        "inventory": merged_inventory
    }
    inventory_file = get_output_path(output_dir, recipe_id, "inventory", suffix)
    with open(inventory_file, 'w', encoding='utf-8') as f:
        json.dump(inventory_result, f, indent=2)

    # Save lifecycle file
    lifecycle_result = {
        "recipe_id": recipe_id,
        "recipe_name": recipe_name,
        "capture_index": capture_index,
        "videos": video_ids,
        "events_by_item": merged_lifecycle
    }
    lifecycle_file = get_output_path(output_dir, recipe_id, "lifecycle", suffix)
    with open(lifecycle_file, 'w', encoding='utf-8') as f:
        json.dump(lifecycle_result, f, indent=2)

    print(f"    Saved: {inventory_file.parent.name}/{inventory_file.name}, {lifecycle_file.name}")

    return lifecycle_result


def load_filtered_narrations_for_videos(filter_dir: Path, video_ids: List[str]) -> Optional[str]:
    """Load and concatenate filtered narrations for multiple videos."""
    all_narrations = []

    for video_id in video_ids:
        filtered_file = filter_dir / f"filtered_{video_id}.txt"

        if not filtered_file.exists():
            print(f"  WARNING: Filtered file not found: {filtered_file}")
            continue

        with open(filtered_file, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if content:
                all_narrations.append(f"=== VIDEO: {video_id} ===\n{content}")

    if not all_narrations:
        return None

    return "\n\n".join(all_narrations)


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
    """
    Load and concatenate RAW narrations (unfiltered) from pickle file.

    Skips the filter step - loads all narrations directly.
    """
    df = load_pickle_data()

    all_narrations = []

    for video_id in video_ids:
        # Filter to this video
        video_df = df[df['video_id'] == video_id].copy()

        if video_df.empty:
            print(f"  WARNING: No narrations found for video: {video_id}")
            continue

        # Sort by timestamp
        video_df = video_df.sort_values('start_timestamp')

        # Format as "narration_id | narration"
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


def run_lifecycle_for_item(
    api: GPTClient,
    item_name: str,
    narrations: str,
    verbose: bool = False,
    use_reasoning: bool = False,
    reasoning_effort: str = "medium"
) -> Optional[List[Dict]]:
    """Run lifecycle event tracking for a single inventory item."""

    # Format prompt with item name
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

        # Add item_name to each event
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

    # Format lists for prompt
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


def list_recipes(recipes: Dict, filter_dir: Path):
    """List available recipes with their captures and filter status."""
    print("\nAvailable Recipes:")
    print("=" * 80)

    for recipe_id, recipe in sorted(recipes.items()):
        captures = get_recipe_captures(recipes, recipe_id)
        if not captures:
            continue

        # Count total videos and available filters
        total_videos = sum(len(c['videos']) for c in captures)
        all_videos = [v for c in captures for v in c['videos']]
        available = [v for v in all_videos if (filter_dir / f"filtered_{v}.txt").exists()]
        missing = [v for v in all_videos if v not in available]

        status = "✓" if len(available) == total_videos else f"({len(available)}/{total_videos})"

        print(f"\n{recipe_id}: {recipe.get('name', 'Unknown')} {status}")
        print(f"  Captures: {len(captures)}" + (" (run per-capture!)" if len(captures) > 1 else ""))

        for cap in captures:
            cap_status = "✓" if all((filter_dir / f"filtered_{v}.txt").exists() for v in cap['videos']) else "✗"
            print(f"    [{cap['index']}] {cap_status} Videos: {', '.join(cap['videos'])}")

        if missing:
            print(f"  Missing filters: {', '.join(missing)}")


def main():
    parser = argparse.ArgumentParser(
        description="Inventory lifecycle pipeline using GPT-5"
    )
    parser.add_argument(
        '--recipe-id',
        help='Recipe ID to process (e.g., P01_R03)'
    )
    parser.add_argument(
        '--list-recipes',
        action='store_true',
        help='List available recipes'
    )
    parser.add_argument(
        '--stage',
        choices=['discovery', 'lifecycle', 'both'],
        default='both',
        help='Which stage to run (default: both)'
    )
    parser.add_argument(
        '--filter-dir',
        type=Path,
        default=DEFAULT_FILTER_DIR,
        help='Directory containing filtered narrations'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help='Output directory for results'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Print verbose output including raw API responses'
    )
    parser.add_argument(
        '--test-api',
        action='store_true',
        help='Just test API connectivity'
    )
    parser.add_argument(
        '--model',
        default='gpt-5.2',
        choices=['gpt-4o', 'o4-mini', 'gpt-4.1-mini', 'gpt-5', 'gpt-5.2', 'o3'],
        help='Model to use (default: gpt-5.2)'
    )
    parser.add_argument(
        '--reasoning',
        action='store_true',
        help='Enable reasoning mode using Responses API (gpt-5/o3 only)'
    )
    parser.add_argument(
        '--reasoning-effort',
        choices=['low', 'medium', 'high'],
        default='high',
        help='Reasoning effort level (default: high)'
    )
    parser.add_argument(
        '--capture',
        type=int,
        default=None,
        help='Specific capture index to process (0-indexed). Required for multi-capture recipes.'
    )
    parser.add_argument(
        '--all-captures',
        action='store_true',
        help='Process all captures separately for multi-capture recipes'
    )
    parser.add_argument(
        '--raw',
        action='store_true',
        help='Use raw narrations from pickle (skip filter step)'
    )
    parser.add_argument(
        '--video-ids',
        nargs='+',
        help='Specific video IDs to process (use with --raw, without --recipe-id)'
    )

    # New video-first processing modes
    parser.add_argument(
        '--process-videos',
        action='store_true',
        help='Process all unique videos (video-first mode)'
    )
    parser.add_argument(
        '--process-video',
        help='Process a single video by ID (e.g., P01-20240202-161354)'
    )
    parser.add_argument(
        '--aggregate-recipes',
        action='store_true',
        help='Aggregate video outputs into recipe-level files'
    )
    parser.add_argument(
        '--participant',
        help='Filter to specific participant (e.g., P01)'
    )
    parser.add_argument(
        '--video-output-dir',
        type=Path,
        default=DEFAULT_VIDEO_OUTPUT_DIR,
        help='Output directory for video-level results'
    )
    parser.add_argument(
        '--skip-mapping',
        action='store_true',
        help='Skip ingredient mapping during aggregation (faster, no API calls)'
    )
    parser.add_argument(
        '--process-recipes',
        action='store_true',
        help='Process all recipes for participant using recipe-based mode (all videos concatenated)'
    )

    args = parser.parse_args()

    # Load recipes
    print("Loading recipes...")
    try:
        recipes = load_recipes(RECIPES_FILE)
        print(f"  Loaded {len(recipes)} recipes")
    except Exception as e:
        print(f"ERROR: Failed to load recipes: {e}")
        return

    # List recipes if requested
    if args.list_recipes:
        list_recipes(recipes, args.filter_dir)
        return

    # Handle aggregate-recipes mode
    if args.aggregate_recipes:
        print("\n" + "="*70)
        print("AGGREGATING VIDEO OUTPUTS TO RECIPES")
        print("="*70)

        # Initialize API for ingredient mapping (unless skipped)
        api = None
        if not args.skip_mapping:
            print(f"Initializing {args.model} API for ingredient mapping...")
            try:
                api = GPTClient(args.model, use_reasoning=args.reasoning)
                print("  API initialized successfully")
            except Exception as e:
                print(f"  WARNING: Failed to initialize API: {e}")
                print("  Proceeding without ingredient mapping")

        recipe_ids = [r for r in recipes.keys() if not args.participant or r.startswith(f"{args.participant}_")]

        if args.recipe_id:
            recipe_ids = [args.recipe_id]

        print(f"Processing {len(recipe_ids)} recipes...")

        for recipe_id in sorted(recipe_ids):
            recipe = recipes[recipe_id]
            captures = recipe.get('captures', [])

            for cap_idx in range(len(captures)):
                aggregate_recipe_from_videos(
                    recipe_id, recipe, cap_idx,
                    args.video_output_dir, args.output_dir,
                    api=api,
                    use_reasoning=args.reasoning,
                    reasoning_effort=args.reasoning_effort,
                    verbose=args.verbose
                )

        print(f"\n{'='*70}")
        print("AGGREGATION COMPLETE")
        print(f"{'='*70}")
        return

    # Initialize API (needed for video processing and recipe processing)
    print(f"Initializing {args.model} API" + (" with reasoning..." if args.reasoning else "..."))
    try:
        api = GPTClient(args.model, use_reasoning=args.reasoning)
        print("  API initialized successfully")
    except Exception as e:
        print(f"ERROR: Failed to initialize API: {e}")
        return

    # Test API if requested
    if args.test_api:
        print("\nTesting API with simple query...")
        try:
            response = api.chat_completion([
                {"role": "user", "content": "Say 'API test successful' and nothing else."}
            ], max_tokens=50)
            print(f"  Response: {response.choices[0].message.content}")
            print("\nAPI test passed!")
        except Exception as e:
            print(f"  ERROR: {e}")
        return

    # Handle process-recipes mode (recipe-based batch processing)
    if args.process_recipes:
        recipe_ids = [r for r in recipes.keys()
                      if not args.participant or r.startswith(f"{args.participant}_")]

        if not recipe_ids:
            print(f"ERROR: No recipes found" + (f" for {args.participant}" if args.participant else ""))
            return

        # Count total captures
        total_captures = sum(len(recipes[r].get('captures', [])) for r in recipe_ids)

        # ============================================================
        # STEP 1: DISCOVERY (outputs _edit.json for user verification)
        # ============================================================
        if args.stage in ['discovery', 'both']:
            print(f"\n{'='*70}")
            print(f"STEP 1: INVENTORY DISCOVERY")
            print(f"{'='*70}")
            print(f"Found {len(recipe_ids)} recipes" + (f" for {args.participant}" if args.participant else ""))
            print(f"Total captures to process: {total_captures}")

            processed = 0
            for recipe_id in sorted(recipe_ids):
                recipe = recipes[recipe_id]
                recipe_name = recipe.get('name', 'Unknown')
                captures = get_recipe_captures(recipes, recipe_id)

                for cap_idx, capture in enumerate(captures):
                    processed += 1
                    videos = capture['videos']
                    ingredients = capture['ingredients']
                    suffix = f"_C{cap_idx}" if len(captures) > 1 else ""

                    print(f"\n{'#'*70}")
                    print(f"[{processed}/{total_captures}] {recipe_id}{suffix}: {recipe_name}")
                    print(f"Videos: {', '.join(videos)}")
                    print(f"{'#'*70}")

                    # Load narrations from all videos
                    narrations = load_raw_narrations_for_videos(videos)
                    if narrations is None:
                        print("ERROR: No narrations found")
                        continue

                    line_count = len(narrations.strip().split('\n'))
                    print(f"  Loaded {line_count} narrations across {len(videos)} videos")

                    # Inventory Discovery
                    print(f"\n  --- INVENTORY DISCOVERY ---")
                    inventory = run_inventory_discovery(
                        api, narrations, args.verbose,
                        use_reasoning=args.reasoning,
                        reasoning_effort=args.reasoning_effort
                    )

                    if not inventory:
                        print("  WARNING: No inventory items found")
                        inventory = []
                    else:
                        print(f"  Found {len(inventory)} inventory items")

                    # Ingredient Mapping
                    if inventory and ingredients:
                        print(f"\n  --- INGREDIENT MAPPING ---")
                        print(f"  Mapping to {len(ingredients)} recipe ingredients...")
                        mappings = run_ingredient_mapping(
                            api, inventory, ingredients, recipe_name, args.verbose,
                            use_reasoning=args.reasoning,
                            reasoning_effort=args.reasoning_effort
                        )

                        if mappings:
                            mapping_dict = {m['inventory_item']: m for m in mappings}
                            for item in inventory:
                                food_name = item.get('food_name', '')
                                if food_name in mapping_dict:
                                    m = mapping_dict[food_name]
                                    item['matched_ingredient'] = m.get('matched_ingredient')
                                    item['amount'] = m.get('amount')
                                    item['amount_unit'] = m.get('amount_unit')
                            matched = sum(1 for item in inventory if item.get('matched_ingredient'))
                            print(f"  Matched: {matched}/{len(inventory)} items with amounts")

                    # Add 'include' field for user editing (default: True)
                    for item in inventory:
                        item['include'] = True

                    # Save as EDIT file (for user verification)
                    edit_file = get_edit_path(args.output_dir, recipe_id, suffix)
                    with open(edit_file, 'w', encoding='utf-8') as f:
                        json.dump({
                            "recipe_id": recipe_id,
                            "recipe_name": recipe_name,
                            "capture_index": cap_idx,
                            "videos": videos,
                            "ground_truth_ingredients": ingredients,
                            "inventory": inventory,
                            "_instructions": "Set 'include': false to exclude items from lifecycle tracking"
                        }, f, indent=2)
                    print(f"\n  Saved: {edit_file.parent.name}/{edit_file.name}")

            # Generate video reports after all discoveries
            if args.participant:
                generate_video_reports(args.output_dir, recipes, args.participant, args.verbose)

            if args.stage == 'discovery':
                print(f"\n{'='*70}")
                print(f"STEP 1 COMPLETE: DISCOVERY")
                print(f"{'='*70}")
                print(f"  Processed {processed} captures from {len(recipe_ids)} recipes")
                print(f"\n  NEXT STEPS:")
                print(f"  1. Review _edit.json files in {args.output_dir}/{args.participant}/")
                print(f"  2. Set 'include': false for items to exclude")
                print(f"  3. Check video_reports/ for potential duplicates")
                print(f"  4. Run: python {Path(__file__).name} --stage lifecycle --participant {args.participant}")
                return

        # ============================================================
        # STEP 2: LIFECYCLE (reads verified _edit.json files)
        # ============================================================
        if args.stage in ['lifecycle', 'both']:
            print(f"\n{'='*70}")
            print(f"STEP 2: LIFECYCLE TRACKING")
            print(f"{'='*70}")

            processed = 0
            for recipe_id in sorted(recipe_ids):
                recipe = recipes[recipe_id]
                recipe_name = recipe.get('name', 'Unknown')
                captures = get_recipe_captures(recipes, recipe_id)

                for cap_idx, capture in enumerate(captures):
                    processed += 1
                    videos = capture['videos']
                    suffix = f"_C{cap_idx}" if len(captures) > 1 else ""

                    # Read from EDIT file
                    edit_file = get_edit_path(args.output_dir, recipe_id, suffix)
                    if not edit_file.exists():
                        print(f"\n  SKIP {recipe_id}{suffix}: No edit file found (run discovery first)")
                        continue

                    with open(edit_file, 'r') as f:
                        edit_data = json.load(f)

                    # Filter inventory: only items with include=True
                    all_inventory = edit_data.get('inventory', [])
                    inventory = [item for item in all_inventory if item.get('include', True)]
                    excluded_count = len(all_inventory) - len(inventory)

                    print(f"\n{'#'*70}")
                    print(f"[{processed}/{total_captures}] {recipe_id}{suffix}: {recipe_name}")
                    print(f"  Inventory: {len(inventory)} items ({excluded_count} excluded)")
                    print(f"{'#'*70}")

                    if not inventory:
                        print("  WARNING: No inventory items to process")
                        continue

                    # Load narrations for lifecycle tracking
                    narrations = load_raw_narrations_for_videos(videos)
                    if narrations is None:
                        print("ERROR: No narrations found")
                        continue

                    # Lifecycle Events
                    print(f"\n  --- LIFECYCLE EVENTS ---")
                    events_by_item = run_lifecycle_events(
                        api, inventory, narrations, args.verbose,
                        use_reasoning=args.reasoning,
                        reasoning_effort=args.reasoning_effort
                    )

                    total_events = sum(len(evts) for evts in events_by_item.values())
                    print(f"  Found {total_events} lifecycle events across {len(events_by_item)} items")

                    # Save final inventory (verified items only)
                    inventory_file = get_output_path(args.output_dir, recipe_id, "inventory", suffix)
                    with open(inventory_file, 'w', encoding='utf-8') as f:
                        json.dump({
                            "recipe_id": recipe_id,
                            "recipe_name": recipe_name,
                            "capture_index": cap_idx,
                            "videos": videos,
                            "inventory": inventory
                        }, f, indent=2)
                    print(f"  Saved: {inventory_file.parent.name}/{inventory_file.name}")

                    # Save lifecycle
                    lifecycle_file = get_output_path(args.output_dir, recipe_id, "lifecycle", suffix)
                    with open(lifecycle_file, 'w', encoding='utf-8') as f:
                        json.dump({
                            "recipe_id": recipe_id,
                            "recipe_name": recipe_name,
                            "capture_index": cap_idx,
                            "videos": videos,
                            "events_by_item": events_by_item
                        }, f, indent=2)
                    print(f"  Saved: {lifecycle_file.parent.name}/{lifecycle_file.name}")

            print(f"\n{'='*70}")
            print(f"STEP 2 COMPLETE: LIFECYCLE")

        print(f"\n{'='*70}")
        print(f"RECIPE PROCESSING COMPLETE")
        print(f"Processed {processed} captures from {len(recipe_ids)} recipes")
        print(f"{'='*70}")
        return

    # Handle process-video mode (single video)
    if args.process_video:
        process_single_video(
            api, args.process_video, args.video_output_dir,
            use_raw=True,  # Always use raw for video-first mode
            use_reasoning=args.reasoning,
            reasoning_effort=args.reasoning_effort,
            verbose=args.verbose
        )
        return

    # Handle process-videos mode (all unique videos)
    if args.process_videos:
        all_videos = get_all_unique_videos(recipes, args.participant)
        print(f"\n{'='*70}")
        print(f"VIDEO-FIRST PROCESSING MODE")
        print(f"{'='*70}")
        print(f"Found {len(all_videos)} unique videos" + (f" for {args.participant}" if args.participant else ""))

        # Check which videos already have outputs
        existing = [v for v in all_videos if (args.video_output_dir / f"{v}.json").exists()]
        remaining = [v for v in all_videos if v not in existing]

        print(f"  Already processed: {len(existing)}")
        print(f"  Remaining: {len(remaining)}")

        if not remaining:
            print("\nAll videos already processed!")
            return

        print(f"\nProcessing {len(remaining)} videos...")

        for i, video_id in enumerate(remaining):
            print(f"\n[{i+1}/{len(remaining)}]", end="")
            process_single_video(
                api, video_id, args.video_output_dir,
                use_raw=True,
                use_reasoning=args.reasoning,
                reasoning_effort=args.reasoning_effort,
                verbose=args.verbose
            )

        print(f"\n{'='*70}")
        print(f"VIDEO PROCESSING COMPLETE")
        print(f"{'='*70}")
        return

    # Check recipe-id is provided for single-recipe mode
    if not args.recipe_id:
        print("ERROR: --recipe-id is required (use --list-recipes to see available recipes)")
        print("       Or use --process-recipes --participant P01  (batch process all recipes)")
        return

    # Get captures for recipe
    captures = get_recipe_captures(recipes, args.recipe_id)
    if not captures:
        print(f"ERROR: Recipe '{args.recipe_id}' not found or has no captures")
        return

    recipe_name = recipes[args.recipe_id].get('name', 'Unknown')

    # Determine which captures to process
    if len(captures) > 1:
        # Multi-capture recipe
        if args.all_captures:
            capture_indices = list(range(len(captures)))
            print(f"\nRecipe: {args.recipe_id} - {recipe_name}")
            print(f"Processing ALL {len(captures)} captures separately...")
        elif args.capture is not None:
            if args.capture < 0 or args.capture >= len(captures):
                print(f"ERROR: Invalid capture index {args.capture}. Recipe has {len(captures)} captures (0-{len(captures)-1})")
                return
            capture_indices = [args.capture]
        else:
            print(f"\nERROR: Recipe '{args.recipe_id}' has {len(captures)} captures (independent cooking sessions).")
            print("       You must specify either:")
            print(f"         --capture <0-{len(captures)-1}>  (process one capture)")
            print("         --all-captures           (process all captures separately)")
            print("\nCaptures:")
            for cap in captures:
                print(f"  [{cap['index']}] Videos: {', '.join(cap['videos'])}")
            return
    else:
        # Single-capture recipe
        capture_indices = [0]

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Process each capture
    for cap_idx in capture_indices:
        capture = captures[cap_idx]
        videos = capture['videos']
        ingredients = capture['ingredients']

        # Output suffix for multi-capture recipes
        suffix = f"_C{cap_idx}" if len(captures) > 1 else ""

        print(f"\n{'#'*70}")
        print(f"Recipe: {args.recipe_id} - {recipe_name}" + (f" [Capture {cap_idx}]" if len(captures) > 1 else ""))
        print(f"Videos: {', '.join(videos)}")
        print(f"{'#'*70}")

        # Load narrations
        if args.raw:
            print(f"\nLoading RAW narrations (skipping filter)...")
            narrations = load_raw_narrations_for_videos(videos)
            if narrations is None:
                print("ERROR: No narrations found in pickle for videos in this capture")
                continue
        else:
            print(f"\nLoading filtered narrations...")
            narrations = load_filtered_narrations_for_videos(args.filter_dir, videos)
            if narrations is None:
                print("ERROR: No filtered narrations found for videos in this capture")
                continue

        line_count = len(narrations.strip().split('\n'))
        print(f"  Loaded {line_count} total lines across {len(videos)} videos")

        inventory = None

        # Stage 1: Inventory Discovery
        if args.stage in ['discovery', 'both']:
            print(f"\n{'='*60}")
            print("STAGE 1: INVENTORY DISCOVERY")
            print(f"{'='*60}")

            inventory = run_inventory_discovery(
                api, narrations, args.verbose,
                use_reasoning=args.reasoning,
                reasoning_effort=args.reasoning_effort
            )

            if inventory:
                print(f"\nFound {len(inventory)} inventory items:")
                for item in inventory:
                    print(f"  - {item.get('food_name', 'N/A')} ({item.get('narration_id', 'N/A')})")

                # Get ground-truth ingredients and run mapping
                if ingredients:
                    print(f"\n--- Mapping to {len(ingredients)} ground-truth ingredients ---")
                    mappings = run_ingredient_mapping(
                        api, inventory, ingredients, recipe_name, args.verbose,
                        use_reasoning=args.reasoning,
                        reasoning_effort=args.reasoning_effort
                    )

                    if mappings:
                        # Enrich inventory with amount info
                        mapping_dict = {m['inventory_item']: m for m in mappings}
                        for item in inventory:
                            food_name = item.get('food_name', '')
                            if food_name in mapping_dict:
                                m = mapping_dict[food_name]
                                item['matched_ingredient'] = m.get('matched_ingredient')
                                item['amount'] = m.get('amount')
                                item['amount_unit'] = m.get('amount_unit')

                        print("\nInventory with amounts:")
                        for item in inventory:
                            amount = item.get('amount')
                            unit = item.get('amount_unit')
                            matched = item.get('matched_ingredient')
                            if amount and unit:
                                print(f"  - {item['food_name']}: {amount} {unit} (← {matched})")
                            else:
                                print(f"  - {item['food_name']}: (no amount)")

                # Save results
                output_file = get_output_path(args.output_dir, args.recipe_id, "inventory", suffix)
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        "recipe_id": args.recipe_id,
                        "recipe_name": recipe_name,
                        "capture_index": cap_idx,
                        "videos": videos,
                        "inventory": inventory
                    }, f, indent=2)
                print(f"\nSaved to: {output_file.parent.name}/{output_file.name}")
            else:
                print("ERROR: Inventory discovery failed")
                continue

        # Load existing inventory if only running lifecycle
        if args.stage == 'lifecycle':
            inventory_file = get_output_path(args.output_dir, args.recipe_id, "inventory", suffix)
            if inventory_file.exists():
                with open(inventory_file, 'r') as f:
                    data = json.load(f)
                    inventory = data.get('inventory', [])
                print(f"Loaded existing inventory: {len(inventory)} items")
            else:
                print(f"ERROR: No inventory file found. Run discovery first.")
                continue

        # Stage 2: Lifecycle Events (per-item)
        if args.stage in ['lifecycle', 'both']:
            print(f"\n{'='*60}")
            print("STAGE 2: LIFECYCLE EVENTS (per-item)")
            print(f"{'='*60}")

            events_by_item = run_lifecycle_events(
                api, inventory, narrations, args.verbose,
                use_reasoning=args.reasoning,
                reasoning_effort=args.reasoning_effort
            )

            # Count total events
            total_events = sum(len(evts) for evts in events_by_item.values())

            if total_events > 0:
                print(f"\nFound {total_events} lifecycle events across {len(events_by_item)} items:")

                for item_name, evts in events_by_item.items():
                    if evts:
                        print(f"\n  {item_name} ({len(evts)} events):")
                        for evt in evts:
                            stage = evt.get('stage', 'N/A')
                            action = evt.get('action', evt.get('description', 'N/A'))
                            method = evt.get('method', '')
                            method_str = f" [{method}]" if method else ""
                            print(f"    [{stage}]{method_str} {action}")

                # Save results
                output_file = get_output_path(args.output_dir, args.recipe_id, "lifecycle", suffix)
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        "recipe_id": args.recipe_id,
                        "recipe_name": recipe_name,
                        "capture_index": cap_idx,
                        "videos": videos,
                        "events_by_item": events_by_item
                    }, f, indent=2)
                print(f"\nSaved to: {output_file.parent.name}/{output_file.name}")
            else:
                print("WARNING: No lifecycle events found for any item")

    print(f"\n{'='*60}")
    print("COMPLETE")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()

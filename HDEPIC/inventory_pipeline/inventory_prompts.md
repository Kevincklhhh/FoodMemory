inventory_discovery_prompt = """
You are an "Ingredient Spotter" for a cooking video.

Your task is to analyze the entire narration log and identify the FIRST APPEARANCE of every distinct food item.



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

    "narration ID": P01-20240202-161948-269

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
1. TARGET ITEM: "{target_item_name}" (e.g., "Milk")
2. NARRATION LOG: Chronological list of all user actions.

YOUR GOAL:
Scan the log and extract ONLY the logistic events for the Target Item.
Ignore all other food items. Ignore cooking/processing steps.

### THE LIFECYCLE STAGES (Look for these):

1. **RETRIEVAL (Start of Cycle):**
   - Bringing the item from a storage zone (Fridge, Cupboard, Pantry) to the workspace.
   - *Example:* "Pick up {target_item_name} from fridge."

2. **ACCESS (Prep for Dispensing):**
   - Opening the container physically.
   - *Example:* "Unscrew cap of {target_item_name}", "Remove lid", "Unwrap foil."

3. **DISPENSING (Quantity Reduction):**
   - The moment quantity leaves the Source Container.
   - **Pouring/Scooping:** "Pour {target_item_name} into bowl."
   - **Dispensing Cuts:** Cutting a portion OFF the main block (e.g., slicing a piece of butter/cheese). *Keep this.*
   - **Scanning:** Taking one unit from a multipack (e.g., taking one egg from the carton).

4. **RESTOCKING (End of Cycle):**
   - Returning the Source Container to a storage zone.
   - *Example:* "Place {target_item_name} back in fridge."
   - *Note:* If the item is EMPTY/TRASHED, record that as the end of lifecycle.

### DISCARD (Noise Filter):
- **Culinary Processing:** Chopping/dicing the portion *after* it has been dispensed (e.g., "Chop the butter slice").
- **Cooking:** Boiling, frying, baking.
- **Other Items:** Actions related to "{other_food}".

### OUTPUT FORMAT (JSON):
Return a list of Lifecycle Events for this item.
If the item is never touched, return an empty list.

Example Output for Target: "Butter":
[
  {
    "stage": "RETRIEVAL",
    "timestamp": 12.5,
    "action": "User retrieves Butter from fridge."
  },
  {
    "stage": "ACCESS",
    "timestamp": 15.2,
    "action": "User unwraps the Butter block."
  },
  {
    "stage": "DISPENSING",
    "timestamp": 18.0,
    "action": "User slices a piece of Butter off the block into the pan.",
    "method": "cut_portion"
  },
  {
    "stage": "RESTOCKING",
    "timestamp": 45.0,
    "action": "User puts the remaining Butter block back in the fridge."
  }
]
"""
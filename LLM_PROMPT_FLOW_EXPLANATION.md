# LLM Prompt Flow & KG Update Explanation

## Overview: From Narration to Knowledge Graph Update

```
Narration → Entity Extraction → KG Retrieval → LLM Prompt → LLM Response → Validation → KG Update
```

---

## 1. LLM Prompt Structure

The prompt consists of **3 main parts**:

### Part 1: System Prompt (Simple & Direct)
```
"You help track food in a kitchen by responding with JSON."
```
- Very minimal - just tells LLM its role
- Expects structured JSON output

### Part 2: Few-Shot Examples
Two examples showing the LLM what to do:

**Example 1 - Creating New Food (TAKE)**
```
Input: "Pick up milk bottle from fridge at 21.4-22.0s"
Output: {
  "update_type": "TAKE",
  "target_food_id": null,
  "updates": {"location": null},
  "history_entry": {
    "start_time": 21.4,
    "end_time": 22.0,
    "action": "pick up",
    "narration_text": "pick up milk bottle",
    "location_context": "zone_fridge_1"
  },
  "new_food_info": {
    "name": "milk bottle",
    "state": "unknown",
    "quantity": "1"
  }
}
```

**Example 2 - Updating Existing Food (PLACE)**
```
Input: "Place milk bottle in fridge at 37.0-38.0s"
Current: food_milk_bottle_1 is in hand
Output: {
  "update_type": "PLACE",
  "target_food_id": "food_milk_bottle_1",
  "updates": {"location": "zone_fridge_1"},
  "history_entry": {
    "start_time": 37.0,
    "end_time": 38.0,
    "action": "place",
    "narration_text": "place milk bottle",
    "location_context": "zone_fridge_1"
  }
}
```

### Part 3: Current Context + New Narration

**If Existing Food Found:**
```
CURRENT FOOD NODE:
- Food ID: food_mug_1
- Name: mug
- State: unknown
- Location: in hand (null)
- Quantity: 1
- Previous interactions: 1

RECENT INTERACTIONS:
  - pick up at 8.8s

NEW ACTION TO PROCESS:
- Time: 9.34s to 11.16s
- Narration: "Pick up a mug from the lower shelf..."
```

**If No Existing Food:**
```
NO EXISTING FOOD NODE FOUND - This appears to be a new food item

NEW ACTION TO PROCESS:
- Time: 8.85s to 9.36s
- Narration: "Stretch the left hand inside the cupboard..."
```

---

## 2. LLM Output Format (JSON Structure)

The LLM must respond with a JSON object containing these fields:

### Required Fields (Always)

#### `update_type` (string)
**What is it?** Specifies the type of kitchen action being performed.

**Valid values:**
- `"TAKE"` - Picking up food from a location (food becomes "in hand")
- `"PLACE"` - Putting food down at a location (food goes to a zone)
- `"MODIFY_STATE"` - Changing food properties (stirring, cutting, opening, etc.)
- `"CREATE_NEW"` - First time seeing this food item

**Note:** LLM can output other action verbs (like "TURN", "OPEN", "POUR"), and these get **automatically mapped** to the 4 valid types during validation.

Mapping examples:
```python
"TURN" → "MODIFY_STATE"
"OPEN" → "MODIFY_STATE"
"POUR" → "MODIFY_STATE"
```

Special auto-conversion rule:
```python
# If LLM provides new_food_info but no target_food_id:
# Automatically convert to CREATE_NEW regardless of update_type
if new_food_info exists AND target_food_id is null:
    update_type = "CREATE_NEW"
```

#### `history_entry` (object)
**What is it?** Records what happened in this narration.

**Required fields:**
- `start_time` (float): When action started (seconds)
- `end_time` (float): When action ended (seconds)
- `action` (string): Action verb (e.g., "pick up", "place", "turn")
- `narration_text` (string): Summary of what happened
- `location_context` (string): Where it happened (zone_id or location name)

**Purpose:** This gets appended to the food's interaction history timeline.

### Conditional Fields

#### `target_food_id` (string or null)
- **Required for:** TAKE, PLACE, MODIFY_STATE (updating existing food)
- **Must be null for:** CREATE_NEW (creating new food)
- **Value:** The food_id of the existing food node (e.g., "food_mug_1")

#### `new_food_info` (object)
- **Required for:** CREATE_NEW
- **Not needed for:** TAKE, PLACE, MODIFY_STATE
- **Fields:**
  - `name` (string, required): Food name (e.g., "mug", "milk bottle")
  - `state` (string, optional): Initial state (default: "unknown")
  - `quantity` (string, optional): Amount (default: "unknown")

#### `updates` (object)
- **Optional for all types**
- **Contains key-value pairs of food properties to update**
- Common fields:
  - `location`: Zone ID or null (null = "in hand")
  - `state`: New state (e.g., "opened", "checked", "chopped")
  - `quantity`: New quantity

---

## 3. Update Type Semantics

### `TAKE` - Picking Up Food
**Meaning:** Moving food from a location INTO hand

**Example:**
```
Narration: "Pick up milk bottle from fridge"
LLM Output:
{
  "update_type": "TAKE",
  "target_food_id": "food_milk_1",  // Existing food at fridge
  "updates": {"location": null},    // null = "in hand"
  ...
}
```

**KG Changes:**
- Sets `location` to `null` (in hand)
- Adds interaction to history

### `PLACE` - Putting Down Food
**Meaning:** Moving food from hand TO a location

**Example:**
```
Narration: "Place milk bottle in fridge"
LLM Output:
{
  "update_type": "PLACE",
  "target_food_id": "food_milk_1",  // Food currently in hand
  "updates": {"location": "zone_fridge_1"},
  ...
}
```

**KG Changes:**
- Sets `location` to the specified zone
- Adds interaction to history

### `MODIFY_STATE` - Changing Food Properties
**Meaning:** Manipulating food without changing location

**Example:**
```
Narration: "Turn the mug to check how clean it is"
LLM Output:
{
  "update_type": "MODIFY_STATE",
  "target_food_id": "food_mug_2",
  "updates": {"state": "checked"},
  ...
}
```

**KG Changes:**
- Updates specified properties (state, quantity, etc.)
- Adds interaction to history

### `CREATE_NEW` - First Time Seeing Food
**Meaning:** This food item doesn't exist in KG yet

**Example:**
```
Narration: "Pick up a mug from shelf"
LLM Output:
{
  "update_type": "CREATE_NEW",
  "target_food_id": null,
  "new_food_info": {
    "name": "mug",
    "state": "unknown",
    "quantity": "1"
  },
  ...
}
```

**KG Changes:**
- Creates new food node with generated food_id
- Sets initial properties from new_food_info
- Adds first interaction to history

---

## 4. Validation Process

After LLM responds, the JSON goes through validation (`llm_context.py:validate_update_command`):

### Step 1: Type Normalization
```python
# Convert common action verbs to valid types
"TURN" → "MODIFY_STATE"
"OPEN" → "MODIFY_STATE"
"POUR" → "MODIFY_STATE"
```

### Step 2: Auto-correction
```python
# If has new_food_info but no target_food_id, must be CREATE_NEW
if new_food_info exists AND not target_food_id:
    update_type = "CREATE_NEW"
```

### Step 3: Required Field Checks
- ✓ Must have `update_type` (one of 4 valid types)
- ✓ Must have `history_entry` with all 5 required fields
- ✓ CREATE_NEW must have `new_food_info` with `name`
- ✓ Other types must have `target_food_id`

**If validation fails:** Retry LLM call (up to 3 attempts)

---

## 5. KG Update Execution

After validation passes, `kg_update_executor.py:execute_kg_update` performs the actual KG modifications:

### For CREATE_NEW:

```python
1. Extract food name from new_food_info
2. Determine initial location (from updates or null)
3. Call add_food_node() to create new node:
   - Generates food_id: "food_{name}_{count}"
   - Sets name, state, quantity, location
   - Records first_seen_time
4. Call add_interaction() to record first interaction
```

**Result:** New food node added to `kg['foods']`

### For TAKE, PLACE, MODIFY_STATE:

```python
1. Get food_id from target_food_id
2. Verify food exists in KG
3. If updates provided:
   - Handle location conversion (name → zone_id)
   - Call update_food_node() to modify properties
4. Call add_interaction() to record interaction
```

**Result:** Existing food node updated, interaction appended

### Zone Creation (Automatic)

If any location is mentioned:
```python
1. Check if zone exists (by name)
2. If not, create new zone:
   - Zone ID: "zone_{name}_{count}"
   - Zone name: from narration
   - Zone type: "Storage"
```

---

## 6. Complete Example Flow

**Narration:** "Pick up a mug from the lower shelf"

### Step 1: Entity Extraction
```python
{
  'food_entity': 'mug',
  'location_entity': 'lower shelf',
  'action': 'pick up',
  'narration': 'Pick up a mug from the lower shelf...',
  'start_time': 9.34,
  'end_time': 11.16
}
```

### Step 2: KG Retrieval
```python
find_food(kg, name_pattern='mug')
# Finds: [food_mug_1] (last interaction: 9.4s)
existing_food = food_mug_1
```

### Step 3: LLM Prompt
```
CURRENT FOOD NODE:
- Food ID: food_mug_1
- Name: mug
- Location: in hand (null)
...

NEW ACTION TO PROCESS:
- Time: 9.34s to 11.16s
- Narration: "Pick up a mug from the lower shelf..."
```

### Step 4: LLM Response
```json
{
  "update_type": "TAKE",
  "target_food_id": null,  // ← PROBLEM! Should be "food_mug_1"
  "new_food_info": {"name": "mug", ...}
}
```

### Step 5: Validation (Auto-fix)
```python
# Has new_food_info but no target_food_id
# → Auto-convert to CREATE_NEW
update_type = "CREATE_NEW"
```

### Step 6: KG Update
```python
# Creates food_mug_2 (duplicate!)
add_food_node(kg, name="mug", ...)
add_interaction(kg, "food_mug_2", ...)
```

---

## 7. Current Issues

### Issue 1: Ambiguous Few-Shot Examples
**Problem:** Example 1 shows `"TAKE"` with `target_food_id: null` and `new_food_info`

This confuses the LLM:
- Should TAKE have `target_food_id` (updating existing) or `null` (creating new)?
- When creating new food, should I use `TAKE` or `CREATE_NEW`?

**Fix:** Separate examples for:
1. Creating new food: Use `CREATE_NEW` explicitly
2. Taking existing food: Use `TAKE` with `target_food_id`

### Issue 2: No "Update Existing In-Hand Food" Example
**Problem:** No example shows what to do when:
- Food already exists IN HAND
- New narration is about the same food

**Fix:** Add Example 3:
```
Input: "Turn the milk bottle to read the label"
Current: food_milk_bottle_1 is in hand
Output: {
  "update_type": "MODIFY_STATE",
  "target_food_id": "food_milk_bottle_1",  // ← Key: use existing food
  "updates": {"state": "inspected"},
  ...
}
```

### Issue 3: Auto-Conversion Rule Creates Duplicates
**Problem:** The validation rule:
```python
if new_food_info exists AND not target_food_id:
    update_type = "CREATE_NEW"
```

This **always creates a new node** when LLM provides `new_food_info`, even if existing food was passed to LLM!

**Fix:** Either:
1. Don't auto-convert - make LLM fix it via retry
2. Add stricter validation to prevent creating duplicate "in hand" foods

---

## 8. Data Flow Summary

```
┌─────────────────┐
│   Narration     │
│ "Pick up mug"   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Entity Extract  │ food_entity: "mug"
│                 │ location: "shelf"
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  KG Retrieval   │ find_food(kg, "mug")
│                 │ → existing_food or None
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Build Prompt   │ System + Examples + Context + Narration
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   LLM Call      │ ollama.chat(model, messages)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Parse Response  │ Extract JSON from text
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Validate      │ Check fields, normalize types
│                 │ Auto-convert if needed
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Execute KG     │ CREATE_NEW → add_food_node()
│  Update         │ Others → update_food_node()
│                 │ All → add_interaction()
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Updated KG     │ New/modified food node
│                 │ + interaction history
└─────────────────┘
```

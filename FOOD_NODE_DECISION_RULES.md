# Food Node Decision Rules: Add New vs Update Existing

## Problem Overview

The system is creating **duplicate food nodes** instead of updating existing ones. This analysis uses evidence from 200+ snapshots in `kg_snapshots/`.

**Note:** The KG is built across **multiple videos**, not just one. Each interaction now includes `video_id` to support multi-video tracking and visualization.

## Current Decision Flow

### 1. Entity Extraction
```
Narration: "Pick up a mug from the lower shelf..."
→ food_entity: "mug"
→ location_entity: "cupboard"
→ action: "pick up"
```

### 2. Food Matching Logic (kg_sequential_pipeline.py:187-200)

```python
matching_foods = find_food(kg, name_pattern=food_name, location=location)
in_hand_foods = find_food(kg, name_pattern=food_name, location=None)

all_matches = in_hand_foods + matching_foods
existing_food = all_matches[0] if all_matches else None  # Most recent
```

### 3. LLM Decision
The LLM receives context about `existing_food` and decides:
- **UPDATE**: Use `target_food_id` of existing food
- **CREATE_NEW**: Add new food node with `new_food_info`

## Real Examples from Snapshots

### Example 1: Duplicate Mug Creation (Narration 2→3)

**Snapshot 2 (narration-2, time 8.85-9.36s):**
- Action: "Stretch hand to reach mug"
- Result: Created `food_mug_13` (new node)
- Also updated `food_mug_2` (added interaction)
- Total foods: 14

**Snapshot 3 (narration-3, time 9.34-11.16s):**
- Action: "Pick up a mug"
- Existing nodes at this time:
  - `food_mug_1`: first_seen=8.85s, last_interaction=9.36s, location=None
  - `food_mug_2`: first_seen=9.34s, last_interaction=9.36s, location=None
  - `food_mug_13`: first_seen=8.85s, last_interaction=9.36s, location=None
  - `food_mug_14`: first_seen=9.34s, last_interaction=11.16s, location=None
- Result: **Created `food_mug_15` (NEW duplicate!)** instead of updating `food_mug_14`
- Total foods: 15 ❌

**Problem:** The system should have updated `food_mug_14` but created a duplicate instead.

### Example 2: Multiple Orange Tracking

From the snapshots, we see 3 distinct oranges correctly tracked:
- `food_orange_7`: first_seen=53.31s, "pick up orange" from counter
- `food_orange_8`: first_seen=55.26s, "pick up orange" from mesh
- `food_orange_9`: first_seen=56.6s, "pick up a third orange" from mesh

**These are correct!** Each represents a physically distinct orange.

## Root Cause: Location Matching Bug

### Problem in `find_food()` (kg_storage.py:121-124)

```python
if location:
    food_location = food_data.get("location")
    if food_location is None:  # ❌ BUG HERE!
        continue  # Excludes foods "in hand"
```

**Impact:**
- When searching for `location="cupboard"`, foods with `location=null` (in hand) are excluded
- This breaks tracking continuity when food is picked up (location becomes null)
- LLM doesn't see the existing food node, so it creates a new one

## Correct Decision Rules

### RULE 1: Update Existing Node
**Update existing food when it's the SAME physical object:**

Conditions:
1. **Temporal continuity**: New action time is close to last interaction (< 5s gap)
2. **Name match**: Same food name (e.g., "mug", "milk bottle")
3. **Quantity consistency**: Narration doesn't indicate "another" or "second"
4. **Logical flow**: Action sequence makes sense (pick up → turn → place)
5. **In-hand priority**: Food with location=null (in hand) takes priority over located food

Example (single video):
```
video: P01-20240202-110250
t=9.34s: "Pick up mug" → food_mug_2 (location=null, video_id=P01-20240202-110250)
t=11.15s: "Turn mug" → UPDATE food_mug_2 (still in hand, video_id=P01-20240202-110250)
t=14.2s: "Put mug under nozzle" → UPDATE food_mug_2 (location=zone_coffee_machine, video_id=P01-20240202-110250)
```

Example (cross-video tracking):
```
video1: P01-20240202-110250
t=30.0s: "Place milk in fridge" → food_milk_1 (location=zone_fridge_1, video_id=P01-20240202-110250)

video2: P01-20240202-161354
t=5.0s: "Take milk from fridge" → UPDATE food_milk_1 (location=null, video_id=P01-20240202-161354)
```

### RULE 2: Create New Node
**Create new food when it's a DIFFERENT physical object:**

Conditions:
1. **Explicit quantifiers**: "another", "second", "third", "a different"
2. **Different locations**: Two foods with same name but different locations
3. **Temporal gap**: Large time gap suggests different object (> 30s within same video, or different video)
4. **Parallel existence**: Narration indicates multiple items exist simultaneously
5. **Contradictory state**: E.g., one is "closed" while new action is "pick up unopened"
6. **Cross-video boundary**: Same food can persist across videos (e.g., milk in fridge), but video_id helps track provenance

Example:
```
t=53.31s: "pick up orange" → CREATE food_orange_7
t=55.26s: "pick up orange" → CREATE food_orange_8 (different orange!)
t=56.6s: "pick up a third orange" → CREATE food_orange_9
```

### RULE 3: Ambiguous Cases - Default to Update
When uncertain:
- **Default to UPDATE** if name matches and recent (< 5s gap)
- Reason: Single object moving through workflow is more common than multiple identical items

## Bug Fix Required

### Fix 1: Improve find_food() to handle "in hand" items

**Current (buggy):**
```python
if location:
    food_location = food_data.get("location")
    if food_location is None:
        continue  # ❌ Excludes in-hand foods
```

**Proposed fix:**
```python
if location:
    food_location = food_data.get("location")

    # Allow null location IF food was recently in this location
    if food_location is None:
        # Check if recent interaction mentions this location
        recent_locations = [
            i.get('location_context', '')
            for i in food_data.get('interaction_history', [])[-3:]
        ]
        location_match = any(location.lower() in str(loc).lower() for loc in recent_locations)
        if not location_match:
            continue
    elif location.lower() not in food_location.lower():
        continue
```

### Fix 2: Improve LLM prompt with temporal reasoning

Add to prompt (llm_context.py:72-80):
```python
DECISION RULES:
1. UPDATE if same food name AND time gap < 5 seconds
2. UPDATE if food is "in hand" (location=null) and name matches
3. CREATE_NEW only if narration says "another", "second", "different"
4. When in doubt, UPDATE (single object workflows are more common)
```

### Fix 3: Add deduplication check

After LLM decision, verify:
```python
if update_type == "CREATE_NEW":
    # Check if very similar food exists
    recent_foods = [f for f in kg['foods'].values()
                    if f['name'] == new_food_name
                    and f['first_seen_time'] > (current_time - 10)]
    if recent_foods:
        # Force UPDATE instead
        update_type = "MODIFY_STATE"
        target_food_id = recent_foods[0]['food_id']
```

## Expected Behavior After Fix

### Good: Mug workflow (single video)
```
video: P01-20240202-110250
t=8.85s: "Stretch to reach mug" → CREATE food_mug_1 (first time)
  interaction_history: [{..., video_id: "P01-20240202-110250"}]
t=9.34s: "Pick up mug" → UPDATE food_mug_1 (same mug)
  interaction_history: [{...}, {..., video_id: "P01-20240202-110250"}]
t=11.15s: "Turn mug" → UPDATE food_mug_1 (same mug)
t=14.2s: "Put mug under nozzle" → UPDATE food_mug_1 (same mug)
Final: 1 mug node with 4 interactions, all from same video ✓
```

### Good: Multiple oranges
```
t=53.31s: "pick up orange" → CREATE food_orange_1
t=55.26s: "pick up orange" → CREATE food_orange_2 (different orange)
t=56.6s: "pick up a third orange" → CREATE food_orange_3
Final: 3 orange nodes ✓
```

### Good: Cross-video food tracking
```
video1: P01-20240202-110250
t=120s: "Place olive oil on counter" → CREATE food_olive_oil_1
  interaction_history: [{..., video_id: "P01-20240202-110250"}]
  location: zone_counter_1

video2: P01-20240202-161354
t=10s: "Pick up olive oil from counter" → UPDATE food_olive_oil_1
  interaction_history: [
    {..., video_id: "P01-20240202-110250"},
    {..., video_id: "P01-20240202-161354"}  ← spans videos
  ]
Final: 1 olive oil node with interactions from 2 videos ✓
```

## Validation

After applying fixes, verify with:
```bash
# Run on first 20 narrations
python3 kg_sequential_pipeline.py --csv participant_P01_narrations.csv \
    --kg food_kg_test.json --snapshots kg_snapshots_test \
    --limit 20 --verbose --model gpt-oss:120b

# Check food count growth
python3 -c "
import json
for i in range(1, 21):
    s = json.load(open(f'kg_snapshots_test/snapshot_P01-20240202-110250-{i}.json'))
    print(f'Narration {i}: {s[\"snapshot_info\"][\"snapshot_metadata\"][\"num_foods\"]} foods')
"
```

Expected: ~5-8 foods (not 15+) after 20 narrations dealing with a single mug and coffee capsule.

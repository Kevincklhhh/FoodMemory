# Changelog: Multi-Video KG Tracking

## Date: 2025-10-16

## Summary
Added `video_id` tracking to food interaction history to support knowledge graphs built across multiple videos.

## Problem
The KG system processes narrations from multiple videos, but the interaction history didn't track which video each interaction came from. This made it difficult to:
1. Trace which video an interaction occurred in
2. Support visualizer features that load specific videos when clicking snapshots
3. Understand food provenance across multi-video sessions

## Solution
Modified the interaction history structure to include `video_id` for every interaction entry.

## Changes Made

### 1. kg_storage.py:204-242
**Modified `add_interaction()` function:**
- Added `video_id: str = None` parameter
- Updated docstring to document the new parameter
- Added `"video_id": video_id` to the `interaction_entry` dictionary

**Before:**
```python
def add_interaction(kg, food_id, start_time, end_time, action,
                   narration_text, location_context):
    interaction_entry = {
        "start_time": start_time,
        "end_time": end_time,
        "action": action,
        "narration_text": narration_text,
        "location_context": location_context
    }
```

**After:**
```python
def add_interaction(kg, food_id, start_time, end_time, action,
                   narration_text, location_context, video_id=None):
    interaction_entry = {
        "start_time": start_time,
        "end_time": end_time,
        "action": action,
        "narration_text": narration_text,
        "location_context": location_context,
        "video_id": video_id  # ← NEW
    }
```

### 2. kg_update_executor.py:65-73, 106-114
**Updated both calls to `add_interaction()`:**
- Added `video_id=narration_info.get('video_id')` parameter to CREATE_NEW path
- Added `video_id=narration_info.get('video_id')` parameter to UPDATE path

**Before:**
```python
add_interaction(
    kg, food_id,
    history_entry['start_time'],
    history_entry['end_time'],
    history_entry['action'],
    history_entry['narration_text'],
    history_entry.get('location_context', '')
)
```

**After:**
```python
add_interaction(
    kg, food_id,
    history_entry['start_time'],
    history_entry['end_time'],
    history_entry['action'],
    history_entry['narration_text'],
    history_entry.get('location_context', ''),
    video_id=narration_info.get('video_id')  # ← NEW
)
```

### 3. FOOD_NODE_DECISION_RULES.md
**Updated documentation:**
- Added note about multi-video KG tracking
- Updated examples to show video_id in interaction history
- Added cross-video tracking examples
- Updated RULE 2 to mention video boundaries

## Data Structure Impact

### Old Interaction Entry:
```json
{
  "start_time": 9.34,
  "end_time": 11.16,
  "action": "pick up",
  "narration_text": "pick up a mug",
  "location_context": "zone_cupboard_1"
}
```

### New Interaction Entry:
```json
{
  "start_time": 9.34,
  "end_time": 11.16,
  "action": "pick up",
  "narration_text": "pick up a mug",
  "location_context": "zone_cupboard_1",
  "video_id": "P01-20240202-110250"
}
```

## Use Cases Enabled

### 1. Cross-Video Food Tracking
Track foods that persist across multiple video sessions:
```
video1 (morning): Place milk in fridge
video2 (afternoon): Take milk from fridge
→ Same food node, different video_id in interaction history
```

### 2. Visualizer Integration
Clicking on a snapshot can now:
- Load the correct video file (not just one video)
- Jump to the correct timestamp within that video
- Show which video each interaction came from

### 3. Food Provenance
Query food history to understand:
- When food first appeared (video + timestamp)
- Which videos the food appears in
- Full timeline across multiple recording sessions

## Backward Compatibility

The change is **backward compatible**:
- `video_id` parameter has a default value of `None`
- Old code that doesn't pass `video_id` will still work
- Existing KG JSON files without `video_id` in interactions are still valid
- New interactions will have `video_id`, old ones will show `null`

## Testing

Verified implementation:
```bash
python3 -c "from kg_storage import add_interaction; import inspect; print(inspect.signature(add_interaction))"
# Output: (kg, food_id, start_time, end_time, action, narration_text, location_context, video_id=None)
```

## Next Steps

To fully utilize this feature:
1. Ensure all narration CSV files include a `video_id` column
2. Update visualizer frontend to use `video_id` from interactions
3. Add video_id filtering to KG query functions
4. Consider adding video metadata to KG (video duration, participant, session info)

## Related Files
- `kg_storage.py` - Core data structures
- `kg_update_executor.py` - KG update logic
- `kg_sequential_pipeline.py` - Pipeline that extracts video_id from CSV
- `kg_visualizer_server.py` - Already has video_id support for snapshots
- `FOOD_NODE_DECISION_RULES.md` - Documentation of decision rules

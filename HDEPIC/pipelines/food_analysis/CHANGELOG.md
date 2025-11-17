# Food Analysis Pipeline - Changelog

## 2024-11-16: Refactored Step 2 to Group Food Items per Narration

### Previous Behavior
- **One entry per food item occurrence**
- Narrations with multiple food items appeared multiple times
- Example: Narration with "milk" and "milk frother" created 2 separate entries

**Old Output Structure:**
```json
[
  {
    "narration_id": "P01-20240202-110250-18",
    "class_id": 64,
    "noun_key": "milk",
    "noun_text": "milk",
    ...
  },
  {
    "narration_id": "P01-20240202-110250-18",  // DUPLICATE
    "class_id": 64,
    "noun_key": "milk",
    "noun_text": "milk frother",
    ...
  }
]
```

### New Behavior
- **One entry per narration**
- All food items grouped together in a `food_items` list
- Added `food_count` field for quick reference

**New Output Structure:**
```json
[
  {
    "narration_id": "P01-20240202-110250-18",
    "narration": "...",
    "start_timestamp": 27.0,
    "end_timestamp": 27.5,
    "narration_timestamp": 27.4,
    "hands": ["left hand", "right hand"],
    "food_items": [
      {
        "class_id": 64,
        "noun_key": "milk",
        "noun_text": "milk"
      },
      {
        "class_id": 64,
        "noun_key": "milk",
        "noun_text": "milk frother"
      }
    ],
    "food_count": 2
  }
]
```

### Key Changes

1. **No duplicate narrations** - Each narration_id appears exactly once
2. **Food items grouped** - All food items in a narration are in a single list
3. **Added `food_count`** - Quick access to number of food items per narration
4. **Clearer semantics** - Goal is to find narrations involving food, not individual food occurrences

### Statistics Comparison

**Previous Output:**
- 3,570 entries (many duplicates)
- Could not distinguish narrations from food items

**New Output:**
- 2,558 unique narrations involving food
  - 1,886 with single food item (74%)
  - 672 with multiple food items (26%)
- 3,570 total food item mentions (same as before)
- 212 narrations have 3+ food items
- No duplicates!

### Benefits

1. **Clearer intent**: Finding narrations involving food, not counting individual food mentions
2. **Easier analysis**: One narration = one timeline entry
3. **Preserves information**: All food items still accessible in `food_items` list
4. **Better for downstream tasks**: Video segmentation, temporal analysis
5. **Reduces redundancy**: 3,570 → 2,558 entries (28% reduction)

### Backward Compatibility

Downstream scripts (Step 3 and 4) need minor updates to work with new structure:
- Access `food_items` list instead of top-level fields
- Use `food_count` for quick statistics
- Iterate through `food_items` for detailed analysis

### Migration Example

**Old code:**
```python
for food in narrations:
    print(food['noun_key'], food['noun_text'])
```

**New code:**
```python
for narration in narrations:
    for food_item in narration['food_items']:
        print(food_item['noun_key'], food_item['noun_text'])
```

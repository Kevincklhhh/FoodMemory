# Per-Video Food Item Extractions

This directory contains food narration data extracted for individual videos from the HD-EPIC P01 dataset.

## Files

### P01-20240203-121517

**Video**: `P01-20240203-121517.mp4`

**Files**:
- `P01-20240203-121517_food_items.json` (27KB) - Complete structured data
- `P01-20240203-121517_food_summary.txt` (12KB) - Human-readable summary

**Content Overview**:
- **51 narrations** involving food items
- **76 total food mentions** (average 1.5 items per narration)
- **14 unique food types**
- **Time span**: 22.35s - 459.39s (~7.3 minutes of activity)

**Narration Distribution**:
- 35 narrations with 1 food item (69%)
- 10 narrations with 2 food items (20%)
- 3 narrations with 3 food items (6%)
- 3 narrations with 4 food items (6%)

**Top Food Items**:
1. **water** - 22 mentions (mainly for yeast/dough preparation)
2. **flour** - 21 mentions (measuring and mixing for dough)
3. **yeast** - 8 mentions (dissolving in water)
4. **sugar** - 3 mentions (for yeast activation)
5. **salt** - 3 mentions (for dough)

**Activity Summary**: 
This video captures bread-making preparation activities including:
- Grocery unpacking (meat, butter, lemon, onions)
- Yeast activation (mixing water, sugar, yeast)
- Flour measurement and mixing
- Dough preparation

## Data Structure

### JSON Format
```json
{
  "video_id": [
    {
      "narration_id": "unique_id",
      "narration": "description of action",
      "start_timestamp": 0.0,
      "end_timestamp": 0.0,
      "narration_timestamp": 0.0,
      "hands": ["left hand", "right hand"],
      "food_items": [
        {
          "class_id": 64,
          "noun_key": "milk",
          "noun_text": "milk frother"
        }
      ],
      "food_count": 1
    }
  ]
}
```

### Field Descriptions

- **narration_id**: Unique identifier (format: `{video_id}-{number}`)
- **narration**: Full text description of the action
- **start_timestamp**: Action start time in seconds
- **end_timestamp**: Action end time in seconds
- **narration_timestamp**: Key moment timestamp (usually midpoint)
- **hands**: List of hands involved in action
- **food_items**: Array of all food items mentioned in this narration
  - **class_id**: HD-EPIC noun class ID
  - **noun_key**: Canonical food category (e.g., "milk")
  - **noun_text**: Specific noun phrase from narration (e.g., "milk frother")
- **food_count**: Number of food items in this narration (for quick filtering)

## Usage

### Load in Python
```python
import json

with open('P01-20240203-121517_food_items.json', 'r') as f:
    data = json.load(f)

# Access narrations for the video
narrations = data['P01-20240203-121517']

# Iterate through narrations
for narration in narrations:
    print(f"Time: {narration['start_timestamp']:.2f}s")
    print(f"Action: {narration['narration']}")
    print(f"Foods: {[item['noun_key'] for item in narration['food_items']]}")
```

### Filter by Food Type
```python
# Find all narrations involving water
water_narrations = [
    n for n in narrations 
    if any(item['noun_key'] == 'water' for item in n['food_items'])
]
print(f"Found {len(water_narrations)} narrations with water")
```

### Extract Time Segments
```python
# Get time segments for video extraction
segments = [
    (n['start_timestamp'], n['end_timestamp']) 
    for n in narrations
]
```

## Source

Extracted from:
- **Source file**: `/outputs/food_analysis/extractions/hdepic_p01_food_items.json`
- **Extraction script**: `/pipelines/food_analysis/2_extract_hdepic_food_items.py`
- **Date**: 2024-11-16 (refactored version)

## Notes

- Each narration appears exactly once (no duplicates)
- Multiple food items in a single narration are grouped together
- Timestamps are in seconds from video start
- Food classifications are based on LLM analysis (Step 1)

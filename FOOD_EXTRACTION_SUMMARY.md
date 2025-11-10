# VISOR Food Extraction Summary

## Overview
Successfully extracted food segments from EPIC-KITCHENS VISOR sparse annotations for video P02_03.

## Results

### Frame Correspondence
- **Total sparse frames**: 564 frames
- **Frames with VISOR annotations**: 564 frames (100% correspondence ✓)
- **Frames with food items**: 364 frames (64.5%)

### Food Items Detected
Extracted **22 unique food items** using the `epic_food_nouns_detailed.json` classification:

| Food Noun | Specific Instance | Frame Count |
|-----------|------------------|-------------|
| onion | onion | 97 |
| broccoli | broccoli | 67 |
| cheese | cheese | 49 |
| nut | pine nut | 37 |
| pasta | pasta | 35 |
| sauce | pesto | 33 |
| food | mixed sauce | 32 |
| water | water | 31 |
| pasta | pasta mixed sauce | 28 |
| water | pasta water | 22 |
| oil | oil | 14 |
| sauce | sauce | 12 |
| food | food | 11 |
| salt | salt | 10 |
| chicken | chicken piece | 8 |
| garlic | garlic | 6 |
| broccoli | broccoli rest | 5 |
| onion | onion end | 5 |
| onion | onion slice rest | 4 |
| chicken | chicken | 3 |
| yoghurt | yoghurt | 1 |
| water | more water | 1 |

## Implementation

### Script: `extract_food_segments.py`

**Key Features:**
1. **Dynamic Food Loading**: Loads food class IDs from `epic_food_nouns_detailed.json` (150 food classes)
2. **Frame-Annotation Matching**: Verifies correspondence between sparse frames and VISOR annotations
3. **Segmentation Extraction**: Creates polygon masks from VISOR segmentation annotations
4. **Visualization**: Generates overlayed images with green highlighting for food regions

**Output Structure:**
```
epic-kitchen-visor/extracted_food_segments/P02_03/
├── P02_03_food_metadata.json          # Complete metadata
├── P02_03_frame_*_food_mask.jpg       # 364 masked images
```

### Food Classification
- Uses EPIC-100 noun class IDs from `epic_food_nouns_detailed.json`
- Each food item has:
  - `class_id`: EPIC-100 noun class identifier
  - `noun_name`: Generic food category (e.g., "onion", "broccoli")
  - `category`: High-level grouping (vegetables, dairy, etc.)
  - Specific instances in annotations (e.g., "onion end", "pine nut")

## Verification

### Correspondence Check ✓
- All 564 sparse frames from `P02_03.zip` have matching entries in `P02_03.json`
- Frame naming format: `P02_03_frame_XXXXXXXXXX.jpg` (10-digit frame number)
- Annotations use same naming convention

### Segmentation Quality
- Polygon-based masks from VISOR annotations
- Multiple food items can appear in single frame
- Green overlay (30% transparency) highlights food regions

## Usage

```bash
python3 extract_food_segments.py
```

**Requirements:**
- OpenCV (cv2)
- NumPy
- Python 3.x

## Next Steps

Potential extensions:
1. Process additional videos (P01-P37 participants)
2. Create food-specific training datasets
3. Analyze food co-occurrence patterns
4. Extract cropped food regions for classification tasks
5. Compare with full video annotations (if available)

## Files Generated

- **Script**: `extract_food_segments.py` - Main extraction script
- **Output**: `epic-kitchen-visor/extracted_food_segments/P02_03/`
  - 364 masked images
  - 1 metadata JSON file (7.8 MB)
  - Total size: ~168 MB

## Validation

✓ Sparse frames exist and are accessible
✓ VISOR annotations loaded successfully
✓ Frame-to-annotation 1:1 correspondence
✓ Food class IDs loaded from JSON (150 classes)
✓ Segmentation masks created successfully
✓ Visualization images generated
✓ Metadata saved in structured format

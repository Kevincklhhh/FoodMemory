# VISOR Food Extraction - Refactoring Summary

## Changes Made

### 1. Script Reorganization

**Before:**
- `extract_food_from_visor.py` - Complex script with multiple outputs
- `extract_food_segments.py` - Duplicate functionality with inline food classification

**After:**
- `1_extract_food_items.py` - **Step 1**: Clean extraction script focused on parsing VISOR annotations
- `2_create_food_segments.py` - **Step 2**: Visualization script that uses Step 1 output

### 2. Key Improvements

#### Separation of Concerns
- **Step 1** handles all annotation parsing and food identification
- **Step 2** handles visualization and image processing
- Clear pipeline: JSON → Segments

#### Eliminated Duplication
- Removed hardcoded food lists
- Single source of truth: `epic_food_nouns_detailed.json`
- No redundant analysis in segmentation script

#### Simplified Output
- **Step 1**: Single JSON file (`visor_food_items.json`)
- **Step 2**: Organized directory of images (`food_segments/`)
- No more multiple CSV, TXT, JSON variants

### 3. Files Removed

Deleted redundant output files:
```
✗ visor_food_mapping.json          (replaced by visor_food_items.json)
✗ visor_food_mapping_by_video.txt  (redundant)
✗ visor_food_mapping_detailed.csv  (redundant)
✗ visor_food_mapping_statistics.txt (redundant)
✗ visor_coverage_*.{json,csv,txt}  (redundant)
✗ wdtcf_food_*.{json,csv,txt}      (redundant)
✗ frame_mapping.json               (redundant)
```

Deleted old scripts:
```
✗ extract_food_from_visor.py       (replaced by 1_extract_food_items.py)
✗ extract_food_segments.py         (replaced by 2_create_food_segments.py)
✗ extract_wdtcf_food.py            (obsolete)
```

### 4. Current Directory Structure

```
epic-kitchen-visor/
├── 1_extract_food_items.py          # Step 1: Extract food items
├── 2_create_food_segments.py        # Step 2: Create visualizations
├── WORKFLOW.md                       # Complete usage guide
├── REFACTORING_SUMMARY.md           # This file
│
├── visor_food_items.json            # Main output (404 MB)
│
├── EPIC_100_noun_classes_v2.csv     # EPIC-100 taxonomy
├── epic_food_nouns_detailed.json    # Food classifications
├── epic_food_nouns_detailed.csv     # Same in CSV
├── epic_food_nouns_names.txt        # Simple list
│
├── classify_epic_food_nouns.py      # Supporting script
├── classify_food_objects.py         # Supporting script
│
├── GroundTruth-SparseAnnotations/   # VISOR data
├── Interpolations-DenseAnnotations/ # VISOR data
│
├── WDTCF_GT.json                    # WDTCF ground truth
├── README.txt                       # VISOR README
└── (PDFs)                           # Documentation
```

## Execution Results

### Step 1: Extract Food Items
```bash
$ python3 1_extract_food_items.py

Results:
✓ 158 videos analyzed (115 train, 43 val)
✓ 154 videos with food items
✓ 32,645 food occurrences detected
✓ 127 unique food classes found
✓ Output: visor_food_items.json (404 MB)
```

### Top Food Classes
1. water - 4,080 occurrences
2. onion - 1,798 occurrences
3. meat - 1,430 occurrences
4. food - 1,178 occurrences
5. cheese - 1,098 occurrences
6. dough - 1,004 occurrences
7. carrot - 998 occurrences
8. aubergine - 827 occurrences
9. garlic - 802 occurrences
10. potato - 792 occurrences

### Example: P02_03 Video
- **Total frames**: 564 (sparse annotations)
- **Food occurrences**: 511
- **Frame correspondence**: 100% ✓
- **Segmentation data**: Polygon coordinates included

Sample occurrence:
```json
{
  "frame_name": "P02_03_frame_0000000797.jpg",
  "frame_number": 797,
  "class_id": 27,
  "noun_key": "water",
  "object_name": "water",
  "object_id": "1925bede3b00f33bc785361826eac8c2",
  "segments": [[[x1, y1], [x2, y2], ...]],
  "exhaustive": "inconclusive"
}
```

## Benefits

### 1. Clarity
- Clear two-step pipeline
- Each script has single responsibility
- Self-documenting workflow

### 2. Efficiency
- Run Step 1 once, reuse results
- Step 2 can process videos selectively
- No redundant computation

### 3. Flexibility
- Easy to process specific videos
- Can limit processing for testing
- Simple to add new visualization types

### 4. Maintainability
- Less code duplication
- Single source of truth for food classes
- Easier to debug and extend

## Usage Examples

### Extract all food items (run once)
```bash
python3 1_extract_food_items.py
```

### Create segments for all videos
```bash
python3 2_create_food_segments.py
```

### Create segments for specific videos
```bash
python3 2_create_food_segments.py --videos P02_03 P01_01 P04_05
```

### Test with limited videos
```bash
python3 2_create_food_segments.py --limit 10
```

## Data Integrity

### Validated
✓ Frame-to-annotation correspondence (100%)
✓ Food class mappings (150 classes)
✓ Polygon segmentation data
✓ Video metadata (participant IDs, frame counts)

### Preserved
✓ Original VISOR annotations untouched
✓ EPIC-100 taxonomy maintained
✓ Food classifications consistent
✓ Segmentation polygons intact

## Next Steps

### Immediate Use Cases
1. Run Step 2 to create visualizations for specific videos
2. Analyze food co-occurrence patterns from JSON
3. Extract temporal food sequences
4. Create training datasets for food detection

### Potential Extensions
1. Add crop extraction (individual food items)
2. Generate food statistics by participant
3. Create temporal food graphs
4. Export to other formats (COCO, YOLO, etc.)

## Performance

- **Step 1**: ~2-3 minutes (all 158 videos)
- **Step 2**: ~1-2 seconds per video
- **Storage**: 404 MB for complete food annotations

## Conclusion

The refactored pipeline provides:
- ✓ Clean separation of extraction and visualization
- ✓ Single comprehensive output file
- ✓ Flexible processing options
- ✓ Reduced redundancy and complexity
- ✓ Better maintainability and extensibility

All food items from VISOR are now available in a single, well-structured JSON file ready for further processing.

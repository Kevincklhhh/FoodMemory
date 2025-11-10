# VISOR Food Extraction Workflow

This directory contains a two-step pipeline for extracting and visualizing food items from EPIC-KITCHENS VISOR annotations.

## Overview

The workflow is split into two independent steps:

1. **Extract Food Items** - Parse VISOR annotations to identify all food occurrences
2. **Create Segmentation Visualizations** - Generate visual masks for the extracted food items

## Prerequisites

### Data Files (Included)
- `EPIC_100_noun_classes_v2.csv` - EPIC-100 noun taxonomy (305 classes)
- `epic_food_nouns_detailed.json` - Food classifications (150 food classes with metadata)
- `GroundTruth-SparseAnnotations/` - VISOR sparse annotations with RGB frames

### Python Dependencies
```bash
pip install opencv-python numpy
```

## Step 1: Extract Food Items

**Script:** `1_extract_food_items.py`

Processes all VISOR annotation files to identify food items using proper class_id mapping.

### Usage
```bash
python3 1_extract_food_items.py
```

### Options
```bash
--visor-dir         # Path to VISOR annotations (default: GroundTruth-SparseAnnotations)
--noun-classes      # Path to EPIC-100 classes (default: EPIC_100_noun_classes_v2.csv)
--food-json         # Path to food classifications (default: epic_food_nouns_detailed.json)
--output            # Output JSON file (default: visor_food_items.json)
--splits            # Splits to process (default: train val)
```

### Output
**File:** `visor_food_items.json` (~404 MB)

Structure:
```json
{
  "P02_03": {
    "video_id": "P02_03",
    "participant_id": "P02",
    "total_frames_annotated": 564,
    "food_occurrences": [
      {
        "frame_name": "P02_03_frame_0000000212.jpg",
        "frame_number": 212,
        "class_id": 16,
        "noun_key": "onion",
        "object_name": "onion bag",
        "object_id": "b8fb39f20ba1f587c97af5c567661ee6",
        "segments": [[[x1,y1], [x2,y2], ...]],
        "exhaustive": "y"
      }
    ]
  }
}
```

### Results
- **158 videos** analyzed (115 train, 43 val)
- **154 videos** with food items
- **32,645 food occurrences** detected
- **127 unique food classes** found

Top food classes:
1. water - 4,080 occurrences
2. onion - 1,798 occurrences
3. meat - 1,430 occurrences
4. food - 1,178 occurrences
5. cheese - 1,098 occurrences

## Step 2: Create Food Segmentations

**Script:** `2_create_food_segments.py`

Creates visual segmentation masks from the extracted food items.

## Step 3: Analyze Food Per Video

**Script:** `3_analyze_food_per_video.py`

Analyzes the extracted food items and generates per-video food lists in multiple formats.

### Usage (Step 3)
```bash
# Analyze all videos and generate reports
python3 3_analyze_food_per_video.py
```

### Options
```bash
--input             # Input JSON from Step 1 (default: visor_food_items.json)
--output-prefix     # Output file prefix (default: food_per_video)
```

### Output
**Files:** Multiple formats for analysis

1. **food_per_video.json** - Detailed JSON with frame ranges and statistics
2. **food_per_video.csv** - Tabular format (one row per food item per video)
3. **food_per_video.txt** - Human-readable detailed summary
4. **food_per_video_simple.txt** - Simple list of food items per video

#### JSON Structure:
```json
{
  "P02_03": {
    "video_id": "P02_03",
    "participant_id": "P02",
    "total_frames_annotated": 564,
    "total_food_occurrences": 511,
    "unique_food_items": 13,
    "food_items": [
      {
        "class_id": 27,
        "noun_key": "water",
        "object_names": ["water", "pasta water"],
        "total_occurrences": 54,
        "first_frame": 797,
        "last_frame": 68260,
        "frame_count": 54,
        "frame_ranges": [[797, 797], [834, 834], ...],
        "frame_numbers": [797, 834, 1292, ...]
      }
    ]
  }
}
```

### Results
- **158 videos** analyzed
- **154 videos** with food items
- **1,049 unique food items** across all videos (counting per-video instances)
- **127 global food classes**
- Average: **6.6 food items per video**

Top videos by food diversity:
1. P04_05 - 22 unique food items
2. P08_21 - 22 unique food items
3. P30_05 - 21 unique food items

## Step 2: Create Food Segmentations

**Script:** `2_create_food_segments.py`

Creates visual segmentation masks from the extracted food items.

### Usage (Step 2)
```bash
# Process all videos
python3 2_create_food_segments.py

# Process specific videos
python3 2_create_food_segments.py --videos P02_03 P01_01

# Process first N videos
python3 2_create_food_segments.py --limit 10
```

### Options
```bash
--input             # Input JSON from Step 1 (default: visor_food_items.json)
--frames-dir        # Base directory for frames (default: GroundTruth-SparseAnnotations/rgb_frames)
--output-dir        # Output directory (default: food_segments)
--videos            # Specific video IDs to process (optional)
--limit             # Limit number of videos (optional)
```

### Output
**Directory:** `food_segments/`

Structure:
```
food_segments/
├── P02_03/
│   ├── P02_03_frame_0000000212_food_mask.jpg
│   ├── P02_03_frame_0000000834_food_mask.jpg
│   └── ...
├── P01_01/
│   └── ...
└── ...
```

Each image shows the original frame with green overlay (30% transparency) highlighting food regions.

## Data Files

### Core Data (Keep)
- `EPIC_100_noun_classes_v2.csv` - EPIC-100 taxonomy
- `epic_food_nouns_detailed.json` - Food class definitions with categories
- `epic_food_nouns_detailed.csv` - Same data in CSV format
- `epic_food_nouns_names.txt` - Simple list of food nouns
- `visor_food_items.json` - **Output from Step 1** (VISOR food extraction, 404 MB)
- `food_image_index.json` - **Output from Step 4** (Complete image index, 1.1 GB)
- `food_inventory_lookup.json` - **Output from Step 4** (Optimized lookup, 16 MB)
- `wdtcf_food_items.json` - **Output from Step 5** (WDTCF food extraction, 87 KB)
- `WDTCF_GT.json` - WDTCF ground truth annotations (input)

### Supporting Scripts (Keep)
- `classify_epic_food_nouns.py` - Script used to classify food nouns
- `classify_food_objects.py` - Object classification script

### Documentation (Keep)
- `README.txt` - VISOR dataset README
- `WORKFLOW.md` - This file

## Technical Details

### Food Classification
Uses EPIC-100 noun taxonomy with 150 food classes across 7 categories:
- Vegetables
- Fruits and nuts
- Meat and substitute
- Dairy and eggs
- Baked goods and grains
- Spices and herbs and sauces
- Drinks
- Prepared food
- Other

### Segmentation Format
VISOR annotations use polygon segmentation:
- Format: List of (x, y) coordinate pairs
- Multiple polygons per object possible
- Coordinates in image pixel space

### Frame Naming Convention
```
{PARTICIPANT_ID}_{VIDEO_NUM}_frame_{FRAME_NUMBER:010d}.jpg
```
Example: `P02_03_frame_0000000212.jpg`

## Step 4: Build Food Image Index

**Script:** `4_build_food_image_index.py`

Unzips all RGB frames and builds a searchable index of food images with annotations.

### Usage (Step 4)
```bash
# Build complete index (first time - unzips all frames)
python3 4_build_food_image_index.py

# Rebuild index without unzipping (if frames already extracted)
python3 4_build_food_image_index.py --skip-unzip
```

### Options
```bash
--frames-dir        # Base directory for RGB frames (default: GroundTruth-SparseAnnotations/rgb_frames)
--food-items        # Food items JSON from Step 1 (default: visor_food_items.json)
--output            # Output index file (default: food_image_index.json)
--lookup-output     # Lookup file for inventory (default: food_inventory_lookup.json)
--skip-unzip        # Skip unzipping if already done
--splits            # Splits to process (default: train val)
```

### Output
**Files:**
1. **food_image_index.json** (1.1 GB) - Complete index with all annotations
2. **food_inventory_lookup.json** (16 MB) - Optimized for food inventory queries

#### Index Structure:
```json
{
  "by_food_class": {
    "onion": [
      {
        "video_id": "P01_01",
        "frame_number": 22843,
        "image_path": "train/P01/P01_01_frame_0000022843.jpg",
        "segments": [[[x1,y1], [x2,y2], ...]]
      }
    ]
  },
  "by_video": { "video_id": [...] },
  "by_frame": { "video_id": { "frame_num": {...} } },
  "metadata": { ... }
}
```

### Results
- **157 ZIP archives** unzipped (115 train, 43 val)
- **19,591 food images** indexed
- **122 food classes** with images
- **119 videos** with indexed food images

## Step 5: Extract WDTCF Food Items

**Script:** `5_extract_wdtcf_food_items.py`

Extracts food items from the WDTCF (Where Did The Container First appear) dataset, which tracks where objects were first stored in the kitchen.

### Usage (Step 5)
```bash
# Extract food items from WDTCF
python3 5_extract_wdtcf_food_items.py
```

### Options
```bash
--wdtcf             # WDTCF ground truth file (default: WDTCF_GT.json)
--food-nouns        # Food nouns JSON (default: epic_food_nouns_detailed.json)
--output            # Output JSON file (default: wdtcf_food_items.json)
--simple-output     # Simple text output (default: wdtcf_food_per_video_simple.txt)
```

### Output
**Files:**
1. **wdtcf_food_items.json** (87 KB) - Complete food instances with queryable instance IDs
2. **wdtcf_food_per_video_simple.txt** (1.7 KB) - Simple list of food per video

#### JSON Structure:
```json
{
  "metadata": {
    "total_videos": 74,
    "total_food_instances": 135,
    "unique_food_classes": 62
  },
  "videos": {
    "P02_03": {
      "video_id": "P02_03",
      "total_food_instances": 5,
      "unique_food_classes": 5,
      "food_classes": ["broccoli", "cheese", "pasta", "salt", "sauce"],
      "food_instances": [
        {
          "instance_id": "P02_03_salt",
          "object_name": "salt",
          "food_class": "salt",
          "query_frame": {
            "video_id": "P02_03",
            "frame_number": 47206,
            "frame_name": "P02_03_frame_0000047206.jpg"
          },
          "evidence_frame": {
            "video_id": "P02_03",
            "frame_number": 31449,
            "frame_name": "P02_03_frame_0000031449.jpg"
          },
          "storage_locations": ["cupboard"]
        }
      ]
    }
  }
}
```

### Results
- **74 videos** with food items
- **135 food instances** extracted
- **62 unique food classes**
- **16 non-food items** filtered out (containers, utensils, etc.)

Top food classes:
1. cheese - 8 instances
2. oil - 8 instances
3. onion - 7 instances
4. milk - 7 instances
5. butter - 5 instances

**Key Features:**
- Each instance has a unique `instance_id` (original WDTCF key) for queryability
- Includes both query frame (where object was used) and evidence frame (where stored)
- Storage locations tracked (fridge, cupboard, drawer, etc.)

### Query Food Inventory

**Script:** `query_food_inventory.py`

Search the index for specific food items (for synthetic household inventory).

**Default behavior:** Returns only the first appearance of each food item per video to provide diverse samples across different cooking sessions. Use `--all-occurrences` to get all frames containing the food.

```bash
# List all available foods
python3 query_food_inventory.py --list

# Query specific food items (returns first appearance per video)
python3 query_food_inventory.py --food onion cheese tomato

# Get sample images (first N unique videos)
python3 query_food_inventory.py --food onion --limit 5

# Get all occurrences (multiple frames from same video)
python3 query_food_inventory.py --food onion --limit 10 --all-occurrences

# Export to CSV
python3 query_food_inventory.py --food onion cheese --export inventory.csv

# Copy images organized by food type
python3 query_food_inventory.py --food onion cheese --copy-images food_samples/
```

## Example Workflow

```bash
# Step 1: Extract all food items from VISOR (run once)
python3 1_extract_food_items.py
# Output: visor_food_items.json (404 MB)

# Step 3: Analyze food items per video (generates reports)
python3 3_analyze_food_per_video.py
# Output: food_per_video.{json,csv,txt} + simple.txt

# Step 4: Build food image index (run once - unzips all frames)
python3 4_build_food_image_index.py
# Output: food_image_index.json (1.1 GB), food_inventory_lookup.json (16 MB)

# Step 5: Extract WDTCF food items (temporal storage tracking)
python3 5_extract_wdtcf_food_items.py
# Output: wdtcf_food_items.json (87 KB), wdtcf_food_per_video_simple.txt (1.7 KB)

# Query for specific foods
python3 query_food_inventory.py --food onion tomato cheese --limit 10

# Step 2a: Create segments for specific video
python3 2_create_food_segments.py --videos P02_03

# Step 2b: Create segments for first 5 videos
python3 2_create_food_segments.py --limit 5

# Step 2c: Process all videos (requires ~168MB per video)
python3 2_create_food_segments.py
```

## Validation

### Step 1 Validation
- ✓ All 158 videos processed
- ✓ 154 videos contain food items
- ✓ 32,645 food occurrences extracted
- ✓ Frame-annotation correspondence verified

### Example: P02_03
- 564 frames annotated
- 511 food occurrences
- 22 unique food items including onion, broccoli, cheese, pasta, sauce, water

## Performance Notes

- **Step 1**: ~2-3 minutes for all 158 videos
- **Step 2**: ~1-2 seconds per video with food
- **Storage**:
  - visor_food_items.json: 404 MB
  - Segmented images: ~168 MB per video (varies)

## Citation

If using this data, please cite:
```
@inproceedings{VISOR2022,
  title={EPIC-KITCHENS VISOR Benchmark: Video Segmentations and Object Relations},
  author={Darkhalil, Ahmad and Shan, Dandan and Zhu, Bin and Ma, Jian and Kar, Amlan and Higgins, Richard and Fidler, Sanja and Fouhey, David and Damen, Dima},
  booktitle={Proceedings of the Neural Information Processing Systems (NeurIPS) Track on Datasets and Benchmarks},
  year={2022}
}
```

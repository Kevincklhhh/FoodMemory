# Query Food Inventory Tool - Usage Guide

## Overview
`query_food_inventory.py` is a tool to search and retrieve food images from the VISOR dataset based on the food inventory index.

---

## Basic Usage

### Default Behavior (First Frame Per Video)
By default, the tool returns only the first frame containing the queried food from each video:

```bash
python3 query_food_inventory.py --food yoghurt
```

**Output:**
- Returns 8 images (one per video that contains yoghurt)
- Useful for getting a representative sample across all videos

---

## Retrieve ALL Frames from ALL Videos

### Get Every Single Frame
Use `--all-occurrences` to retrieve ALL frames containing the queried food from ALL videos:

```bash
python3 query_food_inventory.py --food yoghurt --all-occurrences
```

**Output:**
- Returns 239 images (every single frame with yoghurt across all 8 videos)
- Includes multiple frames per video
- Comprehensive dataset for analysis

---

## Common Use Cases

### 1. Quick Sample (Default)
Get one representative frame per video:
```bash
python3 query_food_inventory.py --food onion
```

### 2. Complete Dataset
Get every frame with the food:
```bash
python3 query_food_inventory.py --food onion --all-occurrences
```

### 3. Export All Frames to CSV
Export all occurrences to CSV for analysis:
```bash
python3 query_food_inventory.py --food yoghurt --all-occurrences --export yoghurt_complete.csv
```

### 4. Limited Sample from All Videos
Get first N frames across all videos:
```bash
python3 query_food_inventory.py --food water --all-occurrences --limit 100
```

### 5. Copy All Images
Copy all frames with a food to a directory:
```bash
python3 query_food_inventory.py --food cheese --all-occurrences --copy-images ./cheese_images
```

### 6. Multiple Food Items
Query multiple foods at once:
```bash
python3 query_food_inventory.py --food onion tomato garlic --all-occurrences
```

---

## Command-Line Options

### Query Options
- `--food FOOD [FOOD ...]` - Food items to query (required)
- `--all-occurrences` - Return ALL frames from ALL videos (default: first per video)
- `--limit N` - Limit to N images per food item
- `--index FILE` - Specify index file (default: food_inventory_lookup.json)

### Output Options
- `--export FILE.csv` - Export results to CSV
- `--copy-images DIR` - Copy images to directory organized by food
- `--frames-base DIR` - Base path for frames (default: GroundTruth-SparseAnnotations/rgb_frames)

### Discovery Options
- `--list` - List all available food items
- `--search PATTERN` - Search for food items matching pattern

---

## Examples with Output

### Example 1: Default Query
```bash
$ python3 query_food_inventory.py --food yoghurt

Querying for: yoghurt (first per video)

YOGHURT
  Total occurrences in dataset: 239
  Total videos with this food: 8
  Unique videos returned: 8
  Images returned: 8
```

### Example 2: All Occurrences
```bash
$ python3 query_food_inventory.py --food yoghurt --all-occurrences

Querying for: yoghurt (all occurrences from all videos)

YOGHURT
  Total occurrences in dataset: 239
  Total videos with this food: 8
  Unique videos returned: 8
  Images returned: 239
```

### Example 3: CSV Export Structure
```bash
$ python3 query_food_inventory.py --food yoghurt --all-occurrences --export output.csv
```

**CSV Format:**
```csv
food_item,image_path,full_path,video_id,frame_number,object_id
yoghurt,train/P02/P02_03_frame_0000004418.jpg,GroundTruth-SparseAnnotations/rgb_frames/train/P02/P02_03_frame_0000004418.jpg,P02_03,4418,5adcac35f5663c4d1bf52c9d33dccdc1
yoghurt,train/P02/P02_102_frame_0000000271.jpg,GroundTruth-SparseAnnotations/rgb_frames/train/P02/P02_102_frame_0000000271.jpg,P02_102,271,2ec7d48f66825ff3b7698b8e436cfea8
...
```

---

## Understanding the Results

### Result Fields
- **Total occurrences in dataset**: Total number of frames with this food across all videos
- **Total videos with this food**: Number of unique videos containing this food
- **Unique videos returned**: Number of videos in the result set
- **Images returned**: Number of frames in the result set

### Filtering Logic
1. **Default (first per video)**:
   - Scans through all frames
   - Returns first occurrence in each video
   - Result: `unique_videos_returned` = `total_videos`

2. **All occurrences**:
   - Returns every single frame
   - Result: `images_returned` = `total_occurrences` (unless limited)

3. **With limit**:
   - Takes first N frames from the result
   - May include multiple frames from same video

---

## Performance Considerations

### Dataset Size
Different foods have different occurrence counts:
- **water**: 4,080 frames across 112 videos
- **onion**: 1,798 frames across 34 videos
- **yoghurt**: 239 frames across 8 videos
- **almond**: 11 frames across 1 video

### Recommendations
- Use default mode for quick sampling
- Use `--all-occurrences` when you need complete data
- Use `--limit` with high-occurrence foods (like water)
- Export to CSV for large datasets instead of copying images

---

## Discovery Commands

### List All Foods
```bash
python3 query_food_inventory.py --list
```

### Search for Foods
```bash
python3 query_food_inventory.py --search yoghurt
```

---

## Use Cases by Research Need

### 1. Visual Analysis
Get representative samples:
```bash
python3 query_food_inventory.py --food onion --copy-images ./samples
```

### 2. Dataset Statistics
Export all occurrences for counting:
```bash
python3 query_food_inventory.py --food water --all-occurrences --export water_all.csv
```

### 3. Training Data Collection
Collect all instances for ML training:
```bash
python3 query_food_inventory.py --food cheese --all-occurrences --copy-images ./training_data
```

### 4. Cross-Video Analysis
Get one sample per video for comparison:
```bash
python3 query_food_inventory.py --food tomato --export tomato_per_video.csv
```

---

## Tips

1. **Start with default mode** to see how many videos contain the food
2. **Use --all-occurrences** when you need comprehensive coverage
3. **Apply --limit** to manage large result sets
4. **Export to CSV** first before copying images (lighter weight)
5. **Check --list** to see available foods before querying

---

**Date**: 2025-11-11
**Updated**: Removed single-video filter, enhanced all-occurrences mode

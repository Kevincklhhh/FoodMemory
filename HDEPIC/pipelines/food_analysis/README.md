# HD-EPIC P01 Food Analysis Pipeline

This directory contains a complete food analysis pipeline for HD-EPIC participant P01, following the same methodology as the epic-kitchen-visor analysis.

## Pipeline Overview

The pipeline consists of 4 main steps:

### Step 1: Classify HD-EPIC Food Nouns with LLM
**Script:** `1_classify_hdepic_food_nouns.py`

- Loads all 303 HD-EPIC noun classes from `HD_EPIC_noun_classes.csv`
- Uses Qwen3-VL LLM to classify each noun as food or non-food
- Generates structured prompts with clear food/non-food criteria
- **Results:** Identified **152 food nouns** out of 303 total noun classes (50.2%)

**Outputs:**
- `hdepic_food_nouns_detailed.json` - Full details with LLM reasoning
- `hdepic_food_nouns_detailed.csv` - Tabular format
- `hdepic_food_nouns_names.txt` - Simple list of food names

### Step 2: Extract Food Items from P01 Narrations
**Script:** `2_extract_hdepic_food_items.py`

- Reads participant P01 narrations from `participant_P01_narrations.csv`
- Maps noun_classes to food items using classifications from Step 1
- Extracts all food occurrences with timestamps and narration context
- **Results:** Found **3,570 food occurrences** across **27 P01 videos**

**Outputs:**
- `hdepic_p01_food_items.json` - All food items mapped by video_id

### Step 3: Analyze Food Per Video
**Script:** `3_analyze_hdepic_food_per_video.py`

- Analyzes food occurrences for each video
- Groups by class_id to identify unique food items per video
- Calculates statistics: first/last appearance, occurrence counts, time ranges
- **Results:**
  - Total unique food items across all videos: **277**
  - Unique food classes: **64**
  - Average food items per video: **10.3**

**Top Videos with Most Food:**
1. P01-20240203-135502 - 23 unique food items
2. P01-20240203-123350 - 20 unique food items
3. P01-20240204-145458 - 20 unique food items

**Outputs:**
- `hdepic_food_per_video.json` - Detailed per-video analysis
- `hdepic_food_per_video.csv` - Tabular format
- `hdepic_food_per_video.txt` - Human-readable summary
- `hdepic_food_per_video_simple.txt` - Concise list

### Step 4: Analyze Food Abundance
**Script:** `4_analyze_hdepic_food_abundance.py`

- Analyzes distribution of food items across videos
- Calculates occurrence statistics, averages, and maximums
- Identifies which foods are most common/rare
- **Results:** 64 unique food classes with varying abundance

**Most Common Foods:**
1. water - 293 occurrences in 23 videos
2. dough - 278 occurrences in 7 videos
3. food - 276 occurrences in 19 videos
4. meat - 247 occurrences in 12 videos
5. carrot - 225 occurrences in 4 videos
6. onion - 207 occurrences in 5 videos
7. biscuit - 146 occurrences in 9 videos
8. garlic - 130 occurrences in 5 videos
9. orange - 125 occurrences in 8 videos
10. mixture - 125 occurrences in 12 videos

**Outputs:**
- `hdepic_food_abundance_analysis.json` - Detailed abundance statistics
- `hdepic_food_abundance_table.csv` - Sortable table

## Summary Statistics

### Overall Results
- **Total noun classes:** 303
- **Food nouns identified:** 152 (50.2%)
- **P01 videos analyzed:** 27
- **Total food occurrences:** 3,570
- **Unique food classes found:** 64
- **Average food items per video:** 10.3

### Food Distribution
- **Most abundant food:** water (293 occurrences, 23 videos)
- **Most diverse food:** roll (appears in 18 videos)
- **Video with most food variety:** P01-20240203-135502 (23 unique items)
- **Video with most food occurrences:** P01-20240203-135502 (714 occurrences)

## Comparison with Epic-Kitchen-VISOR

The HD-EPIC analysis follows the same methodology as epic-kitchen-visor:

| Metric | HD-EPIC P01 | VISOR |
|--------|-------------|-------|
| Dataset | HD-EPIC | EPIC-KITCHENS-VISOR |
| Participant | P01 only | Multiple participants |
| Videos | 27 | 300+ |
| Food classes | 64 | ~50-60 |
| Analysis method | Narration-based | Frame annotation-based |

## Usage

Run the pipeline in sequence:

```bash
# Step 1: Classify food nouns (takes ~5 minutes with LLM API)
python 1_classify_hdepic_food_nouns.py

# Step 2: Extract food items from narrations
python 2_extract_hdepic_food_items.py

# Step 3: Analyze food per video
python 3_analyze_hdepic_food_per_video.py

# Step 4: Analyze food abundance
python 4_analyze_hdepic_food_abundance.py
```

## Key Insights

1. **Food Diversity:** P01 interactions involve 64 different food classes, showing rich variety in kitchen activities
2. **Water Dominance:** Water appears in 85% of videos (23/27), indicating frequent washing/cooking activities
3. **Baking Activities:** High occurrence of dough (278) and biscuit (146) suggests significant baking
4. **Preparation-Heavy:** High counts of preparation items (carrot: 225, onion: 207, garlic: 130) indicate extensive food prep
5. **Temporal Coverage:** Food interactions span entire videos with proper timestamp annotations

## File Structure

```
HDEPIC/
├── 1_classify_hdepic_food_nouns.py
├── 2_extract_hdepic_food_items.py
├── 3_analyze_hdepic_food_per_video.py
├── 4_analyze_hdepic_food_abundance.py
├── hdepic_food_nouns_detailed.json
├── hdepic_food_nouns_detailed.csv
├── hdepic_food_nouns_names.txt
├── hdepic_p01_food_items.json
├── hdepic_food_per_video.json
├── hdepic_food_per_video.csv
├── hdepic_food_per_video.txt
├── hdepic_food_per_video_simple.txt
├── hdepic_food_abundance_analysis.json
├── hdepic_food_abundance_table.csv
└── HDEPIC_FOOD_ANALYSIS_README.md
```

## Dependencies

- Python 3.x
- requests (for LLM API)
- Standard library: json, csv, pathlib, collections, ast

## Notes

- LLM classification uses Qwen3-VL-30B-A3B-Instruct model
- API endpoint: http://saltyfish.eecs.umich.edu:8000/v1/chat/completions
- Classification includes food ingredients, beverages, and prepared dishes
- Analysis focuses on P01 participant only (can be extended to other participants)

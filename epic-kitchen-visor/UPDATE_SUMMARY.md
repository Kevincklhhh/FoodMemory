# VISOR Food Index Update Summary

## Date: 2025-11-11

## Overview
Fixed critical bug in Step 4 (food image indexing) and regenerated all downstream analyses. The bug caused 35 videos to be completely missing from the index, resulting in incomplete food abundance analysis.

---

## Bug Impact

### Before Fix (Buggy Data)
- **Videos indexed**: 119 out of 158 (75.3%)
- **Food images indexed**: 19,591
- **Food classes**: 122
- **Missing videos**: 35 videos (24.7%)
- **Missing images**: 4,342 images

### After Fix (Corrected Data)
- **Videos indexed**: 154 out of 158 (97.5%)
- **Food images indexed**: 23,933
- **Food classes**: 127
- **Recovered videos**: 35 videos
- **Recovered images**: 4,342 images

### Impact Breakdown
| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Videos with food | 119 | 154 | +35 (+29.4%) |
| Food images | 19,591 | 23,933 | +4,342 (+22.2%) |
| Food classes | 122 | 127 | +5 (+4.1%) |
| Coverage | 75.3% | 97.5% | +22.2% |

---

## Specific Example: Yoghurt

### Before Fix
- **Videos**: 6
- **Videos list**: P02_03, P02_102, P04_03, P04_101, P04_109, P08_21
- **Missing**: P02_12 (18 frames), P22_107

### After Fix
- **Videos**: 8
- **Videos list**: P02_03, P02_102, **P02_12**, P04_03, P04_101, P04_109, P08_21, **P22_107**
- **Participants**: 4 (P02, P04, P08, P22)
- **Settings**: 6
- **Risk ratio**: 1.33x (LOW)
- **Distribution**: P02(3 videos), P04(3 videos), P08(1 video), P22(1 video)

**Result**: P02_12 yoghurt frames are now retrievable via `query_food_inventory.py`

---

## Updated Files

### Core Data Files
1. ✅ **food_image_index.json** - Rebuilt with corrected frame paths
2. ✅ **food_inventory_lookup.json** - Rebuilt with all videos
3. ✅ **food_per_video.json** - Regenerated with 154 videos
4. ✅ **food_per_video.csv** - Updated statistics
5. ✅ **food_per_video.txt** - Updated human-readable report
6. ✅ **food_per_video_simple.txt** - Updated simple list
7. ✅ **food_abundance_analysis.json** - Recalculated contamination risks

### Script Changes
1. ✅ **4_build_food_image_index.py** - Fixed split detection logic
2. ✅ **query_food_inventory.py** - Added `--video` filter option

### Documentation
1. ✅ **BUG_REPORT_P02_12_YOGHURT.md** - Detailed bug analysis
2. ✅ **VIDEO_FILTER_FEATURE.md** - New feature documentation
3. ✅ **UPDATE_SUMMARY.md** - This file

---

## Top Food Classes (Updated Statistics)

### Most Common Foods by Occurrence
| Food | Occurrences | Videos | Change |
|------|-------------|--------|--------|
| water | 4,080 | 112 | +652 images |
| onion | 1,798 | 34 | +248 images |
| meat | 1,430 | 14 | +528 images |
| food | 1,178 | 43 | +197 images |
| cheese | 1,098 | 34 | +216 images |
| dough | 1,004 | 11 | +401 images |
| carrot | 998 | 13 | +213 images |

### Contamination Risk Categories
- **HIGH risk**: 2 foods (water, salmon)
- **MEDIUM risk**: 24 foods
- **LOW risk**: 101 foods

---

## Videos with Most Food Items

| Video | Unique Foods | Total Occurrences |
|-------|--------------|-------------------|
| P04_05 | 22 | - |
| P08_21 | 22 | - |
| P30_05 | 21 | 750 |
| P01_09 | 19 | 742 |
| P30_111 | 19 | 510 |
| P10_04 | 18 | 500 |

---

## Query Examples with Corrected Data

### Basic query
```bash
python3 query_food_inventory.py --food yoghurt
# Returns 8 videos (was 6)
```

### Video-specific query (NEW FEATURE)
```bash
python3 query_food_inventory.py --food yoghurt --video P02_12
# Returns 18 frames with yoghurt from P02_12
```

### Export to CSV
```bash
python3 query_food_inventory.py --food yoghurt --video P02_12 --export output.csv
# Exports all 18 yoghurt frames to CSV
```

### Copy images
```bash
python3 query_food_inventory.py --food yoghurt --video P02_12 --copy-images ./images
# Copies all 18 yoghurt frames to directory
```

---

## Verification Status

### ✅ Completed
- [x] Bug identified and fixed in 4_build_food_image_index.py
- [x] Food image index rebuilt (23,933 images from 154 videos)
- [x] Food inventory lookup regenerated
- [x] Food per video analysis regenerated
- [x] Food abundance analysis regenerated
- [x] P02_12 yoghurt frames now retrievable
- [x] Video filter feature added to query script
- [x] All documentation updated

### ✅ Validated
- [x] P02_12 has 18 yoghurt frames in index
- [x] Query returns P02_12 yoghurt correctly
- [x] 35 previously missing videos now indexed
- [x] 4,342 previously missing images now indexed
- [x] Food abundance statistics updated

---

## Key Takeaways

1. **Root Cause**: Script checked for participant directory existence but not video-specific frames, causing videos in the "wrong" split to be skipped

2. **Impact**: 35 videos (22.2% of dataset) were completely missing from the index

3. **Resolution**: Modified script to verify actual video frames exist before accepting a split directory

4. **Design Decision**: Train/val split is for VISOR's segmentation task, not relevant for our food inventory retrieval - we now treat it as an organizational detail

5. **New Feature**: Added `--video` filter to enable per-video food queries

---

## Next Steps

The VISOR food analysis pipeline is now complete and validated:
- ✅ Step 1: Extract food items from annotations
- ✅ Step 2: Create food segments
- ✅ Step 3: Analyze food per video
- ✅ Step 4: Build food image index (FIXED)
- ✅ Step 5: Query and retrieval tools (ENHANCED)

All outputs are now based on the corrected, complete dataset.

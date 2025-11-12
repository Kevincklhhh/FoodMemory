# Bug Report: P02_12 Yoghurt Not Appearing in Food Inventory Query

## Issue Summary
The video P02_12 contains 63 yoghurt annotations in the VISOR ground truth, but `query_food_inventory.py` was not returning any frames from P02_12 when queried for yoghurt.

## Root Cause
**Bug in `4_build_food_image_index.py` at lines 122-132**

The script was checking if a participant directory exists in each split (train/val), but not verifying that the specific video's frames exist in that split.

### The Problem:
1. P02_12 is in the **val** split
2. Participant P02 has videos in both **train** and **val** splits
3. The script found `train/P02/` directory first and assumed all P02 videos were there
4. It never checked `val/P02/` where P02_12 frames actually exist
5. Result: P02_12 frames were never indexed

### Impact:
- **Before fix**: 19,591 images indexed from 119 videos
- **After fix**: 23,933 images indexed from 154 videos
- **Missing data**: 4,342 images from 35 videos were not indexed!

## Investigation Steps

1. **Verified ground truth**: P02_12.json contains 63 yoghurt instances (yoghurt, yoghurt cup, yoghurt lid)

2. **Checked Step 1 output**: visor_food_items.json correctly extracted 18 yoghurt occurrences from P02_12

3. **Checked Step 3 output**: food_per_video.csv correctly has P02_12 yoghurt entries

4. **Checked Step 4 output**: food_image_index.json did NOT have P02_12 yoghurt (only 6 videos with yoghurt instead of 8)

5. **Root cause identified**: Script logic error in split detection

## The Fix

**File**: `4_build_food_image_index.py`
**Lines**: 122-137

### Before (Buggy Code):
```python
# Determine split (check both train and val)
image_path = None
for split in splits:
    potential_path = frames_base_dir / split / participant_id
    if potential_path.exists():
        image_path = potential_path
        break  # <-- BUG: Stops at first participant match, not video match
```

### After (Fixed Code):
```python
# Determine split (check both train and val)
# Need to check if the specific video exists, not just the participant directory
image_path = None
for split in splits:
    potential_path = frames_base_dir / split / participant_id
    if potential_path.exists():
        # Check if this split actually has frames for this video
        # by checking for at least one frame
        sample_frames = list(potential_path.glob(f"{video_id}_frame_*.jpg"))
        if sample_frames:
            image_path = potential_path
            break
```

## Verification

After rebuilding the index with the fix:

```bash
$ python3 query_food_inventory.py --food yoghurt
```

Output now correctly includes P02_12:
```
YOGHURT
  Total occurrences in dataset: 239  (was 135)
  Total videos with this food: 8     (was 6)

  Sample images:
    3. Video P02_12, Frame 4438 -> val/P02/P02_12_frame_0000004438.jpg  ✓
```

## Other Affected Videos

This bug likely affected other videos where a participant has videos split across both train and val. Example cases:
- P02: Has videos in both train (P02_01, P02_03, P02_101, etc.) and val (P02_02, P02_09, P02_12)
- Any participant with videos in both splits would have val videos missed if train directory was checked first

## Resolution

1. ✓ Fixed the bug in `4_build_food_image_index.py`
2. ✓ Rebuilt `food_image_index.json` with correct data
3. ✓ Rebuilt `food_inventory_lookup.json` with correct data
4. ✓ Verified query_food_inventory.py now returns P02_12 yoghurt frames

## Design Decision

The train/val split in VISOR is for segmentation tasks, which is not relevant for our food inventory retrieval use case. The code now simply searches across all splits to find frames for each video, treating the split as an implementation detail of how VISOR organizes their data.

Updated documentation in `4_build_food_image_index.py` to clarify this approach.

## Impact Assessment

**35 videos** were completely missing from the index due to this bug. The workflow should be considered **fixed and validated** as of this rebuild.

---
**Date**: 2025-11-11
**Fixed by**: Claude Code debugging session

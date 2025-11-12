# Corrected Understanding: What Was Actually Affected by the Bug

## Date: 2025-11-11

---

## Key Insight

**The bug ONLY affected Step 4 (food image indexing).** Steps 1, 2, 3, and food abundance analysis were **ALWAYS CORRECT** from the beginning!

---

## Pipeline Data Flow

### Step 1: Extract Food Items ✅ ALWAYS CORRECT
**Script**: `1_extract_food_items.py`
**Input**: VISOR annotations (train + val JSON files)
**Output**: `visor_food_items.json`
**Process**: Reads annotation JSON files from both `train/` and `val/` directories

```python
for split in ['train', 'val']:  # Processes BOTH splits
    split_dir = visor_dir / 'annotations' / split
    for json_file in split_dir.glob('*.json'):
        # Extract food items...
```

**Result**: Always had all 8 yoghurt videos including P02_12 and P22_107

---

### Step 3: Analyze Food Per Video ✅ ALWAYS CORRECT
**Script**: `3_analyze_food_per_video.py`
**Input**: `visor_food_items.json` (from Step 1)
**Output**: `food_per_video.json`, `food_per_video.csv`, etc.
**Process**: Reads Step 1 output and analyzes food occurrences

```python
def load_food_items(json_path: str):
    with open('visor_food_items.json', 'r') as f:
        return json.load(f)
```

**Result**: Always had all 8 yoghurt videos (inherited from Step 1)

---

### Food Abundance Analysis ✅ ALWAYS CORRECT
**Script**: `analyze_food_abundance.py`
**Input**: `food_per_video.json` (from Step 3)
**Output**: `food_abundance_analysis.json`, `food_abundance_table.csv`
**Process**: Reads Step 3 output and calculates risks

```python
def analyze_food_abundance(food_per_video_path: str = 'food_per_video.json'):
    with open(food_per_video_path, 'r') as f:
        data = json.load(f)
```

**Result**: Always showed yoghurt with 8 videos (inherited from Step 3)

---

### Step 4: Build Food Image Index ❌ HAD BUG → ✅ NOW FIXED
**Script**: `4_build_food_image_index.py`
**Input**: `visor_food_items.json` (from Step 1) + RGB frame files
**Output**: `food_image_index.json`, `food_inventory_lookup.json`
**Process**: Reads Step 1 output, then **searches disk for actual frame files**

**THE BUG WAS HERE:**
```python
# BUGGY CODE (before fix):
for split in splits:
    potential_path = frames_base_dir / split / participant_id
    if potential_path.exists():
        image_path = potential_path
        break  # ❌ Stopped at first participant match!
```

**Why it failed:**
1. Step 1 output had all 8 yoghurt videos including P02_12
2. Step 4 tried to find RGB frames for P02_12 on disk
3. Found `train/P02/` directory first
4. Assumed P02_12 frames were there (they weren't - they were in `val/P02/`)
5. Failed to index P02_12 because frames didn't exist at expected path
6. **Result: Only 6 yoghurt videos in image index** (missing P02_12, P22_107)

---

## What Files Were Affected?

### ✅ NEVER HAD THE BUG
These were always correct:
- `visor_food_items.json` (Step 1) - ✅ 8 yoghurt videos
- `food_per_video.json` (Step 3) - ✅ 8 yoghurt videos
- `food_per_video.csv` (Step 3) - ✅ 8 yoghurt videos
- `food_per_video.txt` (Step 3) - ✅ 8 yoghurt videos
- `food_abundance_analysis.json` - ✅ 8 yoghurt videos
- `food_abundance_table.csv` - ✅ 8 yoghurt videos

### ❌ HAD THE BUG (Now Fixed)
These were incomplete:
- `food_image_index.json` (Step 4) - ❌ Only 6 yoghurt videos → ✅ Now 8
- `food_inventory_lookup.json` (Step 4) - ❌ Only 6 yoghurt videos → ✅ Now 8
- `query_food_inventory.py` results - ❌ Couldn't find P02_12 → ✅ Now works

---

## Why This Matters

### The Confusion
When you said "so @food_abundance_table.csv did not change?" - you were RIGHT!

It didn't change because it was **already correct**. The bug was downstream in Step 4.

### The Impact
The bug only affected:
1. **Image-level queries** via `query_food_inventory.py`
2. **Frame path lookups** from the image index
3. **Visual retrieval** of specific food instances

It did NOT affect:
1. **Statistical analysis** (food counts, video counts, participant counts)
2. **Abundance calculations** (contamination risk, multi-participant analysis)
3. **Video-level summaries** (which videos have which foods)

---

## Corrected Timeline

### Before Any Fix
- Step 1: ✅ 8 yoghurt videos
- Step 3: ✅ 8 yoghurt videos
- Food abundance: ✅ 8 yoghurt videos
- Step 4: ❌ 6 yoghurt videos (BUG)
- Query tool: ❌ Can't find P02_12 yoghurt

### After Fix
- Step 1: ✅ 8 yoghurt videos (unchanged)
- Step 3: ✅ 8 yoghurt videos (unchanged)
- Food abundance: ✅ 8 yoghurt videos (unchanged - we regenerated it but data was same)
- Step 4: ✅ 8 yoghurt videos (FIXED)
- Query tool: ✅ Can find P02_12 yoghurt (FIXED)

---

## Why We Regenerated Everything

Even though Steps 1-3 and food abundance were correct, we regenerated them to:
1. **Verify consistency** across all files
2. **Update statistics** that depend on Step 4 being complete
3. **Ensure all outputs match** the fixed state
4. **Provide clean baseline** for future work

But the actual **data values didn't change** for those steps - they were already right!

---

## The Real Bug Impact

**What actually broke:**
- 35 videos (including P02_12, P22_107) couldn't be queried for images
- 4,342 images were not indexed
- Visual retrieval was incomplete

**What never broke:**
- Video counts per food (always 8 for yoghurt)
- Participant analysis (always knew P02, P04, P08, P22 had yoghurt)
- Contamination risk calculations (always showed correct ratios)
- Frame-level metadata (always in visor_food_items.json)

---

## Conclusion

You were absolutely correct to question why `food_abundance_table.csv` didn't change!

**Answer**: Because Step 1 (which Step 3 and food_abundance depend on) reads directly from VISOR annotation JSON files, which include both train and val splits. The bug only occurred in Step 4 when trying to match videos to their RGB frame files on disk.

The confusion arose because:
1. The bug description focused on "35 missing videos"
2. But those videos were only missing from the **image index**, not from the **data analysis**
3. All upstream steps (1, 3, abundance) were always complete

**Key Takeaway**: The split detection bug affected **image retrieval** but not **data analysis**.

---

**Date**: 2025-11-11
**Clarification by**: User observation + Claude verification

# How to Run the Complete Pipeline

**Pipeline**: Food State Change Annotation Benchmark Creation

**Directory**: `/home/kailaic/NeuroTrace/kitchen/HDEPIC/pipelines/state_change_annotation`

---

## Quick Start

```bash
cd /home/kailaic/NeuroTrace/kitchen/HDEPIC/pipelines/state_change_annotation

# Step 1: Create annotation tasks (blocks, frames, clips)
python 01_create_annotation_tasks.py \
  --input ../../outputs/food_analysis/per_video_extractions/P01-20240203-121517_food_items.json \
  --video-path ../../data/HD-EPIC/Videos/P01/P01-20240203-121517.mp4 \
  --output-dir ../../outputs/state_change_annotation \
  --max-block-duration 30.0 \
  --frame-offset 1.0

# Step 2: Add grounding masks (Hands23)
conda activate hands23
python 02_add_grounding_to_tasks.py \
  --tasks-file ../../outputs/state_change_annotation/P01-20240203-121517_annotation_tasks.json \
  --output-dir ../../outputs/state_change_annotation

# Step 3: Add VLM state predictions (Qwen3-VL)
python 03_vlm_state_tracking.py \
  --tasks-file ../../outputs/state_change_annotation/P01-20240203-121517_annotation_tasks.json \
  --output-dir ../../outputs/state_change_annotation \
  --fps 1
```

---

## Step-by-Step Guide

### Prerequisites

1. **Video file**: HD-EPIC video in `data/HD-EPIC/Videos/P01/`
2. **Food extractions**: Run food analysis pipeline first
3. **Hands23 environment**: `conda activate hands23`
4. **VLM API access**: Ensure `saltyfish.eecs.umich.edu:8000` is reachable

### Step 1: Create Annotation Tasks (5-10 minutes)

**Input**:
- Food narrations JSON from food analysis pipeline
- Source video file

**Process**:
- Merge narrations into 30-second blocks
- Extract video clips (one per block)
- Extract frames (before/after each narration)
- Initialize state tables

**Output**:
```
outputs/state_change_annotation/
├── P01-20240203-121517_annotation_tasks.json
└── assets/P01-20240203-121517/
    └── block_XXX/
        ├── *_clip.mp4
        ├── *_narrXX_before.jpg
        └── *_narrXX_after.jpg
```

**Runtime**: ~2-3 minutes

**Command**:
```bash
python 01_create_annotation_tasks.py \
  --input ../../outputs/food_analysis/per_video_extractions/P01-20240203-121517_food_items.json \
  --video-path ../../data/HD-EPIC/Videos/P01/P01-20240203-121517.mp4 \
  --output-dir ../../outputs/state_change_annotation
```

### Step 2: Add Grounding Masks (5-10 minutes)

**Input**:
- Annotation tasks JSON from Step 1
- Extracted frames

**Process**:
- Run Hands23 detector on each frame
- Detect hand-object interactions
- Generate object masks
- Add grounding data to tasks

**Output**:
```
outputs/state_change_annotation/
├── P01-20240203-121517_annotation_tasks.json  (updated)
└── assets/P01-20240203-121517/
    └── XXX/masks/
        └── *_hands23_obj*.png
```

**Runtime**: ~3-5 minutes

**Command**:
```bash
conda activate hands23

python 02_add_grounding_to_tasks.py \
  --tasks-file ../../outputs/state_change_annotation/P01-20240203-121517_annotation_tasks.json \
  --output-dir ../../outputs/state_change_annotation
```

### Step 3: Add VLM State Predictions (10-15 minutes)

**Input**:
- Annotation tasks JSON from Step 2
- Video clips
- State taxonomy

**Process**:
- For each block:
  - Send video clip to Qwen3-VL
  - Provide current food memory
  - Provide state taxonomy schema
  - Get state predictions (before/after)
- Update food memory
- Generate timeline

**Output**:
```
outputs/state_change_annotation/
└── P01-20240203-121517_annotation_tasks.json       (updated with VLM predictions)
```

**Runtime**: ~10-15 minutes (30-60s per block)

**Command**:
```bash
python 03_vlm_state_tracking.py \
  --tasks-file ../../outputs/state_change_annotation/P01-20240203-121517_annotation_tasks.json \
  --output-dir ../../outputs/state_change_annotation \
  --fps 1
```

**Resume if interrupted**:
```bash
python 03_vlm_state_tracking.py \
  --tasks-file ../../outputs/state_change_annotation/P01-20240203-121517_annotation_tasks.json \
  --output-dir ../../outputs/state_change_annotation \
  --fps 1 \
  --start-task 5  # Resume from block 5
```

---

## Step 4: View and Annotate Results

### Purpose
Interactive visualization interface for reviewing VLM predictions and grounding results

### Output
Browser window with interactive annotation interface

### Runtime
Instant (starts HTTP server)

### Command
```bash
cd ../../outputs/state_change_annotation
python view_annotations.py
```

**What happens:**
1. Starts local HTTP server at http://localhost:8000
2. Automatically opens browser to visualization.html
3. You upload the annotation_tasks.json file
4. Video, frames, and masks load automatically

### Interface Features

- **Timeline Navigation**: Click to seek, use arrow keys for fine control
- **Frame Gallery**: View before/after frames for each narration
- **Mask Overlay**: Toggle grounding masks on/off
- **Food State Display**: Review VLM predictions (before/after) with reasoning
- **Click-to-Seek**: Click any frame to jump to that timestamp in video

### Optional Arguments

```bash
# Use different port
python view_annotations.py --port 9000

# Don't auto-open browser
python view_annotations.py --no-browser
```

### Annotation Workflow

1. Start server: `python view_annotations.py`
2. Upload `P01-20240203-121517_annotation_tasks.json` in browser
3. Review each block:
   - Watch video clip
   - Examine before/after frames
   - Check grounding masks
   - Review VLM state predictions
4. Validate or correct states (future: annotation UI)

---

## Expected Outputs

### Final File Structure

```
outputs/state_change_annotation/
├── P01-20240203-121517_annotation_tasks.json        # Complete pipeline output (blocks, frames, clips, grounding, VLM states)
├── visualization.html                               # Interactive annotation interface
├── view_annotations.py                              # Visualization server launcher
└── assets/P01-20240203-121517/
    ├── block_000/
    │   ├── P01-20240203-121517_block_000_clip.mp4
    │   ├── P01-20240203-121517_block_000_narr00_before.jpg
    │   ├── P01-20240203-121517_block_000_narr00_after.jpg
    │   └── ...
    ├── 000/masks/
    │   ├── P01-20240203-121517_block_000_narr00_before_hands23_obj0.png
    │   └── ...
    └── ...
```

### Annotation Tasks JSON Structure

```json
{
  "task_id": "P01-20240203-121517_block_000",
  "video_id": "P01-20240203-121517",
  "block_start_time": 56.11,
  "block_end_time": 85.21,
  "narrations_in_block": ["...", "..."],
  "target_food_nouns": ["lemon", "meat"],
  "assets": {
    "clip_path": "...",
    "before_frames": [...],
    "after_frames": [...]
  },
  "auto_grounding_data": {
    "lemon": {
      "before_frames": [...],
      "after_frames": [...]
    }
  },
  "ground_truth_state_table": {
    "lemon": {
      "vlm_state_before": {
        "container_state": {...},
        "preparation_state": {...},
        "consumption_state": {...},
        "location_state": {...}
      },
      "vlm_state_after": {...},
      "vlm_reasoning": "Lemon halves picked up and put in fridge",
      "block_time_range": {"start": 56.11, "end": 85.21},
      "annotator_state_before": null,
      "annotator_state_after": null
    }
  }
}
```

---

## Troubleshooting

### Step 1 Issues

**Error**: `Video file not found`
- Check `--video-path` points to actual video file
- Verify file exists: `ls ../../data/HD-EPIC/Videos/P01/P01-20240203-121517.mp4`

**Error**: `Food items JSON not found`
- Run food analysis pipeline first
- Check file exists: `ls ../../outputs/food_analysis/per_video_extractions/`

### Step 2 Issues

**Error**: `No module named 'detectron2'`
- Activate hands23 environment: `conda activate hands23`

**Error**: `FIRSTOBJ config error`
- This is fixed in current version
- Ensure using latest `02_add_grounding_to_tasks.py`

### Step 3 Issues

**Error**: `Connection timeout`
- Check VLM API: `curl http://saltyfish.eecs.umich.edu:8000/health`
- Increase timeout in script (currently 300s)

**Error**: `JSON parsing error`
- VLM returned non-JSON response
- Check raw response in error output
- Try reducing `--fps` to speed up processing

**Out of memory**:
- Process blocks one at a time with `--start-task`
- Reduce `--fps` to process fewer frames

---

## Parameters Reference

### 01_create_annotation_tasks.py

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--input` | Food items JSON (required) | - |
| `--video-path` | Source video file (required) | - |
| `--output-dir` | Output directory | `../../outputs/state_change_annotation` |
| `--max-block-duration` | Max block duration (seconds) | 30.0 |
| `--frame-offset` | Frame offset before/after (seconds) | 1.0 |

### 02_add_grounding_to_tasks.py

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--tasks-file` | Annotation tasks JSON (required) | - |
| `--output-dir` | Output directory (required) | - |
| `--hands23-config` | Hands23 config YAML | `../../models/hands23_detector/...yaml` |
| `--hands23-weights` | Hands23 weights PTH | `../../models/hands23_detector/...pth` |

### 03_vlm_state_tracking.py

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--tasks-file` | Annotation tasks JSON (required) | - |
| `--output-dir` | Output directory (required) | - |
| `--taxonomy-file` | State taxonomy JSON | `food_state_taxonomy.json` |
| `--fps` | Video sampling rate (frames/sec) | 1 |
| `--start-task` | Resume from task index | 0 |

---

## Total Runtime

**Per video** (P01-20240203-121517 example):
- Step 1: ~2-3 minutes
- Step 2: ~3-5 minutes
- Step 3: ~10-15 minutes

**Total**: ~15-23 minutes

---

## Next Steps After Pipeline Completion

1. **Verify outputs**:
   ```bash
   ls ../../outputs/state_change_annotation/P01-20240203-121517_*.json
   ```

2. **Inspect VLM predictions**:
   ```bash
   python -m json.tool ../../outputs/state_change_annotation/P01-20240203-121517_annotation_tasks.json | less
   ```

3. **Create visualization tool** (future):
   ```bash
   python 04_visualize_state_timeline.py --video P01-20240203-121517
   ```

4. **Human annotation** (future):
   - Load annotation tasks
   - Review VLM predictions
   - Fill in `annotator_state_before/after` fields
   - Save corrected annotations

---

## Batch Processing Multiple Videos

To process all P01 videos:

```bash
# Create list of videos
find ../../outputs/food_analysis/per_video_extractions/ -name "P01-*_food_items.json" > video_list.txt

# Process each video
while read -r food_json; do
    video_id=$(basename "$food_json" _food_items.json)
    video_path="../../data/HD-EPIC/Videos/P01/${video_id}.mp4"

    echo "Processing $video_id..."

    # Step 1
    python 01_create_annotation_tasks.py \
      --input "$food_json" \
      --video-path "$video_path" \
      --output-dir ../../outputs/state_change_annotation

    # Step 2
    python 02_add_grounding_to_tasks.py \
      --tasks-file "../../outputs/state_change_annotation/${video_id}_annotation_tasks.json" \
      --output-dir ../../outputs/state_change_annotation

    # Step 3
    python 03_vlm_state_tracking.py \
      --tasks-file "../../outputs/state_change_annotation/${video_id}_annotation_tasks.json" \
      --output-dir ../../outputs/state_change_annotation

    echo "✓ Completed $video_id"
done < video_list.txt
```

---

## Summary

**Complete Pipeline**:
1. ✅ Extract blocks, frames, clips
2. ✅ Generate object masks (Hands23)
3. 🔴 **Next**: Predict food states (VLM)
4. Future: Visualize states
5. Future: Human validation
6. Future: Evaluation metrics

**Current Status**: Ready to run Step 3 (VLM state tracking)!

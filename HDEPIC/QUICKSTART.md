# Quick Start Guide: Processing New Videos

This guide shows you how to process a new HD-EPIC video from raw narrations to final food memory states.

## Prerequisites

- HD-EPIC narration CSV file for your participant (e.g., `participant_P01_narrations.csv`)
- Video file (e.g., `P01-20240205-143021.mp4`)
- Qwen3-VL API access

## Two-Step Process

### Step 1: Extract Food Items from Narrations

**Location:** `pipelines/food_analysis/`

```bash
cd pipelines/food_analysis

# If first time: Classify HD-EPIC nouns (run once)
python 1_classify_hdepic_food_nouns.py

# Extract food items from your participant's narrations
python 2_extract_hdepic_food_items.py
```

This creates: `outputs/food_analysis/per_video_extractions/{video_id}_food_items.json`

### Step 2: Track Food State Changes

**Location:** `pipelines/state_change_annotation/`

**Option A: Automated (Recommended)**

```bash
cd pipelines/state_change_annotation

./run_full_pipeline.sh YOUR_VIDEO_ID
```

**Option B: Manual Control**

```bash
cd pipelines/state_change_annotation

# Create annotation tasks
python 01_create_annotation_tasks.py \
    --input ../../outputs/food_analysis/per_video_extractions/YOUR_VIDEO_ID_food_items.json \
    --video-path ../../data/HD-EPIC/Videos/P01/YOUR_VIDEO_ID.mp4

# Run VLM state tracking
python 03_vlm_state_tracking.py --clear-logs
```

## Outputs

All outputs are in `outputs/state_change_annotation/`:

- `{video_id}_annotation_tasks.json` - Complete annotation data (blocks, frames, clips, states)
- `{video_id}_final_food_memory.json` - **End-of-video food states** ⭐
- `vlm_logs/` - VLM prompts and responses for debugging

## Example: Process Video P01-20240205-143021

```bash
# Step 1: Extract food items (if not already done)
cd pipelines/food_analysis
python 2_extract_hdepic_food_items.py

# Step 2: Run state tracking
cd ../state_change_annotation
./run_full_pipeline.sh P01-20240205-143021 \
    ../../outputs/food_analysis/per_video_extractions/P01-20240205-143021_food_items.json \
    ../../data/HD-EPIC/Videos/P01/P01-20240205-143021.mp4
```

## Re-running VLM Experiments

To test different prompts without re-extracting frames:

```bash
cd pipelines/state_change_annotation

# Edit the prompt in 03_vlm_state_tracking.py
# Then re-run:
python 03_vlm_state_tracking.py --clear-logs
```

The `--clear-logs` flag clears old VLM outputs but preserves blocks/frames/clips.

## Troubleshooting

**"Food items JSON not found"**
→ Run `food_analysis/2_extract_hdepic_food_items.py` first

**"VLM creating too many instances"**
→ Check `outputs/state_change_annotation/vlm_logs/` for prompt/response
→ Ensure you're using latest version with accumulation support

**"Video file not found"**
→ Check video path: `data/HD-EPIC/Videos/{participant}/{video_id}.mp4`

## More Information

- **Pipeline overview:** `pipelines/PIPELINE_OVERVIEW.md`
- **State tracking details:** `pipelines/state_change_annotation/README.md`
- **Technical decisions:** `pipelines/state_change_annotation/PROGRESS.md`

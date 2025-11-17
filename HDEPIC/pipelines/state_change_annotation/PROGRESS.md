# State Change Annotation Pipeline - Progress Tracker

**Last Updated**: 2025-11-17

---

## Pipeline Overview

**Goal**: Create a benchmark dataset for food state change tracking in egocentric cooking videos.

**Pipeline Steps**:
1. ✅ `01_create_annotation_tasks.py` - Block merging, frame/clip extraction
2. ✅ `02_add_grounding_to_tasks.py` - Hands23 object mask generation
3. ✅ `03_vlm_state_tracking.py` - VLM state prediction (text-first, persistent memory)
4. ✅ `view_annotations.py` - Interactive visualization with block navigation

**Key Features**:
- **Text-first VLM**: Relies on narrations, validates with video
- **Persistent state**: Each block's `state_after` flows to next block
- **No state_before in JSON**: Auto-derived from previous block's `state_after`
- **30s blocks**: Optimized for VLM context and annotation consistency

---

## Quick Start

```bash
# Full pipeline
cd pipelines/state_change_annotation
python 01_create_annotation_tasks.py --input ... --video-path ... --output-dir ...
python 02_add_grounding_to_tasks.py --tasks-file ... --output-dir ...
python 03_vlm_state_tracking.py --tasks-file ... --output-dir ...
cd ../../outputs/state_change_annotation && python view_annotations.py
```

---

## Current Status

### ✅ Step 1: Create Annotation Tasks

**Script**: `01_create_annotation_tasks.py`

**Status**: COMPLETE

**Implementation**:
- 30-second maximum block duration constraint
- Sequential narration aggregation
- Frame extraction: before/after for each narration (±1s)
- Clip extraction: one per block with ±1s padding
- No boundary frames (only narration-based frames)

**Test Results** (P01-20240203-121517):
- Input: 42 food narrations
- Output: 13 blocks (durations: 3-30s, one 45s edge case)
- Assets: 84 frames (42 pairs), 13 clips
- File: `P01-20240203-121517_annotation_tasks.json`

**Usage**:
```bash
python 01_create_annotation_tasks.py \
  --input ../../outputs/food_analysis/per_video_extractions/P01-20240203-121517_food_items.json \
  --video-path ../../data/HD-EPIC/Videos/P01/P01-20240203-121517.mp4 \
  --output-dir ../../outputs/state_change_annotation \
  --max-block-duration 30.0 \
  --frame-offset 1.0
```

---

### ✅ Step 2: Add Grounding Data

**Script**: `02_add_grounding_to_tasks.py`

**Status**: COMPLETE

**Implementation**:
- Hands23 detector integration (detectron2-based)
- Hand-object interaction detection
- Interaction levels: "first" (hand-object), "second" (object-object)
- Threshold configuration: Hand=0.5, FirstObj=0.3, SecondObj=0.2

**Test Results** (P01-20240203-121517):
- Total grounding results: 306
- Average per block: 23.5 masks
- Unique mask files: ~150 PNG files
- Stored in: `assets/{video_id}/{block_num}/masks/`

**Usage**:
```bash
conda activate hands23

python 02_add_grounding_to_tasks.py \
  --tasks-file ../../outputs/state_change_annotation/P01-20240203-121517_annotation_tasks.json \
  --output-dir ../../outputs/state_change_annotation \
  --hands23-config ../../models/hands23_detector/faster_rcnn_X_101_32x8d_FPN_3x_Hands23.yaml \
  --hands23-weights ../../models/hands23_detector/model_weights/model_hands23.pth
```

---

### ✅ Step 3: VLM State Tracking

**Script**: `03_vlm_state_tracking.py`

**Status**: COMPLETED

**Implementation**:
- Qwen3-VL video analysis (API: `saltyfish.eecs.umich.edu:8000`)
- **Text-first approach**: VLM relies on narration text, uses video for validation
- State taxonomy integration (`food_state_taxonomy.json`)
- Persistent food memory across blocks (each block's `state_after` becomes implicit `state_before` for next block)
- VLM predicts only `state_after` (current state at end of block)

**State Persistence**:
- Block 0: VLM predicts state_after → stored in memory
- Block 1: Memory provides implicit state_before, VLM predicts new state_after
- Block N: Continues state chain from Block N-1

**Instance-Based Tracking**:
- Supports food splits: "Pour flour from bag to bowl" creates two instances
- Each instance has unique ID: `{food_noun}_{semantic_name}_{counter}`
- VLM decides when to create new instances vs update existing
- Simple retrieval: All instances of target foods provided to VLM

**Output Structure**:
Each task's `ground_truth_state_table` contains:
```json
{
  "<food_noun>": {
    "instances": {
      "flour_from_bag_001": {
        "vlm_state_after": { ... },     // Current state at end of block
        "vlm_reasoning": "...",         // Based on narrations + video validation
        "semantic_name": "from_bag",
        "block_time_range": {"start": 56.11, "end": 85.21},
        "annotator_state_after": null   // For human verification
      },
      "flour_in_bowl_002": {
        "vlm_state_after": { ... },
        "vlm_reasoning": "...",
        "semantic_name": "in_bowl",
        "block_time_range": {"start": 123.45, "end": 145.67},
        "annotator_state_after": null
      }
    }
  }
}
```
Note: Multiple instances per food noun supported; VLM has full autonomy

**Generated Files**:
1. `{video_id}_annotation_tasks.json` - Complete output with blocks, frames, clips, grounding masks, and VLM state predictions

**Usage**:
```bash
python 03_vlm_state_tracking.py \
  --tasks-file ../../outputs/state_change_annotation/P01-20240203-121517_annotation_tasks.json \
  --output-dir ../../outputs/state_change_annotation \
  --taxonomy-file food_state_taxonomy.json \
  --fps 1
```

**Runtime**: ~10-15 minutes for 13 blocks (30-60s per block)

**Re-running with Updated Script**:
If you have old JSON with `state_before` fields:
```bash
# Clear old outputs (keeps food extraction data)
rm ../../outputs/state_change_annotation/P01-20240203-121517_annotation_tasks.json
rm -rf ../../outputs/state_change_annotation/assets/P01-20240203-121517/

# Re-run pipeline from Step 1
python 01_create_annotation_tasks.py \
  --input ../../outputs/food_analysis/per_video_extractions/P01-20240203-121517_food_items.json \
  --video-path ../../data/HD-EPIC/Videos/P01/P01-20240203-121517.mp4 \
  --output-dir ../../outputs/state_change_annotation

# Then continue with Steps 2 and 3
```

---

## Step 4: Visualization and Annotation ✅

**Purpose**: Interactive web interface for reviewing and annotating VLM predictions

**Script**: `view_annotations.py`

**Features**:
- Auto-loads video clips, frames, and grounding masks from JSON
- Block navigation (previous/next buttons, arrow keys)
- Frame gallery with before/after images for each narration
- Toggle grounding mask overlays on frames
- Food state comparison:
  - Before: Auto-derived from previous block's state_after
  - After: Current block's VLM prediction
- Click frames to seek video to that timestamp

**Usage**:
```bash
cd ../../outputs/state_change_annotation
python view_annotations.py
# Opens browser at http://localhost:8000/visualization.html
# Upload the annotation_tasks.json file to begin
```

**Optional Arguments**:
```bash
python view_annotations.py --port 9000       # Use different port
python view_annotations.py --no-browser      # Don't auto-open browser
```

**Workflow**:
1. Start server with `python view_annotations.py`
2. Upload `{video_id}_annotation_tasks.json` in the browser
3. Video, frames, and masks load automatically
4. Review VLM predictions and grounding results
5. Use interface to validate/annotate food states

**Interface Controls**:
- **Timeline**: Click to seek, arrow keys for fine control (0.1s/1s)
- **Frame Gallery**: Click frames to jump to that timestamp
- **Show Masks**: Toggle to overlay hand-object detection masks
- **Food States**: View VLM predictions and reasoning for each food item

---

## File Organization

```
pipelines/state_change_annotation/
├── 01_create_annotation_tasks.py      # Block merging, asset extraction
├── 02_add_grounding_to_tasks.py       # Hands23 grounding
├── 03_vlm_state_tracking.py           # VLM state prediction
├── grounding_utils.py                 # Hands23 utilities
├── food_state_taxonomy.json           # State schema
├── README.md                          # Full documentation
└── PROGRESS.md                        # This file

outputs/state_change_annotation/
├── {video_id}_annotation_tasks.json   # Complete pipeline output (blocks, frames, clips, grounding, VLM states)
├── visualization.html                 # Interactive annotation interface
├── view_annotations.py                # Visualization server launcher
└── assets/{video_id}/
    └── block_XXX/
        ├── *_clip.mp4                   # Video clip
        ├── *_narrXX_before.jpg          # Before frames
        ├── *_narrXX_after.jpg           # After frames
        └── masks/
            └── *_hands23_obj*.png       # Grounding masks
```

---

## Data Schema

### State Taxonomy Categories

From `food_state_taxonomy.json`:

1. **Container State**:
   - `container_type`: original_packaging, storage_container, serving_dish, bowl, plate, none

2. **Preparation State**:
   - `form_state`: whole, prepared_ingredient, cooking_in_progress, cooked_dish, leftover, unknown

3. **Consumption State**:
   - `quantity`: full, partial, nearly_empty, consumed, unknown

4. **Location State**:
   - `location_type`: storage, shoping bag, prep_surface, consumption_area, in_hand, unknown

---

## Next Steps

### 🟡 Immediate

1. **Run VLM State Tracking** on P01-20240203-121517
   - Execute `03_vlm_state_tracking.py`
   - Verify VLM predictions quality
   - Check timeline JSON format

2. **Create State Visualization Tool**
   - Script: `04_visualize_state_timeline.py`
   - Play video with synchronized state display
   - Show state changes at block boundaries

### 🟡 Short-term

3. **Scale to All P01 Videos**
   - Batch processing script
   - Process all videos with food narrations
   - Aggregate statistics

4. **Human Annotation Interface**
   - Tool for annotators to validate/correct VLM predictions
   - Update `annotator_state_before/after` fields
   - Track annotation quality metrics

5. **Evaluation Framework**
   - Compare VLM predictions vs. human annotations
   - Metrics: exact match, category accuracy, state transition accuracy
   - Error analysis

### 🟡 Long-term

6. **GroundingSAM Integration**
   - Add food-specific semantic grounding
   - Compare with Hands23 results
   - Ensemble grounding predictions

7. **Multi-Modal VLM Comparison**
   - Test other VLMs: GPT-4o, Gemini
   - Compare prediction quality
   - Ensemble predictions

---

## Technical Decisions

### 30-Second Block Duration
- **Rationale**: VLM context window compatibility
- **Trade-off**: Some semantic coherence lost vs. temporal consistency
- **Edge cases**: Single narrations >30s kept as-is

### VLM-First Annotation
- **Approach**: VLM predictions first, human validation second
- **Benefits**: Faster annotation, consistency baseline
- **Validation**: Human annotators correct VLM errors

### Persistent Food Memory
- **Design**: Food states persist across blocks within a video
- **Example**: Flour state evolves: storage → prep_surface → mixed → cooked
- **Usage**: VLM sees previous states when predicting new states

### Instance-Based Food Tracking
- **Problem**: Food splits create multiple physical instances (e.g., pour flour from bag to bowl)
- **Solution**: Unique instance IDs with semantic names (`flour_from_bag_001`, `flour_in_bowl_002`)
- **VLM Autonomy**: VLM decides when to create new instances vs update existing
- **Retrieval**: Simple filter by food noun - all instances provided to VLM
- **Lifecycle**: Instances removed only when `quantity="consumed"` (explicit consumption)
- **Counter System**: Global counter per food noun prevents duplicate IDs

---

## Known Issues

1. **Hands23 Limitations**:
   - Detects hand-interacted objects (generic)
   - Does not identify specific food types
   - Useful for localization, not classification

2. **VLM API Dependency**:
   - Requires stable connection to `saltyfish.eecs.umich.edu:8000`
   - 30-60s per block processing time
   - May timeout on long videos

3. **State Taxonomy Coverage**:
   - Current schema covers basic cooking states
   - May need expansion for complex preparations
   - Some states difficult to observe visually

---

## References

- **Hands23 Paper**: "Understanding Egocentric Hand-Object Interactions from Hand Pose Estimation"
- **Qwen3-VL**: Multi-modal LLM with video understanding
- **HD-EPIC Dataset**: High-definition egocentric procedural instruction corpus

---

## Contact

For issues or questions:
- Check README.md for detailed documentation
- Review PROGRESS.md for current status
- See pipeline scripts for implementation details

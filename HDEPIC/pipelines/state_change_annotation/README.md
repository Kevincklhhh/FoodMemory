# Food State Change Annotation Pipeline

This pipeline creates annotation tasks for labeling food state changes in HD-EPIC videos, enabling the creation of a benchmark dataset for tracking how food items change during cooking and preparation.

## Overview

The pipeline processes food narrations from HD-EPIC videos and creates structured annotation tasks that include:
- **Interaction blocks**: Groups of temporally-close food narrations (within 10s threshold)
- **Video clips**: Extracted video segments for each block with ±1s temporal context
- **Frame pairs**: Before/after frames for each narration (not block boundaries)
- **Grounding data**: Pre-populated object masks from Hands23 detector
- **State tables**: Templates for annotators to fill in state changes

### Key Design Decisions

**Block Merging Strategy (30-Second Maximum):**
- Narrations are aggregated sequentially until adding another would exceed 30 seconds
- Block duration = `block.end_time - block.start_time`
- **Rationale**: Consistency with VLM context window limits and manageable annotation units
- Edge case: Single narrations >30s are kept as single-narration blocks
- This creates temporally compact, VLM-friendly blocks

**Clip Selection Strategy:**
- Each block gets ONE video clip spanning: `(block.start_time - 1.0s)` to `(block.end_time + 1.0s)`
- This provides full temporal context for all narrations in the block
- Maximum clip duration: ~32 seconds (30s block + 2s padding)
- Annotators can see the complete interaction sequence

**Frame Extraction Strategy:**
- Extracts before/after frames for EACH NARRATION (not block boundaries)
- Before frame: `narration.start_time - 1.0s`
- After frame: `narration.end_time + 1.0s`
- No redundant "boundary" frames for the entire block
- This captures each individual state change precisely

**Interaction Levels (from Hands23):**
- **"first"**: Objects directly touched/grasped by hands (e.g., hand holding knife, hand grasping onion)
- **"second"**: Objects interacted with via first object (e.g., onion being cut by knife)
- This hierarchical detection distinguishes direct hand-food contact from tool-mediated interactions

## Pipeline Components

### 1. Block Creation & Asset Extraction
**Script**: `01_create_annotation_tasks.py`
- 30-second block merging with sequential narration aggregation
- Video clip extraction (block duration + 2s padding)
- Before/after frame extraction for each narration

### 2. Hand-Object Grounding
**Script**: `02_add_grounding_to_tasks.py`
- Hands23 detector integration
- Generates object masks for food items in frames
- Interaction level detection (first/second)

### 3. VLM State Tracking
**Script**: `03_vlm_state_tracking.py`
- Qwen3-VL video analysis with text-first approach
- Persistent state memory across blocks
- **Instance-based tracking**: Supports food splits (e.g., "pour flour from bag to bowl")
- VLM creates instances with semantic IDs: `flour_from_bag_001`, `flour_in_bowl_002`
- Simple retrieval: All instances of target foods provided to VLM
- Uses food state taxonomy schema

### 4. Interactive Visualization
**Script**: `view_annotations.py` (in outputs directory)
- Web-based annotation interface
- Block navigation, frame gallery, mask overlays
- Auto-derives state transitions from block sequence

## Test Run Results

**Video**: P01-20240203-121517 (Bread-making preparation)

**Input**: 42 food narrations with 13 unique food types
- butter, flour, food, lemon, meat, mixture, onion, orange, roll, salt, sandwich, sugar, yeast

**Output**: 13 annotation task blocks (30s max duration)

| Block | Duration | Narrations | Food Nouns |
|-------|----------|------------|------------|
| 0 | 29.10s | 4 | lemon, meat |
| 1 | 15.80s | 3 | butter, meat |
| 2 | 3.08s | 3 | onion, orange |
| 3 | 24.51s | 3 | food, yeast |
| 4 | 8.56s | 3 | food, roll, sugar |
| 5 | 7.27s | 4 | roll, sugar, yeast |
| 6 | 45.63s* | 1 | yeast |
| 7 | 4.24s | 1 | yeast |
| 8 | 26.40s | 2 | flour, sandwich |
| 9 | 19.85s | 5 | flour |
| 10 | 20.92s | 4 | flour, food |
| 11 | 29.58s | 8 | flour, mixture, salt |
| 12 | 4.43s | 1 | flour, salt |

*Single narration >30s (edge case)

**Assets Generated**:
- 84 frames (42 before, 42 after) - one pair per narration
- 13 video clips - one per block
- 306 grounding masks from Hands23 detector
- Located in: `outputs/state_change_annotation/assets/P01-20240203-121517/`

## Directory Structure

```
pipelines/state_change_annotation/
├── create_annotation_tasks.py      # Main pipeline script
├── grounding_utils.py               # Grounding model utilities
├── add_grounding_to_tasks.py        # Post-processing to add masks
└── README.md                        # This file

outputs/state_change_annotation/
├── P01-20240203-121517_annotation_tasks.json  # Generated tasks
└── assets/
    └── P01-20240203-121517/
        ├── block_000/
        │   ├── *_clip.mp4            # Video clip
        │   ├── *_before_*.jpg        # Before frames
        │   ├── *_after_*.jpg         # After frames
        │   └── masks/                 # Generated masks (when grounding works)
        ├── block_001/
        └── ...
```

## Usage

### Complete Pipeline

```bash
cd pipelines/state_change_annotation

# Step 1: Create annotation tasks
python 01_create_annotation_tasks.py \
  --input ../../outputs/food_analysis/per_video_extractions/P01-20240203-121517_food_items.json \
  --video-path ../../data/HD-EPIC/Videos/P01/P01-20240203-121517.mp4 \
  --output-dir ../../outputs/state_change_annotation \
  --max-block-duration 30.0

# Step 2: Add grounding masks
conda activate hands23
python 02_add_grounding_to_tasks.py \
  --tasks-file ../../outputs/state_change_annotation/P01-20240203-121517_annotation_tasks.json \
  --output-dir ../../outputs/state_change_annotation \
  --hands23-config ../../models/hands23_detector/faster_rcnn_X_101_32x8d_FPN_3x_Hands23.yaml \
  --hands23-weights ../../models/hands23_detector/model_weights/model_hands23.pth

# Step 3: VLM state tracking
python 03_vlm_state_tracking.py \
  --tasks-file ../../outputs/state_change_annotation/P01-20240203-121517_annotation_tasks.json \
  --output-dir ../../outputs/state_change_annotation \
  --fps 1

# Step 4: View results
cd ../../outputs/state_change_annotation
python view_annotations.py
# Upload P01-20240203-121517_annotation_tasks.json in browser
```

### Re-running VLM with Updated Script

If you have old JSON with `state_before` fields, regenerate from Step 1:

```bash
# Clear old outputs
rm ../../outputs/state_change_annotation/P01-20240203-121517_annotation_tasks.json
rm -rf ../../outputs/state_change_annotation/assets/P01-20240203-121517/

# Re-run from Step 1 (creates fresh JSON without state_before)
python 01_create_annotation_tasks.py \
  --input ../../outputs/food_analysis/per_video_extractions/P01-20240203-121517_food_items.json \
  --video-path ../../data/HD-EPIC/Videos/P01/P01-20240203-121517.mp4 \
  --output-dir ../../outputs/state_change_annotation

# Continue with Steps 2-4
```

## Output Format

Each annotation task is a JSON object with this structure:

```json
{
  "task_id": "P01-20240203-121517_block_000",
  "video_id": "P01-20240203-121517",
  "block_start_time": 56.11,
  "block_end_time": 58.75,
  "narrations_in_block": [
    "Pick up the two lemon halves from the lower shelf...",
    "Put the two halves of lemon to the right side..."
  ],
  "target_food_nouns": ["lemon"],
  "assets": {
    "clip_path": "assets/.../clip.mp4",  // Video from (block_start-1s) to (block_end+1s)
    "before_frames": [
      {
        "path": "assets/.../narr00_before.jpg",
        "timestamp": 55.11,  // narration.start_time - 1.0s
        "type": "narration",
        "narration_index": 0,
        "narration_id": "P01-20240203-121517-32"
      },
      // One before frame for each narration in block (no boundary frames)
    ],
    "after_frames": [
      {
        "path": "assets/.../narr00_after.jpg",
        "timestamp": 57.25,  // narration.end_time + 1.0s
        "type": "narration",
        "narration_index": 0,
        "narration_id": "P01-20240203-121517-32"
      }
      // One after frame for each narration in block
    ]
  },
  "auto_grounding_data": {
    "lemon": {
      "before_frames": [
        {
          "method": "hands23",
          "food_noun": "lemon",
          "frame_path": "...",
          "mask_path": "assets/.../masks/..._hands23_obj0.png",
          "confidence": 0.85,
          "bbox": [x1, y1, x2, y2],
          "interaction_level": "first"  // "first" = hand-object, "second" = object-object via first
        }
      ],
      "after_frames": [...]
    }
  },
  "ground_truth_state_table": {
    "lemon": {
      "instances": {
        "lemon_on_shelf_001": {
          "vlm_state_after": { ... },      // VLM prediction at end of block
          "vlm_reasoning": "...",          // Based on narrations + video
          "semantic_name": "on_shelf",     // Human-readable instance descriptor
          "block_time_range": {...},
          "annotator_state_after": null    // Human verification
        }
      }
    },
    "flour": {
      "instances": {
        "flour_from_bag_001": {
          "vlm_state_after": {
            "container_state": {"container_type": "original_packaging"},
            "quantity": "partial"
          },
          "vlm_reasoning": "Flour poured from bag, some remains",
          "semantic_name": "from_bag",
          "block_time_range": {...}
        },
        "flour_in_bowl_002": {
          "vlm_state_after": {
            "container_state": {"container_type": "bowl"},
            "quantity": "partial"
          },
          "vlm_reasoning": "Flour transferred to bowl for mixing",
          "semantic_name": "in_bowl",
          "block_time_range": {...}
        }
      }
    }
  }
  // Note: Multiple instances per food noun supported for splits
}
```

## Next Steps

1. **Human Annotation Interface**
   - Add annotation UI to visualization
   - Allow annotators to validate/correct VLM predictions
   - Save annotator_state_after to JSON

2. **Batch Processing**
   - Process multiple P01 videos
   - Automated pipeline for all videos
   - Result aggregation and statistics

3. **GroundingSAM Integration**
   - Text-prompted grounding as alternative to Hands23
   - Compare grounding quality

## Dependencies

- **Core**: Python 3.10+, OpenCV, NumPy
- **Video Processing**: ffmpeg
- **Hands23**: detectron2, torch (CUDA recommended)
- **GroundingSAM**: (to be added)
- **VLM**: Qwen3-VL, GPT-4o API access

## Known Issues

1. ~~Hands23 detector configuration~~ ✅ FIXED
2. GroundingSAM not yet integrated
3. VLM pre-population not implemented
4. Hands23 detects hand-interacted objects (generic), not specific food types - consider GroundingSAM for food-specific grounding

## Technical Notes

### Hands23 Detector Configuration

The Hands23 detector requires specific configuration parameters that must be set before creating the predictor:

```python
cfg.HAND = 0.5        # Hand detection threshold
cfg.FIRSTOBJ = 0.3    # First object threshold
cfg.SECONDOBJ = 0.2   # Second object threshold
cfg.HAND_RELA = 0.3   # Hand relationship threshold
cfg.OBJ_RELA = 0.5    # Object relationship threshold
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = min(cfg.HAND, cfg.FIRSTOBJ, cfg.SECONDOBJ)
```

**Class Mapping**:
- Class 0: Hand
- Class 1: First-level object (directly touched/held by hand)
- Class 2: Second-level object (interacted with via first object)

**Interaction Level Examples**:
- **First-level**: Hand holding knife, hand grasping onion, hand touching flour
- **Second-level**: Onion being cut by knife, dough being rolled by pin

**Key Findings**:
- Lower thresholds (0.3-0.5) needed to capture food objects in egocentric cooking videos
- Hands23 detects objects based on hand-interaction, not semantic food categories
- Average detection: 2-4 objects per frame with hands present

## Configuration Notes

**Merging Threshold**: 10 seconds works well for bread-making video
- Captures related actions (e.g., multiple flour measurements)
- Prevents over-merging of distinct interactions
- Can be adjusted per video type

**Frame Offset**: 1 second provides good temporal context
- Captures pre/post state clearly
- Not too far from actual interaction
- Allows for motion blur recovery

## Performance

**Test Video** (P01-20240203-121517, 459 seconds):
- Processing time: ~2 minutes
- 42 narrations → 11 blocks (73% reduction)
- 106 frames extracted at 1408x1408 resolution
- Total asset size: ~150MB (frames + clips)

## Contact

For questions or issues with this pipeline, refer to the main HDEPIC project documentation.

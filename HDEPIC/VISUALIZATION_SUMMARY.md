# Hand Detection Pipeline - Visualization Summary

## Complete Pipeline with Official Hands23 Visualization

The pipeline now uses the official Hands23 visualization utilities (`vis_utils.py`) which provide professional-quality annotations with segmentation masks.

## Pipeline Components

### 1. Frame Extraction
**Script:** `extract_frames.py`
- Extracts frames at specified FPS with timestamp tracking
- Creates metadata JSON for frame-to-video mapping

### 2. Detection with Mask Output
**Script:** `run_hands_detector_with_masks.py` (NEW)
- Runs Hands23 detector and saves segmentation masks
- Outputs structured JSON with detection results
- Saves masks in format compatible with vis_utils

**Key Features:**
- Saves hand masks (class 2)
- Saves first object masks (class 3)
- Saves second object masks (class 5)
- Preserves all detection attributes (grasp, contact state, touch type)

### 3. Official Visualization
**Script:** `visualize_with_official_utils.py` (NEW)
- Uses official Hands23 `vis_utils.py` for rendering
- Processes ALL frames (including those without detections)
- Applies segmentation masks with semi-transparency
- Professional color scheme and labeling

**Visualization Features:**
- **Segmentation Masks**: Semi-transparent overlays showing exact hand/object regions
- **Color Coding**:
  - Blue hands (left hand darker, right hand lighter)
  - Yellow/orange for first objects
  - Green/teal for second objects
- **Labels**:
  - Contact state (e.g., "ObjC" for object contact)
  - Grasp type (e.g., "Pre-Pris", "NP-Fin")
  - Touch type (e.g., "Cont:held", "Neither:touched")
  - Hand side ("L" or "R")
- **Connection Lines**: Blue lines connecting hands to objects, with colored dots at centers
- **Bounding Boxes**: Thick colored borders matching the mask colors

### 4. Evaluation
**Script:** `evaluate_detections.py`
- Compares detections against narration annotations
- Generates coverage and precision metrics

## Running the Complete Pipeline

### Using Master Script
```bash
chmod +x run_pipeline_with_official_vis.sh
./run_pipeline_with_official_vis.sh
```

### Individual Steps

#### Step 1: Extract Frames
```bash
python extract_frames.py \
    --video_path "HD-EPIC/Videos/P01/P01-20240202-110250.mp4" \
    --output_dir "video_frames/P01-20240202-110250" \
    --fps 1.0 \
    --video_id "P01-20240202-110250"
```

#### Step 2: Run Detector with Masks
```bash
python run_hands_detector_with_masks.py \
    --frames_dir "video_frames/P01-20240202-110250" \
    --output_file "results/P01-20240202-110250_detections.json" \
    --masks_dir "results/masks/P01-20240202-110250" \
    --config_file "hands23_detector/faster_rcnn_X_101_32x8d_FPN_3x_Hands23.yaml" \
    --model_weights "hands23_detector/model_weights/model_hands23.pth"
```

#### Step 3: Evaluate Results
```bash
python evaluate_detections.py \
    --detection_file "results/P01-20240202-110250_detections.json" \
    --narration_csv "participant_P01_narrations.csv" \
    --video_id "P01-20240202-110250" \
    --output_file "results/P01-20240202-110250_evaluation.txt"
```

#### Step 4: Generate Official Visualizations
```bash
python visualize_with_official_utils.py \
    --detection_file "results/P01-20240202-110250_detections.json" \
    --frames_dir "video_frames/P01-20240202-110250" \
    --masks_dir "results/masks/P01-20240202-110250" \
    --output_dir "results/visualizations_official/P01-20240202-110250"
```

## Output Structure

```
results/
├── P01-20240202-110250_detections.json      # Detection results
├── P01-20240202-110250_evaluation.txt       # Evaluation report
├── P01-20240202-110250_evaluation.json      # Evaluation data
├── masks/
│   └── P01-20240202-110250/                 # Segmentation masks
│       ├── 2_0_*.jpg                        # Hand masks
│       ├── 3_0_*.jpg                        # First object masks
│       └── 5_0_*.jpg                        # Second object masks
└── visualizations_official/
    └── P01-20240202-110250/                 # 397 annotated images
        └── *.jpg                            # All frames visualized
```

## Results for P01-20240202-110250

### Detection Statistics
- Total frames: 397
- Frames with hands: 367 (92.4%)
- Frames with interactions: 347 (87.4%)

### Visualization Statistics
- Total visualizations: 397 (100% of frames)
- Frames with detections: 367
- Frames without detections: 30 (copied as-is)

### Evaluation Statistics
- Total narrations: 224
- Narrations with detections: 161 (71.9%)
- Narrations without detections: 63 (28.1%)
- Detections outside narration ranges: 62

## Visualization Examples

### Example 1: Single Hand-Object Interaction
**Frame:** `P01-20240202-110250_frame_00240_ts_8.00.jpg`
- Left hand interacting with cupboard
- Yellow/orange mask on object (cupboard)
- Blue mask on hand
- Label: "ObjC,NP-Fin" (Object contact, Non-Prehensile Finger grasp)
- Object label: "Neither:touched"
- Connection line from hand to object

### Example 2: Two-Handed Interaction with Multiple Objects
**Frame:** `P01-20240202-110250_frame_01020_ts_34.00.jpg`
- Both hands detected
- Multiple objects (containers)
- Left hand: "ObjC,Pre-Pris" (Precision prismatic grasp)
- Right hand: "ObjC,NP-Fin" (Non-prehensile finger)
- Objects labeled: "Cont:held" (Container held)
- Connection lines showing hand-object relationships

### Example 3: Frame Without Detection
**Frame:** `P01-20240202-110250_frame_00060_ts_2.00.jpg`
- No hands visible in frame
- Original frame copied without annotations
- Maintains consistency across all frames

## Key Advantages of Official Visualization

1. **Professional Quality**: Uses the same visualization style as the Hands23 paper
2. **Segmentation Masks**: Shows precise hand/object regions, not just boxes
3. **Rich Information**: Displays grasp types, contact states, and touch types
4. **Publication Ready**: High-quality visualizations suitable for papers/presentations
5. **Complete Coverage**: All frames included, making it easy to create videos
6. **Color Consistency**: Standard color scheme from the official implementation

## Creating Video from Visualizations

```bash
# Using ffmpeg to create video at 1 FPS
ffmpeg -framerate 1 -pattern_type glob -i 'results/visualizations_official/P01-20240202-110250/*.jpg' \
    -c:v libx264 -pix_fmt yuv420p -crf 20 \
    results/P01-20240202-110250_visualization.mp4
```

## Comparison: Custom vs Official Visualization

### Previous Custom Visualization
- Simple bounding boxes with OpenCV
- Basic labels with text
- No segmentation masks
- Custom color scheme
- Only frames with interactions

### Current Official Visualization
- Segmentation masks with transparency
- Professional layout and typography
- Official Hands23 color scheme
- Rich interaction details
- **All frames included** (with and without detections)

## Font Customization

The official visualization uses `hands23_detector/utils/times_b.ttf`. To use a different font:

```bash
python visualize_with_official_utils.py \
    --detection_file "..." \
    --frames_dir "..." \
    --masks_dir "..." \
    --output_dir "..." \
    --font_path "/path/to/your/font.ttf"
```

## Simple Mode

For presentations without detailed labels:

```bash
python visualize_with_official_utils.py \
    --detection_file "..." \
    --frames_dir "..." \
    --masks_dir "..." \
    --output_dir "..." \
    --use_simple
```

This removes the text labels while keeping masks and bounding boxes.

## Notes

- Mask files are named with pattern: `{class_id}_{hand_id}_{filename}`
  - Class 2: Hand
  - Class 3: First object
  - Class 5: Second object
- Each hand has a unique ID (0, 1, 2, ...)
- Masks are saved as BGR images for compatibility with vis_utils
- The visualization script handles missing masks gracefully

## Performance

- Detection: ~8 frames/second on GPU
- Mask saving: Included in detection time
- Visualization: ~5 frames/second
- Total pipeline: ~50 seconds for 397 frames

## Troubleshooting

### Issue: "Font file not found"
- Check that `hands23_detector/utils/times_b.ttf` exists
- Specify custom font with `--font_path`

### Issue: "Mask not exist"
- Ensure detector was run with `run_hands_detector_with_masks.py` (not the original script)
- Check that `--masks_dir` matches the directory used during detection

### Issue: Segmentation masks don't align with boxes
- This is expected if the model's mask prediction is imperfect
- The official visualization shows exactly what the model predicted

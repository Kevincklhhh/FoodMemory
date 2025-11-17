# Hand-Object Interaction Detection Pipeline

This pipeline extracts frames from egocentric videos, detects hand-object interactions using the Hands23 detector with segmentation masks, evaluates results against narration annotations, and generates professional visualizations.

## Pipeline Overview

```
Video (MP4) → Frame Extraction → Hand Detection + Masks → Evaluation + Visualization → Results
```

## Pipeline Components

### 1. Frame Extraction (`1_extract_frames.py`)
Extracts video frames at a specified FPS with timestamp tracking.

**Features:**
- Configurable FPS (default: 1.0)
- Timestamp encoding in filenames
- Metadata JSON with frame information
- Allows tracing back to original video timestamp

**Output:**
- `../../outputs/hand_detection/frames/{VIDEO_ID}/` - Directory with extracted frames
- `frame_metadata.json` - Frame timing metadata

### 2. Hand-Object Interaction Detection (`2_detect_hands_with_masks.py`)
Runs the Hands23 detector on extracted frames to detect hands and their interactions with objects, including segmentation masks.

**Detection Capabilities:**
- Hand detection (left/right classification)
- Hand-object interactions (3-level hierarchy: hand → first object → second object)
- Grasp classification (8 types: NP-Palm, NP-Fin, Pow-Pris, Pre-Pris, Pow-Circ, Pre-Circ, Later, Other)
- Touch type classification (7 types: tool touched/held/used, container touched/held, neither touched/held)
- Contact state (5 types: no_contact, self_contact, object_contact, other_person_contact, obj_to_obj_contact)
- **Segmentation masks** for hands and objects

**Configurable Thresholds:**
- `--hand_thresh` (default: 0.7) - Hand detection confidence
- `--first_obj_thresh` (default: 0.5) - First object detection confidence
- `--second_obj_thresh` (default: 0.3) - Second object detection confidence
- `--hand_rela` (default: 0.3) - Hand-object interaction threshold
- `--obj_rela` (default: 0.7) - Object-object interaction threshold

**Output:**
- `../../outputs/hand_detection/detections/{VIDEO_ID}_detections.json` - Structured detection results
- `../../outputs/hand_detection/masks/{VIDEO_ID}/` - Segmentation masks (class 2: hands, class 3: first object, class 5: second object)

### 3. Evaluation (`3_evaluate_detections.py`)
Compares detected hand-object interactions against narration annotations.

**Evaluation Metrics:**
- **Coverage**: Percentage of narration time ranges that have at least one detection
- **Precision**: Number of detections outside narration time ranges
- **Temporal Alignment**: Checks if each narration's start-end time has corresponding detections

**Output:**
- `../../outputs/hand_detection/evaluations/{VIDEO_ID}_evaluation.txt` - Detailed human-readable report
- `../../outputs/hand_detection/evaluations/{VIDEO_ID}_evaluation.json` - Structured evaluation results

### 4. Official Visualization (`4_visualize_official.py`)
Generates professional-quality annotated images using official Hands23 visualization utilities.

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
- **Complete Coverage**: All frames included (with and without detections)

**Output:**
- `../../outputs/hand_detection/visualizations/{VIDEO_ID}/` - Annotated images for all frames

## Running the Pipeline

### Quick Start
```bash
# Run complete pipeline on a video
./run_pipeline_with_official_vis.sh
```

The master script runs all 4 steps sequentially with proper error handling.

### Individual Steps

#### Step 1: Extract Frames
```bash
python 1_extract_frames.py \
    --video_path "../../data/HD-EPIC/Videos/P01/P01-20240202-110250.mp4" \
    --output_dir "../../outputs/hand_detection/frames/P01-20240202-110250" \
    --fps 1.0 \
    --video_id "P01-20240202-110250"
```

#### Step 2: Run Hand Detector with Masks
```bash
python 2_detect_hands_with_masks.py \
    --frames_dir "../../outputs/hand_detection/frames/P01-20240202-110250" \
    --output_file "../../outputs/hand_detection/detections/P01-20240202-110250_detections.json" \
    --masks_dir "../../outputs/hand_detection/masks/P01-20240202-110250" \
    --config_file "../../models/hands23_detector/faster_rcnn_X_101_32x8d_FPN_3x_Hands23.yaml" \
    --model_weights "../../models/hands23_detector/model_weights/model_hands23.pth"
```

#### Step 3: Evaluate Results
```bash
python 3_evaluate_detections.py \
    --detection_file "../../outputs/hand_detection/detections/P01-20240202-110250_detections.json" \
    --narration_csv "../../participant_P01_narrations.csv" \
    --video_id "P01-20240202-110250" \
    --output_file "../../outputs/hand_detection/evaluations/P01-20240202-110250_evaluation.txt"
```

#### Step 4: Generate Official Visualizations
```bash
python 4_visualize_official.py \
    --detection_file "../../outputs/hand_detection/detections/P01-20240202-110250_detections.json" \
    --frames_dir "../../outputs/hand_detection/frames/P01-20240202-110250" \
    --masks_dir "../../outputs/hand_detection/masks/P01-20240202-110250" \
    --output_dir "../../outputs/hand_detection/visualizations/P01-20240202-110250"
```

## Output Structure

```
outputs/hand_detection/
├── frames/
│   └── P01-20240202-110250/
│       ├── P01-20240202-110250_frame_XXXXX_ts_XX.XX.jpg
│       └── frame_metadata.json
├── detections/
│   └── P01-20240202-110250_detections.json
├── masks/
│   └── P01-20240202-110250/
│       ├── 2_0_*.jpg  # Hand masks
│       ├── 3_0_*.jpg  # First object masks
│       └── 5_0_*.jpg  # Second object masks
├── evaluations/
│   ├── P01-20240202-110250_evaluation.txt
│   └── P01-20240202-110250_evaluation.json
└── visualizations/
    └── P01-20240202-110250/
        └── *.jpg  # All frames with professional annotations
```

## Example Results (P01-20240202-110250)

**Detection Statistics:**
- Total frames: 397
- Frames with hands: 367 (92.4%)
- Frames with interactions: 347 (87.4%)

**Evaluation Statistics:**
- Total narrations: 224
- Narrations with detections: 161 (71.9%)
- Narrations without detections: 63 (28.1%)
- Detections outside narration ranges: 62
- Video duration: 394.72s
- Narration coverage: 78.6% of video

## Understanding the Results

### High Coverage (71.9%)
The detector successfully identifies hand-object interactions for most narrated events, showing good alignment between automated detection and manual annotations.

### Missed Detections (28.1%)
Most uncovered narrations are very short (0.2-1.4 seconds). These brief interactions may:
- Fall between sampling frames (current: 1 FPS)
- Involve quick motions that are hard to detect
- Be ambiguous or subtle interactions

**Recommendation**: Try higher FPS (2-5 FPS) for shorter interactions.

### Extra Detections (62 frames)
Detections outside narration ranges could indicate:
- False positives from the detector
- Hand-object interactions not captured in narrations
- Timing misalignment between video and annotations
- Preparatory or follow-up actions not explicitly narrated

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
ffmpeg -framerate 1 -pattern_type glob -i '../../outputs/hand_detection/visualizations/P01-20240202-110250/*.jpg' \
    -c:v libx264 -pix_fmt yuv420p -crf 20 \
    ../../outputs/hand_detection/P01-20240202-110250_visualization.mp4
```

## Environment

**Required:**
- Conda environment: `hands23` (Python 3.10)
- PyTorch 2.0.0 with CUDA 11.8
- Detectron2
- OpenCV, pandas, tqdm, pillow

**Installation:**
See `../../models/hands23_detector/README.md` for detailed setup instructions.

## Font Customization

The official visualization uses `../../models/hands23_detector/utils/times_b.ttf`. To use a different font:

```bash
python 4_visualize_official.py \
    --detection_file "..." \
    --frames_dir "..." \
    --masks_dir "..." \
    --output_dir "..." \
    --font_path "/path/to/your/font.ttf"
```

## Simple Mode

For presentations without detailed labels:

```bash
python 4_visualize_official.py \
    --detection_file "..." \
    --frames_dir "..." \
    --masks_dir "..." \
    --output_dir "..." \
    --use_simple
```

This removes the text labels while keeping masks and bounding boxes.

## Performance

- Detection: ~8 frames/second on GPU
- Mask saving: Included in detection time
- Visualization: ~5 frames/second
- Total pipeline: ~50 seconds for 397 frames

## Troubleshooting

### NumPy Version Issues
If you get "NumPy 2.x incompatibility" errors:
```bash
pip install "numpy<2"
pip install "opencv-python<4.10"
```

### CUDA Version Mismatch
The detector requires PyTorch compiled with CUDA 11.8. If you have CUDA 12.0, the installation script includes workarounds.

### Missing Pandas
```bash
conda activate hands23
pip install pandas
```

### Issue: "Font file not found"
- Check that `../../models/hands23_detector/utils/times_b.ttf` exists
- Specify custom font with `--font_path`

### Issue: "Mask not exist"
- Ensure detector was run with `2_detect_hands_with_masks.py`
- Check that `--masks_dir` matches the directory used during detection

### Issue: Segmentation masks don't align with boxes
- This is expected if the model's mask prediction is imperfect
- The official visualization shows exactly what the model predicted

## References

- **Hands23 Detector**: https://github.com/ddshan/hand_object_detector
- **Model Paper**: [Detecting Hands and Recognizing Physical Contact in the Wild](https://openaccess.thecvf.com/content/CVPR2023/papers/Shan_Detecting_Hands_and_Recognizing_Physical_Contact_in_the_Wild_CVPR_2023_paper.pdf)

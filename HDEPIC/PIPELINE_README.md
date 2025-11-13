# Hand-Object Interaction Detection Pipeline

This pipeline extracts frames from egocentric videos, detects hand-object interactions using the Hands23 detector, evaluates results against narration annotations, and generates visualizations.

## Pipeline Overview

```
Video (MP4) → Frame Extraction → Hand Detection → Evaluation + Visualization → Results
```

## Components

### 1. Frame Extraction (`extract_frames.py`)
Extracts video frames at a specified FPS with timestamp tracking.

**Features:**
- Configurable FPS (default: 1.0)
- Timestamp encoding in filenames
- Metadata JSON with frame information
- Allows tracing back to original video timestamp

**Output:**
- `video_frames/{VIDEO_ID}/` - Directory with extracted frames
- `video_frames/{VIDEO_ID}/frame_metadata.json` - Frame timing metadata

### 2. Hand-Object Interaction Detection (`run_hands_detector.py`)
Runs the Hands23 detector on extracted frames to detect hands and their interactions with objects.

**Detection Capabilities:**
- Hand detection (left/right classification)
- Hand-object interactions (3-level hierarchy: hand → first object → second object)
- Grasp classification (8 types: NP-Palm, NP-Fin, Pow-Pris, Pre-Pris, Pow-Circ, Pre-Circ, Later, Other)
- Touch type classification (7 types: tool touched/held/used, container touched/held, neither touched/held)
- Contact state (5 types: no_contact, self_contact, object_contact, other_person_contact, obj_to_obj_contact)

**Configurable Thresholds:**
- `--hand_thresh` (default: 0.7) - Hand detection confidence
- `--first_obj_thresh` (default: 0.5) - First object detection confidence
- `--second_obj_thresh` (default: 0.3) - Second object detection confidence
- `--hand_rela` (default: 0.3) - Hand-object interaction threshold
- `--obj_rela` (default: 0.7) - Object-object interaction threshold

**Output:**
- `results/{VIDEO_ID}_detections.json` - Structured detection results with timestamps

### 3. Evaluation (`evaluate_detections.py`)
Compares detected hand-object interactions against narration annotations.

**Evaluation Metrics:**
- **Coverage**: Percentage of narration time ranges that have at least one detection
- **Precision**: Number of detections outside narration time ranges
- **Temporal Alignment**: Checks if each narration's start-end time has corresponding detections

**Output:**
- `results/{VIDEO_ID}_evaluation.txt` - Detailed human-readable report
- `results/{VIDEO_ID}_evaluation.json` - Structured evaluation results

### 4. Visualization (`visualize_detections.py`)
Generates annotated images showing detected hands, objects, and interactions.

**Visualization Features:**
- Color-coded bounding boxes:
  - Green: Hands
  - Blue: First objects
  - Orange: Second objects
- Connection lines showing interaction relationships
- Labels with grasp type, contact state, and touch type
- Timestamp overlay
- Can generate for all frames or only frames with interactions

**Output:**
- `results/visualizations/{VIDEO_ID}/` - Annotated images for each frame

## Running the Pipeline

### Quick Start
```bash
# Run complete pipeline on a video
./run_pipeline.sh
```

The master script runs all 4 steps sequentially with proper error handling.

### Individual Steps

#### Step 1: Extract Frames
```bash
python extract_frames.py \
    --video_path "HD-EPIC/Videos/P01/P01-20240202-110250.mp4" \
    --output_dir "video_frames/P01-20240202-110250" \
    --fps 1.0 \
    --video_id "P01-20240202-110250"
```

#### Step 2: Run Hand Detector
```bash
python run_hands_detector.py \
    --frames_dir "video_frames/P01-20240202-110250" \
    --output_file "results/P01-20240202-110250_detections.json" \
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

#### Step 4: Generate Visualizations
```bash
# Only frames with interactions
python visualize_detections.py \
    --detection_file "results/P01-20240202-110250_detections.json" \
    --frames_dir "video_frames/P01-20240202-110250" \
    --output_dir "results/visualizations/P01-20240202-110250" \
    --only_interactions

# All frames
python visualize_detections.py \
    --detection_file "results/P01-20240202-110250_detections.json" \
    --frames_dir "video_frames/P01-20240202-110250" \
    --output_dir "results/visualizations/P01-20240202-110250"
```

## Configuration

Edit `run_pipeline.sh` to configure:
- `VIDEO_ID` - Video identifier
- `VIDEO_PATH` - Path to input video
- `FPS` - Frame extraction rate
- `NARRATION_CSV` - Path to narration annotations

## Output Structure

```
/home/kailaic/NeuroTrace/kitchen/HDEPIC/
├── video_frames/
│   └── {VIDEO_ID}/
│       ├── {VIDEO_ID}_frame_XXXXX_ts_XX.XX.jpg  # Extracted frames
│       └── frame_metadata.json                  # Frame timing metadata
├── results/
│   ├── {VIDEO_ID}_detections.json               # Detection results
│   ├── {VIDEO_ID}_evaluation.txt                # Text evaluation report
│   ├── {VIDEO_ID}_evaluation.json               # JSON evaluation report
│   └── visualizations/
│       └── {VIDEO_ID}/
│           └── {VIDEO_ID}_frame_XXXXX_ts_XX.XX.jpg  # Annotated frames
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

## Visualization Legend

**Bounding Box Colors:**
- **Green**: Hand (with side label: left_hand/right_hand)
- **Blue**: First object being manipulated
- **Orange**: Second object (if present)

**Connection Lines:**
- Hand → Object: Shows hand-object interaction
- Object → Second Object: Shows object-object relationship

**Labels:**
- Contact state: Describes what the hand is touching
- Grasp type: Describes how the object is held
- Touch type: Describes the nature of object manipulation

## Environment

**Required:**
- Conda environment: `hands23` (Python 3.10)
- PyTorch 2.0.0 with CUDA 11.8
- Detectron2
- OpenCV, pandas, tqdm, pillow

**Installation:**
See `hands23_detector/README.md` for detailed setup instructions.

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

## References

- **Hands23 Detector**: https://github.com/ddshan/hand_object_detector
- **Model Paper**: [Detecting Hands and Recognizing Physical Contact in the Wild](https://openaccess.thecvf.com/content/CVPR2023/papers/Shan_Detecting_Hands_and_Recognizing_Physical_Contact_in_the_Wild_CVPR_2023_paper.pdf)

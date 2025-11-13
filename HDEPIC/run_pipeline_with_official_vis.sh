#!/bin/bash
# Master script to run the complete hand detection pipeline with official visualization

set -e  # Exit on error

# Configuration
VIDEO_ID="P01-20240202-110250"
VIDEO_PATH="/home/kailaic/NeuroTrace/kitchen/HDEPIC/HD-EPIC/Videos/P01/${VIDEO_ID}.mp4"
FRAMES_DIR="/home/kailaic/NeuroTrace/kitchen/HDEPIC/video_frames/${VIDEO_ID}"
MASKS_DIR="/home/kailaic/NeuroTrace/kitchen/HDEPIC/results/masks/${VIDEO_ID}"
DETECTION_FILE="/home/kailaic/NeuroTrace/kitchen/HDEPIC/results/${VIDEO_ID}_detections.json"
NARRATION_CSV="/home/kailaic/NeuroTrace/kitchen/HDEPIC/participant_P01_narrations.csv"
EVAL_REPORT="/home/kailaic/NeuroTrace/kitchen/HDEPIC/results/${VIDEO_ID}_evaluation.txt"
VIS_DIR="/home/kailaic/NeuroTrace/kitchen/HDEPIC/results/visualizations_official/${VIDEO_ID}"

# FPS for frame extraction
FPS=1.0

# Activate conda environment
source ~/.bashrc
conda activate hands23

echo "========================================="
echo "Hand Detection Pipeline (with Official Visualization)"
echo "========================================="
echo "Video ID: ${VIDEO_ID}"
echo "FPS: ${FPS}"
echo ""

# Step 1: Extract frames
echo "Step 1: Extracting frames from video..."
python extract_frames.py \
    --video_path "${VIDEO_PATH}" \
    --output_dir "${FRAMES_DIR}" \
    --fps ${FPS} \
    --video_id "${VIDEO_ID}"

echo ""
echo "Step 2: Running hand-object interaction detector with mask output..."
python run_hands_detector_with_masks.py \
    --frames_dir "${FRAMES_DIR}" \
    --output_file "${DETECTION_FILE}" \
    --masks_dir "${MASKS_DIR}" \
    --config_file "hands23_detector/faster_rcnn_X_101_32x8d_FPN_3x_Hands23.yaml" \
    --model_weights "hands23_detector/model_weights/model_hands23.pth"

echo ""
echo "Step 3: Evaluating detections against narrations..."
python evaluate_detections.py \
    --detection_file "${DETECTION_FILE}" \
    --narration_csv "${NARRATION_CSV}" \
    --video_id "${VIDEO_ID}" \
    --output_file "${EVAL_REPORT}"

echo ""
echo "Step 4: Generating visualizations with official Hands23 vis_utils..."
python visualize_with_official_utils.py \
    --detection_file "${DETECTION_FILE}" \
    --frames_dir "${FRAMES_DIR}" \
    --masks_dir "${MASKS_DIR}" \
    --output_dir "${VIS_DIR}"

echo ""
echo "========================================="
echo "Pipeline Complete!"
echo "========================================="
echo "Results saved to:"
echo "  - Frames: ${FRAMES_DIR}"
echo "  - Masks: ${MASKS_DIR}"
echo "  - Detections: ${DETECTION_FILE}"
echo "  - Evaluation: ${EVAL_REPORT}"
echo "  - Visualizations: ${VIS_DIR}"
echo "========================================="

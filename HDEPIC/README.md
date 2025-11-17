# HDEPIC Analysis Suite

Comprehensive analysis tools for the HD-EPIC egocentric video dataset, including food item analysis and hand-object interaction detection.

## Overview

This repository contains two main analysis pipelines and various utility tools for working with HD-EPIC dataset:

1. **Food Analysis Pipeline** - LLM-based classification and abundance analysis of food items
2. **Hand-Object Interaction Detection** - Automated detection, evaluation, and visualization of hand-object interactions

## Directory Structure

```
HDEPIC/
├── README.md                      # This file
│
├── pipelines/                     # Main analysis workflows
│   ├── food_analysis/             # Food item classification & analysis (4 scripts)
│   └── hand_detection/            # Hand-object interaction detection (4 scripts)
│
├── tools/                         # Standalone utilities
│   ├── data_extraction/           # Extract objects, frames, masks, participants
│   ├── classification/            # LLM-based object classification
│   ├── preprocessing/             # Data cleaning and deduplication
│   └── testing/                   # API testing and development
│
├── models/                        # Model dependencies
│   └── hands23_detector/          # Hands23 detection model
│
├── data/                          # Input datasets
│   ├── HD-EPIC/                   # HD-EPIC video dataset
│   └── hd-epic-annotations/       # Annotation files
│
└── outputs/                       # Generated results
    ├── food_analysis/             # Food analysis outputs
    ├── hand_detection/            # Hand detection outputs
    └── extracted_data/            # Utility extraction outputs
```

## Quick Start

### Food Analysis Pipeline

Analyzes food items in P01 participant videos using LLM classification:

```bash
cd pipelines/food_analysis
python 1_classify_hdepic_food_nouns.py    # Classify 303 noun classes
python 2_extract_hdepic_food_items.py     # Extract food from narrations
python 3_analyze_hdepic_food_per_video.py # Per-video analysis
python 4_analyze_hdepic_food_abundance.py # Abundance statistics
```

**Results**: Identified 152 food nouns, 3,570 food occurrences across 27 videos.

See [pipelines/food_analysis/README.md](pipelines/food_analysis/README.md) for details.

### Hand-Object Interaction Detection

Detects and visualizes hand-object interactions with segmentation masks:

```bash
cd pipelines/hand_detection
./run_pipeline_with_official_vis.sh
```

Or run individual steps:
```bash
python 1_extract_frames.py           # Extract video frames
python 2_detect_hands_with_masks.py  # Detect hands + generate masks
python 3_evaluate_detections.py      # Evaluate against annotations
python 4_visualize_official.py       # Generate professional visualizations
```

**Results**: 92.4% of frames with hands detected, 71.9% narration coverage.

See [pipelines/hand_detection/README.md](pipelines/hand_detection/README.md) for details.

## Utilities

### Data Extraction Tools (`tools/data_extraction/`)

- **`list_p01_objects.py`** - List all objects from P01 videos with tracking details
- **`extract_participant.py`** - Extract participant-specific data from annotations
- **`extract_food_frames.py`** - Extract full frames for food items
- **`extract_mask_images.py`** - Extract bounding box ROI images

### Classification Tools (`tools/classification/`)

- **`classify_food_objects.py`** - LLM-based food object classification (older version)

### Preprocessing Tools (`tools/preprocessing/`)

- **`deduplicate_food_items.py`** - Semantic deduplication of food names

### Testing Tools (`tools/testing/`)

- **`test_qwen3vl_video.py`** - Test Qwen3-VL video analysis API

## Key Features

### Food Analysis
- LLM-based classification with structured prompts
- Temporal tracking of food occurrences
- Per-video and cross-video abundance analysis
- CSV/JSON/TXT output formats

### Hand Detection
- Official Hands23 detector with segmentation masks
- 8 grasp types, 7 touch types, 5 contact states
- Evaluation against manual narrations
- Professional publication-ready visualizations
- Complete pipeline automation

## Environment Setup

### Food Analysis
```bash
pip install requests
# Standard library: json, csv, pathlib, collections, ast
```

### Hand Detection
```bash
conda create -n hands23 python=3.10
conda activate hands23
# See models/hands23_detector/README.md for full installation
```

## Dataset

This suite works with the **HD-EPIC** egocentric video dataset:
- High-definition first-person kitchen videos
- Detailed narrations with timestamps
- Object tracking and segmentation annotations
- Participant P01 data included

Dataset location: `data/HD-EPIC/` and `data/hd-epic-annotations/`

## Outputs

All generated files are organized in `outputs/`:

```
outputs/
├── food_analysis/           # Food classification and abundance data
├── hand_detection/          # Frames, detections, masks, visualizations
└── extracted_data/          # Object lists, frames, ROIs
```

## Citation

If you use the HD-EPIC dataset:
```
[HD-EPIC citation - to be added]
```

If you use the Hands23 detector:
```
@inproceedings{shan2023detecting,
  title={Detecting Hands and Recognizing Physical Contact in the Wild},
  author={Shan, Dandan and Geng, Jiaqi and Shu, Michelle and Fouhey, David F},
  booktitle={CVPR},
  year={2023}
}
```

## Contributing

Each pipeline and tool has its own README with detailed usage instructions. Please refer to:
- [Food Analysis Pipeline](pipelines/food_analysis/README.md)
- [Hand Detection Pipeline](pipelines/hand_detection/README.md)

## Support

For issues or questions:
- Check individual README files in each directory
- Review troubleshooting sections in pipeline READMEs
- Ensure environment setup is correct

## License

[License information - to be added]

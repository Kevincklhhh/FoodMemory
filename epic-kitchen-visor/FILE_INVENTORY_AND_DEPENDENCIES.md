# EPIC Kitchen VISOR - Complete File Inventory & Dependency Analysis

Generated: 2025-11-12
Working Directory: `/home/kailaic/NeuroTrace/kitchen/epic-kitchen-visor/`

---

## Executive Summary

The epic-kitchen-visor directory contains a **2-stage pipeline for food item analysis from VISOR annotations**:

1. **Stage 1: VISOR Food Extraction** (Steps 1-5)
   - Extract food items from VISOR sparse annotations (158 videos, 32,645 food occurrences)
   - Generate searchable indices and analysis reports
   - Build an image inventory for food retrieval

2. **Stage 2: Benchmark Creation** (Supporting tools)
   - Generate synthetic household food inventory
   - Create benchmark datasets for food instance retrieval

The system is **fully independent** from the main NeuroTrace pipeline (narration extraction, VLM analysis, evaluation).

---

## 1. PYTHON SCRIPTS (Numbered Pipeline Steps)

### Core Pipeline Scripts (Execute in order)

#### ✅ **1_extract_food_items.py** (STEP 1)
- **Purpose**: Extract all food items from VISOR sparse annotations
- **Input**: 
  - `GroundTruth-SparseAnnotations/annotations/` (JSON annotation files)
  - `EPIC_100_noun_classes_v2.csv` (noun taxonomy)
  - `epic_food_nouns_detailed.json` (food class mappings)
- **Output**: `visor_food_items.json` (404 MB)
- **Function**: Processes VISOR annotation files to identify food occurrences with frame-level precision
- **Dependencies**: json, csv, pathlib (standard library only)
- **Status**: ✅ Core step - REQUIRED

#### ✅ **2_create_food_segments.py** (STEP 2 - OPTIONAL)
- **Purpose**: Create visual segmentation masks for extracted food items
- **Input**: 
  - `visor_food_items.json` (from Step 1)
  - `GroundTruth-SparseAnnotations/rgb_frames/` (frame images)
- **Output**: `food_segments/` directory with masked images
- **Function**: Generates visual overlays showing food regions
- **Dependencies**: json, cv2, numpy, pathlib, collections
- **Status**: ✅ Optional visualization step

#### ✅ **3_analyze_food_per_video.py** (STEP 3)
- **Purpose**: Generate analysis reports per video
- **Input**: `visor_food_items.json` (from Step 1)
- **Output**: 
  - `food_per_video.json` (2.8 MB) - detailed structured data
  - `food_per_video.csv` - tabular format
  - `food_per_video.txt` - human-readable summary
  - `food_per_video_simple.txt` - simple list format
- **Function**: Aggregates food occurrences and generates statistics per video
- **Dependencies**: json, csv, pathlib, collections
- **Status**: ✅ Core reporting step - REQUIRED

#### ✅ **4_build_food_image_index.py** (STEP 4)
- **Purpose**: Unzip frame archives and build searchable food image index
- **Input**:
  - `GroundTruth-SparseAnnotations/rgb_frames/` (ZIP archives)
  - `visor_food_items.json` (from Step 1)
- **Output**:
  - `food_image_index.json` (1.4 GB) - complete index
  - `food_inventory_lookup.json` (17 MB) - optimized lookup
- **Function**: Indexes all food images for efficient retrieval
- **Dependencies**: json, zipfile, pathlib, collections, os
- **Status**: ✅ Core indexing step - REQUIRED (but slow, ~15-30 mins)
- **Note**: This step unzips 157 ZIP files (~2.5 GB of frames). Contains a previously fixed bug (UPDATE_SUMMARY.md)

#### ✅ **5_extract_wdtcf_food_items.py** (STEP 5)
- **Purpose**: Extract food items from WDTCF temporal annotations
- **Input**:
  - `WDTCF_GT.json` (temporal storage tracking data)
  - `epic_food_nouns_detailed.json` (food classifications)
- **Output**: 
  - `wdtcf_food_items.json` (87 KB)
  - `wdtcf_food_per_video_simple.txt` (1.7 KB)
- **Function**: Tracks where food items were first stored in the kitchen
- **Dependencies**: json, pathlib
- **Status**: ⚠️ **OPTIONAL - Supplementary dataset**
  - **Usage**: Only needed if temporal storage information is required
  - **Not used by**: No other scripts depend on this output
  - **Complementary to**: VISOR food items but different annotation source (74 videos vs 158 videos)

---

### Supporting & Analysis Scripts (Non-Pipeline)

#### **query_food_inventory.py**
- **Purpose**: Query the food inventory index for specific foods
- **Input**: 
  - `food_inventory_lookup.json` (from Step 4)
  - Food names (command-line argument)
- **Output**: 
  - Console output with image paths
  - Optional: CSV export, image copies to directory
- **Dependencies**: json, csv, pathlib, shutil
- **Status**: ✅ Utility script - query tool for inventory

#### **analyze_food_abundance.py**
- **Purpose**: Analyze food distribution across participants and sessions
- **Input**: `visor_food_items.json` (from Step 1)
- **Output**: Food abundance statistics and contamination risk analysis
- **Dependencies**: json, collections
- **Status**: ✅ Analysis script - generates contamination risk metrics

#### **generate_benchmark_metadata.py**
- **Purpose**: Generate metadata for benchmark creation
- **Input**:
  - `retrieve_samples/{food_class}/` (sample images)
  - EPIC-100 pickle files (external)
  - `food_image_index.json` (from Step 4)
- **Output**: JSON metadata for benchmark datasets
- **Dependencies**: json, pickle, argparse, pathlib, re
- **Status**: ✅ Benchmark generation tool

#### **classify_epic_food_nouns.py**
- **Purpose**: Classify EPIC-100 noun classes as food items using LLM
- **Input**: `EPIC_100_noun_classes_v2.csv`
- **Output**: Food classification results (JSON/text)
- **Dependencies**: json, csv, pathlib, requests, time
- **Status**: ⚠️ Classification tool (uses external Qwen API)
- **Note**: Used to generate `epic_food_nouns_detailed.json` - already run, not needed to rerun

#### **classify_food_objects.py**
- **Purpose**: Classify food objects from VISOR annotations using LLM
- **Input**: VISOR annotations + classification model
- **Output**: Food classification results
- **Dependencies**: json, csv, pathlib, requests, time
- **Status**: ⚠️ Classification tool (similar to above)
- **Note**: Core utility for food classification - already executed

#### **list_visor_videos_per_participant.py**
- **Purpose**: List which videos have VISOR annotations per participant
- **Input**: 
  - EPIC-100 CSV files (external path)
  - `GroundTruth-SparseAnnotations/annotations/` (VISOR files)
- **Output**: `visor_video_list_per_participant.csv` and TXT files
- **Dependencies**: pandas, json, pathlib, collections
- **Status**: ✅ Analysis/listing script

#### **analyze_visor_coverage.py**
- **Purpose**: Compare VISOR coverage vs EPIC-100 videos
- **Input**: 
  - EPIC-100 CSV files (external path)
  - `GroundTruth-SparseAnnotations/annotations/` (VISOR files)
- **Output**: Coverage analysis and statistics
- **Dependencies**: pandas, json, pathlib, collections
- **Status**: ✅ Coverage analysis script

---

## 2. DOCUMENTATION FILES

### Workflow Documentation

#### ✅ **WORKFLOW.md** (Main Documentation)
- **Content**: Complete pipeline workflow, usage examples, output descriptions
- **Sections**:
  - Overview of 2 main steps
  - Prerequisites and dependencies
  - Detailed instructions for Steps 1-5
  - Usage examples and options
  - Output structure and file formats
  - Data files organization
  - Performance notes
- **Status**: ✅ Current and comprehensive

#### ✅ **README.txt**
- **Content**: VISOR dataset information from authors
- **Includes**: Dataset details, citation info, license (CC-BY-NC 4.0)
- **Status**: ✅ Reference document

#### ✅ **UPDATE_SUMMARY.md**
- **Content**: Bug fix summary and update log (2025-11-11)
- **Documents**: 
  - Bug in Step 4 (35 videos missing from index)
  - Fix details and impact analysis
  - Updated statistics
  - New features (video filter in query script)
- **Status**: ✅ Recent update documentation

#### 📋 **BENCHMARK_*.md** (Benchmark Documentation)
- **BENCHMARK_CREATION_SUMMARY.md** - Overview of benchmark creation process
- **BENCHMARK_METADATA_GUIDE.md** - Metadata structure guide
- **BENCHMARK_QUICK_REFERENCE.md** - Quick reference for benchmark tools
- **Status**: ✅ Reference documentation

#### 📋 **Other Documentation**
- **QUERY_TOOL_USAGE.md** - Query tool documentation
- **CORRECTED_UNDERSTANDING.md** - Understanding/notes document
- **Status**: ✅ Reference materials

### Summary & Analysis Reports

#### 📊 **Text Report Files**
- `food_per_video.txt` - Detailed food analysis per video
- `food_per_video_simple.txt` - Simple food list per video
- `wdtcf_food_per_video_simple.txt` - WDTCF food simple list
- `food_abundance_summary.txt` - Food abundance analysis
- `distractor_availability_report.txt` - Distractor analysis
- `visor_annotated_videos_only.txt` - List of VISOR videos
- `visor_video_list_per_participant.txt` - Videos per participant
- `wdtcf_extraction_summary.txt` - WDTCF extraction summary
- `epic_food_nouns_names.txt` - Simple food noun list
- **Status**: ✅ Generated reports (auto-generated, can be regenerated)

---

## 3. DATA FILES

### Input Data (Keep - Required)

#### 📁 **VISOR Annotations**
- **Location**: `GroundTruth-SparseAnnotations/`
- **Structure**:
  - `annotations/train/` - 115 JSON files with train annotations
  - `annotations/val/` - 43 JSON files with val annotations
  - `rgb_frames/train/` - 115 ZIP archives with frames
  - `rgb_frames/val/` - 43 ZIP archives with frames
- **Size**: ~2.5 GB (zipped frames)
- **Usage**: Input for Steps 1 & 4
- **Status**: ✅ Keep - essential source data

#### 📄 **Classification Data**
- `EPIC_100_noun_classes_v2.csv` (305 noun classes)
  - Source: EPIC-KITCHENS-100 dataset
  - Usage: Input for Step 1 (class mapping)
- `epic_food_nouns_detailed.json` (150 food classes)
  - Auto-generated by classify_epic_food_nouns.py
  - Usage: Input for Steps 1, 5 (food filtering)
- `epic_food_nouns_detailed.csv` (same as above, CSV format)
- **Status**: ✅ Keep - core classification data

#### 📄 **WDTCF Ground Truth**
- `WDTCF_GT.json` (28 KB)
  - Source: EPIC-KITCHENS WDTCF annotations
  - Usage: Input for Step 5
- **Status**: ✅ Keep - supplementary annotation source

---

### Output Data (Step 1-3 Pipeline)

#### 🔴 **STEP 1 OUTPUT** - Primary Food Extraction
- **`visor_food_items.json`** (404 MB)
  - Contains: All 32,645 food occurrences from 158 videos
  - Structure: Nested JSON by video_id with frame-level annotations
  - **CRITICALITY**: 🔴 **HIGH** - Foundation for all downstream analysis
  - Used by: Steps 3, 4, query tools
  - Dependencies: Must regenerate if input annotations change
  - **Status**: ✅ Current (last updated 2025-11-10)

#### 🟢 **STEP 3 OUTPUTS** - Food Analysis Reports
- **`food_per_video.json`** (2.8 MB)
  - Aggregated food data: per-video statistics, frame ranges
  - **CRITICALITY**: 🟡 **MEDIUM** - Summarized view
  - Used by: Analysis tools, queries
  - **Status**: ✅ Current (last updated 2025-11-11)

- **`food_per_video.csv`** (aggregated tabular format)
  - **CRITICALITY**: 🟡 **MEDIUM** - For spreadsheet analysis
  - **Status**: ✅ Current

- **`food_per_video.txt`** + **`food_per_video_simple.txt`** (human-readable)
  - **CRITICALITY**: 🟢 **LOW** - Reference/reporting only
  - **Status**: ✅ Current

---

### Output Data (Step 4 - Image Indexing)

#### 🔴 **STEP 4 OUTPUTS** - Image Index
- **`food_image_index.json`** (1.4 GB)
  - Contains: Full mapping of all 23,933 food images to annotations
  - Structure: Indexed by food_class, video_id, frame_number
  - **CRITICALITY**: 🔴 **HIGH** - Required for image retrieval
  - Used by: query_food_inventory.py, benchmark generation
  - **Status**: ✅ Current (last updated 2025-11-11, bug fixed)
  - **Note**: Can be regenerated from Steps 1 + unzipped frames

- **`food_inventory_lookup.json`** (17 MB)
  - Contains: Optimized lookup index for food inventory queries
  - **CRITICALITY**: 🔴 **HIGH** - Required for query operations
  - Used by: query_food_inventory.py exclusively
  - **Status**: ✅ Current (last updated 2025-11-11)
  - **Note**: Must be kept synchronized with food_image_index.json

---

### Output Data (Step 5 - WDTCF)

#### 🟡 **STEP 5 OUTPUTS** - WDTCF Food Items
- **`wdtcf_food_items.json`** (87 KB)
  - Contains: 135 food instances from 74 videos (WDTCF task)
  - **CRITICALITY**: 🟢 **LOW** - Supplementary dataset
  - Used by: None (standalone analysis)
  - **Status**: ✅ Current (last updated 2025-11-10)
  - **Relationship**: Different annotation source than VISOR
  - **Note**: Can be regenerated if needed, but rarely used

---

### Supporting Data Files

#### 🟡 **Analysis & Metadata**
- `food_abundance_analysis.json` (83 KB)
  - Contamination risk analysis per food
  - Generated by: analyze_food_abundance.py
  - **Status**: ✅ Current (2025-11-11)

- `distractor_availability_analysis.json` (25 KB)
  - Distractor availability metrics
  - **Status**: ✅ Current

- `benchmark_metadata_template.json` (7.8 KB)
  - Template for benchmark metadata
  - **Status**: ✅ Reference template

- `yoghurt_benchmark_metadata.json` (1.6 MB)
  - Example benchmark metadata for yoghurt
  - **Status**: ✅ Example benchmark

#### 🟢 **Utility Data**
- `.info.json` - System information file
- `visor_video_list_per_participant.csv` - Listing of VISOR videos

---

## 4. DIRECTORY STRUCTURE

```
epic-kitchen-visor/
├── WORKFLOW.md                                    # Main documentation
├── README.txt                                     # VISOR dataset info
├── UPDATE_SUMMARY.md                              # Recent updates
├── BENCHMARK_*.md                                 # Benchmark documentation
│
├── 1_extract_food_items.py                        # STEP 1: Extract
├── 2_create_food_segments.py                      # STEP 2: Visualize (optional)
├── 3_analyze_food_per_video.py                    # STEP 3: Analyze
├── 4_build_food_image_index.py                    # STEP 4: Index
├── 5_extract_wdtcf_food_items.py                  # STEP 5: WDTCF (optional)
├── query_food_inventory.py                        # Query tool
├── analyze_food_abundance.py                      # Analysis tool
├── generate_benchmark_metadata.py                 # Benchmark tool
├── classify_epic_food_nouns.py                    # Classifier (pre-run)
├── classify_food_objects.py                       # Classifier (pre-run)
├── list_visor_videos_per_participant.py           # Listing tool
├── analyze_visor_coverage.py                      # Coverage analysis
│
├── EPIC_100_noun_classes_v2.csv                   # Input: Noun taxonomy (305 classes)
├── epic_food_nouns_detailed.json                  # Input: Food classifications (150 classes)
├── epic_food_nouns_detailed.csv                   # Input: Food classifications (CSV)
├── epic_food_nouns_names.txt                      # Input: Simple food noun list
├── WDTCF_GT.json                                  # Input: WDTCF annotations (28 KB)
│
├── visor_food_items.json                          # OUTPUT Step 1: All food items (404 MB)
├── food_per_video.json                            # OUTPUT Step 3: Analysis JSON (2.8 MB)
├── food_per_video.csv                             # OUTPUT Step 3: Analysis CSV
├── food_per_video.txt                             # OUTPUT Step 3: Detailed report
├── food_per_video_simple.txt                      # OUTPUT Step 3: Simple list
│
├── food_image_index.json                          # OUTPUT Step 4: Image index (1.4 GB)
├── food_inventory_lookup.json                     # OUTPUT Step 4: Query index (17 MB)
│
├── wdtcf_food_items.json                          # OUTPUT Step 5: WDTCF items (87 KB)
├── wdtcf_food_per_video_simple.txt                # OUTPUT Step 5: WDTCF list
│
├── food_abundance_analysis.json                   # Analysis data
├── distractor_availability_analysis.json          # Analysis data
├── benchmark_metadata_template.json               # Benchmark template
├── yoghurt_benchmark_metadata.json                # Benchmark example (1.6 MB)
│
├── [Text report files]                            # Auto-generated reports
├── [CSV listing files]                            # Auto-generated listings
│
├── GroundTruth-SparseAnnotations/                 # VISOR Annotations (INPUT)
│   ├── annotations/
│   │   ├── train/                                 # 115 JSON files
│   │   └── val/                                   # 43 JSON files
│   └── rgb_frames/
│       ├── train/                                 # 115 ZIP archives
│       └── val/                                   # 43 ZIP archives
│
├── Interpolations-DenseAnnotations/               # Dense VISOR annotations (supplementary)
│   ├── train/
│   └── val/
│
└── retrieve_samples/                              # Benchmark sample images
    ├── chicken/                                   # Sample food images
    ├── oil/
    ├── onion/
    ├── pizza/
    └── yoghurt/
```

---

## 5. DEPENDENCY GRAPH

### Pipeline Dependencies

```
VISOR Annotations (GroundTruth-SparseAnnotations/)
├─ EPIC_100_noun_classes_v2.csv
├─ epic_food_nouns_detailed.json
│
└─> STEP 1: 1_extract_food_items.py
    └─> visor_food_items.json
        │
        ├─> STEP 3: 3_analyze_food_per_video.py
        │   └─> food_per_video.{json,csv,txt}
        │
        ├─> STEP 4: 4_build_food_image_index.py
        │   └─> food_image_index.json
        │       └─> food_inventory_lookup.json
        │           └─> query_food_inventory.py
        │
        └─> analyze_food_abundance.py
            └─> food_abundance_analysis.json

WDTCF_GT.json
├─ epic_food_nouns_detailed.json
│
└─> STEP 5: 5_extract_wdtcf_food_items.py
    └─> wdtcf_food_items.json (STANDALONE - not used by other scripts)
```

### Script Dependencies (What each needs)

| Script | Depends On | Optional | Notes |
|--------|-----------|----------|-------|
| 1_extract_food_items.py | VISOR annotations, CSVs/JSONs | No | Core step |
| 2_create_food_segments.py | visor_food_items.json, rgb_frames | Yes | Visualization only |
| 3_analyze_food_per_video.py | visor_food_items.json | No | Core step |
| 4_build_food_image_index.py | visor_food_items.json, rgb_frames | No | Core step (slow) |
| 5_extract_wdtcf_food_items.py | WDTCF_GT.json | Yes | Supplementary only |
| query_food_inventory.py | food_inventory_lookup.json | - | Query tool |
| analyze_food_abundance.py | visor_food_items.json | - | Analysis tool |
| generate_benchmark_metadata.py | food_image_index.json | - | Benchmark tool |
| classify_epic_food_nouns.py | EPIC CSV, external LLM API | - | Pre-run utility |
| list_visor_videos_per_participant.py | VISOR annotations, EPIC CSVs | - | Listing tool |
| analyze_visor_coverage.py | VISOR annotations, EPIC CSVs | - | Coverage tool |

---

## 6. DATA FILE CRITICALITY ASSESSMENT

### Red (Critical - Must Keep)

| File | Size | Purpose | Why Critical |
|------|------|---------|--------------|
| visor_food_items.json | 404 MB | Raw food extraction | Foundation for all analysis |
| food_image_index.json | 1.4 GB | Image retrieval index | Required for image queries |
| food_inventory_lookup.json | 17 MB | Fast lookup index | Required for query_food_inventory.py |
| GroundTruth-SparseAnnotations/ | 2.5 GB | VISOR source data | Source annotations |
| epic_food_nouns_detailed.json | 63 KB | Food classifications | Used by Steps 1 & 5 |
| EPIC_100_noun_classes_v2.csv | ~100 KB | Noun taxonomy | Input for Step 1 |

**Total Critical Size**: ~4.5 GB

### Yellow (Important - Should Keep)

| File | Size | Purpose | Reason |
|------|------|---------|--------|
| food_per_video.json | 2.8 MB | Aggregated analysis | Useful reference |
| food_abundance_analysis.json | 83 KB | Risk analysis | Analytical output |
| wdtcf_food_items.json | 87 KB | WDTCF extraction | Supplementary data |
| All text reports | ~200 KB | Human-readable output | Reference/reporting |

**Total Important Size**: ~3.2 MB

### Green (Optional - Can Delete/Regenerate)

| File | Size | Purpose | Regeneration |
|------|------|---------|--------------|
| food_per_video.csv/txt | ~100 KB | CSV/text reports | Run Step 3 (~1 sec) |
| food_segments/ | ~168 MB/video | Visual masks | Run Step 2 (~2 sec/video) |
| retrieve_samples/ | Variable | Sample images | External source or Step 4 |
| Interpolations-DenseAnnotations/ | Not measured | Dense annotations | External (supplementary) |
| yoghurt_benchmark_metadata.json | 1.6 MB | Benchmark metadata | Run generate_benchmark_metadata.py |

**Total Regenerable Size**: ~170+ MB

---

## 7. STEP 5 (WDTCF) ANALYSIS

### What is Step 5?

**5_extract_wdtcf_food_items.py** extracts food items from the **WDTCF (Where Did The Container First appear)** temporal localization task. This is a different annotation source than VISOR.

### Characteristics

- **Input**: WDTCF_GT.json (28 KB) - Temporal storage tracking
- **Output**: wdtcf_food_items.json (87 KB) - 135 food instances from 74 videos
- **Different from VISOR**: 
  - VISOR: 158 videos, 32,645 food occurrences, pixel-level segmentation
  - WDTCF: 74 videos, 135 food instances, temporal first-appearance tracking
- **Annotation Type**: Tracks where objects were FIRST stored in kitchen
- **Frame Information**: Includes both "query frame" and "evidence frame"

### Is Step 5 Needed?

**No**, unless you specifically need:
- Temporal information about where food was first stored
- Storage location tracking (fridge, cupboard, drawer)
- Different set of 74 videos with different annotation granularity

### Dependencies

- **Nothing depends on wdtcf_food_items.json**
- No other scripts import or use it
- It's a standalone output
- Cannot be regenerated without the input WDTCF_GT.json

### Recommendation

**OPTIONAL - Keep if storing temporal food location data is required**
- Can safely delete if only need VISOR food items
- Rare use case for food item retrieval tasks
- Adds minimal overhead (87 KB output file)

---

## 8. VISOR ANNOTATIONS - ESSENTIAL?

### Are VISOR Annotations Essential to Current System?

**YES - Absolutely critical**

### Why

1. **Data Source**: VISOR annotations (GroundTruth-SparseAnnotations/) are the PRIMARY INPUT
   - 158 videos with 32,645 food occurrences
   - Pixel-level segmentation masks
   - 23,933 food images after extraction

2. **Pipeline Foundation**: All core outputs depend on this:
   - `visor_food_items.json` from Step 1
   - `food_image_index.json` from Step 4
   - All food queries and inventory tools

3. **Cannot be replaced**:
   - These are the actual annotated frames
   - No other data source has this level of detail
   - The entire food inventory system is built on VISOR

4. **Independent from Main NeuroTrace**:
   - This is a separate subsystem for food analysis
   - Does NOT depend on VLM results or narration extraction
   - But IS self-contained and complete

### VISOR Annotations Must-Keep Files

- `GroundTruth-SparseAnnotations/annotations/` - JSON annotation files (156 MB)
- `GroundTruth-SparseAnnotations/rgb_frames/` - Frame ZIPs (2.3 GB)
- Both train/ and val/ splits

---

## 9. REFACTORING RECOMMENDATIONS

### For Main NeuroTrace Integration

#### Current State
- epic-kitchen-visor is **completely independent** from main pipeline
- Runs its own 5-step food analysis workflow
- Produces food inventory data separate from video analysis

#### If Integrating with Main Pipeline

Option 1: Keep Separate (Recommended)
- `epic-kitchen-visor/` remains independent subsystem
- Main pipeline calls it as external tool when food data needed
- Clean separation of concerns

Option 2: Merge Into Pipeline
- Move scripts to main `/kitchen/` directory
- Integrate as Stage 0 or supplementary stage
- Share data through unified output structure
- Requires refactoring import paths

#### Files Safe to Remove from epic-kitchen-visor

**To Clean Up Space** (if consolidating):
- `classify_epic_food_nouns.py` - Already executed, produces `epic_food_nouns_detailed.json`
- `classify_food_objects.py` - Already executed, pre-run utility
- All `*_simple.txt` files - Can regenerate from JSON in seconds
- `food_per_video.csv` - Can regenerate from JSON
- All documentation except `WORKFLOW.md` and `UPDATE_SUMMARY.md`
- `retrieve_samples/` directory if not using benchmark generation
- `Interpolations-DenseAnnotations/` if not using dense annotations

**Total Deletable**: ~200 MB if removing sample data

---

## 10. SUMMARY TABLE

| Category | Item | Size | Keep? | Reason |
|----------|------|------|-------|--------|
| **INPUTS** | VISOR Annotations | 2.5 GB | ✅ YES | Source data |
| | EPIC noun classes | 100 KB | ✅ YES | Classification |
| | Food noun mapping | 63 KB | ✅ YES | Filter criteria |
| | WDTCF data | 28 KB | ✅ YES | Supplementary |
| **CORE OUTPUTS** | visor_food_items.json | 404 MB | ✅ YES | Step 1 foundation |
| | food_image_index.json | 1.4 GB | ✅ YES | Step 4 index |
| | food_inventory_lookup.json | 17 MB | ✅ YES | Query lookup |
| **ANALYSIS OUTPUTS** | food_per_video.json | 2.8 MB | 🟡 MAYBE | Reference data |
| | food_abundance_analysis.json | 83 KB | 🟡 MAYBE | Risk analysis |
| **OPTIONAL OUTPUTS** | wdtcf_food_items.json | 87 KB | 🟢 OPTIONAL | Standalone |
| | Food segment images | 168 MB+/video | 🟢 OPTIONAL | Visualizations |
| | Benchmark metadata | 1.6 MB | 🟢 OPTIONAL | If creating benchmarks |
| **UTILITIES** | Steps 1-5 scripts | ~50 KB | ✅ YES | Pipeline |
| | Query/analysis scripts | ~30 KB | ✅ YES | Tools |
| **DOCUMENTATION** | WORKFLOW.md | ~15 KB | ✅ YES | Critical reference |
| | All other docs | ~50 KB | 🟢 OPTIONAL | Reference only |

---

## 11. EXECUTION ORDER & TIME ESTIMATES

### Recommended Workflow

```bash
# Step 1: Extract all food items (2-3 minutes)
python3 1_extract_food_items.py
# Output: visor_food_items.json (404 MB)

# Step 3: Generate analysis reports (1-2 seconds)
python3 3_analyze_food_per_video.py
# Output: food_per_video.* files

# Step 4: Build image index (15-30 minutes - SLOW)
python3 4_build_food_image_index.py
# Output: food_image_index.json, food_inventory_lookup.json

# Step 2 (Optional): Create segmentations (~1-2 sec per video)
python3 2_create_food_segments.py --limit 5
# Output: food_segments/ directory

# Step 5 (Optional): Extract WDTCF items (1-2 seconds)
python3 5_extract_wdtcf_food_items.py
# Output: wdtcf_food_items.json

# Use the tools:
python3 query_food_inventory.py --food yoghurt --limit 5
python3 analyze_food_abundance.py
python3 generate_benchmark_metadata.py --food yoghurt
```

### Total Time
- **Essential steps (1, 3, 4)**: ~20-35 minutes
- **With optional steps**: ~25-40 minutes
- **Setup**: ~5 minutes (first time dependencies)

---

## 12. KEY FINDINGS

1. **Epic-kitchen-visor is a complete, self-contained system**
   - Not integrated with main NeuroTrace pipeline
   - Separate VISOR food analysis workflow
   - Can run independently

2. **Pipeline is well-documented**
   - WORKFLOW.md is comprehensive
   - UPDATE_SUMMARY.md documents recent bug fix
   - Clear step-by-step instructions

3. **Step 5 (WDTCF) is supplementary**
   - Optional extraction from different annotation source
   - Not used by any other scripts
   - Can be safely skipped if not needed

4. **Data size is substantial**
   - 1.4 GB image index is the largest file
   - 404 MB raw extraction file
   - 2.5 GB source annotations
   - Total: ~4.5 GB critical data

5. **Image indexing (Step 4) is slow**
   - Unzips 157 archives (~2.5 GB)
   - ~15-30 minutes execution time
   - Can be skipped if not doing image queries

6. **VISOR annotations are irreplaceable**
   - Core source data for all food analysis
   - Cannot be deleted or regenerated
   - Must be kept for reproducibility


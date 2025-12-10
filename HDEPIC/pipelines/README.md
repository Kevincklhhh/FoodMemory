# HDEPIC Food State Tracking Pipeline

This pipeline tracks food items through cooking videos, building a spatio-temporal graph of food state changes.

## Pipeline Overview

```
Narrations (CSV)
     │
     ▼
┌─────────────────────────────────┐
│ 01_group_and_classify_food_blocks │  → food_classification/{video}_food_blocks.json
└─────────────────────────────────┘
     │
     ├──────────────────────────────┐
     ▼                              ▼
┌──────────────────┐    ┌────────────────────────┐
│ 02_extract_clips │    │ 03_inventory_discovery │
│ (optional)       │    │ (range of videos)      │
└──────────────────┘    └────────────────────────┘
     │                              │
     ▼                              ▼
food_clips/{video}/         inventory_discovery/{video}_arrivals.json
     │                              │
     └──────────────┬───────────────┘
                    ▼
        ┌─────────────────────────┐
        │ 04_state_description    │  → food_classification/state_descriptions/
        └─────────────────────────┘
                    │
                    ▼
        ┌─────────────────────────┐
        │ 05_food_graph_builder   │  → food_graph/spatio_temporal_graph.json
        └─────────────────────────┘
                    │
                    ▼
        ┌─────────────────────────┐
        │ 06_visualize_graph      │  → food_graph/visualization.png
        └─────────────────────────┘
```

## Pipeline Scripts

| Script | Purpose | Inputs | Outputs |
|--------|---------|--------|---------|
| `01_group_and_classify_food_blocks.py` | Group narrations into 30s blocks, classify food actions | `participant_P01_narrations.csv` | `food_classification/{video}_food_blocks.json` |
| `02_extract_food_clips.py` | Extract video clips for each block | Source videos + food_blocks.json | `food_clips/{video}/block_NNN.mp4` |
| `03_inventory_discovery.py` | Identify first appearances of food items | food_blocks.json | `inventory_discovery/{video}_arrivals.json` |
| `04_state_description.py` | Generate natural language state descriptions | food_blocks.json + video clips | `food_classification/state_descriptions/` |
| `05_food_graph_builder.py` | Build spatio-temporal food graph | arrivals.json + state descriptions | `food_graph/spatio_temporal_graph.json` |
| `06_visualize_graph.py` | Visualize the food graph | spatio_temporal_graph.json | `food_graph/visualization.png` |

## Quick Start

```bash
cd pipelines/

# Step 1: Group narrations into blocks
python 01_group_and_classify_food_blocks.py

# Step 2: Extract video clips (optional, for VLM with video)
python 02_extract_food_clips.py --video-id P01-20240202-110250

# Step 3: Discover inventory (can specify video range)
python 03_inventory_discovery.py --local --start-video P01-20240202-110250 --end-video P01-20240202-161948

# Step 4: Generate state descriptions
python 04_state_description.py --video-id P01-20240202-110250

# Step 5: Build food graph
python 05_food_graph_builder.py --local --start-video P01-20240202-110250 --end-video P01-20240202-161948

# Step 6: Visualize
python 06_visualize_graph.py
```

## Directory Structure

```
HDEPIC/
├── pipelines/
│   ├── 01_group_and_classify_food_blocks.py
│   ├── 02_extract_food_clips.py
│   ├── 03_inventory_discovery.py
│   ├── 04_state_description.py
│   ├── 05_food_graph_builder.py
│   ├── 06_visualize_graph.py
│   ├── food_graph/              # Core data structures
│   │   ├── data_structures.py   # FoodNode, BlockGraph, etc.
│   │   ├── graph_operations.py  # Transaction handling
│   │   └── vlm_prompts.py       # VLM prompt templates
│   ├── food_state_taxonomy.json # State schema
│   └── format_vlm_logs.py       # Utility
│
└── outputs/
    ├── food_classification/     # Blocks + state descriptions
    ├── food_clips/              # Video clips
    ├── inventory_discovery_global/
    ├── inventory_discovery_local/
    ├── food_graph/              # Graph outputs
    └── food_graph_local/
```

## Key Concepts

### Food State Taxonomy
- **form_state**: whole, prepared_ingredient, cooking_in_progress, cooked_dish, leftover
- **quantity**: full, partial, nearly_empty
- **count**: integer for countable items (eggs, slices)

### Transaction Types
- **TRANSFER**: Move food to a different container
- **SPLIT**: Divide food (partial or complete)
- **MERGE**: Combine foods (accumulation, incorporation, transformation)
- **CONSUME**: Food eaten or discarded
- **UPDATE**: Change state in place

### VLM Models
- **Qwen3-VL**: Default, supports video input
- **GPT-4o**: Text-only mode available
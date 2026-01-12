# HDEPIC Pipeline Outputs Directory

This directory contains all outputs from the HDEPIC inventory tracking pipeline. The structure is organized by pipeline stage and output type for easy navigation and maintenance.

## Directory Structure

```
outputs/
├── 01_narrations/              # Narration processing outputs
│   └── filtered/{participant}/ # Filtered food-related narrations by participant
│                                 (from 01_export_narrations.py)
│
├── 02_inventory/               # Inventory tracking outputs
│   ├── lifecycle/              # Recipe-level inventory and lifecycle JSON files
│   │                             (inventory_{recipe_id}.json, lifecycle_{recipe_id}.json)
│   ├── video_lifecycle/        # Video-level inventory and lifecycle data
│   ├── clips/{recipe_id}/      # Video clips of inventory lifecycle events
│   └── edits/                  # Lifecycle edit annotations
│
├── 03_dispensal/               # Dispensal event tracking
│   ├── actions/                # Dispensal action data
│   ├── classified/             # Classified dispensal difficulty/method
│   ├── clips/{recipe_id}/      # Video clips of dispensal events
│   └── timestamps/             # Dispensal event timestamps
│
├── 04_food/                    # Food state and classification
│   ├── classification/         # Food classification results
│   ├── clips/{video_id}/       # Food-related video clips
│   ├── graphs/                 # Food state transition graphs
│   │   ├── gemini/             # Gemini-generated food graphs
│   │   └── local/{video_range}/# Locally-generated food graphs
│   └── state_descriptions/{video_id}/ # Food state descriptions
│
├── legacy/                     # Deprecated pipeline outputs
│   ├── chunks/                 # Old chunked data (deprecated)
│   ├── graphs/                 # Old graph outputs (deprecated)
│   └── video_ranges/           # Old video range outputs (deprecated)
│
└── old/                        # Archived outputs from previous pipeline versions
```

## File Statistics

- **Narrations**: 156 files (filtered narrations)
- **Inventory**: 531 files (48 recipe-level JSONs + clips + edits)
- **Dispensal**: 78 files (action data, classifications, timestamps)
- **Food**: 560 files (classifications, clips, graphs, state descriptions)

## Pipeline Scripts and Output Locations

| Script | Output Directory | Description |
|--------|-----------------|-------------|
| `01_export_narrations.py` | `01_narrations/filtered/{participant}/` | Filters food-related narrations |
| `02_inventory_transactions.py` | `02_inventory/lifecycle/` | Generates recipe-level inventory & lifecycle |
| `03_export_lifecycle_edits.py` | `02_inventory/edits/` | Exports lifecycle event annotations |
| `04_dispensal_classification.py` | `03_dispensal/classified/` | Classifies dispensal events |
| `05_get_timestamps.py` | Various | Extracts timestamps for events |

## Key Output Files

### Inventory Lifecycle Files
- **Format**: `inventory_{recipe_id}[_C{capture}].json`
- **Content**: List of discovered inventory items with metadata
- **Example**: `inventory_P01_R03.json`, `inventory_P01_R01_C0.json`

### Lifecycle Event Files
- **Format**: `lifecycle_{recipe_id}[_C{capture}].json`
- **Content**: Lifecycle events (RETRIEVAL, ACCESS, DISPENSING, RESTOCKING) per item
- **Example**: `lifecycle_P01_R03.json`, `lifecycle_P03_R05.json`

### Filtered Narrations
- **Format**: `filtered/{participant}/filter_{video_id}_response.txt`
- **Content**: JSON with filtered narration IDs
- **Example**: `filtered/P01/filter_P01-20240202-110250_response.txt`

## Recipe ID Naming Convention

- **Format**: `{Participant}_R{Recipe_Number}[_C{Capture}]`
- **Participant**: P01-P09 (participant identifier)
- **Recipe Number**: R01-R10 (unique recipe within participant)
- **Capture**: Optional C0, C1, C2... (for recipes with multiple captures/sessions)

**Examples**:
- `P01_R03` - Participant 1, Recipe 3 (single capture)
- `P01_R01_C0` - Participant 1, Recipe 1, Capture 0 (first of multiple captures)
- `P03_R05` - Participant 3, Recipe 5

## Video ID Format

- **Format**: `{Participant}-{YYYYMMDD}-{HHMMSS}`
- **Example**: `P01-20240202-110250` (Participant 01, February 2 2024, 11:02:50)

## Usage Notes

1. **Recipe-level processing**: Most inventory outputs are organized by recipe, combining multiple videos
2. **Multi-capture recipes**: Some recipes have multiple independent captures (cooking sessions), indicated by `_C{N}` suffix
3. **Video-first mode**: Video-level outputs in `02_inventory/video_lifecycle/` process each video independently
4. **Legacy outputs**: The `legacy/` directory contains deprecated pipeline outputs that may be removed in future versions

## Maintenance

- **Archiving**: Move old/deprecated outputs to `old/` directory with timestamp
- **Cleanup**: Periodically review `legacy/` directory for safe removal
- **Backup**: Recipe-level JSONs in `02_inventory/lifecycle/` are critical outputs

---

*Last updated*: 2026-01-09
*Pipeline version*: Inventory Lifecycle v2.0

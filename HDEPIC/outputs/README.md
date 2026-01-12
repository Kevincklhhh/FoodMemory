# HDEPIC Inventory Pipeline Outputs

This directory contains all outputs from the HDEPIC inventory tracking pipeline.

## Directory Structure

```
outputs/
├── 01_narrations/              # Narration processing outputs
│   └── filtered/
│       ├── P01/                # Participant-specific filtered narrations
│       ├── P02/
│       └── P03/
│
├── 02_inventory/               # Inventory tracking outputs
│   ├── P01/                    # Participant-specific inventory outputs
│   │   ├── P01_discovery_edit.json      # Step 02: Discovered items
│   │   ├── P01_lifecycle.json           # Step 03: Lifecycle events
│   │   ├── P01_dispensal_classified.json # Step 04: Difficulty ratings
│   │   └── P01_known_quantities.json    # Step 05: Items with known quantities
│   ├── P02/
│   └── P03/
│
└── 04_food/                    # Food state graphs (separate pipeline)
    └── graphs/
```

## Pipeline Steps

| Step | Script | Output | Description |
|------|--------|--------|-------------|
| 01 | `01_export_narrations.py` | `01_narrations/filtered/{P}/` | Filter food-related narrations |
| 02 | `02_inventory_discovery.py` | `{P}_discovery_edit.json` | Discover inventory items with deduplication |
| 03 | `03_lifecycle_tracking.py` | `{P}_lifecycle.json` | Track lifecycle events per item |
| 04 | `04_dispensal_classification.py` | `{P}_dispensal_classified.json` | Classify dispensal difficulty |
| 05 | `05_filter_for_annotation.py` | `{P}_known_quantities.json` | Filter items with known/easy quantities |

## Output File Formats

### discovery_edit.json
Contains discovered inventory items with:
- `narration_id`: First appearance narration
- `food_name`: Item description
- `video_range`: List of videos where item appears
- `ingredient_matches`: Recipe ingredient matches (with amounts)
- `include`: Whether to include in lifecycle tracking
- `aliases`: Merged duplicate items (from deduplication)

### lifecycle.json
Contains lifecycle events per item:
- `events`: List of RETRIEVAL, ACCESS, DISPENSING, RESTOCKING events
- `video_range`: Videos analyzed
- Each event has: `narration_id`, `stage`, `action`

### dispensal_classified.json
Contains difficulty ratings per item:
- `difficulty`: LOW (countable), MID (geometric), HIGH (continuous)
- `reasoning`: Explanation for rating
- `events`: All lifecycle events

### known_quantities.json
Items with known or easily measurable quantities:
- Items with `difficulty = LOW`
- Items with matched ingredient that has `amount` field

## Video ID Format

`{Participant}-{YYYYMMDD}-{HHMMSS}`

Example: `P01-20240202-110250` (Participant 01, February 2 2024, 11:02:50)

## Usage

Run the full pipeline:
```bash
cd inventory_pipeline
./run_all_participants.sh P01
```

Run a specific step:
```bash
python 02_inventory_discovery.py --participant P01
python 03_lifecycle_tracking.py --participant P01
python 04_dispensal_classification.py --participant P01
python 05_filter_for_annotation.py --participant P01
```

---
*Last updated*: 2025-01-12
*Pipeline version*: Inventory Lifecycle v3.0

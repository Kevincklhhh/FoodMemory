# Inventory Pipeline Guide

This guide describes how to run the inventory pipeline (steps 01-06) for new participants.

## Available Participants
P01, P02, P03, P04, P05, P06, P07, P08, P09

**Completed:** P01, P03

## Pipeline Overview

```
01_export_narrations.py     → (Optional) Filter food narrations from raw data
        ↓
02_inventory_discovery.py   → Discover all food items entering the scene
        ↓
    [MANUAL EDIT]           → Human review: edit {participant}_discovery_edit.json
        ↓
03_lifecycle_tracking.py    → Track lifecycle events (get, dispense, store) for each item
        ↓
04_dispensal_classification.py → Classify difficulty (LOW/MID/HIGH) for counting
        ↓
05_filter_for_annotation.py → Extract items with known quantities + match to recipes
        ↓
06_timeline_aggregation.py  → Aggregate dispensing events into time segments with counts
        ↓
    [MANUAL ANNOTATION]     → Human review: verify/edit timeline_annotated.json
        ↓
07_vlm_QA.py               → (Optional) Test VLM quantity estimation
```

## Step-by-Step Instructions

### Step 1: Export Narrations (Optional)

Only needed if narrations haven't been pre-filtered. Most participants already have narrations in the pickle file.

```bash
# List available videos for a participant
python inventory_pipeline/01_export_narrations.py list --participant P04

# Filter food-related narrations (uses GPT-5.2)
python inventory_pipeline/01_export_narrations.py filter --participant P04
```

### Step 2: Inventory Discovery

Discovers all food items that enter the cooking scene.

```bash
python inventory_pipeline/02_inventory_discovery.py \
    --participant P04 \
    --model gpt-5.2 \
    --reasoning \
    --reasoning-effort high \
    --verbose
```

**Output:** `outputs/02_inventory/P04/P04_discovery.json`

### Step 2.5: Manual Review (REQUIRED)

Copy discovery output and manually verify/edit:

```bash
cp outputs/02_inventory/P04/P04_discovery.json \
   outputs/02_inventory/P04/P04_discovery_edit.json
```

Edit `P04_discovery_edit.json` to:
- Remove false positives (non-food items, duplicates)
- Add missing food items
- Correct food names if needed

### Step 3: Lifecycle Tracking

Tracks all events (get, dispense, store, etc.) for each verified food item.

```bash
python inventory_pipeline/03_lifecycle_tracking.py \
    --participant P04 \
    --model gpt-5.2 \
    --reasoning \
    --reasoning-effort high \
    --verbose
```

**Input:** `P04_discovery_edit.json`
**Output:** `P04_lifecycle.json`

### Step 4: Dispensal Classification

Classifies each item's counting difficulty:
- **LOW**: Discrete countable (eggs, slices, whole items)
- **MID**: Partially countable (scoops, spoonfuls)
- **HIGH**: Continuous/uncountable (liquids, powders)

```bash
python inventory_pipeline/04_dispensal_classification.py \
    --participant P04 \
    --model gpt-5.2 \
    --verbose
```

**Input:** `P04_lifecycle.json`
**Output:** `P04_dispensal_classified.json`

### Step 5: Filter for Annotation

Extracts items with known quantities and matches them to recipe ingredients.

```bash
python inventory_pipeline/05_filter_for_annotation.py \
    --participant P04 \
    --verbose
```

**Input:** `P04_dispensal_classified.json` + `complete_recipes.json`
**Output:**
- `P04_known_quantities.json` (machine-readable)
- `P04_known_quantities.txt` (human-readable summary)

### Step 6: Timeline Aggregation

Aggregates dispensing events into continuous time segments with counts.

```bash
# Test on first 5 items
python inventory_pipeline/06_timeline_aggregation.py \
    --participant P04 \
    --model gpt-5.2 \
    --reasoning-effort high \
    --test 5 \
    --verbose

# Run on all items
python inventory_pipeline/06_timeline_aggregation.py \
    --participant P04 \
    --model gpt-5.2 \
    --reasoning-effort high \
    --verbose
```

**Input:** `P04_known_quantities.json` + narrations
**Output:** `P04_timeline_aggregated.json`

### Step 6.5: Manual Annotation (REQUIRED)

Copy and manually verify/edit the timeline:

```bash
cp outputs/02_inventory/P04/P04_timeline_aggregated.json \
   outputs/02_inventory/P04/P04_timeline_annotated.json
```

Edit `P04_timeline_annotated.json` to:
- Verify timestamps are correct
- Adjust counts if needed
- Add `video_id` to segments if missing

## Quick Run Script

For a new participant, run all steps in sequence:

```bash
PARTICIPANT=P04

# Step 2: Discovery
python inventory_pipeline/02_inventory_discovery.py \
    --participant $PARTICIPANT --model gpt-5.2 --reasoning --reasoning-effort high

# PAUSE: Manually edit discovery_edit.json

# Step 3: Lifecycle
python inventory_pipeline/03_lifecycle_tracking.py \
    --participant $PARTICIPANT --model gpt-5.2 --reasoning --reasoning-effort high

# Step 4: Classification
python inventory_pipeline/04_dispensal_classification.py \
    --participant $PARTICIPANT --model gpt-5.2

# Step 5: Filter
python inventory_pipeline/05_filter_for_annotation.py \
    --participant $PARTICIPANT

# Step 6: Timeline
python inventory_pipeline/06_timeline_aggregation.py \
    --participant $PARTICIPANT --model gpt-5.2 --reasoning-effort high

# PAUSE: Manually edit timeline_annotated.json
```

## Output File Summary

| Step | Output File | Description |
|------|-------------|-------------|
| 2 | `{P}_discovery.json` | Raw discovered food items |
| 2.5 | `{P}_discovery_edit.json` | Human-verified food items |
| 3 | `{P}_lifecycle.json` | Lifecycle events per item |
| 4 | `{P}_dispensal_classified.json` | Items with difficulty labels |
| 5 | `{P}_known_quantities.json` | Items with known counts + recipe matches |
| 5 | `{P}_known_quantities.txt` | Human-readable summary |
| 6 | `{P}_timeline_aggregated.json` | GPT-aggregated time segments |
| 6.5 | `{P}_timeline_annotated.json` | Human-verified timeline |

## Notes

- Steps 2.5 and 6.5 require manual human review
- GPT-5.2 with high reasoning effort gives best results but is slower
- Use `--test N` flag to process only first N items for testing
- Videos must exist in `data/HD-EPIC/Videos/{participant}/` for VLM QA

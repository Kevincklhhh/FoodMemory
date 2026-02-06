
# Inventory Pipeline Guide

This guide describes how to run the inventory pipeline for new participants.

## Available Participants
P01, P02, P03, P04, P05, P06, P07, P08, P09

**Completed:** P01, P02, P03, P04

## Pipeline Overview

```
01_export_narrations.py        → (Optional) Filter food narrations from raw data
        ↓
02_inventory_discovery.py      → Discover all food items entering the scene
        ↓
    [MANUAL EDIT]              → Human review: edit {participant}_discovery_edit.json
        ↓
03_lifecycle_tracking.py       → Track lifecycle events (get, dispense, store) for each item
        ↓
04_dispensal_classification.py → Classify difficulty (LOW/MID/HIGH) for counting
        ↓
05_filter_for_annotation.py    → Extract items with known quantities + match to recipes
        ↓
06_timeline_aggregation.py     → Aggregate dispensing events into time segments with counts
        ↓
    [MANUAL ANNOTATION]        → Human review: verify/edit timeline_annotated.json
        ↓
07_vlm_QA.py                   → Run VLM quantity estimation on video clips
        ↓
08_evaluate_vlm_count.py       → Evaluate VLM predictions against ground truth
        ↓
09_gather_wrong_predictions.py → Collect wrong predictions for analysis
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
    --participant P09 \
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
    --participant P06 \
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
    --participant P06 \
    --model gpt-5.2 \
    --verbose
```

**Input:** `P04_lifecycle.json`
**Output:** `P04_dispensal_classified.json`

### Step 5: Filter for Annotation

Extracts items with known quantities and matches them to recipe ingredients.

```bash
python inventory_pipeline/05_filter_for_annotation.py \
    --participant P06 \
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
    --participant P06 \
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

### Step 7: VLM Q&A (Quantity Estimation)

Runs VLM on extracted video clips to estimate quantities. Requires video files in `data/HD-EPIC/Videos/{participant}/`.

```bash
# Process a single participant with a tag
python inventory_pipeline/07_vlm_QA.py \
    --participant P04 \
    --tag qwen_v1 \
    --model qwen

# Process all participants with annotated timelines
python inventory_pipeline/07_vlm_QA.py \
    --all \
    --tag qwen_v1

# Process only LOW difficulty items (faster, cleaner evaluation)
python inventory_pipeline/07_vlm_QA.py \
    --all \
    --tag qwen_low \
    --low-only

# Skip participants that already have results for this tag
python inventory_pipeline/07_vlm_QA.py \
    --all \
    --tag qwen_v1 \
    --skip-existing

# Test mode (first N items)
python inventory_pipeline/07_vlm_QA.py \
    --participant P04 \
    --tag test \
    --test 5
```

**Options:**
- `--tag`: Required. Identifies this VLM run (used in output filenames)
- `--model`: `qwen` (default, native video) or `gpt4o` (frame sampling at 2fps)
- `--low-only`: Only process LOW difficulty items
- `--skip-existing`: Skip if result file for this tag already exists
- `--delete-clips`: Remove extracted video clips after processing
- `--no-eval`: Skip automatic evaluation report generation

**Input:** `P04_timeline_annotated.json` + video files
**Output:**
- `P04_vlm_qa_{tag}_results.json` - VLM predictions with match evaluation
- `vlm_clips/` - Extracted video clips (unless `--delete-clips`)

### Step 8: Evaluation

Evaluates VLM predictions against ground truth for LOW difficulty items.

```bash
# Evaluate all results for a specific tag
python inventory_pipeline/08_evaluate_vlm_count.py --tag qwen_v1

# Evaluate with custom output path
python inventory_pipeline/08_evaluate_vlm_count.py \
    --tag qwen_low \
    --output outputs/02_inventory/eval_reports/my_eval_report.json

# Evaluate blind mode results
python inventory_pipeline/08b_evaluate_vlm_blind.py --tag qwen_blind_v2
```

**Metrics:**
- **Mean Accuracy**: Fraction of segments with correct count prediction
- **Mean Absolute Error (MAE)**: Average |predicted - ground_truth|

**Note:** Segments with null ground truth or null predictions are **skipped** in evaluation (marked as `skipped: True` in output).

**Input:** `P*_vlm_qa_{tag}_results.json` files
**Output:** `eval_reports/vlm_qa_{tag}_count_eval_report.json`

### Step 9: Gather Wrong Predictions

Collects all wrong VLM predictions across participants for analysis in the visualizer.

```bash
# Gather wrong predictions for a VLM tag
python inventory_pipeline/09_gather_wrong_predictions.py --tag qwen_v1

# Include "close" matches (off by 1) as well as wrong
python inventory_pipeline/09_gather_wrong_predictions.py \
    --tag qwen_v1 \
    --include-close

# Filter to only "qa" files, excluding baseline
python inventory_pipeline/09_gather_wrong_predictions.py \
    --tag qwen \
    --exclude baseline

# Verbose output showing each wrong prediction
python inventory_pipeline/09_gather_wrong_predictions.py --tag qwen -v
```

**Output:** `prediction_analysis/wrong_predictions_{tag}.json`

This file is compatible with the food-inventory-visualizer and contains:
- All items with wrong predictions across participants
- Sorted by error magnitude (worst first)
- Participant prefix in food names: `[P04] green peas`
- Preserved `clip_path` for video playback
- Stats by participant and by source file

## Quick Run Script

For a new participant, run all steps in sequence:

```bash
PARTICIPANT=P04
TAG=qwen_v1

# Step 2: Discovery
python inventory_pipeline/02_inventory_discovery.py \
    --participant $PARTICIPANT --model gpt-5.2 --reasoning --reasoning-effort high

# PAUSE: Manually create and edit discovery_edit.json
cp outputs/02_inventory/$PARTICIPANT/${PARTICIPANT}_discovery.json \
   outputs/02_inventory/$PARTICIPANT/${PARTICIPANT}_discovery_edit.json
# Edit the file to verify/correct food items

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

# PAUSE: Manually create and edit timeline_annotated.json
cp outputs/02_inventory/$PARTICIPANT/${PARTICIPANT}_timeline_aggregated.json \
   outputs/02_inventory/$PARTICIPANT/${PARTICIPANT}_timeline_annotated.json
# Edit the file to verify timestamps and counts

# Step 7: VLM Q&A
python inventory_pipeline/07_vlm_QA.py \
    --participant $PARTICIPANT --tag $TAG --model qwen

# Step 8: Evaluation (runs automatically after step 7 unless --no-eval)
# To re-run evaluation manually:
python inventory_pipeline/08_evaluate_vlm_count.py --tag $TAG

# Step 9: Gather wrong predictions (after processing multiple participants)
python inventory_pipeline/09_gather_wrong_predictions.py --tag $TAG
```

## Processing Multiple Participants

To process all participants and get aggregate evaluation:

```bash
TAG=qwen_v1

# Run VLM Q&A on all participants (skipping those already processed)
python inventory_pipeline/07_vlm_QA.py --all --tag $TAG --skip-existing

# Generate aggregate evaluation report
python inventory_pipeline/08_evaluate_vlm_count.py --tag $TAG

# Gather all wrong predictions for analysis
python inventory_pipeline/09_gather_wrong_predictions.py --tag $TAG

# View results in visualizer
cd food-inventory-visualizer && npm start
```

## Failure Cases Workflow

For iterative VLM testing on difficult cases, use the failure cases workflow.

### Step 1: Initialize Failure Cases

Create a curated list from wrong predictions:

```bash
# Top 20 items by error magnitude
python inventory_pipeline/09c_init_failure_cases.py --tag qwen --top 20 --name curated

# Filter by minimum error
python inventory_pipeline/09c_init_failure_cases.py --tag qwen --min-error 3 --name high_error

# Filter by participant
python inventory_pipeline/09c_init_failure_cases.py --tag qwen --participant P03 --name p03_cases

# Filter by difficulty
python inventory_pipeline/09c_init_failure_cases.py --tag qwen --difficulty LOW --name low_only
```

**Output:** `outputs/02_inventory/failure_cases/failure_cases_{name}.json`

### Step 2: Edit Failure Cases (Optional)

Manually edit the failure cases file to:
- Set `"include": false` for cases to skip
- Add `"notes"` for interesting observations
- Adjust `"priority"` values (lower = higher priority)

```json
{
  "items": [
    {
      "case_id": "FC001",
      "food_name": "[P03] eggs",
      "include": true,          // Set to false to skip this case
      "priority": 1,            // Lower = higher priority
      "notes": "Partially occluded by hand",
      ...
    }
  ]
}
```

### Step 3: Run VLM on Failure Cases

Process the curated cases with a new VLM run:

```bash
python inventory_pipeline/07_vlm_QA.py \
    --failure-cases failure_cases_curated.json \
    --tag curated_v2 \
    --model qwen
```

**Output:** `outputs/02_inventory/failure_cases/failure_cases_curated_v2.json`

The script automatically versions the output (v2, v3, etc.) to preserve history.

### Step 4: View in Visualizer

```bash
cd food-inventory-visualizer && npm start
```

1. Click "Load from Server"
2. Select "Failure Cases" tab
3. Use the dropdown to select which failure cases file to view
4. Compare original predictions vs re-run results

### Failure Cases vs VLM Results

| Feature | VLM Results View | Failure Cases View |
|---------|-----------------|-------------------|
| Data source | `{P}_vlm_qa_{tag}_results.json` | `failure_cases_{name}.json` |
| Scope | Single participant | Cross-participant |
| Purpose | Full evaluation | Focused debugging |
| Items shown | All processed items | Curated wrong predictions |

## Output File Summary

### Per-Participant Files (in `outputs/02_inventory/{P}/`)

| Step | Output File | Description |
|------|-------------|-------------|
| 2 | `{P}_discovery.json` | Raw discovered food items |
| 2.5 | `{P}_discovery_edit.json` | Human-verified food items |
| 3 | `{P}_lifecycle.json` | Lifecycle events per item |
| 4 | `{P}_dispensal_classified.json` | Items with difficulty labels |
| 5 | `{P}_known_quantities.json` | Items with known counts + recipe matches |
| 5 | `{P}_known_quantities.txt` | Human-readable summary |
| 6 | `{P}_timeline_aggregated.json` | GPT-aggregated time segments (auto-padded to min 5s) |
| 6.5 | `{P}_timeline_annotated.json` | Human-verified timeline |
| 7 | `{P}_vlm_qa_{tag}_results.json` | VLM predictions with match evaluation |
| 7 | `vlm_clips/` | Extracted video clips for VLM |
| 8 | `{P}_vlm_qa_{tag}_results_count_eval.json` | Per-participant evaluation |

### Aggregate Files (in `outputs/02_inventory/`)

| Step | Output File | Description |
|------|-------------|-------------|
| 8 | `eval_reports/vlm_qa_{tag}_count_eval_report.json` | Aggregate evaluation across all participants |
| 8b | `eval_reports/vlm_qa_{tag}_blind_eval_report.json` | Blind mode evaluation |
| 9 | `prediction_analysis/wrong_predictions_{tag}.json` | All wrong predictions for visualizer |
| 9b | `failure_cases/failure_cases_{name}.json` | Curated failure cases (initial) |
| 9b | `failure_cases/failure_cases_{name}_v{N}.json` | VLM re-run results on failure cases |
| 9d | `prediction_analysis/prediction_flips_{a}_vs_{b}.json` | Prediction flip analysis |
| 9d | `failure_cases/failure_cases_diffs_{a}_vs_{b}.json` | Prediction diff failure cases |

## Understanding Null Counts

Items with **null counts** (uncountable items like liquids, powders) flow through the pipeline as follows:

| Pipeline Stage | Handling |
|---------------|----------|
| Step 6 | Created with `count: null` for continuous items |
| Step 7 | Still processed, VLM returns `continuous_estimate` or `gt_uncountable` match |
| Step 8 | **Skipped** in accuracy/MAE calculations (marked `skipped: True`) |
| Visualizer | Shown with "GT N/A" or "Pred N/A" badges |

## Notes

- Steps 2.5 and 6.5 require manual human review
- GPT-5.2 with high reasoning effort gives best results but is slower
- Use `--test N` flag to process only first N items for testing
- Videos must exist in `data/HD-EPIC/Videos/{participant}/` for VLM QA
- Segments shorter than 5 seconds are automatically padded in Step 6
- Use `--skip-existing` in Step 7 to avoid reprocessing completed participants

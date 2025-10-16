# Sequential Knowledge Graph Pipeline

## Overview

This pipeline processes egocentric kitchen narrations sequentially using a local Ollama LLM to build and maintain a Food Knowledge Graph. Unlike batch processing, each narration is processed in order, querying the current KG state before making updates.

## Why Sequential Processing?

**Critical Issue with Batch Processing**: Batch APIs require all requests to be created upfront, meaning each request queries the KG at its initial state (T=0). This causes:
- Duplicate food nodes (e.g., 6 separate "milk bottle" nodes instead of 1)
- Broken entity resolution (can't link interactions to same food item)
- Invalid temporal tracking

**Sequential Solution**: Process narrations one-by-one, updating the KG after each narration. This ensures proper entity resolution and temporal consistency.

## Architecture

```
For each narration in chronological order:
  1. Extract entities (food, location, action) from CSV
  2. Query current KG state for matching food items
  3. Build LLM prompt with KG context
  4. Call Ollama LLM to generate update command
  5. Execute KG update (create/update food node)
  6. Save KG snapshot for evaluation
```

## Pipeline Components

### Core Files

1. **`kg_sequential_pipeline.py`** - Main orchestrator
   - Reads narrations from CSV
   - Processes sequentially with Ollama
   - Manages KG updates and snapshots

2. **`kg_storage.py`** - KG data management
   - JSON persistence
   - CRUD operations for food/zone nodes
   - Query functions for entity resolution

3. **`entity_extractor.py`** - Deterministic entity extraction
   - Leverages pre-parsed CSV nouns/verbs
   - Identifies food items and locations
   - Filters out tools and appliances

4. **`llm_context.py`** - Prompt engineering
   - Builds prompts with current KG context
   - Few-shot examples for structured JSON output
   - Parses and validates LLM responses

5. **`kg_pipeline.py`** - KG update execution
   - `execute_kg_update()` function applies LLM commands
   - Handles food node creation and updates
   - Manages interaction history

6. **`kg_snapshots.py`** - Temporal snapshot management
   - Saves complete KG state after each narration
   - Enables evaluation at any point in time
   - Tracks metadata (success/failure, food count, etc.)

### Reference Files

- **`batch_ollama_csv_to_jsonl.py`** - Reference for Ollama client usage (read-only)
- **`KG.md`** - Knowledge graph schema and design
- **`BUGS_FOUND.md`** - Documentation of batch processing issues

## Usage

### Basic Usage

```bash
python kg_sequential_pipeline.py \
  --csv participant_P01_narrations.csv \
  --kg food_kg.json \
  --snapshots kg_snapshots \
  --model "gpt-oss:120b"
```

### Parameters

- `--csv`, `-c`: Path to narration CSV file (required)
- `--kg`, `-k`: Path to KG JSON file (default: `food_kg_sequential.json`)
- `--snapshots`: Directory for snapshots (default: `kg_snapshots`)
- `--model`, `-m`: Ollama model name (default: `qwen2.5:32b`)
- `--host`: Ollama host URL (default: `http://localhost:11434`)
- `--limit`, `-l`: Limit number of rows for testing
- `--start`, `-s`: Start row index (default: 0)
- `--verbose`, `-v`: Print detailed processing info
- `--save-interval`: Save KG every N rows (default: 10)

### Testing

```bash
# Test with first 20 narrations
python kg_sequential_pipeline.py \
  --csv participant_P01_narrations.csv \
  --limit 20 \
  --verbose

# Test specific range
python kg_sequential_pipeline.py \
  --csv participant_P01_narrations.csv \
  --start 100 \
  --limit 50
```

## Data Flow

### Input (CSV Columns)
- `unique_narration_id`: Unique identifier (e.g., "P01-20240202-110250-1")
- `video_id`: Video identifier
- `start_timestamp`, `end_timestamp`: Time range in seconds
- `narration`: Natural language description
- `nouns`, `verbs`, `main_actions`: Pre-parsed entities

### Output Files

1. **Knowledge Graph JSON** (`food_kg.json`)
   ```json
   {
     "zones": {
       "zone_fridge_1": {
         "zone_id": "zone_fridge_1",
         "name": "fridge",
         "type": "Storage"
       }
     },
     "foods": {
       "food_milk_bottle_1": {
         "food_id": "food_milk_bottle_1",
         "name": "milk bottle",
         "state": "closed",
         "location": "zone_fridge_1",
         "quantity": "1",
         "interaction_history": [...]
       }
     }
   }
   ```

2. **Snapshots** (`kg_snapshots/snapshot_{narration_id}.json`)
   - Complete KG state after each narration
   - Metadata: narration_id, time, success/failure, food counts

3. **Snapshot Metadata** (`kg_snapshots/snapshots_metadata.jsonl`)
   - One line per narration
   - Tracks processing status and KG statistics

## Performance

- **Speed**: ~0.5-2 rows/second (depends on LLM speed)
- **Full dataset**: 7392 narrations ≈ 1-4 hours
- **Memory**: Low (only current KG in memory)
- **Disk**: ~500KB per 100 snapshots

## Entity Resolution

The pipeline correctly resolves entities by:
1. **Prioritizing foods in hand** (location=null) for actions like "put", "place"
2. **Matching by name and location** for contextual disambiguation
3. **Sorting by most recent interaction** to prefer active foods

Example:
```
Row 3: "Pick up mug from shelf" → Creates food_mug_1
Row 4: "Turn mug in hand" → Updates food_mug_1 (found in hand)
Row 7: "Put mug under nozzle" → Updates food_mug_1 (location change)
```

## Error Handling

- **Invalid update types**: LLM sometimes returns actions like "TURN", "FLIP" - validation catches these and retries
- **Missing food entities**: Narrations without food are skipped and logged
- **LLM failures**: Retries up to 3 times, then saves failed snapshot
- **Periodic saves**: KG saved every N rows to prevent data loss

## Evaluation

Snapshots enable temporal evaluation:
- Query KG state at any timestamp
- Compare predicted vs. ground truth food locations
- Analyze interaction history accuracy
- Track food appearance/disappearance

## Next Steps

1. Run full 7392-row dataset
2. Analyze success/failure rates
3. Evaluate entity resolution accuracy
4. Compare with ground truth annotations

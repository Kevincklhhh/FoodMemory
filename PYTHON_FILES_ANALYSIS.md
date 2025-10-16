# Python Files Usage Analysis

## Currently Active Files (KEEP)

### Core Pipeline
1. **`kg_sequential_pipeline.py`** (13.9 KB) ⭐ **MAIN PIPELINE**
   - Entry point for sequential KG processing
   - Uses Ollama for LLM calls
   - Supports both keyword and LLM entity extraction
   - Status: **ACTIVE - PRIMARY FILE**

2. **`kg_storage.py`** (8.4 KB) ✓ **ESSENTIAL**
   - KG data management (CRUD operations)
   - JSON persistence
   - Functions: `load_kg`, `save_kg`, `find_food`, `add_food_node`, etc.
   - Used by: `kg_sequential_pipeline.py`, `kg_pipeline.py`
   - Status: **ACTIVE - CORE LIBRARY**

3. **`entity_extractor.py`** (7.5 KB) ✓ **ACTIVE**
   - Keyword-based entity extraction (fast, deterministic)
   - Default extraction method
   - Used by: `kg_sequential_pipeline.py`, `kg_pipeline.py`
   - Status: **ACTIVE - DEFAULT EXTRACTOR**

4. **`llm_entity_extractor.py`** (6.8 KB) ✓ **ACTIVE**
   - LLM-based entity extraction (accurate, slower)
   - Optional extraction method via `--entity-extraction llm`
   - Used by: `kg_sequential_pipeline.py`
   - Status: **ACTIVE - OPTIONAL EXTRACTOR**

5. **`llm_context.py`** (8.0 KB) ✓ **ESSENTIAL**
   - Prompt building for LLM
   - Response parsing and validation
   - Update type normalization (fixes invalid types)
   - Used by: `kg_sequential_pipeline.py`, `kg_pipeline.py`
   - Status: **ACTIVE - CORE LIBRARY**

6. **`kg_snapshots.py`** (9.5 KB) ✓ **ACTIVE**
   - Snapshot management for temporal evaluation
   - Saves KG state after each narration
   - Used by: `kg_sequential_pipeline.py`
   - Status: **ACTIVE - EVALUATION TOOL**

### Backend & Visualization
7. **`kg_visualizer_server.py`** (5.5 KB) ✓ **ACTIVE**
   - Flask backend for KG visualizer
   - Serves videos and snapshot data
   - Status: **ACTIVE - VISUALIZATION BACKEND**

## Partially Useful Files (REFACTOR)

8. **`kg_pipeline.py`** (10.4 KB) ⚠️ **PARTIALLY OBSOLETE**
   - **Used:** `execute_kg_update()` function (lines 137-241) - called by `kg_sequential_pipeline.py`
   - **Obsolete:**
     - `process_narration_row()` function (lines 25-134) - NOT USED
     - `main()` function (lines 244-325) - NOT USED (was for OpenAI API)
     - OpenAI-specific code
   - **Action:** Extract `execute_kg_update()` into a utility module, delete the rest
   - Status: **REFACTOR NEEDED**

## Obsolete Files (DELETE)

9. **`batch_ollama_csv_to_jsonl.py`** ❌ **REFERENCE ONLY**
   - Was used as reference for Ollama client patterns
   - Not part of KG pipeline
   - Status: **KEEP AS REFERENCE (different project)**

10. **`extract_participant.py`** ❌ **UNRELATED**
   - Not part of KG pipeline
   - Likely a preprocessing script
   - Status: **UNRELATED - NO ACTION**

## Summary Table

| File | Size | Status | Used By | Action |
|------|------|--------|---------|--------|
| `kg_sequential_pipeline.py` | 13.9 KB | ⭐ Active | - | **KEEP** |
| `kg_storage.py` | 8.4 KB | ✓ Active | Sequential pipeline | **KEEP** |
| `llm_context.py` | 8.0 KB | ✓ Active | Sequential pipeline | **KEEP** |
| `kg_snapshots.py` | 9.5 KB | ✓ Active | Sequential pipeline | **KEEP** |
| `entity_extractor.py` | 7.5 KB | ✓ Active | Sequential pipeline | **KEEP** |
| `llm_entity_extractor.py` | 6.8 KB | ✓ Active | Sequential pipeline (optional) | **KEEP** |
| `kg_visualizer_server.py` | 5.5 KB | ✓ Active | Visualizer frontend | **KEEP** |
| `kg_pipeline.py` | 10.4 KB | ⚠️ Partial | Sequential pipeline (1 function) | **REFACTOR** |
| `batch_ollama_csv_to_jsonl.py` | 23.2 KB | Reference | - | **KEEP** |
| `extract_participant.py` | 2.2 KB | Unrelated | - | **NO ACTION** |

## Recommendations

### Immediate Action: Refactor `kg_pipeline.py`

**Problem:** `kg_pipeline.py` has mixed content:
- **Used:** `execute_kg_update()` function (105 lines, 40% of file)
- **Unused:** `process_narration_row()`, `main()`, OpenAI code (219 lines, 60% of file)

**Solution:** Extract useful function into `kg_update_executor.py`

```python
# NEW FILE: kg_update_executor.py
# Contains only execute_kg_update() function
# Clean, focused, no legacy code
```

**Then update imports in `kg_sequential_pipeline.py`:**
```python
# Old: from kg_pipeline import execute_kg_update
# New: from kg_update_executor import execute_kg_update
```

**Then delete `kg_pipeline.py`**

### Benefits of Refactoring

1. **Cleaner codebase**: Remove 60% unused code
2. **Clear purpose**: Each file has one responsibility
3. **Easier maintenance**: No confusion about which code is active
4. **Better documentation**: File names match their actual use

## Implementation Plan

```bash
# Step 1: Create new file with just the useful function
# Extract execute_kg_update() from kg_pipeline.py -> kg_update_executor.py

# Step 2: Update import in kg_sequential_pipeline.py
# Change: from kg_pipeline import execute_kg_update
# To:     from kg_update_executor import execute_kg_update

# Step 3: Test that pipeline still works
python kg_sequential_pipeline.py --csv data.csv --limit 5

# Step 4: Delete obsolete file
rm kg_pipeline.py

# Step 5: Update documentation
```

## Dependency Graph

```
kg_sequential_pipeline.py (MAIN)
├── kg_storage.py (KG CRUD)
├── entity_extractor.py (keyword extraction)
├── llm_entity_extractor.py (LLM extraction)
├── llm_context.py (prompts & validation)
├── kg_snapshots.py (snapshots)
└── kg_update_executor.py (execute updates) [TO BE CREATED]

kg_visualizer_server.py (BACKEND)
└── (no dependencies on pipeline code)
```

## Files to Keep As-Is

- `batch_ollama_csv_to_jsonl.py` - Different project, useful reference
- `extract_participant.py` - Preprocessing script, not part of main pipeline
- All documentation `.md` files

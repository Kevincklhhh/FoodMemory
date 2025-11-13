# EPIC Kitchen VISOR - Quick Reference Guide

**Last Updated**: 2025-11-12  
**Full Analysis**: See `FILE_INVENTORY_AND_DEPENDENCIES.md`

---

## TL;DR: What Is This?

Epic-kitchen-visor is a **standalone food analysis pipeline** that:
- Extracts food items from VISOR annotations (158 videos, 32,645 food occurrences)
- Creates searchable food inventory index
- Generates analysis reports and benchmark datasets
- **Does NOT depend on main NeuroTrace pipeline**

---

## The 5-Step Pipeline

| Step | Script | Purpose | Time | Output | Optional? |
|------|--------|---------|------|--------|-----------|
| 1 | `1_extract_food_items.py` | Extract from annotations | 2-3 min | visor_food_items.json (404 MB) | ❌ REQUIRED |
| 2 | `2_create_food_segments.py` | Visual masks | ~1-2 sec/video | food_segments/ | ✅ Optional |
| 3 | `3_analyze_food_per_video.py` | Generate reports | ~1-2 sec | food_per_video.* | ❌ REQUIRED |
| 4 | `4_build_food_image_index.py` | Index images | 15-30 min | food_image_index.json (1.4 GB) | ❌ REQUIRED |
| 5 | `5_extract_wdtcf_food_items.py` | WDTCF extraction | ~1-2 sec | wdtcf_food_items.json (87 KB) | ✅ Optional |

**Essential pipeline: Steps 1 → 3 → 4** (Total: ~20-35 minutes)

---

## Critical Files to Keep

### Must Have (4.5 GB)
- `GroundTruth-SparseAnnotations/` (2.5 GB) - VISOR annotations
- `visor_food_items.json` (404 MB) - Extracted food items
- `food_image_index.json` (1.4 GB) - Image retrieval index
- `food_inventory_lookup.json` (17 MB) - Query lookup
- `epic_food_nouns_detailed.json` (63 KB) - Food classifications
- `EPIC_100_noun_classes_v2.csv` (100 KB) - Noun taxonomy

### Should Have (3.2 MB)
- `food_per_video.json` (2.8 MB) - Analysis data
- `food_abundance_analysis.json` (83 KB) - Risk metrics
- `wdtcf_food_items.json` (87 KB) - WDTCF data

### Nice to Have (Regenerable)
- `food_per_video.{csv,txt}` - Text reports
- `food_segments/` - Visual masks
- All other `.txt` and `.csv` files

---

## Scripts Overview

### Pipeline Scripts (Execute in order)
```bash
python3 1_extract_food_items.py      # Extract all food from VISOR
python3 3_analyze_food_per_video.py  # Generate analysis reports  
python3 4_build_food_image_index.py  # Build searchable image index
```

### Query & Analysis Tools
```bash
python3 query_food_inventory.py --food onion --limit 5
python3 analyze_food_abundance.py
python3 generate_benchmark_metadata.py --food yoghurt
```

### Informational Scripts
```bash
python3 list_visor_videos_per_participant.py  # List VISOR videos
python3 analyze_visor_coverage.py              # Coverage analysis
```

### Pre-run Utilities (Don't rerun)
- `classify_epic_food_nouns.py` - Already executed
- `classify_food_objects.py` - Already executed

---

## Key Statistics

| Metric | Value |
|--------|-------|
| Total videos | 158 |
| Videos with food | 154 |
| Total food occurrences | 32,645 |
| Food images indexed | 23,933 |
| Unique food classes | 127 |
| Most common food | water (4,080 occurrences) |
| Average foods per video | 6.6 items |

---

## File Dependencies

```
VISOR Annotations
├─ epic_food_nouns_detailed.json
├─ EPIC_100_noun_classes_v2.csv
└─> Step 1: Extract
    └─> visor_food_items.json
        ├─> Step 3: Analyze → food_per_video.*
        ├─> Step 4: Index → food_image_index.json
        │   └─> query_food_inventory.py
        └─> analyze_food_abundance.py

WDTCF_GT.json
└─> Step 5: WDTCF → wdtcf_food_items.json (standalone)
```

---

## Storage Breakdown

| Category | Size | Usage |
|----------|------|-------|
| VISOR annotations (source) | 2.5 GB | Required input |
| Extracted food items | 404 MB | Step 1 output |
| Food image index | 1.4 GB | Step 4 output |
| Image lookup table | 17 MB | Query tool |
| Analysis reports | 3 MB | Reference |
| Scripts & code | ~100 KB | Pipeline |
| **TOTAL CRITICAL** | **4.5 GB** | **Must keep** |

---

## About Step 5 (WDTCF)

**What is it?**
- Extract food from WDTCF (Where Did The Container First appear) annotations
- Different dataset: 74 videos vs 158 for VISOR
- Tracks where food was first stored in kitchen

**Is it needed?**
- ❌ NO - unless you need temporal storage information
- Not used by any other scripts
- Completely optional extraction

**Can be safely deleted?**
- ✅ YES - if not doing temporal analysis
- Minimal overhead (87 KB output)
- Can be regenerated anytime

---

## Documentation Files

| File | Purpose |
|------|---------|
| `WORKFLOW.md` | Complete pipeline documentation (read this first) |
| `README.txt` | VISOR dataset information |
| `UPDATE_SUMMARY.md` | Recent bug fixes and updates |
| `FILE_INVENTORY_AND_DEPENDENCIES.md` | Detailed analysis (this report) |
| `QUICK_REFERENCE.md` | This file |

---

## Recent Changes (2025-11-11)

**Bug Fixed in Step 4**: 35 videos were missing from the image index
- Root cause: Split detection logic error
- Fixed and regenerated all outputs
- Now: 154/158 videos (97.5% coverage)
- 4,342 previously missing images recovered

---

## Refactoring Notes

**Current Status**: Completely independent from main NeuroTrace pipeline

**For Integration**:
- Option 1 (Recommended): Keep as separate subsystem, call externally
- Option 2: Merge into main pipeline (requires path refactoring)

**Files Safe to Delete** (to save space):
- `classify_epic_food_nouns.py` (pre-run utility)
- `classify_food_objects.py` (pre-run utility)
- All `*_simple.txt` files (regenerable)
- `retrieve_samples/` (if not using benchmarks)
- `Interpolations-DenseAnnotations/` (if not using dense annotations)

---

## Common Tasks

### Extract & analyze food
```bash
python3 1_extract_food_items.py
python3 3_analyze_food_per_video.py
```

### Build searchable index
```bash
python3 4_build_food_image_index.py
# Then query:
python3 query_food_inventory.py --food onion --limit 10
```

### Get food per video
```bash
python3 3_analyze_food_per_video.py
cat food_per_video_simple.txt  # Simple format
cat food_per_video.txt         # Detailed format
```

### Analyze contamination risk
```bash
python3 analyze_food_abundance.py
# Output: food_abundance_analysis.json
```

### Create benchmark metadata
```bash
python3 generate_benchmark_metadata.py --food yoghurt
```

---

## Contact/Questions

See `WORKFLOW.md` for detailed documentation and examples.


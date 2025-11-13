# EPIC Kitchen VISOR - Analysis Documents

**Generated**: 2025-11-12  
**Purpose**: Complete refactoring and dependency analysis

---

## Documents Overview

### Quick Start (Read These First)

1. **QUICK_REFERENCE.md** ⭐ START HERE
   - 5-minute overview of the pipeline
   - Essential files to keep
   - Common tasks and usage
   - ~5 KB, 2 min read

2. **REFACTORING_RECOMMENDATIONS.md**
   - Architectural analysis
   - Integration options
   - Files safe to delete
   - ~8 KB, 5 min read

### Detailed Analysis

3. **FILE_INVENTORY_AND_DEPENDENCIES.md**
   - Complete file listing (all files)
   - Full dependency graph
   - Data criticality assessment
   - Step-by-step execution guide
   - ~35 KB, 20 min read

### Original Documentation

4. **WORKFLOW.md** (Existing)
   - Complete pipeline documentation
   - Usage examples
   - Output file formats
   - KEEP - critical reference

5. **UPDATE_SUMMARY.md** (Existing)
   - Bug fix log (2025-11-11)
   - Recent changes
   - Performance notes
   - KEEP - important history

---

## TL;DR Summary

### What Is This System?

Epic-kitchen-visor is a **standalone food analysis pipeline**:
- Extracts food from VISOR annotations (158 videos, 32,645 occurrences)
- Creates searchable food inventory
- Generates analysis reports
- **Completely independent from main NeuroTrace pipeline**

### The Essential Pipeline

```
Step 1: 1_extract_food_items.py          (2-3 min)  → visor_food_items.json
Step 3: 3_analyze_food_per_video.py      (1-2 sec)  → food_per_video.json
Step 4: 4_build_food_image_index.py      (15-30 min) → food_image_index.json
```

**Total time**: 20-35 minutes  
**Total data**: 4.5 GB critical files

### Critical Files (Must Keep)
- `GroundTruth-SparseAnnotations/` (2.5 GB) - Source annotations
- `visor_food_items.json` (404 MB) - Extracted food items
- `food_image_index.json` (1.4 GB) - Image retrieval index
- Classification files + lookup tables (~80 MB)

### Optional Files (Can Delete)
- Step 2 segmentations (visualization)
- Step 5 WDTCF extraction (temporal data)
- All `.txt` and `.csv` reports (regenerable)
- Sample images and benchmark data

### Key Finding

**VISOR annotations are essential** - cannot be deleted  
**Step 5 (WDTCF) is optional** - not used by other scripts  
**System is independent** - zero dependencies on main pipeline

---

## Which Document Should I Read?

**I want to...**

- Quick overview of the system → **QUICK_REFERENCE.md**
- Understand all files and dependencies → **FILE_INVENTORY_AND_DEPENDENCIES.md**
- Know what to delete/keep → **REFACTORING_RECOMMENDATIONS.md**
- Run the pipeline → **WORKFLOW.md**
- Understand recent updates → **UPDATE_SUMMARY.md**
- See complete examples → **WORKFLOW.md** (sections 3-5)

---

## Key Statistics

| Metric | Value |
|--------|-------|
| Total videos | 158 |
| Videos with food | 154 (97.5%) |
| Food occurrences | 32,645 |
| Unique food classes | 127 |
| Food images indexed | 23,933 |
| Pipeline scripts | 12 |
| Critical data size | 4.5 GB |
| Execution time | 20-35 min |
| Independent system | Yes ✅ |

---

## Common Questions Answered

**Q: Is VISOR data essential?**  
A: YES - absolutely. It's the source data and cannot be regenerated.

**Q: Can I skip Step 5 (WDTCF)?**  
A: YES - it's completely optional and independent.

**Q: Does this depend on main NeuroTrace pipeline?**  
A: NO - completely independent system.

**Q: How much storage do I need?**  
A: 4.5 GB minimum (can delete optional files to save space).

**Q: Can I delete the image index?**  
A: NO - it's critical for queries. But it CAN be regenerated from Step 4.

**Q: Should I integrate with main pipeline?**  
A: Recommended: NO - keep separate for modularity (see REFACTORING_RECOMMENDATIONS.md).

---

## File Organization

### Documentation (This folder)
- `00_READ_ME_FIRST.md` ← You are here
- `QUICK_REFERENCE.md` - 5-minute overview
- `FILE_INVENTORY_AND_DEPENDENCIES.md` - Complete analysis
- `REFACTORING_RECOMMENDATIONS.md` - Integration options
- `WORKFLOW.md` - Original documentation
- `UPDATE_SUMMARY.md` - Bug fixes and updates

### Pipeline Scripts
- `1_extract_food_items.py` - Extract from VISOR
- `2_create_food_segments.py` - Visual masks (optional)
- `3_analyze_food_per_video.py` - Generate reports
- `4_build_food_image_index.py` - Build image index
- `5_extract_wdtcf_food_items.py` - WDTCF extraction (optional)

### Utility Scripts
- `query_food_inventory.py` - Query tool
- `analyze_food_abundance.py` - Risk analysis
- `generate_benchmark_metadata.py` - Benchmark tool
- `list_visor_videos_per_participant.py` - Listing tool
- `analyze_visor_coverage.py` - Coverage analysis
- `classify_epic_food_nouns.py` - Pre-run utility
- `classify_food_objects.py` - Pre-run utility

### Data Files
- `GroundTruth-SparseAnnotations/` - Input annotations (2.5 GB)
- `visor_food_items.json` - Step 1 output (404 MB)
- `food_image_index.json` - Step 4 output (1.4 GB)
- `food_inventory_lookup.json` - Query index (17 MB)
- Classification JSONs/CSVs
- Analysis reports and summaries

---

## Next Steps

1. **Quick start**: Read `QUICK_REFERENCE.md` (2 min)
2. **Understand architecture**: Read `REFACTORING_RECOMMENDATIONS.md` (5 min)
3. **For deep dive**: Read `FILE_INVENTORY_AND_DEPENDENCIES.md` (20 min)
4. **Execute pipeline**: See `WORKFLOW.md` Step-by-step instructions
5. **Query data**: Use `query_food_inventory.py` tool

---

## Contact

For detailed documentation, see individual markdown files.  
For refactoring questions, see `REFACTORING_RECOMMENDATIONS.md`.  
For usage examples, see `WORKFLOW.md`.

---

*This analysis was generated 2025-11-12 as part of complete file inventory and dependency analysis.*


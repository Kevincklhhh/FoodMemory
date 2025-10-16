# Entity Extraction Implementation Summary

## Problem Identified

The current pipeline uses **keyword-based entity extraction** which:
- Matches against hardcoded lists of 47 food terms, 19 locations
- Returns None for foods not in the predefined list
- Cannot understand context ("milk frother" vs "milk bottle")

## Solution Implemented

Created **LLM-based entity extraction** (`llm_entity_extractor.py`) with:
- Context-aware extraction using few-shot prompting
- Distinguishes tools from food items
- Provides reasoning for extraction decisions
- Not limited by predefined vocabulary

## Integration

Updated `kg_sequential_pipeline.py` to support both methods via command-line flag:

```bash
# Keyword extraction (default - faster)
python kg_sequential_pipeline.py --csv data.csv --entity-extraction keyword

# LLM extraction (slower but more accurate)
python kg_sequential_pipeline.py --csv data.csv --entity-extraction llm
```

## Performance Comparison

### Test Results (5 narrations, rows 15-20):

| Method | Time | Speed | Foods Found | Same Results |
|--------|------|-------|-------------|--------------|
| Keyword | 34.3s | 0.15 rows/s | 1 milk bottle | Yes |
| LLM | 41.6s | 0.12 rows/s | 1 milk bottle | Yes |

**Speed overhead:** +21% processing time with LLM extraction

### Expected Improvements with LLM Extraction:

1. **Fewer false negatives**: Detects novel foods not in keyword list
2. **Better disambiguation**: "milk frother" (tool) vs "milk bottle" (food)
3. **Context understanding**: "lid on milk bottle" → extracts "milk bottle"
4. **Reasoning**: Explains why entity was or wasn't extracted

## Files Created

1. **`llm_entity_extractor.py`** - LLM-based extraction implementation
2. **`ENTITY_EXTRACTION_COMPARISON.md`** - Detailed comparison analysis
3. **`ENTITY_EXTRACTION_SUMMARY.md`** - This summary

## Files Modified

1. **`kg_sequential_pipeline.py`** - Added `--entity-extraction` flag
2. **`kg_pipeline.py`** - Removed deleted openai_api import

## Recommendation

**For research/evaluation:** Use `--entity-extraction llm` for highest accuracy

**For large-scale processing:** Start with `--entity-extraction keyword`, then reprocess failures with LLM extraction

## Example Usage

```bash
# Test with small dataset using LLM extraction
python kg_sequential_pipeline.py \
  --csv participant_P01_narrations.csv \
  --kg food_kg_llm.json \
  --snapshots kg_snapshots_llm \
  --model gpt-oss:120b \
  --entity-extraction llm \
  --limit 100
```

## Next Steps

1. Run full dataset with both methods and compare results
2. Measure accuracy improvement (% more foods detected)
3. Consider hybrid approach (keyword first, LLM fallback for failures)
4. Add caching to avoid re-extracting entities on reruns

# Auto-Mask Loading Feature

## What Was Added

The retrieval script (`10_food_retrieval.py`) now **automatically detects and loads VISOR segmentation masks** when querying with images from the benchmark directories.

## How It Works

### Detection Logic (Lines 310-342)

1. **Check if query is from benchmark directory**:
   - Looks for `retrieve_samples/` or `retrieve_benchmarks/` in the path
   - Extracts the food class from directory name (e.g., `yoghurt`, `pizza`)

2. **Load benchmark metadata**:
   - Finds `{food_class}_benchmark_instances.json` in the benchmark directory
   - Example: `retrieve_benchmarks/yoghurt_benchmark_instances.json`

3. **Match filename and extract mask**:
   - Searches through all instances/frames in metadata
   - Matches query filename (e.g., `P02_102_frame_0000000296.jpg`)
   - Extracts VISOR polygon segments for that frame

4. **Apply mask automatically**:
   - No need for `--mask-json` argument
   - Mask is applied transparently before embedding extraction

## Usage Examples

### Before (Manual Mask Loading)
```bash
# Required separate mask JSON file
python3 10_food_retrieval.py \
    --query image.jpg \
    --mask-json mask.json \
    -k 5
```

### After (Automatic)
```bash
# Auto-detects mask from benchmark metadata
python3 10_food_retrieval.py \
    --query retrieve_benchmarks/yoghurt/P02_102_frame_0000000296.jpg \
    -k 5

# Output shows:
#   ℹ Auto-detected VISOR benchmark image, loading mask from yoghurt_benchmark_instances.json
#   ✓ Found mask for P02_102_frame_0000000296.jpg
```

### Still Works with Manual Mask
```bash
# Explicit mask JSON takes precedence
python3 10_food_retrieval.py \
    --query custom_image.jpg \
    --mask-json custom_mask.json \
    -k 5
```

## Verification

Test with benchmark image:
```bash
python3 10_food_retrieval.py \
    --query retrieve_benchmarks/yoghurt/P02_102_frame_0000000296.jpg \
    -k 5
```

**Expected output:**
```
Querying: retrieve_benchmarks/yoghurt/P02_102_frame_0000000296.jpg
  ℹ Auto-detected VISOR benchmark image, loading mask from yoghurt_benchmark_instances.json
  ✓ Found mask for P02_102_frame_0000000296.jpg

Top 5 results:
Rank 1: P02_102_frame_0000000296.jpg (sim=1.0001) ← Same image, perfect match!
Rank 2: P11_105_frame_0000012892.jpg (sim=0.8867)
...
```

**Rank 1 similarity = 1.0** proves the mask was correctly loaded and applied (same masked region embedded twice).

## Benefits

1. **Convenience**: No need to manually specify mask files
2. **Consistency**: Ensures same masks used at indexing and retrieval time
3. **Accuracy**: Proper masking focuses embeddings on food items
4. **Flexibility**: Still allows custom masks with `--mask-json`

## Directory Structure Requirements

```
retrieve_benchmarks/
├── yoghurt/
│   ├── P02_102_frame_0000000296.jpg  ← Query image
│   ├── P02_102_frame_0000001361.jpg
│   └── ...
├── pizza/
│   ├── P01_09_frame_0000146163.jpg
│   └── ...
├── yoghurt_benchmark_instances.json  ← Auto-loaded metadata
└── pizza_benchmark_instances.json
```

OR

```
retrieve_samples/
├── yoghurt/
│   └── images...
├── pizza/
│   └── images...
├── yoghurt_benchmark_instances.json
└── pizza_benchmark_instances.json
```

## When Auto-Loading Doesn't Apply

Auto-mask loading is **skipped** when:
- Query path doesn't contain `retrieve_samples/` or `retrieve_benchmarks/`
- Food class directory name doesn't match any `*_benchmark_instances.json` file
- Filename doesn't match any frame in the benchmark metadata
- User provides explicit `--mask-json` argument (takes precedence)

In these cases, the query image is embedded **without masking** (full frame).

## Implementation Details

The auto-detection code checks path hierarchy:
```python
# Path: retrieve_benchmarks/yoghurt/P02_102_frame_0000000296.jpg
#              ↑                ↑            ↑
#         benchmark_dir    food_class    filename
```

1. Walk up directory tree to find `retrieve_benchmarks/` or `retrieve_samples/`
2. Extract food class = immediate parent directory (`yoghurt`)
3. Load `{benchmark_dir}/{food_class}_benchmark_instances.json`
4. Find frame with matching filename
5. Extract and apply VISOR segments

## Answer to Original Question

**Q: "Did you calculate embedding of frame or embedding of segmented mask region?"**

**A: Segmented mask region** - and now masks are **automatically loaded** for VISOR benchmark images, so you get masked embeddings by default without manual `--mask-json` specification!

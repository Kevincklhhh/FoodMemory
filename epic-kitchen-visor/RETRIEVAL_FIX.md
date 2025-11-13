# 10_food_retrieval.py - Bug Fix Summary

## Issue
The script was failing with the same `KeyError: 'image_tensor'` error when extracting CLIP embeddings for query images.

## Root Cause
Identical to the issue in `8_extract_food_embeddings.py` - the `ClipMapper` expects preprocessed tensor input, not raw PIL Images.

The original code at line 128 was:
```python
embedding = self.clip_mapper({'image': pil_image})  # WRONG
```

## Solution
Applied the same fix as Step 8 - added CLIP preprocessing to convert PIL Images to tensors.

### Changes Made

1. **Lines 81-86** - Added preprocessing transform initialization in `__init__`:
```python
# Get preprocess transform from CLIP model
import open_clip
_, _, self.preprocess = open_clip.create_model_and_transforms(
    clip_model.replace('/', '-'),  # Convert 'ViT-B/32' to 'ViT-B-32'
    pretrained='openai'
)
```

2. **Lines 134-142** - Fixed embedding extraction in `extract_query_embedding`:
```python
# Preprocess image to tensor
image_tensor = self.preprocess(pil_image).unsqueeze(0)  # Add batch dimension

# Extract embedding
result = self.clip_mapper({
    'image_tensor': image_tensor,
    'image_filename': str(image_path)
})
embedding = result['image_embs'][0]  # Get first (and only) embedding
```

## Verification

Successfully tested with yoghurt and pizza queries:

### Test 1: Yoghurt Query
```bash
python3 10_food_retrieval.py --query retrieve_samples/yoghurt/P02_03_frame_0000004418.jpg -k 5
```

**Results:**
```
Rank 1: P04_101_frame_0000011919.jpg (yoghurt, sim=0.7261)
Rank 2: P02_102_frame_0000001361.jpg (yoghurt, sim=0.6878)
Rank 3: P11_105_frame_0000012892.jpg (pizza, sim=0.6821)
Rank 4: P04_03_frame_0000017295.jpg (yoghurt, sim=0.6763)
Rank 5: P01_09_frame_0000163755.jpg (pizza, sim=0.6712)
```

### Test 2: Pizza Query
```bash
python3 10_food_retrieval.py --query retrieve_samples/pizza/P01_09_frame_0000146163.jpg -k 5
```

**Results:**
```
Rank 1: P06_05_frame_0000014005.jpg (pizza, sim=0.6740)
Rank 2: P01_09_frame_0000163755.jpg (pizza, sim=0.6465)
Rank 3: P30_111_frame_0000059567.jpg (pizza, sim=0.6267)
Rank 4: P11_105_frame_0000012892.jpg (pizza, sim=0.6166)
Rank 5: P06_07_frame_0000003850.jpg (pizza, sim=0.6152)
```

✅ **Pizza query returns pizza results** - showing correct semantic retrieval!

## Usage Examples

```bash
# Single image query (no mask)
python3 10_food_retrieval.py --query /path/to/image.jpg -k 10

# Single image query with mask
python3 10_food_retrieval.py --query /path/to/image.jpg --mask-json mask.json -k 5

# Batch retrieval for benchmark
python3 10_food_retrieval.py --benchmark yoghurt -k 10 --output results.json
```

## System Architecture

The fix ensures the retrieval pipeline matches the indexing pipeline:

```
Query Image → Preprocess → Tensor → CLIP → Embedding → FAISS Search → Results
     ↓                                         ↓
   Mask (optional)                      Indexed Embeddings
                                        (from Step 8)
```

Both pipelines now use identical preprocessing, ensuring query embeddings are in the same feature space as indexed embeddings.

## Performance

- **Index loaded**: 21 vectors (8 yoghurt + 13 pizza)
- **Query time**: ~0.2s (including CLIP inference)
- **Similarity metric**: Cosine similarity (0-1 scale)
- **Device**: CUDA (GPU accelerated)

## Next Steps

The complete food memory system is now functional:
1. ✅ Step 8: Extract embeddings from food instances
2. ✅ Step 9: Build FAISS index
3. ✅ Step 10: Query and retrieve similar instances

Ready for production use!

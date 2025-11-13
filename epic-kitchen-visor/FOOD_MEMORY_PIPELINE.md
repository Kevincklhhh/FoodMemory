# Food Memory System Pipeline

Complete pipeline for building and querying a visual food memory database.

## Overview

This pipeline implements the 3-module system from `FoodMemory_System_Design.md`:
1. **Logging Module** (Step 8) - Extract visual embeddings and VLM captions
2. **Memory Module** (Step 9) - Build FAISS index for similarity search
3. **Retrieval Module** (Step 10) - Query interface for visual similarity

## Data Flow

```
retrieve_benchmarks/{food}/
    ├── *.jpg (frames)
    └── {food}_benchmark_instances.json (metadata with masks)
         ↓
    [Step 8: Extract Embeddings]
         ↓
memory_database/
    ├── embeddings/
    │   ├── food_embeddings.npy (N x 512 CLIP vectors)
    │   └── food_metadata.json (includes captions)
    └── captions/
        └── vlm_captions.json (GPT-4o descriptions)
         ↓
    [Step 9: Build Index]
         ↓
memory_database/index/
    ├── memory_index.faiss (visual similarity index)
    ├── memory_metadata.parquet (metadata lookup)
    └── index_info.json (index configuration)
         ↓
    [Step 10: Query]
         ↓
Ranked results (by visual similarity)
```

## Pipeline Usage

### Prerequisites

```bash
# Install dependencies
pip install torch torchvision opencv-python pillow open-clip-torch
pip install faiss-cpu autofaiss pandas pyarrow

# Ensure you have:
# - clip-retrieval/ directory with ClipMapper
# - llm-api/ directory with OpenAIAPI
```

### Step 8: Extract Embeddings and Captions

Extract CLIP visual embeddings and GPT-4o captions from benchmark frames.

```bash
# Process single food class
python3 8_extract_food_embeddings.py --food yoghurt

# Process multiple food classes
python3 8_extract_food_embeddings.py --food yoghurt pizza

# Skip VLM captions (faster, visual-only)
python3 8_extract_food_embeddings.py --food yoghurt --skip-captions

# Custom batch size
python3 8_extract_food_embeddings.py --food yoghurt --batch-size 64
```

**Inputs:**
- `retrieve_benchmarks/{food}/` - Frame images
- `retrieve_benchmarks/{food}_benchmark_instances.json` - Instance metadata with VISOR masks

**Outputs:**
- `memory_database/embeddings/food_embeddings.npy` - CLIP ViT-B/32 embeddings (N x 512)
- `memory_database/embeddings/food_metadata.json` - Frame metadata including captions
- `memory_database/captions/vlm_captions.json` - VLM captions only (if not skipped)

**What it does:**
1. Load frames and VISOR segmentation masks from benchmark metadata
2. Apply masks to isolate food items (black out background)
3. Extract CLIP embeddings from masked images
4. Generate GPT-4o captions from original images with green overlay
5. Save embeddings, captions, and metadata

### Step 9: Build FAISS Index

Build searchable FAISS index from extracted embeddings.

```bash
# Default configuration
python3 9_build_memory_index.py

# Custom paths
python3 9_build_memory_index.py \
    --input memory_database/embeddings \
    --output memory_database/index

# Adjust memory limits
python3 9_build_memory_index.py \
    --max-index-memory 8G \
    --available-memory 32G
```

**Inputs:**
- `memory_database/embeddings/food_embeddings.npy`
- `memory_database/embeddings/food_metadata.json`

**Outputs:**
- `memory_database/index/memory_index.faiss` - Searchable index
- `memory_database/index/memory_metadata.parquet` - Metadata for lookups
- `memory_database/index/index_info.json` - Index configuration

**What it does:**
1. Load embeddings and metadata
2. L2-normalize embeddings (for cosine similarity via inner product)
3. Build FAISS index using autofaiss
4. Create parquet metadata mapping for fast retrieval
5. Verify index correctness

### Step 10: Query the Index

Retrieve visually similar food instances.

#### Single Image Query

```bash
# Query with mask
python3 10_food_retrieval.py \
    --query /path/to/query.jpg \
    --mask-json mask.json \
    -k 10

# Query without mask (uses full image)
python3 10_food_retrieval.py \
    --query /path/to/query.jpg \
    -k 5
```

**Output:** Ranked list of top-k similar frames with metadata

#### Batch Benchmark Query

```bash
# Query all frames in a benchmark
python3 10_food_retrieval.py \
    --benchmark yoghurt \
    -k 10

# Save results to file
python3 10_food_retrieval.py \
    --benchmark yoghurt \
    -k 20 \
    --output yoghurt_retrieval_results.json
```

**Output:**
- Console: Summary with top-5 results per query, marked with ✓ if same instance
- JSON file: Complete results for all queries (if --output specified)

**What it does:**
1. Load FAISS index and metadata
2. For each query:
   - Apply mask if provided
   - Extract CLIP embedding (same process as logging)
   - Search index via k-NN
   - Return ranked results with instance_id and metadata

## Architecture Details

### Logging Module (Step 8)

**Visual Embeddings:**
- Model: CLIP ViT-B/32 (512-dim)
- Preprocessing: Black out background using VISOR polygon masks
- Normalization: Handled by ClipMapper

**VLM Captions:**
- Model: GPT-4o
- Input: Original image with green transparent overlay on food
- Prompt: Describes food appearance, state, and kitchen context
- Output: 2-3 sentence description
- Storage: Saved in metadata but NOT used for retrieval

### Memory Module (Step 9)

**Index Construction:**
- Normalization: L2-normalize embeddings for cosine similarity
- Index type: Determined by autofaiss (based on dataset size)
- Metric: Inner product (on normalized vectors = cosine similarity)
- Memory mapping: Enabled for efficient large-scale loading

**Metadata Storage:**
- Format: Parquet for fast columnar access
- Fields: instance_id, frame_id, filename, food_class, semantic_label, video metadata

### Retrieval Module (Step 10)

**Search Process:**
1. Query embedding extracted with identical masking as logging
2. FAISS k-NN search using inner product (cosine similarity)
3. Metadata lookup from parquet file
4. Results include: rank, similarity score, instance_id, filename, source video info

**Evaluation Support:**
- Batch mode processes all frames in a benchmark
- Output includes same-instance markers (✓) for manual evaluation
- Ready for future automated evaluation scripts

## Example Workflow

```bash
# 1. Extract embeddings for yoghurt (with captions)
python3 8_extract_food_embeddings.py --food yoghurt

# 2. Build FAISS index
python3 9_build_memory_index.py

# 3. Test retrieval on yoghurt benchmark
python3 10_food_retrieval.py \
    --benchmark yoghurt \
    -k 10 \
    --output yoghurt_results.json

# 4. Add more food classes
python3 8_extract_food_embeddings.py --food pizza oil

# 5. Rebuild index with all food
python3 9_build_memory_index.py

# 6. Query again
python3 10_food_retrieval.py --benchmark yoghurt -k 10
```

## Design Decisions

### Why CLIP ViT-B/32?
- Proven for visual similarity tasks
- Reasonable embedding size (512-dim)
- Fast inference
- Existing integration via clip-retrieval

### Why Black Out Background?
- Forces CLIP to encode only the food item
- Reduces scene context bias
- Consistent with food re-identification goal

### Why Store but Not Use Captions?
- Primary task is visual re-identification
- Captions provide human-interpretable context
- Future work could explore multi-modal retrieval

### Why FAISS?
- Industry-standard vector search
- Scales to millions of embeddings
- Supports exact and approximate k-NN
- Memory-mappable for large datasets

### Why Separate Caption Storage?
- Keeps metadata JSON focused on retrieval
- Easy to toggle caption generation (--skip-captions)
- Allows future multi-modal extensions

## Performance Notes

**Step 8 (Embedding Extraction):**
- CLIP: ~10-50 images/sec (GPU-dependent)
- VLM captions: ~1-2 images/sec (API rate limits)
- Use `--skip-captions` for faster visual-only extraction

**Step 9 (Index Building):**
- Fast for <100k embeddings (<1 minute)
- autofaiss optimizes index type based on size

**Step 10 (Retrieval):**
- k-NN search: <10ms per query (typical)
- Batch mode: Limited by frame loading, not search

## Future Extensions

1. **Role Assignment**: Add script to randomly assign query/evidence/distractor roles
2. **Automated Evaluation**: Compute retrieval metrics (mAP, Recall@K)
3. **Multi-Modal Retrieval**: Explore text + visual retrieval using stored captions
4. **Cross-Food Evaluation**: Test if pizza embeddings retrieve yoghurt (should not)
5. **Temporal Consistency**: Add video-level instance tracking

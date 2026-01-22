# Food Memory Tracker System

A modular system for tracking food items and interactions using egocentric videos and Vision-Language Models (VLMs).

## System Overview

The Food Memory Tracker processes egocentric videos chronologically to build and maintain a persistent memory of food items, their locations, quantities, and interaction history.

### Architecture

The system consists of 9 modular components:

1. **FoodMemory** (`food_memory.py`) - Persistent storage with full node structure
2. **ReceiptParser** (`receipt_parser.py`) - Parses receipt text files for un-bagging scenarios
3. **VideoProcessor** (`video_processor.py`) - Splits videos into 30-second clips
4. **VLMPerception** (`vlm_perception.py`) - VLM Call 1: Context-free food detection
5. **MemoryRetriever** (`memory_retriever.py`) - Context-aware memory search (OPTIMIZED)
6. **VLMUpdate** (`vlm_update.py`) - VLM Call 2: Context-aware interaction analysis
7. **MemoryUpdater** (`memory_updater.py`) - Executes CREATE/UPDATE commands
8. **MemorySnapshotter** (`memory_snapshotter.py`) - Saves memory snapshots after each video
9. **MainOrchestrator** (`main_orchestrator.py`) - Coordinates the entire pipeline

### Pipeline Flow

```
[Video Files + Receipt Files]
       |
       v
[1. Sort Chronologically]
       |
       v
For each video:
  [2. Check for Receipt] --> (Un-bagging mode if receipt exists)
       |
       v
  [3. Split into 30s Clips]
       |
       v
  For each clip:
    [4. VLM Call 1: Detect Food Items] (context-free)
         |
         v
    [5. Retrieve Memory Context] (OPTIMIZED: smart filtering)
         |
         v
    [6. VLM Call 2: Analyze with Context] (+ receipt expectations if un-bagging)
         |
         v
    [7. Execute UPDATE/CREATE Commands]
       |
       v
  [8. Save Memory State]
       |
       v
  [9. Create Snapshot]
```

## Data Structures

### FoodItem Node
```python
{
  "food_id": "milk_abc12345",
  "primary_label": "Kroger Whole Milk Gallon",
  "description": "Gallon of whole milk",
  "visual_embedding": [0.1, 0.2, 0.3, ...],  # Full embedding vector
  "current_location": "fridge",
  "current_quantity": "half-full",
  "interaction_history": [
    {
      "timestamp": "2025-10-29T19:37:37",
      "action": "purchased and un-bagged"
    },
    {
      "timestamp": "2025-10-29T23:30:28",
      "action": "retrieve from fridge"
    }
  ]
}
```

### Receipt Format
Simple text file with one item per line:
```
Fresh Bunch of Organic Bananas
Garlic
Kroger® Peeled & Deveined Tail Off White Raw Shrimp
Kroger® Vitamin D Whole Milk Gallon
Simple Truth Organic® Firm Tofu
```

Receipt files should be named: `receipt_<video_timestamp>.txt`

### Video File Format
Videos should be named with timestamps: `yyyymmdd-hhmmss.MOV`

Example: `20251029-193737.MOV`

## Installation

### Prerequisites

1. **Python 3.8+**
2. **OpenCV**: `pip install opencv-python`
3. **Requests**: `pip install requests`
4. **FFmpeg**: System package (for video processing)

### VLM Server

The system requires a running Qwen3-VL vLLM server:

```bash
# Default endpoint: http://localhost:8000/v1/chat/completions
# See VLLM_QWEN3VL_DEPLOYMENT_GUIDE.md for setup instructions
```

## Usage

### Quick Start

1. **Prepare your data:**
   ```bash
   cd /path/to/videos/

   # Rename videos by timestamp (if needed)
   python rename_by_timestamp.py --execute

   # Create receipt files for un-bagging videos
   echo "Milk\nBread\nEggs" > receipt_20251029-193737.txt
   ```

2. **Run the pipeline:**
   ```bash
   python main_orchestrator.py
   ```

### Command-Line Options

```bash
# Process videos in specific directory
python main_orchestrator.py --input-dir /path/to/videos

# Custom memory file location
python main_orchestrator.py --memory-file /path/to/food_memory.json

# Custom snapshot directory
python main_orchestrator.py --snapshot-dir /path/to/snapshots

# Custom VLM endpoint
python main_orchestrator.py --vlm-url http://192.168.1.100:8000/v1/chat/completions

# Different model and sampling rate
python main_orchestrator.py --vlm-model Qwen3-VL-30B --vlm-fps 4

# Full example
python main_orchestrator.py \
  --input-dir ./videos \
  --memory-file food_memory.json \
  --snapshot-dir snapshots \
  --clip-dir clips \
  --vlm-url http://localhost:8000/v1/chat/completions \
  --vlm-model Qwen3-VL-30B \
  --vlm-fps 2
```

### Testing Individual Modules

Each module can be tested independently:

```bash
# Test FoodMemory
python food_memory.py

# Test Receipt Parser
python receipt_parser.py

# Test Video Processor
python video_processor.py

# Test VLM Perception (requires VLM server + test clip)
python vlm_perception.py

# Test Memory Retriever
python memory_retriever.py

# Test VLM Update (requires VLM server + test clip)
python vlm_update.py

# Test Memory Updater
python memory_updater.py

# Test Memory Snapshotter
python memory_snapshotter.py
```

## Output Files

### Generated Files

```
project_directory/
├── food_memory.json          # Main persistent memory
├── snapshots/                # Memory snapshots after each video
│   ├── snapshot_20251029-193737.json
│   ├── snapshot_20251029-233028.json
│   └── ...
├── clips/                    # Extracted 30-second video clips
│   ├── 20251029-193737_clip_000.mp4
│   ├── 20251029-193737_clip_001.mp4
│   └── ...
└── videos/                   # Your input files
    ├── 20251029-193737.MOV
    ├── receipt_20251029-193737.txt
    └── ...
```

### Snapshot Format

```json
{
  "metadata": {
    "video_name": "20251029-193737.MOV",
    "video_timestamp": "20251029-193737",
    "snapshot_time": "2025-10-30T04:30:00",
    "total_items": 15
  },
  "memory": {
    "items": {
      "milk_abc12345": { ... },
      "banana_def67890": { ... }
    }
  }
}
```

## Optimizations Implemented

### 1. Context-Aware Retrieval ✅
- Pre-filters candidates using text similarity
- Prioritizes recent interactions
- Limits results per food name (default: 3 items)
- Reduces context size by ~80% compared to passing all items

### 2. Receipt-Guided Creation ✅
- Passes expected items from receipt to VLM Call 2
- Enables batch creation during un-bagging
- Helps VLM match ambiguous items to receipts

## VLM Prompts

### VLM Call 1: Context-Free Detection
```
You are a food tracker. Analyze this egocentric video clip and identify
all food items the user *directly interacts with*.

Output: {"food_names": ["milk", "chicken", ...]}
```

### VLM Call 2A: Un-bagging (Acquisition)
```
You are a food memory manager. The user is UN-BAGGING GROCERIES.

Expected items: [bananas, milk, eggs]
Context: <existing memory items>

Output: [
  {'command': 'CREATE', 'data': {...}},
  {'command': 'UPDATE', 'food_id': '...', 'data': {...}}
]
```

### VLM Call 2B: Standard Interaction
```
You are a food memory manager. Analyze food interactions.

Context: <relevant memory items>

Output: [
  {'command': 'UPDATE', 'food_id': '...', 'data': {'action': 'retrieve from fridge', ...}}
]
```

## Troubleshooting

### VLM Server Connection Issues
```bash
# Check if VLM server is running
curl http://localhost:8000/v1/models

# Test with simple request
python test_qwen3vl_video.py
```

### No Food Items Detected
- Check video quality and lighting
- Adjust VLM FPS: `--vlm-fps 4` (higher = more frames sampled)
- Review VLM prompt sensitivity

### Memory Not Persisting
- Check file permissions for `food_memory.json`
- Verify snapshots are being created in `snapshots/` directory

### Video Processing Errors
- Ensure FFmpeg is installed: `ffmpeg -version`
- Check video file format compatibility
- Try re-encoding problematic videos

## Performance Notes

### Processing Time
- **30s clip extraction**: ~1-2 seconds per clip
- **VLM Call 1**: ~10-30 seconds (depends on video length, FPS)
- **VLM Call 2**: ~15-40 seconds (includes context)
- **Typical 5-minute video**: ~8-15 minutes total processing time

### Resource Requirements
- **RAM**: 4-8 GB (more for VLM server)
- **GPU**: 24+ GB VRAM for Qwen3-VL-30B
- **Disk**: ~100 MB per video (clips + snapshots)

## Future Enhancements

Possible extensions to the system:

1. **Visual Similarity Search**: Use CLIP embeddings for visual matching
2. **Multi-camera Support**: Merge observations from multiple cameras
3. **Real-time Processing**: Stream processing instead of batch
4. **Quantity Estimation**: Computer vision for precise quantity tracking
5. **Nutrition Database**: Link items to nutritional information
6. **Expiration Tracking**: Predict food spoilage dates

## License

Part of the NeuroTrace project.

## References

- Qwen3-VL: https://github.com/QwenLM/Qwen
- vLLM: https://github.com/vllm-project/vllm
- NeuroTrace: Egocentric video interaction analysis system

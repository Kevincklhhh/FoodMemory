# Getting Started: Food Quantity Estimation Benchmark

## What This Is

We extract ~150 food dispensal clips from HD-EPIC egocentric kitchen videos — things like "3 eggs cracked into a bowl" or "2 slices of bread placed on a plate." Each clip has a ground truth count. We then test VLMs on estimating the quantity from video alone.

~36 cases remain where all VLMs consistently fail (food out of view, occluded by hands, too small). The next step is using tool-augmented approaches to solve these.

## Repository Layout

```
HDEPIC/
│
├── inventory_pipeline/              # --- VLM experiment scripts ---
│   ├── 07_vlm_QA.py                #   Main: run VLM on annotated clips
│   ├── 07b_vlm_baseline_QA.py      #   Control: fixed 30-second blocks
│   ├── 07c_vlm_frame.py            #   Evidence frame selection
│   ├── 07d_vlm_frame_only.py       #   Counting from static frames only
│   ├── 08_evaluate_vlm_count.py    #   Evaluation (also auto-runs after 07)
│   ├── 09b_build_failure_cases.py  #   Build failure cases from eval reports
│   └── inventory_utils.py          #   Shared config, API clients, helpers
│
├── food-inventory-visualizer/       # --- Web UI ---
│   ├── video-server.js              #   Data + video server (port 4001)
│   ├── src/                         #   React app (port 3000)
│   └── package.json
│
├── outputs/02_inventory/            # --- All pipeline outputs (checked in) ---
│   ├── P01/ ... P09/                #   Per-participant results (see below)
│   ├── eval_reports/                #   Aggregate accuracy reports
│   └── failure_cases/               #   Curated hard cases
│
├── data/                            # --- NOT in repo, must be set up locally ---
│   └── HD-EPIC/
│       └── Videos/
│           ├── P01/                 #   P01-20240202-110250.mp4, ...
│           ├── P03/                 #   P03-20240216-084005.mp4, ...
│           └── .../                 #   P01–P09
│
├── models/                          #   Model weights (hands23, SAM3, etc.)
├── system_design/                   #   Design docs and experimental scripts
└── .env                             # --- NOT in repo, API keys go here ---
```

### Per-Participant Directory (e.g. `outputs/02_inventory/P03/`)

```
P03/
│
│  # ---- Ground truth (checked in, don't modify) ----
├── P03_timeline_annotated.json        # Human-verified dispensal segments with counts
│
│  # ---- VLM results (one file per experiment, checked in) ----
├── P03_vlm_qa_{tag}_results.json      # Created by 07_vlm_QA.py
│   examples:
│     P03_vlm_qa_hybrid_no_transfer_qwen_results.json
│     P03_vlm_qa_hybrid_gpt5_results.json
│     P03_vlm_qa_qwen_blind_v2_results.json
│
│  # ---- Auto-generated (safe to delete, not checked in) ----
└── vlm_clips/                         # Video clips extracted during VLM runs
    └── P03-20240216-084005-67_seg0_716_726.mp4
```

### What You Need Before Running

| Requirement | Details |
|-------------|---------|
| **Videos** | HD-EPIC `.mp4` files at `data/HD-EPIC/Videos/{P01..P09}/` |
| **API keys** | `.env` file at `HDEPIC/` root (see [VLM Backend Setup](#vlm-backend-setup)) |
| **Node.js** | For the visualizer (`node`, `npm`) |
| **Python deps** | `torch`, `transformers`, `opencv-python`, `moviepy`, `decord`, `pillow`, `python-dotenv`, `openai` |

### What's Already Checked In

All ground truth, existing experiment results, eval reports, and failure case files are committed. You can browse everything in the visualizer without running any VLM or having the video files — you just won't see video playback.

---

## Quick Start

### 1. Start the Visualizer

```bash
cd food-inventory-visualizer

# Terminal 1: data + video server
node video-server.js            # port 4001

# Terminal 2: React frontend
npm install                     # first time only
npm start                       # port 3000
```

Open http://localhost:3000. Both the server and React app must be running.

### 2. Browse Existing Results (No VLM Needed)

All previous experiment results are checked in, so you can explore immediately:

1. Select a participant (e.g., **P03**) from the top dropdown
2. **Aggregated** tab — ground truth: food items, dispensal segments, counts, video playback
3. **VLM** tab — select a tag from the dropdown (e.g., `hybrid_no_transfer_qwen`). Shows per-segment GT vs prediction with match badges: `exact` / `close` / `wrong`
4. **Failure Cases** tab — select `failure_cases_real_difficult.json` to see the 36 hardest cases
5. **Comparison** tab — pick two tags side-by-side to see where predictions differ

The tag dropdown auto-discovers result files by scanning `{P}_vlm_qa_*_results.json` in each participant's folder.

### 3. Run a New VLM Experiment

```bash
cd inventory_pipeline

# Quick test: 3 items, one participant
python 07_vlm_QA.py --participant P03 --tag my_test --model qwen --low-only --test 3

# Full run: one participant
python 07_vlm_QA.py --participant P03 --tag my_experiment --model qwen --low-only

# Full run: all participants
python 07_vlm_QA.py --all --tag my_experiment --model qwen --low-only

# Run on the 36 hardest failure cases only
python 07_vlm_QA.py --all --tag my_experiment_fc --model qwen \
    --failure-cases failure_cases/failure_cases_real_difficult.json
```

What happens:
1. Reads segments from `{P}_timeline_annotated.json` (or the failure cases file)
2. Extracts short video clips to `vlm_clips/`
3. Queries the VLM for each clip
4. Saves results to `outputs/02_inventory/{P}/{P}_vlm_qa_{tag}_results.json`
5. Auto-generates eval report in `outputs/02_inventory/eval_reports/`

Then refresh the visualizer — your new tag appears in the VLM dropdown.

**Key flags:**

| Flag | What it does |
|------|-------------|
| `--tag NAME` | **(required)** Names your experiment. Becomes part of the output filename. |
| `--model MODEL` | VLM backend: `qwen` (default), `gpt4o`, `gpt5`, or `gemini` |
| `--low-only` | Only process countable items (LOW difficulty — the ones we evaluate on) |
| `--participant P03` | Single participant (use `--all` for everyone) |
| `--test N` | Only process first N items (quick sanity check) |
| `--blind` | Don't tell the VLM the item name (tests detection + counting) |
| `--failure-cases PATH` | Run on a failure cases file instead of full timeline |
| `--verbose` | Print raw VLM responses |
| `--skip-existing` | Skip if result file for this tag already exists |
| `--delete-clips` | Clean up extracted video clips after processing |

### 4. Evaluate Results

Evaluation runs automatically after step 3. To re-run or evaluate independently:

```bash
python 08_evaluate_vlm_count.py --tag my_experiment
```

**Metrics** (computed over LOW difficulty segments only):
- **Mean Accuracy**: exact count match rate
- **MAE**: mean absolute error

Reports saved to `outputs/02_inventory/eval_reports/vlm_qa_{tag}_count_eval_report.json`.

---

## VLM Backend Setup

| Backend | How it connects | Environment variables |
|---------|----------------|----------------------|
| `qwen` | Local HTTP server (Qwen3-VL) | Server at `saltyfish.eecs.umich.edu:8000` (no key needed) |
| `gpt4o` | Azure OpenAI API | `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT` |
| `gpt5` | Azure OpenAI API | `AZURE_OPENAI_API_KEY_2`, `AZURE_OPENAI_ENDPOINT_2` |
| `gemini` | Google Gemini API | `GOOGLE_API_KEY` |

Put these in a `.env` file at the `HDEPIC/` root:
```
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=...
AZURE_OPENAI_API_KEY_2=...
AZURE_OPENAI_ENDPOINT_2=...
GOOGLE_API_KEY=...
```

## Other VLM Scripts

| Script | What it does | Example |
|--------|-------------|---------|
| `07b_vlm_baseline_QA.py` | Control: 30-second blocks instead of precise clips | `python 07b_vlm_baseline_QA.py --all --tag baseline_30s --model qwen` |
| `07c_vlm_frame.py` | VLM selects evidence frames (before/after state) | `python 07c_vlm_frame.py --all --tag evidence_qwen --prompt hybrid_no_transfer` |
| `07d_vlm_frame_only.py` | Count from static evidence frames (no video) | `python 07d_vlm_frame_only.py --all --source-tag evidence_qwen --tag frameonly` |
| `09b_build_failure_cases.py` | Build failure cases from multi-model eval reports | (hardcoded config inside script) |

## VLM Result File Format

Each `{P}_vlm_qa_{tag}_results.json` contains:

```json
{
  "participant": "P03",
  "model": "qwen",
  "tag": "my_experiment",
  "items": [
    {
      "narration_id": "P03-20240216-084005-67",
      "food_name": "bread (loaf/bag of bread)",
      "difficulty": "LOW",
      "segments": [
        {
          "segment_id": "seg_2bac7566",
          "video_id": "P03-20240216-084005",
          "start_timestamp": 715.74,
          "end_timestamp": 726.49,
          "ground_truth_count": 2,
          "predicted_count": 3,
          "match": "close",
          "clip_path": "vlm_clips/..."
        }
      ]
    }
  ]
}
```

`match` values: `exact` (correct), `close` (off by 1), `wrong`.

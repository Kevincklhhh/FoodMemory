# Knowledge Graph Visualizer

A web-based visualizer for exploring Food Knowledge Graph snapshots with synchronized video playback.

## Features

- **Snapshot Navigation**: Browse through all KG snapshots with arrow keys or buttons
- **Video Synchronization**: Automatically loads videos from HD-EPIC/Videos/ and jumps to snapshot timestamps
- **KG State Display**: View foods, zones, and interaction history at each snapshot
- **Interactive Timeline**: Click on any snapshot to jump to that point in time
- **Success/Failure Tracking**: See which narrations were successfully processed

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  React Frontend                     │
│  (visualizer/ - runs on http://localhost:3000)     │
│                                                     │
│  - KGVisualizer component                          │
│  - Video player with controls                       │
│  - KG state display (foods, zones)                 │
│  - Snapshot navigation sidebar                      │
└───────────────────┬─────────────────────────────────┘
                    │
                    │ HTTP API
                    │
┌───────────────────▼─────────────────────────────────┐
│               Flask Backend                         │
│  (kg_visualizer_server.py - runs on :5000)        │
│                                                     │
│  API Endpoints:                                     │
│  - /api/videos - List available videos             │
│  - /api/video/<id> - Stream video file            │
│  - /api/snapshots/directories - List snapshot dirs │
│  - /api/snapshots/<dir>/metadata - Get all snaps  │
│  - /api/snapshots/<dir>/<id> - Get specific snap  │
└───────────────────┬─────────────────────────────────┘
                    │
                    │ File System
                    │
      ┌─────────────┴──────────────┐
      │                            │
┌─────▼──────────┐        ┌───────▼────────┐
│ HD-EPIC/Videos │        │ kg_snapshots_* │
│                │        │                │
│ P01/           │        │ snapshot_*.json│
│ ├─ video1.mp4  │        │ metadata.jsonl │
│ └─ video2.mp4  │        └────────────────┘
└────────────────┘
```

## Setup

### 1. Install Backend Dependencies

```bash
pip install flask flask-cors
```

### 2. Install Frontend Dependencies

```bash
cd visualizer
npm install
```

### 3. Start Backend Server

```bash
# From kitchen/ directory
python kg_visualizer_server.py
```

The backend will start on http://localhost:5000

### 4. Start Frontend

```bash
# In another terminal, from kitchen/visualizer/ directory
npm start
```

The frontend will open at http://localhost:3000

## Usage

1. **Select Snapshot Directory**: Choose from available snapshot directories (e.g., `kg_snapshots_100`)

2. **Navigate Snapshots**:
   - Use ← → arrow keys
   - Click Previous/Next buttons
   - Click on any snapshot in the left sidebar

3. **Watch Video**: The video automatically loads and jumps to the snapshot's timestamp

4. **Explore KG State**:
   - View all tracked foods with their locations
   - See zone information
   - Check interaction history count
   - Monitor success/failure status

5. **Keyboard Shortcuts**:
   - `←` Previous snapshot
   - `→` Next snapshot
   - `Space` Play/pause video

## Data Flow

1. User selects a snapshot directory
2. Frontend loads snapshot metadata from backend
3. User navigates to a snapshot
4. Backend retrieves full snapshot JSON
5. Frontend:
   - Extracts video_id from snapshot
   - Loads video from HD-EPIC/Videos/{participant}/{video_id}.mp4
   - Displays KG state (foods, zones, interactions)
   - Jumps video to snapshot start_time

## API Reference

### GET /api/snapshots/directories
Returns list of available snapshot directories:
```json
[
  {
    "name": "kg_snapshots_100",
    "num_snapshots": 100,
    "metadata_file": "kg_snapshots_100/snapshots_metadata.jsonl"
  }
]
```

### GET /api/snapshots/{dir}/metadata
Returns all snapshot metadata entries:
```json
[
  {
    "narration_id": "P01-20240202-110250-1",
    "video_id": "P01-20240202-110250",
    "start_time": 7.44,
    "end_time": 8.75,
    "narration_text": "Open the upper cupboard...",
    "success": false,
    "num_foods": 0,
    "num_zones": 0
  }
]
```

### GET /api/snapshots/{dir}/{narration_id}
Returns full snapshot with KG state:
```json
{
  "snapshot_info": {
    "narration_id": "P01-20240202-110250-3",
    "video_id": "P01-20240202-110250",
    "start_time": 9.3,
    "end_time": 11.2,
    "narration_text": "Pick up a mug...",
    "update_success": true
  },
  "kg_state": {
    "foods": {
      "food_mug_1": {
        "food_id": "food_mug_1",
        "name": "mug",
        "state": "unknown",
        "location": null,
        "quantity": "1",
        "interaction_history": [...]
      }
    },
    "zones": {...}
  }
}
```

### GET /api/video/{participant}/{video_id}
Streams video file (MP4).

## Troubleshooting

### Backend Won't Start
- Check that you're in the `kitchen/` directory
- Verify HD-EPIC/Videos/ directory exists
- Ensure snapshot directories are present

### Video Not Loading
- Check browser console for CORS errors
- Verify video file exists in HD-EPIC/Videos/
- Check that video_id in snapshot matches filename

### Snapshots Not Showing
- Verify snapshot directory has `snapshots_metadata.jsonl`
- Check that snapshot JSON files exist
- Ensure backend is running on port 5000

### CORS Errors
- Make sure flask-cors is installed: `pip install flask-cors`
- Backend must be running before frontend

## Directory Structure

```
kitchen/
├── kg_visualizer_server.py       # Flask backend
├── kg_snapshots_100/              # Snapshot directory
│   ├── snapshot_*.json            # Individual snapshots
│   └── snapshots_metadata.jsonl   # Metadata index
├── HD-EPIC/
│   └── Videos/
│       └── P01/
│           └── *.mp4              # Video files
└── visualizer/                    # React app
    ├── src/
    │   ├── App.js
    │   └── components/
    │       └── KGVisualizer.jsx   # Main visualizer component
    ├── package.json
    └── public/
```

## Development

### Adding New Features

1. **Backend**: Edit `kg_visualizer_server.py` to add new API endpoints
2. **Frontend**: Edit `visualizer/src/components/KGVisualizer.jsx`

### Testing with Different Snapshots

```bash
# Run pipeline with different limits
python kg_sequential_pipeline.py \
  --csv participant_P01_narrations.csv \
  --snapshots kg_snapshots_test \
  --limit 50

# Visualizer will automatically detect new snapshot directory
```

## Future Enhancements

- [ ] Multi-participant support (switch between P01, P02, etc.)
- [ ] Interactive KG graph visualization
- [ ] Diff view between snapshots
- [ ] Export snapshot timeline as video
- [ ] Search/filter foods and interactions
- [ ] Annotation mode for correcting KG errors

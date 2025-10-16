# AI-Assisted Action Annotator - Cooking Dataset

A React-based tool for annotating video interactions with AI assistance, specifically designed for the cooking dataset format.

## Features

- Load video files and automatically match JSON annotation files from the cooking dataset
- Review and verify AI-generated interaction predictions
- Keyboard shortcuts for efficient annotation workflow
- Export verified annotations with video name as filename

## Setup

1. Install dependencies:
```bash
npm install
```

2. Run the application:
```bash
npm start
```

This will start the React development server on http://localhost:3000

## Usage

### Input Format

The tool expects:
- **Video files**: `{basename}_preview_rgb.mp4` or `{basename}.mp4`
- **JSON files**: Auto-loaded from `dataset_by_script/cooking/{basename}.json`

The JSON format should match the cooking dataset structure:
```json
{
  "analysis_results": [
    {
      "interaction_id": "interaction_0001",
      "event_description": "opens cabinet",
      "view_classification": "HOI in the view",
      "predicted_timestamps": {
        "start_time": "03:35",
        "end_time": "03:36"
      }
    }
  ]
}
```

### Keyboard Shortcuts

- **←/→**: Navigate between tasks
- **Space**: Play/pause video
- **A**: Accept current task
- **R**: Reject current task
- **Q/W**: Adjust start time (-1s/+1s)
- **Z/X**: Adjust end time (-1s/+1s)
- **J/K**: Navigate video time (-1s/+1s)
- **1/2/3**: Set view classification (HOI in view/partially/not in view)

### Output

Verified annotations are downloaded as `{video_name}.json` in the same format as the input, with added status information:

```json
{
  "analysis_results": [
    {
      "interaction_id": "task_0",
      "event_description": "opens cabinet",
      "view_classification": "HOI in the view",
      "predicted_timestamps": {
        "start_time": "3:35",
        "end_time": "3:36"
      },
      "status": "accepted",
      "confidence_score": 1.0,
      "reference_time_window": "",
      "notes": ""
    }
  ]
}
```

## Changes Made

1. **Input Format**: Updated `DataLoader.jsx` to handle the cooking dataset format with `predicted_timestamps` structure
2. **Output Format**: Modified `TaskQueue.jsx` to export JSON in cooking dataset format with video name as filename
3. **Simple Download**: Files are downloaded directly to the browser's default download location
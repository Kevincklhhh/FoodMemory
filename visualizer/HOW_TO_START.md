# Video Annotation Tool - Startup Guide

## Quick Start

### Prerequisites
- Node.js (version 14 or higher)
- npm (comes with Node.js)

### Starting the Tool

1. **Navigate to the project directory:**
   ```bash
   cd video_annotation_tool/action-annotator
   ```

2. **Install dependencies (first time only):**
   ```bash
   npm install
   ```

3. **Start the application:**
   ```bash
   npm start
   ```
   
   If port 3000 is already in use, start on a different port:
   ```bash
   PORT=3001 npm start
   ```

4. **Open your browser:**
   - Default: http://localhost:3000
   - Custom port: http://localhost:3001 (or whatever port you specified)

## Using the Tool

### Basic Features
- **Load Video**: Click "Choose File" to upload a video file
- **Playback Controls**: Use the video player controls or adjust playback rate (0.5x to 8x)
- **Frame Navigation**: Set FPS and use "Last Frame"/"Next Frame" buttons for precise navigation
- **Jump to Time**: Enter a specific time in seconds to jump to that position

### Creating Annotations
1. Enter a **Label** (required) and optional **Description**
2. Click **Start** to mark the beginning of an event
3. Click **End** to complete the annotation
4. Annotations appear in the right panel with start/end times

### Time Display
- **Current Time Bar**: Shows current playback position with precise timestamp (MM:SS.mmm)
- **Progress Bar**: Visual indicator of video progress

### Hotkey Controls
Select any annotation by clicking on it, then use:
- **Q**: Decrease start time by 1 second
- **W**: Increase start time by 1 second  
- **A**: Decrease end time by 1 second
- **S**: Increase end time by 1 second

Selected annotations are highlighted in blue with visual feedback.

### Export Data
Click **Export CSV** to download annotations in CSV format with columns:
- video_id
- label  
- start_time
- end_time

## Troubleshooting

### Port Issues
If you get "port already in use" errors:
1. Kill existing processes: `pkill -f "react-scripts start"`
2. Use a different port: `PORT=3001 npm start`

### Missing Dependencies
If you get module not found errors:
```bash
npm install
```

### Performance Issues
- Use appropriate FPS settings (default: 30)
- Consider lower playback rates for detailed annotation work
- Close other browser tabs to free up memory

## File Structure
```
action-annotator/
├── public/          # Static files
├── src/
│   └── components/
│       └── VideoAnnotationTool.jsx  # Main component
├── package.json     # Dependencies and scripts
└── HOW_TO_START.md  # This file
```
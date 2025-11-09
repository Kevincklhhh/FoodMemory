#!/usr/bin/env python3
"""
KG Visualizer Backend Server
Serves videos from HD-EPIC/Videos/ and snapshot data from snapshot directories.
"""

from flask import Flask, jsonify, send_file, request
from flask_cors import CORS
import json
import os
from pathlib import Path
import glob

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Configuration
VIDEO_BASE_DIR = Path("HD-EPIC/Videos")
SNAPSHOTS_BASE_DIR = Path(".")  # Current directory contains snapshot folders


@app.route('/api/videos', methods=['GET'])
def list_videos():
    """List all available videos grouped by participant."""
    videos = {}

    # Scan for video files in P01, P02, etc.
    for participant_dir in VIDEO_BASE_DIR.glob("P*"):
        if participant_dir.is_dir():
            participant_id = participant_dir.name
            videos[participant_id] = []

            for video_file in participant_dir.glob("*.mp4"):
                video_id = video_file.stem  # Filename without extension
                videos[participant_id].append({
                    "video_id": video_id,
                    "filename": video_file.name,
                    "path": str(video_file.relative_to(VIDEO_BASE_DIR))
                })

    return jsonify(videos)


@app.route('/api/video/<participant>/<video_id>', methods=['GET'])
def serve_video(participant, video_id):
    """Serve a video file."""
    video_path = VIDEO_BASE_DIR / participant / f"{video_id}.mp4"

    if not video_path.exists():
        return jsonify({"error": "Video not found"}), 404

    return send_file(video_path, mimetype='video/mp4')


@app.route('/api/snapshots/directories', methods=['GET'])
def list_snapshot_directories():
    """List all snapshot directories."""
    snapshot_dirs = []

    # Find all directories matching kg_snapshots*
    for snapshot_dir in glob.glob("kg_snapshots*"):
        if os.path.isdir(snapshot_dir):
            metadata_file = Path(snapshot_dir) / "snapshots_metadata.jsonl"

            if metadata_file.exists():
                # Count snapshots
                with open(metadata_file, 'r') as f:
                    num_snapshots = sum(1 for _ in f)

                snapshot_dirs.append({
                    "name": snapshot_dir,
                    "num_snapshots": num_snapshots,
                    "metadata_file": str(metadata_file)
                })

    return jsonify(snapshot_dirs)


@app.route('/api/snapshots/<snapshot_dir>/metadata', methods=['GET'])
def get_snapshot_metadata(snapshot_dir):
    """Get metadata for all snapshots in a directory."""
    metadata_file = Path(snapshot_dir) / "snapshots_metadata.jsonl"

    if not metadata_file.exists():
        return jsonify({"error": "Metadata file not found"}), 404

    snapshots = []
    with open(metadata_file, 'r') as f:
        for line in f:
            snapshots.append(json.loads(line))

    return jsonify(snapshots)


@app.route('/api/snapshots/<snapshot_dir>/<narration_id>', methods=['GET'])
def get_snapshot(snapshot_dir, narration_id):
    """Get a specific snapshot by narration ID."""
    snapshot_file = Path(snapshot_dir) / f"snapshot_{narration_id}.json"

    if not snapshot_file.exists():
        return jsonify({"error": "Snapshot not found"}), 404

    with open(snapshot_file, 'r') as f:
        snapshot_data = json.load(f)

    return jsonify(snapshot_data)


@app.route('/api/snapshots/<snapshot_dir>/at_time', methods=['GET'])
def get_snapshot_at_time(snapshot_dir):
    """Get the snapshot closest to a specific time."""
    video_id = request.args.get('video_id')
    timestamp = float(request.args.get('timestamp', 0))

    if not video_id:
        return jsonify({"error": "video_id parameter required"}), 400

    metadata_file = Path(snapshot_dir) / "snapshots_metadata.jsonl"

    if not metadata_file.exists():
        return jsonify({"error": "Metadata file not found"}), 404

    # Find snapshot closest to the timestamp
    closest_snapshot = None
    min_distance = float('inf')

    with open(metadata_file, 'r') as f:
        for line in f:
            entry = json.loads(line)
            if entry['video_id'] == video_id:
                # Use end_time as the reference point
                distance = abs(entry['end_time'] - timestamp)
                if distance < min_distance:
                    min_distance = distance
                    closest_snapshot = entry

    if not closest_snapshot:
        return jsonify({"error": "No snapshot found for this video"}), 404

    # Load the full snapshot
    snapshot_file = Path(snapshot_dir) / f"snapshot_{closest_snapshot['narration_id']}.json"
    with open(snapshot_file, 'r') as f:
        snapshot_data = json.load(f)

    return jsonify(snapshot_data)


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "ok",
        "video_base_dir": str(VIDEO_BASE_DIR),
        "snapshots_base_dir": str(SNAPSHOTS_BASE_DIR)
    })


if __name__ == '__main__':
    print("=" * 80)
    print("KG Visualizer Backend Server")
    print("=" * 80)
    print(f"Video directory: {VIDEO_BASE_DIR.absolute()}")
    print(f"Snapshots directory: {SNAPSHOTS_BASE_DIR.absolute()}")
    print("\nStarting server on http://localhost:5000")
    print("=" * 80)

    app.run(debug=True, port=5000, host='0.0.0.0')

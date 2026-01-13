import React, { useState } from "react";

const OutputDataLoader = ({ onDataLoad }) => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [videoFile, setVideoFile] = useState(null);
  const [jsonFile, setJsonFile] = useState(null);

  const handleLoadData = async () => {
    if (!videoFile || !jsonFile) {
      setError('Both video and JSON files are required');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      await handleFileLoad(videoFile, jsonFile);
    } catch (err) {
      setError(`Failed to load files: ${err.message}`);
      console.error('Data loading error:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleFileLoad = async (videoFile, jsonFile) => {
    try {
      // Create video URL
      const videoUrl = URL.createObjectURL(videoFile);

      // Extract base name from video file
      const fileName = videoFile.name;
      const baseName = fileName.replace('_preview_rgb.mp4', '').replace('.mp4', '');

      // Use provided JSON file (now required)
      const jsonText = await jsonFile.text();
      const jsonData = JSON.parse(jsonText);
      
      // Extract tasks from outputs JSON format
      const tasks = [];
      
      if (jsonData.analysis_results) {
        jsonData.analysis_results.forEach((result, resultIndex) => {
          // Convert time format (MM:SS to seconds)
          const parseTimeToSeconds = (timeStr) => {
            const [minutes, seconds] = timeStr.split(':').map(Number);
            return minutes * 60 + seconds;
          };
          
          // Use predicted_timestamps object for outputs format
          const timestamps = result.predicted_timestamps;
          if (timestamps && timestamps.start_time && timestamps.end_time) {
            tasks.push({
              id: result.interaction_id, // Use original interaction_id
              description: result.event_description,
              viewClassification: result.view_classification,
              startTime: parseTimeToSeconds(timestamps.start_time),
              endTime: parseTimeToSeconds(timestamps.end_time),
              confidence: result.confidence_score || 1.0, // default confidence if not present
              timeWindow: result.reference_time_window || '', // may not be present
              status: 'pending', // pending, rejected
              originalData: result
            });
          }
        });
      }
      
      // Sort tasks by interaction ID to maintain original order
      tasks.sort((a, b) => {
        // Extract the numeric part from interaction_0001, interaction_0002, etc.
        const aNum = parseInt(a.id.replace('interaction_', ''));
        const bNum = parseInt(b.id.replace('interaction_', ''));
        return aNum - bNum;
      });
      
      onDataLoad({
        videoUrl,
        videoFile,
        jsonData,
        tasks,
        fileName: videoFile.name,
        baseName: baseName
      });

    } catch (err) {
      throw err;
    }
  };

  const handleVideoFileSelect = (e) => {
    const file = e.target.files[0];
    if (file && file.type.startsWith('video/')) {
      setVideoFile(file);
      setError(null);
    } else {
      setError('Please select a valid video file');
    }
  };

  const handleJsonFileSelect = (e) => {
    const file = e.target.files[0];
    if (file && file.name.endsWith('.json')) {
      setJsonFile(file);
      setError(null);
    } else {
      setError('Please select a valid JSON file');
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    const files = Array.from(e.dataTransfer.files);
    const droppedVideoFile = files.find(f => f.type.startsWith('video/'));
    const droppedJsonFile = files.find(f => f.name.endsWith('.json'));

    if (droppedVideoFile && !videoFile) {
      setVideoFile(droppedVideoFile);
    }
    if (droppedJsonFile && !jsonFile) {
      setJsonFile(droppedJsonFile);
    }

    setError(null);
  };

  return (
    <div style={{
      padding: '40px',
      textAlign: 'center',
      border: '2px dashed #ccc',
      borderRadius: '8px',
      backgroundColor: '#f9f9f9'
    }}>
      <h2>Output Edit Mode - Video Annotation Tool</h2>
      <p>Upload both a video file and its corresponding checked JSON output file to start annotation editing.</p>

      <div style={{ display: 'flex', gap: '20px', justifyContent: 'center', marginBottom: '30px' }}>
        {/* Video Upload */}
        <div
          onDrop={handleDrop}
          onDragOver={(e) => e.preventDefault()}
          style={{
            border: videoFile ? '2px solid #28a745' : '2px dashed #007bff',
            borderRadius: '8px',
            padding: '30px',
            backgroundColor: 'white',
            cursor: 'pointer',
            minWidth: '200px'
          }}
        >
          <p style={{ fontSize: '2em' }}>📹</p>
          <p><strong>Video File</strong></p>
          <p style={{ fontSize: '0.9em', color: '#666', marginBottom: '10px' }}>
            {videoFile ? videoFile.name : 'Drop video file or click to select'}
          </p>

          <input
            type="file"
            accept="video/*"
            onChange={handleVideoFileSelect}
            style={{ display: 'none' }}
            id="video-file-input"
          />

          <label
            htmlFor="video-file-input"
            style={{
              display: 'inline-block',
              padding: '8px 16px',
              backgroundColor: videoFile ? '#28a745' : '#007bff',
              color: 'white',
              borderRadius: '4px',
              cursor: 'pointer',
              fontSize: '0.9em'
            }}
          >
            {videoFile ? '✓ Selected' : 'Select Video'}
          </label>
        </div>

        {/* JSON Upload */}
        <div
          onDrop={handleDrop}
          onDragOver={(e) => e.preventDefault()}
          style={{
            border: jsonFile ? '2px solid #28a745' : '2px dashed #ffc107',
            borderRadius: '8px',
            padding: '30px',
            backgroundColor: 'white',
            cursor: 'pointer',
            minWidth: '200px'
          }}
        >
          <p style={{ fontSize: '2em' }}>📋</p>
          <p><strong>Checked JSON</strong></p>
          <p style={{ fontSize: '0.9em', color: '#666', marginBottom: '10px' }}>
            {jsonFile ? jsonFile.name : 'Drop JSON file or click to select'}
          </p>

          <input
            type="file"
            accept=".json"
            onChange={handleJsonFileSelect}
            style={{ display: 'none' }}
            id="json-file-input"
          />

          <label
            htmlFor="json-file-input"
            style={{
              display: 'inline-block',
              padding: '8px 16px',
              backgroundColor: jsonFile ? '#28a745' : '#ffc107',
              color: jsonFile ? 'white' : '#000',
              borderRadius: '4px',
              cursor: 'pointer',
              fontSize: '0.9em'
            }}
          >
            {jsonFile ? '✓ Selected' : 'Select JSON'}
          </label>
        </div>
      </div>

      {/* Load Button */}
      <button
        onClick={handleLoadData}
        disabled={!videoFile || !jsonFile || loading}
        style={{
          padding: '15px 30px',
          fontSize: '1.1em',
          backgroundColor: (!videoFile || !jsonFile || loading) ? '#6c757d' : '#007bff',
          color: 'white',
          border: 'none',
          borderRadius: '4px',
          cursor: (!videoFile || !jsonFile || loading) ? 'not-allowed' : 'pointer',
          marginBottom: '20px'
        }}
      >
        {loading ? 'Loading...' : 'Load Annotation Data'}
      </button>
      
      {loading && (
        <div style={{ padding: '20px', color: '#007bff' }}>
          <p>Loading and processing files...</p>
        </div>
      )}
      
      {error && (
        <div style={{ 
          padding: '15px', 
          backgroundColor: '#ffebee', 
          color: '#c62828',
          borderRadius: '4px',
          marginTop: '20px'
        }}>
          {error}
        </div>
      )}
      
      <div style={{
        marginTop: '20px',
        padding: '20px',
        backgroundColor: '#e8f5e8',
        borderRadius: '4px',
        fontSize: '0.9em'
      }}>
        <h4>Required Files:</h4>
        <ul style={{ textAlign: 'left', display: 'inline-block' }}>
          <li><strong>Video file:</strong> Any video format (e.g., basename_preview_rgb.mp4)</li>
          <li><strong>Checked JSON:</strong> Validated output JSON file with annotation results</li>
        </ul>
        <p style={{ marginTop: '10px', fontStyle: 'italic', color: '#666' }}>
          Both files must be uploaded before you can start editing annotations.
        </p>
      </div>
    </div>
  );
};

export default OutputDataLoader;
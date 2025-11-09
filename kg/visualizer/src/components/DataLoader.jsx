import React, { useState } from "react";

const DataLoader = ({ onDataLoad }) => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const findJsonFile = async (baseName) => {
    // Search in dataset_by_script subdirectories
    const searchPaths = [
      `/dataset_by_script/cooking/${baseName}.json`,
      `/dataset_by_script/simon_says/${baseName}.json`,
      `/dataset_by_script/laundary/${baseName}.json`,
      `/dataset_by_script/housekeeping/${baseName}.json`
    ];
    
    for (const path of searchPaths) {
      try {
        const response = await fetch(path);
        if (response.ok) {
          console.log(`Found JSON file at: ${path}`);
          return await response.json();
        }
      } catch (error) {
        console.log(`File not found at: ${path}`);
      }
    }
    
    console.error(`Could not find ${baseName}.json in dataset_by_script directories`);
    return null;
  };

  const handleFileLoad = async (videoFile, jsonFile = null) => {
    setLoading(true);
    setError(null);
    
    try {
      // Create video URL
      const videoUrl = URL.createObjectURL(videoFile);
      
      let jsonData = null;
      
      // Extract base name from video file
      const fileName = videoFile.name;
      const baseName = fileName.replace('_preview_rgb.mp4', '').replace('.mp4', '');
      
      if (jsonFile) {
        // Use provided JSON file
        const jsonText = await jsonFile.text();
        jsonData = JSON.parse(jsonText);
      } else {
        // Auto-find JSON file based on base name in dataset_by_script
        jsonData = await findJsonFile(baseName);
        if (!jsonData) {
          throw new Error(`Could not find ${baseName}.json in dataset_by_script directories`);
        }
      }
      
      // Extract tasks from cooking dataset JSON format
      const tasks = [];
      
      if (jsonData.analysis_results) {
        jsonData.analysis_results.forEach((result, resultIndex) => {
          // Convert time format (MM:SS to seconds)
          const parseTimeToSeconds = (timeStr) => {
            const [minutes, seconds] = timeStr.split(':').map(Number);
            return minutes * 60 + seconds;
          };
          
          // Use predicted_timestamps object for cooking dataset format
          const timestamps = result.predicted_timestamps;
          if (timestamps && timestamps.start_time && timestamps.end_time) {
            tasks.push({
              id: result.interaction_id, // Use original interaction_id
              description: result.event_description || result.primary_description, // Handle merged format
              viewClassification: result.view_classification,
              startTime: parseTimeToSeconds(timestamps.start_time),
              endTime: parseTimeToSeconds(timestamps.end_time),
              confidence: result.confidence_score || 1.0, // default confidence if not present
              timeWindow: result.reference_time_window || '', // may not be present
              status: 'pending', // pending, rejected
              hand: result.hand || null, // Load hand attribute from JSON
              originalData: result,
              // Merged interaction specific fields
              isMerged: result.is_merged || false,
              mergedDescriptions: result.merged_descriptions || [result.event_description || result.primary_description].filter(Boolean),
              primaryDescription: result.primary_description,
              mergeCount: result.merge_count || 1,
              originalIds: result.original_ids || [result.interaction_id],
              originalTasks: result.original_interactions ? result.original_interactions.map(origTask => ({
                ...origTask,
                id: origTask.interaction_id,
                description: origTask.event_description,
                viewClassification: origTask.view_classification,
                startTime: parseTimeToSeconds(origTask.predicted_timestamps.start_time),
                endTime: parseTimeToSeconds(origTask.predicted_timestamps.end_time),
                confidence: origTask.confidence_score || 1.0,
                timeWindow: origTask.reference_time_window || '',
                status: 'pending',
                hand: origTask.hand || null, // Load hand attribute for original tasks too
                originalIds: [origTask.interaction_id]
              })) : null
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
      setError(`Failed to load files: ${err.message}`);
      console.error('Data loading error:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    const files = Array.from(e.dataTransfer.files);
    const videoFile = files.find(f => f.type.startsWith('video/'));
    const jsonFile = files.find(f => f.name.endsWith('.json'));
    
    if (videoFile) {
      handleFileLoad(videoFile, jsonFile); // jsonFile can be null, will auto-find
    } else {
      setError('Please drop a video file (JSON file will be auto-detected)');
    }
  };

  const handleFileChange = (e) => {
    const files = Array.from(e.target.files);
    const videoFile = files.find(f => f.type.startsWith('video/'));
    const jsonFile = files.find(f => f.name.endsWith('.json'));
    
    if (videoFile) {
      handleFileLoad(videoFile, jsonFile); // jsonFile can be null, will auto-find
    } else {
      setError('Please select a video file (JSON file will be auto-detected)');
    }
  };

  return (
    <div style={{ 
      padding: '40px', 
      textAlign: 'center',
      border: '2px dashed #ccc',
      borderRadius: '8px',
      backgroundColor: '#f9f9f9'
    }}>
      <h2>AI-Assisted Video Annotation Tool</h2>
      <p>Load a video file (format: basename_preview_rgb.mp4). The corresponding JSON file will be automatically loaded from the dataset_by_script directory.</p>
      
      <div 
        onDrop={handleDrop}
        onDragOver={(e) => e.preventDefault()}
        style={{
          border: '2px dashed #007bff',
          borderRadius: '8px',
          padding: '40px',
          margin: '20px 0',
          backgroundColor: 'white',
          cursor: 'pointer'
        }}
      >
        <p>📹</p>
        <p>Drop video file here</p>
        <p style={{ fontSize: '0.9em', color: '#666' }}>
          or click to select video file
        </p>
        
        <input 
          type="file"
          accept="video/*"
          onChange={handleFileChange}
          style={{
            position: 'absolute',
            left: '-9999px'
          }}
          id="file-input"
        />
        
        <label 
          htmlFor="file-input"
          style={{
            display: 'inline-block',
            padding: '10px 20px',
            backgroundColor: '#007bff',
            color: 'white',
            borderRadius: '4px',
            cursor: 'pointer',
            marginTop: '10px'
          }}
        >
          Select Files
        </label>
      </div>
      
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
        marginTop: '30px', 
        padding: '20px', 
        backgroundColor: '#e8f5e8',
        borderRadius: '4px',
        fontSize: '0.9em'
      }}>
        <h4>Expected Files:</h4>
        <ul style={{ textAlign: 'left', display: 'inline-block' }}>
          <li><strong>Video file:</strong> basename_preview_rgb.mp4</li>
          <li><strong>JSON file:</strong> Auto-loaded from dataset_by_script/cooking/ or dataset_by_script/simon_says/</li>
        </ul>
      </div>
    </div>
  );
};

export default DataLoader;
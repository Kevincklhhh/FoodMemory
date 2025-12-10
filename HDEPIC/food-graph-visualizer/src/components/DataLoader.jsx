import React, { useState } from 'react';

const styles = {
  container: {
    padding: '20px',
    backgroundColor: '#f5f5f5',
    borderRadius: '8px',
    marginBottom: '20px',
  },
  title: {
    margin: '0 0 15px 0',
    fontSize: '18px',
    fontWeight: 'bold',
  },
  inputGroup: {
    marginBottom: '15px',
  },
  label: {
    display: 'block',
    marginBottom: '5px',
    fontWeight: '500',
  },
  input: {
    width: '100%',
    padding: '8px',
    border: '1px solid #ccc',
    borderRadius: '4px',
    fontSize: '14px',
    boxSizing: 'border-box',
  },
  button: {
    padding: '10px 20px',
    backgroundColor: '#2196F3',
    color: 'white',
    border: 'none',
    borderRadius: '4px',
    cursor: 'pointer',
    fontSize: '14px',
    marginRight: '10px',
  },
  buttonDisabled: {
    backgroundColor: '#ccc',
    cursor: 'not-allowed',
  },
  error: {
    color: '#d32f2f',
    marginTop: '10px',
    fontSize: '14px',
  },
  success: {
    color: '#388e3c',
    marginTop: '10px',
    fontSize: '14px',
  },
  fileInput: {
    marginBottom: '10px',
  },
};

function DataLoader({ onDataLoaded }) {
  const [stateChangeFile, setStateChangeFile] = useState(null);
  const [graphFile, setGraphFile] = useState(null);
  const [videoFile, setVideoFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);

  const handleFileChange = (setter) => (e) => {
    if (e.target.files && e.target.files[0]) {
      setter(e.target.files[0]);
      setError(null);
    }
  };

  const readJsonFile = (file) => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = (e) => {
        try {
          const data = JSON.parse(e.target.result);
          resolve(data);
        } catch (err) {
          reject(new Error(`Failed to parse ${file.name}: ${err.message}`));
        }
      };
      reader.onerror = () => reject(new Error(`Failed to read ${file.name}`));
      reader.readAsText(file);
    });
  };

  const handleLoad = async () => {
    if (!stateChangeFile || !graphFile) {
      setError('Please select both state_change.json and spatio_temporal_graph.json files');
      return;
    }

    setLoading(true);
    setError(null);
    setSuccess(null);

    try {
      const [stateChangeData, graphData] = await Promise.all([
        readJsonFile(stateChangeFile),
        readJsonFile(graphFile),
      ]);

      // Create video URL if video file provided
      let videoUrl = null;
      if (videoFile) {
        videoUrl = URL.createObjectURL(videoFile);
      }

      onDataLoaded({
        stateChange: stateChangeData,
        graph: graphData,
        videoUrl: videoUrl,
      });

      const eventCount = Array.isArray(stateChangeData) ? stateChangeData.length : (stateChangeData.events?.length || 0);
      setSuccess(`Loaded ${eventCount} events and ${graphData.block_graphs?.length || 0} snapshots`);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={styles.container}>
      <h2 style={styles.title}>Load Data Files</h2>

      <div style={styles.inputGroup}>
        <label style={styles.label}>state_change.json (required)</label>
        <input
          type="file"
          accept=".json"
          onChange={handleFileChange(setStateChangeFile)}
          style={styles.fileInput}
        />
      </div>

      <div style={styles.inputGroup}>
        <label style={styles.label}>spatio_temporal_graph.json (required)</label>
        <input
          type="file"
          accept=".json"
          onChange={handleFileChange(setGraphFile)}
          style={styles.fileInput}
        />
      </div>

      <div style={styles.inputGroup}>
        <label style={styles.label}>Video file (optional)</label>
        <input
          type="file"
          accept="video/*"
          onChange={handleFileChange(setVideoFile)}
          style={styles.fileInput}
        />
      </div>

      <button
        onClick={handleLoad}
        disabled={loading || !stateChangeFile || !graphFile}
        style={{
          ...styles.button,
          ...(loading || !stateChangeFile || !graphFile ? styles.buttonDisabled : {}),
        }}
      >
        {loading ? 'Loading...' : 'Load Data'}
      </button>

      {error && <div style={styles.error}>{error}</div>}
      {success && <div style={styles.success}>{success}</div>}
    </div>
  );
}

export default DataLoader;

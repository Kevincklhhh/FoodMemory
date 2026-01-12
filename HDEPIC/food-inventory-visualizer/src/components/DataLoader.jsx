import React, { useState } from 'react';

const styles = {
  container: {
    padding: '15px 20px',
    backgroundColor: '#f5f5f5',
    borderRadius: '8px',
    marginBottom: '20px',
  },
  title: {
    margin: '0 0 12px 0',
    fontSize: '16px',
    fontWeight: 'bold',
  },
  inputRow: {
    display: 'flex',
    gap: '15px',
    alignItems: 'flex-end',
    flexWrap: 'wrap',
  },
  inputGroup: {
    flex: '1',
    minWidth: '200px',
  },
  label: {
    display: 'block',
    marginBottom: '4px',
    fontWeight: '500',
    fontSize: '13px',
  },
  hint: {
    fontSize: '11px',
    color: '#666',
    marginTop: '2px',
  },
  button: {
    padding: '8px 16px',
    backgroundColor: '#2196F3',
    color: 'white',
    border: 'none',
    borderRadius: '4px',
    cursor: 'pointer',
    fontSize: '13px',
    height: '34px',
  },
  buttonDisabled: {
    backgroundColor: '#ccc',
    cursor: 'not-allowed',
  },
  error: {
    color: '#d32f2f',
    marginTop: '8px',
    fontSize: '13px',
  },
  success: {
    color: '#388e3c',
    marginTop: '8px',
    fontSize: '13px',
  },
  fileInput: {
    fontSize: '13px',
  },
};

function DataLoader({ onDataLoaded }) {
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
    if (!graphFile) {
      setError('Please select spatio_temporal_graph.json');
      return;
    }

    setLoading(true);
    setError(null);
    setSuccess(null);

    try {
      const graphData = await readJsonFile(graphFile);

      // Extract state_changes from graph (embedded during pipeline)
      // Fall back to empty array if not present
      const stateChangeData = graphData.state_changes || [];

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

      const eventCount = Array.isArray(stateChangeData) ? stateChangeData.length : 0;
      const vlmLogCount = Object.keys(graphData.vlm_logs || {}).length;
      setSuccess(`Loaded ${eventCount} events, ${graphData.block_graphs?.length || 0} snapshots, ${vlmLogCount} VLM logs`);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={styles.container}>
      <h2 style={styles.title}>Load Food Graph</h2>

      <div style={styles.inputRow}>
        <div style={styles.inputGroup}>
          <label style={styles.label}>spatio_temporal_graph.json</label>
          <input
            type="file"
            accept=".json"
            onChange={handleFileChange(setGraphFile)}
            style={styles.fileInput}
          />
          <div style={styles.hint}>Contains graph, events, and VLM logs</div>
        </div>

        <div style={styles.inputGroup}>
          <label style={styles.label}>Video (optional)</label>
          <input
            type="file"
            accept="video/*"
            onChange={handleFileChange(setVideoFile)}
            style={styles.fileInput}
          />
        </div>

        <button
          onClick={handleLoad}
          disabled={loading || !graphFile}
          style={{
            ...styles.button,
            ...(loading || !graphFile ? styles.buttonDisabled : {}),
          }}
        >
          {loading ? 'Loading...' : 'Load'}
        </button>
      </div>

      {error && <div style={styles.error}>{error}</div>}
      {success && <div style={styles.success}>{success}</div>}
    </div>
  );
}

export default DataLoader;

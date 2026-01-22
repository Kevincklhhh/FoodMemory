import React, { useState, useRef, useCallback, useEffect } from 'react';
import InventoryItemList from './components/InventoryItemList';
import InventoryVideoPlayer from './components/InventoryVideoPlayer';
import TimelineView from './components/TimelineView';
import AggregatedView from './components/AggregatedView';
import VLMResultsView from './components/VLMResultsView';
import { parseNarrationTimestampsJSON, parseNarrationId } from './utils/narrationParser';

const VIDEO_SERVER = 'http://localhost:3001';

const styles = {
  app: {
    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
    minHeight: '100vh',
    backgroundColor: '#f5f5f5',
  },
  header: {
    backgroundColor: '#2E7D32',
    color: 'white',
    padding: '15px 20px',
    fontSize: '20px',
    fontWeight: 'bold',
    boxShadow: '0 2px 4px rgba(0,0,0,0.2)',
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  serverStatus: {
    fontSize: '12px',
    padding: '4px 10px',
    borderRadius: '12px',
    backgroundColor: 'rgba(255,255,255,0.2)',
  },
  serverOnline: {
    backgroundColor: 'rgba(76, 175, 80, 0.3)',
  },
  serverOffline: {
    backgroundColor: 'rgba(244, 67, 54, 0.3)',
  },
  content: {
    padding: '20px',
    maxWidth: '1600px',
    margin: '0 auto',
  },
  loaderSection: {
    backgroundColor: 'white',
    borderRadius: '8px',
    padding: '20px',
    marginBottom: '20px',
    boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
  },
  loaderTitle: {
    margin: '0 0 15px 0',
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
  select: {
    width: '100%',
    padding: '8px',
    borderRadius: '4px',
    border: '1px solid #ddd',
    fontSize: '13px',
  },
  button: {
    padding: '8px 16px',
    backgroundColor: '#2E7D32',
    color: 'white',
    border: 'none',
    borderRadius: '4px',
    cursor: 'pointer',
    fontSize: '13px',
    height: '34px',
  },
  buttonSecondary: {
    backgroundColor: '#1976D2',
  },
  mainLayout: {
    display: 'grid',
    gridTemplateColumns: '400px 1fr',
    gap: '20px',
    height: 'calc(100vh - 240px)',
    maxHeight: '800px',
  },
  statsBar: {
    padding: '10px 15px',
    backgroundColor: '#e8f5e9',
    borderRadius: '4px',
    fontSize: '13px',
    color: '#1b5e20',
    marginBottom: '20px',
  },
  error: {
    color: '#d32f2f',
    marginTop: '10px',
    fontSize: '13px',
  },
  success: {
    color: '#388e3c',
    marginTop: '10px',
    fontSize: '13px',
  },
  quickLoad: {
    display: 'flex',
    gap: '10px',
    marginTop: '15px',
    paddingTop: '15px',
    borderTop: '1px solid #e0e0e0',
  },
  viewToggle: {
    display: 'flex',
    gap: '0',
    marginLeft: '20px',
  },
  viewToggleButton: {
    padding: '8px 16px',
    border: '1px solid #2E7D32',
    backgroundColor: 'white',
    color: '#2E7D32',
    fontSize: '13px',
    cursor: 'pointer',
    transition: 'all 0.2s',
  },
  viewToggleButtonFirst: {
    borderRadius: '4px 0 0 4px',
  },
  viewToggleButtonLast: {
    borderRadius: '0 4px 4px 0',
    borderLeft: 'none',
  },
  viewToggleButtonActive: {
    backgroundColor: '#2E7D32',
    color: 'white',
  },
};

function InventoryApp() {
  // Server state
  const [serverOnline, setServerOnline] = useState(false);
  const [availableVideos, setAvailableVideos] = useState([]);

  // Data state
  const [inventoryData, setInventoryData] = useState(null);
  const [lifecycleData, setLifecycleData] = useState(null);
  const [aggregatedData, setAggregatedData] = useState(null);
  const [vlmDataByModel, setVlmDataByModel] = useState({}); // { modelName: data }
  const [selectedVlmModel, setSelectedVlmModel] = useState(null);
  const [narrationTimestamps, setNarrationTimestamps] = useState({});
  const [participant, setParticipant] = useState('P01');
  const [viewMode, setViewMode] = useState('items'); // 'items', 'timeline', 'aggregated', or 'vlm'

  // UI state
  const [selectedItem, setSelectedItem] = useState(null);
  const [currentVideo, setCurrentVideo] = useState(null);
  const [currentVideoUrl, setCurrentVideoUrl] = useState(null);
  const [currentTime, setCurrentTime] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);

  // Refs
  const videoRef = useRef(null);

  // Check server status on mount
  useEffect(() => {
    checkServerStatus();
    const interval = setInterval(checkServerStatus, 5000);
    return () => clearInterval(interval);
  }, []);

  const checkServerStatus = async () => {
    try {
      const response = await fetch(`${VIDEO_SERVER}/list/${participant}`, { method: 'GET' });
      if (response.ok) {
        const data = await response.json();
        setServerOnline(true);
        setAvailableVideos(data.videos || []);
      } else {
        setServerOnline(false);
        setAvailableVideos([]);
      }
    } catch {
      setServerOnline(false);
      setAvailableVideos([]);
    }
  };

  // Load data from server
  const loadFromServer = async () => {
    setLoading(true);
    setError(null);
    setSuccess(null);

    try {
      // Load inventory data (known_quantities.json)
      const inventoryResponse = await fetch(
        `${VIDEO_SERVER}/data/outputs/02_inventory/${participant}/${participant}_known_quantities.json`
      );
      if (!inventoryResponse.ok) {
        throw new Error(`Failed to load inventory data: ${inventoryResponse.statusText}`);
      }
      const inventoryJson = await inventoryResponse.json();
      setInventoryData(inventoryJson);

      // Load lifecycle data (lifecycle.json) for timeline view
      const lifecycleResponse = await fetch(
        `${VIDEO_SERVER}/data/outputs/02_inventory/${participant}/${participant}_lifecycle.json`
      );
      if (lifecycleResponse.ok) {
        const lifecycleJson = await lifecycleResponse.json();
        setLifecycleData(lifecycleJson);
      }

      // Load timeline data for aggregated view
      // First try annotated (user's saved annotations), then fall back to aggregated (original)
      let aggregatedLoaded = false;
      const annotatedResponse = await fetch(
        `${VIDEO_SERVER}/data/outputs/02_inventory/${participant}/${participant}_timeline_annotated.json`
      );
      if (annotatedResponse.ok) {
        const annotatedJson = await annotatedResponse.json();
        setAggregatedData(annotatedJson);
        aggregatedLoaded = true;
        console.log('Loaded annotated timeline data');
      }

      if (!aggregatedLoaded) {
        const aggregatedResponse = await fetch(
          `${VIDEO_SERVER}/data/outputs/02_inventory/${participant}/${participant}_timeline_aggregated.json`
        );
        if (aggregatedResponse.ok) {
          const aggregatedJson = await aggregatedResponse.json();
          setAggregatedData(aggregatedJson);
          console.log('Loaded aggregated timeline data');
        }
      }

      // Load VLM QA results from multiple models
      // Try multiple file naming conventions for each model
      // Note: Try _qwen_ naming first as it may have newer fields
      const vlmModels = [
        { key: 'qwen', files: [
          `${participant}_vlm_qa_qwen_results.json`,
          `${participant}_vlm_qa_results.json`,
        ]},
        { key: 'gpt4o', files: [
          `${participant}_vlm_qa_gpt4o_results.json`,
        ]},
        { key: 'baseline', files: [
          `${participant}_vlm_baseline_qwen_results.json`,
        ]},
      ];
      const loadedVlmData = {};
      let firstModel = null;

      for (const model of vlmModels) {
        for (const file of model.files) {
          try {
            const vlmResponse = await fetch(
              `${VIDEO_SERVER}/data/outputs/02_inventory/${participant}/${file}`
            );
            if (vlmResponse.ok) {
              const vlmJson = await vlmResponse.json();
              loadedVlmData[model.key] = vlmJson;
              if (!firstModel) firstModel = model.key;
              console.log(`Loaded VLM QA results for ${model.key} from ${file}`);
              break; // Found file for this model, move to next model
            }
          } catch (err) {
            // Continue to next file pattern
          }
        }
      }

      setVlmDataByModel(loadedVlmData);
      if (firstModel) {
        setSelectedVlmModel(firstModel);
      }

      // Load narration timestamps from central JSON file
      const narrationsResponse = await fetch(`${VIDEO_SERVER}/narrations`);
      if (narrationsResponse.ok) {
        const narrationsJson = await narrationsResponse.json();
        const timestamps = parseNarrationTimestampsJSON(narrationsJson);
        setNarrationTimestamps(timestamps);
      }

      // Refresh available videos
      await checkServerStatus();

      const itemCount = inventoryJson.items?.length || 0;
      setSuccess(`Loaded ${itemCount} inventory items for ${participant}`);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  // Handle item selection
  const handleSelectItem = useCallback((item) => {
    setSelectedItem(item);
  }, []);

  // Handle narration click - load video and seek to timestamp
  const handleNarrationClick = useCallback((narrationId) => {
    const parsed = parseNarrationId(narrationId);
    if (!parsed) return;

    const { videoId } = parsed;
    const timestamp = narrationTimestamps[narrationId];

    // Build video URL from server
    const videoUrl = `${VIDEO_SERVER}/videos/${parsed.participant}/${videoId}.mp4`;

    if (currentVideo !== videoId) {
      setCurrentVideo(videoId);
      setCurrentVideoUrl(videoUrl);
      // Wait for video to load before seeking
      setTimeout(() => {
        if (timestamp !== undefined && videoRef.current) {
          videoRef.current.seekTo(timestamp);
          setCurrentTime(timestamp);
        }
      }, 500);
    } else if (timestamp !== undefined && videoRef.current) {
      videoRef.current.seekTo(timestamp);
      setCurrentTime(timestamp);
    }
  }, [narrationTimestamps, currentVideo]);

  const handleTimeUpdate = useCallback((time) => {
    setCurrentTime(time);
  }, []);

  // Handle video load with specific timestamp (for aggregated view)
  const handleLoadVideoAtTime = useCallback((videoId, timestamp) => {
    if (!videoId) return;

    // Extract participant from videoId (e.g., P01-20240202-110250 -> P01)
    const participantId = videoId.split('-')[0];
    const videoUrl = `${VIDEO_SERVER}/videos/${participantId}/${videoId}.mp4`;

    if (currentVideo !== videoId) {
      setCurrentVideo(videoId);
      setCurrentVideoUrl(videoUrl);
      // Wait for video to load before seeking
      setTimeout(() => {
        if (timestamp !== undefined && videoRef.current) {
          videoRef.current.seekTo(timestamp);
          setCurrentTime(timestamp);
        }
      }, 500);
    } else if (timestamp !== undefined && videoRef.current) {
      videoRef.current.seekTo(timestamp);
      setCurrentTime(timestamp);
    }
  }, [currentVideo]);

  // Handle save for aggregated data modifications - saves to _timeline_annotated.json
  const handleSaveAggregated = useCallback(async (updatedData) => {
    const response = await fetch(
      `${VIDEO_SERVER}/data/outputs/02_inventory/${participant}/${participant}_timeline_annotated.json`,
      {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(updatedData, null, 2),
      }
    );

    if (!response.ok) {
      throw new Error(`Failed to save: ${response.statusText}`);
    }

    // Update local state with saved data
    setAggregatedData(updatedData);
  }, [participant]);

  return (
    <div style={styles.app}>
      <div style={styles.header}>
        <span>🥕 Inventory Lifecycle Visualizer</span>
        <span
          style={{
            ...styles.serverStatus,
            ...(serverOnline ? styles.serverOnline : styles.serverOffline),
          }}
        >
          Server: {serverOnline ? `Online (${availableVideos.length} videos)` : 'Offline'}
        </span>
      </div>

      <div style={styles.content}>
        {/* Data Loader Section */}
        <div style={styles.loaderSection}>
          <h2 style={styles.loaderTitle}>Load Data</h2>

          <div style={styles.inputRow}>
            <div style={styles.inputGroup}>
              <label style={styles.label}>Participant</label>
              <select
                style={styles.select}
                value={participant}
                onChange={(e) => setParticipant(e.target.value)}
              >
                <option value="P01">P01</option>
                <option value="P02">P02</option>
                <option value="P03">P03</option>
                <option value="P04">P04</option>
                <option value="P05">P05</option>
              </select>
              <div style={styles.hint}>Select participant to load</div>
            </div>

            <button
              onClick={loadFromServer}
              disabled={loading || !serverOnline}
              style={{
                ...styles.button,
                opacity: loading || !serverOnline ? 0.6 : 1,
                cursor: loading || !serverOnline ? 'not-allowed' : 'pointer',
              }}
            >
              {loading ? 'Loading...' : 'Load from Server'}
            </button>

            {/* View Mode Toggle */}
            {inventoryData && (
              <div style={styles.viewToggle}>
                <button
                  style={{
                    ...styles.viewToggleButton,
                    ...styles.viewToggleButtonFirst,
                    ...(viewMode === 'items' ? styles.viewToggleButtonActive : {}),
                  }}
                  onClick={() => setViewMode('items')}
                >
                  Item View
                </button>
                <button
                  style={{
                    ...styles.viewToggleButton,
                    borderLeft: 'none',
                    ...(viewMode === 'timeline' ? styles.viewToggleButtonActive : {}),
                  }}
                  onClick={() => setViewMode('timeline')}
                >
                  Timeline View
                </button>
                <button
                  style={{
                    ...styles.viewToggleButton,
                    borderLeft: 'none',
                    ...(viewMode === 'aggregated' ? styles.viewToggleButtonActive : {}),
                  }}
                  onClick={() => setViewMode('aggregated')}
                  disabled={!aggregatedData}
                  title={!aggregatedData ? 'No aggregated data available' : ''}
                >
                  Aggregated
                </button>
                <button
                  style={{
                    ...styles.viewToggleButton,
                    ...styles.viewToggleButtonLast,
                    ...(viewMode === 'vlm' ? styles.viewToggleButtonActive : {}),
                  }}
                  onClick={() => setViewMode('vlm')}
                  disabled={Object.keys(vlmDataByModel).length === 0}
                  title={Object.keys(vlmDataByModel).length === 0 ? 'No VLM results available' : ''}
                >
                  VLM Results
                </button>
              </div>
            )}
          </div>

          {!serverOnline && (
            <div style={{ ...styles.error, marginTop: '15px' }}>
              Video server is offline. Start it with: <code>node video-server.js</code>
            </div>
          )}

          {error && <div style={styles.error}>{error}</div>}
          {success && <div style={styles.success}>{success}</div>}
        </div>

        {/* Main Content */}
        {inventoryData && (
          <>
            <div style={styles.statsBar}>
              <strong>Participant:</strong> {inventoryData.participant} |
              <strong> Total Items:</strong> {inventoryData.total_lifecycle_items} |
              <strong> Selected:</strong> {inventoryData.selected_count} |
              <strong> Difficulty:</strong> LOW: {inventoryData.difficulty_breakdown?.LOW || 0},
              MID: {inventoryData.difficulty_breakdown?.MID || 0},
              HIGH: {inventoryData.difficulty_breakdown?.HIGH || 0} |
              <strong> Videos Available:</strong> {availableVideos.length}
            </div>

            {viewMode === 'items' ? (
              <div style={styles.mainLayout}>
                {/* Left Panel: Inventory List */}
                <InventoryItemList
                  items={inventoryData.items || []}
                  selectedItem={selectedItem}
                  onSelectItem={handleSelectItem}
                  onNarrationClick={handleNarrationClick}
                  narrationTimestamps={narrationTimestamps}
                />

                {/* Right Panel: Video Player */}
                <InventoryVideoPlayer
                  ref={videoRef}
                  videoUrl={currentVideoUrl}
                  videoId={currentVideo}
                  currentTime={currentTime}
                  onTimeUpdate={handleTimeUpdate}
                  onNarrationClick={handleNarrationClick}
                  selectedItem={selectedItem}
                  narrationTimestamps={narrationTimestamps}
                />
              </div>
            ) : viewMode === 'timeline' ? (
              <div style={{ ...styles.mainLayout, gridTemplateColumns: '1fr' }}>
                {/* Timeline View with integrated video player */}
                <TimelineView
                  ref={videoRef}
                  lifecycleData={lifecycleData}
                  onEventClick={handleNarrationClick}
                  narrationTimestamps={narrationTimestamps}
                  videoUrl={currentVideoUrl}
                  videoId={currentVideo}
                  currentTime={currentTime}
                  onTimeUpdate={handleTimeUpdate}
                />
              </div>
            ) : viewMode === 'aggregated' ? (
              <div style={{ ...styles.mainLayout, gridTemplateColumns: '1fr' }}>
                {/* Aggregated View with timestamp editing */}
                <AggregatedView
                  ref={videoRef}
                  aggregatedData={aggregatedData}
                  onLoadVideoAtTime={handleLoadVideoAtTime}
                  narrationTimestamps={narrationTimestamps}
                  videoUrl={currentVideoUrl}
                  videoId={currentVideo}
                  currentTime={currentTime}
                  onTimeUpdate={handleTimeUpdate}
                  onSave={handleSaveAggregated}
                  participant={participant}
                />
              </div>
            ) : (
              <div style={{ ...styles.mainLayout, gridTemplateColumns: '1fr' }}>
                {/* VLM Results View */}
                <VLMResultsView
                  ref={videoRef}
                  vlmData={vlmDataByModel[selectedVlmModel]}
                  availableModels={Object.keys(vlmDataByModel)}
                  selectedModel={selectedVlmModel}
                  onModelChange={setSelectedVlmModel}
                  onLoadVideoAtTime={handleLoadVideoAtTime}
                  videoUrl={currentVideoUrl}
                  videoId={currentVideo}
                  currentTime={currentTime}
                  onTimeUpdate={handleTimeUpdate}
                />
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}

export default InventoryApp;

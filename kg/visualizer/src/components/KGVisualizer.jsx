import React, { useState, useRef, useEffect } from "react";

const API_BASE = "http://localhost:5000/api";

export default function KGVisualizer() {
  // State
  const [snapshotDirs, setSnapshotDirs] = useState([]);
  const [selectedSnapshotDir, setSelectedSnapshotDir] = useState("");
  const [snapshots, setSnapshots] = useState([]);
  const [currentSnapshotIndex, setCurrentSnapshotIndex] = useState(0);
  const [currentSnapshot, setCurrentSnapshot] = useState(null);
  const [videoSrc, setVideoSrc] = useState("");
  const [currentTime, setCurrentTime] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);

  const videoRef = useRef(null);
  const snapshotListRef = useRef(null);

  // Load snapshot directories on mount
  useEffect(() => {
    fetch(`${API_BASE}/snapshots/directories`)
      .then(res => res.json())
      .then(data => {
        setSnapshotDirs(data);
        if (data.length > 0) {
          setSelectedSnapshotDir(data[0].name);
        }
      })
      .catch(err => console.error("Error loading snapshot directories:", err));
  }, []);

  // Load snapshots when directory is selected
  useEffect(() => {
    if (selectedSnapshotDir) {
      fetch(`${API_BASE}/snapshots/${selectedSnapshotDir}/metadata`)
        .then(res => res.json())
        .then(data => {
          setSnapshots(data);
          if (data.length > 0) {
            setCurrentSnapshotIndex(0);
            loadSnapshot(data[0].narration_id);
          }
        })
        .catch(err => console.error("Error loading snapshots:", err));
    }
  }, [selectedSnapshotDir]);

  // Load a specific snapshot
  const loadSnapshot = (narrationId) => {
    fetch(`${API_BASE}/snapshots/${selectedSnapshotDir}/${narrationId}`)
      .then(res => res.json())
      .then(data => {
        setCurrentSnapshot(data);

        // Auto-load video if video_id is present
        const videoId = data.snapshot_info.video_id;
        if (videoId) {
          // Extract participant ID from video_id (e.g., P01-20240202-110250 -> P01)
          const participantId = videoId.split('-')[0];
          const videoUrl = `${API_BASE}/video/${participantId}/${videoId}`;
          setVideoSrc(videoUrl);

          // Jump to the snapshot time
          setTimeout(() => {
            if (videoRef.current) {
              videoRef.current.currentTime = data.snapshot_info.start_time;
            }
          }, 100);
        }
      })
      .catch(err => console.error("Error loading snapshot:", err));
  };

  // Navigate to snapshot by index
  const goToSnapshot = (index) => {
    if (index >= 0 && index < snapshots.length) {
      setCurrentSnapshotIndex(index);
      loadSnapshot(snapshots[index].narration_id);
    }
  };

  // Navigate with keyboard
  useEffect(() => {
    const handleKeyPress = (e) => {
      if (e.key === 'ArrowLeft') {
        goToSnapshot(currentSnapshotIndex - 1);
      } else if (e.key === 'ArrowRight') {
        goToSnapshot(currentSnapshotIndex + 1);
      } else if (e.key === ' ') {
        e.preventDefault();
        if (videoRef.current) {
          if (isPlaying) {
            videoRef.current.pause();
          } else {
            videoRef.current.play();
          }
        }
      }
    };

    window.addEventListener('keydown', handleKeyPress);
    return () => window.removeEventListener('keydown', handleKeyPress);
  }, [currentSnapshotIndex, snapshots, isPlaying]);

  // Auto-scroll to current snapshot
  useEffect(() => {
    if (snapshotListRef.current) {
      const activeElement = snapshotListRef.current.querySelector('.snapshot-item.active');
      if (activeElement) {
        activeElement.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
      }
    }
  }, [currentSnapshotIndex]);

  const formatTime = (time) => {
    const minutes = Math.floor(time / 60);
    const seconds = (time % 60).toFixed(1);
    return `${minutes}:${seconds.padStart(4, '0')}`;
  };

  // Render KG state
  const renderKGState = () => {
    if (!currentSnapshot) return <p>No snapshot loaded</p>;

    const kg = currentSnapshot.kg_state;
    const foods = Object.values(kg.foods || {});
    const zones = Object.values(kg.zones || {});

    return (
      <div>
        <h3>Knowledge Graph State</h3>

        {/* Snapshot Info */}
        <div style={{
          backgroundColor: '#f8f9fa',
          padding: '10px',
          borderRadius: '4px',
          marginBottom: '15px'
        }}>
          <strong>Narration ID:</strong> {currentSnapshot.snapshot_info.narration_id}<br/>
          <strong>Time:</strong> {formatTime(currentSnapshot.snapshot_info.start_time)} - {formatTime(currentSnapshot.snapshot_info.end_time)}<br/>
          <strong>Status:</strong> <span style={{
            color: currentSnapshot.snapshot_info.update_success ? 'green' : 'red'
          }}>
            {currentSnapshot.snapshot_info.update_success ? '✓ Success' : '✗ Failed'}
          </span><br/>
          <strong>Narration:</strong> <i>{currentSnapshot.snapshot_info.narration_text}</i>
        </div>

        {/* Foods */}
        <div style={{ marginBottom: '20px' }}>
          <h4>🍕 Foods ({foods.length})</h4>
          {foods.length === 0 ? (
            <p style={{ color: '#999' }}>No foods tracked yet</p>
          ) : (
            <div style={{ maxHeight: '250px', overflowY: 'auto' }}>
              {foods.map(food => {
                const locationName = food.location && kg.zones[food.location]
                  ? kg.zones[food.location].name
                  : food.location === null ? 'in hand' : 'unknown';

                return (
                  <div key={food.food_id} style={{
                    border: '1px solid #dee2e6',
                    borderRadius: '4px',
                    padding: '10px',
                    marginBottom: '8px',
                    backgroundColor: 'white'
                  }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '5px' }}>
                      <strong>{food.name}</strong>
                      <span style={{
                        fontSize: '0.8em',
                        color: '#666',
                        backgroundColor: '#e9ecef',
                        padding: '2px 6px',
                        borderRadius: '3px'
                      }}>
                        {food.food_id}
                      </span>
                    </div>
                    <div style={{ fontSize: '0.9em', color: '#666' }}>
                      📍 {locationName} |
                      State: {food.state} |
                      Qty: {food.quantity}
                    </div>
                    <div style={{ fontSize: '0.85em', color: '#999', marginTop: '3px' }}>
                      {food.interaction_history.length} interaction(s)
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>

        {/* Zones */}
        <div>
          <h4>📦 Zones ({zones.length})</h4>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
            {zones.map(zone => (
              <span key={zone.zone_id} style={{
                backgroundColor: '#e7f3ff',
                padding: '5px 10px',
                borderRadius: '15px',
                fontSize: '0.85em',
                border: '1px solid #bee5eb'
              }}>
                {zone.name}
              </span>
            ))}
          </div>
        </div>
      </div>
    );
  };

  return (
    <div style={{ display: 'flex', height: 'calc(100vh - 60px)' }}>
      {/* Left Sidebar - Snapshot Navigation */}
      <div style={{
        width: '300px',
        borderRight: '1px solid #dee2e6',
        display: 'flex',
        flexDirection: 'column',
        backgroundColor: '#f8f9fa'
      }}>
        {/* Snapshot Directory Selector */}
        <div style={{ padding: '15px', borderBottom: '1px solid #dee2e6' }}>
          <label style={{ display: 'block', marginBottom: '5px', fontSize: '0.9em', fontWeight: 'bold' }}>
            Snapshot Directory:
          </label>
          <select
            value={selectedSnapshotDir}
            onChange={(e) => setSelectedSnapshotDir(e.target.value)}
            style={{
              width: '100%',
              padding: '8px',
              borderRadius: '4px',
              border: '1px solid #dee2e6'
            }}
          >
            {snapshotDirs.map(dir => (
              <option key={dir.name} value={dir.name}>
                {dir.name} ({dir.num_snapshots})
              </option>
            ))}
          </select>
        </div>

        {/* Navigation Controls */}
        <div style={{ padding: '15px', borderBottom: '1px solid #dee2e6' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' }}>
            <span style={{ fontSize: '0.9em', fontWeight: 'bold' }}>
              Snapshot {currentSnapshotIndex + 1} / {snapshots.length}
            </span>
          </div>
          <div style={{ display: 'flex', gap: '5px' }}>
            <button
              onClick={() => goToSnapshot(0)}
              disabled={currentSnapshotIndex === 0}
              style={{
                flex: 1,
                padding: '8px',
                fontSize: '0.85em',
                border: '1px solid #dee2e6',
                borderRadius: '4px',
                cursor: currentSnapshotIndex === 0 ? 'not-allowed' : 'pointer',
                backgroundColor: currentSnapshotIndex === 0 ? '#e9ecef' : 'white'
              }}
            >
              ⏮️ First
            </button>
            <button
              onClick={() => goToSnapshot(currentSnapshotIndex - 1)}
              disabled={currentSnapshotIndex === 0}
              style={{
                flex: 1,
                padding: '8px',
                fontSize: '0.85em',
                border: '1px solid #dee2e6',
                borderRadius: '4px',
                cursor: currentSnapshotIndex === 0 ? 'not-allowed' : 'pointer',
                backgroundColor: currentSnapshotIndex === 0 ? '#e9ecef' : 'white'
              }}
            >
              ◀️ Prev
            </button>
            <button
              onClick={() => goToSnapshot(currentSnapshotIndex + 1)}
              disabled={currentSnapshotIndex === snapshots.length - 1}
              style={{
                flex: 1,
                padding: '8px',
                fontSize: '0.85em',
                border: '1px solid #dee2e6',
                borderRadius: '4px',
                cursor: currentSnapshotIndex === snapshots.length - 1 ? 'not-allowed' : 'pointer',
                backgroundColor: currentSnapshotIndex === snapshots.length - 1 ? '#e9ecef' : 'white'
              }}
            >
              Next ▶️
            </button>
            <button
              onClick={() => goToSnapshot(snapshots.length - 1)}
              disabled={currentSnapshotIndex === snapshots.length - 1}
              style={{
                flex: 1,
                padding: '8px',
                fontSize: '0.85em',
                border: '1px solid #dee2e6',
                borderRadius: '4px',
                cursor: currentSnapshotIndex === snapshots.length - 1 ? 'not-allowed' : 'pointer',
                backgroundColor: currentSnapshotIndex === snapshots.length - 1 ? '#e9ecef' : 'white'
              }}
            >
              Last ⏭️
            </button>
          </div>
          <div style={{ marginTop: '10px', fontSize: '0.8em', color: '#666' }}>
            Use ← → arrow keys to navigate
          </div>
        </div>

        {/* Snapshot List */}
        <div ref={snapshotListRef} style={{ flex: 1, overflowY: 'auto', padding: '10px' }}>
          {snapshots.map((snapshot, index) => (
            <div
              key={snapshot.narration_id}
              className={`snapshot-item ${index === currentSnapshotIndex ? 'active' : ''}`}
              onClick={() => goToSnapshot(index)}
              style={{
                padding: '10px',
                marginBottom: '5px',
                borderRadius: '4px',
                cursor: 'pointer',
                backgroundColor: index === currentSnapshotIndex ? '#007bff' : 'white',
                color: index === currentSnapshotIndex ? 'white' : '#333',
                border: '1px solid #dee2e6',
                fontSize: '0.85em'
              }}
            >
              <div style={{ fontWeight: 'bold', marginBottom: '3px' }}>
                #{index + 1} {formatTime(snapshot.start_time)}
              </div>
              <div style={{ fontSize: '0.9em', opacity: 0.9 }}>
                {snapshot.narration_text.substring(0, 60)}...
              </div>
              <div style={{ marginTop: '3px', fontSize: '0.85em' }}>
                Foods: {snapshot.num_foods} | {snapshot.success ? '✓' : '✗'}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Main Content Area */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
        {/* Video Player */}
        <div style={{ height: '50%', backgroundColor: 'black', position: 'relative' }}>
          {videoSrc ? (
            <video
              ref={videoRef}
              src={videoSrc}
              controls
              style={{ width: '100%', height: '100%', objectFit: 'contain' }}
              onTimeUpdate={() => setCurrentTime(videoRef.current?.currentTime || 0)}
              onPlay={() => setIsPlaying(true)}
              onPause={() => setIsPlaying(false)}
            />
          ) : (
            <div style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              height: '100%',
              color: 'white',
              fontSize: '1.2em'
            }}>
              No video loaded
            </div>
          )}
        </div>

        {/* KG State Display */}
        <div style={{
          height: '50%',
          overflowY: 'auto',
          padding: '20px',
          backgroundColor: 'white'
        }}>
          {renderKGState()}
        </div>
      </div>
    </div>
  );
}

import React, { useState, useRef, useEffect } from "react";

export default function ClusterVisualizer() {
  const [videoSrc, setVideoSrc] = useState("");
  const [metadata, setMetadata] = useState(null);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [playbackRate, setPlaybackRate] = useState(1);
  const [selectedNode, setSelectedNode] = useState(null);
  const [expandedNodes, setExpandedNodes] = useState(new Set());

  const videoRef = useRef(null);

  // Convert frame number to video playtime (15 fps)
  const frameToTime = (frameNumber) => {
    return frameNumber / 15.0;
  };

  const formatTime = (time) => {
    const minutes = Math.floor(time / 60);
    const seconds = Math.floor(time % 60);
    const milliseconds = Math.floor((time % 1) * 1000);
    return `${minutes}:${seconds.toString().padStart(2, '0')}.${milliseconds.toString().padStart(3, '0')}`;
  };

  const jumpToTime = (time) => {
    if (videoRef.current) {
      videoRef.current.currentTime = time;
    }
  };

  const handleVideoUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      const url = URL.createObjectURL(file);
      setVideoSrc(url);
    }
  };

  const handleMetadataUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        try {
          const data = JSON.parse(e.target.result);
          setMetadata(data);
        } catch (error) {
          console.error('Error parsing metadata:', error);
          alert('Invalid JSON file. Please select a valid metadata.json file.');
        }
      };
      reader.readAsText(file);
    }
  };

  useEffect(() => {
    if (videoRef.current) {
      videoRef.current.playbackRate = playbackRate;
    }
  }, [playbackRate]);

  const handleTimeUpdate = () => {
    if (videoRef.current) {
      setCurrentTime(videoRef.current.currentTime);
    }
  };

  const handleLoadedMetadata = () => {
    if (videoRef.current) {
      setDuration(videoRef.current.duration);
    }
  };

  const toggleNodeExpansion = (nodeId) => {
    const newExpanded = new Set(expandedNodes);
    if (newExpanded.has(nodeId)) {
      newExpanded.delete(nodeId);
    } else {
      newExpanded.add(nodeId);
    }
    setExpandedNodes(newExpanded);
  };

  const getNodeScenes = (nodeId) => {
    if (!metadata || !metadata.scenes) return [];
    return metadata.scenes.filter(scene => scene.label === nodeId);
  };

  const getNodeColor = (nodeId) => {
    const colors = [
      '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
      '#DDA0DD', '#98D8C8', '#FFA07A', '#87CEEB', '#F0E68C',
      '#FFB6C1', '#20B2AA', '#87CEFA', '#DEB887', '#F5DEB3',
      '#FF69B4', '#00CED1', '#9370DB', '#3CB371', '#F4A460'
    ];
    return colors[nodeId % colors.length];
  };

  return (
    <div style={{ padding: "20px", display: "flex", gap: "20px", height: "calc(100vh - 100px)" }}>
      {/* Video Player */}
      <div style={{ flex: 2, display: "flex", flexDirection: "column" }}>
        <div style={{ marginBottom: "15px", padding: "15px", backgroundColor: "#f8f9fa", borderRadius: "6px", border: "1px solid #dee2e6" }}>
          <div style={{ marginBottom: "10px" }}>
            <label style={{ display: "block", marginBottom: "5px", fontWeight: "bold", color: "#495057" }}>
              Upload Video File:
            </label>
            <input
              type="file"
              accept="video/*"
              onChange={handleVideoUpload}
              style={{
                padding: "8px",
                border: "1px solid #ced4da",
                borderRadius: "4px",
                width: "100%",
                backgroundColor: "white"
              }}
            />
            <small style={{ color: "#6c757d", fontSize: "0.85em" }}>
              Select the video file (e.g., 20230828_s0_kaylee_johnson_act4_lt2qge_preview_rgb.mp4)
            </small>
          </div>

          <div>
            <label style={{ display: "block", marginBottom: "5px", fontWeight: "bold", color: "#495057" }}>
              Upload Cluster Metadata:
            </label>
            <input
              type="file"
              accept=".json"
              onChange={handleMetadataUpload}
              style={{
                padding: "8px",
                border: "1px solid #ced4da",
                borderRadius: "4px",
                width: "100%",
                backgroundColor: "white"
              }}
            />
            <small style={{ color: "#6c757d", fontSize: "0.85em" }}>
              Select the metadata.json file from clustering results
            </small>
          </div>
        </div>

        {videoSrc && (
          <>
            <video
              src={videoSrc}
              controls
              ref={videoRef}
              onTimeUpdate={handleTimeUpdate}
              onLoadedMetadata={handleLoadedMetadata}
              style={{ width: "100%", marginBottom: "10px", flex: 1 }}
            />

            {/* Video Controls */}
            <div style={{
              padding: "10px",
              backgroundColor: "#f8f9fa",
              borderRadius: "4px",
              border: "1px solid #dee2e6",
              marginBottom: "10px"
            }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "5px" }}>
                <span style={{ fontWeight: "bold" }}>Current Time: {formatTime(currentTime)}</span>
                <span style={{ color: "#666" }}>Duration: {formatTime(duration)}</span>
              </div>
              <div style={{
                width: "100%",
                height: "8px",
                backgroundColor: "#e9ecef",
                borderRadius: "4px",
                overflow: "hidden"
              }}>
                <div style={{
                  width: duration > 0 ? `${(currentTime / duration) * 100}%` : "0%",
                  height: "100%",
                  backgroundColor: "#007bff",
                  transition: "width 0.1s ease"
                }} />
              </div>
            </div>

            {/* Playback Rate */}
            <div style={{ marginBottom: "10px" }}>
              Playback Rate: {" "}
              {[0.5, 1, 2, 4].map((r) => (
                <button
                  key={r}
                  style={{
                    margin: "0 5px",
                    padding: "4px 8px",
                    backgroundColor: playbackRate === r ? "#333" : "#eee",
                    color: playbackRate === r ? "white" : "black",
                    border: "1px solid #ccc",
                    borderRadius: "3px",
                    cursor: "pointer"
                  }}
                  onClick={() => setPlaybackRate(r)}
                >
                  {r}x
                </button>
              ))}
            </div>
          </>
        )}
      </div>

      {/* Cluster Navigation */}
      <div style={{ flex: 1, overflowY: "auto" }}>
        <h3 style={{ marginTop: 0, marginBottom: "15px" }}>Cluster Nodes</h3>

        {metadata && metadata.nodes ? (
          <div>
            {metadata.nodes.map((nodeId) => {
              const nodeScenes = getNodeScenes(nodeId);
              const isExpanded = expandedNodes.has(nodeId);
              const nodeColor = getNodeColor(nodeId);

              return (
                <div key={nodeId} style={{ marginBottom: "10px" }}>
                  <div
                    style={{
                      padding: "8px 12px",
                      backgroundColor: nodeColor + "20",
                      border: `2px solid ${nodeColor}`,
                      borderRadius: "6px",
                      cursor: "pointer",
                      display: "flex",
                      justifyContent: "space-between",
                      alignItems: "center"
                    }}
                    onClick={() => toggleNodeExpansion(nodeId)}
                  >
                    <div>
                      <strong style={{ color: nodeColor }}>Node {nodeId}</strong>
                      <div style={{ fontSize: "0.8em", color: "#666" }}>
                        {nodeScenes.length} scene{nodeScenes.length !== 1 ? 's' : ''}
                      </div>
                    </div>
                    <span style={{ fontSize: "1.2em", color: nodeColor }}>
                      {isExpanded ? '▼' : '▶'}
                    </span>
                  </div>

                  {isExpanded && nodeScenes.length > 0 && (
                    <div style={{ marginTop: "5px", marginLeft: "10px" }}>
                      {nodeScenes.map((scene, index) => {
                        const startTime = frameToTime(scene.start);
                        const endTime = frameToTime(scene.end);
                        const sceneDuration = endTime - startTime;

                        return (
                          <div
                            key={index}
                            style={{
                              padding: "6px 10px",
                              margin: "3px 0",
                              backgroundColor: "white",
                              border: "1px solid #ddd",
                              borderRadius: "4px",
                              cursor: "pointer",
                              fontSize: "0.9em",
                              transition: "background-color 0.2s"
                            }}
                            onMouseEnter={(e) => e.target.style.backgroundColor = "#f5f5f5"}
                            onMouseLeave={(e) => e.target.style.backgroundColor = "white"}
                            onClick={() => jumpToTime(startTime)}
                          >
                            <div style={{ fontWeight: "bold" }}>
                              Scene {index + 1}
                            </div>
                            <div style={{ color: "#666" }}>
                              {formatTime(startTime)} - {formatTime(endTime)}
                            </div>
                            <div style={{ color: "#888", fontSize: "0.8em" }}>
                              Duration: {formatTime(sceneDuration)} | Frames: {scene.start}-{scene.end}
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        ) : (
          <div style={{
            padding: "20px",
            textAlign: "center",
            color: "#666",
            border: "2px dashed #ccc",
            borderRadius: "8px"
          }}>
            {metadata === null ? "Upload cluster metadata to load visualization" : "No cluster data found"}
          </div>
        )}

        {/* Statistics */}
        {metadata && metadata.scene_stats && (
          <div style={{ marginTop: "20px", padding: "10px", backgroundColor: "#f8f9fa", borderRadius: "4px" }}>
            <h4 style={{ margin: "0 0 10px 0" }}>Cluster Statistics</h4>
            <div style={{ fontSize: "0.9em" }}>
              <div>Total Nodes: {metadata.nodes?.length || 0}</div>
              <div>Total Scenes: {metadata.scenes?.length || 0}</div>
              <div>Total Transitions: {metadata.transitions?.length || 0}</div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
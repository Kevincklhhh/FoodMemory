import React, { useState, useRef, useEffect } from "react";

export default function ActivityGraphVisualizer() {
  const [videoSrc, setVideoSrc] = useState("");
  const [activityGraph, setActivityGraph] = useState(null);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [playbackRate, setPlaybackRate] = useState(1);
  const [selectedZone, setSelectedZone] = useState(null);
  const [selectedInteraction, setSelectedInteraction] = useState(null);
  const [expandedZones, setExpandedZones] = useState(new Set());

  const videoRef = useRef(null);

  // Convert frame number to video playtime (assumes 1fps frame extraction)
  const frameToTime = (frameNumber) => {
    return frameNumber / 1.0; // 1fps
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

  const handleActivityGraphUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        try {
          const data = JSON.parse(e.target.result);
          setActivityGraph(data);
          console.log('Activity graph loaded:', data);
        } catch (error) {
          console.error('Error parsing activity graph:', error);
          alert('Invalid JSON file. Please select a valid activity graph JSON file.');
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

  const toggleZoneExpansion = (zoneId) => {
    const newExpanded = new Set(expandedZones);
    if (newExpanded.has(zoneId)) {
      newExpanded.delete(zoneId);
    } else {
      newExpanded.add(zoneId);
    }
    setExpandedZones(newExpanded);
  };

  const getZoneColor = (zoneIndex) => {
    const colors = [
      '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
      '#DDA0DD', '#98D8C8', '#FFA07A', '#87CEEB', '#F0E68C',
      '#FFB6C1', '#20B2AA', '#87CEFA', '#DEB887', '#F5DEB3',
      '#FF69B4', '#00CED1', '#9370DB', '#3CB371', '#F4A460'
    ];
    return colors[zoneIndex % colors.length];
  };

  const getActionColor = (action) => {
    const actionColors = {
      'taking': '#FF6B6B',
      'putting': '#4ECDC4',
      'put': '#4ECDC4',
      'puts': '#4ECDC4',
      'touching': '#45B7D1',
      'folding': '#96CEB4',
      'pouring': '#FFEAA7',
      'takes': '#FF8A8A',
      'default': '#999999'
    };
    return actionColors[action.toLowerCase()] || actionColors['default'];
  };

  const jumpToInteraction = (temporal_window) => {
    if (temporal_window && temporal_window.length >= 2) {
      const startTime = frameToTime(temporal_window[0]);
      jumpToTime(startTime);
      setSelectedInteraction(temporal_window);
    }
  };

  const renderActivityGraph = () => {
    if (!activityGraph) return null;

    const videoName = Object.keys(activityGraph)[0];
    const graphData = activityGraph[videoName];

    if (!graphData || !graphData["Layer2: activity_zones"]) return null;

    const zones = graphData["Layer2: activity_zones"];

    return (
      <div>
        <h3 style={{ marginTop: 0, marginBottom: "15px" }}>Activity Zones</h3>

        {zones.map((zone, zoneIndex) => {
          const isExpanded = expandedZones.has(zone.zone_id);
          const zoneColor = getZoneColor(zoneIndex);
          const interactions = zone["Layer1: Human and object"];

          return (
            <div key={zone.zone_id} style={{ marginBottom: "15px" }}>
              {/* Zone Header */}
              <div
                style={{
                  padding: "12px 16px",
                  backgroundColor: zoneColor + "20",
                  border: `2px solid ${zoneColor}`,
                  borderRadius: "8px",
                  cursor: "pointer",
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "center"
                }}
                onClick={() => toggleZoneExpansion(zone.zone_id)}
              >
                <div>
                  <div style={{ fontWeight: "bold", color: zoneColor, fontSize: "1.1em" }}>
                    {zone.location_name || `Zone ${zoneIndex}`}
                  </div>
                  <div style={{ fontSize: "0.85em", color: "#666", marginTop: "2px" }}>
                    {Object.keys(interactions || {}).length} interactions
                  </div>
                </div>
                <span style={{ fontSize: "1.2em", color: zoneColor }}>
                  {isExpanded ? '▼' : '▶'}
                </span>
              </div>

              {/* Interactions */}
              {isExpanded && interactions && (
                <div style={{ marginTop: "8px", marginLeft: "12px" }}>
                  {Object.entries(interactions).map(([interactionKey, interaction], index) => {
                    const actionColor = getActionColor(interaction.action);
                    const startTime = frameToTime(interaction.temporal_window[0]);
                    const endTime = frameToTime(interaction.temporal_window[1]);
                    const duration = endTime - startTime;
                    const isSelected = selectedInteraction &&
                      selectedInteraction[0] === interaction.temporal_window[0] &&
                      selectedInteraction[1] === interaction.temporal_window[1];

                    return (
                      <div
                        key={`${zone.zone_id}-${index}`}
                        style={{
                          padding: "10px 12px",
                          margin: "4px 0",
                          backgroundColor: isSelected ? "#e3f2fd" : "white",
                          border: isSelected ? "2px solid #2196f3" : "1px solid #ddd",
                          borderRadius: "6px",
                          cursor: "pointer",
                          transition: "all 0.2s ease"
                        }}
                        onMouseEnter={(e) => {
                          if (!isSelected) {
                            e.target.style.backgroundColor = "#f8f9fa";
                            e.target.style.borderColor = actionColor;
                          }
                        }}
                        onMouseLeave={(e) => {
                          if (!isSelected) {
                            e.target.style.backgroundColor = "white";
                            e.target.style.borderColor = "#ddd";
                          }
                        }}
                        onClick={() => jumpToInteraction(interaction.temporal_window)}
                      >
                        {/* Interaction Header */}
                        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "start", marginBottom: "6px" }}>
                          <div style={{ flex: 1 }}>
                            <div style={{ fontWeight: "bold", color: "#333" }}>
                              <span style={{
                                display: "inline-block",
                                padding: "2px 8px",
                                backgroundColor: actionColor,
                                color: "white",
                                borderRadius: "12px",
                                fontSize: "0.8em",
                                marginRight: "8px"
                              }}>
                                {interaction.action}
                              </span>
                              {interaction.object}
                            </div>
                            <div style={{ fontSize: "0.85em", color: "#666", marginTop: "2px" }}>
                              {interaction.spatial_relation} hand{interaction.spatial_relation === 'both' ? 's' : ''}
                            </div>
                          </div>
                          <div style={{ textAlign: "right", fontSize: "0.8em", color: "#666" }}>
                            <div>{formatTime(startTime)} - {formatTime(endTime)}</div>
                            <div>Duration: {formatTime(duration)}</div>
                          </div>
                        </div>

                        {/* Interaction Description */}
                        {interaction.interaction && (
                          <div style={{
                            fontSize: "0.85em",
                            color: "#555",
                            lineHeight: "1.4",
                            backgroundColor: "#f8f9fa",
                            padding: "8px",
                            borderRadius: "4px",
                            marginTop: "6px"
                          }}>
                            {interaction.interaction.length > 200
                              ? interaction.interaction.substring(0, 200) + "..."
                              : interaction.interaction
                            }
                          </div>
                        )}

                        {/* Timeline indicator */}
                        <div style={{ marginTop: "6px", fontSize: "0.75em", color: "#888" }}>
                          Frames: {interaction.temporal_window[0]} - {interaction.temporal_window[1]}
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
    );
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
              Select the video file (e.g., *_preview_rgb.mp4)
            </small>
          </div>

          <div>
            <label style={{ display: "block", marginBottom: "5px", fontWeight: "bold", color: "#495057" }}>
              Upload Activity Graph:
            </label>
            <input
              type="file"
              accept=".json"
              onChange={handleActivityGraphUpload}
              style={{
                padding: "8px",
                border: "1px solid #ced4da",
                borderRadius: "4px",
                width: "100%",
                backgroundColor: "white"
              }}
            />
            <small style={{ color: "#6c757d", fontSize: "0.85em" }}>
              Select the activity_graph_*.json file
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

      {/* Activity Graph Navigation */}
      <div style={{ flex: 1, overflowY: "auto" }}>
        {activityGraph ? (
          renderActivityGraph()
        ) : (
          <div style={{
            padding: "20px",
            textAlign: "center",
            color: "#666",
            border: "2px dashed #ccc",
            borderRadius: "8px"
          }}>
            Upload activity graph JSON to load visualization
          </div>
        )}

        {/* Statistics */}
        {activityGraph && (
          <div style={{ marginTop: "20px", padding: "10px", backgroundColor: "#f8f9fa", borderRadius: "4px" }}>
            <h4 style={{ margin: "0 0 10px 0" }}>Activity Statistics</h4>
            <div style={{ fontSize: "0.9em" }}>
              {(() => {
                const videoName = Object.keys(activityGraph)[0];
                const graphData = activityGraph[videoName];
                const zones = graphData?.["Layer2: activity_zones"] || [];
                const totalInteractions = zones.reduce((sum, zone) =>
                  sum + Object.keys(zone["Layer1: Human and object"] || {}).length, 0
                );

                return (
                  <>
                    <div>Total Zones: {zones.length}</div>
                    <div>Total Interactions: {totalInteractions}</div>
                    <div>Video: {videoName}</div>
                  </>
                );
              })()}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
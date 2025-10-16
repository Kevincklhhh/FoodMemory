import React, { useState, useRef, useEffect } from "react";

export default function ContextCaptionsViewer() {
  const [videoSrc, setVideoSrc] = useState("");
  const [captions, setCaptions] = useState(null);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [videoName, setVideoName] = useState("");

  const videoRef = useRef(null);
  const captionsListRef = useRef(null);

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

  const handleCaptionsUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        try {
          const data = JSON.parse(e.target.result);
          setCaptions(data);

          // Extract video name from the data (first key)
          const videoKeys = Object.keys(data);
          if (videoKeys.length > 0) {
            setVideoName(videoKeys[0]);
          }

          console.log('Context captions loaded:', data);
        } catch (error) {
          console.error('Error parsing context captions JSON:', error);
          alert('Error parsing context captions file. Please ensure it is a valid JSON file.');
        }
      };
      reader.readAsText(file);
    }
  };

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

  // Get current caption based on video time
  const getCurrentCaption = () => {
    if (!captions || !videoName) return null;

    const captionsData = captions[videoName];
    if (!captionsData) return null;

    // Since captions are indexed by frame number and frames are extracted at 1fps,
    // we can use Math.floor(currentTime) as the frame index
    const frameIndex = Math.floor(currentTime);
    const captionData = captionsData[frameIndex.toString()];

    return captionData || null;
  };

  // Get all captions as an array for the sidebar
  const getCaptionsArray = () => {
    if (!captions || !videoName) return [];

    const captionsData = captions[videoName];
    if (!captionsData) return [];

    return Object.entries(captionsData)
      .map(([frameIndex, captionData]) => ({
        frameIndex: parseInt(frameIndex),
        time: parseInt(frameIndex), // 1fps means frame index = time in seconds
        caption: captionData.caption || captionData, // Support both new and old formats
        context: captionData.context || null
      }))
      .sort((a, b) => a.frameIndex - b.frameIndex);
  };

  // Auto-scroll to current caption in the sidebar
  useEffect(() => {
    if (captionsListRef.current && currentTime > 0) {
      const currentFrameIndex = Math.floor(currentTime);
      const captionElement = captionsListRef.current.querySelector(`[data-frame="${currentFrameIndex}"]`);
      if (captionElement) {
        captionElement.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
      }
    }
  }, [currentTime]);

  const currentCaptionData = getCurrentCaption();
  const captionsArray = getCaptionsArray();
  const currentFrameIndex = Math.floor(currentTime);

  // Extract current caption and context
  const currentCaption = currentCaptionData?.caption || currentCaptionData;
  const currentContext = currentCaptionData?.context;

  return (
    <div style={{ display: 'flex', height: '100vh', backgroundColor: '#f8f9fa' }}>
      {/* Left Panel - Video and Controls */}
      <div style={{
        flex: 1,
        padding: '20px',
        display: 'flex',
        flexDirection: 'column'
      }}>
        {/* Upload Section */}
        <div style={{
          backgroundColor: 'white',
          padding: '20px',
          borderRadius: '8px',
          marginBottom: '20px',
          boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
        }}>
          <h2 style={{ margin: '0 0 15px 0', fontSize: '1.2em' }}>Context-Aided Captions Viewer</h2>

          <div style={{ marginBottom: '15px' }}>
            <label style={{ display: 'block', marginBottom: '5px', fontWeight: 'bold' }}>
              Upload Video:
            </label>
            <input
              type="file"
              accept="video/*"
              onChange={handleVideoUpload}
              style={{
                width: '100%',
                padding: '8px',
                border: '1px solid #ddd',
                borderRadius: '4px'
              }}
            />
          </div>

          <div>
            <label style={{ display: 'block', marginBottom: '5px', fontWeight: 'bold' }}>
              Upload Context Captions JSON:
            </label>
            <input
              type="file"
              accept=".json"
              onChange={handleCaptionsUpload}
              style={{
                width: '100%',
                padding: '8px',
                border: '1px solid #ddd',
                borderRadius: '4px'
              }}
            />
            <p style={{ margin: '5px 0 0 0', fontSize: '0.8em', color: '#666' }}>
              Upload: {videoName}_captions_context.json
            </p>
          </div>

          {videoName && (
            <p style={{ margin: '10px 0 0 0', color: '#666', fontSize: '0.9em' }}>
              Video: {videoName}
            </p>
          )}
        </div>

        {/* Video Player */}
        {videoSrc && (
          <div style={{
            backgroundColor: 'white',
            padding: '20px',
            borderRadius: '8px',
            marginBottom: '20px',
            boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
            flex: 1
          }}>
            <video
              ref={videoRef}
              src={videoSrc}
              controls
              onTimeUpdate={handleTimeUpdate}
              onLoadedMetadata={handleLoadedMetadata}
              style={{
                width: '100%',
                maxHeight: '60vh',
                backgroundColor: '#000'
              }}
            />

            {/* Video Info */}
            <div style={{
              marginTop: '15px',
              padding: '10px',
              backgroundColor: '#f8f9fa',
              borderRadius: '4px',
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center'
            }}>
              <span>
                Time: {formatTime(currentTime)} / {formatTime(duration)}
              </span>
              <span>
                Frame: {currentFrameIndex}
              </span>
            </div>
          </div>
        )}

        {/* Current Caption Display */}
        {currentCaption && (
          <div style={{
            backgroundColor: 'white',
            padding: '20px',
            borderRadius: '8px',
            boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
            border: '2px solid #28a745',
            marginBottom: '15px'
          }}>
            <h3 style={{ margin: '0 0 10px 0', color: '#28a745' }}>
              Current Caption (Frame {currentFrameIndex})
            </h3>
            <p style={{ margin: 0, fontSize: '1.1em', lineHeight: '1.5' }}>
              {currentCaption}
            </p>
          </div>
        )}

        {/* Current Context Display */}
        {currentContext && (
          <div style={{
            backgroundColor: 'white',
            padding: '20px',
            borderRadius: '8px',
            boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
            border: '2px solid #ffc107'
          }}>
            <h3 style={{ margin: '0 0 15px 0', color: '#ffc107' }}>
              Context Information
            </h3>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '15px' }}>
              <div>
                <h4 style={{ margin: '0 0 8px 0', fontSize: '0.9em', color: '#666' }}>
                  Location & Time
                </h4>
                <p style={{ margin: '0 0 5px 0', fontSize: '0.9em' }}>
                  <strong>Location:</strong> {currentContext.current_location || 'Unknown'}
                </p>
                <p style={{ margin: '0', fontSize: '0.9em' }}>
                  <strong>Time:</strong> {currentContext.current_time || 'N/A'}
                </p>
              </div>

              <div>
                <h4 style={{ margin: '0 0 8px 0', fontSize: '0.9em', color: '#666' }}>
                  Current Activity
                </h4>
                <p style={{ margin: '0', fontSize: '0.9em' }}>
                  <strong>Interaction ID:</strong> {currentContext.current_interaction_id || 'None'}
                </p>
              </div>
            </div>

            {currentContext.recent_interactions && currentContext.recent_interactions.length > 0 && (
              <div style={{ marginTop: '15px' }}>
                <h4 style={{ margin: '0 0 10px 0', fontSize: '0.9em', color: '#666' }}>
                  Recent Activities
                </h4>
                <div style={{
                  maxHeight: '120px',
                  overflowY: 'auto',
                  border: '1px solid #e9ecef',
                  borderRadius: '4px',
                  padding: '8px'
                }}>
                  {currentContext.recent_interactions.map((interaction, index) => (
                    <div key={index} style={{
                      padding: '6px 0',
                      borderBottom: index < currentContext.recent_interactions.length - 1 ? '1px solid #f8f9fa' : 'none',
                      fontSize: '0.85em'
                    }}>
                      <strong style={{ color: '#007bff' }}>{interaction.time}:</strong> {interaction.action} {interaction.object}
                      {interaction.source && <span style={{ color: '#666' }}> from {interaction.source}</span>}
                      {interaction.destination && <span style={{ color: '#666' }}> to {interaction.destination}</span>}
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Right Panel - Captions List */}
      <div style={{
        width: '450px',
        backgroundColor: 'white',
        borderLeft: '1px solid #dee2e6',
        display: 'flex',
        flexDirection: 'column'
      }}>
        <div style={{
          padding: '20px',
          borderBottom: '1px solid #dee2e6',
          backgroundColor: '#f8f9fa'
        }}>
          <h3 style={{ margin: 0 }}>
            Context Captions ({captionsArray.length})
          </h3>
        </div>

        <div
          ref={captionsListRef}
          style={{
            flex: 1,
            overflowY: 'auto',
            padding: '10px'
          }}
        >
          {captionsArray.map((item) => (
            <div
              key={item.frameIndex}
              data-frame={item.frameIndex}
              onClick={() => jumpToTime(item.time)}
              style={{
                padding: '12px',
                margin: '8px 0',
                backgroundColor: currentFrameIndex === item.frameIndex ? '#e8f5e8' : '#f8f9fa',
                border: currentFrameIndex === item.frameIndex ? '2px solid #28a745' : '1px solid #dee2e6',
                borderRadius: '6px',
                cursor: 'pointer',
                transition: 'all 0.2s ease',
              }}
              onMouseEnter={(e) => {
                if (currentFrameIndex !== item.frameIndex) {
                  e.target.style.backgroundColor = '#e9ecef';
                }
              }}
              onMouseLeave={(e) => {
                if (currentFrameIndex !== item.frameIndex) {
                  e.target.style.backgroundColor = '#f8f9fa';
                }
              }}
            >
              <div style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                marginBottom: '8px'
              }}>
                <strong style={{ color: '#28a745' }}>
                  Frame {item.frameIndex}
                </strong>
                <span style={{
                  fontSize: '0.8em',
                  color: '#666',
                  backgroundColor: '#fff',
                  padding: '2px 6px',
                  borderRadius: '3px',
                  border: '1px solid #ddd'
                }}>
                  {formatTime(item.time)}
                </span>
              </div>

              {/* Caption */}
              <p style={{
                margin: '0 0 8px 0',
                fontSize: '0.9em',
                lineHeight: '1.4',
                color: '#333',
                fontWeight: '500'
              }}>
                {item.caption}
              </p>

              {/* Context Summary */}
              {item.context && (
                <div style={{
                  padding: '8px',
                  backgroundColor: '#fff3cd',
                  borderRadius: '4px',
                  fontSize: '0.8em',
                  color: '#856404',
                  marginTop: '8px',
                  border: '1px solid #ffeaa7'
                }}>
                  {item.context.current_location && (
                    <div style={{ marginBottom: '3px' }}>
                      📍 <strong>{item.context.current_location}</strong>
                    </div>
                  )}
                  {item.context.recent_interactions && item.context.recent_interactions.length > 0 && (
                    <div style={{ fontSize: '0.75em' }}>
                      💡 {item.context.recent_interactions.length} recent activities
                    </div>
                  )}
                </div>
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
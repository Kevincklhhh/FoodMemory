import React, { useState, useRef, useEffect, useCallback } from 'react';
import { formatTimestamp } from '../utils/narrationParser';

const VIDEO_SERVER = 'http://localhost:4001';

const styles = {
  container: {
    display: 'flex',
    flexDirection: 'column',
    height: '100%',
    minHeight: '400px',
    backgroundColor: '#1a1a1a',
    borderRadius: '8px',
    overflow: 'hidden',
  },
  mainViewer: {
    flex: 1,
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    padding: '10px',
    minHeight: '300px',
    position: 'relative',
    overflow: 'hidden',
  },
  frameImage: {
    maxWidth: '100%',
    maxHeight: '100%',
    objectFit: 'contain',
    borderRadius: '4px',
  },
  frameInfo: {
    position: 'absolute',
    bottom: '15px',
    left: '15px',
    right: '15px',
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: '8px 12px',
    backgroundColor: 'rgba(0,0,0,0.7)',
    borderRadius: '6px',
    color: 'white',
    fontSize: '12px',
  },
  frameInfoLeft: {
    display: 'flex',
    alignItems: 'center',
    gap: '15px',
  },
  frameNumber: {
    fontWeight: 'bold',
    fontFamily: 'monospace',
  },
  timestamp: {
    color: '#4CAF50',
    fontFamily: 'monospace',
  },
  detectionInfo: {
    display: 'flex',
    gap: '8px',
    alignItems: 'center',
  },
  badge: {
    padding: '2px 6px',
    borderRadius: '3px',
    fontSize: '10px',
    fontWeight: '500',
  },
  handsBadge: {
    backgroundColor: '#2196F3',
    color: 'white',
  },
  contactBadge: {
    backgroundColor: '#4CAF50',
    color: 'white',
  },
  noContactBadge: {
    backgroundColor: '#9E9E9E',
    color: 'white',
  },
  graspBadge: {
    backgroundColor: '#FF9800',
    color: 'white',
  },
  // Filmstrip
  filmstripContainer: {
    backgroundColor: '#2a2a2a',
    borderTop: '1px solid #444',
    padding: '10px',
  },
  filmstripControls: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    marginBottom: '8px',
  },
  navButton: {
    padding: '6px 12px',
    backgroundColor: '#444',
    color: 'white',
    border: 'none',
    borderRadius: '4px',
    cursor: 'pointer',
    fontSize: '14px',
    fontWeight: 'bold',
  },
  navButtonDisabled: {
    opacity: 0.5,
    cursor: 'not-allowed',
  },
  filmstripInfo: {
    flex: 1,
    textAlign: 'center',
    color: '#aaa',
    fontSize: '11px',
  },
  filmstripScroll: {
    display: 'flex',
    gap: '4px',
    overflowX: 'auto',
    paddingBottom: '5px',
    scrollBehavior: 'smooth',
  },
  thumbnail: {
    width: '80px',
    height: '60px',
    objectFit: 'cover',
    borderRadius: '3px',
    cursor: 'pointer',
    border: '2px solid transparent',
    flexShrink: 0,
    transition: 'border-color 0.15s, opacity 0.15s',
    opacity: 0.7,
  },
  thumbnailSelected: {
    borderColor: '#4CAF50',
    opacity: 1,
  },
  thumbnailHasHands: {
    borderColor: '#2196F3',
  },
  thumbnailLabel: {
    position: 'absolute',
    bottom: '2px',
    left: '2px',
    right: '2px',
    backgroundColor: 'rgba(0,0,0,0.7)',
    color: 'white',
    fontSize: '8px',
    padding: '1px 2px',
    borderRadius: '2px',
    textAlign: 'center',
  },
  thumbnailWrapper: {
    position: 'relative',
    flexShrink: 0,
  },
  // Loading states
  noFrames: {
    color: '#888',
    textAlign: 'center',
    padding: '40px',
    fontSize: '14px',
  },
  loadingOverlay: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: 'rgba(0,0,0,0.5)',
    color: 'white',
    fontSize: '14px',
  },
  // Toggle bar
  toggleBar: {
    display: 'flex',
    gap: '4px',
    marginLeft: 'auto',
  },
  toggleButton: {
    padding: '4px 8px',
    fontSize: '10px',
    border: '1px solid #555',
    borderRadius: '3px',
    cursor: 'pointer',
    backgroundColor: '#333',
    color: '#aaa',
  },
  toggleButtonActive: {
    backgroundColor: '#4CAF50',
    borderColor: '#4CAF50',
    color: 'white',
  },
};

const FrameGallery = ({
  hands23Data,
  videoId,
  currentTime,
  onTimeChange,
  participant: propParticipant,
  onSwitchToVideo,
}) => {
  // Extract participant from videoId if not provided, or use provided participant
  // This handles failure cases that span multiple participants
  const participantFromVideo = videoId?.split('-')[0];
  const participant = participantFromVideo || propParticipant;
  const [currentFrameIndex, setCurrentFrameIndex] = useState(0);
  const [showVisualization, setShowVisualization] = useState(true);
  const [imageLoading, setImageLoading] = useState(false);
  const filmstripRef = useRef(null);

  // Get frames for the current video (memoized to prevent dependency issues)
  const videoData = hands23Data?.videos?.find(v => v.video_id === videoId);
  const frames = React.useMemo(() => videoData?.frames || [], [videoData]);

  // Find nearest frame when currentTime changes
  useEffect(() => {
    if (frames.length === 0) return;

    let nearestIdx = 0;
    let minDiff = Infinity;
    frames.forEach((frame, idx) => {
      const diff = Math.abs(frame.timestamp - currentTime);
      if (diff < minDiff) {
        minDiff = diff;
        nearestIdx = idx;
      }
    });

    if (nearestIdx !== currentFrameIndex) {
      setCurrentFrameIndex(nearestIdx);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [currentTime, frames]);

  // Scroll filmstrip to center current frame
  useEffect(() => {
    if (filmstripRef.current && frames.length > 0) {
      const container = filmstripRef.current;
      const thumb = container.children[currentFrameIndex];
      if (thumb) {
        thumb.scrollIntoView({ behavior: 'smooth', inline: 'center', block: 'nearest' });
      }
    }
  }, [currentFrameIndex, frames]);

  const handleFrameClick = useCallback((idx) => {
    setCurrentFrameIndex(idx);
    const frame = frames[idx];
    if (frame && onTimeChange) {
      onTimeChange(frame.timestamp);
    }
  }, [frames, onTimeChange]);

  const handlePrevFrame = useCallback(() => {
    if (currentFrameIndex > 0) {
      handleFrameClick(currentFrameIndex - 1);
    }
  }, [currentFrameIndex, handleFrameClick]);

  const handleNextFrame = useCallback(() => {
    if (currentFrameIndex < frames.length - 1) {
      handleFrameClick(currentFrameIndex + 1);
    }
  }, [currentFrameIndex, frames.length, handleFrameClick]);

  // Keyboard navigation
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === 'ArrowLeft') {
        handlePrevFrame();
        e.preventDefault();
      } else if (e.key === 'ArrowRight') {
        handleNextFrame();
        e.preventDefault();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [handlePrevFrame, handleNextFrame]);

  if (!videoId || !hands23Data) {
    return (
      <div style={styles.container}>
        <div style={styles.noFrames}>
          No hands23 detection data available.
          <br />
          <span style={{ fontSize: '12px', color: '#666' }}>
            Run the hands23 detector first.
          </span>
        </div>
      </div>
    );
  }

  if (!videoData) {
    return (
      <div style={styles.container}>
        <div style={styles.noFrames}>
          No frame data for video: {videoId}
          <br />
          <span style={{ fontSize: '12px', color: '#666' }}>
            Available videos: {hands23Data.videos?.map(v => v.video_id).join(', ') || 'none'}
          </span>
        </div>
      </div>
    );
  }

  const currentFrame = frames[currentFrameIndex];
  const totalHands = currentFrame?.num_hands || 0;

  // Get detection summary for badge display
  const detections = currentFrame?.detections || [];
  const hasContact = detections.some(d => d.contact_state === 'object_contact');
  const graspTypes = [...new Set(detections.map(d => d.grasp).filter(Boolean))];

  // Build image URLs
  // The frame_path already includes videoId/frames/filename, so we just need participant prefix
  // e.g. "P03-20240216-084005/frames/frame_00000000_t0.00s.jpg"
  // URL structure: /frames/{participant}/{frame_path} -> /frames/P03/P03-20240216-084005/frames/frame_00000000_t0.00s.jpg
  // But our server expects: /frames/{participant}/{videoId}/{filename}
  // So we need to split frame_path: videoId/frames/filename -> use only filename
  const getFrameUrl = (framePath) => {
    if (!framePath) return null;
    // frame_path format: "P03-20240216-084005/frames/frame_00000000_t0.00s.jpg"
    const parts = framePath.split('/');
    const videoIdFromPath = parts[0];
    const filename = parts.slice(2).join('/'); // skip videoId/frames/
    return `${VIDEO_SERVER}/frames/${participant}/${videoIdFromPath}/${filename}`;
  };

  const getVisualizationUrl = (visPath) => {
    if (!visPath) return null;
    // visualization_path format: "P03-20240216-084005/visualizations/vis_frame_00000000_t0.00s.jpg"
    const parts = visPath.split('/');
    const videoIdFromPath = parts[0];
    const filename = parts.slice(2).join('/'); // skip videoId/visualizations/
    return `${VIDEO_SERVER}/visualizations/${participant}/${videoIdFromPath}/${filename}`;
  };

  const imageUrl = showVisualization && currentFrame?.visualization_path
    ? getVisualizationUrl(currentFrame.visualization_path)
    : currentFrame?.frame_path
    ? getFrameUrl(currentFrame.frame_path)
    : null;

  return (
    <div style={styles.container}>
      {/* Main Frame Viewer */}
      <div style={styles.mainViewer}>
        {imageUrl ? (
          <>
            <img
              src={imageUrl}
              alt={`Frame ${currentFrameIndex}`}
              style={styles.frameImage}
              onLoadStart={() => setImageLoading(true)}
              onLoad={() => setImageLoading(false)}
              onError={() => setImageLoading(false)}
            />
            {imageLoading && (
              <div style={styles.loadingOverlay}>Loading...</div>
            )}
          </>
        ) : (
          <div style={styles.noFrames}>No frame available</div>
        )}

        {/* Frame Info Overlay */}
        {currentFrame && (
          <div style={styles.frameInfo}>
            <div style={styles.frameInfoLeft}>
              <span style={styles.frameNumber}>
                Frame {currentFrameIndex + 1} / {frames.length}
              </span>
              <span style={styles.timestamp}>
                {formatTimestamp(currentFrame.timestamp)}
              </span>
            </div>
            <div style={styles.detectionInfo}>
              <span style={{ ...styles.badge, ...styles.handsBadge }}>
                {totalHands} hand{totalHands !== 1 ? 's' : ''}
              </span>
              {hasContact ? (
                <span style={{ ...styles.badge, ...styles.contactBadge }}>
                  contact
                </span>
              ) : totalHands > 0 && (
                <span style={{ ...styles.badge, ...styles.noContactBadge }}>
                  no contact
                </span>
              )}
              {graspTypes.map(grasp => (
                <span key={grasp} style={{ ...styles.badge, ...styles.graspBadge }}>
                  {grasp}
                </span>
              ))}
              <div style={styles.toggleBar}>
                {onSwitchToVideo && (
                  <button
                    style={{
                      ...styles.toggleButton,
                      marginRight: '8px',
                      borderRadius: '4px',
                    }}
                    onClick={onSwitchToVideo}
                    title="Switch to video view"
                  >
                    Video
                  </button>
                )}
                <button
                  style={{
                    ...styles.toggleButton,
                    ...(showVisualization ? styles.toggleButtonActive : {}),
                  }}
                  onClick={() => setShowVisualization(true)}
                  title="Show HOI visualization"
                >
                  HOI
                </button>
                <button
                  style={{
                    ...styles.toggleButton,
                    ...(!showVisualization ? styles.toggleButtonActive : {}),
                  }}
                  onClick={() => setShowVisualization(false)}
                  title="Show raw frame"
                >
                  Raw
                </button>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Filmstrip */}
      <div style={styles.filmstripContainer}>
        <div style={styles.filmstripControls}>
          <button
            style={{
              ...styles.navButton,
              ...(currentFrameIndex === 0 ? styles.navButtonDisabled : {}),
            }}
            onClick={handlePrevFrame}
            disabled={currentFrameIndex === 0}
          >
            &lt;
          </button>
          <div style={styles.filmstripInfo}>
            Use arrow keys to navigate | Click thumbnail to jump
          </div>
          <button
            style={{
              ...styles.navButton,
              ...(currentFrameIndex >= frames.length - 1 ? styles.navButtonDisabled : {}),
            }}
            onClick={handleNextFrame}
            disabled={currentFrameIndex >= frames.length - 1}
          >
            &gt;
          </button>
        </div>
        <div style={styles.filmstripScroll} ref={filmstripRef}>
          {frames.map((frame, idx) => {
            const isSelected = idx === currentFrameIndex;
            const hasHands = frame.num_hands > 0;
            // Use raw frame for thumbnail (smaller file usually)
            const thumbUrl = getFrameUrl(frame.frame_path);

            return (
              <div key={idx} style={styles.thumbnailWrapper}>
                {thumbUrl && (
                  <img
                    src={thumbUrl}
                    alt={`Frame ${idx}`}
                    style={{
                      ...styles.thumbnail,
                      ...(isSelected ? styles.thumbnailSelected : {}),
                      ...(hasHands && !isSelected ? styles.thumbnailHasHands : {}),
                    }}
                    onClick={() => handleFrameClick(idx)}
                    loading="lazy"
                  />
                )}
                <div style={styles.thumbnailLabel}>
                  {formatTimestamp(frame.timestamp)}
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};

export default FrameGallery;

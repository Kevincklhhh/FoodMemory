import React, { useRef, useImperativeHandle, forwardRef, useState } from 'react';
import { formatTimestamp, getStageColor, parseNarrationId } from '../utils/narrationParser';

const styles = {
  container: {
    backgroundColor: 'white',
    borderRadius: '8px',
    boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
    display: 'flex',
    flexDirection: 'column',
    overflow: 'hidden',
  },
  videoWrapper: {
    backgroundColor: '#000',
    position: 'relative',
  },
  video: {
    width: '100%',
    maxHeight: '500px',
    display: 'block',
  },
  noVideo: {
    padding: '100px 40px',
    textAlign: 'center',
    color: '#999',
    backgroundColor: '#1a1a1a',
  },
  controls: {
    padding: '12px 15px',
    backgroundColor: '#2a2a2a',
    color: 'white',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  timeDisplay: {
    fontFamily: 'monospace',
    fontSize: '14px',
    display: 'flex',
    gap: '10px',
    alignItems: 'center',
  },
  videoIdBadge: {
    padding: '4px 8px',
    backgroundColor: '#4CAF50',
    borderRadius: '4px',
    fontSize: '12px',
    fontFamily: 'monospace',
  },
  itemPanel: {
    flex: 1,
    overflowY: 'auto',
    maxHeight: '300px',
  },
  itemHeader: {
    padding: '15px',
    borderBottom: '1px solid #e0e0e0',
    backgroundColor: '#fafafa',
  },
  itemTitle: {
    margin: 0,
    fontSize: '16px',
    fontWeight: 'bold',
    color: '#333',
  },
  itemMeta: {
    fontSize: '13px',
    color: '#666',
    marginTop: '5px',
  },
  eventTimeline: {
    padding: '10px 15px',
  },
  timelineTitle: {
    fontSize: '12px',
    fontWeight: '600',
    color: '#666',
    marginBottom: '10px',
  },
  timelineEvent: {
    display: 'flex',
    alignItems: 'center',
    gap: '10px',
    padding: '8px',
    marginBottom: '4px',
    backgroundColor: '#f5f5f5',
    borderRadius: '4px',
    cursor: 'pointer',
    transition: 'all 0.2s',
    border: '2px solid transparent',
  },
  timelineEventActive: {
    backgroundColor: '#e8f5e9',
    borderColor: '#4CAF50',
  },
  timelineEventHover: {
    backgroundColor: '#f0f0f0',
  },
  stageDot: {
    width: '10px',
    height: '10px',
    borderRadius: '50%',
  },
  eventText: {
    flex: 1,
    fontSize: '12px',
    color: '#333',
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
  },
  eventTimestamp: {
    fontSize: '11px',
    fontFamily: 'monospace',
    color: '#666',
  },
  noSelection: {
    padding: '40px',
    textAlign: 'center',
    color: '#999',
    fontSize: '14px',
  },
};

const InventoryVideoPlayer = forwardRef(({
  videoUrl,
  videoId,
  currentTime,
  onTimeUpdate,
  onNarrationClick,
  selectedItem,
  narrationTimestamps,
}, ref) => {
  const videoRef = useRef(null);
  const [hoveredEvent, setHoveredEvent] = useState(null);
  const [duration, setDuration] = useState(0);

  // Expose methods to parent
  useImperativeHandle(ref, () => ({
    seekTo: (time) => {
      if (videoRef.current) {
        videoRef.current.currentTime = time;
      }
    },
    play: () => {
      if (videoRef.current) {
        videoRef.current.play();
      }
    },
    pause: () => {
      if (videoRef.current) {
        videoRef.current.pause();
      }
    },
  }));

  const handleTimeUpdate = () => {
    if (videoRef.current && onTimeUpdate) {
      onTimeUpdate(videoRef.current.currentTime);
    }
  };

  const handleLoadedMetadata = () => {
    if (videoRef.current) {
      setDuration(videoRef.current.duration);
    }
  };

  // Find current/active event based on current time
  const activeEventIndex = selectedItem?.events?.findIndex((event, idx) => {
    const timestamp = narrationTimestamps[event.narration_id];
    const nextEvent = selectedItem.events[idx + 1];
    const nextTimestamp = nextEvent ? narrationTimestamps[nextEvent.narration_id] : null;

    if (timestamp === undefined) return false;

    // Check if current time is within this event's range
    if (nextTimestamp !== undefined) {
      return currentTime >= timestamp && currentTime < nextTimestamp;
    }
    // Last event or no next timestamp
    return currentTime >= timestamp;
  });

  const handleEventClick = (event) => {
    // Use parent's narration click handler to properly load video and seek
    if (onNarrationClick) {
      onNarrationClick(event.narration_id);
    }
  };

  // Check if event's video matches current video
  const isEventInCurrentVideo = (event) => {
    const parsed = parseNarrationId(event.narration_id);
    return parsed && parsed.videoId === videoId;
  };

  return (
    <div style={styles.container}>
      {/* Video Section */}
      <div style={styles.videoWrapper}>
        {videoUrl ? (
          <video
            ref={videoRef}
            src={videoUrl}
            style={styles.video}
            controls
            onTimeUpdate={handleTimeUpdate}
            onLoadedMetadata={handleLoadedMetadata}
          />
        ) : (
          <div style={styles.noVideo}>
            <p style={{ fontSize: '18px', marginBottom: '10px' }}>No video loaded</p>
            <p style={{ fontSize: '13px' }}>
              Click on an event with a timestamp to load its video
            </p>
          </div>
        )}
      </div>

      {/* Video Controls */}
      {videoUrl && (
        <div style={styles.controls}>
          <div style={styles.timeDisplay}>
            <span>{formatTimestamp(currentTime)} / {formatTimestamp(duration)}</span>
          </div>
          {videoId && (
            <span style={styles.videoIdBadge}>{videoId}</span>
          )}
        </div>
      )}

      {/* Selected Item Panel */}
      <div style={styles.itemPanel}>
        {selectedItem ? (
          <>
            <div style={styles.itemHeader}>
              <h3 style={styles.itemTitle}>{selectedItem.food_name}</h3>
              <div style={styles.itemMeta}>
                Difficulty: <strong>{selectedItem.difficulty}</strong>
                {' | '}
                Events: <strong>{selectedItem.num_events}</strong>
                {' | '}
                Dispensing: <strong>{selectedItem.num_dispensing}</strong>
                {selectedItem.matched_ingredient_weight && (
                  <>
                    {' | '}
                    Recipe: <strong>{selectedItem.matched_ingredient_weight.amount}{selectedItem.matched_ingredient_weight.unit}</strong>
                  </>
                )}
              </div>
            </div>

            <div style={styles.eventTimeline}>
              <div style={styles.timelineTitle}>
                Event Timeline ({selectedItem.events?.length || 0})
              </div>
              {selectedItem.events?.map((event, idx) => {
                const timestamp = narrationTimestamps[event.narration_id];
                const hasTimestamp = timestamp !== undefined;
                const isInCurrentVideo = isEventInCurrentVideo(event);
                const isActive = idx === activeEventIndex && isInCurrentVideo;

                return (
                  <div
                    key={idx}
                    style={{
                      ...styles.timelineEvent,
                      ...(isActive ? styles.timelineEventActive : {}),
                      ...(hoveredEvent === idx ? styles.timelineEventHover : {}),
                      opacity: isInCurrentVideo || !videoId ? 1 : 0.5,
                    }}
                    onMouseEnter={() => setHoveredEvent(idx)}
                    onMouseLeave={() => setHoveredEvent(null)}
                    onClick={() => handleEventClick(event)}
                  >
                    <div
                      style={{
                        ...styles.stageDot,
                        backgroundColor: getStageColor(event.stage),
                      }}
                      title={event.stage}
                    />
                    <span style={styles.eventText} title={event.action}>
                      <strong style={{ color: getStageColor(event.stage) }}>
                        {event.stage}
                      </strong>
                      {': '}
                      {event.action}
                    </span>
                    <span style={styles.eventTimestamp}>
                      {hasTimestamp ? formatTimestamp(timestamp) : '--:--'}
                    </span>
                  </div>
                );
              })}
            </div>
          </>
        ) : (
          <div style={styles.noSelection}>
            <p>Select an inventory item to see its event timeline</p>
          </div>
        )}
      </div>
    </div>
  );
});

InventoryVideoPlayer.displayName = 'InventoryVideoPlayer';

export default InventoryVideoPlayer;

import React, { useState, useEffect, useRef, useImperativeHandle, forwardRef } from 'react';
import { getStageColor, formatTimestamp } from '../utils/narrationParser';

const styles = {
  container: {
    display: 'grid',
    gridTemplateColumns: '350px 1fr',
    gap: '20px',
    height: 'calc(100vh - 280px)',
    minHeight: '500px',
  },
  panel: {
    backgroundColor: 'white',
    borderRadius: '8px',
    boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
    display: 'flex',
    flexDirection: 'column',
    overflow: 'hidden',
    height: '100%',
  },
  panelHeader: {
    padding: '8px 12px',
    borderBottom: '1px solid #e0e0e0',
    backgroundColor: '#fafafa',
    fontWeight: 'bold',
    fontSize: '12px',
  },
  panelContent: {
    flex: 1,
    overflowY: 'auto',
    padding: '8px',
  },
  videoSeparator: {
    padding: '4px 8px',
    backgroundColor: '#e3f2fd',
    color: '#1565c0',
    fontWeight: '600',
    fontSize: '10px',
    fontFamily: 'monospace',
    marginTop: '6px',
    marginBottom: '4px',
    borderRadius: '3px',
    position: 'sticky',
    top: 0,
    zIndex: 1,
  },
  eventRow: {
    display: 'flex',
    gap: '6px',
    alignItems: 'center',
    padding: '5px 8px',
    marginBottom: '2px',
    backgroundColor: '#fafafa',
    borderRadius: '4px',
    cursor: 'pointer',
    transition: 'all 0.15s',
    borderLeft: '3px solid transparent',
  },
  eventRowSelected: {
    backgroundColor: '#e8f5e9',
    borderLeftColor: '#2e7d32',
  },
  eventRowHighlight: {
    backgroundColor: '#fff3e0',
    borderLeftColor: '#ff9800',
  },
  timestamp: {
    fontFamily: 'monospace',
    fontSize: '10px',
    color: '#666',
    minWidth: '50px',
  },
  stageBadge: {
    padding: '1px 4px',
    borderRadius: '3px',
    fontSize: '8px',
    fontWeight: '600',
    color: 'white',
    minWidth: '55px',
    textAlign: 'center',
  },
  foodName: {
    fontSize: '11px',
    fontWeight: '500',
    color: '#333',
    flex: 1,
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
  },
  actionText: {
    fontSize: '10px',
    color: '#666',
    marginTop: '1px',
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
  },
  assignmentBadge: {
    fontSize: '10px',
    padding: '2px 6px',
    backgroundColor: '#e8f5e9',
    color: '#2e7d32',
    borderRadius: '4px',
    marginTop: '4px',
    display: 'inline-block',
  },
  noAssignment: {
    backgroundColor: '#fff3e0',
    color: '#e65100',
  },
  stats: {
    padding: '6px 10px',
    backgroundColor: '#f0f0f0',
    fontSize: '10px',
    color: '#666',
    borderTop: '1px solid #e0e0e0',
  },
  // Right column styles
  rightColumn: {
    display: 'flex',
    flexDirection: 'column',
    gap: '15px',
    height: '100%',
    overflow: 'hidden',
  },
  // Video player styles
  videoContainer: {
    backgroundColor: 'white',
    borderRadius: '8px',
    boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
    overflow: 'hidden',
    flexShrink: 0,
  },
  videoWrapper: {
    backgroundColor: '#000',
    position: 'relative',
  },
  video: {
    width: '100%',
    maxHeight: '450px',
    display: 'block',
  },
  noVideo: {
    padding: '100px 40px',
    textAlign: 'center',
    color: '#999',
    backgroundColor: '#1a1a1a',
  },
  videoControls: {
    padding: '10px 15px',
    backgroundColor: '#2a2a2a',
    color: 'white',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    fontSize: '13px',
  },
  videoIdBadge: {
    padding: '4px 8px',
    backgroundColor: '#4CAF50',
    borderRadius: '4px',
    fontSize: '11px',
    fontFamily: 'monospace',
  },
  // Same item events styles (horizontal below video)
  sameItemPanel: {
    backgroundColor: 'white',
    borderRadius: '8px',
    boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
    flex: 1,
    display: 'flex',
    flexDirection: 'column',
    overflow: 'hidden',
    minHeight: '150px',
  },
  sameItemHeader: {
    padding: '10px 15px',
    borderBottom: '1px solid #e0e0e0',
    backgroundColor: '#fafafa',
  },
  sameItemName: {
    fontSize: '14px',
    fontWeight: '600',
    marginBottom: '4px',
  },
  sameItemMeta: {
    fontSize: '11px',
    color: '#666',
  },
  sameItemContent: {
    flex: 1,
    overflowY: 'auto',
    padding: '10px',
  },
  sameItemEvent: {
    display: 'flex',
    gap: '8px',
    alignItems: 'center',
    padding: '8px 10px',
    marginBottom: '4px',
    backgroundColor: '#fafafa',
    borderRadius: '4px',
    cursor: 'pointer',
    transition: 'all 0.15s',
    borderLeft: '3px solid transparent',
  },
  sameItemEventCurrent: {
    backgroundColor: '#e8f5e9',
    borderLeftColor: '#2e7d32',
  },
  eventIndex: {
    fontSize: '11px',
    fontWeight: '600',
    color: '#888',
    minWidth: '24px',
  },
  videoLabel: {
    fontSize: '10px',
    color: '#1565c0',
    backgroundColor: '#e3f2fd',
    padding: '1px 4px',
    borderRadius: '3px',
    fontFamily: 'monospace',
  },
  noSelection: {
    padding: '20px',
    textAlign: 'center',
    color: '#888',
    fontSize: '13px',
  },
};

const TimelineView = forwardRef(({
  lifecycleData,
  onEventClick,
  narrationTimestamps,
  videoUrl,
  videoId,
  currentTime,
  onTimeUpdate,
}, ref) => {
  const [allEvents, setAllEvents] = useState([]);
  const [selectedEvent, setSelectedEvent] = useState(null);
  const [sameItemEvents, setSameItemEvents] = useState([]);
  const [itemInfo, setItemInfo] = useState(null);

  const selectedEventRef = useRef(null);
  const timelineRef = useRef(null);
  const videoRef = useRef(null);
  const [duration, setDuration] = useState(0);

  // Expose video methods to parent
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

  const handleVideoTimeUpdate = () => {
    if (videoRef.current && onTimeUpdate) {
      onTimeUpdate(videoRef.current.currentTime);
    }
  };

  const handleLoadedMetadata = () => {
    if (videoRef.current) {
      setDuration(videoRef.current.duration);
    }
  };

  // Build flat list of all events with item context
  useEffect(() => {
    if (!lifecycleData?.items) return;

    const events = [];

    Object.entries(lifecycleData.items).forEach(([itemId, item]) => {
      (item.events || []).forEach((event) => {
        // Extract video ID from narration_id
        const narrationId = event.narration_id || '';
        const parts = narrationId.split('-');
        const eventVideoId = parts.length >= 3 ? `${parts[0]}-${parts[1]}-${parts[2]}` : '';

        events.push({
          ...event,
          itemId,
          foodName: item.food_name,
          videoId: eventVideoId,
          videoRange: item.video_range,
          ingredientMatches: item.ingredient_matches,
        });
      });
    });

    // Sort by video ID first, then by timestamp
    events.sort((a, b) => {
      if (a.videoId !== b.videoId) {
        return a.videoId.localeCompare(b.videoId);
      }
      const aTime = narrationTimestamps[a.narration_id] || 0;
      const bTime = narrationTimestamps[b.narration_id] || 0;
      return aTime - bTime;
    });

    setAllEvents(events);
  }, [lifecycleData, narrationTimestamps]);

  // Update same-item events when selection changes
  useEffect(() => {
    if (!selectedEvent) {
      setSameItemEvents([]);
      setItemInfo(null);
      return;
    }

    const itemId = selectedEvent.itemId;
    const itemData = lifecycleData?.items?.[itemId];

    if (itemData) {
      setItemInfo({
        id: itemId,
        foodName: itemData.food_name,
        videoRange: itemData.video_range,
        ingredientMatches: itemData.ingredient_matches,
        totalEvents: itemData.events?.length || 0,
      });

      // Get all events for this item with video context
      const events = (itemData.events || []).map((evt) => {
        const narrationId = evt.narration_id || '';
        const parts = narrationId.split('-');
        const eventVideoId = parts.length >= 3 ? `${parts[0]}-${parts[1]}-${parts[2]}` : '';
        return { ...evt, videoId: eventVideoId };
      });

      setSameItemEvents(events);
    }
  }, [selectedEvent, lifecycleData]);

  const handleEventClick = (event) => {
    setSelectedEvent(event);
    if (onEventClick && event.narration_id) {
      onEventClick(event.narration_id);
    }
  };

  const handleSameItemEventClick = (event) => {
    // Find and scroll to this event in the main timeline
    const targetEvent = allEvents.find(
      e => e.narration_id === event.narration_id && e.itemId === selectedEvent?.itemId
    );

    if (targetEvent) {
      setSelectedEvent(targetEvent);
      if (onEventClick && event.narration_id) {
        onEventClick(event.narration_id);
      }
    }
  };

  // Scroll selected event into view
  useEffect(() => {
    if (selectedEventRef.current && timelineRef.current) {
      selectedEventRef.current.scrollIntoView({
        behavior: 'smooth',
        block: 'center',
      });
    }
  }, [selectedEvent?.narration_id]);

  // Group events by video for rendering
  const groupedEvents = [];
  let currentVideoId = null;

  allEvents.forEach((event, index) => {
    if (event.videoId !== currentVideoId) {
      groupedEvents.push({ type: 'separator', videoId: event.videoId });
      currentVideoId = event.videoId;
    }
    groupedEvents.push({ type: 'event', event, index });
  });

  const dispensingCount = allEvents.filter(e => e.stage === 'DISPENSING').length;
  const assignedCount = allEvents.filter(e => e.stage === 'DISPENSING' && e.assigned_recipe_id).length;

  return (
    <div style={styles.container}>
      {/* Left Panel: Timeline - All Events */}
      <div style={styles.panel}>
        <div style={styles.panelHeader}>
          Timeline - All Events ({allEvents.length})
        </div>
        <div style={styles.panelContent} ref={timelineRef}>
          {groupedEvents.map((item, idx) => {
            if (item.type === 'separator') {
              return (
                <div key={`sep-${item.videoId}`} style={styles.videoSeparator}>
                  {item.videoId}
                </div>
              );
            }

            const event = item.event;
            const isSelected = selectedEvent?.narration_id === event.narration_id
                            && selectedEvent?.itemId === event.itemId;
            const isSameItem = selectedEvent && event.itemId === selectedEvent.itemId && !isSelected;

            return (
              <div
                key={`${event.narration_id}-${event.itemId}`}
                ref={isSelected ? selectedEventRef : null}
                style={{
                  ...styles.eventRow,
                  ...(isSelected ? styles.eventRowSelected : {}),
                  ...(isSameItem ? styles.eventRowHighlight : {}),
                }}
                onClick={() => handleEventClick(event)}
                title={event.action}
              >
                <span style={styles.timestamp}>
                  {formatTimestamp(narrationTimestamps[event.narration_id])}
                </span>
                <span style={{ ...styles.stageBadge, backgroundColor: getStageColor(event.stage) }}>
                  {event.stage}
                </span>
                <div style={styles.foodName}>{event.foodName}</div>
                {event.stage === 'DISPENSING' && event.assigned_recipe_id && (
                  <span style={{ ...styles.assignmentBadge, marginTop: 0, fontSize: '8px', padding: '1px 4px' }}>
                    {event.assigned_amount}{event.assigned_amount_unit}
                  </span>
                )}
              </div>
            );
          })}
        </div>
        <div style={styles.stats}>
          Dispensing: {dispensingCount} | Assigned: {assignedCount} | Videos: {new Set(allEvents.map(e => e.videoId)).size}
        </div>
      </div>

      {/* Right Column: Video Player + Same Item Events */}
      <div style={styles.rightColumn}>
        {/* Video Player */}
        <div style={styles.videoContainer}>
          <div style={styles.videoWrapper}>
            {videoUrl ? (
              <video
                ref={videoRef}
                src={videoUrl}
                style={styles.video}
                controls
                onTimeUpdate={handleVideoTimeUpdate}
                onLoadedMetadata={handleLoadedMetadata}
              />
            ) : (
              <div style={styles.noVideo}>
                <p style={{ fontSize: '16px', marginBottom: '8px' }}>No video loaded</p>
                <p style={{ fontSize: '12px' }}>
                  Click on an event to load its video
                </p>
              </div>
            )}
          </div>
          {videoUrl && (
            <div style={styles.videoControls}>
              <span style={{ fontFamily: 'monospace' }}>
                {formatTimestamp(currentTime)} / {formatTimestamp(duration)}
              </span>
              {videoId && (
                <span style={styles.videoIdBadge}>{videoId}</span>
              )}
            </div>
          )}
        </div>

        {/* Same Item Events */}
        <div style={styles.sameItemPanel}>
          <div style={styles.panelHeader}>
            Same Item Events
          </div>
          {itemInfo ? (
            <>
              <div style={styles.sameItemHeader}>
                <div style={styles.sameItemName}>{itemInfo.foodName}</div>
                <div style={styles.sameItemMeta}>
                  {itemInfo.totalEvents} events across {itemInfo.videoRange?.length || 0} video(s)
                </div>
              </div>
              <div style={styles.sameItemContent}>
                {sameItemEvents.map((event, idx) => {
                  const isCurrent = selectedEvent?.narration_id === event.narration_id;
                  const prevVideoId = idx > 0 ? sameItemEvents[idx - 1].videoId : null;
                  const showVideoLabel = event.videoId !== prevVideoId;

                  return (
                    <React.Fragment key={event.narration_id}>
                      {showVideoLabel && (
                        <div style={{ ...styles.videoLabel, marginBottom: '6px', marginTop: idx > 0 ? '10px' : 0 }}>
                          {event.videoId}
                        </div>
                      )}
                      <div
                        style={{
                          ...styles.sameItemEvent,
                          ...(isCurrent ? styles.sameItemEventCurrent : {}),
                        }}
                        onClick={() => handleSameItemEventClick(event)}
                      >
                        <span style={styles.eventIndex}>{idx + 1}.</span>
                        <span style={styles.timestamp}>
                          {formatTimestamp(narrationTimestamps[event.narration_id])}
                        </span>
                        <span style={{ ...styles.stageBadge, backgroundColor: getStageColor(event.stage), fontSize: '8px', minWidth: '55px' }}>
                          {event.stage}
                        </span>
                        {event.stage === 'DISPENSING' && event.assigned_recipe_id && (
                          <span style={{ ...styles.assignmentBadge, marginTop: 0, fontSize: '9px' }}>
                            {event.assigned_recipe_id.replace(/^P\d+_/, '')} {event.assigned_amount}{event.assigned_amount_unit}
                          </span>
                        )}
                      </div>
                    </React.Fragment>
                  );
                })}
              </div>
            </>
          ) : (
            <div style={styles.noSelection}>
              Select an event to see all events for that item
            </div>
          )}
        </div>
      </div>
    </div>
  );
});

TimelineView.displayName = 'TimelineView';

export default TimelineView;

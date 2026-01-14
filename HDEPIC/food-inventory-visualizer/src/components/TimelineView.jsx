import React, { useState, useEffect, useRef } from 'react';
import { getStageColor, formatTimestamp } from '../utils/narrationParser';

const styles = {
  container: {
    display: 'grid',
    gridTemplateColumns: '1fr 350px',
    gap: '20px',
    height: '100%',
  },
  panel: {
    backgroundColor: 'white',
    borderRadius: '8px',
    boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
    display: 'flex',
    flexDirection: 'column',
    overflow: 'hidden',
  },
  panelHeader: {
    padding: '12px 15px',
    borderBottom: '1px solid #e0e0e0',
    backgroundColor: '#fafafa',
    fontWeight: 'bold',
    fontSize: '14px',
  },
  panelContent: {
    flex: 1,
    overflowY: 'auto',
    padding: '10px',
  },
  videoSeparator: {
    padding: '8px 12px',
    backgroundColor: '#e3f2fd',
    color: '#1565c0',
    fontWeight: '600',
    fontSize: '12px',
    fontFamily: 'monospace',
    marginTop: '10px',
    marginBottom: '6px',
    borderRadius: '4px',
    position: 'sticky',
    top: 0,
    zIndex: 1,
  },
  eventRow: {
    display: 'flex',
    gap: '8px',
    alignItems: 'flex-start',
    padding: '8px 10px',
    marginBottom: '4px',
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
    fontSize: '11px',
    color: '#666',
    minWidth: '60px',
  },
  stageBadge: {
    padding: '2px 6px',
    borderRadius: '4px',
    fontSize: '9px',
    fontWeight: '600',
    color: 'white',
    minWidth: '65px',
    textAlign: 'center',
  },
  foodName: {
    fontSize: '12px',
    fontWeight: '500',
    color: '#333',
    flex: 1,
  },
  actionText: {
    fontSize: '11px',
    color: '#666',
    marginTop: '2px',
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
  // Right panel styles
  itemHeader: {
    padding: '10px',
    backgroundColor: '#f5f5f5',
    borderBottom: '1px solid #e0e0e0',
  },
  itemName: {
    fontSize: '14px',
    fontWeight: '600',
    marginBottom: '4px',
  },
  itemMeta: {
    fontSize: '11px',
    color: '#666',
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
  stats: {
    padding: '8px 12px',
    backgroundColor: '#f0f0f0',
    fontSize: '11px',
    color: '#666',
    borderTop: '1px solid #e0e0e0',
  },
};

function TimelineView({ lifecycleData, onEventClick, narrationTimestamps }) {
  const [allEvents, setAllEvents] = useState([]);
  const [selectedEvent, setSelectedEvent] = useState(null);
  const [sameItemEvents, setSameItemEvents] = useState([]);
  const [itemInfo, setItemInfo] = useState(null);

  const selectedEventRef = useRef(null);
  const timelineRef = useRef(null);

  // Build flat list of all events with item context
  useEffect(() => {
    if (!lifecycleData?.items) return;

    const events = [];

    Object.entries(lifecycleData.items).forEach(([itemId, item]) => {
      (item.events || []).forEach((event) => {
        // Extract video ID from narration_id
        const narrationId = event.narration_id || '';
        const parts = narrationId.split('-');
        const videoId = parts.length >= 3 ? `${parts[0]}-${parts[1]}-${parts[2]}` : '';

        events.push({
          ...event,
          itemId,
          foodName: item.food_name,
          videoId,
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
      const aTime = narrationTimestamps[a.narration_id]?.start_timestamp || 0;
      const bTime = narrationTimestamps[b.narration_id]?.start_timestamp || 0;
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
        const videoId = parts.length >= 3 ? `${parts[0]}-${parts[1]}-${parts[2]}` : '';
        return { ...evt, videoId };
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
      {/* Left Panel: Timeline */}
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
              >
                <span style={styles.timestamp}>
                  {formatTimestamp(narrationTimestamps[event.narration_id]?.start_timestamp)}
                </span>
                <span style={{ ...styles.stageBadge, backgroundColor: getStageColor(event.stage) }}>
                  {event.stage}
                </span>
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={styles.foodName}>{event.foodName}</div>
                  <div style={styles.actionText} title={event.action}>
                    {event.action}
                  </div>
                  {event.stage === 'DISPENSING' && (
                    <span style={{
                      ...styles.assignmentBadge,
                      ...(event.assigned_recipe_id ? {} : styles.noAssignment),
                    }}>
                      {event.assigned_recipe_id
                        ? `${event.assigned_recipe_id} ${event.assigned_amount}${event.assigned_amount_unit}`
                        : 'No assignment'}
                    </span>
                  )}
                </div>
              </div>
            );
          })}
        </div>
        <div style={styles.stats}>
          Dispensing: {dispensingCount} | Assigned: {assignedCount} | Videos: {new Set(allEvents.map(e => e.videoId)).size}
        </div>
      </div>

      {/* Right Panel: Same Item Events */}
      <div style={styles.panel}>
        <div style={styles.panelHeader}>
          Same Item Events
        </div>
        {itemInfo ? (
          <>
            <div style={styles.itemHeader}>
              <div style={styles.itemName}>{itemInfo.foodName}</div>
              <div style={styles.itemMeta}>
                {itemInfo.totalEvents} events across {itemInfo.videoRange?.length || 0} video(s)
              </div>
            </div>
            <div style={styles.panelContent}>
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
                        {formatTimestamp(narrationTimestamps[event.narration_id]?.start_timestamp)}
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
  );
}

export default TimelineView;

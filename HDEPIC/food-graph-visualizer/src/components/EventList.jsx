import React from 'react';

const styles = {
  container: {
    backgroundColor: '#f9f9f9',
    borderRadius: '8px',
    overflow: 'hidden',
    display: 'flex',
    flexDirection: 'column',
    height: '100%',
  },
  header: {
    padding: '15px',
    backgroundColor: '#2196F3',
    color: 'white',
    fontWeight: 'bold',
    fontSize: '16px',
  },
  list: {
    flex: 1,
    overflowY: 'auto',
    padding: '0',
    margin: '0',
    listStyle: 'none',
  },
  eventItem: {
    padding: '12px 15px',
    borderBottom: '1px solid #e0e0e0',
    cursor: 'pointer',
    transition: 'background-color 0.2s',
  },
  eventItemHover: {
    backgroundColor: '#e3f2fd',
  },
  eventItemSelected: {
    backgroundColor: '#bbdefb',
    borderLeft: '4px solid #2196F3',
  },
  eventId: {
    fontWeight: 'bold',
    fontSize: '14px',
    color: '#1976D2',
    marginBottom: '4px',
  },
  eventAction: {
    fontSize: '13px',
    color: '#333',
    marginBottom: '4px',
  },
  eventTime: {
    fontSize: '12px',
    color: '#666',
    fontFamily: 'monospace',
  },
  eventDescription: {
    fontSize: '12px',
    color: '#555',
    marginTop: '4px',
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
    maxWidth: '100%',
  },
  noEvents: {
    padding: '20px',
    textAlign: 'center',
    color: '#666',
  },
  currentIndicator: {
    display: 'inline-block',
    width: '8px',
    height: '8px',
    borderRadius: '50%',
    backgroundColor: '#4CAF50',
    marginRight: '8px',
  },
};

const formatTime = (seconds) => {
  if (seconds === undefined || seconds === null) return '--:--';
  const mins = Math.floor(seconds / 60);
  const secs = (seconds % 60).toFixed(1);
  return `${mins}:${secs.padStart(4, '0')}`;
};

function EventList({ events, selectedIndex, onSelectEvent, currentTime }) {
  if (!events || events.length === 0) {
    return (
      <div style={styles.container}>
        <div style={styles.header}>State Change Events</div>
        <div style={styles.noEvents}>No events loaded</div>
      </div>
    );
  }

  // Find which event is currently active based on video time
  const activeEventIndex = events.findIndex(
    (event) =>
      currentTime >= event.timestamp_start && currentTime <= event.timestamp_end
  );

  return (
    <div style={styles.container}>
      <div style={styles.header}>
        State Change Events ({events.length})
      </div>
      <ul style={styles.list}>
        {events.map((event, index) => {
          const isSelected = index === selectedIndex;
          const isActive = index === activeEventIndex;

          return (
            <li
              key={event.event_id || index}
              onClick={() => onSelectEvent(index)}
              style={{
                ...styles.eventItem,
                ...(isSelected ? styles.eventItemSelected : {}),
              }}
              onMouseEnter={(e) => {
                if (!isSelected) {
                  e.currentTarget.style.backgroundColor = '#e3f2fd';
                }
              }}
              onMouseLeave={(e) => {
                if (!isSelected) {
                  e.currentTarget.style.backgroundColor = 'transparent';
                }
              }}
            >
              <div style={styles.eventId}>
                {isActive && <span style={styles.currentIndicator} />}
                Event {event.event_id}
              </div>
              <div style={styles.eventAction}>
                {event.primary_action || event.action || 'Unknown action'}
              </div>
              <div style={styles.eventTime}>
                {formatTime(event.timestamp_start)} - {formatTime(event.timestamp_end)}
              </div>
              {event.state_description && (
                <div style={styles.eventDescription} title={event.state_description}>
                  {event.state_description}
                </div>
              )}
            </li>
          );
        })}
      </ul>
    </div>
  );
}

export default EventList;

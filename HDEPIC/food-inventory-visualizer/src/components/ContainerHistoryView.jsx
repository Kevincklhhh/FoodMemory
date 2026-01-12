import React from 'react';

const styles = {
  container: {
    backgroundColor: '#fff',
    borderRadius: '8px',
    border: '1px solid #ddd',
    padding: '15px',
    marginTop: '15px',
  },
  header: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: '15px',
  },
  title: {
    fontSize: '14px',
    fontWeight: 'bold',
    color: '#333',
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
  },
  containerIcon: {
    fontSize: '18px',
  },
  closeBtn: {
    background: 'none',
    border: 'none',
    fontSize: '18px',
    cursor: 'pointer',
    color: '#666',
    padding: '0 5px',
  },
  historyContainer: {
    display: 'flex',
    alignItems: 'flex-start',
    overflowX: 'auto',
    padding: '10px 0',
    gap: '20px',
  },
  // Event card with vertical layout
  eventCard: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    minWidth: '160px',
  },
  eventHeader: {
    fontSize: '10px',
    color: '#888',
    marginBottom: '8px',
    textAlign: 'center',
  },
  eventId: {
    fontWeight: 'bold',
    color: '#e65100',
  },
  // Source layer (top)
  sourceLayer: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    marginBottom: '0',
  },
  sourceBox: {
    padding: '8px 12px',
    backgroundColor: '#fff3e0',
    border: '2px solid #ff9800',
    borderRadius: '6px',
    textAlign: 'center',
    minWidth: '120px',
  },
  sourceLabel: {
    fontSize: '9px',
    color: '#888',
    fontWeight: 'bold',
    textTransform: 'uppercase',
    marginBottom: '4px',
  },
  sourceId: {
    fontWeight: 'bold',
    fontSize: '11px',
    color: '#e65100',
    wordBreak: 'break-all',
  },
  sourceNoun: {
    fontSize: '10px',
    color: '#666',
    marginTop: '2px',
  },
  sourceLocation: {
    fontSize: '9px',
    color: '#888',
    marginTop: '4px',
    display: 'flex',
    alignItems: 'center',
    gap: '3px',
    justifyContent: 'center',
  },
  // Vertical edge connecting source to destination
  verticalEdge: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    padding: '4px 0',
  },
  verticalLine: {
    width: '2px',
    height: '20px',
    backgroundColor: '#4CAF50',
  },
  edgeLabel: {
    fontSize: '9px',
    fontWeight: 'bold',
    color: '#4CAF50',
    backgroundColor: '#e8f5e9',
    padding: '2px 6px',
    borderRadius: '3px',
    textTransform: 'uppercase',
  },
  verticalArrow: {
    width: 0,
    height: 0,
    borderLeft: '5px solid transparent',
    borderRight: '5px solid transparent',
    borderTop: '8px solid #4CAF50',
  },
  // Destination layer (bottom) - the container
  destLayer: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
  },
  destBox: {
    padding: '8px 12px',
    backgroundColor: '#e3f2fd',
    border: '2px solid #2196F3',
    borderRadius: '6px',
    textAlign: 'center',
    minWidth: '120px',
  },
  destLabel: {
    fontSize: '9px',
    color: '#888',
    fontWeight: 'bold',
    textTransform: 'uppercase',
    marginBottom: '4px',
  },
  destId: {
    fontWeight: 'bold',
    fontSize: '11px',
    color: '#1565c0',
    wordBreak: 'break-all',
  },
  destNoun: {
    fontSize: '10px',
    color: '#666',
    marginTop: '2px',
  },
  destState: {
    fontSize: '9px',
    color: '#888',
    marginTop: '2px',
  },
  // Horizontal arrow between events
  horizontalArrow: {
    display: 'flex',
    alignItems: 'center',
    alignSelf: 'center',
    padding: '0 8px',
  },
  horizontalLine: {
    width: '30px',
    height: '2px',
    backgroundColor: '#ccc',
  },
  horizontalArrowHead: {
    width: 0,
    height: 0,
    borderTop: '5px solid transparent',
    borderBottom: '5px solid transparent',
    borderLeft: '8px solid #ccc',
  },
  noHistory: {
    color: '#999',
    textAlign: 'center',
    padding: '20px',
    fontStyle: 'italic',
  },
  summary: {
    fontSize: '12px',
    color: '#666',
    marginBottom: '10px',
    padding: '8px 12px',
    backgroundColor: '#f5f5f5',
    borderRadius: '4px',
  },
  // No source styling
  noSource: {
    padding: '8px 12px',
    backgroundColor: '#f5f5f5',
    border: '1px dashed #ccc',
    borderRadius: '6px',
    textAlign: 'center',
    minWidth: '120px',
    color: '#999',
    fontSize: '10px',
    fontStyle: 'italic',
  },
  // Spacer to align destination boxes for non-split events
  spacer: {
    height: '120px', // Approximate height of source box + edge
  },
};

const CONTAINER_ICONS = {
  pan: '🍳',
  pot: '🍲',
  bowl: '🥣',
  plate: '🍽️',
  cup: '☕',
  mug: '☕',
  glass: '🥛',
  jar: '🫙',
  bottle: '🍾',
  bag: '🛍️',
  blender: '🥤',
  container: '📦',
  box: '📦',
  environment: '📍',
  default: '📦',
};

const EDGE_COLORS = {
  split: { bg: '#e8f5e9', text: '#4CAF50', line: '#4CAF50' },
  transfer: { bg: '#e3f2fd', text: '#2196F3', line: '#2196F3' },
  merge: { bg: '#f3e5f5', text: '#9c27b0', line: '#9c27b0' },
  default: { bg: '#f5f5f5', text: '#666', line: '#666' },
};

function getContainerIcon(containerId) {
  if (!containerId) return CONTAINER_ICONS.environment;
  const loc = containerId.toLowerCase();
  for (const [key, icon] of Object.entries(CONTAINER_ICONS)) {
    if (key !== 'default' && key !== 'environment' && loc.includes(key)) return icon;
  }
  return CONTAINER_ICONS.default;
}

function formatTimestamp(seconds) {
  if (seconds === null || seconds === undefined) return '--:--';
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${secs.toString().padStart(2, '0')}`;
}

function formatContainerName(containerId) {
  if (!containerId) return 'Unknown';
  return containerId.replace(/_/g, ' ');
}

function formatLocation(location) {
  if (!location) return 'Environment';
  return location.replace(/_/g, ' ');
}

function HistoryEventCard({ item, containerId, isLast }) {
  const source = item.source;
  const isSplit = source?.derivation_type === 'split';
  const edgeColors = source ? (EDGE_COLORS[source.derivation_type] || EDGE_COLORS.default) : EDGE_COLORS.default;
  const containerIcon = getContainerIcon(containerId);

  return (
    <>
      <div style={styles.eventCard}>
        {/* Event header */}
        <div style={styles.eventHeader}>
          <span style={styles.eventId}>Event {item.event_id}</span>
          <span> | {formatTimestamp(item.timestamp)}</span>
        </div>

        {/* Source layer - only shown for split events */}
        {isSplit && source ? (
          <>
            <div style={styles.sourceLayer}>
              <div style={styles.sourceBox}>
                <div style={styles.sourceLabel}>Source</div>
                <div style={styles.sourceId}>{source.parent_id}</div>
                <div style={styles.sourceNoun}>{source.parent_noun}</div>
                <div style={styles.sourceLocation}>
                  <span>{getContainerIcon(source.parent_location)}</span>
                  <span>{formatLocation(source.parent_location)}</span>
                </div>
              </div>
            </div>

            {/* Vertical edge with label - only for split */}
            <div style={styles.verticalEdge}>
              <div style={{ ...styles.verticalLine, backgroundColor: edgeColors.line }} />
              <div style={{ ...styles.edgeLabel, backgroundColor: edgeColors.bg, color: edgeColors.text }}>
                {source.derivation_type}
              </div>
              <div style={{ ...styles.verticalArrow, borderTopColor: edgeColors.line }} />
            </div>
          </>
        ) : (
          /* Spacer to align destination box at bottom for non-split events */
          <div style={styles.spacer} />
        )}

        {/* Destination layer (container with the food) */}
        <div style={styles.destLayer}>
          <div style={styles.destBox}>
            <div style={styles.destLabel}>
              {containerIcon} {formatContainerName(containerId)}
            </div>
            <div style={styles.destId}>{item.food_id}</div>
            <div style={styles.destNoun}>{item.food_noun}</div>
            {item.state && (
              <div style={styles.destState}>
                {item.state.form_state || 'unknown'} | {item.state.quantity || 'unknown'}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Horizontal arrow to next event */}
      {!isLast && (
        <div style={styles.horizontalArrow}>
          <div style={styles.horizontalLine} />
          <div style={styles.horizontalArrowHead} />
        </div>
      )}
    </>
  );
}

function ContainerHistoryView({ containerId, history, onClose }) {
  if (!containerId) {
    return null;
  }

  const icon = getContainerIcon(containerId);
  const displayName = formatContainerName(containerId);

  return (
    <div style={styles.container}>
      <div style={styles.header}>
        <div style={styles.title}>
          <span style={styles.containerIcon}>{icon}</span>
          <span>Container History: {displayName}</span>
        </div>
        <button style={styles.closeBtn} onClick={onClose} title="Close">
          x
        </button>
      </div>

      {history && history.length > 0 ? (
        <>
          <div style={styles.summary}>
            {history.length} item{history.length !== 1 ? 's' : ''} added to this container
          </div>
          <div style={styles.historyContainer}>
            {history.map((item, idx) => (
              <HistoryEventCard
                key={`${item.food_id}-${item.block_idx}`}
                item={item}
                containerId={containerId}
                isLast={idx === history.length - 1}
              />
            ))}
          </div>
        </>
      ) : (
        <div style={styles.noHistory}>
          No items have been added to {displayName}
        </div>
      )}
    </div>
  );
}

export default ContainerHistoryView;

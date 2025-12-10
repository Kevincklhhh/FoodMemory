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
  },
  closeBtn: {
    background: 'none',
    border: 'none',
    fontSize: '18px',
    cursor: 'pointer',
    color: '#666',
    padding: '0 5px',
  },
  pathContainer: {
    display: 'flex',
    alignItems: 'center',
    overflowX: 'auto',
    padding: '10px 0',
    gap: '0',
  },
  stepWrapper: {
    display: 'flex',
    alignItems: 'center',
  },
  step: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    minWidth: '100px',
    padding: '8px',
  },
  nodeBox: {
    padding: '8px 12px',
    borderRadius: '6px',
    border: '2px solid #2196F3',
    backgroundColor: '#e3f2fd',
    textAlign: 'center',
    minWidth: '80px',
  },
  nodeBoxCurrent: {
    border: '2px solid #4CAF50',
    backgroundColor: '#c8e6c9',
  },
  nodeBoxConsumed: {
    border: '2px solid #9e9e9e',
    backgroundColor: '#f5f5f5',
    opacity: 0.7,
  },
  nodeId: {
    fontWeight: 'bold',
    fontSize: '11px',
    color: '#1565c0',
    wordBreak: 'break-all',
  },
  nodeIdCurrent: {
    color: '#2e7d32',
  },
  foodNoun: {
    fontSize: '10px',
    color: '#666',
    marginTop: '2px',
  },
  location: {
    fontSize: '9px',
    color: '#888',
    marginTop: '4px',
    display: 'flex',
    alignItems: 'center',
    gap: '3px',
  },
  arrow: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    padding: '0 8px',
  },
  arrowLine: {
    width: '40px',
    height: '2px',
    backgroundColor: '#999',
    position: 'relative',
  },
  arrowHead: {
    width: 0,
    height: 0,
    borderTop: '5px solid transparent',
    borderBottom: '5px solid transparent',
    borderLeft: '8px solid #999',
    marginLeft: '-1px',
  },
  derivationType: {
    fontSize: '9px',
    fontWeight: 'bold',
    padding: '2px 6px',
    borderRadius: '3px',
    marginBottom: '4px',
    textTransform: 'uppercase',
  },
  deriveSplit: { backgroundColor: '#c8e6c9', color: '#2e7d32' },
  deriveMerge: { backgroundColor: '#e1bee7', color: '#7b1fa2' },
  deriveUpdate: { backgroundColor: '#bbdefb', color: '#1565c0' },
  deriveIdentity: { backgroundColor: '#ffe0b2', color: '#e65100' },
  deriveTransfer: { backgroundColor: '#cfd8dc', color: '#455a64' },
  deriveConsume: { backgroundColor: '#ffcdd2', color: '#c62828' },
  deriveDefault: { backgroundColor: '#e0e0e0', color: '#424242' },
  noPath: {
    color: '#999',
    textAlign: 'center',
    padding: '20px',
    fontStyle: 'italic',
  },
  stateInfo: {
    fontSize: '9px',
    color: '#777',
    marginTop: '3px',
  },
};

const DERIVATION_STYLES = {
  split: styles.deriveSplit,
  merge: styles.deriveMerge,
  update: styles.deriveUpdate,
  identity_transform: styles.deriveIdentity,
  transfer: styles.deriveTransfer,
  consume: styles.deriveConsume,
};

const LOCATION_ICONS = {
  pan: '🍳',
  pot: '🍲',
  bowl: '🥣',
  plate: '🍽️',
  cup: '☕',
  mug: '☕',
  fridge: '❄️',
  counter: '📍',
  stove: '🔥',
  hand: '✋',
  default: '📍',
};

function getLocationIcon(location) {
  if (!location) return '';
  const loc = location.toLowerCase();
  for (const [key, icon] of Object.entries(LOCATION_ICONS)) {
    if (loc.includes(key)) return icon;
  }
  return LOCATION_ICONS.default;
}

function formatNodeId(id) {
  if (!id) return '?';
  // Remove trailing _001, _002 etc for display
  return id.replace(/_\d+$/, '').replace(/_/g, ' ');
}

function PathStep({ step, isLast }) {
  const isCurrent = step.isCurrent;
  const isConsumed = step.status === 'consumed';
  const derivationType = step.action_next?.toLowerCase().replace(/\s*\(.*\)/, '');

  return (
    <div style={styles.stepWrapper}>
      <div style={styles.step}>
        <div
          style={{
            ...styles.nodeBox,
            ...(isCurrent ? styles.nodeBoxCurrent : {}),
            ...(isConsumed ? styles.nodeBoxConsumed : {}),
          }}
        >
          <div style={{ ...styles.nodeId, ...(isCurrent ? styles.nodeIdCurrent : {}) }}>
            {step.node}
          </div>
          {step.food_noun && (
            <div style={styles.foodNoun}>{step.food_noun}</div>
          )}
          {step.state && (
            <div style={styles.stateInfo}>
              {step.state.form_state} | {step.state.quantity}
            </div>
          )}
        </div>
        {step.location && (
          <div style={styles.location}>
            <span>{getLocationIcon(step.location)}</span>
            <span>{step.location}</span>
          </div>
        )}
      </div>

      {!isLast && step.action_next && (
        <div style={styles.arrow}>
          <div
            style={{
              ...styles.derivationType,
              ...(DERIVATION_STYLES[derivationType] || styles.deriveDefault),
            }}
          >
            {step.action_next}
          </div>
          <div style={{ display: 'flex', alignItems: 'center' }}>
            <div style={styles.arrowLine} />
            <div style={styles.arrowHead} />
          </div>
        </div>
      )}
    </div>
  );
}

function LineageView({ focusNode, path, onClose }) {
  if (!focusNode) {
    return null;
  }

  return (
    <div style={styles.container}>
      <div style={styles.header}>
        <div style={styles.title}>
          Lineage: {focusNode}
        </div>
        <button style={styles.closeBtn} onClick={onClose} title="Close">
          ×
        </button>
      </div>

      {path && path.length > 0 ? (
        <div style={styles.pathContainer}>
          {path.map((step, idx) => (
            <PathStep
              key={`${step.node}-${idx}`}
              step={step}
              isLast={idx === path.length - 1}
            />
          ))}
        </div>
      ) : (
        <div style={styles.noPath}>
          No ancestry path found for {focusNode}
        </div>
      )}
    </div>
  );
}

export default LineageView;

import React, { useState } from 'react';

const styles = {
  container: {
    backgroundColor: '#f5f5f5',
    borderRadius: '8px',
    border: '1px solid #ddd',
    overflow: 'hidden',
  },
  header: {
    padding: '10px 15px',
    backgroundColor: '#455a64',
    color: 'white',
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    cursor: 'pointer',
    fontSize: '13px',
    fontWeight: 'bold',
  },
  toggleIcon: {
    fontSize: '14px',
    transition: 'transform 0.2s',
  },
  content: {
    padding: '12px 15px',
    maxHeight: '250px',
    overflowY: 'auto',
    fontSize: '12px',
  },
  section: {
    marginBottom: '12px',
  },
  sectionTitle: {
    fontWeight: 'bold',
    color: '#37474f',
    marginBottom: '4px',
    fontSize: '11px',
    textTransform: 'uppercase',
  },
  sectionContent: {
    backgroundColor: '#fff',
    padding: '8px 10px',
    borderRadius: '4px',
    border: '1px solid #e0e0e0',
    whiteSpace: 'pre-wrap',
    wordBreak: 'break-word',
    fontFamily: 'monospace',
    fontSize: '11px',
    lineHeight: '1.4',
    maxHeight: '100px',
    overflowY: 'auto',
  },
  transactionsList: {
    backgroundColor: '#fff',
    borderRadius: '4px',
    border: '1px solid #e0e0e0',
  },
  transaction: {
    padding: '8px 10px',
    borderBottom: '1px solid #eee',
  },
  transactionType: {
    display: 'inline-block',
    padding: '2px 6px',
    borderRadius: '3px',
    fontSize: '10px',
    fontWeight: 'bold',
    textTransform: 'uppercase',
    marginRight: '8px',
  },
  transactionSplit: { backgroundColor: '#c8e6c9', color: '#2e7d32' },
  transactionMerge: { backgroundColor: '#e1bee7', color: '#7b1fa2' },
  transactionUpdate: { backgroundColor: '#bbdefb', color: '#1565c0' },
  transactionIdentity: { backgroundColor: '#ffe0b2', color: '#e65100' },
  transactionTransfer: { backgroundColor: '#cfd8dc', color: '#455a64' },
  transactionConsume: { backgroundColor: '#ffcdd2', color: '#c62828' },
  transactionCreate: { backgroundColor: '#b2dfdb', color: '#00695c' },
  transactionDefault: { backgroundColor: '#e0e0e0', color: '#424242' },
  transactionFoodId: {
    color: '#1565c0',
    fontWeight: '500',
  },
  transactionReasoning: {
    marginTop: '4px',
    padding: '6px 8px',
    backgroundColor: '#fafafa',
    borderRadius: '3px',
    fontSize: '11px',
    color: '#555',
    fontStyle: 'italic',
  },
  noData: {
    color: '#999',
    textAlign: 'center',
    padding: '20px',
    fontStyle: 'italic',
  },
};

const TRANSACTION_STYLES = {
  split: styles.transactionSplit,
  merge: styles.transactionMerge,
  update: styles.transactionUpdate,
  identity_transform: styles.transactionIdentity,
  transfer: styles.transactionTransfer,
  consume: styles.transactionConsume,
  create: styles.transactionCreate,
};

function DebugFooter({ selectedEventIndex, events, vlmLogs }) {
  const [isExpanded, setIsExpanded] = useState(true);

  if (selectedEventIndex === null || selectedEventIndex === undefined) {
    return null;
  }

  const selectedEvent = events?.[selectedEventIndex];
  if (!selectedEvent) {
    return null;
  }

  // Get VLM log for this event
  const eventKey = `event_${String(selectedEventIndex + 1).padStart(3, '0')}`;
  const vlmLog = vlmLogs?.[eventKey];

  const transactions = vlmLog?.parsed_response?.transactions || [];

  return (
    <div style={styles.container}>
      <div style={styles.header} onClick={() => setIsExpanded(!isExpanded)}>
        <span>
          VLM Reasoning - Event {selectedEvent.event_id}: {selectedEvent.primary_action}
        </span>
        <span style={{ ...styles.toggleIcon, transform: isExpanded ? 'rotate(0deg)' : 'rotate(-90deg)' }}>
          ▼
        </span>
      </div>

      {isExpanded && (
        <div style={styles.content}>
          {/* Visual Evidence from state_change.json */}
          {selectedEvent.visual_evidence && (
            <div style={styles.section}>
              <div style={styles.sectionTitle}>Visual Evidence</div>
              <div style={styles.sectionContent}>
                {selectedEvent.visual_evidence}
              </div>
            </div>
          )}

          {/* State Description from state_change.json */}
          {selectedEvent.state_description && (
            <div style={styles.section}>
              <div style={styles.sectionTitle}>State Description</div>
              <div style={styles.sectionContent}>
                {selectedEvent.state_description}
              </div>
            </div>
          )}

          {/* VLM Transactions */}
          <div style={styles.section}>
            <div style={styles.sectionTitle}>
              VLM Transactions ({transactions.length})
            </div>
            {transactions.length > 0 ? (
              <div style={styles.transactionsList}>
                {transactions.map((txn, idx) => {
                  const typeStyle = TRANSACTION_STYLES[txn.type] || styles.transactionDefault;
                  return (
                    <div key={idx} style={styles.transaction}>
                      <div>
                        <span style={{ ...styles.transactionType, ...typeStyle }}>
                          {txn.type}
                        </span>
                        <span style={styles.transactionFoodId}>
                          {txn.food_id || txn.parent_id || txn.source_id || '?'}
                        </span>
                        {txn.child_id && txn.child_id !== txn.food_id && (
                          <span> → <span style={styles.transactionFoodId}>{txn.child_id}</span></span>
                        )}
                        {txn.destination && (
                          <span> @ {txn.destination}</span>
                        )}
                      </div>
                      {txn.reasoning && (
                        <div style={styles.transactionReasoning}>
                          {txn.reasoning}
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            ) : (
              <div style={styles.noData}>
                {vlmLog ? 'No transactions parsed' : 'No VLM log available for this event'}
              </div>
            )}
          </div>

          {/* Raw VLM Response (collapsed by default) */}
          {vlmLog?.raw_response && (
            <details style={styles.section}>
              <summary style={{ ...styles.sectionTitle, cursor: 'pointer' }}>
                Raw VLM Response
              </summary>
              <div style={{ ...styles.sectionContent, marginTop: '4px' }}>
                {vlmLog.raw_response}
              </div>
            </details>
          )}
        </div>
      )}
    </div>
  );
}

export default DebugFooter;

import React, { useState } from 'react';
import { getDifficultyColor, getStageColor, formatTimestamp } from '../utils/narrationParser';

const styles = {
  container: {
    backgroundColor: 'white',
    borderRadius: '8px',
    boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
    height: '100%',
    maxHeight: '100%',
    display: 'flex',
    flexDirection: 'column',
    overflow: 'hidden',
  },
  header: {
    padding: '15px',
    borderBottom: '1px solid #e0e0e0',
    backgroundColor: '#fafafa',
  },
  title: {
    margin: 0,
    fontSize: '16px',
    fontWeight: 'bold',
  },
  filterRow: {
    display: 'flex',
    gap: '10px',
    marginTop: '10px',
    flexWrap: 'wrap',
  },
  filterChip: {
    padding: '4px 10px',
    borderRadius: '16px',
    fontSize: '12px',
    cursor: 'pointer',
    border: '1px solid #ddd',
    backgroundColor: 'white',
    transition: 'all 0.2s',
  },
  filterChipActive: {
    backgroundColor: '#2E7D32',
    color: 'white',
    borderColor: '#2E7D32',
  },
  list: {
    flex: 1,
    overflowY: 'auto',
    padding: '10px',
  },
  itemCard: {
    backgroundColor: '#fafafa',
    borderRadius: '6px',
    marginBottom: '8px',
    border: '1px solid #e0e0e0',
    overflow: 'hidden',
    transition: 'all 0.2s',
  },
  itemCardSelected: {
    borderColor: '#2E7D32',
    boxShadow: '0 2px 4px rgba(46, 125, 50, 0.2)',
  },
  itemHeader: {
    padding: '12px',
    cursor: 'pointer',
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
  },
  itemInfo: {
    flex: 1,
  },
  foodName: {
    fontSize: '14px',
    fontWeight: '600',
    marginBottom: '4px',
    color: '#333',
  },
  itemMeta: {
    display: 'flex',
    gap: '8px',
    flexWrap: 'wrap',
    alignItems: 'center',
  },
  badge: {
    padding: '2px 8px',
    borderRadius: '4px',
    fontSize: '11px',
    fontWeight: '500',
  },
  ingredientBadge: {
    backgroundColor: '#e3f2fd',
    color: '#1565c0',
    fontSize: '11px',
    padding: '2px 6px',
    borderRadius: '4px',
  },
  eventCount: {
    fontSize: '12px',
    color: '#666',
  },
  expandIcon: {
    fontSize: '18px',
    color: '#666',
    marginLeft: '10px',
  },
  itemDetails: {
    borderTop: '1px solid #e0e0e0',
    backgroundColor: 'white',
    maxHeight: '400px',
    overflowY: 'auto',
  },
  detailSection: {
    padding: '12px',
    borderBottom: '1px solid #f0f0f0',
  },
  sectionTitle: {
    fontSize: '12px',
    fontWeight: '600',
    color: '#666',
    marginBottom: '8px',
  },
  videoRanges: {
    display: 'flex',
    flexWrap: 'wrap',
    gap: '4px',
  },
  videoChip: {
    padding: '2px 6px',
    backgroundColor: '#f0f0f0',
    borderRadius: '4px',
    fontSize: '10px',
    fontFamily: 'monospace',
  },
  eventList: {
    display: 'flex',
    flexDirection: 'column',
    gap: '6px',
  },
  eventRow: {
    display: 'flex',
    gap: '8px',
    alignItems: 'flex-start',
    padding: '8px',
    backgroundColor: '#fafafa',
    borderRadius: '4px',
    cursor: 'pointer',
    transition: 'background-color 0.2s',
  },
  eventRowHover: {
    backgroundColor: '#e8f5e9',
  },
  stageBadge: {
    padding: '2px 6px',
    borderRadius: '4px',
    fontSize: '10px',
    fontWeight: '600',
    color: 'white',
    minWidth: '70px',
    textAlign: 'center',
  },
  eventAction: {
    flex: 1,
    fontSize: '12px',
    color: '#333',
    lineHeight: '1.4',
  },
  timestampBadge: {
    padding: '2px 6px',
    backgroundColor: '#e3f2fd',
    borderRadius: '4px',
    fontSize: '10px',
    fontFamily: 'monospace',
    color: '#1565c0',
    cursor: 'pointer',
  },
  timestampMissing: {
    backgroundColor: '#ffebee',
    color: '#c62828',
  },
  narrationId: {
    fontSize: '10px',
    color: '#888',
    fontFamily: 'monospace',
  },
  methodBadge: {
    padding: '1px 4px',
    backgroundColor: '#fff3e0',
    borderRadius: '3px',
    fontSize: '9px',
    color: '#e65100',
  },
};

function InventoryItemList({ items, selectedItem, onSelectItem, onNarrationClick, narrationTimestamps }) {
  const [difficultyFilter, setDifficultyFilter] = useState(null);
  const [expandedItem, setExpandedItem] = useState(null);
  const [hoveredEvent, setHoveredEvent] = useState(null);

  // Filter items by difficulty
  const filteredItems = difficultyFilter
    ? items.filter(item => item.difficulty === difficultyFilter)
    : items;

  const handleItemClick = (item) => {
    if (expandedItem?.narration_id === item.narration_id) {
      setExpandedItem(null);
    } else {
      setExpandedItem(item);
      onSelectItem(item);
    }
  };

  const handleEventClick = (e, event) => {
    e.stopPropagation();
    if (event.narration_id) {
      onNarrationClick(event.narration_id);
    }
  };

  return (
    <div style={styles.container}>
      <div style={styles.header}>
        <h3 style={styles.title}>Inventory Items ({filteredItems.length})</h3>
        <div style={styles.filterRow}>
          <button
            style={{
              ...styles.filterChip,
              ...(difficultyFilter === null ? styles.filterChipActive : {}),
            }}
            onClick={() => setDifficultyFilter(null)}
          >
            All
          </button>
          {['LOW', 'MID', 'HIGH'].map(diff => (
            <button
              key={diff}
              style={{
                ...styles.filterChip,
                ...(difficultyFilter === diff ? styles.filterChipActive : {}),
              }}
              onClick={() => setDifficultyFilter(diff)}
            >
              {diff}
            </button>
          ))}
        </div>
      </div>

      <div style={styles.list}>
        {filteredItems.map((item, idx) => {
          const isExpanded = expandedItem?.narration_id === item.narration_id;
          const isSelected = selectedItem?.narration_id === item.narration_id;

          return (
            <div
              key={item.narration_id || idx}
              style={{
                ...styles.itemCard,
                ...(isSelected ? styles.itemCardSelected : {}),
              }}
            >
              <div
                style={styles.itemHeader}
                onClick={() => handleItemClick(item)}
              >
                <div style={styles.itemInfo}>
                  <div style={styles.foodName}>{item.food_name}</div>
                  <div style={styles.itemMeta}>
                    <span
                      style={{
                        ...styles.badge,
                        backgroundColor: getDifficultyColor(item.difficulty) + '20',
                        color: getDifficultyColor(item.difficulty),
                      }}
                    >
                      {item.difficulty}
                    </span>
                    <span style={styles.eventCount}>
                      {item.num_events} events, {item.num_dispensing} dispensing
                    </span>
                    {item.matched_ingredient_weight && (
                      <span style={styles.ingredientBadge}>
                        {item.matched_ingredient_weight.amount}{item.matched_ingredient_weight.unit}
                        {' '}
                        {item.matched_ingredient_weight.ingredient}
                      </span>
                    )}
                  </div>
                </div>
                <span style={styles.expandIcon}>{isExpanded ? '▼' : '▶'}</span>
              </div>

              {isExpanded && (
                <div style={styles.itemDetails}>
                  {/* Video Ranges */}
                  <div style={styles.detailSection}>
                    <div style={styles.sectionTitle}>Video Range</div>
                    <div style={styles.videoRanges}>
                      {item.video_range?.map((vid, i) => (
                        <span key={i} style={styles.videoChip}>{vid}</span>
                      ))}
                    </div>
                  </div>

                  {/* Matched Ingredient */}
                  {item.matched_ingredient_weight && (
                    <div style={styles.detailSection}>
                      <div style={styles.sectionTitle}>Recipe Match</div>
                      <div style={{ fontSize: '12px' }}>
                        <strong>{item.matched_ingredient_weight.ingredient}</strong>
                        {': '}
                        {item.matched_ingredient_weight.amount} {item.matched_ingredient_weight.unit}
                        {' '}
                        <span style={{ color: '#666' }}>
                          (Recipe: {item.matched_ingredient_weight.recipe})
                        </span>
                      </div>
                    </div>
                  )}

                  {/* Events Timeline */}
                  <div style={styles.detailSection}>
                    <div style={styles.sectionTitle}>
                      Events Timeline ({item.events?.length || 0})
                    </div>
                    <div style={styles.eventList}>
                      {item.events?.map((event, i) => {
                        const timestamp = narrationTimestamps[event.narration_id];
                        const hasTimestamp = timestamp !== undefined;

                        return (
                          <div
                            key={i}
                            style={{
                              ...styles.eventRow,
                              ...(hoveredEvent === i ? styles.eventRowHover : {}),
                            }}
                            onMouseEnter={() => setHoveredEvent(i)}
                            onMouseLeave={() => setHoveredEvent(null)}
                            onClick={(e) => handleEventClick(e, event)}
                          >
                            <span
                              style={{
                                ...styles.stageBadge,
                                backgroundColor: getStageColor(event.stage),
                              }}
                            >
                              {event.stage}
                            </span>
                            <div style={{ flex: 1 }}>
                              <div style={styles.eventAction}>
                                {event.action}
                                {event.method && (
                                  <span style={styles.methodBadge}>{event.method}</span>
                                )}
                              </div>
                              <div style={styles.narrationId}>{event.narration_id}</div>
                            </div>
                            <span
                              style={{
                                ...styles.timestampBadge,
                                ...(hasTimestamp ? {} : styles.timestampMissing),
                              }}
                              title={hasTimestamp ? 'Click to jump to timestamp' : 'No timestamp data'}
                            >
                              {hasTimestamp ? formatTimestamp(timestamp) : 'N/A'}
                            </span>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

export default InventoryItemList;

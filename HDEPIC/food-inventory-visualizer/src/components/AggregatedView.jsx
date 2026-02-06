import React, { useState, useEffect, useRef, useImperativeHandle, forwardRef } from 'react';
import { formatTimestamp } from '../utils/narrationParser';

const styles = {
  container: {
    display: 'grid',
    gridTemplateColumns: '350px 1fr',
    gap: '15px',
    height: 'calc(100vh - 260px)',
    minHeight: '600px',
  },
  // Left panel - item list
  listPanel: {
    backgroundColor: 'white',
    borderRadius: '8px',
    boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
    display: 'flex',
    flexDirection: 'column',
    overflow: 'hidden',
  },
  listHeader: {
    padding: '10px 12px',
    borderBottom: '1px solid #e0e0e0',
    backgroundColor: '#fafafa',
    fontWeight: 'bold',
    fontSize: '13px',
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  listContent: {
    flex: 1,
    overflowY: 'auto',
    padding: '6px',
  },
  itemRow: {
    padding: '6px 8px',
    marginBottom: '3px',
    backgroundColor: '#fafafa',
    borderRadius: '4px',
    cursor: 'pointer',
    transition: 'all 0.15s',
    borderLeft: '3px solid transparent',
  },
  itemRowSelected: {
    backgroundColor: '#e3f2fd',
    borderLeftColor: '#1976D2',
  },
  itemRowModified: {
    borderLeftColor: '#ff9800',
  },
  itemName: {
    fontSize: '11px',
    fontWeight: '600',
    color: '#333',
    marginBottom: '2px',
  },
  itemMeta: {
    display: 'flex',
    gap: '4px',
    flexWrap: 'wrap',
    alignItems: 'center',
  },
  badge: {
    fontSize: '8px',
    padding: '1px 4px',
    borderRadius: '3px',
    fontWeight: '500',
  },
  difficultyLow: { backgroundColor: '#e8f5e9', color: '#2e7d32' },
  difficultyMid: { backgroundColor: '#fff3e0', color: '#e65100' },
  difficultyHigh: { backgroundColor: '#ffebee', color: '#c62828' },
  amountBadge: { backgroundColor: '#e3f2fd', color: '#1565c0' },
  recipeBadge: { backgroundColor: '#f3e5f5', color: '#7b1fa2' },
  segmentsBadge: { backgroundColor: '#eceff1', color: '#455a64' },
  // Right panel
  rightColumn: {
    display: 'grid',
    gridTemplateRows: 'auto 1fr',
    gap: '10px',
    height: '100%',
    overflow: 'hidden',
  },
  // Video player
  videoContainer: {
    backgroundColor: 'white',
    borderRadius: '8px',
    boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
    overflow: 'hidden',
  },
  videoWrapper: { backgroundColor: '#000', position: 'relative' },
  video: { width: '100%', maxHeight: '400px', display: 'block' },
  noVideo: {
    padding: '40px',
    textAlign: 'center',
    color: '#999',
    backgroundColor: '#1a1a1a',
  },
  videoControls: {
    padding: '8px 12px',
    backgroundColor: '#2a2a2a',
    color: 'white',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    fontSize: '12px',
  },
  videoIdBadge: {
    padding: '3px 6px',
    backgroundColor: '#4CAF50',
    borderRadius: '4px',
    fontSize: '10px',
    fontFamily: 'monospace',
  },
  // Detail panel
  detailPanel: {
    backgroundColor: 'white',
    borderRadius: '8px',
    boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
    display: 'flex',
    flexDirection: 'column',
    overflow: 'hidden',
    minHeight: 0,
  },
  detailHeader: {
    padding: '10px 12px',
    borderBottom: '1px solid #e0e0e0',
    backgroundColor: '#fafafa',
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  detailTitle: { fontWeight: 'bold', fontSize: '13px' },
  detailContent: {
    flex: 1,
    overflowY: 'auto',
    padding: '10px',
  },
  fieldGroup: { marginBottom: '10px' },
  fieldLabel: {
    fontSize: '9px',
    fontWeight: '600',
    color: '#666',
    marginBottom: '2px',
    textTransform: 'uppercase',
  },
  fieldValue: { fontSize: '12px', color: '#333' },
  // Segment styles
  segmentCard: {
    border: '1px solid #e0e0e0',
    borderRadius: '6px',
    padding: '10px',
    marginBottom: '8px',
    backgroundColor: '#fafafa',
  },
  segmentCardDeleted: {
    backgroundColor: '#ffebee',
    borderColor: '#ef9a9a',
    opacity: 0.7,
  },
  segmentHeader: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: '8px',
  },
  segmentTitle: {
    fontSize: '11px',
    fontWeight: '600',
    color: '#333',
  },
  segmentActions: {
    display: 'flex',
    gap: '6px',
  },
  deleteToggle: {
    padding: '3px 8px',
    fontSize: '10px',
    border: 'none',
    borderRadius: '3px',
    cursor: 'pointer',
  },
  deleteToggleActive: {
    backgroundColor: '#ef5350',
    color: 'white',
  },
  deleteToggleInactive: {
    backgroundColor: '#e0e0e0',
    color: '#666',
  },
  timestampRow: {
    display: 'flex',
    gap: '8px',
    alignItems: 'center',
    marginBottom: '6px',
  },
  timestampLabel: {
    fontSize: '10px',
    color: '#666',
    width: '35px',
  },
  timestampInput: {
    flex: 1,
    padding: '4px 6px',
    border: '1px solid #ddd',
    borderRadius: '3px',
    fontSize: '11px',
    fontFamily: 'monospace',
    maxWidth: '80px',
  },
  timestampInputModified: {
    borderColor: '#ff9800',
    backgroundColor: '#fff8e1',
  },
  smallButton: {
    padding: '3px 8px',
    fontSize: '10px',
    border: 'none',
    borderRadius: '3px',
    cursor: 'pointer',
    backgroundColor: '#1976D2',
    color: 'white',
  },
  setCurrentButton: {
    backgroundColor: '#ff9800',
  },
  segmentMeta: {
    fontSize: '10px',
    color: '#666',
    marginTop: '6px',
  },
  buttonRow: {
    display: 'flex',
    gap: '8px',
    marginTop: '10px',
    paddingTop: '10px',
    borderTop: '1px solid #e0e0e0',
  },
  saveButton: {
    padding: '8px 16px',
    backgroundColor: '#4CAF50',
    color: 'white',
    border: 'none',
    borderRadius: '4px',
    cursor: 'pointer',
    fontSize: '12px',
    fontWeight: '500',
  },
  resetButton: {
    padding: '8px 16px',
    backgroundColor: '#ff9800',
    color: 'white',
    border: 'none',
    borderRadius: '4px',
    cursor: 'pointer',
    fontSize: '12px',
  },
  saveAllButton: {
    padding: '6px 12px',
    backgroundColor: '#2196F3',
    color: 'white',
    border: 'none',
    borderRadius: '4px',
    cursor: 'pointer',
    fontSize: '11px',
    fontWeight: '500',
  },
  reasoning: {
    fontSize: '10px',
    color: '#666',
    lineHeight: '1.3',
    backgroundColor: '#f5f5f5',
    padding: '6px',
    borderRadius: '4px',
    maxHeight: '60px',
    overflowY: 'auto',
  },
  noSelection: {
    padding: '30px',
    textAlign: 'center',
    color: '#888',
    fontSize: '13px',
  },
  saveStatus: {
    fontSize: '11px',
    padding: '6px 10px',
    borderRadius: '4px',
    marginTop: '8px',
  },
  saveSuccess: { backgroundColor: '#e8f5e9', color: '#2e7d32' },
  saveError: { backgroundColor: '#ffebee', color: '#c62828' },
  modifiedCount: { fontSize: '10px', color: '#ff9800', fontWeight: '500' },
};

// Helper to get segments from item (supports both old and new format)
const getSegments = (item) => {
  if (item.dispensal_segments && Array.isArray(item.dispensal_segments)) {
    return item.dispensal_segments;
  }
  if (item.dispensal_segment) {
    return [item.dispensal_segment];
  }
  return [];
};

const AggregatedView = forwardRef(({
  aggregatedData,
  onLoadVideoAtTime,
  narrationTimestamps,
  videoUrl,
  videoId,
  currentTime,
  onTimeUpdate,
  onSave,
  participant,
}, ref) => {
  const [selectedItem, setSelectedItem] = useState(null);
  // modifications: { [itemId]: { segments: [...], deleted: Set<index> } }
  const [modifications, setModifications] = useState({});
  const [saveStatus, setSaveStatus] = useState(null);

  const videoRef = useRef(null);
  const [duration, setDuration] = useState(0);

  useImperativeHandle(ref, () => ({
    seekTo: (time) => {
      if (videoRef.current) videoRef.current.currentTime = time;
    },
    play: () => { if (videoRef.current) videoRef.current.play(); },
    pause: () => { if (videoRef.current) videoRef.current.pause(); },
  }));

  const handleVideoTimeUpdate = () => {
    if (videoRef.current && onTimeUpdate) {
      onTimeUpdate(videoRef.current.currentTime);
    }
  };

  const handleLoadedMetadata = () => {
    if (videoRef.current) setDuration(videoRef.current.duration);
  };

  const items = aggregatedData?.items || [];

  const handleItemClick = (item) => {
    setSelectedItem(item);
    const segments = getItemSegments(item);
    if (segments.length > 0 && onLoadVideoAtTime) {
      const firstSegment = segments[0];
      // Use segment's video_id if available, otherwise fall back to first video_range
      const videoId = firstSegment.video_id || item.video_range?.[0];
      if (videoId) {
        onLoadVideoAtTime(videoId, firstSegment.start_timestamp);
      }
    }
  };

  // Get segments for an item, applying any modifications
  const getItemSegments = (item) => {
    const mod = modifications[item.narration_id];
    if (mod && mod.segments) {
      return mod.segments;
    }
    return getSegments(item);
  };

  // Get deleted indices for an item
  const getDeletedIndices = (item) => {
    const mod = modifications[item.narration_id];
    return mod?.deleted || new Set();
  };

  // Check if item has any modifications
  const isItemModified = (itemId) => {
    return !!modifications[itemId];
  };

  // Update a segment's timestamp
  const handleSegmentChange = (segmentIndex, field, value) => {
    if (!selectedItem) return;
    const numValue = parseFloat(value);
    if (isNaN(numValue)) return;

    const itemId = selectedItem.narration_id;
    const currentSegments = getItemSegments(selectedItem);
    const currentDeleted = getDeletedIndices(selectedItem);

    const newSegments = currentSegments.map((seg, idx) => {
      if (idx === segmentIndex) {
        return { ...seg, [field]: numValue };
      }
      return seg;
    });

    setModifications(prev => ({
      ...prev,
      [itemId]: {
        segments: newSegments,
        deleted: currentDeleted,
      },
    }));
  };

  // Toggle segment deletion
  const handleToggleDelete = (segmentIndex) => {
    if (!selectedItem) return;
    const itemId = selectedItem.narration_id;
    const currentSegments = getItemSegments(selectedItem);
    const currentDeleted = new Set(getDeletedIndices(selectedItem));

    if (currentDeleted.has(segmentIndex)) {
      currentDeleted.delete(segmentIndex);
    } else {
      currentDeleted.add(segmentIndex);
    }

    setModifications(prev => ({
      ...prev,
      [itemId]: {
        segments: currentSegments,
        deleted: currentDeleted,
      },
    }));
  };

  // Seek video to timestamp, optionally loading a different video
  const handleSeekTo = (timestamp, segmentVideoId) => {
    if (timestamp === undefined) return;

    // If segment has a different video_id than currently loaded, load that video first
    if (segmentVideoId && segmentVideoId !== videoId && onLoadVideoAtTime) {
      onLoadVideoAtTime(segmentVideoId, timestamp);
    } else if (videoRef.current) {
      videoRef.current.currentTime = timestamp;
      if (onTimeUpdate) onTimeUpdate(timestamp);
    }
  };

  // Set current video time as segment start/end
  const handleSetCurrent = (segmentIndex, field) => {
    if (selectedItem && videoRef.current) {
      handleSegmentChange(segmentIndex, field, videoRef.current.currentTime);
    }
  };

  // Reset modifications for selected item
  const handleResetItem = () => {
    if (!selectedItem) return;
    setModifications(prev => {
      const newMods = { ...prev };
      delete newMods[selectedItem.narration_id];
      return newMods;
    });
  };

  // Save all modifications
  const handleSaveAll = async () => {
    if (Object.keys(modifications).length === 0) {
      setSaveStatus({ type: 'error', message: 'No modifications to save' });
      setTimeout(() => setSaveStatus(null), 3000);
      return;
    }

    // Build updated data, excluding deleted segments
    const updatedItems = items.map(item => {
      const mod = modifications[item.narration_id];
      if (!mod) return item;

      const segments = mod.segments || getSegments(item);
      const deleted = mod.deleted || new Set();

      // Filter out deleted segments
      const filteredSegments = segments.filter((_, idx) => !deleted.has(idx));

      // Recalculate total_count
      let totalCount = null;
      const countUnit = item.count_unit;
      if (filteredSegments.length > 0 && filteredSegments.some(s => s.count !== null)) {
        totalCount = filteredSegments.reduce((sum, s) => sum + (s.count || 0), 0);
      }

      return {
        ...item,
        dispensal_segments: filteredSegments,
        total_count: totalCount,
        num_segments: filteredSegments.length,
      };
    });

    const updatedData = {
      ...aggregatedData,
      items: updatedItems,
      last_modified: new Date().toISOString(),
    };

    try {
      if (onSave) {
        await onSave(updatedData);
        setSaveStatus({ type: 'success', message: `Saved ${Object.keys(modifications).length} items` });
        setModifications({});
      }
    } catch (error) {
      setSaveStatus({ type: 'error', message: `Save failed: ${error.message}` });
    }
    setTimeout(() => setSaveStatus(null), 3000);
  };

  const getDifficultyStyle = (difficulty) => {
    switch (difficulty) {
      case 'LOW': return styles.difficultyLow;
      case 'MID': return styles.difficultyMid;
      case 'HIGH': return styles.difficultyHigh;
      default: return {};
    }
  };

  const modifiedCount = Object.keys(modifications).length;
  const segments = selectedItem ? getItemSegments(selectedItem) : [];
  const deletedIndices = selectedItem ? getDeletedIndices(selectedItem) : new Set();

  return (
    <div style={styles.container}>
      {/* Left Panel: Item List */}
      <div style={styles.listPanel}>
        <div style={styles.listHeader}>
          <span>Items ({items.length})</span>
          {modifiedCount > 0 && (
            <span style={styles.modifiedCount}>{modifiedCount} modified</span>
          )}
        </div>
        <div style={styles.listContent}>
          {items.map((item, idx) => {
            const isSelected = selectedItem?.narration_id === item.narration_id;
            const isModified = isItemModified(item.narration_id);
            const itemSegments = getItemSegments(item);
            const numSegments = itemSegments.length;

            return (
              <div
                key={item.narration_id || idx}
                style={{
                  ...styles.itemRow,
                  ...(isSelected ? styles.itemRowSelected : {}),
                  ...(isModified ? styles.itemRowModified : {}),
                }}
                onClick={() => handleItemClick(item)}
              >
                <div style={styles.itemName}>
                  {item.food_name}
                  {isModified && <span style={{ color: '#ff9800', marginLeft: '4px' }}>*</span>}
                </div>
                <div style={styles.itemMeta}>
                  <span style={{ ...styles.badge, ...getDifficultyStyle(item.difficulty) }}>
                    {item.difficulty}
                  </span>
                  {item.total_count !== null && (
                    <span style={{ ...styles.badge, ...styles.amountBadge }}>
                      {item.total_count} {item.count_unit}
                    </span>
                  )}
                  {item.matched_ingredient_weight && (
                    <span style={{ ...styles.badge, ...styles.recipeBadge }}>
                      {item.matched_ingredient_weight.amount}{item.matched_ingredient_weight.unit}
                    </span>
                  )}
                  <span style={{ ...styles.badge, ...styles.segmentsBadge }}>
                    {numSegments} seg
                  </span>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Right Column: Video + Details */}
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
                <p style={{ fontSize: '14px', marginBottom: '6px' }}>No video loaded</p>
                <p style={{ fontSize: '11px' }}>Select an item to load its video</p>
              </div>
            )}
          </div>
          {videoUrl && (
            <div style={styles.videoControls}>
              <span style={{ fontFamily: 'monospace' }}>
                {formatTimestamp(currentTime)} / {formatTimestamp(duration)}
              </span>
              {videoId && <span style={styles.videoIdBadge}>{videoId}</span>}
            </div>
          )}
        </div>

        {/* Detail Panel */}
        <div style={styles.detailPanel}>
          <div style={styles.detailHeader}>
            <span style={styles.detailTitle}>
              {selectedItem ? selectedItem.food_name : 'Select an item'}
            </span>
            {modifiedCount > 0 && (
              <button style={styles.saveAllButton} onClick={handleSaveAll}>
                Save All ({modifiedCount})
              </button>
            )}
          </div>

          {selectedItem ? (
            <div style={styles.detailContent}>
              {/* Item info */}
              <div style={styles.fieldGroup}>
                <div style={styles.itemMeta}>
                  <span style={{ ...styles.badge, ...getDifficultyStyle(selectedItem.difficulty) }}>
                    {selectedItem.difficulty}
                  </span>
                  {selectedItem.total_count !== null && (
                    <span style={{ ...styles.badge, ...styles.amountBadge }}>
                      Total: {selectedItem.total_count} {selectedItem.count_unit}
                    </span>
                  )}
                  {selectedItem.matched_ingredient_weight && (
                    <span style={{ ...styles.badge, ...styles.recipeBadge }}>
                      Recipe: {selectedItem.matched_ingredient_weight.amount}{selectedItem.matched_ingredient_weight.unit} ({selectedItem.matched_ingredient_weight.recipe})
                    </span>
                  )}
                </div>
              </div>

              {/* Segments */}
              <div style={styles.fieldGroup}>
                <div style={styles.fieldLabel}>Time Segments ({segments.length})</div>
                {segments.map((segment, idx) => {
                  const isDeleted = deletedIndices.has(idx);
                  return (
                    <div
                      key={idx}
                      style={{
                        ...styles.segmentCard,
                        ...(isDeleted ? styles.segmentCardDeleted : {}),
                      }}
                    >
                      <div style={styles.segmentHeader}>
                        <div>
                          <span style={styles.segmentTitle}>
                            Segment {idx + 1}
                            {segment.count !== null && ` - ${segment.count} ${segment.count_unit || ''}`}
                          </span>
                          {segment.video_id && (
                            <span style={{
                              fontSize: '9px',
                              color: '#1565c0',
                              backgroundColor: '#e3f2fd',
                              padding: '1px 4px',
                              borderRadius: '3px',
                              marginLeft: '6px',
                              fontFamily: 'monospace',
                            }}>
                              {segment.video_id}
                            </span>
                          )}
                        </div>
                        <div style={styles.segmentActions}>
                          <button
                            style={{
                              ...styles.deleteToggle,
                              ...(isDeleted ? styles.deleteToggleActive : styles.deleteToggleInactive),
                            }}
                            onClick={() => handleToggleDelete(idx)}
                          >
                            {isDeleted ? 'Deleted' : 'Delete'}
                          </button>
                        </div>
                      </div>

                      {/* Start timestamp */}
                      <div style={styles.timestampRow}>
                        <span style={styles.timestampLabel}>Start:</span>
                        <input
                          type="number"
                          step="0.01"
                          value={segment.start_timestamp ?? ''}
                          onChange={(e) => handleSegmentChange(idx, 'start_timestamp', e.target.value)}
                          style={{
                            ...styles.timestampInput,
                            ...(isItemModified(selectedItem.narration_id) ? styles.timestampInputModified : {}),
                          }}
                          disabled={isDeleted}
                        />
                        <button
                          style={styles.smallButton}
                          onClick={() => handleSeekTo(segment.start_timestamp, segment.video_id)}
                          disabled={isDeleted}
                        >
                          Seek
                        </button>
                        <button
                          style={{ ...styles.smallButton, ...styles.setCurrentButton }}
                          onClick={() => handleSetCurrent(idx, 'start_timestamp')}
                          disabled={isDeleted}
                        >
                          Set
                        </button>
                      </div>

                      {/* End timestamp */}
                      <div style={styles.timestampRow}>
                        <span style={styles.timestampLabel}>End:</span>
                        <input
                          type="number"
                          step="0.01"
                          value={segment.end_timestamp ?? ''}
                          onChange={(e) => handleSegmentChange(idx, 'end_timestamp', e.target.value)}
                          style={{
                            ...styles.timestampInput,
                            ...(isItemModified(selectedItem.narration_id) ? styles.timestampInputModified : {}),
                          }}
                          disabled={isDeleted}
                        />
                        <button
                          style={styles.smallButton}
                          onClick={() => handleSeekTo(segment.end_timestamp, segment.video_id)}
                          disabled={isDeleted}
                        >
                          Seek
                        </button>
                        <button
                          style={{ ...styles.smallButton, ...styles.setCurrentButton }}
                          onClick={() => handleSetCurrent(idx, 'end_timestamp')}
                          disabled={isDeleted}
                        >
                          Set
                        </button>
                      </div>

                      {segment.count !== null && (
                        <div style={styles.segmentMeta}>
                          Count: {segment.count} {segment.count_unit || ''}
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>

              {/* Reasoning */}
              {selectedItem.reasoning && (
                <div style={styles.fieldGroup}>
                  <div style={styles.fieldLabel}>Reasoning</div>
                  <div style={styles.reasoning}>{selectedItem.reasoning}</div>
                </div>
              )}

              {/* Reset button */}
              {isItemModified(selectedItem.narration_id) && (
                <div style={styles.buttonRow}>
                  <button style={styles.resetButton} onClick={handleResetItem}>
                    Reset Changes
                  </button>
                </div>
              )}

              {saveStatus && (
                <div style={{
                  ...styles.saveStatus,
                  ...(saveStatus.type === 'success' ? styles.saveSuccess : styles.saveError),
                }}>
                  {saveStatus.message}
                </div>
              )}
            </div>
          ) : (
            <div style={styles.noSelection}>
              Select an item to view and edit its time segments
            </div>
          )}
        </div>
      </div>
    </div>
  );
});

AggregatedView.displayName = 'AggregatedView';

export default AggregatedView;

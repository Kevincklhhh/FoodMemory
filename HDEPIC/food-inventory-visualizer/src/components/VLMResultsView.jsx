import React, { useState, useRef, useImperativeHandle, forwardRef, useEffect } from 'react';
import { formatTimestamp } from '../utils/narrationParser';

const styles = {
  container: {
    display: 'grid',
    gridTemplateColumns: '380px 1fr',
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
  statsRow: {
    display: 'flex',
    gap: '8px',
    fontSize: '10px',
  },
  modelSelector: {
    padding: '4px 8px',
    fontSize: '11px',
    borderRadius: '4px',
    border: '1px solid #ddd',
    backgroundColor: 'white',
    cursor: 'pointer',
    fontWeight: '500',
  },
  headerRow: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    gap: '10px',
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
  // Match status badges
  matchExact: { backgroundColor: '#c8e6c9', color: '#2e7d32' },
  matchClose: { backgroundColor: '#fff9c4', color: '#f57f17' },
  matchWrong: { backgroundColor: '#ffcdd2', color: '#c62828' },
  matchUncountable: { backgroundColor: '#e0e0e0', color: '#616161' },
  gtBadge: { backgroundColor: '#e3f2fd', color: '#1565c0' },
  recipeBadge: { backgroundColor: '#fff8e1', color: '#ff8f00' },
  predBadge: { backgroundColor: '#f3e5f5', color: '#7b1fa2' },
  segmentsBadge: { backgroundColor: '#eceff1', color: '#455a64' },
  categoryDiscrete: { backgroundColor: '#e8f5e9', color: '#2e7d32' },
  categoryContinuous: { backgroundColor: '#fff3e0', color: '#e65100' },
  errorPositive: { backgroundColor: '#ffcdd2', color: '#c62828' },
  errorNegative: { backgroundColor: '#fff9c4', color: '#f57f17' },
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
  video: { width: '100%', maxHeight: '250px', display: 'block' },
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
  // Segment styles
  segmentCard: {
    border: '1px solid #e0e0e0',
    borderRadius: '6px',
    padding: '10px',
    marginBottom: '8px',
    backgroundColor: '#fafafa',
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
  comparisonRow: {
    display: 'grid',
    gridTemplateColumns: '1fr 1fr',
    gap: '10px',
    marginBottom: '8px',
  },
  comparisonBox: {
    padding: '8px',
    borderRadius: '4px',
    fontSize: '11px',
  },
  gtBox: {
    backgroundColor: '#e3f2fd',
    border: '1px solid #90caf9',
  },
  predBox: {
    backgroundColor: '#f3e5f5',
    border: '1px solid #ce93d8',
  },
  comparisonLabel: {
    fontSize: '9px',
    fontWeight: '600',
    color: '#666',
    marginBottom: '4px',
    textTransform: 'uppercase',
  },
  comparisonValue: {
    fontSize: '14px',
    fontWeight: '600',
  },
  comparisonAmount: {
    fontSize: '11px',
    fontStyle: 'italic',
    color: '#555',
    marginTop: '4px',
    lineHeight: '1.3',
  },
  comparisonUnit: {
    fontSize: '10px',
    color: '#666',
    marginLeft: '4px',
  },
  visualEvidence: {
    fontSize: '10px',
    color: '#555',
    lineHeight: '1.4',
    backgroundColor: '#e8f5e9',
    padding: '8px',
    borderRadius: '4px',
    marginTop: '6px',
    borderLeft: '3px solid #4caf50',
  },
  timestampRow: {
    display: 'flex',
    gap: '8px',
    alignItems: 'center',
    marginBottom: '6px',
    fontSize: '11px',
  },
  timestampLabel: {
    fontSize: '10px',
    color: '#666',
    width: '35px',
  },
  timestampValue: {
    fontFamily: 'monospace',
    fontSize: '11px',
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
  matchStatusRow: {
    display: 'flex',
    gap: '8px',
    alignItems: 'center',
    marginBottom: '6px',
    padding: '6px 8px',
    borderRadius: '4px',
    fontSize: '11px',
  },
  reasoning: {
    fontSize: '10px',
    color: '#666',
    lineHeight: '1.4',
    backgroundColor: '#f5f5f5',
    padding: '8px',
    borderRadius: '4px',
    maxHeight: '80px',
    overflowY: 'auto',
  },
  noSelection: {
    padding: '30px',
    textAlign: 'center',
    color: '#888',
    fontSize: '13px',
  },
  summaryStats: {
    display: 'flex',
    gap: '12px',
    padding: '8px',
    backgroundColor: '#f5f5f5',
    borderRadius: '4px',
    marginBottom: '10px',
    fontSize: '11px',
  },
  statItem: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
  },
  statValue: {
    fontSize: '16px',
    fontWeight: '600',
  },
  statLabel: {
    fontSize: '9px',
    color: '#666',
    textTransform: 'uppercase',
  },
};

const VLMResultsView = forwardRef(({
  vlmData,
  availableModels = [],
  selectedModel,
  onModelChange,
  onLoadVideoAtTime,
  videoUrl,
  videoId,
  currentTime,
  onTimeUpdate,
}, ref) => {
  const [selectedItem, setSelectedItem] = useState(null);
  const [filterMatch, setFilterMatch] = useState('all'); // all, exact, close, wrong, uncountable

  const videoRef = useRef(null);
  const [duration, setDuration] = useState(0);

  // Reset selected item when model/data changes
  useEffect(() => {
    setSelectedItem(null);
  }, [vlmData, selectedModel]);

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

  const items = vlmData?.items || [];

  // Calculate summary stats
  const stats = {
    total: items.length,
    exact: 0,
    close: 0,
    wrong: 0,
    uncountable: 0,
  };

  items.forEach(item => {
    item.segments?.forEach(seg => {
      if (seg.match === 'exact') stats.exact++;
      else if (seg.match === 'close') stats.close++;
      else if (seg.match === 'wrong') stats.wrong++;
      else stats.uncountable++;
    });
  });

  // Filter items based on match type
  const filteredItems = filterMatch === 'all'
    ? items
    : items.filter(item =>
        item.segments?.some(seg => {
          if (filterMatch === 'uncountable') {
            return ['pred_uncountable', 'gt_uncountable', 'both_uncountable'].includes(seg.match);
          }
          return seg.match === filterMatch;
        })
      );

  const handleItemClick = (item) => {
    setSelectedItem(item);
    const segments = item.segments || [];
    if (segments.length > 0 && onLoadVideoAtTime) {
      const firstSegment = segments[0];
      const segVideoId = firstSegment.video_id || item.video_range?.[0];
      if (segVideoId) {
        onLoadVideoAtTime(segVideoId, firstSegment.start_timestamp);
      }
    }
  };

  const handleSeekTo = (timestamp, segmentVideoId) => {
    if (timestamp === undefined) return;
    if (segmentVideoId && segmentVideoId !== videoId && onLoadVideoAtTime) {
      onLoadVideoAtTime(segmentVideoId, timestamp);
    } else if (videoRef.current) {
      videoRef.current.currentTime = timestamp;
      if (onTimeUpdate) onTimeUpdate(timestamp);
    }
  };

  const getDifficultyStyle = (difficulty) => {
    switch (difficulty) {
      case 'LOW': return styles.difficultyLow;
      case 'MID': return styles.difficultyMid;
      case 'HIGH': return styles.difficultyHigh;
      default: return {};
    }
  };

  const getMatchStyle = (match) => {
    if (match === 'exact') return styles.matchExact;
    if (match === 'close') return styles.matchClose;
    if (match === 'wrong') return styles.matchWrong;
    return styles.matchUncountable;
  };

  const getMatchLabel = (match) => {
    if (match === 'exact') return 'Exact Match';
    if (match === 'close') return 'Close';
    if (match === 'wrong') return 'Wrong';
    if (match === 'pred_uncountable') return 'Pred N/A';
    if (match === 'gt_uncountable') return 'GT N/A';
    if (match === 'both_uncountable') return 'Both N/A';
    return match;
  };

  const formatCount = (count, unit) => {
    if (count === null || count === undefined) return 'N/A';
    return `${count} ${unit || ''}`;
  };

  const segments = selectedItem?.segments || [];

  return (
    <div style={styles.container}>
      {/* Left Panel: Item List */}
      <div style={styles.listPanel}>
        <div style={styles.listHeader}>
          <div style={styles.headerRow}>
            <span>VLM Results ({filteredItems.length}/{items.length})</span>
            {availableModels.length > 1 && (
              <select
                style={styles.modelSelector}
                value={selectedModel || ''}
                onChange={(e) => onModelChange && onModelChange(e.target.value)}
              >
                {availableModels.map(model => (
                  <option key={model} value={model}>
                    {model.toUpperCase()}
                  </option>
                ))}
              </select>
            )}
          </div>
          <div style={styles.statsRow}>
            <span
              style={{ ...styles.badge, ...styles.matchExact, cursor: 'pointer' }}
              onClick={() => setFilterMatch(filterMatch === 'exact' ? 'all' : 'exact')}
            >
              {stats.exact}
            </span>
            <span
              style={{ ...styles.badge, ...styles.matchClose, cursor: 'pointer' }}
              onClick={() => setFilterMatch(filterMatch === 'close' ? 'all' : 'close')}
            >
              {stats.close}
            </span>
            <span
              style={{ ...styles.badge, ...styles.matchWrong, cursor: 'pointer' }}
              onClick={() => setFilterMatch(filterMatch === 'wrong' ? 'all' : 'wrong')}
            >
              {stats.wrong}
            </span>
            <span
              style={{ ...styles.badge, ...styles.matchUncountable, cursor: 'pointer' }}
              onClick={() => setFilterMatch(filterMatch === 'uncountable' ? 'all' : 'uncountable')}
            >
              {stats.uncountable}
            </span>
          </div>
        </div>
        <div style={styles.listContent}>
          {filteredItems.map((item, idx) => {
            const isSelected = selectedItem?.narration_id === item.narration_id;
            const numSegments = item.segments?.length || 0;
            // Determine overall match status for item
            const matchTypes = item.segments?.map(s => s.match) || [];
            const hasWrong = matchTypes.includes('wrong');
            const hasClose = matchTypes.includes('close');
            const hasExact = matchTypes.includes('exact');

            let overallMatch = 'uncountable';
            if (hasWrong) overallMatch = 'wrong';
            else if (hasClose) overallMatch = 'close';
            else if (hasExact) overallMatch = 'exact';

            return (
              <div
                key={item.narration_id || idx}
                style={{
                  ...styles.itemRow,
                  ...(isSelected ? styles.itemRowSelected : {}),
                }}
                onClick={() => handleItemClick(item)}
              >
                <div style={styles.itemName}>{item.food_name}</div>
                <div style={styles.itemMeta}>
                  <span style={{ ...styles.badge, ...getDifficultyStyle(item.difficulty) }}>
                    {item.difficulty}
                  </span>
                  <span style={{ ...styles.badge, ...getMatchStyle(overallMatch) }}>
                    {overallMatch}
                  </span>
                  {item.total_ground_truth !== null ? (
                    <span style={{ ...styles.badge, ...styles.gtBadge }}>
                      GT: {item.total_ground_truth}
                    </span>
                  ) : item.recipe_amount ? (
                    <span style={{ ...styles.badge, ...styles.recipeBadge }}>
                      Recipe: {item.recipe_amount.amount}{item.recipe_amount.unit}
                    </span>
                  ) : null}
                  {item.total_predicted !== null && (
                    <span style={{ ...styles.badge, ...styles.predBadge }}>
                      Pred: {item.total_predicted}
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
            {selectedItem && (
              <span style={{ ...styles.badge, ...getDifficultyStyle(selectedItem.difficulty) }}>
                {selectedItem.difficulty}
              </span>
            )}
          </div>

          {selectedItem ? (
            <div style={styles.detailContent}>
              {/* Summary */}
              <div style={styles.summaryStats}>
                <div style={styles.statItem}>
                  <span style={{ ...styles.statValue, color: selectedItem.total_ground_truth !== null ? '#1565c0' : selectedItem.recipe_amount ? '#ff8f00' : '#999' }}>
                    {selectedItem.total_ground_truth ?? (selectedItem.recipe_amount ? `${selectedItem.recipe_amount.amount}${selectedItem.recipe_amount.unit}` : 'N/A')}
                  </span>
                  <span style={styles.statLabel}>
                    {selectedItem.total_ground_truth !== null ? 'Ground Truth' : selectedItem.recipe_amount ? 'Recipe' : 'Ground Truth'}
                  </span>
                </div>
                <div style={styles.statItem}>
                  <span style={{ ...styles.statValue, color: '#7b1fa2' }}>
                    {selectedItem.total_predicted ?? 'N/A'}
                  </span>
                  <span style={styles.statLabel}>Predicted</span>
                </div>
                <div style={styles.statItem}>
                  <span style={styles.statValue}>{segments.length}</span>
                  <span style={styles.statLabel}>Segments</span>
                </div>
              </div>

              {/* Segments */}
              <div style={styles.fieldGroup}>
                <div style={styles.fieldLabel}>Segments</div>
                {segments.map((segment, idx) => (
                  <div key={idx} style={styles.segmentCard}>
                    <div style={styles.segmentHeader}>
                      <div>
                        <span style={styles.segmentTitle}>Segment {idx + 1}</span>
                        {segment.quantity_category && (
                          <span style={{
                            ...styles.badge,
                            ...(segment.quantity_category === 'discrete' ? styles.categoryDiscrete : styles.categoryContinuous),
                            marginLeft: '6px',
                            fontSize: '8px',
                          }}>
                            {segment.quantity_category}
                          </span>
                        )}
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
                      <span style={{ ...styles.badge, ...getMatchStyle(segment.match), fontSize: '9px' }}>
                        {getMatchLabel(segment.match)}
                        {segment.error !== null && segment.error !== 0 && (
                          <span style={{ marginLeft: '4px' }}>
                            ({segment.error > 0 ? '+' : ''}{segment.error})
                          </span>
                        )}
                      </span>
                    </div>

                    {/* Ground Truth vs Prediction */}
                    <div style={styles.comparisonRow}>
                      <div style={{
                        ...styles.comparisonBox,
                        ...styles.gtBox,
                        ...(segment.ground_truth_count === null && selectedItem.recipe_amount ? { backgroundColor: '#fff8e1', borderColor: '#ffcc80' } : {}),
                      }}>
                        <div style={styles.comparisonLabel}>
                          {segment.ground_truth_count !== null ? 'Ground Truth' : selectedItem.recipe_amount ? 'Recipe Amount' : 'Ground Truth'}
                        </div>
                        {segment.ground_truth_count !== null ? (
                          <>
                            <span style={styles.comparisonValue}>
                              {segment.ground_truth_count}
                            </span>
                            <span style={styles.comparisonUnit}>
                              {segment.ground_truth_unit || ''}
                            </span>
                          </>
                        ) : selectedItem.recipe_amount ? (
                          <>
                            <span style={styles.comparisonValue}>
                              {selectedItem.recipe_amount.amount}
                            </span>
                            <span style={styles.comparisonUnit}>
                              {selectedItem.recipe_amount.unit}
                            </span>
                            <div style={{ fontSize: '9px', color: '#666', marginTop: '2px' }}>
                              ({selectedItem.recipe_amount.recipe})
                            </div>
                          </>
                        ) : (
                          <span style={styles.comparisonValue}>N/A</span>
                        )}
                      </div>
                      <div style={{ ...styles.comparisonBox, ...styles.predBox }}>
                        <div style={styles.comparisonLabel}>Prediction</div>
                        {segment.predicted_count !== null ? (
                          <>
                            <span style={styles.comparisonValue}>
                              {segment.predicted_count}
                            </span>
                            <span style={styles.comparisonUnit}>
                              {segment.predicted_unit || ''}
                            </span>
                          </>
                        ) : segment.predicted_amount ? (
                          <div style={styles.comparisonAmount}>
                            {segment.predicted_amount}
                          </div>
                        ) : (
                          <span style={styles.comparisonValue}>N/A</span>
                        )}
                      </div>
                    </div>

                    {/* Timestamps */}
                    <div style={styles.timestampRow}>
                      <span style={styles.timestampLabel}>Time:</span>
                      <span style={styles.timestampValue}>
                        {formatTimestamp(segment.start_timestamp)} - {formatTimestamp(segment.end_timestamp)}
                      </span>
                      <button
                        style={styles.smallButton}
                        onClick={() => handleSeekTo(segment.start_timestamp, segment.video_id)}
                      >
                        Start
                      </button>
                      <button
                        style={styles.smallButton}
                        onClick={() => handleSeekTo(segment.end_timestamp, segment.video_id)}
                      >
                        End
                      </button>
                    </div>

                    {/* Confidence */}
                    {segment.confidence && (
                      <div style={{ fontSize: '10px', color: '#666', marginBottom: '6px' }}>
                        Confidence: <strong>{segment.confidence}</strong>
                      </div>
                    )}

                    {/* Reasoning */}
                    {segment.reasoning && (
                      <div style={styles.reasoning}>{segment.reasoning}</div>
                    )}

                    {/* Visual Evidence */}
                    {segment.visual_evidence && (
                      <div style={styles.visualEvidence}>
                        <strong>Visual:</strong> {segment.visual_evidence}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <div style={styles.noSelection}>
              Select an item to view VLM predictions vs ground truth
            </div>
          )}
        </div>
      </div>
    </div>
  );
});

VLMResultsView.displayName = 'VLMResultsView';

export default VLMResultsView;

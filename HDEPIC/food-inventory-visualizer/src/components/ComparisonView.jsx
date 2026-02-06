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
  },
  headerRow: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    gap: '10px',
    marginBottom: '6px',
  },
  tagSelectRow: {
    display: 'flex',
    gap: '6px',
    alignItems: 'center',
    marginBottom: '6px',
  },
  tagLabel: {
    fontSize: '10px',
    fontWeight: '600',
    padding: '2px 5px',
    borderRadius: '3px',
    minWidth: '28px',
    textAlign: 'center',
  },
  tagLabelA: {
    backgroundColor: '#e3f2fd',
    color: '#1565c0',
  },
  tagLabelB: {
    backgroundColor: '#f3e5f5',
    color: '#7b1fa2',
  },
  tagSelector: {
    padding: '3px 6px',
    fontSize: '10px',
    borderRadius: '4px',
    border: '1px solid #ddd',
    backgroundColor: 'white',
    cursor: 'pointer',
    fontWeight: '500',
    flex: 1,
    minWidth: 0,
  },
  statsRow: {
    display: 'flex',
    gap: '8px',
    fontSize: '10px',
  },
  badge: {
    fontSize: '8px',
    padding: '1px 4px',
    borderRadius: '3px',
    fontWeight: '500',
  },
  filterBadge: {
    fontSize: '9px',
    padding: '2px 6px',
    borderRadius: '3px',
    fontWeight: '500',
    cursor: 'pointer',
    border: '1px solid transparent',
  },
  filterActive: {
    outline: '2px solid #333',
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
  diffIndicator: {
    width: '8px',
    height: '8px',
    borderRadius: '50%',
    display: 'inline-block',
    marginRight: '4px',
  },
  // Difficulty
  difficultyLow: { backgroundColor: '#e8f5e9', color: '#2e7d32' },
  difficultyMid: { backgroundColor: '#fff3e0', color: '#e65100' },
  difficultyHigh: { backgroundColor: '#ffebee', color: '#c62828' },
  // Match
  matchExact: { backgroundColor: '#c8e6c9', color: '#2e7d32' },
  matchClose: { backgroundColor: '#fff9c4', color: '#f57f17' },
  matchWrong: { backgroundColor: '#ffcdd2', color: '#c62828' },
  matchUncountable: { backgroundColor: '#e0e0e0', color: '#616161' },
  // Right panel
  rightColumn: {
    display: 'grid',
    gridTemplateRows: 'auto 1fr',
    gap: '10px',
    height: '100%',
    overflow: 'hidden',
  },
  videoContainer: {
    backgroundColor: 'white',
    borderRadius: '8px',
    boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
    overflow: 'hidden',
    display: 'flex',
    flexDirection: 'column',
    maxHeight: '350px',
  },
  videoWrapper: { backgroundColor: '#000', position: 'relative', flex: 1, minHeight: 0, overflow: 'hidden' },
  video: { width: '100%', height: '100%', objectFit: 'contain', display: 'block' },
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
  noSelection: {
    padding: '30px',
    textAlign: 'center',
    color: '#888',
    fontSize: '13px',
  },
  // Segment cards
  segmentCard: {
    border: '1px solid #e0e0e0',
    borderRadius: '6px',
    padding: '10px',
    marginBottom: '8px',
    backgroundColor: '#fafafa',
    borderLeftWidth: '4px',
    borderLeftStyle: 'solid',
  },
  segmentDiffers: {
    borderLeftColor: '#ef5350',
  },
  segmentAgrees: {
    borderLeftColor: '#66bb6a',
  },
  segmentMissing: {
    borderLeftColor: '#bdbdbd',
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
  // Comparison columns
  comparisonColumns: {
    display: 'grid',
    gridTemplateColumns: '1fr 1fr',
    gap: '8px',
    marginTop: '6px',
  },
  tagColumnA: {
    padding: '8px',
    borderRadius: '4px',
    backgroundColor: '#e3f2fd',
    border: '1px solid #90caf9',
    fontSize: '11px',
  },
  tagColumnB: {
    padding: '8px',
    borderRadius: '4px',
    backgroundColor: '#f3e5f5',
    border: '1px solid #ce93d8',
    fontSize: '11px',
  },
  tagColumnMissing: {
    padding: '8px',
    borderRadius: '4px',
    backgroundColor: '#f5f5f5',
    border: '1px dashed #bdbdbd',
    fontSize: '11px',
    color: '#999',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    fontStyle: 'italic',
  },
  columnLabel: {
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
  comparisonUnit: {
    fontSize: '10px',
    color: '#666',
    marginLeft: '4px',
  },
  // GT box (shared)
  gtBox: {
    padding: '6px 8px',
    borderRadius: '4px',
    backgroundColor: '#e3f2fd',
    border: '1px solid #90caf9',
    fontSize: '11px',
    marginBottom: '6px',
  },
  timestampRow: {
    display: 'flex',
    gap: '8px',
    alignItems: 'center',
    marginBottom: '6px',
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
  visualEvidence: {
    fontSize: '10px',
    color: '#555',
    lineHeight: '1.4',
    backgroundColor: '#e8f5e9',
    padding: '6px',
    borderRadius: '4px',
    marginTop: '4px',
    borderLeft: '3px solid #4caf50',
  },
  // Evidence frames
  evidenceFramesRow: {
    display: 'flex',
    gap: '6px',
    flexWrap: 'wrap',
    marginTop: '4px',
  },
  evidenceFrameCard: {
    flex: '1 1 100px',
    maxWidth: '140px',
    padding: '6px',
    borderRadius: '4px',
    fontSize: '10px',
    border: '1px solid #e0e0e0',
    backgroundColor: '#fafafa',
  },
  synthesisBox: {
    padding: '6px',
    borderRadius: '4px',
    backgroundColor: '#f3e5f5',
    border: '1px solid #ce93d8',
    marginTop: '4px',
    fontSize: '10px',
    lineHeight: '1.4',
  },
  pathsSummary: {
    display: 'flex',
    gap: '4px',
    flexWrap: 'wrap',
    marginTop: '4px',
  },
  pathStatusBadge: {
    fontSize: '8px',
    padding: '2px 5px',
    borderRadius: '3px',
    fontWeight: '600',
    display: 'inline-block',
  },
  pathValid: { backgroundColor: '#c8e6c9', color: '#2e7d32' },
  pathInvalid: { backgroundColor: '#ffcdd2', color: '#c62828' },
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
  if (match === 'exact') return 'Exact';
  if (match === 'close') return 'Close';
  if (match === 'wrong') return 'Wrong';
  if (match === 'pred_uncountable') return 'Pred N/A';
  if (match === 'gt_uncountable') return 'GT N/A';
  if (match === 'both_uncountable') return 'Both N/A';
  if (match === 'continuous_estimate') return 'Continuous';
  if (match === 'item_mismatch') return 'Mismatch';
  return match || 'N/A';
};

const ComparisonView = forwardRef(({
  comparisonData,
  tagA,
  tagB,
  availableTags,
  onTagAChange,
  onTagBChange,
  onLoadVideoAtTime,
  videoUrl,
  videoId,
  currentTime,
  onTimeUpdate,
  participant,
  // Failure cases overlay props
  failureCasesFiles = [],
  overlayFailureCasesFile,
  overlayFailureCases,
  overlayDirty,
  onLoadOverlayFailureCases,
  onSaveOverlayFailureCases,
  onToggleSegmentInFailureCases,
  onUpdateFailureCaseNotes,
}, ref) => {
  const [selectedItem, setSelectedItem] = useState(null);
  const [filter, setFilter] = useState('all'); // all, differs, agrees, onlyA, onlyB
  const [overlayExpanded, setOverlayExpanded] = useState(false);
  const [overlayFileSelection, setOverlayFileSelection] = useState('');

  const videoRef = useRef(null);
  const [duration, setDuration] = useState(0);

  useEffect(() => {
    setSelectedItem(null);
    setFilter('all');
  }, [comparisonData, tagA, tagB]);

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

  const items = comparisonData?.items || [];

  // Stats
  const totalItems = items.length;
  const differsCount = items.filter(i => i.hasDifferences).length;
  const agreesCount = items.filter(i => !i.hasDifferences && !i.onlyInA && !i.onlyInB).length;
  const onlyACount = items.filter(i => i.onlyInA).length;
  const onlyBCount = items.filter(i => i.onlyInB).length;

  // Build overlay lookup
  const overlaySegmentIds = new Set();
  const overlayCaseLookup = {};
  if (overlayFailureCases?.cases) {
    for (const c of overlayFailureCases.cases) {
      if (c.segment_id) {
        overlaySegmentIds.add(c.segment_id);
        overlayCaseLookup[c.segment_id] = c;
      }
    }
  }

  const overlayCountByItem = {};
  if (overlaySegmentIds.size > 0) {
    items.forEach(item => {
      const count = (item.segments || []).filter(s => overlaySegmentIds.has(s.segment_id)).length;
      if (count > 0) overlayCountByItem[item.narration_id] = count;
    });
  }

  // Filter
  const filteredItems = items.filter(item => {
    if (filter === 'all') return true;
    if (filter === 'differs') return item.hasDifferences;
    if (filter === 'agrees') return !item.hasDifferences && !item.onlyInA && !item.onlyInB;
    if (filter === 'onlyA') return item.onlyInA;
    if (filter === 'onlyB') return item.onlyInB;
    return true;
  });

  const handleItemClick = (item) => {
    setSelectedItem(item);
    const segments = item.segments || [];
    if (segments.length > 0 && onLoadVideoAtTime) {
      const firstSeg = segments[0];
      const segVideoId = firstSeg.video_id;
      if (segVideoId) {
        onLoadVideoAtTime(segVideoId, firstSeg.start_timestamp);
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

  const renderTagColumn = (segData, tagName, columnStyle) => {
    if (!segData) {
      return (
        <div style={styles.tagColumnMissing}>
          Not in {tagName}
        </div>
      );
    }

    return (
      <div style={columnStyle}>
        <div style={styles.columnLabel}>{tagName}</div>
        {/* Prediction */}
        <div style={{ marginBottom: '4px' }}>
          <span style={styles.comparisonValue}>
            {segData.predicted_count != null ? segData.predicted_count : 'N/A'}
          </span>
          {segData.predicted_unit && (
            <span style={styles.comparisonUnit}>{segData.predicted_unit}</span>
          )}
          {segData.predicted_amount && segData.predicted_count == null && (
            <div style={{ fontSize: '10px', fontStyle: 'italic', color: '#555' }}>
              {segData.predicted_amount}
            </div>
          )}
        </div>
        {/* Match badge */}
        {segData.match && (
          <span style={{ ...styles.badge, ...getMatchStyle(segData.match), fontSize: '8px' }}>
            {getMatchLabel(segData.match)}
          </span>
        )}
        {/* Confidence */}
        {segData.confidence && (
          <div style={{ fontSize: '9px', color: '#666', marginTop: '3px' }}>
            Confidence: {segData.confidence}
          </div>
        )}
        {/* Visual evidence */}
        {segData.visual_evidence && (
          <div style={{ ...styles.visualEvidence, marginTop: '4px' }}>
            {segData.visual_evidence}
          </div>
        )}
        {/* Evidence frames */}
        {segData.evidence_frames && segData.evidence_frames.length > 0 && (
          <div style={styles.evidenceFramesRow}>
            {segData.evidence_frames.map((frame, fIdx) => (
              <div key={fIdx} style={styles.evidenceFrameCard}>
                <div style={{ fontWeight: '600', fontSize: '9px', textTransform: 'uppercase', marginBottom: '2px' }}>
                  {(frame.role || 'frame').replace(/_/g, ' ')}
                </div>
                <div style={{ fontFamily: 'monospace', fontSize: '10px' }}>
                  {frame.absolute_timestamp != null ? formatTimestamp(frame.absolute_timestamp) : frame.timestamp_raw || '?'}
                </div>
                {frame.description && (
                  <div style={{ fontSize: '9px', color: '#555', marginTop: '2px' }}>{frame.description}</div>
                )}
                {frame.container_description && (
                  <div style={{ fontSize: '8px', color: '#444', marginTop: '2px', fontStyle: 'italic' }}>{frame.container_description}</div>
                )}
                {frame.visible_count != null && (
                  <div style={{ fontSize: '9px', fontWeight: '600', marginTop: '2px' }}>Count: {frame.visible_count}</div>
                )}
                {frame.absolute_timestamp != null && (
                  <button
                    style={{ ...styles.smallButton, fontSize: '9px', padding: '2px 6px', marginTop: '2px' }}
                    onClick={() => handleSeekTo(frame.absolute_timestamp, segData.video_id)}
                  >
                    Seek
                  </button>
                )}
              </div>
            ))}
          </div>
        )}
        {/* Paths (multipath) */}
        {segData.paths && (
          <div style={styles.pathsSummary}>
            {['source', 'destination', 'transfer'].map(pathKey => {
              const path = segData.paths[pathKey];
              if (!path) return null;
              const isValid = path.status === 'VALID';
              return (
                <span key={pathKey} style={{
                  ...styles.pathStatusBadge,
                  ...(isValid ? styles.pathValid : styles.pathInvalid),
                }}>
                  {pathKey}: {path.status}
                </span>
              );
            })}
          </div>
        )}
        {/* Synthesis */}
        {segData.final_synthesis && (
          <div style={styles.synthesisBox}>
            <strong>Synth:</strong> {segData.final_synthesis.best_path_selected}
            {segData.final_synthesis.final_count_estimate != null && (
              <> = <strong>{segData.final_synthesis.final_count_estimate}</strong></>
            )}
          </div>
        )}
      </div>
    );
  };

  const selectedMergedItem = selectedItem;
  const selectedSegments = selectedMergedItem?.segments || [];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: 'calc(100vh - 260px)', minHeight: '600px' }}>
      {/* Failure Cases Overlay Toolbar */}
      {failureCasesFiles.length > 0 && onLoadOverlayFailureCases && (
        <div style={{
          backgroundColor: overlayFailureCasesFile ? '#fff3e0' : '#f5f5f5',
          borderRadius: '6px',
          padding: overlayExpanded ? '8px 12px' : '4px 12px',
          marginBottom: '8px',
          border: overlayFailureCasesFile ? '1px solid #ffcc80' : '1px solid #e0e0e0',
          fontSize: '11px',
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer' }}
            onClick={() => setOverlayExpanded(!overlayExpanded)}
          >
            <span style={{ fontWeight: '600', color: overlayFailureCasesFile ? '#e65100' : '#666' }}>
              {overlayExpanded ? '\u25BC' : '\u25B6'} Failure Cases Overlay
            </span>
            {overlayFailureCasesFile && (
              <span style={{ fontSize: '10px', color: '#e65100' }}>
                ({overlaySegmentIds.size} cases from {overlayFailureCasesFile})
              </span>
            )}
            {overlayDirty && (
              <span style={{ fontSize: '9px', color: '#d32f2f', fontWeight: '600' }}>UNSAVED</span>
            )}
          </div>
          {overlayExpanded && (
            <div style={{ display: 'flex', gap: '8px', alignItems: 'center', marginTop: '6px' }}>
              <select
                style={{ padding: '4px 6px', fontSize: '11px', borderRadius: '4px', border: '1px solid #ddd', flex: 1, minWidth: 0 }}
                value={overlayFileSelection || overlayFailureCasesFile || ''}
                onChange={(e) => setOverlayFileSelection(e.target.value)}
              >
                <option value="">-- Select file --</option>
                {failureCasesFiles.map(f => (
                  <option key={f.filename} value={f.filename}>{f.filename} ({f.case_count} cases)</option>
                ))}
              </select>
              <button
                onClick={() => {
                  const file = overlayFileSelection || overlayFailureCasesFile;
                  if (file) onLoadOverlayFailureCases(file);
                }}
                style={{ padding: '4px 10px', fontSize: '10px', backgroundColor: '#1976D2', color: 'white', border: 'none', borderRadius: '4px', cursor: 'pointer' }}
              >
                Load
              </button>
              {overlayFailureCasesFile && (
                <>
                  <button
                    onClick={onSaveOverlayFailureCases}
                    disabled={!overlayDirty}
                    style={{
                      padding: '4px 10px', fontSize: '10px', border: 'none', borderRadius: '4px', cursor: overlayDirty ? 'pointer' : 'not-allowed',
                      backgroundColor: overlayDirty ? '#2E7D32' : '#bbb', color: 'white',
                    }}
                  >
                    Save
                  </button>
                  <button
                    onClick={() => onLoadOverlayFailureCases(null)}
                    style={{ padding: '4px 10px', fontSize: '10px', backgroundColor: '#757575', color: 'white', border: 'none', borderRadius: '4px', cursor: 'pointer' }}
                  >
                    Clear
                  </button>
                </>
              )}
            </div>
          )}
        </div>
      )}

      <div style={{ ...styles.container, flex: 1, minHeight: 0, height: 'auto' }}>
      {/* Left Panel */}
      <div style={styles.listPanel}>
        <div style={styles.listHeader}>
          <div style={styles.headerRow}>
            <span>Compare ({filteredItems.length}/{totalItems})</span>
          </div>
          {/* Tag selectors */}
          <div style={styles.tagSelectRow}>
            <span style={{ ...styles.tagLabel, ...styles.tagLabelA }}>A</span>
            <select
              style={styles.tagSelector}
              value={tagA || ''}
              onChange={(e) => onTagAChange(e.target.value)}
            >
              {availableTags.map(tag => (
                <option key={tag} value={tag}>{tag}</option>
              ))}
            </select>
          </div>
          <div style={styles.tagSelectRow}>
            <span style={{ ...styles.tagLabel, ...styles.tagLabelB }}>B</span>
            <select
              style={styles.tagSelector}
              value={tagB || ''}
              onChange={(e) => onTagBChange(e.target.value)}
            >
              {availableTags.map(tag => (
                <option key={tag} value={tag}>{tag}</option>
              ))}
            </select>
          </div>
          {/* Filter badges */}
          <div style={styles.statsRow}>
            <span
              style={{
                ...styles.filterBadge,
                backgroundColor: '#e0e0e0', color: '#333',
                ...(filter === 'all' ? styles.filterActive : {}),
              }}
              onClick={() => setFilter('all')}
            >
              All: {totalItems}
            </span>
            <span
              style={{
                ...styles.filterBadge,
                backgroundColor: '#ffcdd2', color: '#c62828',
                ...(filter === 'differs' ? styles.filterActive : {}),
              }}
              onClick={() => setFilter(filter === 'differs' ? 'all' : 'differs')}
            >
              Diff: {differsCount}
            </span>
            <span
              style={{
                ...styles.filterBadge,
                backgroundColor: '#c8e6c9', color: '#2e7d32',
                ...(filter === 'agrees' ? styles.filterActive : {}),
              }}
              onClick={() => setFilter(filter === 'agrees' ? 'all' : 'agrees')}
            >
              Agree: {agreesCount}
            </span>
            {onlyACount > 0 && (
              <span
                style={{
                  ...styles.filterBadge,
                  backgroundColor: '#e3f2fd', color: '#1565c0',
                  ...(filter === 'onlyA' ? styles.filterActive : {}),
                }}
                onClick={() => setFilter(filter === 'onlyA' ? 'all' : 'onlyA')}
              >
                A only: {onlyACount}
              </span>
            )}
            {onlyBCount > 0 && (
              <span
                style={{
                  ...styles.filterBadge,
                  backgroundColor: '#f3e5f5', color: '#7b1fa2',
                  ...(filter === 'onlyB' ? styles.filterActive : {}),
                }}
                onClick={() => setFilter(filter === 'onlyB' ? 'all' : 'onlyB')}
              >
                B only: {onlyBCount}
              </span>
            )}
          </div>
        </div>
        <div style={styles.listContent}>
          {filteredItems.map((item, idx) => {
            const isSelected = selectedItem?.narration_id === item.narration_id;
            // Diff indicator color
            let indicatorColor = '#66bb6a'; // green = agrees
            if (item.onlyInA) indicatorColor = '#42a5f5'; // blue
            else if (item.onlyInB) indicatorColor = '#ab47bc'; // purple
            else if (item.hasDifferences) indicatorColor = '#ef5350'; // red

            // Compact pred display
            const predA = item.segments?.reduce((s, seg) => {
              const p = seg.tagA?.predicted_count;
              return p != null ? s + p : s;
            }, 0);
            const predB = item.segments?.reduce((s, seg) => {
              const p = seg.tagB?.predicted_count;
              return p != null ? s + p : s;
            }, 0);
            const hasPredA = item.segments?.some(seg => seg.tagA?.predicted_count != null);
            const hasPredB = item.segments?.some(seg => seg.tagB?.predicted_count != null);

            return (
              <div
                key={item.narration_id || idx}
                style={{
                  ...styles.itemRow,
                  ...(isSelected ? styles.itemRowSelected : {}),
                }}
                onClick={() => handleItemClick(item)}
              >
                <div style={styles.itemName}>
                  <span style={{ ...styles.diffIndicator, backgroundColor: indicatorColor }} />
                  {item.food_name}
                </div>
                <div style={styles.itemMeta}>
                  <span style={{ ...styles.badge, ...getDifficultyStyle(item.difficulty) }}>
                    {item.difficulty}
                  </span>
                  <span style={{ ...styles.badge, backgroundColor: '#e3f2fd', color: '#1565c0' }}>
                    A:{hasPredA ? predA : '\u2014'}
                  </span>
                  <span style={{ ...styles.badge, backgroundColor: '#f3e5f5', color: '#7b1fa2' }}>
                    B:{hasPredB ? predB : '\u2014'}
                  </span>
                  <span style={{ ...styles.badge, backgroundColor: '#eceff1', color: '#455a64' }}>
                    {item.segments?.length || 0} seg
                  </span>
                  {overlayCountByItem[item.narration_id] && (
                    <span style={{ ...styles.badge, backgroundColor: '#ffcdd2', color: '#c62828' }} title="Segments in failure cases">
                      FC:{overlayCountByItem[item.narration_id]}
                    </span>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Right Column */}
      <div style={styles.rightColumn}>
        {/* Video */}
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
              {selectedMergedItem ? selectedMergedItem.food_name : 'Select an item'}
            </span>
            {selectedMergedItem && (
              <span style={{ ...styles.badge, ...getDifficultyStyle(selectedMergedItem.difficulty) }}>
                {selectedMergedItem.difficulty}
              </span>
            )}
          </div>

          {selectedMergedItem ? (
            <div style={styles.detailContent}>
              {/* Segments */}
              {selectedSegments.map((seg, idx) => {
                const cardBorderStyle = seg.differs
                  ? styles.segmentDiffers
                  : (!seg.tagA || !seg.tagB)
                    ? styles.segmentMissing
                    : styles.segmentAgrees;

                return (
                  <div key={idx} style={{
                    ...styles.segmentCard,
                    ...cardBorderStyle,
                    ...(overlaySegmentIds.has(seg.segment_id) ? { borderRightWidth: '4px', borderRightStyle: 'solid', borderRightColor: '#d32f2f' } : {}),
                  }}>
                    {/* Shared header */}
                    <div style={styles.segmentHeader}>
                      <div>
                        <span style={styles.segmentTitle}>Segment {idx + 1}</span>
                        {seg.segment_id && (
                          <span
                            style={{
                              fontSize: '9px',
                              color: '#6a1b9a',
                              backgroundColor: '#f3e5f5',
                              padding: '1px 4px',
                              borderRadius: '3px',
                              marginLeft: '6px',
                              fontFamily: 'monospace',
                              cursor: 'pointer',
                            }}
                            title="Click to copy"
                            onClick={(e) => {
                              e.stopPropagation();
                              navigator.clipboard.writeText(seg.segment_id);
                            }}
                          >
                            {seg.segment_id}
                          </span>
                        )}
                        {seg.video_id && (
                          <span style={{
                            fontSize: '9px',
                            color: '#1565c0',
                            backgroundColor: '#e3f2fd',
                            padding: '1px 4px',
                            borderRadius: '3px',
                            marginLeft: '6px',
                            fontFamily: 'monospace',
                          }}>
                            {seg.video_id}
                          </span>
                        )}
                      </div>
                      <span style={{
                        ...styles.badge,
                        ...(seg.differs
                          ? { backgroundColor: '#ffcdd2', color: '#c62828' }
                          : (!seg.tagA || !seg.tagB)
                            ? { backgroundColor: '#e0e0e0', color: '#616161' }
                            : { backgroundColor: '#c8e6c9', color: '#2e7d32' }),
                        fontSize: '9px',
                      }}>
                        {seg.differs ? 'Differs' : (!seg.tagA || !seg.tagB) ? 'Missing' : 'Agrees'}
                      </span>
                    </div>

                    {/* GT (shared) */}
                    <div style={styles.gtBox}>
                      <div style={{ fontSize: '9px', fontWeight: '600', color: '#666', textTransform: 'uppercase', marginBottom: '2px' }}>
                        Ground Truth
                      </div>
                      <span style={styles.comparisonValue}>
                        {seg.ground_truth_count != null ? seg.ground_truth_count : 'N/A'}
                      </span>
                      {seg.ground_truth_unit && (
                        <span style={styles.comparisonUnit}>{seg.ground_truth_unit}</span>
                      )}
                    </div>

                    {/* Timestamps + seek */}
                    {seg.start_timestamp != null && (
                      <div style={styles.timestampRow}>
                        <span style={{ fontSize: '10px', color: '#666' }}>Time:</span>
                        <span style={{ fontFamily: 'monospace', fontSize: '11px' }}>
                          {formatTimestamp(seg.start_timestamp)} - {formatTimestamp(seg.end_timestamp)}
                        </span>
                        <button
                          style={styles.smallButton}
                          onClick={() => handleSeekTo(seg.start_timestamp, seg.video_id)}
                        >
                          Start
                        </button>
                        <button
                          style={styles.smallButton}
                          onClick={() => handleSeekTo(seg.end_timestamp, seg.video_id)}
                        >
                          End
                        </button>
                      </div>
                    )}

                    {/* Two-column Tag A | Tag B */}
                    <div style={styles.comparisonColumns}>
                      {renderTagColumn(seg.tagA, tagA, styles.tagColumnA)}
                      {renderTagColumn(seg.tagB, tagB, styles.tagColumnB)}
                    </div>

                    {/* Failure Cases Overlay: toggle + notes */}
                    {overlayFailureCasesFile && seg.segment_id && (
                      <div style={{
                        marginTop: '8px',
                        paddingTop: '8px',
                        borderTop: '1px dashed #ccc',
                        display: 'flex',
                        flexDirection: 'column',
                        gap: '4px',
                      }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              onToggleSegmentInFailureCases(
                                participant,
                                selectedMergedItem.narration_id,
                                seg.segment_id
                              );
                            }}
                            style={{
                              padding: '2px 8px',
                              fontSize: '10px',
                              border: 'none',
                              borderRadius: '3px',
                              cursor: 'pointer',
                              backgroundColor: overlaySegmentIds.has(seg.segment_id) ? '#d32f2f' : '#4caf50',
                              color: 'white',
                              fontWeight: '600',
                            }}
                          >
                            {overlaySegmentIds.has(seg.segment_id) ? '\u2212 Remove from FC' : '+ Add to FC'}
                          </button>
                          {overlaySegmentIds.has(seg.segment_id) && (
                            <span style={{ fontSize: '9px', color: '#d32f2f', fontWeight: '600' }}>
                              {overlayCaseLookup[seg.segment_id]?.case_id}
                            </span>
                          )}
                        </div>
                        {overlaySegmentIds.has(seg.segment_id) && (
                          <input
                            type="text"
                            placeholder="Notes..."
                            value={overlayCaseLookup[seg.segment_id]?.notes || ''}
                            onChange={(e) => onUpdateFailureCaseNotes(seg.segment_id, e.target.value)}
                            onClick={(e) => e.stopPropagation()}
                            style={{
                              padding: '4px 6px',
                              fontSize: '10px',
                              border: '1px solid #ddd',
                              borderRadius: '3px',
                              width: '100%',
                              boxSizing: 'border-box',
                            }}
                          />
                        )}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          ) : (
            <div style={styles.noSelection}>
              Select an item to compare Tag A vs Tag B predictions
            </div>
          )}
        </div>
      </div>
    </div>
    </div>
  );
});

ComparisonView.displayName = 'ComparisonView';

export default ComparisonView;

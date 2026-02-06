import React, { useState, useRef, useImperativeHandle, forwardRef, useEffect } from 'react';
import { formatTimestamp } from '../utils/narrationParser';
import FrameGallery from './FrameGallery';

const styles = {
  container: {
    display: 'grid',
    gridTemplateColumns: '380px 1fr',
    gap: '15px',
    height: 'calc(100vh - 90px)',
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
  tagSelector: {
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
  remainingBadge: { backgroundColor: '#e0f2f1', color: '#00695c' },
  segmentsBadge: { backgroundColor: '#eceff1', color: '#455a64' },
  categoryDiscrete: { backgroundColor: '#e8f5e9', color: '#2e7d32' },
  categoryContinuous: { backgroundColor: '#fff3e0', color: '#e65100' },
  errorPositive: { backgroundColor: '#ffcdd2', color: '#c62828' },
  errorNegative: { backgroundColor: '#fff9c4', color: '#f57f17' },
  // Blind mode styles
  itemMismatch: { backgroundColor: '#fff3e0', color: '#e65100' },
  itemMatch: { backgroundColor: '#e8f5e9', color: '#2e7d32' },
  detectedItemBox: {
    padding: '8px',
    borderRadius: '4px',
    marginBottom: '8px',
    fontSize: '11px',
  },
  similarityBar: {
    height: '4px',
    borderRadius: '2px',
    backgroundColor: '#e0e0e0',
    marginTop: '4px',
    overflow: 'hidden',
  },
  similarityFill: {
    height: '100%',
    borderRadius: '2px',
    transition: 'width 0.3s',
  },
  warningBadge: {
    backgroundColor: '#fff3e0',
    color: '#e65100',
    fontSize: '8px',
    padding: '1px 4px',
    borderRadius: '3px',
    fontWeight: '500',
  },
  // Right panel
  rightColumn: {
    display: 'grid',
    gridTemplateRows: '1fr 1fr',
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
    display: 'flex',
    flexDirection: 'column',
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
  viewModeToggle: {
    display: 'flex',
    gap: '0',
    marginLeft: '10px',
  },
  viewModeButton: {
    padding: '4px 10px',
    fontSize: '10px',
    border: '1px solid #666',
    backgroundColor: '#333',
    color: '#aaa',
    cursor: 'pointer',
    transition: 'all 0.15s',
  },
  viewModeButtonFirst: {
    borderRadius: '4px 0 0 4px',
  },
  viewModeButtonLast: {
    borderRadius: '0 4px 4px 0',
    borderLeft: 'none',
  },
  viewModeButtonActive: {
    backgroundColor: '#4CAF50',
    borderColor: '#4CAF50',
    color: 'white',
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
  remainingBox: {
    backgroundColor: '#e0f2f1',
    border: '1px solid #80cbc4',
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
  // Keyframe selection styles
  keyframeSection: {
    border: '1px solid #bbdefb',
    borderRadius: '6px',
    padding: '10px',
    marginBottom: '8px',
    backgroundColor: '#e3f2fd',
  },
  keyframeSectionLabel: {
    fontSize: '10px',
    fontWeight: '600',
    color: '#1565c0',
    marginBottom: '8px',
    textTransform: 'uppercase',
  },
  keyframeRow: {
    display: 'flex',
    gap: '10px',
    marginBottom: '6px',
  },
  keyframeCard: {
    flex: 1,
    padding: '8px',
    borderRadius: '4px',
    fontSize: '11px',
    border: '1px solid #e0e0e0',
  },
  keyframeCardSource: {
    backgroundColor: '#fff3e0',
    borderColor: '#ffcc80',
  },
  keyframeCardDest: {
    backgroundColor: '#e8f5e9',
    borderColor: '#a5d6a7',
  },
  keyframeLabel: {
    fontSize: '9px',
    fontWeight: '600',
    color: '#666',
    marginBottom: '4px',
    textTransform: 'uppercase',
  },
  keyframeTimestamp: {
    fontFamily: 'monospace',
    fontSize: '13px',
    fontWeight: '600',
    marginBottom: '4px',
  },
  keyframeDescription: {
    fontSize: '10px',
    color: '#555',
    lineHeight: '1.3',
    marginBottom: '4px',
  },
  keyframeVisibility: {
    fontSize: '8px',
    padding: '1px 4px',
    borderRadius: '3px',
    fontWeight: '500',
    display: 'inline-block',
  },
  visibilityClearView: { backgroundColor: '#c8e6c9', color: '#2e7d32' },
  visibilityPartial: { backgroundColor: '#fff9c4', color: '#f57f17' },
  visibilityOccluded: { backgroundColor: '#ffcdd2', color: '#c62828' },
  keyframeSeekButton: {
    padding: '4px 10px',
    fontSize: '10px',
    border: 'none',
    borderRadius: '3px',
    cursor: 'pointer',
    color: 'white',
    fontWeight: '600',
    marginTop: '4px',
  },
  seekButtonSource: { backgroundColor: '#ef6c00' },
  seekButtonDest: { backgroundColor: '#2e7d32' },
  countingStrategy: {
    fontSize: '10px',
    color: '#555',
    lineHeight: '1.4',
    backgroundColor: '#e8eaf6',
    padding: '8px',
    borderRadius: '4px',
    marginTop: '6px',
    borderLeft: '3px solid #3f51b5',
  },
  // Evidence frame styles (unified for keyframe + multipath)
  evidenceFramesRow: {
    display: 'flex',
    gap: '8px',
    flexWrap: 'wrap',
    marginBottom: '6px',
  },
  evidenceFrameCard: {
    flex: '1 1 140px',
    maxWidth: '200px',
    padding: '8px',
    borderRadius: '4px',
    fontSize: '11px',
    border: '1px solid #e0e0e0',
  },
  evidenceRoleLabel: {
    fontSize: '8px',
    fontWeight: '700',
    textTransform: 'uppercase',
    marginBottom: '4px',
    padding: '1px 4px',
    borderRadius: '3px',
    display: 'inline-block',
  },
  // Role colors
  roleSource: { backgroundColor: '#fff3e0', borderColor: '#ffcc80' },
  roleSourceBefore: { backgroundColor: '#fff3e0', borderColor: '#ffcc80' },
  roleSourceAfter: { backgroundColor: '#ffe0b2', borderColor: '#ffb74d' },
  roleDest: { backgroundColor: '#e8f5e9', borderColor: '#a5d6a7' },
  roleDestBefore: { backgroundColor: '#e8f5e9', borderColor: '#a5d6a7' },
  roleDestAfter: { backgroundColor: '#c8e6c9', borderColor: '#81c784' },
  roleTransfer: { backgroundColor: '#e3f2fd', borderColor: '#90caf9' },
  // Role label colors
  roleLabelSource: { backgroundColor: '#ef6c00', color: 'white' },
  roleLabelSourceBefore: { backgroundColor: '#ef6c00', color: 'white' },
  roleLabelSourceAfter: { backgroundColor: '#e65100', color: 'white' },
  roleLabelDest: { backgroundColor: '#2e7d32', color: 'white' },
  roleLabelDestBefore: { backgroundColor: '#2e7d32', color: 'white' },
  roleLabelDestAfter: { backgroundColor: '#1b5e20', color: 'white' },
  roleLabelTransfer: { backgroundColor: '#1565c0', color: 'white' },
  // Path status badge
  pathStatusBadge: {
    fontSize: '8px',
    padding: '2px 5px',
    borderRadius: '3px',
    fontWeight: '600',
    display: 'inline-block',
    marginRight: '4px',
  },
  pathValid: { backgroundColor: '#c8e6c9', color: '#2e7d32' },
  pathInvalid: { backgroundColor: '#ffcdd2', color: '#c62828' },
  // Synthesis box
  synthesisBox: {
    padding: '8px',
    borderRadius: '4px',
    backgroundColor: '#f3e5f5',
    border: '1px solid #ce93d8',
    marginTop: '6px',
    fontSize: '10px',
    lineHeight: '1.4',
  },
  pathsSummary: {
    display: 'flex',
    gap: '6px',
    flexWrap: 'wrap',
    marginTop: '6px',
    marginBottom: '4px',
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
  availableTags = [],
  selectedTag,
  onTagChange,
  onRefresh,
  onLoadVideoAtTime,
  videoUrl,
  videoId,
  currentTime,
  onTimeUpdate,
  hands23Data,
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
  // When true, hides VLM-specific UI (GT/Pred boxes, match badges) and shows case data
  failureCasesMode = false,
}, ref) => {
  const [selectedItem, setSelectedItem] = useState(null);
  const [filterMatch, setFilterMatch] = useState('all'); // all, exact, close, wrong, uncountable
  const [filterTag, setFilterTag] = useState(null);
  const [viewMode, setViewMode] = useState('video'); // 'video' | 'frames'
  const [overlayExpanded, setOverlayExpanded] = useState(false);
  const [overlayFileSelection, setOverlayFileSelection] = useState('');

  const videoRef = useRef(null);
  const [duration, setDuration] = useState(0);

  // Reset selected item and filter when tag/data changes
  useEffect(() => {
    setSelectedItem(null);
    setFilterMatch('all');
    setFilterTag(null);
  }, [vlmData, selectedTag]);

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
    itemMismatch: 0,
    categoryMismatch: 0,
  };

  items.forEach(item => {
    let hasItemMismatch = false;
    let hasCatMismatch = false;
    item.segments?.forEach(seg => {
      const match = seg.match;
      if (!match) return; // Skip segments with errors/no VLM response
      if (match === 'exact') stats.exact++;
      else if (match === 'close') stats.close++;
      else if (match === 'wrong') stats.wrong++;
      else if (match === 'item_mismatch') stats.wrong++; // Count item_mismatch as wrong
      else stats.uncountable++;

      // Blind mode stats
      if (seg.item_match === false || match === 'item_mismatch') hasItemMismatch = true;
      if (seg.quantity_category === 'continuous' && seg.ground_truth_count !== null) hasCatMismatch = true;
    });
    if (hasItemMismatch) stats.itemMismatch++;
    if (hasCatMismatch) stats.categoryMismatch++;
  });

  // Collect unique tags across all segments (for tag filter in all modes)
  const allTags = (() => {
    const tagCounts = {};
    items.forEach(item => {
      item.segments?.forEach(seg => {
        (seg.tags || []).forEach(tag => {
          tagCounts[tag] = (tagCounts[tag] || 0) + 1;
        });
      });
    });
    return Object.entries(tagCounts)
      .sort((a, b) => b[1] - a[1])
      .map(([tag, count]) => ({ tag, count }));
  })();

  // Build overlay lookup: segment_id -> case object
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

  // Count overlay hits per item (for badge in list)
  const overlayCountByItem = {};
  if (overlaySegmentIds.size > 0) {
    items.forEach(item => {
      const count = (item.segments || []).filter(s => overlaySegmentIds.has(s.segment_id)).length;
      if (count > 0) overlayCountByItem[item.narration_id] = count;
    });
  }

  // Filter items based on match type and tag
  const filteredItems = (() => {
    let result = filterMatch === 'all'
      ? items
      : items.filter(item => {
          // Blind mode filters
          if (filterMatch === 'itemMismatch') {
            return item.segments?.some(seg => seg.item_match === false);
          }
          if (filterMatch === 'categoryMismatch') {
            return item.segments?.some(seg =>
              seg.quantity_category === 'continuous' && seg.ground_truth_count !== null
            );
          }
          // Standard count match filters
          return item.segments?.some(seg => {
            const match = seg.match;
            if (filterMatch === 'uncountable') {
              return ['pred_uncountable', 'gt_uncountable', 'both_uncountable', 'continuous_estimate'].includes(match);
            }
            if (filterMatch === 'wrong') {
              return match === 'wrong' || match === 'item_mismatch';
            }
            return match === filterMatch;
          });
        });
    // Tag filter (failure cases mode)
    if (filterTag) {
      result = result.filter(item =>
        item.segments?.some(seg => (seg.tags || []).includes(filterTag))
      );
    }
    return result;
  })();

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
    if (match === 'item_mismatch') return styles.itemMismatch;
    return styles.matchUncountable;
  };

  const getMatchLabel = (match) => {
    if (match === 'exact') return 'Exact Match';
    if (match === 'close') return 'Close';
    if (match === 'wrong') return 'Wrong';
    if (match === 'item_mismatch') return 'Item Mismatch';
    if (match === 'pred_uncountable') return 'Pred N/A';
    if (match === 'gt_uncountable') return 'GT N/A';
    if (match === 'both_uncountable') return 'Both N/A';
    if (match === 'continuous_estimate') return 'Continuous';
    return match;
  };

  const formatCount = (count, unit) => {
    if (count === null || count === undefined) return 'N/A';
    return `${count} ${unit || ''}`;
  };

  const segments = selectedItem?.segments || [];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: 'calc(100vh - 90px)', minHeight: '600px' }}>
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
      {/* Left Panel: Item List */}
      <div style={styles.listPanel}>
        <div style={styles.listHeader}>
          <div style={styles.headerRow}>
            <span>VLM Results ({filteredItems.length}/{items.length})</span>
            <div style={{ display: 'flex', gap: '4px', alignItems: 'center' }}>
              {availableTags.length > 0 && (
                <select
                  style={styles.tagSelector}
                  value={selectedTag || ''}
                  onChange={(e) => onTagChange && onTagChange(e.target.value)}
                >
                  {availableTags.map(tag => (
                    <option key={tag} value={tag}>
                      {tag}
                    </option>
                  ))}
                </select>
              )}
              {onRefresh && (
                <button
                  onClick={onRefresh}
                  style={{
                    padding: '2px 6px',
                    fontSize: '12px',
                    cursor: 'pointer',
                    background: '#444',
                    color: '#fff',
                    border: '1px solid #666',
                    borderRadius: '4px',
                  }}
                  title="Refresh data from file"
                >
                  ↻
                </button>
              )}
            </div>
          </div>
          <div style={styles.statsRow}>
            {!failureCasesMode && <>
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
            </span></>}
            {allTags.length > 0 && allTags.map(({ tag, count }) => (
              <span
                key={tag}
                style={{
                  ...styles.badge,
                  backgroundColor: filterTag === tag ? '#1565c0' : '#e3f2fd',
                  color: filterTag === tag ? 'white' : '#1565c0',
                  cursor: 'pointer',
                  outline: filterTag === tag ? '2px solid #0d47a1' : 'none',
                }}
                onClick={() => setFilterTag(filterTag === tag ? null : tag)}
                title={`${count} items with tag "${tag}"`}
              >
                {tag} ({count})
              </span>
            ))}
            {stats.itemMismatch > 0 && (
              <span
                style={{
                  ...styles.badge,
                  ...styles.itemMismatch,
                  cursor: 'pointer',
                  ...(filterMatch === 'itemMismatch' ? { outline: '2px solid #e65100' } : {}),
                }}
                onClick={() => setFilterMatch(filterMatch === 'itemMismatch' ? 'all' : 'itemMismatch')}
                title="Items with detected item mismatch"
              >
                ⚠{stats.itemMismatch}
              </span>
            )}
            {stats.categoryMismatch > 0 && (
              <span
                style={{
                  ...styles.badge,
                  backgroundColor: '#fff8e1',
                  color: '#f57f17',
                  cursor: 'pointer',
                  ...(filterMatch === 'categoryMismatch' ? { outline: '2px solid #f57f17' } : {}),
                }}
                onClick={() => setFilterMatch(filterMatch === 'categoryMismatch' ? 'all' : 'categoryMismatch')}
                title="Items with category mismatch (continuous vs discrete)"
              >
                ⚠{stats.categoryMismatch}
              </span>
            )}
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

            // Blind mode: check for item mismatches
            const hasItemMismatch = item.segments?.some(s => s.item_match === false);
            const hasCategoryMismatch = item.segments?.some(s =>
              s.quantity_category === 'continuous' && s.ground_truth_count !== null
            );

            // For GT badge, prefer segment-level ground_truth_count (sum across segments)
            // This is more accurate for failure cases where each segment has its own GT
            const segmentGtSum = item.segments?.reduce((sum, seg) => {
              const gt = seg.ground_truth_count;
              return gt !== null && gt !== undefined ? sum + gt : sum;
            }, 0);
            const hasSegmentGt = item.segments?.some(seg => seg.ground_truth_count !== null && seg.ground_truth_count !== undefined);
            const displayGt = hasSegmentGt ? segmentGtSum : item.total_ground_truth;

            // For Pred badge, sum segment predictions
            const segmentPredSum = item.segments?.reduce((sum, seg) => {
              const pred = seg.predicted_count;
              return pred !== null && pred !== undefined ? sum + pred : sum;
            }, 0);
            const hasSegmentPred = item.segments?.some(seg => seg.predicted_count !== null && seg.predicted_count !== undefined);
            const displayPred = hasSegmentPred ? segmentPredSum : item.total_predicted;

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
                  {item.difficulty && (
                  <span style={{ ...styles.badge, ...getDifficultyStyle(item.difficulty) }}>
                    {item.difficulty}
                  </span>
                  )}
                  {!failureCasesMode && (
                  <span style={{ ...styles.badge, ...getMatchStyle(overallMatch) }}>
                    {overallMatch}
                  </span>
                  )}
                  {!failureCasesMode && displayGt !== null && displayGt !== undefined ? (
                    <span style={{ ...styles.badge, ...styles.gtBadge }}>
                      GT: {displayGt}
                    </span>
                  ) : !failureCasesMode && item.recipe_amount ? (
                    <span style={{ ...styles.badge, ...styles.recipeBadge }}>
                      Recipe: {item.recipe_amount.amount}{item.recipe_amount.unit}
                    </span>
                  ) : null}
                  {!failureCasesMode && displayPred !== null && displayPred !== undefined && displayPred > 0 && (
                    <span style={{ ...styles.badge, ...styles.predBadge }}>
                      Pred: {displayPred}
                    </span>
                  )}
                  <span style={{ ...styles.badge, ...styles.segmentsBadge }}>
                    {numSegments} seg
                  </span>
                  {hasItemMismatch && (
                    <span style={{ ...styles.badge, ...styles.itemMismatch }} title="Item detection mismatch">
                      ⚠ item
                    </span>
                  )}
                  {hasCategoryMismatch && (
                    <span style={{ ...styles.badge, backgroundColor: '#fff8e1', color: '#f57f17' }} title="Category mismatch (continuous vs discrete)">
                      ⚠ cat
                    </span>
                  )}
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

      {/* Right Column: Video + Details */}
      <div style={styles.rightColumn}>
        {/* Video Player or Frame Gallery */}
        <div style={styles.videoContainer}>
          {/* Video/Frames Mode Toggle - shown when hands23 data is available */}
          {hands23Data && (
            <div style={{
              display: 'flex',
              justifyContent: 'flex-end',
              padding: '6px 10px',
              gap: '8px',
              alignItems: 'center',
              backgroundColor: '#f5f5f5',
              borderBottom: '1px solid #e0e0e0',
            }}>
              <span style={{ fontSize: '11px', color: '#666' }}>
                HOI Frames: {hands23Data.videos?.length || 0} videos
              </span>
              <div style={styles.viewModeToggle}>
                <button
                  style={{
                    ...styles.viewModeButton,
                    ...styles.viewModeButtonFirst,
                    ...(viewMode === 'video' ? styles.viewModeButtonActive : {}),
                  }}
                  onClick={() => setViewMode('video')}
                >
                  Video
                </button>
                <button
                  style={{
                    ...styles.viewModeButton,
                    ...styles.viewModeButtonLast,
                    ...(viewMode === 'frames' ? styles.viewModeButtonActive : {}),
                  }}
                  onClick={() => setViewMode('frames')}
                >
                  Frames
                </button>
              </div>
            </div>
          )}
          {viewMode === 'frames' && hands23Data ? (
            <FrameGallery
              hands23Data={hands23Data}
              videoId={videoId}
              currentTime={currentTime}
              onTimeChange={onTimeUpdate}
              participant={participant}
              onSwitchToVideo={() => setViewMode('video')}
            />
          ) : (
            <>
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
            </>
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
              {/* Segments */}
              <div style={styles.fieldGroup}>
                <div style={styles.fieldLabel}>Segments</div>
                {segments.map((segment, idx) => {
                  // Check if segment has error (no VLM response)
                  const hasError = typeof segment.error === 'string';
                  const segmentMatch = segment.match;

                  return (
                  <div key={idx} style={{
                    ...styles.segmentCard,
                    ...(hasError ? { backgroundColor: '#fff3e0', borderColor: '#ffcc80' } : {}),
                    ...(overlaySegmentIds.has(segment.segment_id) ? { borderLeft: '4px solid #d32f2f' } : {}),
                  }}>
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
                        {segment.segment_id && (
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
                            title="Click to copy segment_id"
                            onClick={(e) => {
                              e.stopPropagation();
                              navigator.clipboard.writeText(segment.segment_id);
                            }}
                          >
                            {segment.segment_id}
                          </span>
                        )}
                      </div>
                      {hasError ? (
                        <span style={{ ...styles.badge, backgroundColor: '#ffcc80', color: '#e65100', fontSize: '9px' }}>
                          Error
                        </span>
                      ) : !failureCasesMode ? (
                        <span style={{ ...styles.badge, ...getMatchStyle(segmentMatch), fontSize: '9px' }}>
                          {getMatchLabel(segmentMatch)}
                          {typeof segment.error === 'number' && segment.error !== 0 && (
                            <span style={{ marginLeft: '4px' }}>
                              ({segment.error > 0 ? '+' : ''}{segment.error})
                            </span>
                          )}
                        </span>
                      ) : segment.case_id ? (
                        <span style={{ ...styles.badge, backgroundColor: '#e0e0e0', color: '#333', fontSize: '9px' }}>
                          {segment.case_id}
                        </span>
                      ) : null}
                    </div>

                    {hasError ? (
                      <div style={{ padding: '8px', color: '#e65100', fontSize: '11px' }}>
                        {segment.error}
                      </div>
                    ) : (
                    <>

                    {/* Blind Mode: Detected Item vs Expected Item */}
                    {segment.detected_item_name && (
                      <div style={{
                        ...styles.detectedItemBox,
                        ...(segment.item_match === false ? {
                          backgroundColor: '#fff3e0',
                          border: '1px solid #ffcc80',
                        } : segment.item_match === true ? {
                          backgroundColor: '#e8f5e9',
                          border: '1px solid #a5d6a7',
                        } : {
                          backgroundColor: '#f5f5f5',
                          border: '1px solid #e0e0e0',
                        }),
                      }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '4px' }}>
                          <div style={styles.comparisonLabel}>Detected Item (Blind Mode)</div>
                          {segment.item_match === false && (
                            <span style={styles.warningBadge}>MISMATCH</span>
                          )}
                          {segment.item_match === true && (
                            <span style={{ ...styles.badge, ...styles.itemMatch, fontSize: '8px' }}>MATCH</span>
                          )}
                        </div>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                          <div>
                            <span style={{ fontWeight: '600', color: segment.item_match === false ? '#e65100' : '#2e7d32' }}>
                              {segment.detected_item_name}
                            </span>
                            {segment.detected_item_name !== selectedItem.food_name && (
                              <span style={{ fontSize: '10px', color: '#666', marginLeft: '6px' }}>
                                (expected: {selectedItem.food_name})
                              </span>
                            )}
                          </div>
                          {segment.item_similarity !== undefined && (
                            <span style={{
                              fontSize: '10px',
                              fontFamily: 'monospace',
                              color: segment.item_similarity >= 0.8 ? '#2e7d32' : segment.item_similarity >= 0.5 ? '#f57f17' : '#c62828',
                            }}>
                              {(segment.item_similarity * 100).toFixed(0)}% sim
                            </span>
                          )}
                        </div>
                        {segment.item_similarity !== undefined && (
                          <div style={styles.similarityBar}>
                            <div style={{
                              ...styles.similarityFill,
                              width: `${segment.item_similarity * 100}%`,
                              backgroundColor: segment.item_similarity >= 0.8 ? '#4caf50' : segment.item_similarity >= 0.5 ? '#ff9800' : '#f44336',
                            }} />
                          </div>
                        )}
                      </div>
                    )}

                    {/* Category Mismatch Warning: continuous prediction for discrete GT */}
                    {segment.quantity_category === 'continuous' && segment.ground_truth_count !== null && (
                      <div style={{
                        ...styles.detectedItemBox,
                        backgroundColor: '#fff8e1',
                        border: '1px solid #ffe082',
                        marginBottom: '8px',
                      }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                          <span style={styles.warningBadge}>CATEGORY MISMATCH</span>
                          <span style={{ fontSize: '10px', color: '#f57f17' }}>
                            VLM predicted continuous, but GT is discrete ({segment.ground_truth_count} {segment.ground_truth_unit || ''})
                          </span>
                        </div>
                      </div>
                    )}

                    {/* Ground Truth vs Prediction (hidden in FC mode) */}
                    {!failureCasesMode && (
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
                      <div style={{
                        ...styles.comparisonBox,
                        ...styles.predBox,
                      }}>
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
                    )}

                    {/* Remaining Amount (hidden in FC mode) */}
                    {!failureCasesMode && (segment.remaining_count != null || segment.remaining_description || segment.remaining_fraction != null) && (
                      <div style={{
                        ...styles.comparisonBox,
                        ...styles.remainingBox,
                        marginBottom: '8px',
                      }}>
                        <div style={styles.comparisonLabel}>Remaining</div>
                        {segment.remaining_count != null && (
                          <div>
                            <span style={styles.comparisonValue}>{segment.remaining_count}</span>
                            <span style={styles.comparisonUnit}> left</span>
                          </div>
                        )}
                        {segment.remaining_description && (
                          <div style={styles.comparisonAmount}>
                            {segment.remaining_description}
                          </div>
                        )}
                        {segment.remaining_fraction != null && (
                          <div>
                            <span style={styles.comparisonValue}>
                              {Math.round(segment.remaining_fraction * 100)}%
                            </span>
                            <span style={styles.comparisonUnit}> remaining</span>
                          </div>
                        )}
                      </div>
                    )}

                    {/* Failure Cases: path estimates & conflict info */}
                    {failureCasesMode && segment.path_estimates && (
                      <div style={{
                        padding: '8px',
                        borderRadius: '4px',
                        backgroundColor: '#fff3e0',
                        border: '1px solid #ffcc80',
                        marginBottom: '8px',
                        fontSize: '11px',
                      }}>
                        <div style={styles.comparisonLabel}>Path Estimates</div>
                        <div style={{ display: 'flex', gap: '10px', flexWrap: 'wrap', marginBottom: '4px' }}>
                          {Object.entries(segment.path_estimates).map(([path, value]) => {
                            const isTrusted = segment.trusted_paths?.includes(path);
                            return (
                              <span key={path} style={{
                                padding: '2px 6px',
                                borderRadius: '3px',
                                fontWeight: '600',
                                fontSize: '11px',
                                backgroundColor: isTrusted ? '#c8e6c9' : '#ffcdd2',
                                color: isTrusted ? '#2e7d32' : '#c62828',
                                border: isTrusted ? '1px solid #81c784' : '1px solid #ef9a9a',
                              }}>
                                {path}: {value}
                                {isTrusted && ' \u2713'}
                              </span>
                            );
                          })}
                        </div>
                        {segment.trusted_paths?.length > 0 && (
                          <div style={{ fontSize: '9px', color: '#666' }}>
                            VLM chose: <strong>{segment.trusted_paths.join(', ')}</strong>
                            {segment.ignored_paths?.length > 0 && (
                              <> | ignored: {segment.ignored_paths.join(', ')}</>
                            )}
                          </div>
                        )}
                      </div>
                    )}

                    {/* Failure Cases: notes */}
                    {failureCasesMode && segment.notes && (
                      <div style={{
                        padding: '6px 8px',
                        borderRadius: '4px',
                        backgroundColor: '#f5f5f5',
                        border: '1px solid #e0e0e0',
                        marginBottom: '6px',
                        fontSize: '10px',
                        color: '#555',
                      }}>
                        <strong>Notes:</strong> {segment.notes}
                      </div>
                    )}

                    {/* Tags (from timeline_annotated or failure cases) */}
                    {segment.tags?.length > 0 && (
                      <div style={{ display: 'flex', gap: '4px', flexWrap: 'wrap', marginBottom: '6px' }}>
                        {segment.tags.map((tag, tIdx) => (
                          <span key={tIdx} style={{
                            ...styles.badge,
                            backgroundColor: filterTag === tag ? '#1565c0' : '#e3f2fd',
                            color: filterTag === tag ? 'white' : '#1565c0',
                            fontSize: '9px',
                            cursor: 'pointer',
                          }}
                            onClick={(e) => {
                              e.stopPropagation();
                              setFilterTag(filterTag === tag ? null : tag);
                            }}
                            title={`Filter by "${tag}"`}
                          >
                            {tag}
                          </span>
                        ))}
                      </div>
                    )}

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

                    {/* Evidence Frames (unified: keyframe + multipath) */}
                    {(segment.evidence_frames || (segment.source_frame && segment.destination_frame)) && (() => {
                      const getRoleCardStyle = (role) => {
                        if (role === 'source') return styles.roleSource;
                        if (role === 'source_before') return styles.roleSourceBefore;
                        if (role === 'source_after') return styles.roleSourceAfter;
                        if (role === 'destination') return styles.roleDest;
                        if (role === 'dest_before') return styles.roleDestBefore;
                        if (role === 'dest_after') return styles.roleDestAfter;
                        if (role === 'transfer') return styles.roleTransfer;
                        return {};
                      };
                      const getRoleLabelStyle = (role) => {
                        if (role?.startsWith('source')) return styles.roleLabelSource;
                        if (role?.startsWith('dest')) return styles.roleLabelDest;
                        if (role === 'transfer') return styles.roleLabelTransfer;
                        return { backgroundColor: '#666', color: 'white' };
                      };
                      const getSeekButtonStyle = (role) => {
                        if (role?.startsWith('source')) return styles.seekButtonSource;
                        if (role?.startsWith('dest')) return styles.seekButtonDest;
                        return { backgroundColor: '#1565c0' };
                      };

                      // Use evidence_frames if available, otherwise build from old format
                      const frames = segment.evidence_frames || [
                        {
                          role: 'source',
                          timestamp_raw: segment.source_frame?.timestamp_raw,
                          absolute_timestamp: segment.source_frame?.absolute_timestamp,
                          description: segment.source_frame?.description,
                          visibility_status: segment.source_frame?.visibility_status,
                        },
                        {
                          role: 'destination',
                          timestamp_raw: segment.destination_frame?.timestamp_raw,
                          absolute_timestamp: segment.destination_frame?.absolute_timestamp,
                          description: segment.destination_frame?.description,
                          visibility_status: segment.destination_frame?.visibility_status,
                        },
                      ];

                      return (
                      <div style={styles.keyframeSection}>
                        <div style={styles.keyframeSectionLabel}>
                          {segment.paths ? 'Multi-Path Evidence' : 'VLM Keyframe Selection'}
                          <span style={{ fontSize: '9px', fontWeight: '400', marginLeft: '6px', color: '#666' }}>
                            ({frames.length} frames)
                          </span>
                        </div>

                        <div style={styles.evidenceFramesRow}>
                          {frames.map((frame, fIdx) => (
                            <div key={fIdx} style={{
                              ...styles.evidenceFrameCard,
                              ...getRoleCardStyle(frame.role),
                            }}>
                              <span style={{ ...styles.evidenceRoleLabel, ...getRoleLabelStyle(frame.role) }}>
                                {(frame.role || 'unknown').replace(/_/g, ' ')}
                              </span>
                              <div style={styles.keyframeTimestamp}>
                                {frame.absolute_timestamp != null
                                  ? formatTimestamp(frame.absolute_timestamp)
                                  : frame.timestamp_raw || '?'}
                              </div>
                              {frame.description && (
                                <div style={styles.keyframeDescription}>{frame.description}</div>
                              )}
                              {frame.visibility_status && (
                                <span style={{
                                  ...styles.keyframeVisibility,
                                  ...(frame.visibility_status === 'clear_view'
                                    ? styles.visibilityClearView
                                    : frame.visibility_status === 'partially_in_frame'
                                      ? styles.visibilityPartial
                                      : styles.visibilityOccluded),
                                }}>
                                  {frame.visibility_status.replace(/_/g, ' ')}
                                </span>
                              )}
                              {frame.container_description && (
                                <div style={{ fontSize: '9px', color: '#444', marginTop: '2px', fontStyle: 'italic' }}>
                                  {frame.container_description}
                                </div>
                              )}
                              {frame.visible_count != null && (
                                <div style={{ fontSize: '10px', marginTop: '2px', fontWeight: '600' }}>
                                  Count: {frame.visible_count}
                                </div>
                              )}
                              {frame.absolute_timestamp != null && (
                                <div>
                                  <button
                                    style={{ ...styles.keyframeSeekButton, ...getSeekButtonStyle(frame.role) }}
                                    onClick={() => handleSeekTo(frame.absolute_timestamp, segment.video_id)}
                                  >
                                    Seek
                                  </button>
                                </div>
                              )}
                            </div>
                          ))}
                        </div>

                        {/* Path status badges (multipath only) */}
                        {segment.paths && (
                          <div style={styles.pathsSummary}>
                            {['source', 'destination', 'transfer'].map(pathKey => {
                              const path = segment.paths[pathKey];
                              if (!path) return null;
                              const isValid = path.status === 'VALID';
                              return (
                                <span key={pathKey} style={{
                                  ...styles.pathStatusBadge,
                                  ...(isValid ? styles.pathValid : styles.pathInvalid),
                                }}>
                                  {pathKey}: {path.status}
                                  {path.confidence && ` (${path.confidence})`}
                                  {path.observed_delta != null && ` d=${path.observed_delta}`}
                                  {path.total_transfer_count != null && ` n=${path.total_transfer_count}`}
                                </span>
                              );
                            })}
                          </div>
                        )}

                        {/* Final synthesis (multipath only) */}
                        {segment.final_synthesis && (
                          <div style={styles.synthesisBox}>
                            <strong>Synthesis:</strong>{' '}
                            Best path: <strong>{segment.final_synthesis.best_path_selected}</strong>
                            {segment.final_synthesis.final_count_estimate != null && (
                              <> | Count: <strong>{segment.final_synthesis.final_count_estimate}</strong></>
                            )}
                            {segment.final_synthesis.reasoning && (
                              <div style={{ marginTop: '4px', color: '#666' }}>
                                {segment.final_synthesis.reasoning}
                              </div>
                            )}
                          </div>
                        )}

                        {/* Counting Strategy (keyframe only) */}
                        {segment.counting_strategy && (
                          <div style={styles.countingStrategy}>
                            <strong>Strategy:</strong> {segment.counting_strategy}
                          </div>
                        )}
                      </div>
                      );
                    })()}

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

                    {/* Failure Cases Overlay: toggle + notes */}
                    {overlayFailureCasesFile && segment.segment_id && (
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
                                selectedItem.participant || participant,
                                selectedItem.narration_id,
                                segment.segment_id
                              );
                            }}
                            style={{
                              padding: '2px 8px',
                              fontSize: '10px',
                              border: 'none',
                              borderRadius: '3px',
                              cursor: 'pointer',
                              backgroundColor: overlaySegmentIds.has(segment.segment_id) ? '#d32f2f' : '#4caf50',
                              color: 'white',
                              fontWeight: '600',
                            }}
                          >
                            {overlaySegmentIds.has(segment.segment_id) ? '\u2212 Remove from FC' : '+ Add to FC'}
                          </button>
                          {overlaySegmentIds.has(segment.segment_id) && (
                            <span style={{ fontSize: '9px', color: '#d32f2f', fontWeight: '600' }}>
                              {overlayCaseLookup[segment.segment_id]?.case_id}
                            </span>
                          )}
                        </div>
                        {overlaySegmentIds.has(segment.segment_id) && (
                          <input
                            type="text"
                            placeholder="Notes..."
                            value={overlayCaseLookup[segment.segment_id]?.notes || ''}
                            onChange={(e) => onUpdateFailureCaseNotes(segment.segment_id, e.target.value)}
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
                    </>
                    )}
                  </div>
                  );
                })}
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
    </div>
  );
});

VLMResultsView.displayName = 'VLMResultsView';

export default VLMResultsView;

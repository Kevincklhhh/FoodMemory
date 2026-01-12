import React, { useState, useRef, useCallback, useMemo } from 'react';
import DataLoader from './components/DataLoader';
import VideoPlayer from './components/VideoPlayer';
import EventList from './components/EventList';
import GraphView from './components/GraphView';
import DebugFooter from './components/DebugFooter';
import LineageView from './components/LineageView';
import ContainerHistoryView from './components/ContainerHistoryView';
import ThreeBlockView from './components/ThreeBlockView';
import { buildLineageGraph, traceAncestry, traceContainerHistory } from './utils/lineageGraph';

const styles = {
  app: {
    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
    minHeight: '100vh',
    backgroundColor: '#f0f0f0',
  },
  header: {
    backgroundColor: '#1976D2',
    color: 'white',
    padding: '15px 20px',
    fontSize: '20px',
    fontWeight: 'bold',
    boxShadow: '0 2px 4px rgba(0,0,0,0.2)',
  },
  content: {
    padding: '20px',
    maxWidth: '1800px',
    margin: '0 auto',
  },
  mainLayout: {
    display: 'grid',
    gridTemplateColumns: '1fr 300px 1fr',
    gap: '20px',
    minHeight: 'calc(100vh - 200px)',
  },
  leftPanel: {
    display: 'flex',
    flexDirection: 'column',
    gap: '20px',
  },
  centerPanel: {
    height: '350px',
    maxHeight: '400px',
    minHeight: '280px',
    overflowY: 'auto',
  },
  rightPanel: {
    height: 'calc(100vh - 200px)',
    maxHeight: '700px',
    minHeight: '500px',
  },
  stats: {
    padding: '10px 15px',
    backgroundColor: '#e8e8e8',
    borderRadius: '4px',
    fontSize: '12px',
    color: '#666',
  },
};

function App() {
  // Data state
  const [stateChange, setStateChange] = useState(null);
  const [graph, setGraph] = useState(null);
  const [videoUrl, setVideoUrl] = useState(null);

  // UI state
  const [selectedEventIndex, setSelectedEventIndex] = useState(null);
  const [currentTime, setCurrentTime] = useState(0);
  const [focusedNodeId, setFocusedNodeId] = useState(null);
  const [focusedContainerId, setFocusedContainerId] = useState(null);
  const [showOnlyInvolved, setShowOnlyInvolved] = useState(false);
  const [showThreeBlockView, setShowThreeBlockView] = useState(true);

  // Refs
  const videoRef = useRef(null);
  const hasAutoPausedRef = useRef(false);  // Track if we've auto-paused for current event

  // Pre-compute lineage graph when graph data changes
  const lineageGraph = useMemo(() => {
    if (!graph) return null;
    return buildLineageGraph(graph.lineage_edges, graph.block_graphs);
  }, [graph]);

  // Compute ancestry path for focused node
  const ancestryPath = useMemo(() => {
    if (!focusedNodeId || !lineageGraph) return null;
    return traceAncestry(focusedNodeId, lineageGraph);
  }, [focusedNodeId, lineageGraph]);

  // Compute container history for focused container
  const containerHistory = useMemo(() => {
    if (!focusedContainerId || !lineageGraph) return null;
    const events = stateChange?.events || [];
    return traceContainerHistory(focusedContainerId, lineageGraph, events);
  }, [focusedContainerId, lineageGraph, stateChange]);

  // Handle data loaded from DataLoader
  const handleDataLoaded = useCallback((data) => {
    // state_change.json is an array directly, not {events: [...]}
    const stateChangeData = Array.isArray(data.stateChange)
      ? { events: data.stateChange }
      : data.stateChange;
    setStateChange(stateChangeData);
    setGraph(data.graph);
    setVideoUrl(data.videoUrl);
    setSelectedEventIndex(null);
    setCurrentTime(0);
    setFocusedNodeId(null);
    setFocusedContainerId(null);
  }, []);

  // Handle food node click for lineage tracing
  const handleFoodNodeClick = useCallback((nodeId) => {
    setFocusedNodeId(nodeId);
    setFocusedContainerId(null); // Clear container selection when clicking food
  }, []);

  // Handle container click for container history
  const handleContainerClick = useCallback((containerId) => {
    setFocusedContainerId(containerId);
    setFocusedNodeId(null); // Clear food node selection when clicking container
  }, []);

  // Handle close lineage view
  const handleCloseLineage = useCallback(() => {
    setFocusedNodeId(null);
  }, []);

  // Handle close container history view
  const handleCloseContainerHistory = useCallback(() => {
    setFocusedContainerId(null);
  }, []);

  // Handle event selection from EventList
  const handleSelectEvent = useCallback((index) => {
    setSelectedEventIndex(index);
    hasAutoPausedRef.current = false;  // Reset auto-pause flag for new event

    // Get the event and seek video to its start time
    const events = stateChange?.events || [];
    const event = events[index];
    if (event && event.timestamp_start !== undefined && videoRef.current) {
      videoRef.current.seekTo(event.timestamp_start);
      setCurrentTime(event.timestamp_start);
    }
  }, [stateChange]);

  // Handle time update from VideoPlayer
  // Feature 1: Auto-pause at event end timestamp (only once per event selection)
  const handleTimeUpdate = useCallback((time) => {
    setCurrentTime(time);
    // Auto-pause at event end (only once)
    const events = stateChange?.events || [];
    const selectedEvent = events[selectedEventIndex];
    if (selectedEvent?.timestamp_end &&
        time >= selectedEvent.timestamp_end &&
        !hasAutoPausedRef.current) {
      hasAutoPausedRef.current = true;
      videoRef.current?.pause();
    }
  }, [stateChange, selectedEventIndex]);

  const events = stateChange?.events || [];

  return (
    <div style={styles.app}>
      <div style={styles.header}>
        Food Graph State Visualizer
      </div>

      <div style={styles.content}>
        <DataLoader onDataLoaded={handleDataLoaded} />

        {(stateChange || graph) && (
          <>
            <div style={{ ...styles.stats, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span>Loaded: {events.length} events | {graph?.block_graphs?.length || 0} snapshots | {graph?.lineage_edges?.length || 0} edges | {graph?.inventory?.length || 0} inventory items</span>
              <div style={{ display: 'flex', gap: '16px' }}>
                <label style={{ display: 'flex', alignItems: 'center', gap: '6px', cursor: 'pointer' }}>
                  <input
                    type="checkbox"
                    checked={showThreeBlockView}
                    onChange={(e) => setShowThreeBlockView(e.target.checked)}
                  />
                  Three-Block View
                </label>
                <label style={{ display: 'flex', alignItems: 'center', gap: '6px', cursor: 'pointer' }}>
                  <input
                    type="checkbox"
                    checked={showOnlyInvolved}
                    onChange={(e) => setShowOnlyInvolved(e.target.checked)}
                  />
                  Show only involved nodes
                </label>
              </div>
            </div>

            <div style={{ ...styles.mainLayout, marginTop: '20px' }}>
              {/* Left Panel: Video Player */}
              <div style={styles.leftPanel}>
                <VideoPlayer
                  ref={videoRef}
                  videoUrl={videoUrl}
                  currentTime={currentTime}
                  onTimeUpdate={handleTimeUpdate}
                />
              </div>

              {/* Center Panel: Event List */}
              <div style={styles.centerPanel}>
                <EventList
                  events={events}
                  selectedIndex={selectedEventIndex}
                  onSelectEvent={handleSelectEvent}
                  currentTime={currentTime}
                />
              </div>

              {/* Right Panel: Graph View */}
              <div style={styles.rightPanel}>
                <GraphView
                  graph={graph}
                  selectedEventIndex={selectedEventIndex}
                  events={events}
                  onFoodNodeClick={handleFoodNodeClick}
                  onContainerClick={handleContainerClick}
                  focusedNodeId={focusedNodeId}
                  focusedContainerId={focusedContainerId}
                  showOnlyInvolved={showOnlyInvolved}
                />
              </div>
            </div>

            {/* Lineage View - shows when a food node is clicked */}
            {focusedNodeId && (
              <LineageView
                focusNode={focusedNodeId}
                path={ancestryPath}
                onClose={handleCloseLineage}
              />
            )}

            {/* Container History View - shows when a container is clicked */}
            {focusedContainerId && (
              <ContainerHistoryView
                containerId={focusedContainerId}
                history={containerHistory}
                onClose={handleCloseContainerHistory}
              />
            )}

            {/* Three Block View - shows three consecutive graph states */}
            {showThreeBlockView && (
              <ThreeBlockView
                graph={graph}
                selectedEventIndex={selectedEventIndex}
                events={events}
                showOnlyInvolved={showOnlyInvolved}
                onClose={() => setShowThreeBlockView(false)}
              />
            )}

            {/* Debug Footer - VLM Reasoning */}
            <div style={{ marginTop: '20px' }}>
              <DebugFooter
                selectedEventIndex={selectedEventIndex}
                events={events}
                vlmLogs={graph?.vlm_logs}
              />
            </div>
          </>
        )}
      </div>
    </div>
  );
}

export default App;

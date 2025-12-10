import React, { useState, useRef, useCallback } from 'react';
import DataLoader from './components/DataLoader';
import VideoPlayer from './components/VideoPlayer';
import EventList from './components/EventList';
import GraphView from './components/GraphView';

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
    minHeight: '500px',
  },
  rightPanel: {
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

  // Refs
  const videoRef = useRef(null);

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
  }, []);

  // Handle event selection from EventList
  const handleSelectEvent = useCallback((index) => {
    setSelectedEventIndex(index);

    // Get the event and seek video to its start time
    const events = stateChange?.events || [];
    const event = events[index];
    if (event && event.timestamp_start !== undefined && videoRef.current) {
      videoRef.current.seekTo(event.timestamp_start);
      setCurrentTime(event.timestamp_start);
    }
  }, [stateChange]);

  // Handle time update from VideoPlayer
  const handleTimeUpdate = useCallback((time) => {
    setCurrentTime(time);
  }, []);

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
            <div style={styles.stats}>
              Loaded: {events.length} events | {graph?.block_graphs?.length || 0} snapshots | {graph?.lineage_edges?.length || 0} edges | {graph?.inventory?.length || 0} inventory items
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
                />
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
}

export default App;

import React, { useState } from 'react';
import ClusterVisualizer from './components/ClusterVisualizer';
import ActivityGraphVisualizer from './components/ActivityGraphVisualizer';
import CaptionsViewer from './components/CaptionsViewer';
import ContextCaptionsViewer from './components/ContextCaptionsViewer';
import KGVisualizer from './components/KGVisualizer';

function App() {
  const [currentMode, setCurrentMode] = useState('kg');

  return (
    <div className="App">
      {/* Header */}
      <div style={{
        padding: '10px 20px',
        backgroundColor: '#f8f9fa',
        borderBottom: '1px solid #dee2e6',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center'
      }}>
        <h1 style={{ margin: 0, fontSize: '1.5em' }}>NeuroTrace Visualizer</h1>

        {/* Mode Selector */}
        <div>
          <button
            style={{
              margin: '0 5px',
              padding: '8px 16px',
              backgroundColor: currentMode === 'kg' ? '#28a745' : '#f8f9fa',
              color: currentMode === 'kg' ? 'white' : '#333',
              border: '1px solid #dee2e6',
              borderRadius: '4px',
              cursor: 'pointer',
              fontSize: '0.9em',
              fontWeight: currentMode === 'kg' ? 'bold' : 'normal'
            }}
            onClick={() => setCurrentMode('kg')}
          >
            🍕 Knowledge Graph
          </button>
          <button
            style={{
              margin: '0 5px',
              padding: '8px 16px',
              backgroundColor: currentMode === 'captions' ? '#007bff' : '#f8f9fa',
              color: currentMode === 'captions' ? 'white' : '#333',
              border: '1px solid #dee2e6',
              borderRadius: '4px',
              cursor: 'pointer',
              fontSize: '0.9em'
            }}
            onClick={() => setCurrentMode('captions')}
          >
            Captions Mode
          </button>
          <button
            style={{
              margin: '0 5px',
              padding: '8px 16px',
              backgroundColor: currentMode === 'context-captions' ? '#28a745' : '#f8f9fa',
              color: currentMode === 'context-captions' ? 'white' : '#333',
              border: '1px solid #dee2e6',
              borderRadius: '4px',
              cursor: 'pointer',
              fontSize: '0.9em'
            }}
            onClick={() => setCurrentMode('context-captions')}
          >
            Context Captions
          </button>
          <button
            style={{
              margin: '0 5px',
              padding: '8px 16px',
              backgroundColor: currentMode === 'cluster' ? '#007bff' : '#f8f9fa',
              color: currentMode === 'cluster' ? 'white' : '#333',
              border: '1px solid #dee2e6',
              borderRadius: '4px',
              cursor: 'pointer',
              fontSize: '0.9em'
            }}
            onClick={() => setCurrentMode('cluster')}
          >
            Cluster Mode
          </button>
          <button
            style={{
              margin: '0 5px',
              padding: '8px 16px',
              backgroundColor: currentMode === 'activity' ? '#007bff' : '#f8f9fa',
              color: currentMode === 'activity' ? 'white' : '#333',
              border: '1px solid #dee2e6',
              borderRadius: '4px',
              cursor: 'pointer',
              fontSize: '0.9em'
            }}
            onClick={() => setCurrentMode('activity')}
          >
            Activity Graph Mode
          </button>
        </div>
      </div>

      {/* Content */}
      {currentMode === 'kg' && <KGVisualizer />}
      {currentMode === 'captions' && <CaptionsViewer />}
      {currentMode === 'context-captions' && <ContextCaptionsViewer />}
      {currentMode === 'cluster' && <ClusterVisualizer />}
      {currentMode === 'activity' && <ActivityGraphVisualizer />}
    </div>
  );
}

export default App;

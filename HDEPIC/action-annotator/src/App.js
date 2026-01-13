import React, { useState } from 'react';
import VideoAnnotationTool from './components/VideoAnnotationTool';
import AIAssistedAnnotator from './components/AIAssistedAnnotator';
import OutputEditAnnotator from './components/OutputEditAnnotator';

function App() {
  const [mode, setMode] = useState('ai-assisted'); // 'manual', 'ai-assisted', or 'output-edit'

  return (
    <div className="App">
      {/* Mode Selector */}
      <div style={{ 
        padding: '10px 20px', 
        backgroundColor: '#f8f9fa', 
        borderBottom: '1px solid #dee2e6',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center'
      }}>
        <h1 style={{ margin: 0, fontSize: '1.5em' }}>NeuroTrace Video Annotation Tool</h1>
        <div>
          <button
            onClick={() => setMode('ai-assisted')}
            style={{
              padding: '8px 16px',
              marginRight: '10px',
              backgroundColor: mode === 'ai-assisted' ? '#007bff' : '#6c757d',
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              cursor: 'pointer'
            }}
          >
            AI-Assisted Mode
          </button>
          <button
            onClick={() => setMode('output-edit')}
            style={{
              padding: '8px 16px',
              marginRight: '10px',
              backgroundColor: mode === 'output-edit' ? '#007bff' : '#6c757d',
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              cursor: 'pointer'
            }}
          >
            Output Edit Mode
          </button>
          <button
            onClick={() => setMode('manual')}
            style={{
              padding: '8px 16px',
              backgroundColor: mode === 'manual' ? '#007bff' : '#6c757d',
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              cursor: 'pointer'
            }}
          >
            Manual Mode
          </button>
        </div>
      </div>

      {/* Content */}
      {mode === 'ai-assisted' ? (
        <AIAssistedAnnotator />
      ) : mode === 'output-edit' ? (
        <OutputEditAnnotator />
      ) : (
        <VideoAnnotationTool />
      )}
    </div>
  );
}

export default App;

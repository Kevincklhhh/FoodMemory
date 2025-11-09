import React, { useState, useEffect } from "react";

const ControlPanel = ({ 
  task, 
  currentTaskIndex, 
  totalTasks,
  onReject, 
  onNext, 
  onPrevious,
  onJumpToTask,
  onTaskUpdate,
  videoRef 
}) => {
  const [notes, setNotes] = useState(task?.notes || "");
  const [showNotes, setShowNotes] = useState(false);
  const [viewClassification, setViewClassification] = useState(task?.viewClassification || "HOI in the view");
  const [hand, setHand] = useState(task?.hand || null);
  
  // Update state when task changes
  useEffect(() => {
    setNotes(task?.notes || "");
    setViewClassification(task?.viewClassification || "HOI in the view");
    setHand(task?.hand || null);
  }, [task?.id, task?.notes, task?.viewClassification, task?.hand]);
  
  if (!task) {
    return (
      <div style={{ 
        padding: '20px', 
        backgroundColor: '#f5f5f5',
        borderRadius: '4px',
        textAlign: 'center',
        color: '#666'
      }}>
        No task available
      </div>
    );
  }

  const handleReject = () => {
    const updatedTask = { ...task, status: 'rejected', notes, viewClassification, hand };
    onTaskUpdate(updatedTask);
    onReject(updatedTask);
    // Don't auto-navigate on reject - let user decide
  };

  const handleAccept = () => {
    const updatedTask = { ...task, status: 'accepted', notes, viewClassification, hand };
    onTaskUpdate(updatedTask);
    // Don't auto-navigate on accept - let user decide
  };

  const handleViewClassificationChange = (newClassification) => {
    setViewClassification(newClassification);
    const updatedTask = { ...task, viewClassification: newClassification, notes, hand };
    onTaskUpdate(updatedTask);
  };

  const handleHandChange = (newHand) => {
    setHand(newHand);
    const updatedTask = { ...task, hand: newHand, notes, viewClassification };
    onTaskUpdate(updatedTask);
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'accepted': return '#4caf50';
      case 'rejected': return '#f44336';
      default: return '#2196f3';
    }
  };

  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const handlePlaySegment = () => {
    if (videoRef?.current) {
      videoRef.current.currentTime = task.startTime;
      videoRef.current.play();
      
      // Pause at end time
      const checkTime = () => {
        if (videoRef.current.currentTime >= task.endTime) {
          videoRef.current.pause();
        } else {
          requestAnimationFrame(checkTime);
        }
      };
      requestAnimationFrame(checkTime);
    }
  };

  const handleJumpToStart = () => {
    if (videoRef?.current) {
      videoRef.current.currentTime = Math.max(0, task.startTime - 2); // 2 seconds before for context
    }
  };

  return (
    <div style={{ 
      padding: '20px', 
      backgroundColor: 'white', 
      borderRadius: '4px',
      border: '1px solid #ddd'
    }}>
      {/* Progress and Task Info */}
      <div style={{ marginBottom: '20px' }}>
        <div style={{ 
          display: 'flex', 
          justifyContent: 'space-between', 
          alignItems: 'center',
          marginBottom: '10px'
        }}>
          <h3 style={{ margin: 0, color: '#333' }}>
            Task {currentTaskIndex + 1} of {totalTasks}
          </h3>
          <div style={{
            padding: '4px 12px',
            borderRadius: '12px',
            backgroundColor: getStatusColor(task.status),
            color: 'white',
            fontSize: '0.8em',
            fontWeight: 'bold',
            textTransform: 'capitalize'
          }}>
            {task.status}
          </div>
        </div>
        
        <div style={{ 
          padding: '15px',
          backgroundColor: '#f8f9fa',
          borderRadius: '4px',
          border: '1px solid #e9ecef'
        }}>
          <div style={{ fontSize: '1.2em', fontWeight: 'bold', marginBottom: '8px' }}>
            "{task.description}"
          </div>
          <div style={{ fontSize: '0.9em', color: '#666' }}>
            Time: {formatTime(task.startTime)} → {formatTime(task.endTime)} 
            ({formatTime(task.endTime - task.startTime)} duration)
          </div>
          <div style={{ fontSize: '0.9em', color: '#666', marginTop: '4px' }}>
            Confidence: {(task.confidence * 100).toFixed(1)}% | 
            Window: {task.timeWindow}
          </div>
          <div style={{ fontSize: '0.9em', color: '#666', marginTop: '4px' }}>
            View: {task.viewClassification || viewClassification}
          </div>
          <div style={{ fontSize: '0.9em', color: '#666', marginTop: '4px' }}>
            Hand: {task.hand || hand || 'Not specified'}
          </div>
        </div>
      </div>

      {/* Video Controls */}
      <div style={{ marginBottom: '20px' }}>
        <h4 style={{ marginBottom: '10px', color: '#333' }}>Video Controls</h4>
        <div style={{ display: 'flex', gap: '10px', flexWrap: 'wrap' }}>
          <button
            onClick={handleJumpToStart}
            style={{
              padding: '8px 16px',
              backgroundColor: '#6c757d',
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              cursor: 'pointer'
            }}
          >
            Jump to Event
          </button>
          <button
            onClick={handlePlaySegment}
            style={{
              padding: '8px 16px',
              backgroundColor: '#17a2b8',
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              cursor: 'pointer'
            }}
          >
            ▶ Play Segment
          </button>
        </div>
      </div>

      {/* View Classification */}
      <div style={{ marginBottom: '20px' }}>
        <h4 style={{ marginBottom: '10px', color: '#333' }}>View Classification</h4>
        <div style={{ display: 'flex', gap: '5px', flexWrap: 'wrap' }}>
          {[
            { label: 'HOI in the view', hotkey: '1' },
            { label: 'HOI partially in the view', hotkey: '2' },
            { label: 'Not in the view', hotkey: '3' }
          ].map(({ label, hotkey }) => (
            <button
              key={label}
              onClick={() => handleViewClassificationChange(label)}
              style={{
                padding: '8px 12px',
                backgroundColor: viewClassification === label ? '#2196f3' : '#f8f9fa',
                color: viewClassification === label ? 'white' : '#333',
                border: '1px solid ' + (viewClassification === label ? '#2196f3' : '#ddd'),
                borderRadius: '4px',
                cursor: 'pointer',
                fontSize: '0.9em'
              }}
            >
              [{hotkey}] {label}
            </button>
          ))}
        </div>
      </div>

      {/* Hand Selection */}
      <div style={{ marginBottom: '20px' }}>
        <h4 style={{ marginBottom: '10px', color: '#333' }}>Hand Used</h4>
        <div style={{ display: 'flex', gap: '5px', flexWrap: 'wrap' }}>
          {[
            { label: 'Left', value: 'left', hotkey: '4' },
            { label: 'Right', value: 'right', hotkey: '5' },
            { label: 'Both', value: 'both', hotkey: '6' },
            { label: 'None', value: null, hotkey: '7' }
          ].map(({ label, value, hotkey }) => (
            <button
              key={label}
              onClick={() => handleHandChange(value)}
              style={{
                padding: '8px 12px',
                backgroundColor: hand === value ? '#ff9800' : '#f8f9fa',
                color: hand === value ? 'white' : '#333',
                border: '1px solid ' + (hand === value ? '#ff9800' : '#ddd'),
                borderRadius: '4px',
                cursor: 'pointer',
                fontSize: '0.9em'
              }}
            >
              [{hotkey}] {label}
            </button>
          ))}
        </div>
      </div>

      {/* Main Actions */}
      <div style={{ marginBottom: '20px' }}>
        <h4 style={{ marginBottom: '10px', color: '#333' }}>Review Actions</h4>
        <div style={{ display: 'flex', gap: '10px', flexWrap: 'wrap' }}>
          <button
            onClick={handleAccept}
            disabled={task.status === 'accepted'}
            style={{
              padding: '12px 24px',
              backgroundColor: task.status === 'accepted' ? '#c8e6c9' : '#4caf50',
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              cursor: task.status === 'accepted' ? 'default' : 'pointer',
              fontWeight: 'bold',
              fontSize: '1em'
            }}
          >
            ✓ Accept
          </button>
          
          <button
            onClick={handleReject}
            disabled={task.status === 'rejected'}
            style={{
              padding: '12px 24px',
              backgroundColor: task.status === 'rejected' ? '#ffcdd2' : '#f44336',
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              cursor: task.status === 'rejected' ? 'default' : 'pointer',
              fontWeight: 'bold',
              fontSize: '1em'
            }}
          >
            ✗ Reject
          </button>
        </div>
        <div style={{ marginTop: '10px', fontSize: '0.9em', color: '#666' }}>
          Note: Pending tasks are automatically accepted when navigating to another task.
        </div>
      </div>

      {/* Navigation */}
      <div style={{ marginBottom: '15px' }}>
        <h4 style={{ marginBottom: '10px', color: '#333' }}>Navigation</h4>
        <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
          <button
            onClick={onPrevious}
            disabled={currentTaskIndex <= 0}
            style={{
              padding: '8px 16px',
              backgroundColor: currentTaskIndex <= 0 ? '#e0e0e0' : '#007bff',
              color: currentTaskIndex <= 0 ? '#999' : 'white',
              border: 'none',
              borderRadius: '4px',
              cursor: currentTaskIndex <= 0 ? 'default' : 'pointer'
            }}
          >
            ← Previous
          </button>
          
          <span style={{ 
            padding: '8px 16px',
            backgroundColor: '#f8f9fa',
            borderRadius: '4px',
            fontSize: '0.9em'
          }}>
            {currentTaskIndex + 1} / {totalTasks}
          </span>
          
          <button
            onClick={onNext}
            disabled={currentTaskIndex >= totalTasks - 1}
            style={{
              padding: '8px 16px',
              backgroundColor: currentTaskIndex >= totalTasks - 1 ? '#e0e0e0' : '#007bff',
              color: currentTaskIndex >= totalTasks - 1 ? '#999' : 'white',
              border: 'none',
              borderRadius: '4px',
              cursor: currentTaskIndex >= totalTasks - 1 ? 'default' : 'pointer'
            }}
          >
            Next →
          </button>
          
          <input
            type="number"
            min="1"
            max={totalTasks}
            placeholder="Go to task"
            style={{
              width: '80px',
              padding: '6px',
              borderRadius: '4px',
              border: '1px solid #ddd'
            }}
            onKeyPress={(e) => {
              if (e.key === 'Enter') {
                const taskNum = parseInt(e.target.value);
                if (taskNum >= 1 && taskNum <= totalTasks) {
                  onJumpToTask(taskNum - 1);
                  e.target.value = '';
                }
              }
            }}
          />
        </div>
      </div>

      {/* Notes Section */}
      <div>
        <button
          onClick={() => setShowNotes(!showNotes)}
          style={{
            padding: '8px 16px',
            backgroundColor: '#6c757d',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            cursor: 'pointer',
            marginBottom: showNotes ? '10px' : '0'
          }}
        >
          {showNotes ? 'Hide Notes' : 'Add Notes'} 📝
        </button>
        
        {showNotes && (
          <textarea
            value={notes}
            onChange={(e) => setNotes(e.target.value)}
            placeholder="Add notes about this task..."
            style={{
              width: '100%',
              height: '80px',
              padding: '10px',
              borderRadius: '4px',
              border: '1px solid #ddd',
              resize: 'vertical'
            }}
          />
        )}
      </div>
    </div>
  );
};

export default ControlPanel;
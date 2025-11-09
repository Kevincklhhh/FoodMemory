import React, { useState, useRef, useCallback } from "react";

const Timeline = ({ 
  task, 
  videoDuration, 
  onTimeUpdate, 
  currentTime = 0,
  onTaskUpdate 
}) => {
  const [isDragging, setIsDragging] = useState(false);
  const [dragType, setDragType] = useState(null); // 'start', 'end', 'move'
  const [dragStart, setDragStart] = useState({ x: 0, time: 0 });
  const timelineRef = useRef(null);

  const timeToPixel = useCallback((time) => {
    if (!timelineRef.current || !videoDuration) return 0;
    const timelineWidth = timelineRef.current.offsetWidth - 40; // Account for padding
    return (time / videoDuration) * timelineWidth + 20;
  }, [videoDuration]);

  const pixelToTime = useCallback((pixel) => {
    if (!timelineRef.current || !videoDuration) return 0;
    const timelineWidth = timelineRef.current.offsetWidth - 40;
    const relativePixel = Math.max(0, Math.min(pixel - 20, timelineWidth));
    return (relativePixel / timelineWidth) * videoDuration;
  }, [videoDuration]);

  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const handleMouseDown = (e, type) => {
    e.preventDefault();
    setIsDragging(true);
    setDragType(type);
    setDragStart({
      x: e.clientX,
      time: type === 'start' ? task.startTime : task.endTime
    });
    
    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);
  };

  const handleMouseMove = useCallback((e) => {
    if (!isDragging || !timelineRef.current) return;

    const rect = timelineRef.current.getBoundingClientRect();
    const newTime = pixelToTime(e.clientX - rect.left);
    
    if (dragType === 'start') {
      const newStartTime = Math.max(0, Math.min(newTime, task.endTime - 0.1));
      onTaskUpdate({
        ...task,
        startTime: newStartTime
      });
    } else if (dragType === 'end') {
      const newEndTime = Math.max(task.startTime + 0.1, Math.min(newTime, videoDuration));
      onTaskUpdate({
        ...task,
        endTime: newEndTime
      });
    } else if (dragType === 'move') {
      const deltaTime = pixelToTime(e.clientX - dragStart.x) - pixelToTime(0);
      const duration = task.endTime - task.startTime;
      let newStartTime = dragStart.time + deltaTime;
      
      // Constrain to video bounds
      newStartTime = Math.max(0, newStartTime);
      newStartTime = Math.min(newStartTime, videoDuration - duration);
      
      onTaskUpdate({
        ...task,
        startTime: newStartTime,
        endTime: newStartTime + duration
      });
    }
  }, [isDragging, dragType, task, videoDuration, pixelToTime, dragStart, onTaskUpdate]);

  const handleMouseUp = useCallback(() => {
    setIsDragging(false);
    setDragType(null);
    document.removeEventListener('mousemove', handleMouseMove);
    document.removeEventListener('mouseup', handleMouseUp);
  }, [handleMouseMove]);

  const handleTimelineClick = (e) => {
    if (isDragging) return;
    const rect = timelineRef.current.getBoundingClientRect();
    const clickTime = pixelToTime(e.clientX - rect.left);
    onTimeUpdate(clickTime);
  };

  if (!task || !videoDuration) {
    return (
      <div style={{ 
        padding: '20px', 
        backgroundColor: '#f5f5f5',
        borderRadius: '4px',
        textAlign: 'center',
        color: '#666'
      }}>
        No task selected
      </div>
    );
  }

  const startPixel = timeToPixel(task.startTime);
  const endPixel = timeToPixel(task.endTime);
  const currentPixel = timeToPixel(currentTime);

  return (
    <div style={{ padding: '20px', backgroundColor: 'white', borderRadius: '4px' }}>
      <div style={{ marginBottom: '15px' }}>
        <h4 style={{ margin: '0 0 10px 0', color: '#333' }}>
          Interactive Timeline: {task.description}
        </h4>
        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.9em', color: '#666' }}>
          <span>Start: {formatTime(task.startTime)} | End: {formatTime(task.endTime)}</span>
          <span>Duration: {formatTime(task.endTime - task.startTime)}</span>
        </div>
      </div>
      
      <div 
        ref={timelineRef}
        onClick={handleTimelineClick}
        style={{
          position: 'relative',
          height: '60px',
          backgroundColor: '#e0e0e0',
          borderRadius: '4px',
          cursor: 'pointer',
          border: '1px solid #ccc'
        }}
      >
        {/* Time markers */}
        <div style={{ position: 'absolute', top: '-20px', left: '20px', fontSize: '0.8em', color: '#666' }}>
          0:00
        </div>
        <div style={{ position: 'absolute', top: '-20px', right: '20px', fontSize: '0.8em', color: '#666' }}>
          {formatTime(videoDuration)}
        </div>
        
        {/* Current time indicator */}
        <div 
          style={{
            position: 'absolute',
            left: currentPixel,
            top: 0,
            bottom: 0,
            width: '2px',
            backgroundColor: '#ff0000',
            zIndex: 10,
            pointerEvents: 'none'
          }}
        />
        
        {/* Task time range */}
        <div
          style={{
            position: 'absolute',
            left: startPixel,
            width: endPixel - startPixel,
            top: '10px',
            height: '40px',
            backgroundColor: task.status === 'confirmed' ? '#4caf50' : 
                           task.status === 'rejected' ? '#f44336' :
                           task.status === 'modified' ? '#ff9800' : '#2196f3',
            opacity: 0.7,
            borderRadius: '4px',
            cursor: 'move',
            border: '1px solid rgba(0,0,0,0.3)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            color: 'white',
            fontSize: '0.8em',
            fontWeight: 'bold'
          }}
          onMouseDown={(e) => handleMouseDown(e, 'move')}
        >
          {task.description}
        </div>
        
        {/* Start handle */}
        <div
          style={{
            position: 'absolute',
            left: startPixel - 5,
            top: '8px',
            width: '10px',
            height: '44px',
            backgroundColor: '#1976d2',
            cursor: 'ew-resize',
            borderRadius: '2px 0 0 2px',
            border: '1px solid #0d47a1'
          }}
          onMouseDown={(e) => handleMouseDown(e, 'start')}
        />
        
        {/* End handle */}
        <div
          style={{
            position: 'absolute',
            left: endPixel - 5,
            top: '8px',
            width: '10px',
            height: '44px',
            backgroundColor: '#1976d2',
            cursor: 'ew-resize',
            borderRadius: '0 2px 2px 0',
            border: '1px solid #0d47a1'
          }}
          onMouseDown={(e) => handleMouseDown(e, 'end')}
        />
      </div>
      
      <div style={{ 
        marginTop: '15px', 
        padding: '10px', 
        backgroundColor: '#f5f5f5',
        borderRadius: '4px',
        fontSize: '0.9em'
      }}>
        <strong>Instructions:</strong>
        <ul style={{ margin: '5px 0', paddingLeft: '20px' }}>
          <li>Click anywhere on timeline to jump to that time</li>
          <li>Drag the blue handles to adjust start/end times</li>
          <li>Drag the colored region to move the entire annotation</li>
          <li>Use the control panel below to confirm, reject, or modify</li>
        </ul>
      </div>
    </div>
  );
};

export default Timeline;
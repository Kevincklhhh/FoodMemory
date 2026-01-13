import React, { useState, useRef, useEffect } from "react";
import OutputDataLoader from "./OutputDataLoader";
import Timeline from "./Timeline";
import ControlPanel from "./ControlPanel";
import TaskQueue from "./TaskQueue";

const OutputEditAnnotator = () => {
  const [data, setData] = useState(null);
  const [tasks, setTasks] = useState([]);
  const [currentTaskIndex, setCurrentTaskIndex] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);
  const [videoDuration, setVideoDuration] = useState(0);
  const [playbackRate, setPlaybackRate] = useState(1);
  
  const videoRef = useRef(null);

  // Handle data loading from OutputDataLoader component
  const handleDataLoad = (loadedData) => {
    setData(loadedData);
    setTasks(loadedData.tasks);
    setCurrentTaskIndex(0);
    
    // Jump to first task when data loads
    if (loadedData.tasks.length > 0) {
      setTimeout(() => {
        jumpToTask(0);
      }, 500); // Small delay to ensure video is loaded
    }
  };

  // Auto-accept current task if it's still pending when navigating away
  const autoAcceptCurrentTask = () => {
    const currentTask = tasks[currentTaskIndex];
    if (currentTask && currentTask.status === 'pending') {
      console.log('Auto-accepting task:', currentTask.description);
      const updatedTask = { ...currentTask, status: 'accepted' };
      handleTaskUpdate(updatedTask);
    }
  };

  // Jump to a specific task and set video time
  const jumpToTask = (taskIndex) => {
    if (taskIndex >= 0 && taskIndex < tasks.length) {
      // Auto-accept current task before navigating away
      if (taskIndex !== currentTaskIndex) {
        autoAcceptCurrentTask();
      }
      
      setCurrentTaskIndex(taskIndex);
      const task = tasks[taskIndex];
      
      if (videoRef.current && task) {
        // Jump to 2 seconds before the event for context
        const jumpTime = Math.max(0, task.startTime);
        videoRef.current.currentTime = jumpTime;
        setCurrentTime(jumpTime);
      }
    }
  };

  // Navigate to next task
  const handleNext = () => {
    if (currentTaskIndex < tasks.length - 1) {
      jumpToTask(currentTaskIndex + 1);
    }
  };

  // Navigate to previous task
  const handlePrevious = () => {
    if (currentTaskIndex > 0) {
      jumpToTask(currentTaskIndex - 1);
    }
  };

  // Update a task (for timeline modifications or status changes)
  const handleTaskUpdate = (updatedTask) => {
    setTasks(prevTasks => 
      prevTasks.map(task => 
        task.id === updatedTask.id ? updatedTask : task
      )
    );
  };

  // Handle task acceptance (auto-accept if not rejected)
  const handleAccept = (task) => {
    console.log('Task accepted:', task);
    // Additional acceptance logic can be added here
  };

  // Handle task rejection
  const handleReject = (task) => {
    console.log('Task rejected:', task);
    // Additional rejection logic can be added here
  };

  // Adjust start time of current task
  const adjustStartTime = (delta) => {
    const currentTask = tasks[currentTaskIndex];
    if (currentTask) {
      const newStartTime = Math.max(0, currentTask.startTime + delta);
      const updatedTask = { ...currentTask, startTime: newStartTime };
      handleTaskUpdate(updatedTask);
      console.log(`Adjusted start time by ${delta}s: ${currentTask.startTime} → ${newStartTime}`);
    }
  };

  // Adjust end time of current task
  const adjustEndTime = (delta) => {
    const currentTask = tasks[currentTaskIndex];
    if (currentTask) {
      const newEndTime = Math.max(0, currentTask.endTime + delta);
      const updatedTask = { ...currentTask, endTime: newEndTime };
      handleTaskUpdate(updatedTask);
      console.log(`Adjusted end time by ${delta}s: ${currentTask.endTime} → ${newEndTime}`);
    }
  };

  // Navigate video time by seconds
  const navigateVideoTime = (delta) => {
    if (videoRef.current && videoDuration) {
      const newTime = Math.max(0, Math.min(videoDuration, videoRef.current.currentTime + delta));
      videoRef.current.currentTime = newTime;
      setCurrentTime(newTime);
      console.log(`Navigated video time by ${delta}s to ${newTime.toFixed(2)}s`);
    }
  };

  // Handle manual time update from timeline or controls
  const handleTimeUpdate = (newTime) => {
    if (videoRef.current) {
      videoRef.current.currentTime = newTime;
      setCurrentTime(newTime);
    }
  };

  // Handle video metadata load
  const handleVideoLoad = () => {
    if (videoRef.current) {
      setVideoDuration(videoRef.current.duration);
    }
  };

  // Handle video time updates
  const handleVideoTimeUpdate = () => {
    if (videoRef.current) {
      setCurrentTime(videoRef.current.currentTime);
    }
  };

  // Handle playback rate changes
  const handlePlaybackRateChange = (rate) => {
    setPlaybackRate(rate);
    if (videoRef.current) {
      videoRef.current.playbackRate = rate;
    }
  };

  // Frame navigation
  const handlePrevFrame = () => {
    if (videoRef.current) {
      const fps = 30; // Assuming 30 FPS, could be made configurable
      videoRef.current.currentTime = Math.max(0, videoRef.current.currentTime - 1 / fps);
    }
  };

  const handleNextFrame = () => {
    if (videoRef.current && videoDuration) {
      const fps = 30;
      videoRef.current.currentTime = Math.min(videoDuration, videoRef.current.currentTime + 1 / fps);
    }
  };

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e) => {
      // Only handle shortcuts if no input is focused
      if (document.activeElement.tagName === 'INPUT' || document.activeElement.tagName === 'TEXTAREA') {
        return;
      }

      switch (e.key) {
        case 'ArrowRight':
          e.preventDefault();
          handleNext();
          break;
        case 'ArrowLeft':
          e.preventDefault();
          handlePrevious();
          break;
        case ' ':
          e.preventDefault();
          if (videoRef.current) {
            if (videoRef.current.paused) {
              videoRef.current.play();
            } else {
              videoRef.current.pause();
            }
          }
          break;
        case 'a':
          e.preventDefault();
          if (tasks[currentTaskIndex] && tasks[currentTaskIndex].status !== 'accepted') {
            handleAccept(tasks[currentTaskIndex]);
            handleTaskUpdate({ ...tasks[currentTaskIndex], status: 'accepted' });
          }
          break;
        case 'r':
          e.preventDefault();
          if (tasks[currentTaskIndex] && tasks[currentTaskIndex].status !== 'rejected') {
            handleReject(tasks[currentTaskIndex]);
            handleTaskUpdate({ ...tasks[currentTaskIndex], status: 'rejected' });
          }
          break;
        case 'q':
          e.preventDefault();
          adjustStartTime(-1);
          break;
        case 'w':
          e.preventDefault();
          adjustStartTime(1);
          break;
        case 'z':
          e.preventDefault();
          adjustEndTime(-1);
          break;
        case 'x':
          e.preventDefault();
          adjustEndTime(1);
          break;
        case 'j':
          e.preventDefault();
          navigateVideoTime(-1);
          break;
        case 'k':
          e.preventDefault();
          navigateVideoTime(1);
          break;
        case '1':
          e.preventDefault();
          if (tasks[currentTaskIndex]) {
            const updatedTask = { ...tasks[currentTaskIndex], viewClassification: 'HOI in the view' };
            handleTaskUpdate(updatedTask);
          }
          break;
        case '2':
          e.preventDefault();
          if (tasks[currentTaskIndex]) {
            const updatedTask = { ...tasks[currentTaskIndex], viewClassification: 'HOI partially in the view' };
            handleTaskUpdate(updatedTask);
          }
          break;
        case '3':
          e.preventDefault();
          if (tasks[currentTaskIndex]) {
            const updatedTask = { ...tasks[currentTaskIndex], viewClassification: 'Not in the view' };
            handleTaskUpdate(updatedTask);
          }
          break;
        default:
          break;
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [tasks, currentTaskIndex, handleNext, handlePrevious, handleAccept, handleReject, handleTaskUpdate, adjustStartTime, adjustEndTime, navigateVideoTime]);

  // Format time helper
  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  // Show data loader if no data is loaded
  if (!data) {
    return <OutputDataLoader onDataLoad={handleDataLoad} />;
  }

  const currentTask = tasks[currentTaskIndex];

  return (
    <div style={{ 
      display: 'flex', 
      height: '100vh',
      backgroundColor: '#f5f5f5'
    }}>
      {/* Left Panel - Video and Timeline */}
      <div style={{ 
        flex: '2', 
        padding: '20px',
        display: 'flex',
        flexDirection: 'column',
        gap: '20px'
      }}>
        {/* Video Player */}
        <div style={{ 
          backgroundColor: 'white', 
          borderRadius: '4px',
          padding: '20px',
          border: '1px solid #ddd'
        }}>
          <div style={{ marginBottom: '10px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <h3 style={{ margin: 0 }}>Video: {data.fileName}</h3>
            <div style={{ fontSize: '0.9em', color: '#666' }}>
              {formatTime(currentTime)} / {formatTime(videoDuration)}
            </div>
          </div>
          
          <video
            ref={videoRef}
            src={data.videoUrl}
            controls
            onLoadedMetadata={handleVideoLoad}
            onTimeUpdate={handleVideoTimeUpdate}
            style={{ 
              width: '100%', 
              maxHeight: '400px',
              borderRadius: '4px'
            }}
          />
          
          {/* Video Controls */}
          <div style={{ 
            marginTop: '15px', 
            display: 'flex', 
            gap: '15px', 
            alignItems: 'center',
            flexWrap: 'wrap'
          }}>
            <div>
              <label style={{ marginRight: '10px', fontSize: '0.9em' }}>Speed:</label>
              {[0.25, 0.5, 1, 1.5, 2].map(rate => (
                <button
                  key={rate}
                  onClick={() => handlePlaybackRateChange(rate)}
                  style={{
                    margin: '0 2px',
                    padding: '4px 8px',
                    backgroundColor: playbackRate === rate ? '#007bff' : '#f8f9fa',
                    color: playbackRate === rate ? 'white' : '#333',
                    border: '1px solid #ddd',
                    borderRadius: '3px',
                    cursor: 'pointer',
                    fontSize: '0.8em'
                  }}
                >
                  {rate}x
                </button>
              ))}
            </div>
            
            <div>
              <button onClick={handlePrevFrame} style={{ marginRight: '5px', padding: '6px 12px', fontSize: '0.8em' }}>
                ⏮ Frame
              </button>
              <button onClick={handleNextFrame} style={{ padding: '6px 12px', fontSize: '0.8em' }}>
                Frame ⏭
              </button>
            </div>
          </div>
          
          {/* Keyboard Shortcuts Help */}
          <div style={{ 
            marginTop: '10px', 
            padding: '10px', 
            backgroundColor: '#f8f9fa',
            borderRadius: '4px',
            fontSize: '0.8em',
            color: '#666'
          }}>
            <strong>Shortcuts:</strong> ← Prev | → Next | Space Play/Pause | A Accept | R Reject<br/>
            <strong>Time Adjust:</strong> Q Start-1s | W Start+1s | Z End-1s | X End+1s<br/>
            <strong>Video Nav:</strong> J Video-1s | K Video+1s<br/>
            <strong>View Classification:</strong> 1 HOI in view | 2 HOI partially | 3 Not in view
          </div>
        </div>

        {/* Timeline */}
        <Timeline
          task={currentTask}
          videoDuration={videoDuration}
          currentTime={currentTime}
          onTimeUpdate={handleTimeUpdate}
          onTaskUpdate={handleTaskUpdate}
        />

        {/* Control Panel */}
        <ControlPanel
          task={currentTask}
          currentTaskIndex={currentTaskIndex}
          totalTasks={tasks.length}
          onReject={handleReject}
          onNext={handleNext}
          onPrevious={handlePrevious}
          onJumpToTask={jumpToTask}
          onTaskUpdate={handleTaskUpdate}
          videoRef={videoRef}
        />
      </div>

      {/* Right Panel - Task Queue */}
      <div style={{ 
        flex: '1', 
        padding: '20px',
        maxWidth: '400px'
      }}>
        <TaskQueue
          tasks={tasks}
          currentTaskIndex={currentTaskIndex}
          onTaskSelect={jumpToTask}
          onExport={(exportData) => {
            console.log('Exported data:', exportData);
            // Additional export handling can be added here
          }}
          baseName={data.baseName}
        />
      </div>
    </div>
  );
};

export default OutputEditAnnotator;
import React, { useState, useRef, useEffect } from "react";
import DataLoader from "./DataLoader";
import Timeline from "./Timeline";
import ControlPanel from "./ControlPanel";
import TaskQueue from "./TaskQueue";

const AIAssistedAnnotator = () => {
  const [data, setData] = useState(null);
  const [tasks, setTasks] = useState([]);
  const [currentTaskIndex, setCurrentTaskIndex] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);
  const [videoDuration, setVideoDuration] = useState(0);
  const [playbackRate, setPlaybackRate] = useState(1);
  
  // Merge/Unmerge state management
  const [selectedTasks, setSelectedTasks] = useState(new Set());
  const [mergeMode, setMergeMode] = useState(false);
  const [actionHistory, setActionHistory] = useState([]);
  const [nextMergedId, setNextMergedId] = useState(1000); // Start custom merged IDs from 1000
  
  const videoRef = useRef(null);

  // Handle data loading from DataLoader component
  const handleDataLoad = (loadedData) => {
    setData(loadedData);
    const sortedTasks = loadedData.tasks.sort((a, b) => a.startTime - b.startTime);
    setTasks(sortedTasks);
    setCurrentTaskIndex(0);
    
    // Jump to first task when data loads
    if (sortedTasks.length > 0) {
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

  // Generate unique merged ID
  const generateMergedId = () => {
    const newId = `merged_${nextMergedId}`;
    setNextMergedId(prev => prev + 1);
    return newId;
  };

  // Calculate timestamp union for merging different-time events
  const calculateTimestampUnion = (tasks) => {
    const startTimes = tasks.map(t => t.startTime);
    const endTimes = tasks.map(t => t.endTime);
    return {
      startTime: Math.min(...startTimes),
      endTime: Math.max(...endTimes)
    };
  };

  // Save action to history for undo functionality
  const saveActionToHistory = (action) => {
    setActionHistory(prev => [...prev.slice(-9), action]); // Keep last 10 actions
  };

  // Toggle merge mode
  const toggleMergeMode = () => {
    setMergeMode(prev => !prev);
    if (mergeMode) {
      setSelectedTasks(new Set()); // Clear selections when exiting merge mode
    }
  };

  // Toggle task selection for merging
  const toggleTaskSelection = (taskId) => {
    setSelectedTasks(prev => {
      const newSet = new Set(prev);
      if (newSet.has(taskId)) {
        newSet.delete(taskId);
      } else {
        newSet.add(taskId);
      }
      return newSet;
    });
  };

  // Merge selected tasks
  const mergeSelectedTasks = () => {
    if (selectedTasks.size < 2) return;
    
    const tasksToMerge = tasks.filter(task => selectedTasks.has(task.id));
    const timestamps = calculateTimestampUnion(tasksToMerge);
    
    // Create merged task
    const mergedTask = {
      id: generateMergedId(),
      description: tasksToMerge[0].description, // Default to first task's description
      viewClassification: tasksToMerge[0].viewClassification,
      startTime: timestamps.startTime,
      endTime: timestamps.endTime,
      confidence: Math.max(...tasksToMerge.map(t => t.confidence)),
      timeWindow: '',
      status: 'pending',
      isMerged: true,
      mergedDescriptions: tasksToMerge.map(t => t.description),
      primaryDescription: tasksToMerge[0].description,
      mergeCount: tasksToMerge.length,
      originalIds: tasksToMerge.flatMap(t => t.originalIds || [t.id]),
      originalTasks: tasksToMerge.map(t => ({
        ...t,
        originalIds: t.originalIds || [t.id]
      })),
      originalData: tasksToMerge[0].originalData
    };
    
    // Save action to history
    saveActionToHistory({
      type: 'merge',
      mergedTask,
      originalTasks: tasksToMerge,
      timestamp: Date.now()
    });
    
    // Update tasks list
    setTasks(prevTasks => {
      const newTasks = prevTasks.filter(task => !selectedTasks.has(task.id));
      newTasks.push(mergedTask);
      return newTasks.sort((a, b) => a.startTime - b.startTime);
    });
    
    // Clear selections and exit merge mode
    setSelectedTasks(new Set());
    setMergeMode(false);
    
    console.log(`Merged ${tasksToMerge.length} tasks into ${mergedTask.id}`);
  };

  // Unmerge a specific sub-event from a merged task
  const unmergeSubEvent = (mergedTask, subEventIndex) => {
    if (!mergedTask.originalTasks || subEventIndex >= mergedTask.originalTasks.length) {
      console.error('Invalid unmerge operation: missing originalTasks or invalid index');
      return;
    }
    
    const subEventToUnmerge = mergedTask.originalTasks[subEventIndex];
    const remainingSubEvents = mergedTask.originalTasks.filter((_, idx) => idx !== subEventIndex);
    
    // Save action to history
    saveActionToHistory({
      type: 'unmerge_sub',
      originalMergedTask: mergedTask,
      unmergedTask: subEventToUnmerge,
      remainingMerged: remainingSubEvents.length > 1 ? remainingSubEvents : null,
      timestamp: Date.now()
    });
    
    // Create new task from sub-event
    const newTask = {
      ...subEventToUnmerge,
      id: subEventToUnmerge.originalIds?.[0] || subEventToUnmerge.id,
      isMerged: false,
      status: 'pending'
    };
    
    setTasks(prevTasks => {
      let newTasks = prevTasks.filter(task => task.id !== mergedTask.id);
      
      // Add the unmerged task
      newTasks.push(newTask);
      
      // If there are remaining sub-events, create a new merged task
      if (remainingSubEvents.length > 1) {
        const newTimestamps = calculateTimestampUnion(remainingSubEvents);
        const newMergedTask = {
          ...mergedTask,
          id: generateMergedId(),
          startTime: newTimestamps.startTime,
          endTime: newTimestamps.endTime,
          mergedDescriptions: remainingSubEvents.map(t => t.description),
          mergeCount: remainingSubEvents.length,
          originalIds: remainingSubEvents.flatMap(t => t.originalIds || [t.id]),
          originalTasks: remainingSubEvents
        };
        newTasks.push(newMergedTask);
      } else if (remainingSubEvents.length === 1) {
        // If only one remains, convert back to regular task
        const lastTask = {
          ...remainingSubEvents[0],
          id: remainingSubEvents[0].originalIds?.[0] || remainingSubEvents[0].id,
          isMerged: false,
          status: 'pending'
        };
        newTasks.push(lastTask);
      }
      
      return newTasks.sort((a, b) => a.startTime - b.startTime);
    });
    
    console.log(`Unmerged sub-event from ${mergedTask.id}`);
  };

  // Undo last action
  const undoLastAction = () => {
    if (actionHistory.length === 0) return;
    
    const lastAction = actionHistory[actionHistory.length - 1];
    
    if (lastAction.type === 'merge') {
      // Undo merge: restore original tasks, remove merged task
      setTasks(prevTasks => {
        const newTasks = prevTasks.filter(task => task.id !== lastAction.mergedTask.id);
        return [...newTasks, ...lastAction.originalTasks].sort((a, b) => a.startTime - b.startTime);
      });
    } else if (lastAction.type === 'unmerge_sub') {
      // Undo unmerge: restore original merged task, remove split tasks
      setTasks(prevTasks => {
        let newTasks = prevTasks.filter(task => 
          task.id !== lastAction.unmergedTask.id && 
          !task.originalIds?.some(id => lastAction.originalMergedTask.originalIds?.includes(id))
        );
        newTasks.push(lastAction.originalMergedTask);
        return newTasks.sort((a, b) => a.startTime - b.startTime);
      });
    }
    
    // Remove last action from history
    setActionHistory(prev => prev.slice(0, -1));
    console.log(`Undid ${lastAction.type} action`);
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
      
      // Jump video to new start time
      if (videoRef.current) {
        videoRef.current.currentTime = newStartTime;
        setCurrentTime(newStartTime);
      }
      
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
      
      // Jump video to new end time
      if (videoRef.current) {
        videoRef.current.currentTime = newEndTime;
        setCurrentTime(newEndTime);
      }
      
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
          if (e.ctrlKey || e.metaKey) {
            // Ctrl+Z for undo
            undoLastAction();
          } else {
            adjustEndTime(-1);
          }
          break;
        case 'x':
          e.preventDefault();
          adjustEndTime(1);
          break;
        case 'm':
          e.preventDefault();
          toggleMergeMode();
          break;
        case 'Enter':
          e.preventDefault();
          if (mergeMode && selectedTasks.size >= 2) {
            mergeSelectedTasks();
          }
          break;
        case 'Escape':
          e.preventDefault();
          if (mergeMode) {
            setMergeMode(false);
            setSelectedTasks(new Set());
          }
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
        case '4':
          e.preventDefault();
          if (tasks[currentTaskIndex]) {
            const updatedTask = { ...tasks[currentTaskIndex], hand: 'left' };
            handleTaskUpdate(updatedTask);
          }
          break;
        case '5':
          e.preventDefault();
          if (tasks[currentTaskIndex]) {
            const updatedTask = { ...tasks[currentTaskIndex], hand: 'right' };
            handleTaskUpdate(updatedTask);
          }
          break;
        case '6':
          e.preventDefault();
          if (tasks[currentTaskIndex]) {
            const updatedTask = { ...tasks[currentTaskIndex], hand: 'both' };
            handleTaskUpdate(updatedTask);
          }
          break;
        case '7':
          e.preventDefault();
          if (tasks[currentTaskIndex]) {
            const updatedTask = { ...tasks[currentTaskIndex], hand: null };
            handleTaskUpdate(updatedTask);
          }
          break;
        default:
          break;
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [tasks, currentTaskIndex, handleNext, handlePrevious, handleAccept, handleReject, handleTaskUpdate]);

  // Format time helper
  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  // Show data loader if no data is loaded
  if (!data) {
    return <DataLoader onDataLoad={handleDataLoad} />;
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
          <div style={{ marginBottom: '10px' }}>
            <h3 style={{ margin: 0 }}>Video: {data.fileName}</h3>
          </div>
          
          <div style={{ position: 'relative', display: 'inline-block', width: '100%' }}>
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
            
            {/* Centered time display overlay */}
            <div style={{
              position: 'absolute',
              top: '20px',
              left: '50%',
              transform: 'translateX(-50%)',
              backgroundColor: 'rgba(0, 0, 0, 0.7)',
              color: 'white',
              padding: '8px 16px',
              borderRadius: '6px',
              fontSize: '1.2em',
              fontWeight: 'bold',
              fontFamily: 'monospace',
              pointerEvents: 'none',
              zIndex: 10,
              boxShadow: '0 2px 8px rgba(0, 0, 0, 0.3)',
              textAlign: 'center'
            }}>
              <div>{formatTime(currentTime)} / {formatTime(videoDuration)}</div>
              {tasks[currentTaskIndex] && (
                <div style={{
                  fontSize: '0.8em',
                  marginTop: '4px',
                  color: '#ffcc80',
                  borderTop: '1px solid rgba(255, 255, 255, 0.3)',
                  paddingTop: '4px'
                }}>
                  Clip: {formatTime(tasks[currentTaskIndex].startTime)} - {formatTime(tasks[currentTaskIndex].endTime)}
                </div>
              )}
            </div>
          </div>
          
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
            <strong>Navigation:</strong> ← Prev | → Next | Space Play/Pause | A Accept | R Reject<br/>
            <strong>Time Adjust:</strong> Q Start-1s | W Start+1s | Z End-1s | X End+1s<br/>
            <strong>Video Nav:</strong> J Video-1s | K Video+1s<br/>
            <strong>View Class:</strong> 1 HOI in view | 2 HOI partially | 3 Not in view<br/>
            <strong>Hand:</strong> 4 Left | 5 Right | 6 Both | 7 None<br/>
            <strong>Merge/Unmerge:</strong> M Toggle merge mode | Enter Merge selected | Esc Exit mode | Ctrl+Z Undo
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
          onTaskUpdate={handleTaskUpdate}
          onExport={(exportData) => {
            console.log('Exported data:', exportData);
            // Additional export handling can be added here
          }}
          baseName={data.baseName}
          mergeMode={mergeMode}
          selectedTasks={selectedTasks}
          onToggleTaskSelection={toggleTaskSelection}
          onToggleMergeMode={toggleMergeMode}
          onMergeSelected={mergeSelectedTasks}
          onUnmergeSubEvent={unmergeSubEvent}
          onUndo={undoLastAction}
          canUndo={actionHistory.length > 0}
        />
      </div>
    </div>
  );
};

export default AIAssistedAnnotator;
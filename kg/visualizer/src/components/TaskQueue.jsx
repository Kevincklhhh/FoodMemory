import React, { useState, useEffect, useRef } from "react";

const TaskQueue = ({
  tasks = [],
  currentTaskIndex,
  onTaskSelect,
  onExport,
  baseName,
  onTaskUpdate,
  mergeMode = false,
  selectedTasks = new Set(),
  onToggleTaskSelection,
  onToggleMergeMode,
  onMergeSelected,
  onUnmergeSubEvent,
  onUndo,
  canUndo = false
}) => {
  const [filter, setFilter] = useState('all'); // 'all', 'pending', 'accepted', 'rejected'
  const [collapsed, setCollapsed] = useState(false);
  const [selectedDescriptions, setSelectedDescriptions] = useState({}); // Track selected descriptions for merged tasks
  const taskListRef = useRef(null);
  const activeTaskRef = useRef(null);

  const getStatusColor = (status) => {
    switch (status) {
      case 'accepted': return '#4caf50';
      case 'rejected': return '#f44336';
      default: return '#e0e0e0';
    }
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case 'accepted': return '✓';
      case 'rejected': return '✗';
      default: return '○';
    }
  };

  const filteredTasks = tasks.filter(task =>
    filter === 'all' || task.status === filter
  );

  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const handleDescriptionSelect = (taskId, selectedDesc, originalId) => {
    setSelectedDescriptions(prev => ({
      ...prev,
      [taskId]: { description: selectedDesc, originalId: originalId }
    }));

    // Update the task with the selected description
    if (onTaskUpdate) {
      const taskIndex = tasks.findIndex(t => t.id === taskId);
      if (taskIndex >= 0) {
        const updatedTask = {
          ...tasks[taskIndex],
          description: selectedDesc,
          selectedOriginalId: originalId
        };
        onTaskUpdate(updatedTask);
      }
    }
  };

  const getProgressStats = () => {
    const accepted = tasks.filter(t => t.status === 'accepted').length;
    const rejected = tasks.filter(t => t.status === 'rejected').length;
    const pending = tasks.filter(t => t.status === 'pending').length;
    const merged = tasks.filter(t => t.isMerged).length;

    return { accepted, rejected, pending, merged };
  };

  const stats = getProgressStats();
  const completedCount = stats.accepted + stats.rejected;
  const progressPercent = tasks.length > 0 ? (completedCount / tasks.length) * 100 : 0;

  // Auto-scroll to active task when currentTaskIndex changes
  useEffect(() => {
    if (activeTaskRef.current && taskListRef.current && !collapsed) {
      activeTaskRef.current.scrollIntoView({
        behavior: 'smooth',
        block: 'nearest',
        inline: 'nearest'
      });
    }
  }, [currentTaskIndex, collapsed, filter]);

  const handleExportResults = () => {
    // Only include accepted interactions (exclude rejected ones)
    const acceptedTasks = tasks.filter(t => t.status === 'accepted');
    const rejectedTasks = tasks.filter(t => t.status === 'rejected');

    console.log(`Exporting ${acceptedTasks.length} accepted interactions, excluding ${rejectedTasks.length} rejected interactions`);

    const exportData = {
      analysis_results: acceptedTasks.map(task => {
        // Use selected original ID if available, otherwise use task ID
        const finalId = task.selectedOriginalId ||
                       (selectedDescriptions[task.id]?.originalId) ||
                       (task.originalIds && task.originalIds[0]) ||
                       task.id;

        const result = {
          interaction_id: finalId.replace('_merged', ''), // Remove _merged suffix if present
          event_description: task.description,
          view_classification: task.viewClassification || 'HOI in the view',
          predicted_timestamps: {
            start_time: formatTime(task.startTime),
            end_time: formatTime(task.endTime)
          },
          is_merged: task.isMerged || false,
          merge_count: task.mergeCount || 1
          // Note: includes any time adjustments and view classification changes made during annotation
        };

        // Only include hand attribute if it's specified
        if (task.hand !== null && task.hand !== undefined) {
          result.hand = task.hand;
        }

        return result;
      })
    };

    // Create JSON content and download with video name as filename
    const jsonContent = JSON.stringify(exportData, null, 2);
    const blob = new Blob([jsonContent], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${baseName || 'annotations'}_checked.json`;
    a.click();
    URL.revokeObjectURL(url);

    if (onExport) onExport(exportData);
  };

  return (
    <div style={{
      backgroundColor: 'white',
      borderRadius: '4px',
      border: '1px solid #ddd',
      height: '100%',
      display: 'flex',
      flexDirection: 'column'
    }}>
      {/* Header */}
      <div style={{
        padding: '15px',
        borderBottom: '1px solid #eee',
        backgroundColor: '#f8f9fa'
      }}>
        <div style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          marginBottom: '10px'
        }}>
          <h3 style={{ margin: 0, color: '#333' }}>Task Queue</h3>
          <button
            onClick={() => setCollapsed(!collapsed)}
            style={{
              background: 'none',
              border: 'none',
              fontSize: '1.2em',
              cursor: 'pointer',
              color: '#666'
            }}
          >
            {collapsed ? '▼' : '▲'}
          </button>
        </div>

        {!collapsed && (
          <>
            {/* Progress Bar */}
            <div style={{ marginBottom: '10px' }}>
              <div style={{
                display: 'flex',
                justifyContent: 'space-between',
                fontSize: '0.9em',
                marginBottom: '5px'
              }}>
                <span>Progress: {completedCount} / {tasks.length}</span>
                <span>{progressPercent.toFixed(1)}%</span>
              </div>
              <div style={{
                width: '100%',
                height: '8px',
                backgroundColor: '#e0e0e0',
                borderRadius: '4px',
                overflow: 'hidden'
              }}>
                <div style={{
                  width: `${progressPercent}%`,
                  height: '100%',
                  backgroundColor: '#4caf50',
                  transition: 'width 0.3s ease'
                }} />
              </div>
            </div>

            {/* Stats */}
            <div style={{
              display: 'flex',
              gap: '15px',
              fontSize: '0.8em',
              marginBottom: '10px'
            }}>
              <span style={{ color: '#4caf50' }}>✓ {stats.accepted}</span>
              <span style={{ color: '#f44336' }}>✗ {stats.rejected}</span>
              <span style={{ color: '#666' }}>○ {stats.pending}</span>
              {stats.merged > 0 && (
                <span style={{ color: '#2196f3' }}>⟨⟩ {stats.merged}</span>
              )}
              {mergeMode && (
                <span style={{ color: '#ff9800', fontWeight: 'bold' }}>🔗 MERGE MODE</span>
              )}
            </div>

            {/* Merge Controls */}
            <div style={{
              display: 'flex',
              gap: '8px',
              marginBottom: '10px',
              flexWrap: 'wrap'
            }}>
              <button
                onClick={onToggleMergeMode}
                style={{
                  padding: '4px 8px',
                  fontSize: '0.8em',
                  backgroundColor: mergeMode ? '#ff9800' : '#2196f3',
                  color: 'white',
                  border: 'none',
                  borderRadius: '3px',
                  cursor: 'pointer'
                }}
              >
                {mergeMode ? 'Exit Merge' : 'Merge Mode'}
              </button>

              {mergeMode && selectedTasks.size >= 2 && (
                <button
                  onClick={onMergeSelected}
                  style={{
                    padding: '4px 8px',
                    fontSize: '0.8em',
                    backgroundColor: '#4caf50',
                    color: 'white',
                    border: 'none',
                    borderRadius: '3px',
                    cursor: 'pointer'
                  }}
                >
                  Merge ({selectedTasks.size})
                </button>
              )}

              {canUndo && (
                <button
                  onClick={onUndo}
                  style={{
                    padding: '4px 8px',
                    fontSize: '0.8em',
                    backgroundColor: '#757575',
                    color: 'white',
                    border: 'none',
                    borderRadius: '3px',
                    cursor: 'pointer'
                  }}
                >
                  Undo
                </button>
              )}
            </div>

            {/* Filter */}
            <select
              value={filter}
              onChange={(e) => setFilter(e.target.value)}
              style={{
                padding: '6px 12px',
                borderRadius: '4px',
                border: '1px solid #ddd',
                fontSize: '0.9em',
                width: '100%'
              }}
            >
              <option value="all">All Tasks ({tasks.length})</option>
              <option value="pending">Pending ({stats.pending})</option>
              <option value="accepted">Accepted ({stats.accepted})</option>
              <option value="rejected">Rejected ({stats.rejected})</option>
            </select>
          </>
        )}
      </div>

      {/* Task List */}
      {!collapsed && (
        <div
          ref={taskListRef}
          style={{
            flex: 1,
            overflowY: 'auto',
            maxHeight: 'calc(100vh - 400px)'
          }}
        >
          {filteredTasks.map((task, index) => {
            const originalIndex = tasks.findIndex(t => t.id === task.id);
            const isActive = originalIndex === currentTaskIndex;
            const isSelected = selectedTasks.has(task.id);
            const hasMergedDescriptions = task.isMerged && task.mergedDescriptions && task.mergedDescriptions.length > 1;
            const selectedDesc = selectedDescriptions[task.id];

            return (
              <div
                key={task.id}
                ref={isActive ? activeTaskRef : null}
                style={{
                  borderBottom: '1px solid #f0f0f0',
                  backgroundColor: isActive ? '#e3f2fd' : 'transparent',
                  borderLeft: isActive ? '4px solid #2196f3' : '4px solid transparent',
                  transition: 'all 0.2s ease'
                }}
              >
                {/* Main task content */}
                <div
                  onClick={(e) => {
                    if (mergeMode) {
                      e.preventDefault();
                      onToggleTaskSelection(task.id);
                    } else {
                      onTaskSelect(originalIndex);
                    }
                  }}
                  style={{
                    padding: '12px 15px',
                    cursor: mergeMode ? 'pointer' : 'pointer',
                    backgroundColor: isSelected ? '#fff3e0' : 'transparent'
                  }}
                  onMouseEnter={(e) => {
                    if (!isActive && !isSelected) e.currentTarget.style.backgroundColor = '#f5f5f5';
                  }}
                  onMouseLeave={(e) => {
                    if (!isActive && !isSelected) e.currentTarget.style.backgroundColor = 'transparent';
                  }}
                >
                  <div style={{
                    display: 'flex',
                    alignItems: 'center',
                    marginBottom: '4px'
                  }}>
                    <span style={{
                      display: 'inline-block',
                      width: '20px',
                      height: '20px',
                      borderRadius: '50%',
                      backgroundColor: isSelected ? '#ff9800' : getStatusColor(task.status),
                      color: 'white',
                      textAlign: 'center',
                      lineHeight: '20px',
                      fontSize: '0.7em',
                      marginRight: '10px',
                      border: isSelected ? '2px solid #f57c00' : 'none'
                    }}>
                      {isSelected ? '✓' : getStatusIcon(task.status)}
                    </span>

                    <span style={{
                      fontSize: '0.8em',
                      color: '#666',
                      marginRight: '10px'
                    }}>
                      #{originalIndex + 1}
                    </span>

                    {task.isMerged && (
                      <span style={{
                        display: 'inline-block',
                        fontSize: '0.7em',
                        backgroundColor: '#2196f3',
                        color: 'white',
                        padding: '2px 6px',
                        borderRadius: '10px',
                        marginRight: '8px'
                      }}>
                        {task.mergeCount}x
                      </span>
                    )}

                    <span style={{ fontSize: '0.8em', color: '#666' }}>
                      {formatTime(task.startTime)} - {formatTime(task.endTime)}
                    </span>

                  </div>

                  <div style={{
                    fontSize: '0.9em',
                    fontWeight: isActive ? 'bold' : 'normal',
                    color: '#333',
                    marginBottom: '4px'
                  }}>
                    {task.description}
                  </div>

                  <div style={{
                    fontSize: '0.75em',
                    color: '#999',
                    marginBottom: '2px'
                  }}>
                    View: {task.viewClassification || 'HOI in the view'}
                    <span style={{ marginLeft: '10px' }}>
                      Hand: {task.hand || 'unset'}
                    </span>
                  </div>

                  <div style={{
                    fontSize: '0.75em',
                    color: '#999',
                    display: 'flex',
                    justifyContent: 'space-between'
                  }}>
                    <span>{task.timeWindow}</span>
                    <span>{(task.confidence * 100).toFixed(0)}%</span>
                  </div>
                </div>

                {/* Compact selectable descriptions for merged interactions */}
                {hasMergedDescriptions && (
                  <div style={{
                    padding: '8px 15px',
                    backgroundColor: '#f8f9fa',
                    borderTop: '1px solid #e9ecef'
                  }}>
                    <div style={{
                      display: 'flex',
                      flexDirection: 'column',
                      gap: '4px'
                    }}>
                      {task.mergedDescriptions.map((desc, idx) => {
                        const originalId = task.originalIds ? task.originalIds[idx] : `${task.id}_${idx}`;
                        const isDescSelected = selectedDesc?.description === desc || (!selectedDesc && idx === 0);

                        return (
                          <div key={idx} style={{
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'space-between',
                            padding: '2px 4px',
                            borderRadius: '3px',
                            backgroundColor: isDescSelected ? '#e3f2fd' : 'transparent',
                            fontSize: '0.8em'
                          }}>
                            <label style={{
                              display: 'flex',
                              alignItems: 'center',
                              cursor: 'pointer',
                              flex: 1
                            }}>
                              <input
                                type="radio"
                                name={`desc_${task.id}`}
                                checked={isDescSelected}
                                onChange={() => handleDescriptionSelect(task.id, desc, originalId)}
                                style={{
                                  marginRight: '6px',
                                  transform: 'scale(0.8)'
                                }}
                              />
                              <span style={{
                                color: isDescSelected ? '#1976d2' : '#333'
                              }}>
                                {desc}
                              </span>
                            </label>

                            {/* Unmerge button for individual sub-events */}
                            <button
                              onClick={(e) => {
                                e.preventDefault();
                                e.stopPropagation();
                                if (onUnmergeSubEvent) {
                                  onUnmergeSubEvent(task, idx);
                                } else {
                                  console.error('onUnmergeSubEvent not available');
                                }
                              }}
                              style={{
                                padding: '2px 4px',
                                fontSize: '0.7em',
                                backgroundColor: '#f44336',
                                color: 'white',
                                border: 'none',
                                borderRadius: '2px',
                                cursor: 'pointer',
                                marginLeft: '8px'
                              }}
                              title="Unmerge this sub-event"
                            >
                              ✂
                            </button>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                )}
              </div>
            );
          })}

          {filteredTasks.length === 0 && (
            <div style={{
              padding: '40px 20px',
              textAlign: 'center',
              color: '#666'
            }}>
              No tasks match the current filter
            </div>
          )}
        </div>
      )}

      {/* Export Button */}
      {!collapsed && stats.accepted > 0 && (
        <div style={{
          padding: '15px',
          borderTop: '1px solid #eee',
          backgroundColor: '#f8f9fa'
        }}>
          <button
            onClick={handleExportResults}
            style={{
              width: '100%',
              padding: '12px',
              backgroundColor: '#28a745',
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              cursor: 'pointer',
              fontWeight: 'bold',
              fontSize: '0.9em'
            }}
          >
            📊 Export Accepted ({stats.accepted} interactions)
          </button>
        </div>
      )}
    </div>
  );
};

export default TaskQueue;
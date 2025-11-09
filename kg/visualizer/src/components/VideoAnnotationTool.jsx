import React, { useState, useRef, useEffect } from "react";
import Papa from "papaparse";
import { saveAs } from "file-saver";

export default function VideoAnnotationTool() {
  const [videoSrc, setVideoSrc] = useState("");
  const [playbackRate, setPlaybackRate] = useState(1);
  const [annotations, setAnnotations] = useState([]);
  const [currentLabel, setCurrentLabel] = useState("");
  const [currentDesc, setCurrentDesc] = useState("");
  const [videoId, setVideoId] = useState("video1");
  const [labelStart, setLabelStart] = useState(null);
  const [fps, setFps] = useState(30);
  const [jumpTime, setJumpTime] = useState("");
  const [selectedAnnotationIndex, setSelectedAnnotationIndex] = useState(null);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);

  const videoRef = useRef(null);

  const handleVideoLoad = (e) => {
    const file = e.target.files[0];
    if (file) {
      const url = URL.createObjectURL(file);
      setVideoSrc(url);
      setVideoId(file.name);
    }
  };

  const handleStartLabel = () => {
    if (videoRef.current) {
      setLabelStart(videoRef.current.currentTime);
    }
  };

  const handleEndLabel = () => {
    if (videoRef.current && labelStart !== null && currentLabel !== "") {
      const endTime = videoRef.current.currentTime;
      const newAnnotation = {
        videoId,
        label: currentLabel,
        start: labelStart.toFixed(2),
        end: endTime.toFixed(2),
        description: currentDesc,
      };
      setAnnotations([...annotations, newAnnotation]);
      setLabelStart(null);
    }
  };

  const handleExport = () => {
    const csv = Papa.unparse(
      annotations.map((a) => ({
        video_id: a.videoId,
        label: a.label,
        start_time: a.start,
        end_time: a.end,
      }))
    );
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
    saveAs(blob, `${videoId}_annotations.csv`);
  };

  const handlePrevFrame = () => {
    if (videoRef.current && fps > 0) {
      videoRef.current.currentTime = Math.max(0, videoRef.current.currentTime - 1 / fps);
    }
  };

  const handleNextFrame = () => {
    if (videoRef.current && fps > 0) {
      videoRef.current.currentTime = videoRef.current.currentTime + 1 / fps;
    }
  };

  const handleJumpToTime = () => {
    const time = parseFloat(jumpTime);
    if (videoRef.current && !isNaN(time)) {
      videoRef.current.currentTime = time;
    }
  };

  const adjustStartTime = React.useCallback((delta) => {
    console.log('adjustStartTime called with delta:', delta, 'selectedIndex:', selectedAnnotationIndex);
    if (selectedAnnotationIndex !== null && selectedAnnotationIndex < annotations.length) {
      const newAnnotations = [...annotations];
      const currentStart = parseFloat(newAnnotations[selectedAnnotationIndex].start);
      const newStart = Math.max(0, currentStart + delta);
      console.log('Changing start time from', currentStart, 'to', newStart);
      newAnnotations[selectedAnnotationIndex].start = newStart.toFixed(2);
      setAnnotations(newAnnotations);
    }
  }, [selectedAnnotationIndex, annotations]);

  const adjustEndTime = React.useCallback((delta) => {
    console.log('adjustEndTime called with delta:', delta, 'selectedIndex:', selectedAnnotationIndex);
    if (selectedAnnotationIndex !== null && selectedAnnotationIndex < annotations.length) {
      const newAnnotations = [...annotations];
      const currentEnd = parseFloat(newAnnotations[selectedAnnotationIndex].end);
      const newEnd = Math.max(0, currentEnd + delta);
      console.log('Changing end time from', currentEnd, 'to', newEnd);
      newAnnotations[selectedAnnotationIndex].end = newEnd.toFixed(2);
      setAnnotations(newAnnotations);
    }
  }, [selectedAnnotationIndex, annotations]);

  const handleKeyDown = React.useCallback((e) => {
    console.log('Key pressed:', e.key, 'Selected annotation:', selectedAnnotationIndex);
    if (selectedAnnotationIndex === null) {
      console.log('No annotation selected');
      return;
    }
    
    switch (e.key.toLowerCase()) {
      case 'q':
        console.log('Adjusting start time -1');
        e.preventDefault();
        adjustStartTime(-1);
        break;
      case 'w':
        console.log('Adjusting start time +1');
        e.preventDefault();
        adjustStartTime(1);
        break;
      case 'a':
        console.log('Adjusting end time -1');
        e.preventDefault();
        adjustEndTime(-1);
        break;
      case 's':
        console.log('Adjusting end time +1');
        e.preventDefault();
        adjustEndTime(1);
        break;
      default:
        break;
    }
  }, [selectedAnnotationIndex, adjustStartTime, adjustEndTime]);

  useEffect(() => {
    if (videoRef.current) {
      videoRef.current.playbackRate = playbackRate;
    }
  }, [playbackRate]);

  useEffect(() => {
    document.addEventListener('keydown', handleKeyDown);
    return () => {
      document.removeEventListener('keydown', handleKeyDown);
    };
  }, [handleKeyDown]);

  const handleTimeUpdate = () => {
    if (videoRef.current) {
      setCurrentTime(videoRef.current.currentTime);
    }
  };

  const handleLoadedMetadata = () => {
    if (videoRef.current) {
      setDuration(videoRef.current.duration);
    }
  };

  const formatTime = (time) => {
    const minutes = Math.floor(time / 60);
    const seconds = Math.floor(time % 60);
    const milliseconds = Math.floor((time % 1) * 1000);
    return `${minutes}:${seconds.toString().padStart(2, '0')}.${milliseconds.toString().padStart(3, '0')}`;
  };

  return (
    <div style={{ padding: "20px", display: "flex", gap: "20px" }}>
      <div style={{ flex: 2 }}>
        <input type="file" accept="video/*" onChange={handleVideoLoad} />
        {videoSrc && (
          <>
            <video
              src={videoSrc}
              controls
              ref={videoRef}
              onTimeUpdate={handleTimeUpdate}
              onLoadedMetadata={handleLoadedMetadata}
              style={{ width: "100%", marginTop: "10px" }}
            />
            <div style={{ 
              marginTop: "10px", 
              padding: "10px", 
              backgroundColor: "#f8f9fa", 
              borderRadius: "4px",
              border: "1px solid #dee2e6"
            }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "5px" }}>
                <span style={{ fontWeight: "bold" }}>Current Time: {formatTime(currentTime)}</span>
                <span style={{ color: "#666" }}>Duration: {formatTime(duration)}</span>
              </div>
              <div style={{ 
                width: "100%", 
                height: "8px", 
                backgroundColor: "#e9ecef", 
                borderRadius: "4px", 
                overflow: "hidden" 
              }}>
                <div style={{ 
                  width: duration > 0 ? `${(currentTime / duration) * 100}%` : "0%", 
                  height: "100%", 
                  backgroundColor: "#007bff", 
                  transition: "width 0.1s ease" 
                }} />
              </div>
            </div>
          </>
        )}
        <div style={{ marginTop: "10px" }}>
          Playback Rate: {" "}
          {[0.5, 1, 2, 4, 8].map((r) => (
            <button
              key={r}
              style={{
                margin: "0 5px",
                backgroundColor: playbackRate === r ? "#333" : "#eee",
                color: playbackRate === r ? "white" : "black",
              }}
              onClick={() => setPlaybackRate(r)}
            >
              {r}x
            </button>
          ))}
        </div>
        <div style={{ marginTop: "10px" }}>
          <label>
            FPS:
            <input
              type="number"
              value={fps}
              onChange={(e) => setFps(Number(e.target.value))}
              style={{ width: "60px", marginLeft: "5px", marginRight: "15px" }}
            />
          </label>
          <button onClick={handlePrevFrame}>⏮ Last Frame</button>
          <button onClick={handleNextFrame} style={{ marginLeft: "10px" }}>
            ⏭ Next Frame
          </button>
        </div>
        <div style={{ marginTop: "10px" }}>
          Jump to (sec):
          <input
            type="number"
            value={jumpTime}
            onChange={(e) => setJumpTime(e.target.value)}
            style={{ width: "80px", marginLeft: "5px" }}
          />
          <button onClick={handleJumpToTime} style={{ marginLeft: "10px" }}>
            Jump
          </button>
        </div>
        <div style={{ marginTop: "10px" }}>
          <input
            placeholder="Label"
            value={currentLabel}
            onChange={(e) => setCurrentLabel(e.target.value)}
            style={{ marginRight: "10px" }}
          />
          <input
            placeholder="Description"
            value={currentDesc}
            onChange={(e) => setCurrentDesc(e.target.value)}
            style={{ marginRight: "10px" }}
          />
          <button onClick={handleStartLabel}>Start</button>
          <button onClick={handleEndLabel} style={{ marginLeft: "5px" }}>
            End
          </button>
          <button onClick={handleExport} style={{ marginLeft: "10px" }}>
            Export CSV
          </button>
        </div>
        <div style={{ marginTop: "15px", padding: "10px", backgroundColor: "#f5f5f5", borderRadius: "4px" }}>
          <h4>Hotkeys (select annotation first):</h4>
          <div style={{ fontSize: "0.9em" }}>
            <strong>Q</strong>: Start time -1s | <strong>W</strong>: Start time +1s<br />
            <strong>A</strong>: End time -1s | <strong>S</strong>: End time +1s
          </div>
        </div>
      </div>
      <div style={{ flex: 1 }}>
        <h3>Annotations</h3>
        {annotations.map((a, i) => (
          <div
            key={i}
            style={{ 
              border: selectedAnnotationIndex === i ? "2px solid #007bff" : "1px solid #ccc", 
              marginBottom: "5px", 
              padding: "5px",
              backgroundColor: selectedAnnotationIndex === i ? "#e7f3ff" : "white",
              cursor: "pointer"
            }}
            onClick={() => setSelectedAnnotationIndex(i)}
          >
            <strong>{a.label}</strong> ({a.start}s - {a.end}s)
            <div style={{ fontSize: "0.9em" }}>{a.description}</div>
            {selectedAnnotationIndex === i && (
              <div style={{ fontSize: "0.8em", color: "#007bff", marginTop: "5px" }}>
                ✓ Selected - Use Q/W/A/S to adjust times
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
import React, { useRef, useEffect, useImperativeHandle, forwardRef } from 'react';

const styles = {
  container: {
    backgroundColor: '#000',
    borderRadius: '8px',
    overflow: 'hidden',
    display: 'flex',
    flexDirection: 'column',
  },
  video: {
    width: '100%',
    maxHeight: '400px',
    backgroundColor: '#000',
  },
  controls: {
    padding: '10px',
    backgroundColor: '#1a1a1a',
    color: 'white',
    display: 'flex',
    alignItems: 'center',
    gap: '10px',
  },
  timeDisplay: {
    fontFamily: 'monospace',
    fontSize: '14px',
    minWidth: '120px',
  },
  noVideo: {
    padding: '40px',
    textAlign: 'center',
    color: '#666',
    backgroundColor: '#1a1a1a',
    borderRadius: '8px',
  },
};

const formatTime = (seconds) => {
  if (isNaN(seconds) || seconds === null) return '00:00.0';
  const mins = Math.floor(seconds / 60);
  const secs = (seconds % 60).toFixed(1);
  return `${mins.toString().padStart(2, '0')}:${secs.padStart(4, '0')}`;
};

const VideoPlayer = forwardRef(({ videoUrl, onTimeUpdate, currentTime }, ref) => {
  const videoRef = useRef(null);

  // Expose seekTo method to parent
  useImperativeHandle(ref, () => ({
    seekTo: (time) => {
      if (videoRef.current) {
        videoRef.current.currentTime = time;
      }
    },
    play: () => {
      if (videoRef.current) {
        videoRef.current.play();
      }
    },
    pause: () => {
      if (videoRef.current) {
        videoRef.current.pause();
      }
    },
  }));

  const handleTimeUpdate = () => {
    if (videoRef.current && onTimeUpdate) {
      onTimeUpdate(videoRef.current.currentTime);
    }
  };

  // Sync video when currentTime changes externally
  useEffect(() => {
    if (videoRef.current && currentTime !== undefined) {
      const diff = Math.abs(videoRef.current.currentTime - currentTime);
      // Only seek if difference is significant (> 0.5s)
      if (diff > 0.5) {
        videoRef.current.currentTime = currentTime;
      }
    }
  }, [currentTime]);

  if (!videoUrl) {
    return (
      <div style={styles.noVideo}>
        <p>No video loaded</p>
        <p style={{ fontSize: '12px', marginTop: '10px' }}>
          Load a video file using the data loader above
        </p>
      </div>
    );
  }

  const duration = videoRef.current?.duration || 0;

  return (
    <div style={styles.container}>
      <video
        ref={videoRef}
        src={videoUrl}
        style={styles.video}
        controls
        onTimeUpdate={handleTimeUpdate}
      />
      <div style={styles.controls}>
        <span style={styles.timeDisplay}>
          {formatTime(currentTime)} / {formatTime(duration)}
        </span>
      </div>
    </div>
  );
});

VideoPlayer.displayName = 'VideoPlayer';

export default VideoPlayer;

#!/bin/bash
# Start both the video server and the React app

echo "🎬 Starting Video Server on port 3001..."
node video-server.js &
VIDEO_SERVER_PID=$!

echo "⚛️  Starting React App on port 3000..."
npm start &
REACT_PID=$!

echo ""
echo "=========================================="
echo "  Inventory Visualizer is starting..."
echo "=========================================="
echo ""
echo "  React App:    http://localhost:3000"
echo "  Video Server: http://localhost:3001"
echo ""
echo "  Press Ctrl+C to stop both servers"
echo "=========================================="

# Handle Ctrl+C to kill both processes
trap "echo ''; echo 'Stopping servers...'; kill $VIDEO_SERVER_PID $REACT_PID 2>/dev/null; exit" INT

# Wait for either process to exit
wait

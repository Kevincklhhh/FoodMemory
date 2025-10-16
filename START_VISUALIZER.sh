#!/bin/bash

echo "=========================================="
echo "    KG Visualizer Startup Script"
echo "=========================================="
echo ""

# Kill any existing instances
echo "1. Cleaning up existing processes..."
pkill -f kg_visualizer_server.py 2>/dev/null
fuser -k 5000/tcp 2>/dev/null
sleep 1

# Start backend
echo "2. Starting backend server (Flask)..."
nohup python kg_visualizer_server.py > kg_viz_server.log 2>&1 &
BACKEND_PID=$!
echo "   Backend PID: $BACKEND_PID"
echo "   Waiting for backend to start..."
sleep 3

# Test backend
echo "3. Testing backend health..."
HEALTH=$(curl -s http://localhost:5000/api/health 2>/dev/null)
if [ $? -eq 0 ]; then
    echo "   ✓ Backend is running on http://localhost:5000"
else
    echo "   ✗ Backend failed to start. Check kg_viz_server.log"
    exit 1
fi

# Check snapshot directories
echo "4. Checking snapshot directories..."
SNAPSHOTS=$(curl -s http://localhost:5000/api/snapshots/directories 2>/dev/null | grep -o '"name":"[^"]*"' | wc -l)
echo "   Found $SNAPSHOTS snapshot directory(ies)"

echo ""
echo "=========================================="
echo "Backend is ready!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Open a new terminal"
echo "  2. cd visualizer"
echo "  3. npm start"
echo ""
echo "Or run manually:"
echo "  cd visualizer && npm start"
echo ""
echo "To stop backend:"
echo "  kill $BACKEND_PID"
echo ""
echo "Backend log:"
echo "  tail -f kg_viz_server.log"
echo ""

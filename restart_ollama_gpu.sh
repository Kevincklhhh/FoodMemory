#!/bin/bash
# Script to restart Ollama Docker container with proper GPU support
# For sequential processing - only 1 instance needed

echo "🔄 Restarting Ollama container with GPU support..."

# Update Ollama Docker image
echo "⬆️  Updating Ollama Docker image..."
docker pull ollama/ollama:latest

# Stop and remove existing container if it exists
echo "⏹️  Stopping existing Ollama container..."
docker stop ollama1 2>/dev/null || echo "No existing container to stop"
docker rm ollama1 2>/dev/null || echo "No existing container to remove"

# Start new container with GPU support
echo "🚀 Starting Ollama container with GPU access..."
docker run -d --gpus all --name ollama1 -p 11434:11434 -v ollama:/root/.ollama ollama/ollama:latest

# Wait for container to start
echo "⏳ Waiting for container to initialize..."
sleep 10

# Check container status
echo "📋 Container status:"
docker ps | grep ollama

# Test GPU access in container
echo "🖥️  Testing GPU access:"
docker exec ollama1 nvidia-smi | head -10

# Load the required model
echo "📦 Loading gpt-oss:120b model..."
docker exec ollama1 ollama pull gpt-oss:120b

# Test API connectivity
echo "🔗 Testing API connectivity..."
curl -s http://localhost:11434/api/tags > /dev/null && echo "✅ ollama1 (port 11434) is responding" || echo "❌ ollama1 (port 11434) is not responding"

# List available models
echo "📋 Available models:"
docker exec ollama1 ollama list

echo "🎉 Ollama GPU setup complete!"
echo "Use 'nvidia-smi' to monitor GPU usage"
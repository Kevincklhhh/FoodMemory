/**
 * Simple video server for the Inventory Visualizer
 * Serves video files from the HDEPIC directory
 *
 * Usage: node video-server.js
 * Videos will be available at: http://localhost:3001/videos/{participant}/{video_id}.mp4
 */

const http = require('http');
const fs = require('fs');
const path = require('path');

const PORT = 3001;
// Absolute path to HDEPIC data
const HDEPIC_ROOT = '/Users/kailaicui/FoodMemory/HDEPIC';

// MIME types
const MIME_TYPES = {
  '.mp4': 'video/mp4',
  '.webm': 'video/webm',
  '.mov': 'video/quicktime',
  '.json': 'application/json',
  '.csv': 'text/csv',
};

const server = http.createServer((req, res) => {
  // Enable CORS
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Range');
  res.setHeader('Access-Control-Expose-Headers', 'Content-Range, Content-Length, Accept-Ranges');

  if (req.method === 'OPTIONS') {
    res.writeHead(204);
    res.end();
    return;
  }

  const url = new URL(req.url, `http://localhost:${PORT}`);
  const pathname = decodeURIComponent(url.pathname);

  console.log(`[${new Date().toISOString()}] ${req.method} ${pathname}`);

  // Route: /videos/{participant}/{video_id}.mp4
  // Maps to: HDEPIC/{participant}/{video_id}.mp4
  if (pathname.startsWith('/videos/')) {
    const relativePath = pathname.replace('/videos/', '');
    const filePath = path.join(HDEPIC_ROOT, relativePath);

    serveVideo(req, res, filePath);
    return;
  }

  // Route: /data/{path} - serve JSON/CSV files
  if (pathname.startsWith('/data/')) {
    const relativePath = pathname.replace('/data/', '');
    const filePath = path.join(HDEPIC_ROOT, relativePath);

    serveFile(res, filePath);
    return;
  }

  // Route: /list/{participant} - list available videos
  if (pathname.startsWith('/list/')) {
    const participant = pathname.replace('/list/', '');
    const dirPath = path.join(HDEPIC_ROOT, participant);

    listVideos(res, dirPath, participant);
    return;
  }

  // Default: 404
  res.writeHead(404, { 'Content-Type': 'application/json' });
  res.end(JSON.stringify({ error: 'Not found', availableRoutes: ['/videos/{participant}/{video_id}.mp4', '/data/{path}', '/list/{participant}'] }));
});

function serveVideo(req, res, filePath) {
  if (!fs.existsSync(filePath)) {
    res.writeHead(404, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: 'Video not found', path: filePath }));
    return;
  }

  const stat = fs.statSync(filePath);
  const fileSize = stat.size;
  const ext = path.extname(filePath).toLowerCase();
  const mimeType = MIME_TYPES[ext] || 'application/octet-stream';

  const range = req.headers.range;

  if (range) {
    // Handle range request for video seeking
    const parts = range.replace(/bytes=/, '').split('-');
    const start = parseInt(parts[0], 10);
    const end = parts[1] ? parseInt(parts[1], 10) : fileSize - 1;
    const chunkSize = end - start + 1;

    res.writeHead(206, {
      'Content-Range': `bytes ${start}-${end}/${fileSize}`,
      'Accept-Ranges': 'bytes',
      'Content-Length': chunkSize,
      'Content-Type': mimeType,
    });

    const stream = fs.createReadStream(filePath, { start, end });
    stream.pipe(res);
  } else {
    // Full file request
    res.writeHead(200, {
      'Content-Length': fileSize,
      'Content-Type': mimeType,
      'Accept-Ranges': 'bytes',
    });

    fs.createReadStream(filePath).pipe(res);
  }
}

function serveFile(res, filePath) {
  if (!fs.existsSync(filePath)) {
    res.writeHead(404, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: 'File not found' }));
    return;
  }

  const ext = path.extname(filePath).toLowerCase();
  const mimeType = MIME_TYPES[ext] || 'text/plain';

  const content = fs.readFileSync(filePath);
  res.writeHead(200, { 'Content-Type': mimeType });
  res.end(content);
}

function listVideos(res, dirPath, participant) {
  if (!fs.existsSync(dirPath)) {
    res.writeHead(404, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: 'Directory not found' }));
    return;
  }

  const files = fs.readdirSync(dirPath)
    .filter(f => f.endsWith('.mp4'))
    .map(f => ({
      filename: f,
      videoId: f.replace('.mp4', ''),
      url: `http://localhost:${PORT}/videos/${participant}/${f}`,
    }));

  res.writeHead(200, { 'Content-Type': 'application/json' });
  res.end(JSON.stringify({ participant, videos: files }));
}

server.listen(PORT, () => {
  console.log(`\n🎬 Video Server running at http://localhost:${PORT}`);
  console.log(`\nAvailable endpoints:`);
  console.log(`  GET /videos/{participant}/{video_id}.mp4  - Stream video`);
  console.log(`  GET /data/{path}                          - Serve JSON/CSV files`);
  console.log(`  GET /list/{participant}                   - List available videos`);
  console.log(`\nExample:`);
  console.log(`  http://localhost:${PORT}/videos/P01/P01-20240202-110250.mp4`);
  console.log(`  http://localhost:${PORT}/list/P01`);
  console.log(`  http://localhost:${PORT}/data/outputs/02_inventory/lifecycle/P01/P01_known_quantities.json`);
  console.log(`\nPress Ctrl+C to stop\n`);
});

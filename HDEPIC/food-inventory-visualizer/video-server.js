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

const PORT = 4001;
// Absolute path to HDEPIC data
const HDEPIC_ROOT = path.resolve(__dirname, '..');

// MIME types
const MIME_TYPES = {
  '.mp4': 'video/mp4',
  '.webm': 'video/webm',
  '.mov': 'video/quicktime',
  '.json': 'application/json',
  '.csv': 'text/csv',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.png': 'image/png',
};

const server = http.createServer((req, res) => {
  // Enable CORS
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, PUT, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Range, Content-Type');
  res.setHeader('Access-Control-Expose-Headers', 'Content-Range, Content-Length, Accept-Ranges');

  if (req.method === 'OPTIONS') {
    res.writeHead(204);
    res.end();
    return;
  }

  const url = new URL(req.url, `http://localhost:${PORT}`);
  const pathname = decodeURIComponent(url.pathname);

  // Handle PUT requests for saving JSON files
  if (req.method === 'PUT' && pathname.startsWith('/data/')) {
    const relativePath = pathname.replace('/data/', '');
    const filePath = path.join(HDEPIC_ROOT, relativePath);

    let body = '';
    req.on('data', chunk => {
      body += chunk.toString();
    });
    req.on('end', () => {
      try {
        // Validate JSON
        JSON.parse(body);

        // Ensure directory exists
        const dir = path.dirname(filePath);
        if (!fs.existsSync(dir)) {
          fs.mkdirSync(dir, { recursive: true });
        }

        // Write file
        fs.writeFileSync(filePath, body);
        console.log(`[${new Date().toISOString()}] Saved: ${filePath}`);

        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ success: true, path: filePath }));
      } catch (err) {
        console.error(`[${new Date().toISOString()}] Save error:`, err.message);
        res.writeHead(400, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: err.message }));
      }
    });
    return;
  }

  console.log(`[${new Date().toISOString()}] ${req.method} ${pathname}`);

  // Route: /videos/{participant}/{video_id}.mp4
  // Maps to: HDEPIC/data/HD-EPIC/Videos/{participant}/{video_id}.mp4
  if (pathname.startsWith('/videos/')) {
    const relativePath = pathname.replace('/videos/', '');
    const filePath = path.join(HDEPIC_ROOT, 'data', 'HD-EPIC', 'Videos', relativePath);

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
    const dirPath = path.join(HDEPIC_ROOT, 'data', 'HD-EPIC', 'Videos', participant);

    listVideos(res, dirPath, participant);
    return;
  }

  // Route: /ping - simple health check
  if (pathname === '/ping') {
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ status: 'ok', time: new Date().toISOString(), routes: ['/vlm-tags/', '/videos/', '/data/', '/list/', '/failure-cases'] }));
    return;
  }

  // Route: /narrations - serve all narration timestamps
  if (pathname === '/narrations') {
    const filePath = path.join(HDEPIC_ROOT, 'data', 'hd-epic-annotations', 'narrations_timestamps.json');
    serveFile(res, filePath);
    return;
  }

  // Route: /vlm-tags/{participant} - list available VLM result tags
  if (pathname.startsWith('/vlm-tags/')) {
    const participant = pathname.replace('/vlm-tags/', '');
    const dirPath = path.join(HDEPIC_ROOT, 'outputs', '02_inventory', participant);
    listVlmTags(res, dirPath, participant);
    return;
  }

  // Route: /failure-cases - list all failure_cases*.json files
  if (pathname === '/failure-cases') {
    const dirPath = path.join(HDEPIC_ROOT, 'outputs', '02_inventory', 'failure_cases');
    listFailureCases(res, dirPath);
    return;
  }

  // Route: /hands23/{participant} - get hands23 detection results
  if (pathname.startsWith('/hands23/')) {
    const participant = pathname.replace('/hands23/', '');
    const filePath = path.join(HDEPIC_ROOT, 'outputs', '02_inventory', participant, 'hands23_detection', `${participant}_hands23_results.json`);
    serveFile(res, filePath);
    return;
  }

  // Route: /frames/{participant}/{videoId}/{filename} - serve raw frame images
  if (pathname.startsWith('/frames/')) {
    const parts = pathname.replace('/frames/', '').split('/');
    if (parts.length >= 3) {
      const participant = parts[0];
      const videoId = parts[1];
      const filename = parts.slice(2).join('/');
      const filePath = path.join(HDEPIC_ROOT, 'outputs', '02_inventory', participant, 'hands23_detection', videoId, 'frames', filename);
      serveFile(res, filePath);
      return;
    }
  }

  // Route: /visualizations/{participant}/{videoId}/{filename} - serve HOI visualization images
  if (pathname.startsWith('/visualizations/')) {
    const parts = pathname.replace('/visualizations/', '').split('/');
    if (parts.length >= 3) {
      const participant = parts[0];
      const videoId = parts[1];
      const filename = parts.slice(2).join('/');
      const filePath = path.join(HDEPIC_ROOT, 'outputs', '02_inventory', participant, 'hands23_detection', videoId, 'visualizations', filename);
      serveFile(res, filePath);
      return;
    }
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

function listVlmTags(res, dirPath, participant) {
  if (!fs.existsSync(dirPath)) {
    res.writeHead(404, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: 'Directory not found', tags: [] }));
    return;
  }

  // Find all VLM result files: _vlm_qa_{tag}_results.json and _vlm_frame_{tag}_results.json
  const qaPattern = new RegExp(`^${participant}_vlm_qa_(.+)_results\\.json$`);
  const framePattern = new RegExp(`^${participant}_vlm_frame_(.+)_results\\.json$`);
  const files = fs.readdirSync(dirPath);

  const tags = files
    .map(f => {
      const qaMatch = f.match(qaPattern);
      if (qaMatch) return { tag: qaMatch[1], filename: f };
      const frameMatch = f.match(framePattern);
      if (frameMatch) return { tag: `frame:${frameMatch[1]}`, filename: f };
      return null;
    })
    .filter(Boolean)
    .sort((a, b) => a.tag.localeCompare(b.tag));

  res.writeHead(200, { 'Content-Type': 'application/json' });
  res.end(JSON.stringify({ participant, tags }));
}

function listFailureCases(res, dirPath) {
  if (!fs.existsSync(dirPath)) {
    res.writeHead(404, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: 'Directory not found', files: [] }));
    return;
  }

  // Find all failure_cases*.json files
  const pattern = /^failure_cases_(.+)\.json$/;
  const files = fs.readdirSync(dirPath);

  const failureCases = files
    .map(f => {
      const match = f.match(pattern);
      if (!match) return null;

      // Try to read file to get metadata
      const filePath = path.join(dirPath, f);
      try {
        const content = JSON.parse(fs.readFileSync(filePath, 'utf-8'));
        return {
          filename: `failure_cases/${f}`,
          name: match[1],
          description: content.description || '',
          tag: content.tag || '',
          totalItems: content.total_items || content.items?.length || 0,
          createdAt: content.created_at || null,
          rerunTag: content.rerun_tag || null,
        };
      } catch (err) {
        return { filename: `failure_cases/${f}`, name: match[1], error: err.message };
      }
    })
    .filter(Boolean)
    .sort((a, b) => {
      // Sort by name, with versions in order
      if (a.name === b.name) return 0;
      // Extract base name and version for proper sorting
      const aMatch = a.name.match(/^(.+?)(?:_v(\d+))?$/);
      const bMatch = b.name.match(/^(.+?)(?:_v(\d+))?$/);
      if (aMatch && bMatch && aMatch[1] === bMatch[1]) {
        // Same base name, sort by version
        const aVer = parseInt(aMatch[2] || '1', 10);
        const bVer = parseInt(bMatch[2] || '1', 10);
        return aVer - bVer;
      }
      return a.name.localeCompare(b.name);
    });

  res.writeHead(200, { 'Content-Type': 'application/json' });
  res.end(JSON.stringify({ files: failureCases }));
}

server.listen(PORT, () => {
  console.log(`\n🎬 Video Server running at http://localhost:${PORT}`);
  console.log(`\nAvailable endpoints:`);
  console.log(`  GET  /videos/{participant}/{video_id}.mp4  - Stream video`);
  console.log(`  GET  /data/{path}                          - Serve JSON/CSV files`);
  console.log(`  PUT  /data/{path}                          - Save JSON files`);
  console.log(`  GET  /list/{participant}                   - List available videos`);
  console.log(`  GET  /vlm-tags/{participant}               - List VLM result tags`);
  console.log(`  GET  /failure-cases                        - List failure case files`);
  console.log(`  GET  /hands23/{participant}                - Get hands23 detection results`);
  console.log(`  GET  /frames/{participant}/{videoId}/{filename}        - Serve raw frames`);
  console.log(`  GET  /visualizations/{participant}/{videoId}/{filename} - Serve HOI vis`);
  console.log(`\nExample:`);
  console.log(`  http://localhost:${PORT}/videos/P01/P01-20240202-110250.mp4`);
  console.log(`  http://localhost:${PORT}/list/P01`);
  console.log(`  http://localhost:${PORT}/data/outputs/02_inventory/P01/P01_known_quantities.json`);
  console.log(`  http://localhost:${PORT}/failure-cases`);
  console.log(`  http://localhost:${PORT}/hands23/P03`);
  console.log(`\nPress Ctrl+C to stop\n`);
});

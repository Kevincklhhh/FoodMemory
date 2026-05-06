const express = require('express');
const cors = require('cors');
const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

const app = express();
const PORT = 4002;

const SELF_DATA = path.resolve(__dirname, '..');
const PARTICIPANTS_DIR = path.join(SELF_DATA, 'participants');

function participantDir(participant) {
  return path.join(PARTICIPANTS_DIR, participant);
}

app.use(cors());
app.use(express.json());

// List participants
app.get('/api/participants', (req, res) => {
  if (!fs.existsSync(PARTICIPANTS_DIR)) return res.json([]);
  const participants = fs.readdirSync(PARTICIPANTS_DIR)
    .filter(d => fs.statSync(path.join(PARTICIPANTS_DIR, d)).isDirectory())
    .sort();
  res.json(participants);
});

// List sessions for a participant
app.get('/api/participants/:participant/sessions', (req, res) => {
  const videosDir = path.join(participantDir(req.params.participant), 'videos');
  if (!fs.existsSync(videosDir)) return res.json([]);
  const sessions = fs.readdirSync(videosDir)
    .filter(d => fs.statSync(path.join(videosDir, d)).isDirectory())
    .sort();
  res.json(sessions);
});

// Get clip metadata for a session
app.get('/api/participants/:participant/sessions/:session/clips', (req, res) => {
  const sessionDir = path.join(participantDir(req.params.participant), 'videos', req.params.session);
  if (!fs.existsSync(sessionDir)) return res.status(404).json({ error: 'Session not found' });

  const clips = fs.readdirSync(sessionDir)
    .filter(f => f.endsWith('.mp4'))
    .sort()
    .map(filename => {
      const filepath = path.join(sessionDir, filename);
      try {
        const dur = execSync(
          `ffprobe -v quiet -show_entries format=duration -of csv=p=0 "${filepath}"`,
          { encoding: 'utf-8' }
        ).trim();
        return { filename, duration: parseFloat(dur) };
      } catch {
        return { filename, duration: 0 };
      }
    });

  res.json(clips);
});

// Stream video with range request support
app.get('/videos/:participant/:session/:filename', (req, res) => {
  const filepath = path.join(participantDir(req.params.participant), 'videos', req.params.session, req.params.filename);
  if (!fs.existsSync(filepath)) return res.status(404).send('Not found');

  const stat = fs.statSync(filepath);
  const fileSize = stat.size;
  const range = req.headers.range;

  if (range) {
    const parts = range.replace(/bytes=/, '').split('-');
    const start = parseInt(parts[0], 10);
    const end = parts[1] ? parseInt(parts[1], 10) : fileSize - 1;
    const chunkSize = end - start + 1;
    const file = fs.createReadStream(filepath, { start, end });
    res.writeHead(206, {
      'Content-Range': `bytes ${start}-${end}/${fileSize}`,
      'Accept-Ranges': 'bytes',
      'Content-Length': chunkSize,
      'Content-Type': 'video/mp4',
    });
    file.pipe(res);
  } else {
    res.writeHead(200, {
      'Content-Length': fileSize,
      'Content-Type': 'video/mp4',
    });
    fs.createReadStream(filepath).pipe(res);
  }
});

// Get ledger for a participant
app.get('/api/participants/:participant/ledger', (req, res) => {
  const ledgerPath = path.join(participantDir(req.params.participant), 'ledger.json');
  if (!fs.existsSync(ledgerPath)) return res.json({ items: {}, events: [], snapshots: {} });
  res.json(JSON.parse(fs.readFileSync(ledgerPath, 'utf-8')));
});

// Get actions for a session
app.get('/api/participants/:participant/sessions/:session/actions', (req, res) => {
  const actionsPath = path.join(participantDir(req.params.participant), 'annotations', req.params.session, 'actions.json');
  if (!fs.existsSync(actionsPath)) return res.json([]);
  res.json(JSON.parse(fs.readFileSync(actionsPath, 'utf-8')));
});

// Save actions for a session
app.put('/api/participants/:participant/sessions/:session/actions', (req, res) => {
  const annDir = path.join(participantDir(req.params.participant), 'annotations', req.params.session);
  if (!fs.existsSync(annDir)) fs.mkdirSync(annDir, { recursive: true });
  const actionsPath = path.join(annDir, 'actions.json');
  fs.writeFileSync(actionsPath, JSON.stringify(req.body, null, 2) + '\n');
  res.json({ ok: true, count: req.body.length });
});

// List available VLM result tags for a session
app.get('/api/participants/:participant/sessions/:session/vlm-results', (req, res) => {
  const outputDir = path.join(participantDir(req.params.participant), 'outputs', req.params.session);
  if (!fs.existsSync(outputDir)) return res.json([]);
  const files = fs.readdirSync(outputDir)
    .filter(f => f.startsWith('vlm_') && f.endsWith('_results.json'))
    .sort();
  // Return tag name + summary for each
  const results = files.map(f => {
    const tag = f.replace('vlm_', '').replace('_results.json', '');
    try {
      const data = JSON.parse(fs.readFileSync(path.join(outputDir, f), 'utf-8'));
      return {
        tag,
        filename: f,
        method: data.method || '',
        num_detections: data.num_detections || 0,
        total_tokens: data.token_stats?.total_tokens || 0,
        block_duration: data.block_duration,
      };
    } catch {
      return { tag, filename: f };
    }
  });
  res.json(results);
});

// Get VLM result data by tag
app.get('/api/participants/:participant/sessions/:session/vlm-results/:tag', (req, res) => {
  const filepath = path.join(
    participantDir(req.params.participant), 'outputs', req.params.session,
    `vlm_${req.params.tag}_results.json`
  );
  if (!fs.existsSync(filepath)) return res.status(404).json({ error: 'Result not found' });
  res.json(JSON.parse(fs.readFileSync(filepath, 'utf-8')));
});

// List available VLM action result tags for a session
app.get('/api/participants/:participant/sessions/:session/vlm-actions', (req, res) => {
  const outputDir = path.join(participantDir(req.params.participant), 'annotations', req.params.session);
  if (!fs.existsSync(outputDir)) return res.json([]);
  const files = fs.readdirSync(outputDir)
    .filter(f => f.startsWith('vlm_') && f.endsWith('_actions.json'))
    .sort();
  const results = files.map(f => {
    const tag = f.replace('vlm_', '').replace('_actions.json', '');
    try {
      const data = JSON.parse(fs.readFileSync(path.join(outputDir, f), 'utf-8'));
      return {
        tag,
        filename: f,
        model: data.model || '',
        num_actions: data.num_actions || 0,
        block_duration: data.block_duration,
      };
    } catch {
      return { tag, filename: f };
    }
  });
  res.json(results);
});

// Get VLM action results by tag
app.get('/api/participants/:participant/sessions/:session/vlm-actions/:tag', (req, res) => {
  const filepath = path.join(
    participantDir(req.params.participant), 'annotations', req.params.session,
    `vlm_${req.params.tag}_actions.json`
  );
  if (!fs.existsSync(filepath)) return res.status(404).json({ error: 'Result not found' });
  res.json(JSON.parse(fs.readFileSync(filepath, 'utf-8')));
});

// Get TAD (Temporal Action Detection) proposals for a session
// Loads adatad_detections.json, maps per-video timestamps to session-level, unions noun+verb
const TAD_RESULTS_DIR = path.resolve(__dirname, '..', 'models', 'OpenTAD', 'data', 'self_data', 'results');

app.get('/api/participants/:participant/sessions/:session/tad-detections', (req, res) => {
  // Look for per-session results first, then fall back to legacy global file
  const perSessionFile = path.join(participantDir(req.params.participant), 'outputs', req.params.session, 'adatad_detections.json');
  const legacyFile = path.join(TAD_RESULTS_DIR, 'adatad_detections.json');
  const detFile = fs.existsSync(perSessionFile) ? perSessionFile : legacyFile;
  if (!fs.existsSync(detFile)) return res.status(404).json({ error: 'No TAD detections found' });

  // Get clip durations to build offset map
  const sessionDir = path.join(participantDir(req.params.participant), 'videos', req.params.session);
  if (!fs.existsSync(sessionDir)) return res.status(404).json({ error: 'Session not found' });

  const clipFiles = fs.readdirSync(sessionDir).filter(f => f.endsWith('.mp4')).sort();
  const clipOffsets = {};
  let offset = 0;
  for (const f of clipFiles) {
    const videoId = f.replace('.mp4', '');
    try {
      const dur = parseFloat(execSync(
        `ffprobe -v quiet -show_entries format=duration -of csv=p=0 "${path.join(sessionDir, f)}"`,
        { encoding: 'utf-8' }
      ).trim());
      clipOffsets[videoId] = { offset, duration: dur };
      offset += dur;
    } catch {
      clipOffsets[videoId] = { offset, duration: 0 };
    }
  }

  const data = JSON.parse(fs.readFileSync(detFile, 'utf-8'));
  const segments = [];

  for (const category of ['noun', 'verb']) {
    if (!data[category]) continue;
    for (const videoId of Object.keys(data[category])) {
      if (!(videoId in clipOffsets)) continue;
      const off = clipOffsets[videoId].offset;
      for (const det of data[category][videoId]) {
        segments.push({
          start: det.segment[0] + off,
          end: det.segment[1] + off,
          score: det.score,
          label: det.label,
          type: category,
        });
      }
    }
  }

  res.json({ total: segments.length, segments });
});

// Get merged hands23 + siglip results for a session
app.get('/api/participants/:participant/sessions/:session/hands23-results', (req, res) => {
  const h23Dir = path.join(participantDir(req.params.participant), 'outputs', req.params.session, 'hands23_detection');
  if (!fs.existsSync(h23Dir)) return res.status(404).json({ error: 'No hands23 data' });

  // Find hands23 results file
  const h23Files = fs.readdirSync(h23Dir).filter(f => f.endsWith('_hands23_results.json'));
  if (h23Files.length === 0) return res.status(404).json({ error: 'No hands23 results' });
  const h23Data = JSON.parse(fs.readFileSync(path.join(h23Dir, h23Files[0]), 'utf-8'));

  // Find siglip matches file (optional)
  const slipFiles = fs.readdirSync(h23Dir).filter(f => f.endsWith('_siglip_matches.json'));
  let siglipByKey = {};
  let foodItems = [];
  if (slipFiles.length > 0) {
    const slipData = JSON.parse(fs.readFileSync(path.join(h23Dir, slipFiles[0]), 'utf-8'));
    foodItems = slipData.food_items || [];
    for (const vid of (slipData.videos || [])) {
      for (const m of (vid.matches || [])) {
        const key = `${vid.video_id}|${m.timestamp}`;
        if (!siglipByKey[key]) siglipByKey[key] = [];
        siglipByKey[key].push(m);
      }
    }
  }

  // Find dino matches file (optional)
  const dinoFiles = fs.readdirSync(h23Dir).filter(f => f.endsWith('_dino_matches.json'));
  let dinoByKey = {};
  let dinoRefItems = [];
  if (dinoFiles.length > 0) {
    const dinoData = JSON.parse(fs.readFileSync(path.join(h23Dir, dinoFiles[0]), 'utf-8'));
    dinoRefItems = dinoData.reference_items || [];
    for (const vid of (dinoData.videos || [])) {
      for (const m of (vid.matches || [])) {
        const key = `${vid.video_id}|${m.timestamp}`;
        if (!dinoByKey[key]) dinoByKey[key] = [];
        dinoByKey[key].push(m);
      }
    }
  }

  // Flatten and merge
  const frames = [];
  for (const video of (h23Data.videos || [])) {
    for (const frame of (video.frames || [])) {
      const key = `${video.video_id}|${frame.session_timestamp_s}`;
      frames.push({
        session_timestamp_s: frame.session_timestamp_s,
        video_id: video.video_id,
        clip_timestamp_s: frame.clip_timestamp_s,
        frame_path: frame.frame_path,
        visualization_path: frame.visualization_path || null,
        num_hands: frame.num_hands,
        detections: frame.detections,
        siglip_matches: siglipByKey[key] || [],
        dino_matches: dinoByKey[key] || [],
      });
    }
  }
  frames.sort((a, b) => a.session_timestamp_s - b.session_timestamp_s);

  res.json({
    total_frames: h23Data.total_frames,
    total_hands: h23Data.total_hands,
    fps: h23Data.fps,
    food_items: foodItems,
    dino_reference_items: dinoRefItems,
    frames,
  });
});

// Serve hands23 frame/visualization images
// Route: /hands23/:participant/:session/:videoId/:subdir/:filename
app.get('/hands23/:participant/:session/:videoId/:subdir/:filename', (req, res) => {
  const filepath = path.join(
    participantDir(req.params.participant), 'outputs', req.params.session,
    'hands23_detection', req.params.videoId, req.params.subdir, req.params.filename
  );
  if (!fs.existsSync(filepath)) return res.status(404).send('Not found');
  res.sendFile(filepath);
});

// ---- Amount Estimation Endpoints ----
//
// Data-driven pipeline registry. Each entry defines how to discover prediction
// files on disk and map them to a display tag, pipeline type, cache directory,
// and associated eval/planner files. Adding a new pipeline only requires a new
// entry here — no if/else changes in endpoints.

// Unified naming: {prefix}{model}_{tag}_{filetype}.json
// All pipelines use the same pattern: /^{prefix}(.+)_preds\.json$/
// The captured group is always {model}_{tag}.

const CACHE_DIRS = {
  upperbound:          path.resolve(__dirname, '..', 'cache', 'upperbound'),
  upperbound_video:    path.resolve(__dirname, '..', 'cache', 'upperbound_video'),
  upperbound_remaining:path.resolve(__dirname, '..', 'cache', 'upperbound_remaining'),
  lowerbound:          path.resolve(__dirname, '..', 'cache', 'lowerbound'),
  lowerbound_remaining:path.resolve(__dirname, '..', 'cache', 'lowerbound_remaining'),
  avp:                 path.resolve(__dirname, '..', 'cache', 'avp'),
  avp_remaining:       path.resolve(__dirname, '..', 'cache', 'avp_remaining'),
  avp_remaining_noTAD: path.resolve(__dirname, '..', 'cache', 'avp_remaining_noTAD'),
  avp_remaining_noplanner: path.resolve(__dirname, '..', 'cache', 'avp_remaining_noplanner'),
  avp_remaining_CandList_HOI: path.resolve(__dirname, '..', 'cache', 'avp_remaining_CandList_HOI'),
  avp_remaining_CandList_HOI_PerItem: path.resolve(__dirname, '..', 'cache', 'avp_remaining_CandList_HOI_PerItem'),
  avp_remaining_Iterative: path.resolve(__dirname, '..', 'cache', 'avp_remaining_Iterative'),
  avp_remaining_minimal: path.resolve(__dirname, '..', 'cache', 'avp_remaining_minimal'),
};

// More specific prefixes must come before shorter ones (e.g. upperbound_video_
// before upperbound_, avp_remaining_ before avp_) so the regex matches correctly.
const PIPELINE_DEFS = [
  {
    pipeline: 'upperbound',
    filePattern: /^upperbound_video_(.+)_preds\.json$/,
    tagPrefix: 'ubv_',
    predsTemplate:    (t) => `upperbound_video_${t}_preds.json`,
    evalTemplate:     (t) => `upperbound_video_${t}_preds_eval.json`,
    cacheKey: 'upperbound_video',
  },
  {
    pipeline: 'upperbound',
    filePattern: /^upperbound_remaining_(.+)_preds\.json$/,
    tagPrefix: 'ubr_',
    predsTemplate:    (t) => `upperbound_remaining_${t}_preds.json`,
    evalTemplate:     (t) => `upperbound_remaining_${t}_preds_eval.json`,
    cacheKey: 'upperbound_remaining',
  },
  {
    pipeline: 'upperbound',
    filePattern: /^upperbound_(.+)_preds\.json$/,
    tagPrefix: 'ub_',
    predsTemplate:    (t) => `upperbound_${t}_preds.json`,
    evalTemplate:     (t) => `upperbound_${t}_preds_eval.json`,
    cacheKey: 'upperbound',
  },
  {
    pipeline: 'lowerbound',
    filePattern: /^lowerbound_remaining_(.+)_preds\.json$/,
    tagPrefix: 'lbr_',
    predsTemplate:    (t) => `lowerbound_remaining_${t}_preds.json`,
    evalTemplate:     (t) => `lowerbound_remaining_${t}_preds_eval.json`,
    cacheKey: 'lowerbound_remaining',
  },
  {
    pipeline: 'lowerbound',
    filePattern: /^lowerbound_(.+)_preds\.json$/,
    tagPrefix: 'lb_',
    predsTemplate:    (t) => `lowerbound_${t}_preds.json`,
    evalTemplate:     (t) => `lowerbound_${t}_preds_eval.json`,
    cacheKey: 'lowerbound',
  },
  {
    // avp_Iterative_remaining_ — multi-round plan-observe loop (batched
    // windows per round, multi-segment per window). pipeline='iterative'
    // triggers the dedicated AmountViewIterative panel which renders the
    // per-round planner/observer trace.
    pipeline: 'iterative',
    filePattern: /^avp_Iterative_remaining_(.+)_preds\.json$/,
    tagPrefix: 'iter_',
    predsTemplate:   (t) => `avp_Iterative_remaining_${t}_preds.json`,
    evalTemplate:    (t) => `avp_Iterative_remaining_${t}_preds_eval.json`,
    plannerTemplate: (t) => `avp_Iterative_remaining_${t}_planner.json`,
    cacheKey: 'avp_remaining_Iterative',
  },
  {
    // avp_minimal_remaining_ — sweep-only branch (06_avp_round1_remaining_minimal.py).
    // Per session: planner emits sweep_segments + per-round R2/R3 dispensal-window
    // replans on still-derivative items. We adapt to pipeline='iterative' so the
    // existing AmountView/IterativeTrace renders the per-round windows and items
    // without UI changes. The observer_rounds[] log expected by the iterative
    // branch is synthesized at read time from sessionLog.sweep / sweep_r2 / rounds[].
    pipeline: 'iterative',
    filePattern: /^avp_minimal_remaining_(.+)_preds\.json$/,
    tagPrefix: 'sweep_',
    predsTemplate:   (t) => `avp_minimal_remaining_${t}_preds.json`,
    evalTemplate:    (t) => `avp_minimal_remaining_${t}_preds_eval.json`,
    plannerTemplate: (t) => `avp_minimal_remaining_${t}_planner.json`,
    cacheKey: 'avp_remaining_minimal',
  },
  {
    // avp_CandList_HOI_PerItem_remaining_ — PerItem branch (per-item
    // evidence + journey-aware sampling). Must come before other avp_* defs.
    pipeline: 'avp',
    filePattern: /^avp_CandList_HOI_PerItem_remaining_(.+)_preds\.json$/,
    tagPrefix: 'avpcpi_',
    predsTemplate:   (t) => `avp_CandList_HOI_PerItem_remaining_${t}_preds.json`,
    evalTemplate:    (t) => `avp_CandList_HOI_PerItem_remaining_${t}_preds_eval.json`,
    plannerTemplate: (t) => `avp_CandList_HOI_PerItem_remaining_${t}_planner.json`,
    cacheKey: 'avp_remaining_CandList_HOI_PerItem',
  },
  {
    // avp_CandList_HOI_remaining_ — CandList_HOI baseline (per-frame, per-hand blocks).
    pipeline: 'avp',
    filePattern: /^avp_CandList_HOI_remaining_(.+)_preds\.json$/,
    tagPrefix: 'avpch_',
    predsTemplate:   (t) => `avp_CandList_HOI_remaining_${t}_preds.json`,
    evalTemplate:    (t) => `avp_CandList_HOI_remaining_${t}_preds_eval.json`,
    plannerTemplate: (t) => `avp_CandList_HOI_remaining_${t}_planner.json`,
    cacheKey: 'avp_remaining_CandList_HOI',
  },
  {
    // avp_noplanner_remaining_ must come before avp_remaining_ and avp_
    pipeline: 'avp',
    filePattern: /^avp_noplanner_remaining_(.+)_preds\.json$/,
    tagPrefix: 'avpnp_',
    predsTemplate:    (t) => `avp_noplanner_remaining_${t}_preds.json`,
    evalTemplate:     (t) => `avp_noplanner_remaining_${t}_preds_eval.json`,
    plannerTemplate:  (t) => `avp_noplanner_remaining_${t}_planner.json`,
    cacheKey: 'avp_remaining_noplanner',
  },
  {
    // avp_noTAD_remaining_ must come before avp_remaining_ and avp_
    pipeline: 'avp',
    filePattern: /^avp_noTAD_remaining_(.+)_preds\.json$/,
    tagPrefix: 'avpnt_',
    predsTemplate:    (t) => `avp_noTAD_remaining_${t}_preds.json`,
    evalTemplate:     (t) => `avp_noTAD_remaining_${t}_preds_eval.json`,
    plannerTemplate:  (t) => `avp_noTAD_remaining_${t}_planner.json`,
    cacheKey: 'avp_remaining_noTAD',
  },
  {
    // avp_remaining_ must come before avp_ (more specific prefix first)
    pipeline: 'avp',
    filePattern: /^avp_remaining_(.+)_preds\.json$/,
    tagPrefix: 'avpr_',
    predsTemplate:    (t) => `avp_remaining_${t}_preds.json`,
    evalTemplate:     (t) => `avp_remaining_${t}_preds_eval.json`,
    plannerTemplate:  (t) => `avp_remaining_${t}_planner.json`,
    cacheKey: 'avp_remaining',
  },
  {
    pipeline: 'avp',
    filePattern: /^avp_(.+)_preds\.json$/,
    tagPrefix: 'avp_',
    predsTemplate:    (t) => `avp_${t}_preds.json`,
    evalTemplate:     (t) => `avp_${t}_preds_eval.json`,
    plannerTemplate:  (t) => `avp_${t}_planner.json`,
    cacheKey: 'avp',
  },
];

// Scan participant outputs once and return structured registry entries.
function buildTagRegistry(participant) {
  const outputDir = path.join(participantDir(participant), 'outputs');
  if (!fs.existsSync(outputDir)) return [];
  const files = fs.readdirSync(outputDir).sort();
  const tags = [];

  for (const f of files) {
    for (const def of PIPELINE_DEFS) {
      if (def.skipIf && def.skipIf(f)) continue;
      const m = f.match(def.filePattern);
      if (!m) continue;
      const rawTag = def.fixedRawTag || m[1];
      tags.push({
        tag:            def.tagPrefix + rawTag,
        pipeline:       def.pipeline,
        predsFile:       path.join(outputDir, def.predsTemplate(rawTag)),
        evalFile:        path.join(outputDir, def.evalTemplate(rawTag)),
        plannerFile:     def.plannerTemplate ? path.join(outputDir, def.plannerTemplate(rawTag)) : null,
        cacheDir:        CACHE_DIRS[def.cacheKey],
        cacheLookupTag:  rawTag,
      });
      break; // first matching def wins
    }
  }
  return tags;
}

// ---------------------------------------------------------------------------
// Cache helpers
// ---------------------------------------------------------------------------

// Hardcoded in upperbound_amount.py and 06_avp_round1.py — keep in sync.
// Naming convention: {prefix}{model}_{tag}_{filetype}.json
const VLM_FRAME_PADDING_S = 2.0;
const FRAME_GAP_THRESHOLD_S = 1.6;

// Cache writers lay out files as {sessionDir}/{model_tag}/{run_tag}/{filename}.
// The UI passes a flat tag like "gemini-2.5-pro_noisyprior_v1". Walk subdirs,
// join {model}_{run} with "_", return the path that matches flatTag.
function findTaggedLog(sessionDir, flatTag, filename) {
  if (!fs.existsSync(sessionDir)) return null;
  let entries;
  try { entries = fs.readdirSync(sessionDir, { withFileTypes: true }); }
  catch (e) { return null; }
  for (const e of entries) {
    if (!e.isDirectory()) continue;
    const modelDir = path.join(sessionDir, e.name);
    let runEntries;
    try { runEntries = fs.readdirSync(modelDir, { withFileTypes: true }); }
    catch (err) { continue; }
    for (const r of runEntries) {
      if (!r.isDirectory()) continue;
      if (`${e.name}_${r.name}` === flatTag) {
        const candidate = path.join(modelDir, r.name, filename);
        if (fs.existsSync(candidate)) return candidate;
      }
    }
  }
  const legacy = path.join(sessionDir, filename);
  return fs.existsSync(legacy) ? legacy : null;
}

function findTaggedDir(sessionDir, flatTag) {
  if (!fs.existsSync(sessionDir)) return null;
  let entries;
  try { entries = fs.readdirSync(sessionDir, { withFileTypes: true }); }
  catch (e) { return null; }
  for (const e of entries) {
    if (!e.isDirectory()) continue;
    const modelDir = path.join(sessionDir, e.name);
    let runEntries;
    try { runEntries = fs.readdirSync(modelDir, { withFileTypes: true }); }
    catch (err) { continue; }
    for (const r of runEntries) {
      if (!r.isDirectory()) continue;
      if (`${e.name}_${r.name}` === flatTag) return path.join(modelDir, r.name);
    }
  }
  return sessionDir;
}

function clusterFramesToBands(timestamps) {
  if (!timestamps || timestamps.length === 0) return [];
  const sorted = timestamps.slice().sort((a, b) => a - b);
  const bands = [];
  let bandStart = sorted[0], bandEnd = sorted[0];
  for (let i = 1; i < sorted.length; i++) {
    if (sorted[i] - bandEnd > FRAME_GAP_THRESHOLD_S) {
      bands.push({ start: bandStart, end: bandEnd });
      bandStart = sorted[i];
    }
    bandEnd = sorted[i];
  }
  bands.push({ start: bandStart, end: bandEnd });
  return bands;
}

function mergeBands(bands) {
  if (!bands || bands.length === 0) return [];
  const sorted = bands.slice().sort((a, b) => a.start - b.start);
  const out = [sorted[0]];
  for (let i = 1; i < sorted.length; i++) {
    const last = out[out.length - 1];
    if (sorted[i].start <= last.end) {
      last.end = Math.max(last.end, sorted[i].end);
    } else {
      out.push(sorted[i]);
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// Sweep (avp_minimal_remaining) → observer_rounds shim
// ---------------------------------------------------------------------------
// The sweep-only branch (06_avp_round1_remaining_minimal.py) writes a
// session_log shaped:
//   { planner: {...}, sweep: { sweep_segments, items, ... },
//     planner_r2: {...}, sweep_r2: { sweep_segments, items, ... },
//     rounds: [ { planner: {...}, sweep: {...} }, ... ] }
// The iterative-pipeline AmountView reads sessionLog.observer_rounds[].
// Convert sweep rounds into that same shape so the existing rendering works.

function _sweepRoundToObserverRound(roundN, sweepBlock, plannerBlock) {
  // The new journey/dense schema stores `unified_segments` (interleaved
  // journey + dense, sorted by start time). Older sweep runs stored a flat
  // `sweep_segments` list — fall back to that for backward compat.
  const segs = (sweepBlock?.unified_segments) || (sweepBlock?.sweep_segments) || [];
  const items = (sweepBlock?.items) || [];
  const pso  = (sweepBlock?.per_segment_observations) || [];
  // Iterative window.segments is [[start, end], ...]; carry target_items +
  // per-segment observations as siblings for downstream UI.
  const segPairs = segs.map(s => [s.start, s.end]);
  const targetItemsPerSegment = segs.map(s => s.target_items || []);
  const segKinds = segs.map(s => s.kind || 'dense');
  // Map segment_idx (1-based) -> observation text.
  const obsByIdx = {};
  for (const e of pso) {
    if (!e || e.segment_idx == null) continue;
    obsByIdx[Number(e.segment_idx)] = String(e.observation || '');
  }
  const perSegmentObs = segs.map((_, i) => obsByIdx[i + 1] || '');
  // Build a human-readable window_observation by joining per-segment notes.
  const windowObsLines = segs.map((s, i) => {
    const note = perSegmentObs[i] || '(no observation)';
    const kindTag = segKinds[i] === 'journey' ? '[journey]' : '[dense]';
    const tStr = (s.end !== s.start)
      ? `t=${s.start.toFixed(1)}-${s.end.toFixed(1)}s`
      : `t=${s.start.toFixed(1)}s`;
    return `Seg${i + 1} ${kindTag} [${tStr}]: ${note}`;
  });
  const windowObservation = windowObsLines.join('\n');
  // Per-instance entries: one per sweep item return. Carry the new
  // independent-amount triple + amount_kind selector through so the UI can
  // surface them.
  const perInstance = items.map(it => ({
    instance_id:        it.instance_id,
    visual_class:       it.visual_class,
    reasoning:          it.reasoning || '',
    evidence_frames:    it.evidence_frames || [],
    handling_status:    it.status || null,
    remaining:          it.amount_remaining ?? null,
    amount_used:        it.amount_derivative ?? null,
    amount_starting:    it.amount_starting ?? null,
    amount_remaining:   it.amount_remaining ?? null,
    amount_derivative:  it.amount_derivative ?? null,
  }));
  return {
    round: roundN,
    window: {
      segments:                  segPairs,
      target_items:              targetItemsPerSegment,
      per_segment_observations:  perSegmentObs,
      confidence:                '',
      visual_class:              '',
    },
    per_instance:                perInstance,
    window_observation:          windowObservation,
    raw_response:                sweepBlock?.raw_response || '',
    prompt:                      sweepBlock?.prompt || '',
    planner_raw_response:        plannerBlock?.raw_response || '',
  };
}

function synthesizeSweepObserverRounds(sessionLog) {
  if (!sessionLog || sessionLog.observer_rounds) return null; // already shaped
  if (!sessionLog.sweep && !sessionLog.sweep_r2) return null; // not sweep
  const out = [];
  if (sessionLog.sweep) {
    out.push(_sweepRoundToObserverRound(1, sessionLog.sweep, sessionLog.planner));
  }
  if (sessionLog.sweep_r2) {
    out.push(_sweepRoundToObserverRound(2, sessionLog.sweep_r2, sessionLog.planner_r2));
  }
  // Round 3+ live under sessionLog.rounds[]; each entry has {planner, sweep}.
  for (const r of (sessionLog.rounds || [])) {
    const sw = r.sweep;
    if (!sw) continue;
    // Determine round number from sw.stats.round (set by run_sweep_observer_round_n)
    // or by index offset (R3 = first entry, R4 = second, ...).
    const rn = sw.stats?.round || (3 + out.length - 2);
    out.push(_sweepRoundToObserverRound(rn, sw, r.planner));
  }
  return out;
}

// Read the per-session planner JSON for an iterative/sweep tag and, if it's
// a sweep run, mutate sessionLog so observer_rounds[] is populated. Returns
// the (possibly-mutated) sessionLog or null on failure.
function loadSessionLogWithSweepShim(entry, participant, session) {
  const sessPlannerPath = path.join(
    participantDir(participant), 'outputs', session,
    path.basename(entry.plannerFile || '')
  );
  if (!entry.plannerFile || !fs.existsSync(sessPlannerPath)) return null;
  let sessionLog;
  try {
    const sd = JSON.parse(fs.readFileSync(sessPlannerPath, 'utf-8'));
    sessionLog = sd.session || sd;
  } catch (e) { return null; }
  if (!sessionLog) return null;
  if (!sessionLog.observer_rounds) {
    const synth = synthesizeSweepObserverRounds(sessionLog);
    if (synth) sessionLog.observer_rounds = synth;
  }
  return sessionLog;
}

// ---------------------------------------------------------------------------
// Prediction normalization — unify field names across pipelines
// ---------------------------------------------------------------------------

function normalizePrediction(pred) {
  // Segments: AVP uses [start, end] arrays; normalize to {start, end[, round]} objects
  const segments = (pred.segments || []).map(seg => {
    if (Array.isArray(seg)) {
      const out = { start: seg[0], end: seg[1] };
      if (seg.length >= 3 && seg[2] != null) out.round = seg[2];
      return out;
    }
    return seg;
  });

  return {
    session:              pred.session,
    instance_id:          pred.instance_id,
    item:                 pred.item,
    amount_used:          pred.amount_used ?? pred.amount_derivative ?? null,
    amount_remaining:     pred.amount_remaining ?? null,
    // Sweep schema (06_avp_round1_remaining_minimal_r2free*.py): independent
    // start/remain/derivative triple + the round 1/2 frame citations the
    // observer attaches to start/remain. Frame-citations are seconds in
    // session-cumulative time and may be null when the observer didn't
    // ground the value in a single frame (e.g. computed_remaining).
    amount_starting:          pred.amount_starting ?? null,
    amount_derivative:        pred.amount_derivative ?? null,
    amount_kind:              pred.amount_kind ?? null,
    amount_starting_frame:    pred.amount_starting_frame ?? null,
    amount_remaining_frame:   pred.amount_remaining_frame ?? null,
    amount_derivative_frame:  pred.amount_derivative_frame ?? null,
    // Sweep r2free evframe schema: per-amount evidence strings of form
    // `t=<sec>s` or `t=<lo>s..<hi>s` plus optional ≤6-word caption.
    // Non-null iff the corresponding amount is non-null. The annotator's
    // parseEvidence helper extracts the first timestamp for the seek target.
    amount_starting_evidence:    pred.amount_starting_evidence ?? null,
    amount_remaining_evidence:   pred.amount_remaining_evidence ?? null,
    amount_derivative_evidence:  pred.amount_derivative_evidence ?? null,
    // Unified: AVP calls them evidence_frames, others use evidence_timestamps
    evidence_timestamps:  pred.evidence_timestamps || pred.evidence_frames || [],
    segments,
    // Per-item thinking (baseline has this; AVP/wholevideo do not)
    thinking:             pred.thinking || null,
    // Per-item reasoning (AVP observer output; baseline VLM reasoning)
    reasoning:            pred.reasoning || null,
    // AVP-specific planner fields (null for others)
    planner_reasoning:    pred.planner_reasoning || null,
    planner_confidence:   pred.planner_confidence || null,
    // Iterative-specific: list of round numbers the observer touched this iid.
    rounds_touched:       pred.rounds_touched || null,
    // Iterative-specific: observer handling_status verdict (handled / not_visible / ...)
    handling_status:      pred.handling_status || null,
    // Stats (flatten AVP's nested {planner, observer} structure)
    stats:                flattenStats(pred.stats),
  };
}

function flattenStats(stats) {
  if (!stats) return null;
  // AVP has nested { planner: {...}, observer: {...} }
  if (stats.observer) {
    return {
      inference_time_s:         stats.observer.inference_time_s || null,
      input_tokens:             stats.observer.input_tokens || null,
      num_frames:               stats.observer.num_frames || null,
      planner_inference_time_s: stats.planner?.inference_time_s || null,
    };
  }
  // Baseline/wholevideo: already flat
  return stats;
}

// ---------------------------------------------------------------------------
// Session-level reasoning (unified across pipelines)
// ---------------------------------------------------------------------------

function loadSessionReasoning(entry, participant, session) {
  const result = {
    thinking: null, prompt: null, activity_summary: null, stats: null,
    raw_blocks: [],
  };

  if (entry.pipeline === 'wholevideo') {
    const sessionDir = path.join(entry.cacheDir, participant, session);
    const logPath = findTaggedLog(sessionDir, entry.cacheLookupTag, 'session_log.json');
    if (logPath) {
      try {
        const log = JSON.parse(fs.readFileSync(logPath, 'utf-8'));
        result.thinking = log.thinking || null;
        result.prompt = log.prompt || null;
        result.stats = log.stats || null;
        if (log.prompt) result.raw_blocks.push({ label: 'Prompt', text: log.prompt });
        if (log.thinking) result.raw_blocks.push({ label: 'Response', text: log.thinking });
      } catch (e) { /* ignore */ }
    }
  } else if ((entry.pipeline === 'avp' || entry.pipeline === 'iterative') && entry.plannerFile) {
    // Prefer per-session planner file (always current); fall back to aggregated
    // (which may be stale if the script was interrupted before rewriting it).
    const sessPlannerPath = path.join(
      participantDir(participant), 'outputs', session,
      path.basename(entry.plannerFile)
    );
    let sessLog = null;
    if (fs.existsSync(sessPlannerPath)) {
      try {
        const sd = JSON.parse(fs.readFileSync(sessPlannerPath, 'utf-8'));
        sessLog = sd.session || sd || null;
      } catch (e) { /* ignore */ }
    }
    if (!sessLog && fs.existsSync(entry.plannerFile)) {
      try {
        const plannerData = JSON.parse(fs.readFileSync(entry.plannerFile, 'utf-8'));
        sessLog = (plannerData.sessions || []).find(s => s.session === session) || null;
      } catch (e) { /* ignore */ }
    }
    const sessPlanner = sessLog?.planner || null;
    if (sessPlanner) {
      result.prompt = sessPlanner.prompt || null;
      result.activity_summary = sessPlanner.activity_summary || null;
      result.stats = sessPlanner.stats || null;
    }
    if (sessLog) {
      const blocks = [
        ['R1 Planner Prompt',   sessLog.planner?.prompt],
        ['R1 Planner Response', sessLog.planner?.raw_response],
        ['R1 Observer Prompt',  sessLog.sweep?.prompt],
        ['R1 Observer Response',sessLog.sweep?.raw_response],
        ['R2 Planner Prompt',   sessLog.planner_r2?.prompt],
        ['R2 Planner Response', sessLog.planner_r2?.raw_response],
        ['R2 Observer Prompt',  sessLog.sweep_r2?.prompt],
        ['R2 Observer Response',sessLog.sweep_r2?.raw_response],
      ];
      for (const [label, text] of blocks) {
        if (text) result.raw_blocks.push({ label, text });
      }
      // Iterative rounds ≥3
      const rounds = sessLog.rounds || [];
      for (let i = 0; i < rounds.length; i++) {
        const r = rounds[i];
        const rn = r?.sweep?.stats?.round || (3 + i);
        if (r?.planner?.prompt)        result.raw_blocks.push({ label: `R${rn} Planner Prompt`,   text: r.planner.prompt });
        if (r?.planner?.raw_response)  result.raw_blocks.push({ label: `R${rn} Planner Response`, text: r.planner.raw_response });
        if (r?.sweep?.prompt)          result.raw_blocks.push({ label: `R${rn} Observer Prompt`,  text: r.sweep.prompt });
        if (r?.sweep?.raw_response)    result.raw_blocks.push({ label: `R${rn} Observer Response`,text: r.sweep.raw_response });
      }
    }
  }
  // baseline: no session-level reasoning (per-item only via item log)
  return result;
}

// ---------------------------------------------------------------------------
// VLM input segment reconstruction
// ---------------------------------------------------------------------------

function buildVlmInputSegments(entry, predictions, participant, session) {
  const out = {};
  if (entry.pipeline === 'wholevideo') return out;

  if (entry.pipeline === 'baseline') {
    const sessionDir = path.join(entry.cacheDir, participant, session);
    const taggedDir = findTaggedDir(sessionDir, entry.cacheLookupTag);
    if (!taggedDir || !fs.existsSync(taggedDir)) return out;
    let files;
    try { files = fs.readdirSync(taggedDir).filter(f => f.endsWith('_log.json')); }
    catch (e) { return out; }
    for (const f of files) {
      const iid = f.replace(/_log\.json$/, '');
      try {
        const log = JSON.parse(fs.readFileSync(path.join(taggedDir, f), 'utf-8'));
        const ts = log.frame_timestamps || [];
        if (ts.length > 0) out[iid] = clusterFramesToBands(ts);
      } catch (e) { /* skip */ }
    }
    return out;
  }

  if (entry.pipeline === 'avp' || entry.pipeline === 'iterative') {
    for (const pred of predictions) {
      const expanded = (pred.segments || []).map(seg => ({
        start: Math.max(0, seg.start - VLM_FRAME_PADDING_S),
        end: seg.end + VLM_FRAME_PADDING_S,
      }));
      if (expanded.length > 0) out[pred.instance_id] = mergeBands(expanded);
    }
    return out;
  }

  return out;
}

// ---------------------------------------------------------------------------
// Endpoints
// ---------------------------------------------------------------------------

app.get('/api/participants/:participant/amount-tags', (req, res) => {
  res.json(buildTagRegistry(req.params.participant).map(e => ({
    tag: e.tag, pipeline: e.pipeline,
  })));
});

// Sessions present in a given preds file. Used by PriorView so the session
// dropdown only shows sessions the eval/preds file actually contains.
app.get('/api/participants/:participant/amount-tag-sessions/:tag', (req, res) => {
  const { participant, tag } = req.params;
  const entry = buildTagRegistry(participant).find(r => r.tag === tag);
  if (!entry) return res.status(404).json({ error: 'Tag not found' });
  if (!fs.existsSync(entry.predsFile)) return res.status(404).json({ error: 'Predictions not found' });
  const allPreds = JSON.parse(fs.readFileSync(entry.predsFile, 'utf-8'));
  const sessionSet = new Set(allPreds.map(p => p.session).filter(Boolean));
  res.json({ sessions: [...sessionSet].sort() });
});

// Cross-session evidence store written by 06_avp_round1_remaining_minimal_r2free_evidence_v1.py.
// The index file is `observer_evidence_<rawTag>.json` next to the preds file,
// where rawTag is the same string the predsTemplate uses (e.g.
// `gemini-3.1-pro-preview_late_evon_v1`).
app.get('/api/participants/:participant/prior-evidence/:tag', (req, res) => {
  const { participant, tag } = req.params;
  const entry = buildTagRegistry(participant).find(r => r.tag === tag);
  if (!entry) return res.status(404).json({ error: 'Tag not found' });
  // Recover rawTag from the registry entry. cacheLookupTag is set to rawTag
  // by buildTagRegistry — the same string the predsTemplate uses, e.g.
  // "gemini-3.1-pro-preview_late_evon_v1".
  const rawTag = entry.cacheLookupTag || entry.tag;
  const indexFile = path.join(
    participantDir(participant), 'outputs',
    `observer_evidence_${rawTag}.json`
  );
  if (!fs.existsSync(indexFile)) {
    return res.status(404).json({ error: 'No prior-evidence index for this tag' });
  }
  res.json(JSON.parse(fs.readFileSync(indexFile, 'utf-8')));
});

// Serve a prior-evidence JPEG by its index-recorded relative path. The path
// is rooted at `participants/<p>/outputs/`; we sandbox by re-resolving and
// rejecting anything that escapes that directory.
app.get('/prior-evidence-image/:participant/{*rest}', (req, res) => {
  const segments = Array.isArray(req.params.rest) ? req.params.rest : [req.params.rest];
  const rel = segments.join('/');
  const baseOutputs = path.resolve(participantDir(req.params.participant), 'outputs');
  const fp = path.resolve(baseOutputs, rel);
  if (!fp.startsWith(baseOutputs + path.sep)) {
    return res.status(400).send('Bad path');
  }
  if (!fs.existsSync(fp)) return res.status(404).send('Not found');
  res.sendFile(fp);
});

app.get('/api/participants/:participant/sessions/:session/amount-results/:tag', (req, res) => {
  const { participant, session, tag } = req.params;
  const registry = buildTagRegistry(participant);
  const entry = registry.find(r => r.tag === tag);
  if (!entry) return res.status(404).json({ error: 'Tag not found' });
  if (!fs.existsSync(entry.predsFile)) return res.status(404).json({ error: 'Predictions not found' });

  // Predictions — normalized to common schema
  const allPreds = JSON.parse(fs.readFileSync(entry.predsFile, 'utf-8'));
  const predictions = allPreds
    .filter(p => p.session === session)
    .map(normalizePrediction);

  // AVP: synthesize null-amount "predictions" for observer traces that the
  // script dropped (item_confirmed but amount_remaining was null). The trace
  // lives in participants/{P}/outputs/{session}/avp_*_planner.json under
  // session.observer[]. Surfacing them lets the UI show the VLM's reasoning
  // for missed items instead of a blank panel.
  if (entry.pipeline === 'avp') {
    const sessPlannerPath = path.join(
      participantDir(participant), 'outputs', session,
      path.basename(entry.plannerFile || '')
    );
    if (entry.plannerFile && fs.existsSync(sessPlannerPath)) {
      try {
        const sd = JSON.parse(fs.readFileSync(sessPlannerPath, 'utf-8'));
        const sessionLog = sd.session || sd;
        const observers = (sessionLog && sessionLog.observer) || [];
        // Planner observation_plan holds segments + reasoning per iid.
        let plannerPlan = (sessionLog && sessionLog.planner && sessionLog.planner.observation_plan) || [];
        // Fallback: older runs don't save observation_plan; parse from raw_response.
        if (plannerPlan.length === 0 && sessionLog && sessionLog.planner && sessionLog.planner.raw_response) {
          try {
            const m = sessionLog.planner.raw_response.match(/\{[\s\S]*"observation_plan"[\s\S]*\}/);
            if (m) plannerPlan = (JSON.parse(m[0]).observation_plan) || [];
          } catch (e) { /* ignore parse failure */ }
        }
        const planByIid = {};
        for (const p of plannerPlan) {
          if (p.instance_id) planByIid[p.instance_id] = p;
        }
        const haveIids = new Set(predictions.map(p => p.instance_id));
        for (const obs of observers) {
          // PerItem schema: one observer call per visual_class, with a
          // per_instance list holding each candidate iid's verdict.
          if (Array.isArray(obs.per_instance) && obs.per_instance.length > 0) {
            for (const inst of obs.per_instance) {
              const iid = inst.instance_id;
              if (!iid || haveIids.has(iid)) continue;
              const plan = planByIid[iid] || {};
              predictions.push(normalizePrediction({
                session,
                instance_id: iid,
                item: obs.visual_class || iid,
                amount_remaining: null,
                amount_used: null,
                reasoning: inst.reasoning || '',
                evidence_frames: inst.evidence_frames || [],
                planner_reasoning: plan.reasoning || '',
                planner_confidence: plan.confidence || '',
                segments: plan.segments || [],
                stats: obs.stats ? { observer: obs.stats } : null,
              }));
              haveIids.add(iid);
            }
            continue;
          }
          // Legacy schema: one observer entry per iid.
          if (!obs.instance_id || haveIids.has(obs.instance_id)) continue;
          const plan = planByIid[obs.instance_id] || {};
          predictions.push(normalizePrediction({
            session,
            instance_id: obs.instance_id,
            item: obs.visual_class || obs.instance_id,
            amount_remaining: null,
            amount_used: null,
            reasoning: obs.reasoning || '',
            evidence_frames: obs.evidence_frames || [],
            planner_reasoning: plan.reasoning || '',
            planner_confidence: plan.confidence || '',
            segments: plan.segments || [],
            stats: obs.stats ? { observer: obs.stats } : null,
          }));
          haveIids.add(obs.instance_id);
        }
      } catch (e) { /* ignore malformed per-session planner log */ }
    }
  }

  // Iterative pipeline: same "synthesize null-amount entry per observed iid"
  // idea, but the log shape is observer_rounds[] (each has round, window,
  // per_instance[]) rather than observer[]. We also collect every round a
  // given iid appeared in, so the UI can annotate "touched in rounds 0,1".
  if (entry.pipeline === 'iterative') {
    const sessionLog = loadSessionLogWithSweepShim(entry, participant, session);
    if (sessionLog) {
      try {
        const obsRounds = sessionLog.observer_rounds || [];
        const roundsByIid = {};   // iid -> Set of round numbers
        const segsByIid = {};     // iid -> list of [start,end,round]
        const lastInstByIid = {}; // iid -> latest per_instance entry
        for (const r of obsRounds) {
          const win = r.window || {};
          const segs = win.segments || [];
          // Sweep: window.target_items is parallel to window.segments and lists
          // which iids each segment was meant for. Use it to ATTRIBUTE per-segment
          // ownership instead of attaching every segment to every observed iid.
          const targetsParallel = Array.isArray(win.target_items) ? win.target_items : null;
          for (let segIdx = 0; segIdx < segs.length; segIdx++) {
            const s = segs[segIdx];
            if (!Array.isArray(s) || s.length < 2) continue;
            const tgts = targetsParallel ? (targetsParallel[segIdx] || []) : null;
            // If this round has per-segment targets (sweep), only attach the
            // segment to its declared iids. Otherwise (iterative legacy), it
            // applies to every iid the per_instance loop processes (kept below).
            if (tgts) {
              for (const iid of tgts) {
                if (!iid) continue;
                if (!segsByIid[iid]) segsByIid[iid] = [];
                segsByIid[iid].push([s[0], s[1], r.round || 0]);
              }
            }
          }
          for (const inst of (r.per_instance || [])) {
            const iid = inst.instance_id;
            if (!iid) continue;
            if (!roundsByIid[iid]) roundsByIid[iid] = new Set();
            roundsByIid[iid].add(r.round || 0);
            if (!segsByIid[iid]) segsByIid[iid] = [];
            // Legacy iterative: window.target_items absent → attach all segments
            // in this round to every per_instance iid (broad ownership).
            if (!targetsParallel) {
              for (const s of segs) {
                if (Array.isArray(s) && s.length >= 2) {
                  segsByIid[iid].push([s[0], s[1], r.round || 0]);
                }
              }
            }
            lastInstByIid[iid] = { ...inst, _window: win, _round: r.round || 0,
                                   _window_observation: r.window_observation || '' };
          }
        }
        const haveIids = new Set(predictions.map(p => p.instance_id));
        // Annotate existing predictions with rounds_touched + segments from all rounds.
        for (const p of predictions) {
          if (roundsByIid[p.instance_id]) {
            p.rounds_touched = Array.from(roundsByIid[p.instance_id]).sort((a,b)=>a-b);
          }
          if (segsByIid[p.instance_id] && segsByIid[p.instance_id].length) {
            p.segments = segsByIid[p.instance_id].map(([s,e,r]) => ({ start: s, end: e, round: r }));
          }
        }
        // Synthesize null-amount entries for iids the observer saw but did
        // not resolve (not_visible / visible_untouched / parse-failed).
        for (const [iid, inst] of Object.entries(lastInstByIid)) {
          if (haveIids.has(iid)) continue;
          const segs = (segsByIid[iid] || []).map(([s,e,r]) => ({ start: s, end: e, round: r }));
          predictions.push(normalizePrediction({
            session,
            instance_id: iid,
            item: (inst._window && inst._window.visual_class) || iid,
            amount_remaining: null,
            amount_used: null,
            handling_status: inst.handling_status || null,
            reasoning: inst.reasoning || '',
            evidence_frames: inst.evidence_frames || [],
            planner_reasoning: inst._window_observation || '',
            planner_confidence: (inst._window && inst._window.confidence) || '',
            segments: segs,
            rounds_touched: Array.from(roundsByIid[iid] || []).sort((a,b)=>a-b),
          }));
          haveIids.add(iid);
        }
      } catch (e) { /* ignore malformed */ }
    }
  }

  // Eval entries
  let eval_entries = [];
  if (fs.existsSync(entry.evalFile)) {
    const evalData = JSON.parse(fs.readFileSync(entry.evalFile, 'utf-8'));
    eval_entries = (evalData.entries || []).filter(e => e.session === session);
  }

  // Inventory from ledger snapshot, augmented with non-GT items that the
  // planner/observer touched (so the annotator can surface their traces too).
  const ledgerPath = path.join(participantDir(participant), 'ledger.json');
  let inventory = [];
  let ledger = null;
  if (fs.existsSync(ledgerPath)) {
    ledger = JSON.parse(fs.readFileSync(ledgerPath, 'utf-8'));
    const snap = (ledger.snapshots || {})[session] || {};
    for (const [iid, state] of Object.entries(snap)) {
      const item = (ledger.items || {})[iid] || {};
      inventory.push({
        instance_id: iid,
        visual_class: item.visual_class || iid,
        unit: item.unit || 'g',
        package_amount: item.package_amount || '',
        starting_amount: state.starting,
        gt_used: state.used,
        gt_remaining: state.remaining,
        in_gt_snapshot: true,
      });
    }
  }

  // Augment with non-snapshot iids that appear in this run's predictions
  // (already extended above) or in the planner trace (food_journeys /
  // observation_windows / skipped_items / observer per_instance). Keeps GT
  // info empty (null) but lets the user see the traces.
  if (ledger) {
    const haveInv = new Set(inventory.map(i => i.instance_id));
    const referenced = new Set();
    for (const p of predictions) {
      if (p.instance_id) referenced.add(p.instance_id);
    }
    if (entry.pipeline === 'iterative' && entry.plannerFile) {
      const sessPlannerPath = path.join(
        participantDir(participant), 'outputs', session,
        path.basename(entry.plannerFile)
      );
      if (fs.existsSync(sessPlannerPath)) {
        try {
          const sd = JSON.parse(fs.readFileSync(sessPlannerPath, 'utf-8'));
          const sLog = sd.session || sd;
          const collect = (j) => {
            for (const iid of (j.candidate_instance_ids || [])) {
              if (iid) referenced.add(iid);
            }
          };
          for (const fj of (sLog.final_journeys || [])) collect(fj);
          for (const sk of (sLog.skipped_items || [])) {
            if (sk && sk.instance_id) referenced.add(sk.instance_id);
          }
          for (const pr of (sLog.planner_rounds || [])) {
            const parsed = pr.parsed || {};
            for (const fj of (parsed.food_journeys || [])) collect(fj);
            for (const sk of (parsed.skipped_items || [])) {
              if (sk && sk.instance_id) referenced.add(sk.instance_id);
            }
            const ws = (parsed.action || {}).observation_windows || [];
            for (const w of ws) collect(w);
          }
          for (const obs of (sLog.observer_rounds || [])) {
            for (const inst of (obs.per_instance || [])) {
              if (inst && inst.instance_id) referenced.add(inst.instance_id);
            }
          }
        } catch (e) { /* ignore */ }
      }
    }
    for (const iid of referenced) {
      if (haveInv.has(iid)) continue;
      const item = (ledger.items || {})[iid] || {};
      inventory.push({
        instance_id: iid,
        visual_class: item.visual_class || iid,
        unit: item.unit || 'g',
        package_amount: item.package_amount || '',
        starting_amount: null,
        gt_used: null,
        gt_remaining: null,
        in_gt_snapshot: false,
      });
    }
  }

  // Session-level reasoning (unified)
  const session_reasoning = loadSessionReasoning(entry, participant, session);

  // VLM input segments (predictions are already normalized at this point)
  const vlm_input_segments_by_iid = buildVlmInputSegments(entry, predictions, participant, session);

  res.json({ predictions, eval_entries, inventory, pipeline: entry.pipeline,
             session_reasoning, vlm_input_segments_by_iid });
});

// Iterative planner prompts — round-0 includes the full HOI/DINO/SigLIP
// evidence block that the planner bootstraps from. Later rounds are the
// follow-up user turns carrying observer reports + rolling ledger. Endpoint
// returns both when available, keyed by round index (0 = round-0 user
// prompt stripped of system preamble; N≥2 = round N follow-up user text).
app.get('/api/participants/:participant/sessions/:session/iterative-prompts/:tag', (req, res) => {
  const { participant, session, tag } = req.params;
  const registry = buildTagRegistry(participant);
  const entry = registry.find(r => r.tag === tag);
  if (!entry || entry.pipeline !== 'iterative') {
    return res.status(404).json({ error: 'Iterative tag not found' });
  }
  // Cache layout is {sessionDir}/{model_tag}/{run_tag}/; cacheLookupTag
  // is the flat {model_tag}_{run_tag} joined form, so findTaggedDir walks
  // subdirs and returns the actual run_tag directory.
  const sessionDir = path.join(entry.cacheDir, participant, session);
  const tagDir = findTaggedDir(sessionDir, entry.cacheLookupTag);
  const prompts = {};   // { round_idx: { system, user } }
  if (tagDir && fs.existsSync(tagDir)) {
    try {
      // Round 0: planner_round0_prompt.txt (system + ---USER--- + user body)
      const r0 = path.join(tagDir, 'planner_round0_prompt.txt');
      if (fs.existsSync(r0)) {
        const txt = fs.readFileSync(r0, 'utf-8');
        const idx = txt.indexOf('---USER---');
        if (idx >= 0) {
          prompts[0] = {
            system: txt.slice(0, idx).trim(),
            user:   txt.slice(idx + '---USER---'.length).trim(),
          };
        } else {
          prompts[0] = { system: '', user: txt };
        }
      }
      // Round 2+: planner_round{N}_user.txt (these are the follow-up user
      // turns fed into the planner call that produces round N's response).
      for (const f of fs.readdirSync(tagDir)) {
        const m = f.match(/^planner_round(\d+)_user\.txt$/);
        if (!m) continue;
        const n = parseInt(m[1]);
        if (n < 2) continue;
        prompts[n] = { system: '', user: fs.readFileSync(path.join(tagDir, f), 'utf-8') };
      }
    } catch (e) { /* fall through with whatever we collected */ }
  }
  res.json({ participant, session, tag, prompts });
});

// Iterative pipeline trace — full per-round planner + observer records.
// Returns the session_log object from the per-session planner JSON (shape:
// { planner_rounds: [...], observer_rounds: [...], skipped_items: [...],
//   final_journeys: [...] }). Used by AmountViewIterative.
app.get('/api/participants/:participant/sessions/:session/iterative-trace/:tag', (req, res) => {
  const { participant, session, tag } = req.params;
  const registry = buildTagRegistry(participant);
  const entry = registry.find(r => r.tag === tag);
  if (!entry || entry.pipeline !== 'iterative') {
    return res.status(404).json({ error: 'Iterative tag not found' });
  }
  const sessPlannerPath = path.join(
    participantDir(participant), 'outputs', session,
    path.basename(entry.plannerFile || '')
  );
  if (!entry.plannerFile || !fs.existsSync(sessPlannerPath)) {
    return res.status(404).json({ error: 'Session planner log not found' });
  }
  try {
    const sd = JSON.parse(fs.readFileSync(sessPlannerPath, 'utf-8'));
    const sessionLog = sd.session || sd;
    // Sweep runs (avp_minimal_remaining_*) lack observer_rounds[]; build it
    // from sessionLog.sweep / sweep_r2 / rounds[] so the iterative trace UI
    // can render per-round windows + items uniformly.
    if (!sessionLog.observer_rounds) {
      const synth = synthesizeSweepObserverRounds(sessionLog);
      if (synth) sessionLog.observer_rounds = synth;
    }
    // Same for planner_rounds: synthesize a minimal trace so the UI shows
    // per-round planner reasoning + segments.
    let plannerRounds = sessionLog.planner_rounds || [];
    if ((!plannerRounds || plannerRounds.length === 0) && sessionLog.planner) {
      const out = [];
      const r1Plan = sessionLog.planner;
      // New journey/dense planner emits two parallel lists; reconstruct a
      // unified sweep_segments view (interleaved + sorted) for backward UI.
      const r1Unified = (() => {
        if (r1Plan.sweep_segments && r1Plan.sweep_segments.length) return r1Plan.sweep_segments;
        const u = [];
        for (const j of (r1Plan.journey_samples || [])) {
          u.push({ kind: 'journey', start: j.t, end: j.t, target_items: j.target_items || [] });
        }
        for (const d of (r1Plan.dense_windows || [])) {
          u.push({ kind: 'dense', start: d.start, end: d.end, target_items: d.target_items || [] });
        }
        u.sort((a, b) => a.start - b.start);
        return u;
      })();
      out.push({
        round: 1,
        prompt: r1Plan.prompt || '',
        raw_response: r1Plan.raw_response || '',
        parsed: {
          item_decisions: r1Plan.item_decisions || [],
          sweep_segments: r1Unified,
          journey_samples: r1Plan.journey_samples || [],
          dense_windows: r1Plan.dense_windows || [],
        },
        stats: r1Plan.stats || null,
      });
      if (sessionLog.planner_r2) {
        out.push({
          round: 2,
          prompt: sessionLog.planner_r2.prompt || '',
          raw_response: sessionLog.planner_r2.raw_response || '',
          parsed: { item_decisions: sessionLog.planner_r2.item_decisions || [],
                    sweep_segments: sessionLog.planner_r2.sweep_segments || [] },
          stats: sessionLog.planner_r2.stats || null,
        });
      }
      for (const r of (sessionLog.rounds || [])) {
        const pl = r.planner;
        if (!pl) continue;
        out.push({
          round: pl.stats?.round || (3 + out.length - 2),
          prompt: pl.prompt || '',
          raw_response: pl.raw_response || '',
          parsed: { item_decisions: pl.item_decisions || [], sweep_segments: pl.sweep_segments || [] },
          stats: pl.stats || null,
        });
      }
      plannerRounds = out;
    }
    res.json({
      participant, session, tag,
      timestamp: sd.timestamp || null,
      model: sd.model || null,
      planner_rounds:  plannerRounds,
      observer_rounds: sessionLog.observer_rounds || [],
      skipped_items:   sessionLog.skipped_items || [],
      final_journeys:  sessionLog.final_journeys || [],
    });
  } catch (e) {
    res.status(500).json({ error: 'Malformed planner log', detail: String(e) });
  }
});

// Per-item log (prompt, response, thinking)
app.get('/api/participants/:participant/sessions/:session/amount-item-log/:tag/:instanceId', (req, res) => {
  const { participant, session, tag, instanceId } = req.params;
  const registry = buildTagRegistry(participant);
  const entry = registry.find(r => r.tag === tag);
  if (!entry) return res.status(404).json({ error: 'Tag not found' });

  const sessionDir = path.join(entry.cacheDir, participant, session);
  let logPath = findTaggedLog(sessionDir, entry.cacheLookupTag, `${instanceId}_log.json`);

  // AVP fallback: findTaggedLog joins model+run but we only have run_tag.
  // Walk model subdirs matching on run_tag alone.
  if (!logPath && entry.pipeline === 'avp') {
    try {
      for (const modelDir of fs.readdirSync(sessionDir, { withFileTypes: true })) {
        if (!modelDir.isDirectory()) continue;
        const candidate = path.join(sessionDir, modelDir.name, entry.cacheLookupTag, `${instanceId}_log.json`);
        if (fs.existsSync(candidate)) { logPath = candidate; break; }
      }
    } catch (e) { /* ignore */ }
  }

  if (!logPath) return res.status(404).json({ error: 'Item log not found' });
  res.json(JSON.parse(fs.readFileSync(logPath, 'utf-8')));
});

// Get AdaTAD + DINOv2 item-labeled segments for a session
app.get('/api/participants/:participant/sessions/:session/tad-item-labels', (req, res) => {
  const labelsFile = path.join(participantDir(req.params.participant), 'outputs', 'adatad_item_labels.json');
  if (!fs.existsSync(labelsFile)) return res.status(404).json({ error: 'No item labels found' });

  const data = JSON.parse(fs.readFileSync(labelsFile, 'utf-8'));
  const sessionData = (data.sessions || []).find(s => s.session === req.params.session);
  if (!sessionData) return res.status(404).json({ error: 'Session not found in labels' });

  res.json(sessionData);
});

// Scene tags from 07b_siglip_scene_tags.py (one whole-frame scene tag per HOI
// trigger frame: fridge / cabinet / countertop / sink / unknown).
app.get('/api/participants/:participant/sessions/:session/scene-tags', (req, res) => {
  const sceneFile = path.join(
    participantDir(req.params.participant),
    'outputs', req.params.session, 'scene_tags.json'
  );
  if (!fs.existsSync(sceneFile)) return res.status(404).json({ error: 'No scene tags' });
  res.json(JSON.parse(fs.readFileSync(sceneFile, 'utf-8')));
});

// OWLv2 vs GPT-5.4 agreement eval from 07e_owlv2_gpt_eval.py.
// Per-session sample of frames at target_fps (default 0.1) labeled by both
// OWLv2 (multi-query detection) and GPT, with per-frame agreement.
app.get('/api/participants/:participant/sessions/:session/owlv2-gpt-eval', (req, res) => {
  const f = path.join(
    participantDir(req.params.participant),
    'outputs', req.params.session, 'owlv2_gpt_scene_eval.json'
  );
  if (!fs.existsSync(f)) return res.status(404).json({ error: 'No OWLv2/GPT eval results' });
  res.json(JSON.parse(fs.readFileSync(f, 'utf-8')));
});

// Static serving of eval frames (hands23 frames, referenced by relative path
// inside the eval JSON). The /hands23 route below already serves these, but
// the eval JSON's frame_file paths are relative to the participant outputs
// dir, so we add a thin /eval-frame proxy that joins them correctly.
app.get('/eval-frame/:participant/:session/{*rest}', (req, res) => {
  // Express 5 / path-to-regexp v6 requires named splats; `rest` is an array
  // of path segments (e.g. ['hands23_detection', '20260310-195710', 'frames', 'frame_*.jpg']).
  const segments = Array.isArray(req.params.rest) ? req.params.rest : [req.params.rest];
  const rel = segments.join('/');
  const fp = path.join(participantDir(req.params.participant), 'outputs', req.params.session, rel);
  if (!fs.existsSync(fp)) return res.status(404).send('Not found');
  res.sendFile(fp);
});

// Scene tags from 07d_owlv2_scene_detect.py (zero-shot OWLv2 detection of
// fridge/sink/stove on HOI trigger frames; derives per-frame scene tag from
// top detection + confidence).
app.get('/api/participants/:participant/sessions/:session/owlv2-scene', (req, res) => {
  const f = path.join(
    participantDir(req.params.participant),
    'outputs', req.params.session, 'scene_tags_owlv2.json'
  );
  if (!fs.existsSync(f)) return res.status(404).json({ error: 'No OWLv2 scene results' });
  res.json(JSON.parse(fs.readFileSync(f, 'utf-8')));
});

// Per-item temporal segments from 05b_per_item_segments.py (HOI + SigLIP +
// DINOv2 morphological aggregation; no AdaTAD required).
app.get('/api/participants/:participant/sessions/:session/per-item-segments', (req, res) => {
  const f = path.join(
    participantDir(req.params.participant),
    'outputs', req.params.session, 'per_item_segments.json'
  );
  if (!fs.existsSync(f)) return res.status(404).json({ error: 'No per-item segments' });
  res.json(JSON.parse(fs.readFileSync(f, 'utf-8')));
});

// Package-vs-derivative results from 07a_siglip_pkg_vs_deriv_proto.py.
// Currently this is a per-participant file (prototype scope), so we filter
// down to the requested session client-side via this endpoint.
app.get('/api/participants/:participant/sessions/:session/pkg-vs-deriv', (req, res) => {
  const f = path.join(participantDir(req.params.participant), 'outputs', 'scene_context_proto_results.json');
  if (!fs.existsSync(f)) return res.status(404).json({ error: 'No pkg-vs-deriv results' });
  const all = JSON.parse(fs.readFileSync(f, 'utf-8'));
  const entries = (Array.isArray(all) ? all : []).filter(e => e.session === req.params.session);
  res.json({ entries });
});

// ---- VLM eval HTML browser ----
//
// Serves the static `*_eval.html` files generated by
// annotating/visualize_vlm_amount.py (auto-generated by evaluate_amount.py's
// write_report). The annotator runs on a remote machine, so the user views
// these from a browser via the same port.
//
// GET /eval-html/                       -> index page listing every eval HTML
//                                          across participants
// GET /eval-html/:participant/:filename -> stream a single .html file from
//                                          participants/{P}/outputs/

function listEvalHtmls() {
  if (!fs.existsSync(PARTICIPANTS_DIR)) return [];
  const out = [];
  const participants = fs.readdirSync(PARTICIPANTS_DIR)
    .filter(d => fs.statSync(path.join(PARTICIPANTS_DIR, d)).isDirectory())
    .sort();
  for (const p of participants) {
    const outputs = path.join(PARTICIPANTS_DIR, p, 'outputs');
    if (!fs.existsSync(outputs)) continue;
    const files = fs.readdirSync(outputs).filter(f => f.endsWith('_eval.html')).sort();
    for (const f of files) {
      const full = path.join(outputs, f);
      let mtime = null, size = null;
      try {
        const st = fs.statSync(full);
        mtime = st.mtime.toISOString().replace('T', ' ').replace(/\..*/, '');
        size = st.size;
      } catch (e) { /* ignore */ }
      out.push({ participant: p, filename: f, mtime, size });
    }
  }
  return out;
}

app.get('/eval-html', (req, res) => {
  const items = listEvalHtmls();
  const byParticipant = {};
  for (const it of items) {
    (byParticipant[it.participant] ||= []).push(it);
  }
  const escape = (s) => String(s).replace(/[&<>"']/g, c => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
  }[c]));
  const fmtSize = (n) => n == null ? '' : n > 1024 * 1024
    ? `${(n / 1024 / 1024).toFixed(1)}M`
    : n > 1024 ? `${(n / 1024).toFixed(0)}K` : `${n}B`;
  let html = `<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>VLM Amount Eval — index</title>
<style>
  body { font-family: -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
         background: #f5f5f5; color: #222; padding: 24px; max-width: 980px; margin: 0 auto; }
  h1 { font-size: 1.4em; margin-bottom: 4px; }
  .subtitle { color: #777; font-size: 0.85em; margin-bottom: 20px; }
  h2 { font-size: 1.05em; margin: 18px 0 6px 0; color: #3949ab;
       border-bottom: 1px solid #c5cae9; padding-bottom: 3px; }
  ul { list-style: none; padding: 0; margin: 0; }
  li { padding: 4px 0; font-size: 0.92em; display: flex; gap: 14px; align-items: baseline; }
  li a { color: #1a237e; text-decoration: none; font-weight: 500; flex: 1; min-width: 0;
         overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  li a:hover { text-decoration: underline; }
  li .mtime { color: #888; font-size: 0.78em; font-variant-numeric: tabular-nums; }
  li .size { color: #aaa; font-size: 0.78em; font-variant-numeric: tabular-nums; min-width: 38px; text-align: right; }
  .empty { color: #999; font-style: italic; }
</style>
</head><body>
<h1>VLM Amount Eval — generated HTMLs</h1>
<p class="subtitle">${items.length} file${items.length === 1 ? '' : 's'} across ${Object.keys(byParticipant).length} participant${Object.keys(byParticipant).length === 1 ? '' : 's'}. Auto-generated by <code>evaluate_amount.write_report</code>.</p>
`;
  if (items.length === 0) {
    html += '<p class="empty">No <code>*_eval.html</code> files found yet.</p>';
  } else {
    for (const p of Object.keys(byParticipant).sort()) {
      html += `<h2>${escape(p)}</h2>\n<ul>\n`;
      for (const it of byParticipant[p]) {
        const href = `/eval-html/${encodeURIComponent(p)}/${encodeURIComponent(it.filename)}`;
        html += `  <li><a href="${href}">${escape(it.filename)}</a>`
              + `<span class="mtime">${escape(it.mtime || '')}</span>`
              + `<span class="size">${fmtSize(it.size)}</span></li>\n`;
      }
      html += '</ul>\n';
    }
  }
  html += '</body></html>\n';
  res.set('Content-Type', 'text/html; charset=utf-8');
  res.send(html);
});

app.get('/eval-html/:participant/:filename', (req, res) => {
  const { participant, filename } = req.params;
  // Path-traversal guard: filename must end in .html and not contain slashes.
  if (!filename.endsWith('.html') || filename.includes('/') || filename.includes('\\') || filename.includes('..')) {
    return res.status(400).send('Invalid filename');
  }
  const outputs = path.join(participantDir(participant), 'outputs');
  const full = path.join(outputs, filename);
  // Belt-and-suspenders: ensure resolved path stays under the outputs dir.
  if (!path.resolve(full).startsWith(path.resolve(outputs) + path.sep)) {
    return res.status(400).send('Invalid path');
  }
  if (!fs.existsSync(full)) return res.status(404).send('Not found');
  res.sendFile(full);
});

// ---- Ledger HTML viewer ----
//
// Serves the ledger.html files generated by annotating/visualize_ledger.py.
//
// GET /ledger/                  -> index page listing ledger.html per participant
// GET /ledger/:participant      -> serve that participant's ledger.html

app.get('/ledger', (req, res) => {
  if (!fs.existsSync(PARTICIPANTS_DIR)) {
    return res.send('No participants directory found.');
  }
  const participants = fs.readdirSync(PARTICIPANTS_DIR)
    .filter(d => fs.statSync(path.join(PARTICIPANTS_DIR, d)).isDirectory())
    .sort();

  const escape = (s) => String(s).replace(/[&<>"']/g, c => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
  }[c]));
  const fmtSize = (n) => n == null ? '' : n > 1024 * 1024
    ? `${(n / 1024 / 1024).toFixed(1)}M`
    : n > 1024 ? `${(n / 1024).toFixed(0)}K` : `${n}B`;

  const entries = [];
  for (const p of participants) {
    const ledgerPath = path.join(PARTICIPANTS_DIR, p, 'ledger.html');
    if (!fs.existsSync(ledgerPath)) continue;
    let mtime = null, size = null;
    try {
      const st = fs.statSync(ledgerPath);
      mtime = st.mtime.toISOString().replace('T', ' ').replace(/\..*/, '');
      size = st.size;
    } catch (e) { /* ignore */ }
    entries.push({ participant: p, mtime, size });
  }

  let html = `<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>Inventory Ledger — index</title>
<style>
  body { font-family: -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
         background: #f5f5f5; color: #222; padding: 24px; max-width: 980px; margin: 0 auto; }
  h1 { font-size: 1.4em; margin-bottom: 4px; }
  .subtitle { color: #777; font-size: 0.85em; margin-bottom: 20px; }
  ul { list-style: none; padding: 0; margin: 0; }
  li { padding: 4px 0; font-size: 0.92em; display: flex; gap: 14px; align-items: baseline; }
  li a { color: #1a237e; text-decoration: none; font-weight: 500; flex: 1; min-width: 0; }
  li a:hover { text-decoration: underline; }
  li .mtime { color: #888; font-size: 0.78em; font-variant-numeric: tabular-nums; }
  li .size { color: #aaa; font-size: 0.78em; font-variant-numeric: tabular-nums; min-width: 38px; text-align: right; }
  .empty { color: #999; font-style: italic; }
</style>
</head><body>
<h1>Inventory Ledger</h1>
<p class="subtitle">${entries.length} participant${entries.length === 1 ? '' : 's'} with ledger.html. Generated by <code>visualize_ledger.py</code>.</p>
`;
  if (entries.length === 0) {
    html += '<p class="empty">No <code>ledger.html</code> files found yet.</p>';
  } else {
    html += '<ul>\n';
    for (const e of entries) {
      const href = `/ledger/${encodeURIComponent(e.participant)}`;
      html += `  <li><a href="${href}">${escape(e.participant)}</a>`
            + `<span class="mtime">${escape(e.mtime || '')}</span>`
            + `<span class="size">${fmtSize(e.size)}</span></li>\n`;
    }
    html += '</ul>\n';
  }
  html += '</body></html>\n';
  res.set('Content-Type', 'text/html; charset=utf-8');
  res.send(html);
});

app.get('/ledger/:participant', (req, res) => {
  const { participant } = req.params;
  if (participant.includes('/') || participant.includes('\\') || participant.includes('..')) {
    return res.status(400).send('Invalid participant');
  }
  const full = path.join(PARTICIPANTS_DIR, participant, 'ledger.html');
  if (!path.resolve(full).startsWith(path.resolve(PARTICIPANTS_DIR) + path.sep)) {
    return res.status(400).send('Invalid path');
  }
  if (!fs.existsSync(full)) return res.status(404).send('Not found');
  res.sendFile(full);
});

app.listen(PORT, '127.0.0.1', () => {
  console.log(`Annotator server running on http://localhost:${PORT}`);
  console.log(`Participants dir: ${PARTICIPANTS_DIR}`);
  console.log(`Eval HTML index:  http://localhost:${PORT}/eval-html`);
  console.log(`Ledger index:     http://localhost:${PORT}/ledger`);
});

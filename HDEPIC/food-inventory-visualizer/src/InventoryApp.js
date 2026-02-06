import React, { useState, useRef, useCallback, useEffect, useMemo } from 'react';
import InventoryItemList from './components/InventoryItemList';
import InventoryVideoPlayer from './components/InventoryVideoPlayer';
import TimelineView from './components/TimelineView';
import AggregatedView from './components/AggregatedView';
import VLMResultsView from './components/VLMResultsView';
import ComparisonView from './components/ComparisonView';
import { parseNarrationTimestampsJSON, parseNarrationId } from './utils/narrationParser';

const VIDEO_SERVER = 'http://localhost:4001';

const styles = {
  app: {
    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
    minHeight: '100vh',
    backgroundColor: '#f5f5f5',
  },
  header: {
    backgroundColor: '#2E7D32',
    color: 'white',
    padding: '4px 20px',
    fontSize: '14px',
    fontWeight: 'bold',
    boxShadow: '0 2px 4px rgba(0,0,0,0.2)',
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  serverStatus: {
    fontSize: '12px',
    padding: '4px 10px',
    borderRadius: '12px',
    backgroundColor: 'rgba(255,255,255,0.2)',
  },
  serverOnline: {
    backgroundColor: 'rgba(76, 175, 80, 0.3)',
  },
  serverOffline: {
    backgroundColor: 'rgba(244, 67, 54, 0.3)',
  },
  content: {
    padding: '6px 20px',
    maxWidth: '1600px',
    margin: '0 auto',
  },
  loaderSection: {
    backgroundColor: 'white',
    borderRadius: '6px',
    padding: '6px 16px',
    marginBottom: '6px',
    boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
  },
  loaderTitle: {
    margin: '0 0 4px 0',
    fontSize: '13px',
    fontWeight: 'bold',
  },
  inputRow: {
    display: 'flex',
    gap: '15px',
    alignItems: 'flex-end',
    flexWrap: 'wrap',
  },
  inputGroup: {
    flex: '1',
    minWidth: '200px',
  },
  label: {
    display: 'block',
    marginBottom: '4px',
    fontWeight: '500',
    fontSize: '13px',
  },
  hint: {
    fontSize: '11px',
    color: '#666',
    marginTop: '2px',
  },
  select: {
    width: '100%',
    padding: '8px',
    borderRadius: '4px',
    border: '1px solid #ddd',
    fontSize: '13px',
  },
  button: {
    padding: '8px 16px',
    backgroundColor: '#2E7D32',
    color: 'white',
    border: 'none',
    borderRadius: '4px',
    cursor: 'pointer',
    fontSize: '13px',
    height: '34px',
  },
  buttonSecondary: {
    backgroundColor: '#1976D2',
  },
  mainLayout: {
    display: 'grid',
    gridTemplateColumns: '400px 1fr',
    gap: '20px',
    height: 'calc(100vh - 90px)',
  },
  statsBar: {
    padding: '4px 12px',
    backgroundColor: '#e8f5e9',
    borderRadius: '4px',
    fontSize: '11px',
    color: '#1b5e20',
    marginBottom: '6px',
  },
  error: {
    color: '#d32f2f',
    marginTop: '10px',
    fontSize: '13px',
  },
  success: {
    color: '#388e3c',
    marginTop: '10px',
    fontSize: '13px',
  },
  quickLoad: {
    display: 'flex',
    gap: '10px',
    marginTop: '15px',
    paddingTop: '15px',
    borderTop: '1px solid #e0e0e0',
  },
  viewToggle: {
    display: 'flex',
    gap: '0',
    marginLeft: '20px',
  },
  viewToggleButton: {
    padding: '8px 16px',
    border: '1px solid #2E7D32',
    backgroundColor: 'white',
    color: '#2E7D32',
    fontSize: '13px',
    cursor: 'pointer',
    transition: 'all 0.2s',
  },
  viewToggleButtonFirst: {
    borderRadius: '4px 0 0 4px',
  },
  viewToggleButtonLast: {
    borderRadius: '0 4px 4px 0',
    borderLeft: 'none',
  },
  viewToggleButtonActive: {
    backgroundColor: '#2E7D32',
    color: 'white',
  },
};

function InventoryApp() {
  // Server state
  const [serverOnline, setServerOnline] = useState(false);
  const [availableVideos, setAvailableVideos] = useState([]);

  // Data state
  const [inventoryData, setInventoryData] = useState(null);
  const [lifecycleData, setLifecycleData] = useState(null);
  const [aggregatedData, setAggregatedData] = useState(null);
  const [vlmDataByTag, setVlmDataByTag] = useState({}); // { tag: data }
  const [selectedVlmTag, setSelectedVlmTag] = useState(null);
  const [narrationTimestamps, setNarrationTimestamps] = useState({});
  const [participant, setParticipant] = useState('P01');
  const [viewMode, setViewMode] = useState('items'); // 'items', 'timeline', 'aggregated', 'vlm', 'failure-cases', or 'comparison'

  // Comparison mode state
  const [comparisonTagA, setComparisonTagA] = useState(null);
  const [comparisonTagB, setComparisonTagB] = useState(null);

  // Failure cases state
  const [failureCasesFiles, setFailureCasesFiles] = useState([]); // List of available files
  const [selectedFailureCasesFile, setSelectedFailureCasesFile] = useState(null);
  const [failureCasesData, setFailureCasesData] = useState(null);

  // Overlay failure cases state (for VLM Results and Comparison views)
  const [overlayFailureCasesFile, setOverlayFailureCasesFile] = useState(null);
  const [overlayFailureCases, setOverlayFailureCases] = useState(null);
  const [overlayDirty, setOverlayDirty] = useState(false);

  // Hands23 detection data
  const [hands23Data, setHands23Data] = useState(null);

  // UI state
  const [selectedItem, setSelectedItem] = useState(null);
  const [currentVideo, setCurrentVideo] = useState(null);
  const [currentVideoUrl, setCurrentVideoUrl] = useState(null);
  const [currentTime, setCurrentTime] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);

  // Refs
  const videoRef = useRef(null);

  // Check server status on mount
  useEffect(() => {
    checkServerStatus();
    const interval = setInterval(checkServerStatus, 5000);
    return () => clearInterval(interval);
  }, []);

  const checkServerStatus = async () => {
    try {
      const response = await fetch(`${VIDEO_SERVER}/list/${participant}`, { method: 'GET' });
      if (response.ok) {
        const data = await response.json();
        setServerOnline(true);
        setAvailableVideos(data.videos || []);
      } else {
        setServerOnline(false);
        setAvailableVideos([]);
      }
    } catch {
      setServerOnline(false);
      setAvailableVideos([]);
    }
  };

  // Merge v2 failure cases with timeline data (converts to v1 format for display)
  // Uses embedded case data when available; only falls back to timeline for missing fields
  const mergeFailureCasesV2 = async (failureCasesData) => {
    const cases = failureCasesData.cases || [];

    // Check if cases have embedded data (food_name or video_id present)
    const hasEmbeddedData = cases.some(c => c.food_name || c.video_id);

    // Collect unique participants
    const participants = [...new Set(cases.map(c => c.participant))];
    console.log(`Loading timeline data for participants: ${participants.join(', ')}`);

    // Load timeline_annotated for each participant (for GT and basic segment info as fallback)
    const segmentLookup = {}; // segment_id -> {item, segment}

    for (const p of participants) {
      try {
        const response = await fetch(
          `${VIDEO_SERVER}/data/outputs/02_inventory/${p}/${p}_timeline_annotated.json?t=${Date.now()}`
        );
        if (response.ok) {
          const data = await response.json();
          for (const item of data.items || []) {
            for (const seg of item.dispensal_segments || []) {
              const segId = seg.segment_id;
              if (segId) {
                segmentLookup[segId] = { item, segment: seg, participant: p };
              }
            }
          }
        }
      } catch (err) {
        console.warn(`Failed to load timeline for ${p}:`, err);
      }
    }

    console.log(`Loaded ${Object.keys(segmentLookup).length} segments from timeline (no VLM results loaded)`);

    // Build v1-style items from v2 cases
    const items = [];
    for (const c of cases) {
      const segId = c.segment_id;
      const timelineData = segmentLookup[segId];

      // Use embedded case data with timeline as fallback
      const timelineItem = timelineData?.item;
      const timelineSeg = timelineData?.segment;
      const segParticipant = c.participant || timelineData?.participant;

      // Skip if we have neither embedded data nor timeline data
      if (!hasEmbeddedData && !timelineData) {
        console.warn(`Segment ${segId} not found in timeline data and no embedded data`);
        continue;
      }

      // Build segment data from embedded case fields, falling back to timeline
      const segmentData = {
        segment_id: segId,
        segment_idx: 0,
        video_id: c.video_id || timelineSeg?.video_id,
        start_timestamp: c.start_timestamp ?? timelineSeg?.start_timestamp,
        end_timestamp: c.end_timestamp ?? timelineSeg?.end_timestamp,
        ground_truth_count: c.ground_truth_count ?? timelineSeg?.count,
        ground_truth_unit: c.ground_truth_unit || timelineSeg?.count_unit,
        case_id: c.case_id,
        include: c.include,
        priority: c.priority,
        notes: c.notes,
        tags: c.tags,
        // Embedded prediction data (from the case itself, not fetched from VLM)
        predicted_count: c.predicted_count,
        predicted_unit: c.predicted_unit,
        match: c.match,
        confidence: c.confidence,
        visual_evidence: c.visual_evidence,
        // Multipath data
        paths: c.paths,
        path_estimates: c.path_estimates,
        trusted_paths: c.trusted_paths,
        ignored_paths: c.ignored_paths,
        evidence_frames: c.evidence_frames,
      };

      const foodName = c.food_name || timelineItem?.food_name || segId;

      items.push({
        narration_id: c.narration_id,
        food_name: `[${segParticipant}] ${foodName}`,
        difficulty: c.difficulty || timelineItem?.difficulty,
        video_range: timelineItem?.video_range,
        total_ground_truth: c.ground_truth_count ?? timelineItem?.total_count,
        total_ground_truth_unit: c.ground_truth_unit || timelineItem?.count_unit,
        participant: segParticipant,
        recipe_amount: timelineItem?.matched_ingredient_weight,
        segments: [segmentData],
        total_predicted: c.predicted_count,
      });
    }

    // Return v1-style structure
    return {
      ...failureCasesData,
      items,
    };
  };

  // Enrich VLM results with GT/metadata from timeline_annotated (aggData)
  // Works for all prompt modes. Only fills missing fields (backward-compatible with old files).
  const enrichFrameResults = (vlmJson, aggData) => {
    if (!vlmJson) return vlmJson;

    // Build lookups from timeline_annotated (aggData)
    const segmentLookup = {}; // segment_id -> {video_id, start_timestamp, end_timestamp, count, count_unit, tags, notes}
    const itemLookup = {};    // narration_id -> {food_name, difficulty, video_range, total_count, count_unit, recipe_amount}

    if (aggData) {
      for (const item of aggData.items || []) {
        const narr = item.narration_id;
        if (narr) {
          itemLookup[narr] = {
            food_name: item.food_name,
            difficulty: item.difficulty,
            video_range: item.video_range,
            total_count: item.total_count,
            count_unit: item.count_unit,
            recipe_amount: item.matched_ingredient_weight,
          };
        }
        for (const seg of item.dispensal_segments || []) {
          if (seg.segment_id) {
            segmentLookup[seg.segment_id] = {
              video_id: seg.video_id,
              start_timestamp: seg.start_timestamp,
              end_timestamp: seg.end_timestamp,
              count: seg.count,
              count_unit: seg.count_unit,
              tags: seg.tags,
              notes: seg.notes,
            };
          }
        }
      }
    }

    // Enrich each item and segment — fill missing fields only
    for (const item of vlmJson.items || []) {
      const tl = itemLookup[item.narration_id] || {};

      // Item-level enrichment (only fill missing)
      if (item.food_name == null) item.food_name = tl.food_name;
      if (item.difficulty == null) item.difficulty = tl.difficulty;
      if (item.video_range == null) item.video_range = tl.video_range;
      if (item.total_ground_truth == null) item.total_ground_truth = tl.total_count;
      if (item.total_ground_truth_unit == null) item.total_ground_truth_unit = tl.count_unit;
      if (item.recipe_amount == null) item.recipe_amount = tl.recipe_amount;

      for (const seg of item.segments || []) {
        const tlSeg = segmentLookup[seg.segment_id] || {};

        // Segment-level enrichment (only fill missing)
        if (seg.video_id == null) seg.video_id = tlSeg.video_id;
        if (seg.start_timestamp == null) seg.start_timestamp = tlSeg.start_timestamp;
        if (seg.end_timestamp == null) seg.end_timestamp = tlSeg.end_timestamp;
        if (seg.ground_truth_count == null && tlSeg.count != null) {
          seg.ground_truth_count = tlSeg.count;
          seg.ground_truth_unit = tlSeg.count_unit;
        }

        // Always merge tags from timeline (VLM files don't have tags)
        if (tlSeg.tags && tlSeg.tags.length > 0) {
          seg.tags = [...(seg.tags || [])];
          for (const t of tlSeg.tags) {
            if (!seg.tags.includes(t)) seg.tags.push(t);
          }
        }

        // Extract predicted_count from final_synthesis if not already set
        if (seg.predicted_count == null) {
          const synth = seg.final_synthesis;
          if (synth && synth.final_count_estimate != null) {
            seg.predicted_count = synth.final_count_estimate;
          }
        }

        // Compute match status if not already set
        if (seg.match == null) {
          if (seg.predicted_count == null || seg.ground_truth_count == null) {
            seg.match = null;
          } else {
            const diff = Math.abs(seg.predicted_count - seg.ground_truth_count);
            if (diff === 0) seg.match = 'exact';
            else if (diff === 1) seg.match = 'close';
            else seg.match = 'wrong';
          }
        }
      }
    }

    return vlmJson;
  };

  // Load data from server
  const loadFromServer = async () => {
    setLoading(true);
    setError(null);
    setSuccess(null);

    try {
      // Load inventory data (known_quantities.json)
      const inventoryResponse = await fetch(
        `${VIDEO_SERVER}/data/outputs/02_inventory/${participant}/${participant}_known_quantities.json`
      );
      if (!inventoryResponse.ok) {
        throw new Error(`No data for ${participant}. Run pipeline steps 2-5 first.`);
      }
      const inventoryJson = await inventoryResponse.json();
      setInventoryData(inventoryJson);

      // Load lifecycle data (lifecycle.json) for timeline view
      const lifecycleResponse = await fetch(
        `${VIDEO_SERVER}/data/outputs/02_inventory/${participant}/${participant}_lifecycle.json`
      );
      if (lifecycleResponse.ok) {
        const lifecycleJson = await lifecycleResponse.json();
        setLifecycleData(lifecycleJson);
      }

      // Load timeline data for aggregated view
      // First try annotated (user's saved annotations), then fall back to aggregated (original)
      let loadedAggData = null;
      const annotatedResponse = await fetch(
        `${VIDEO_SERVER}/data/outputs/02_inventory/${participant}/${participant}_timeline_annotated.json`
      );
      if (annotatedResponse.ok) {
        loadedAggData = await annotatedResponse.json();
        setAggregatedData(loadedAggData);
        console.log('Loaded annotated timeline data');
      }

      if (!loadedAggData) {
        const aggregatedResponse = await fetch(
          `${VIDEO_SERVER}/data/outputs/02_inventory/${participant}/${participant}_timeline_aggregated.json`
        );
        if (aggregatedResponse.ok) {
          loadedAggData = await aggregatedResponse.json();
          setAggregatedData(loadedAggData);
          console.log('Loaded aggregated timeline data');
        }
      }

      // Load VLM QA results - discover available tags dynamically
      const loadedVlmData = {};
      let firstTag = null;

      try {
        const tagsResponse = await fetch(`${VIDEO_SERVER}/vlm-tags/${participant}`);
        if (tagsResponse.ok) {
          const tagsJson = await tagsResponse.json();
          const availableTags = tagsJson.tags || [];

          for (const { tag, filename } of availableTags) {
            try {
              const vlmResponse = await fetch(
                `${VIDEO_SERVER}/data/outputs/02_inventory/${participant}/${filename}`
              );
              if (vlmResponse.ok) {
                const vlmJson = await vlmResponse.json();
                loadedVlmData[tag] = enrichFrameResults(vlmJson, loadedAggData);
                if (!firstTag) firstTag = tag;
                console.log(`Loaded VLM QA results for tag '${tag}' from ${filename}`);
              }
            } catch (err) {
              console.warn(`Failed to load VLM results for tag '${tag}':`, err);
            }
          }
        }
      } catch (err) {
        console.warn('Failed to fetch VLM tags:', err);
      }

      setVlmDataByTag(loadedVlmData);
      const allTags = Object.keys(loadedVlmData);
      // Preserve current tag selection if it exists for the new participant
      if (selectedVlmTag && allTags.includes(selectedVlmTag)) {
        // keep current selection
      } else if (firstTag) {
        setSelectedVlmTag(firstTag);
      }

      // Auto-init comparison tags to first two available
      if (allTags.length >= 2) {
        setComparisonTagA(prev => allTags.includes(prev) ? prev : allTags[0]);
        setComparisonTagB(prev => allTags.includes(prev) ? prev : allTags[1]);
      } else if (allTags.length === 1) {
        setComparisonTagA(allTags[0]);
        setComparisonTagB(allTags[0]);
      }

      // Load failure cases list (cross-participant)
      try {
        const cacheBuster = `?t=${Date.now()}`;
        const failureCasesResponse = await fetch(`${VIDEO_SERVER}/failure-cases${cacheBuster}`);
        if (failureCasesResponse.ok) {
          const failureCasesJson = await failureCasesResponse.json();
          const files = failureCasesJson.files || [];
          setFailureCasesFiles(files);
          if (files.length > 0 && !selectedFailureCasesFile) {
            // Auto-select first file and load its data
            const firstFile = files[0];
            setSelectedFailureCasesFile(firstFile.filename);
            const dataResponse = await fetch(
              `${VIDEO_SERVER}/data/outputs/02_inventory/${firstFile.filename}${cacheBuster}`
            );
            if (dataResponse.ok) {
              let data = await dataResponse.json();
              // Handle v2 format (reference-based)
              if (data.schema_version === 2) {
                console.log(`Detected v2 failure cases, merging with timeline/vlm data...`);
                data = await mergeFailureCasesV2(data);
              }
              setFailureCasesData(data);
              console.log(`Loaded failure cases: ${firstFile.filename}`);
            }
          }
        }
      } catch (err) {
        console.warn('Failed to fetch failure cases:', err);
      }

      // Load narration timestamps from central JSON file
      const narrationsResponse = await fetch(`${VIDEO_SERVER}/narrations`);
      if (narrationsResponse.ok) {
        const narrationsJson = await narrationsResponse.json();
        const timestamps = parseNarrationTimestampsJSON(narrationsJson);
        setNarrationTimestamps(timestamps);
      }

      // Load hands23 detection data
      try {
        const hands23Response = await fetch(`${VIDEO_SERVER}/hands23/${participant}`);
        if (hands23Response.ok) {
          const hands23Json = await hands23Response.json();
          setHands23Data(hands23Json);
          console.log(`Loaded hands23 data: ${hands23Json.videos?.length || 0} videos`);
        } else {
          setHands23Data(null);
          console.log('No hands23 data available');
        }
      } catch (err) {
        setHands23Data(null);
        console.warn('Failed to load hands23 data:', err);
      }

      // Refresh available videos
      await checkServerStatus();

      const itemCount = inventoryJson.items?.length || 0;
      setSuccess(`Loaded ${itemCount} inventory items for ${participant}`);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  // Handle item selection
  const handleSelectItem = useCallback((item) => {
    setSelectedItem(item);
  }, []);

  // Handle narration click - load video and seek to timestamp
  const handleNarrationClick = useCallback((narrationId) => {
    const parsed = parseNarrationId(narrationId);
    if (!parsed) return;

    const { videoId } = parsed;
    const timestamp = narrationTimestamps[narrationId];

    // Build video URL from server
    const videoUrl = `${VIDEO_SERVER}/videos/${parsed.participant}/${videoId}.mp4`;

    if (currentVideo !== videoId) {
      setCurrentVideo(videoId);
      setCurrentVideoUrl(videoUrl);
      // Wait for video to load before seeking
      setTimeout(() => {
        if (timestamp !== undefined && videoRef.current) {
          videoRef.current.seekTo(timestamp);
          setCurrentTime(timestamp);
        }
      }, 500);
    } else if (timestamp !== undefined && videoRef.current) {
      videoRef.current.seekTo(timestamp);
      setCurrentTime(timestamp);
    }
  }, [narrationTimestamps, currentVideo]);

  const handleTimeUpdate = useCallback((time) => {
    setCurrentTime(time);
  }, []);

  // Handle video load with specific timestamp (for aggregated view)
  const handleLoadVideoAtTime = useCallback((videoId, timestamp) => {
    if (!videoId) return;

    // Extract participant from videoId (e.g., P01-20240202-110250 -> P01)
    const participantId = videoId.split('-')[0];
    const videoUrl = `${VIDEO_SERVER}/videos/${participantId}/${videoId}.mp4`;

    if (currentVideo !== videoId) {
      setCurrentVideo(videoId);
      setCurrentVideoUrl(videoUrl);
      // Wait for video to load before seeking
      setTimeout(() => {
        if (timestamp !== undefined && videoRef.current) {
          videoRef.current.seekTo(timestamp);
          setCurrentTime(timestamp);
        }
      }, 500);
    } else if (timestamp !== undefined && videoRef.current) {
      videoRef.current.seekTo(timestamp);
      setCurrentTime(timestamp);
    }
  }, [currentVideo]);

  // Handle save for aggregated data modifications - saves to _timeline_annotated.json
  const handleSaveAggregated = useCallback(async (updatedData) => {
    const response = await fetch(
      `${VIDEO_SERVER}/data/outputs/02_inventory/${participant}/${participant}_timeline_annotated.json`,
      {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(updatedData, null, 2),
      }
    );

    if (!response.ok) {
      throw new Error(`Failed to save: ${response.statusText}`);
    }

    // Update local state with saved data
    setAggregatedData(updatedData);
  }, [participant]);

  // Handle failure cases file selection
  const handleFailureCasesFileChange = useCallback(async (filename) => {
    setSelectedFailureCasesFile(filename);
    setFailureCasesData(null); // Clear stale data immediately while loading
    try {
      // Add cache-busting timestamp to force reload
      const cacheBuster = `?t=${Date.now()}`;
      const response = await fetch(
        `${VIDEO_SERVER}/data/outputs/02_inventory/${filename}${cacheBuster}`
      );
      if (response.ok) {
        let data = await response.json();

        // Handle v2 format (reference-based)
        if (data.schema_version === 2) {
          console.log(`Detected v2 failure cases, merging with timeline/vlm data...`);
          data = await mergeFailureCasesV2(data);
        }

        setFailureCasesData(data);
        console.log(`Loaded failure cases: ${filename}`);
      }
    } catch (err) {
      console.error('Failed to load failure cases:', err);
    }
  }, []);

  // Refresh failure cases list and data
  const refreshFailureCases = useCallback(async () => {
    try {
      const cacheBuster = `?t=${Date.now()}`;
      const failureCasesResponse = await fetch(`${VIDEO_SERVER}/failure-cases${cacheBuster}`);
      if (failureCasesResponse.ok) {
        const failureCasesJson = await failureCasesResponse.json();
        const files = failureCasesJson.files || [];
        setFailureCasesFiles(files);

        // Reload current file if selected
        if (selectedFailureCasesFile) {
          const dataResponse = await fetch(
            `${VIDEO_SERVER}/data/outputs/02_inventory/${selectedFailureCasesFile}${cacheBuster}`
          );
          if (dataResponse.ok) {
            let data = await dataResponse.json();
            // Handle v2 format (reference-based)
            if (data.schema_version === 2) {
              console.log(`Detected v2 failure cases, merging with timeline/vlm data...`);
              data = await mergeFailureCasesV2(data);
            }
            setFailureCasesData(data);
            console.log(`Refreshed failure cases: ${selectedFailureCasesFile}`);
          }
        }
      }
    } catch (err) {
      console.error('Failed to refresh failure cases:', err);
    }
  }, [selectedFailureCasesFile]);

  // Load a failure cases file as overlay (raw v2 data, not merged)
  const loadOverlayFailureCases = useCallback(async (filename) => {
    if (!filename) {
      setOverlayFailureCasesFile(null);
      setOverlayFailureCases(null);
      setOverlayDirty(false);
      return;
    }
    try {
      const cacheBuster = `?t=${Date.now()}`;
      const response = await fetch(
        `${VIDEO_SERVER}/data/outputs/02_inventory/${filename}${cacheBuster}`
      );
      if (response.ok) {
        const data = await response.json();
        setOverlayFailureCasesFile(filename);
        setOverlayFailureCases(data);
        setOverlayDirty(false);
        console.log(`Loaded overlay failure cases: ${filename} (${(data.cases || []).length} cases)`);
      }
    } catch (err) {
      console.error('Failed to load overlay failure cases:', err);
    }
  }, []);

  // Save the overlay failure cases back to server
  const saveOverlayFailureCases = useCallback(async () => {
    if (!overlayFailureCasesFile || !overlayFailureCases) return;
    try {
      const response = await fetch(
        `${VIDEO_SERVER}/data/outputs/02_inventory/${overlayFailureCasesFile}`,
        {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(overlayFailureCases, null, 2),
        }
      );
      if (response.ok) {
        setOverlayDirty(false);
        console.log(`Saved overlay failure cases: ${overlayFailureCasesFile}`);
      } else {
        console.error('Failed to save overlay failure cases:', response.statusText);
      }
    } catch (err) {
      console.error('Failed to save overlay failure cases:', err);
    }
  }, [overlayFailureCasesFile, overlayFailureCases]);

  // Toggle a segment in/out of the overlay failure cases
  const toggleSegmentInFailureCases = useCallback((participantId, narrationId, segmentId) => {
    if (!overlayFailureCases) return;
    setOverlayFailureCases(prev => {
      const cases = [...(prev.cases || [])];
      const existingIdx = cases.findIndex(c => c.segment_id === segmentId);
      if (existingIdx >= 0) {
        // Remove it
        cases.splice(existingIdx, 1);
      } else {
        // Add it — generate next case_id
        const maxNum = cases.reduce((max, c) => {
          const m = c.case_id?.match(/FC(\d+)/);
          return m ? Math.max(max, parseInt(m[1], 10)) : max;
        }, 0);
        cases.push({
          case_id: `FC${String(maxNum + 1).padStart(3, '0')}`,
          participant: participantId,
          narration_id: narrationId,
          segment_id: segmentId,
          include: true,
          priority: 0,
          notes: '',
          tags: [],
        });
      }
      return { ...prev, cases };
    });
    setOverlayDirty(true);
  }, [overlayFailureCases]);

  // Update notes on a failure case by segment_id
  const updateFailureCaseNotes = useCallback((segmentId, notes) => {
    if (!overlayFailureCases) return;
    setOverlayFailureCases(prev => {
      const cases = (prev.cases || []).map(c =>
        c.segment_id === segmentId ? { ...c, notes } : c
      );
      return { ...prev, cases };
    });
    setOverlayDirty(true);
  }, [overlayFailureCases]);

  // Build merged comparison data from two VLM tags (memoized for stable reference)
  const comparisonData = useMemo(() => {
    const dataA = vlmDataByTag[comparisonTagA];
    const dataB = vlmDataByTag[comparisonTagB];
    if (!dataA && !dataB) return { items: [] };

    const itemsA = dataA?.items || [];
    const itemsB = dataB?.items || [];

    // Index items by narration_id
    const mapA = {};
    itemsA.forEach(item => { if (item.narration_id) mapA[item.narration_id] = item; });
    const mapB = {};
    itemsB.forEach(item => { if (item.narration_id) mapB[item.narration_id] = item; });

    const allIds = new Set([...Object.keys(mapA), ...Object.keys(mapB)]);
    const mergedItems = [];

    allIds.forEach(nid => {
      const itemA = mapA[nid];
      const itemB = mapB[nid];
      const baseItem = itemA || itemB;

      const onlyInA = !!itemA && !itemB;
      const onlyInB = !itemA && !!itemB;

      // Index segments by segment_id
      const segsA = {};
      (itemA?.segments || []).forEach(s => { if (s.segment_id) segsA[s.segment_id] = s; });
      const segsB = {};
      (itemB?.segments || []).forEach(s => { if (s.segment_id) segsB[s.segment_id] = s; });

      const allSegIds = new Set([...Object.keys(segsA), ...Object.keys(segsB)]);
      let hasDifferences = false;

      const mergedSegments = [];
      allSegIds.forEach(sid => {
        const sA = segsA[sid];
        const sB = segsB[sid];
        const baseSeg = sA || sB;

        // Determine if predictions differ
        const differs = !!(sA && sB && sA.predicted_count !== sB.predicted_count);
        const missing = !sA || !sB;
        if (differs || missing) hasDifferences = true;

        mergedSegments.push({
          segment_id: sid,
          video_id: baseSeg.video_id,
          start_timestamp: baseSeg.start_timestamp,
          end_timestamp: baseSeg.end_timestamp,
          ground_truth_count: baseSeg.ground_truth_count,
          ground_truth_unit: baseSeg.ground_truth_unit,
          tagA: sA || null,
          tagB: sB || null,
          differs,
        });
      });

      mergedItems.push({
        narration_id: nid,
        food_name: baseItem.food_name,
        difficulty: baseItem.difficulty,
        recipe_amount: baseItem.recipe_amount,
        hasDifferences,
        onlyInA,
        onlyInB,
        segments: mergedSegments,
      });
    });

    return { items: mergedItems };
  }, [vlmDataByTag, comparisonTagA, comparisonTagB]);

  return (
    <div style={styles.app}>
      <div style={styles.header}>
        <span>🥕 Inventory Lifecycle Visualizer</span>
        <span
          style={{
            ...styles.serverStatus,
            ...(serverOnline ? styles.serverOnline : styles.serverOffline),
          }}
        >
          Server: {serverOnline ? `Online (${availableVideos.length} videos)` : 'Offline'}
        </span>
      </div>

      <div style={styles.content}>
        {/* Compact Toolbar */}
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '8px',
          padding: '4px 12px',
          marginBottom: '6px',
          backgroundColor: 'white',
          borderRadius: '6px',
          boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
          flexWrap: 'wrap',
          fontSize: '12px',
        }}>
          <select
            style={{ padding: '3px 6px', borderRadius: '4px', border: '1px solid #ddd', fontSize: '12px', fontWeight: '600' }}
            value={participant}
            onChange={(e) => setParticipant(e.target.value)}
          >
            {['P01','P02','P03','P04','P05','P06','P07','P08','P09'].map(p => (
              <option key={p} value={p}>{p}</option>
            ))}
          </select>

          <button
            onClick={loadFromServer}
            disabled={loading || !serverOnline}
            style={{
              ...styles.button,
              padding: '3px 10px',
              fontSize: '11px',
              height: 'auto',
              opacity: loading || !serverOnline ? 0.6 : 1,
              cursor: loading || !serverOnline ? 'not-allowed' : 'pointer',
            }}
          >
            {loading ? 'Loading...' : 'Load'}
          </button>

          {/* View Mode Toggle */}
          {inventoryData && (
            <div style={styles.viewToggle}>
              {[
                { key: 'items', label: 'Items' },
                { key: 'timeline', label: 'Timeline' },
                { key: 'aggregated', label: 'Aggregated', disabled: !aggregatedData },
                { key: 'vlm', label: 'VLM', disabled: Object.keys(vlmDataByTag).length === 0 },
                { key: 'failure-cases', label: 'FC', disabled: failureCasesFiles.length === 0 },
                { key: 'comparison', label: 'Compare', disabled: Object.keys(vlmDataByTag).length < 2 },
              ].map((tab, i, arr) => (
                <button
                  key={tab.key}
                  style={{
                    ...styles.viewToggleButton,
                    padding: '3px 10px',
                    fontSize: '11px',
                    ...(i === 0 ? styles.viewToggleButtonFirst : {}),
                    ...(i === arr.length - 1 ? styles.viewToggleButtonLast : {}),
                    ...(i > 0 ? { borderLeft: 'none' } : {}),
                    ...(viewMode === tab.key ? styles.viewToggleButtonActive : {}),
                  }}
                  onClick={() => setViewMode(tab.key)}
                  disabled={tab.disabled}
                >
                  {tab.label}
                </button>
              ))}
            </div>
          )}

          {inventoryData && (
            <span style={{ color: '#666', fontSize: '10px', marginLeft: 'auto' }}>
              {inventoryData.selected_count} items | L:{inventoryData.difficulty_breakdown?.LOW || 0} M:{inventoryData.difficulty_breakdown?.MID || 0} H:{inventoryData.difficulty_breakdown?.HIGH || 0}
            </span>
          )}

          {error && <span style={{ color: '#d32f2f', fontSize: '11px' }}>{error}</span>}
        </div>

        {!serverOnline && (
          <div style={{ ...styles.error, marginBottom: '6px', fontSize: '11px' }}>
            Server offline. Start with: <code>node video-server.js</code>
          </div>
        )}

        {/* Main Content */}
        {inventoryData && (
          <>

            {viewMode === 'items' ? (
              <div style={styles.mainLayout}>
                {/* Left Panel: Inventory List */}
                <InventoryItemList
                  items={inventoryData.items || []}
                  selectedItem={selectedItem}
                  onSelectItem={handleSelectItem}
                  onNarrationClick={handleNarrationClick}
                  narrationTimestamps={narrationTimestamps}
                />

                {/* Right Panel: Video Player */}
                <InventoryVideoPlayer
                  ref={videoRef}
                  videoUrl={currentVideoUrl}
                  videoId={currentVideo}
                  currentTime={currentTime}
                  onTimeUpdate={handleTimeUpdate}
                  onNarrationClick={handleNarrationClick}
                  selectedItem={selectedItem}
                  narrationTimestamps={narrationTimestamps}
                />
              </div>
            ) : viewMode === 'timeline' ? (
              <div style={{ ...styles.mainLayout, gridTemplateColumns: '1fr' }}>
                {/* Timeline View with integrated video player */}
                <TimelineView
                  ref={videoRef}
                  lifecycleData={lifecycleData}
                  onEventClick={handleNarrationClick}
                  narrationTimestamps={narrationTimestamps}
                  videoUrl={currentVideoUrl}
                  videoId={currentVideo}
                  currentTime={currentTime}
                  onTimeUpdate={handleTimeUpdate}
                />
              </div>
            ) : viewMode === 'aggregated' ? (
              <div style={{ ...styles.mainLayout, gridTemplateColumns: '1fr' }}>
                {/* Aggregated View with timestamp editing */}
                <AggregatedView
                  ref={videoRef}
                  aggregatedData={aggregatedData}
                  onLoadVideoAtTime={handleLoadVideoAtTime}
                  narrationTimestamps={narrationTimestamps}
                  videoUrl={currentVideoUrl}
                  videoId={currentVideo}
                  currentTime={currentTime}
                  onTimeUpdate={handleTimeUpdate}
                  onSave={handleSaveAggregated}
                  participant={participant}
                />
              </div>
            ) : viewMode === 'vlm' ? (
              <div style={{ ...styles.mainLayout, gridTemplateColumns: '1fr' }}>
                {/* VLM Results View */}
                <VLMResultsView
                  ref={videoRef}
                  vlmData={vlmDataByTag[selectedVlmTag]}
                  availableTags={Object.keys(vlmDataByTag)}
                  selectedTag={selectedVlmTag}
                  onTagChange={setSelectedVlmTag}
                  onLoadVideoAtTime={handleLoadVideoAtTime}
                  videoUrl={currentVideoUrl}
                  videoId={currentVideo}
                  currentTime={currentTime}
                  onTimeUpdate={handleTimeUpdate}
                  hands23Data={hands23Data}
                  participant={participant}
                  failureCasesFiles={failureCasesFiles}
                  overlayFailureCasesFile={overlayFailureCasesFile}
                  overlayFailureCases={overlayFailureCases}
                  overlayDirty={overlayDirty}
                  onLoadOverlayFailureCases={loadOverlayFailureCases}
                  onSaveOverlayFailureCases={saveOverlayFailureCases}
                  onToggleSegmentInFailureCases={toggleSegmentInFailureCases}
                  onUpdateFailureCaseNotes={updateFailureCaseNotes}
                />
              </div>
            ) : viewMode === 'failure-cases' ? (
              <div style={{ ...styles.mainLayout, gridTemplateColumns: '1fr' }}>
                {/* Failure Cases View - reuses VLMResultsView in FC mode */}
                <VLMResultsView
                  ref={videoRef}
                  vlmData={failureCasesData}
                  availableTags={failureCasesFiles.map(f => f.filename)}
                  selectedTag={selectedFailureCasesFile}
                  onTagChange={handleFailureCasesFileChange}
                  onRefresh={refreshFailureCases}
                  onLoadVideoAtTime={handleLoadVideoAtTime}
                  videoUrl={currentVideoUrl}
                  videoId={currentVideo}
                  currentTime={currentTime}
                  onTimeUpdate={handleTimeUpdate}
                  hands23Data={hands23Data}
                  participant={participant}
                  failureCasesMode
                />
              </div>
            ) : viewMode === 'comparison' ? (
              <div style={{ ...styles.mainLayout, gridTemplateColumns: '1fr' }}>
                {/* Comparison View - side-by-side two VLM tags */}
                <ComparisonView
                  ref={videoRef}
                  comparisonData={comparisonData}
                  tagA={comparisonTagA}
                  tagB={comparisonTagB}
                  availableTags={Object.keys(vlmDataByTag)}
                  onTagAChange={setComparisonTagA}
                  onTagBChange={setComparisonTagB}
                  onLoadVideoAtTime={handleLoadVideoAtTime}
                  videoUrl={currentVideoUrl}
                  videoId={currentVideo}
                  currentTime={currentTime}
                  onTimeUpdate={handleTimeUpdate}
                  participant={participant}
                  failureCasesFiles={failureCasesFiles}
                  overlayFailureCasesFile={overlayFailureCasesFile}
                  overlayFailureCases={overlayFailureCases}
                  overlayDirty={overlayDirty}
                  onLoadOverlayFailureCases={loadOverlayFailureCases}
                  onSaveOverlayFailureCases={saveOverlayFailureCases}
                  onToggleSegmentInFailureCases={toggleSegmentInFailureCases}
                  onUpdateFailureCaseNotes={updateFailureCaseNotes}
                />
              </div>
            ) : null}
          </>
        )}
      </div>
    </div>
  );
}

export default InventoryApp;

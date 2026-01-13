/**
 * Parse narration timestamps from JSON content (from HD_EPIC_Narrations.pkl converted to JSON)
 * @param {Object} jsonData - Parsed JSON object { narration_id: { timestamp, video_id, ... } }
 * @returns {Object} Map of narration_id to timestamp
 */
export function parseNarrationTimestampsJSON(jsonData) {
  const timestamps = {};
  for (const [narrationId, data] of Object.entries(jsonData)) {
    if (data.timestamp !== undefined && !isNaN(data.timestamp)) {
      timestamps[narrationId] = data.timestamp;
    }
  }
  return timestamps;
}

/**
 * Parse narration timestamps from CSV content
 * CSV format: unique_narration_id,participant_id,video_id,narration,start_timestamp,end_timestamp,...
 * @param {string} csvContent - Raw CSV file content
 * @returns {Object} Map of narration_id to start_timestamp
 */
export function parseNarrationTimestamps(csvContent) {
  const lines = csvContent.split('\n');
  const timestamps = {};

  // Skip header line
  for (let i = 1; i < lines.length; i++) {
    const line = lines[i].trim();
    if (!line) continue;

    // Parse CSV line (handling quoted fields)
    const fields = parseCSVLine(line);
    if (fields.length >= 5) {
      const narrationId = fields[0];
      const startTimestamp = parseFloat(fields[4]);

      if (narrationId && !isNaN(startTimestamp)) {
        timestamps[narrationId] = startTimestamp;
      }
    }
  }

  return timestamps;
}

/**
 * Parse a single CSV line, handling quoted fields
 */
function parseCSVLine(line) {
  const fields = [];
  let current = '';
  let inQuotes = false;

  for (let i = 0; i < line.length; i++) {
    const char = line[i];

    if (char === '"') {
      inQuotes = !inQuotes;
    } else if (char === ',' && !inQuotes) {
      fields.push(current.trim());
      current = '';
    } else {
      current += char;
    }
  }

  fields.push(current.trim());
  return fields;
}

/**
 * Parse narration ID to extract video ID and narration number
 * Format: P01-20240202-110250-29
 * @param {string} narrationId
 * @returns {Object|null} { participant, videoId, narrationNumber }
 */
export function parseNarrationId(narrationId) {
  const match = narrationId.match(/^(P\d+)-(\d+-\d+)-(\d+)$/);
  if (!match) return null;

  return {
    participant: match[1],
    videoId: `${match[1]}-${match[2]}`,
    narrationNumber: parseInt(match[3], 10),
  };
}

/**
 * Format timestamp as MM:SS.s
 */
export function formatTimestamp(seconds) {
  if (isNaN(seconds) || seconds === null || seconds === undefined) {
    return '--:--';
  }
  const mins = Math.floor(seconds / 60);
  const secs = (seconds % 60).toFixed(1);
  return `${mins.toString().padStart(2, '0')}:${secs.padStart(4, '0')}`;
}

/**
 * Get stage color for display
 */
export function getStageColor(stage) {
  const colors = {
    'RETRIEVAL': '#4CAF50',
    'ACCESS': '#2196F3',
    'DISPENSING': '#FF9800',
    'RESTOCKING': '#9C27B0',
    'DISCARD': '#f44336',
  };
  return colors[stage] || '#757575';
}

/**
 * Get difficulty color for display
 */
export function getDifficultyColor(difficulty) {
  const colors = {
    'LOW': '#4CAF50',
    'MID': '#FF9800',
    'HIGH': '#f44336',
  };
  return colors[difficulty] || '#757575';
}

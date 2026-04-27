import React, { useState, useEffect, useMemo, useCallback } from 'react';
import IterativeTrace from './IterativeTrace';

const API = 'http://localhost:4002';

const ITEM_PALETTE = [
  '#FF6B6B','#4ECDC4','#45B7D1','#96CEB4','#FFEAA7','#DDA0DD','#98D8C8',
  '#F7DC6F','#BB8FCE','#85C1E9','#F8C471','#82E0AA','#F1948A','#AED6F1',
  '#D7BDE2','#A3E4D7','#FAD7A0','#A9CCE3','#D5DBDB','#F9E79F',
];

function formatAmt(val, unit) {
  if (val == null) return '—';
  if (unit === 'count') return Number.isInteger(val) ? String(val) : val.toFixed(1);
  return `${Math.round(val)}g`;
}

function formatTime(seconds) {
  if (seconds == null || isNaN(seconds)) return '--:--';
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  return `${m}:${s.toFixed(1).padStart(4, '0')}`;
}

export default function AmountView({
  participant, session, globalTime, totalDuration, seekTo, ledger, actions: gtActions,
  videoRef, playing, togglePlay, activeClipIdx, clipMap, onTimeUpdate, onEnded, setPlaying, scrubbingRef,
  onFilterSessions,
}) {
  const [tags, setTags] = useState([]);
  const [selectedTag, setSelectedTag] = useState(null);
  const [data, setData] = useState(null);

  const [selectedItem, setSelectedItem] = useState(null);
  const [itemLog, setItemLog] = useState(null);
  const [filterItem, setFilterItem] = useState(null);
  const [promptExpanded, setPromptExpanded] = useState(false);
  const [detData, setDetData] = useState(null);       // hands23 + siglip + dino + scene per frame
  const [selectedDetFrame, setSelectedDetFrame] = useState(null);

  // Item search/filter state
  const [itemSearch, setItemSearch] = useState('');
  const [activeItemId, setActiveItemId] = useState(null);
  const [searchOpen, setSearchOpen] = useState(false);

  // All items from ledger, sorted by visual_class
  const allItems = useMemo(() => {
    if (!ledger?.items) return [];
    return Object.entries(ledger.items)
      .map(([id, item]) => ({ instance_id: id, visual_class: item.visual_class || id }))
      .sort((a, b) => a.visual_class.localeCompare(b.visual_class));
  }, [ledger]);

  // Filtered items based on search text
  const searchResults = useMemo(() => {
    if (!itemSearch.trim()) return [];
    const q = itemSearch.toLowerCase();
    return allItems.filter(it => it.visual_class.toLowerCase().includes(q) || it.instance_id.toLowerCase().includes(q));
  }, [allItems, itemSearch]);

  // Count sessions for active item
  const activeItemSessionCount = useMemo(() => {
    if (!activeItemId || !ledger?.snapshots) return 0;
    return Object.keys(ledger.snapshots).filter(s => ledger.snapshots[s][activeItemId]).length;
  }, [activeItemId, ledger]);

  // Apply session filter when active item changes
  useEffect(() => {
    if (!onFilterSessions) return;
    if (!activeItemId || !ledger?.snapshots) {
      onFilterSessions(null);
      return;
    }
    const matching = Object.keys(ledger.snapshots)
      .filter(s => ledger.snapshots[s][activeItemId])
      .sort();
    onFilterSessions(matching.length > 0 ? matching : null);
  }, [activeItemId, ledger, onFilterSessions]);

  // Clear filter on unmount
  useEffect(() => {
    return () => { if (onFilterSessions) onFilterSessions(null); };
  }, [onFilterSessions]);

  const handleSelectItem = useCallback((instanceId) => {
    const item = allItems.find(it => it.instance_id === instanceId);
    setActiveItemId(instanceId);
    setItemSearch(item ? item.visual_class : instanceId);
    setSearchOpen(false);
    setFilterItem(instanceId);
    setSelectedItem(instanceId);
  }, [allItems]);

  const handleClearItemFilter = useCallback(() => {
    setActiveItemId(null);
    setItemSearch('');
    setSearchOpen(false);
    setFilterItem(null);
  }, []);

  // Load available tags
  useEffect(() => {
    if (!participant) return;
    fetch(`${API}/api/participants/${participant}/amount-tags`)
      .then(r => r.json())
      .then(t => { setTags(t); if (t.length && !selectedTag) setSelectedTag(t[0].tag); })
      .catch(() => setTags([]));
  }, [participant]);

  // Load session data for selected tag
  useEffect(() => {
    if (!participant || !session || !selectedTag) return;
    fetch(`${API}/api/participants/${participant}/sessions/${session}/amount-results/${selectedTag}`)
      .then(r => r.ok ? r.json() : null)
      .then(d => setData(d))
      .catch(() => setData(null));
  }, [participant, session, selectedTag]);

  // Load per-frame detections (hands23 + siglip + dino) and OWLv2 scene tags
  // for AVP runs. These replace the AdaTAD timeline bar.
  useEffect(() => {
    if (!participant || !session || !data?.pipeline || (data.pipeline !== 'avp' && data.pipeline !== 'iterative')) {
      setDetData(null);
      return;
    }
    let cancelled = false;
    Promise.all([
      fetch(`${API}/api/participants/${participant}/sessions/${session}/hands23-results`).then(r => r.ok ? r.json() : null).catch(() => null),
      fetch(`${API}/api/participants/${participant}/sessions/${session}/owlv2-scene`).then(r => r.ok ? r.json() : null).catch(() => null),
    ]).then(([h23, scene]) => {
      if (cancelled) return;
      if (!h23) { setDetData(null); return; }
      const sceneByT = {};
      if (scene?.frames) {
        for (const v of Object.values(scene.frames)) {
          if (v?.timestamp != null) sceneByT[Math.round(v.timestamp * 100) / 100] = v.scene || 'unknown';
        }
      }
      const frames = (h23.frames || []).map(f => {
        const key = Math.round(f.session_timestamp_s * 100) / 100;
        return { ...f, scene_tag: sceneByT[key] || null };
      });
      setDetData({ ...h23, frames });
    });
    return () => { cancelled = true; };
  }, [participant, session, data?.pipeline]);

  const actions = gtActions || [];

  // Load item log when selected item changes
  useEffect(() => {
    if (!participant || !session || !selectedTag || !selectedItem) { setItemLog(null); return; }
    fetch(`${API}/api/participants/${participant}/sessions/${session}/amount-item-log/${selectedTag}/${selectedItem}`)
      .then(r => r.ok ? r.json() : null)
      .then(log => setItemLog(log))
      .catch(() => setItemLog(null));
  }, [participant, session, selectedTag, selectedItem]);

  useEffect(() => { setSelectedItem(null); setItemLog(null); }, [session]);

  // Lookup maps
  const itemColorMap = useMemo(() => {
    if (!data) return {};
    const map = {};
    data.inventory.forEach((inv, i) => { map[inv.instance_id] = ITEM_PALETTE[i % ITEM_PALETTE.length]; });
    return map;
  }, [data]);

  const evalMap = useMemo(() => {
    if (!data) return {};
    const m = {};
    for (const e of data.eval_entries) m[e.instance_id] = e;
    return m;
  }, [data]);

  const predMap = useMemo(() => {
    if (!data) return {};
    const m = {};
    for (const p of data.predictions) m[p.instance_id] = p;
    return m;
  }, [data]);

  // Timeline segments — load GT actions from actions.json. Drop only the
  // VLM-overlay actions (source starts with 'vlm'); everything else (human,
  // human_verified, no source) is rendered on the GT row.
  const gtSegments = useMemo(() =>
    actions
      .filter(a => !a.source || !String(a.source).startsWith('vlm'))
      .map(a => ({ ...a, source: 'gt' })),
  [actions]);

  const vlmSegments = useMemo(() => {
    if (!data) return [];
    const segs = [];
    const padded = data.vlm_input_segments_by_iid || {};
    const havePadded = Object.keys(padded).length > 0;

    if (havePadded) {
      const predByIid = {};
      for (const p of data.predictions) predByIid[p.instance_id] = p;
      for (const [iid, bands] of Object.entries(padded)) {
        const pred = predByIid[iid];
        for (const band of bands) {
          segs.push({ start: band.start, end: band.end, item: iid, action: pred ? pred.item : iid, source: 'vlm_input' });
        }
      }
      return segs;
    }

    // Fallback: raw pred.segments (already normalized to {start, end} objects)
    for (const pred of data.predictions) {
      for (const seg of (pred.segments || [])) {
        segs.push({ start: seg.start, end: seg.end, item: pred.instance_id, action: pred.item, source: 'vlm_input' });
      }
    }
    return segs;
  }, [data]);

  // Per-frame detection ticks (replaces the AdaTAD bar). Each tick is one
  // HOI-contact frame, colored by the strongest inventory iid match (DINO
  // top, falling back to SigLIP top). Frames without an inventory match get
  // a neutral colour so you still see HOI activity timing.
  const DET_MIN_SCORE = 0.15;
  const detectionSegments = useMemo(() => {
    if (!detData?.frames) return [];
    const vcToIid = {};
    if (data?.inventory) {
      for (const inv of data.inventory) {
        // Map visual_class (used in siglip_matches.food_name) to an instance_id
        // so we can color the tick. If duplicate iids share a class, last one
        // wins — fine for display.
        if (inv.visual_class) vcToIid[inv.visual_class.toLowerCase()] = inv.instance_id;
      }
    }
    const out = [];
    for (const f of detData.frames) {
      const hasContact = (f.detections || []).some(d => d.contact_state === 'object_contact');
      if (!hasContact) continue; // skip non-contact frames to avoid clutter
      // Pick the strongest inventory match across all hands for this frame.
      let topIid = null; let topScore = 0;
      for (const m of (f.dino_matches || [])) {
        for (const tm of (m.top_matches || [])) {
          if (tm.similarity >= DET_MIN_SCORE && tm.similarity > topScore) {
            topScore = tm.similarity;
            topIid = tm.instance_id;
          }
        }
      }
      if (!topIid) {
        for (const m of (f.siglip_matches || [])) {
          for (const tm of (m.top_matches || [])) {
            if (tm.similarity >= DET_MIN_SCORE && tm.similarity > topScore) {
              topScore = tm.similarity;
              topIid = vcToIid[(tm.food_name || '').toLowerCase()] || null;
            }
          }
        }
      }
      const t = f.session_timestamp_s;
      out.push({
        start: t - 0.25, end: t + 0.25,
        item: topIid || '_contact_', // unique sentinel so unmatched ticks aren't filtered out
        action: topIid ? `${topIid.replace(/_\d{8}$/, '')} (${topScore.toFixed(2)})` : `contact · ${f.scene_tag || '?'}`,
        source: 'detection', _raw: f, _idx: out.length,
      });
    }
    return out;
  }, [detData, data?.inventory]);

  // Planner segments (AVP: segments the observer receives)
  const plannerSegments = useMemo(() => {
    if (!data || data.pipeline !== 'avp') return [];
    const segs = [];
    for (const pred of data.predictions) {
      for (const seg of (pred.segments || [])) {
        segs.push({
          start: seg.start, end: seg.end, item: pred.instance_id,
          action: `${pred.item} (${pred.planner_confidence || '?'})`, source: 'planner',
        });
      }
    }
    return segs;
  }, [data]);

  // Iterative trace (per-round planner/observer log) — loaded separately so we
  // can split timeline segments by round. When the selected tag isn't
  // iterative, this stays null and the code falls back to the legacy view.
  const [iterTrace, setIterTrace] = useState(null);
  useEffect(() => {
    if (!participant || !session || !selectedTag || data?.pipeline !== 'iterative') {
      setIterTrace(null);
      return;
    }
    let cancelled = false;
    fetch(`${API}/api/participants/${participant}/sessions/${session}/iterative-trace/${selectedTag}`)
      .then(r => r.ok ? r.json() : null)
      .then(t => { if (!cancelled) setIterTrace(t); })
      .catch(() => { if (!cancelled) setIterTrace(null); });
    return () => { cancelled = true; };
  }, [participant, session, selectedTag, data?.pipeline]);

  // Iterative per-round segments (for the multi-band timeline).
  // Each observer_round contributes N segments (one per segment in its
  // window.segments list) under the round number it belongs to.
  // Sweep runs (avp_minimal_remaining_*) attach `window.target_items` —
  // a parallel array of [iid, ...] per segment — so each segment may
  // belong to multiple iids. Emit one row per (segment, target_iid) so
  // the per-iid filter/coloring works for shared windows.
  const iterSegmentsByRound = useMemo(() => {
    if (!iterTrace?.observer_rounds) return {};
    const byRound = {};
    for (const r of iterTrace.observer_rounds) {
      const roundNum = r.round || 0;
      const win = r.window || {};
      const segs = win.segments || [];
      const sweepTargets = Array.isArray(win.target_items) ? win.target_items : null;
      const fallbackIid = (win.candidate_instance_ids || [])[0] || null;
      for (let i = 0; i < segs.length; i++) {
        const s = segs[i];
        if (!Array.isArray(s) || s.length < 2) continue;
        if (!byRound[roundNum]) byRound[roundNum] = [];
        // Determine which iids this segment belongs to.
        let iids;
        if (sweepTargets && Array.isArray(sweepTargets[i])) {
          iids = sweepTargets[i].length ? sweepTargets[i] : [fallbackIid || win.visual_class || '_'];
        } else {
          iids = [fallbackIid || win.visual_class || '_'];
        }
        const tgtCount = (sweepTargets && Array.isArray(sweepTargets[i])) ? sweepTargets[i].length : 0;
        for (const iid of iids) {
          byRound[roundNum].push({
            start: s[0], end: s[1],
            item: iid,
            action: tgtCount > 1
              ? `R${roundNum} · ${tgtCount} co-targets`
              : `${win.visual_class || iid || '?'} (R${roundNum})`,
            source: 'iter_observer',
          });
        }
      }
    }
    return byRound;
  }, [iterTrace]);

  const iterRoundNumbers = useMemo(
    () => Object.keys(iterSegmentsByRound).map(n => parseInt(n)).sort((a,b) => a-b),
    [iterSegmentsByRound]
  );

  // Filtered segments
  const filteredGt = useMemo(() => filterItem ? gtSegments.filter(s => s.item === filterItem) : gtSegments, [gtSegments, filterItem]);
  const filteredVlm = useMemo(() => filterItem ? vlmSegments.filter(s => s.item === filterItem) : vlmSegments, [vlmSegments, filterItem]);
  // Detections: when filterItem is set, only keep ticks whose top-matched iid
  // is the filter (unmatched ticks are hidden during a filter). With no filter
  // we render everything including the grey `_contact_` ticks.
  const filteredDet = useMemo(() => filterItem ? detectionSegments.filter(s => s.item === filterItem) : detectionSegments, [detectionSegments, filterItem]);
  const filteredPlanner = useMemo(() => filterItem ? plannerSegments.filter(s => s.item === filterItem) : plannerSegments, [plannerSegments, filterItem]);

  const isAvp = data?.pipeline === 'avp';
  const isIterative = data?.pipeline === 'iterative';
  const isWholevideo = data?.pipeline === 'wholevideo';
  const sr = data?.session_reasoning;

  // Per-round filtered segments for the Iterative timeline.
  const filteredIterByRound = useMemo(() => {
    const out = {};
    for (const n of iterRoundNumbers) {
      const all = iterSegmentsByRound[n] || [];
      out[n] = filterItem ? all.filter(s => s.item === filterItem) : all;
    }
    return out;
  }, [iterSegmentsByRound, iterRoundNumbers, filterItem]);

  const handleItemClick = useCallback((instanceId) => {
    setSelectedItem(prev => prev === instanceId ? null : instanceId);
    setFilterItem(prev => prev === instanceId ? null : instanceId);
    setPromptExpanded(false);
  }, []);

  const handleTimelineClick = useCallback((e, containerRef) => {
    if (!containerRef.current || !totalDuration) return;
    const rect = containerRef.current.getBoundingClientRect();
    const pct = (e.clientX - rect.left) / rect.width;
    seekTo(pct * totalDuration);
  }, [totalDuration, seekTo]);

  if (!participant || !session) {
    return <div style={S.empty}>Select a participant and session</div>;
  }

  const inventory = data?.inventory || [];
  const selectedEval = selectedItem ? evalMap[selectedItem] : null;
  const selectedPred = selectedItem ? predMap[selectedItem] : null;

  // Tag display label
  const tagLabel = (t) => {
    if (t.pipeline === 'avp') return `[AVP] ${t.tag}`;
    if (t.pipeline === 'iterative') return `[ITR] ${t.tag}`;
    if (t.pipeline === 'wholevideo') return `[WV] ${t.tag}`;
    return t.tag;
  };

  return (
    <div style={S.main}>
      {/* Left panel: video + timeline */}
      <div style={S.leftPanel}>
        <div style={S.videoWrap}>
          <video
            ref={videoRef} style={S.video}
            onTimeUpdate={onTimeUpdate} onEnded={onEnded}
            onPlay={() => setPlaying(true)} onPause={() => setPlaying(false)}
          />
          <div style={S.videoOverlay}>
            Clip {activeClipIdx + 1}/{(clipMap || []).length} &nbsp;
            {(clipMap || [])[activeClipIdx]?.filename}
          </div>
        </div>

        <div style={S.transport}>
          <button onClick={togglePlay} style={S.playBtn}>{playing ? '\u23F8' : '\u25B6'}</button>
          <span style={S.timeDisplay}>{formatTime(globalTime)}</span>
          <input
            type="range" min={0} max={totalDuration || 1} step={0.1} value={globalTime}
            onPointerDown={() => { if (scrubbingRef) scrubbingRef.current = true; }}
            onPointerUp={() => { if (scrubbingRef) scrubbingRef.current = false; }}
            onChange={e => seekTo(parseFloat(e.target.value))}
            style={S.scrubber}
          />
          <span style={S.timeDisplay}>{formatTime(totalDuration)}</span>
        </div>

        <div style={S.clipBar}>
          {(clipMap || []).map((c, i) => (
            <div
              key={i}
              style={{
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                width: `${(c.duration / (totalDuration || 1)) * 100}%`,
                background: i === activeClipIdx ? '#4CAF50' : i % 2 === 0 ? '#444' : '#555',
                cursor: 'pointer', borderRight: '1px solid #222',
              }}
              onClick={() => seekTo(c.offset)}
              title={`${c.filename} (${formatTime(c.duration)})`}
            >
              <span style={{ fontSize: 9, color: '#fff' }}>{i + 1}</span>
            </div>
          ))}
        </div>

        {/* Tag selector + item toggles */}
        <div style={S.header}>
          <div style={S.headerLeft}>
            <label style={S.label}>Input: </label>
            <select value={selectedTag || ''} onChange={e => setSelectedTag(e.target.value)} style={S.select}>
              {tags.map(t => <option key={t.tag} value={t.tag}>{tagLabel(t)}</option>)}
            </select>
          </div>
          <div style={S.itemToggles}>
            <button
              style={{ ...S.toggleBtn, background: !filterItem ? '#4CAF50' : '#333' }}
              onClick={() => setFilterItem(null)}
            >All</button>
            {inventory.map(inv => (
              <button
                key={inv.instance_id}
                style={{
                  ...S.toggleBtn,
                  background: filterItem === inv.instance_id ? itemColorMap[inv.instance_id] : '#333',
                  color: filterItem === inv.instance_id ? '#000' : '#ccc',
                  borderColor: itemColorMap[inv.instance_id],
                }}
                onClick={() => handleItemClick(inv.instance_id)}
              >{inv.visual_class}</button>
            ))}
          </div>
        </div>

        {/* Timeline */}
        <TimelineBar
          rows={isIterative ? [
            { label: 'GT', segments: filteredGt, opacity: 0.7 },
            { label: 'Detections', segments: filteredDet, opacity: 0.6 },
            ...iterRoundNumbers.map(n => ({
              label: `R${n}`,
              segments: filteredIterByRound[n] || [],
              opacity: 1.0,
            })),
          ] : isAvp ? [
            { label: 'GT', segments: filteredGt, opacity: 0.7 },
            { label: 'Detections', segments: filteredDet, opacity: 0.6 },
            { label: 'VLM In', segments: filteredVlm.length ? filteredVlm : filteredPlanner, opacity: 1.0 },
          ] : isWholevideo ? [
            { label: 'GT', segments: filteredGt, opacity: 0.7 },
          ] : [
            { label: 'GT', segments: filteredGt, opacity: 0.7 },
            { label: 'VLM In', segments: filteredVlm, opacity: 1.0 },
          ]}
          onSegmentClick={(seg) => {
            if (seg.source === 'detection' && seg._raw) {
              setSelectedDetFrame(seg._raw);
              seekTo(seg._raw.session_timestamp_s);
            } else {
              setSelectedDetFrame(null);
            }
          }}
          totalDuration={totalDuration} globalTime={globalTime}
          itemColorMap={itemColorMap} seekTo={seekTo}
        />
      </div>

      {/* Right panel: inventory + detail */}
      <div style={S.rightPanel}>

      {/* Item search / cross-session filter */}
      <div style={S.itemFilterBar}>
        <div style={{ position: 'relative', flex: 1 }}>
          <input
            type="text"
            placeholder="Filter by item..."
            value={itemSearch}
            onChange={e => { setItemSearch(e.target.value); setSearchOpen(true); if (!e.target.value) handleClearItemFilter(); }}
            onFocus={() => { if (itemSearch) setSearchOpen(true); }}
            onBlur={() => setTimeout(() => setSearchOpen(false), 150)}
            style={S.itemSearchInput}
          />
          {activeItemId && (
            <span
              style={S.itemFilterClear}
              onClick={handleClearItemFilter}
              title="Clear item filter"
            >x</span>
          )}
          {searchOpen && searchResults.length > 0 && (
            <div style={S.itemSearchDropdown}>
              {searchResults.slice(0, 12).map(it => (
                <div
                  key={it.instance_id}
                  style={{
                    ...S.itemSearchOption,
                    background: it.instance_id === activeItemId ? '#2a3a2a' : 'transparent',
                  }}
                  onClick={() => handleSelectItem(it.instance_id)}
                >
                  <span style={{ color: '#ddd' }}>{it.visual_class}</span>
                  <span style={{ color: '#666', fontSize: 9, marginLeft: 6 }}>{it.instance_id}</span>
                </div>
              ))}
            </div>
          )}
        </div>
        {activeItemId && (
          <span style={{ fontSize: 10, color: '#FFA726', whiteSpace: 'nowrap', marginLeft: 6 }}>
            {activeItemSessionCount} sessions
          </span>
        )}
      </div>

      <div style={S.inventorySection}>
        <div style={S.sectionTitle}>
          Session Inventory
          {selectedTag && <span style={{ fontSize: 10, color: '#888', marginLeft: 8 }}>({selectedTag})</span>}
        </div>
        <div style={S.tableHeader}>
          <div style={S.tableColItem}>Item</div>
          <div style={S.tableColNum}>GT Used</div>
          <div style={S.tableColNum}>Pred</div>
          <div style={S.tableColCnpe}>CNPE</div>
          <div style={S.tableColNum}>GT Rem</div>
          <div style={S.tableColNum}>Pred</div>
          <div style={S.tableColCnpe}>CNPE</div>
        </div>
        {inventory.map(inv => {
          const pred = predMap[inv.instance_id];
          const ev = evalMap[inv.instance_id];
          const isSelected = selectedItem === inv.instance_id;
          const cnpeU = ev?.cnpe_used;
          const cnpeR = ev?.cnpe_rem;
          const cnpeColor = (v) => v == null ? '#555' : v < 5 ? '#66BB6A' : v < 15 ? '#FFA726' : '#EF5350';
          return (
            <div
              key={inv.instance_id}
              style={{
                ...S.tableRow,
                background: isSelected ? '#2a2a2a' : 'transparent',
                borderLeft: `3px solid ${itemColorMap[inv.instance_id] || '#555'}`,
              }}
              onClick={() => handleItemClick(inv.instance_id)}
            >
              <div style={S.tableColItem} title={inv.instance_id}>{inv.visual_class}</div>
              {/* Prefer eval's tare-corrected values over raw ledger snapshot */}
              <div style={{ ...S.tableColNum, color: '#66BB6A' }}>{formatAmt(ev?.used ?? inv.gt_used, inv.unit)}</div>
              <div style={{ ...S.tableColNum, color: pred ? '#fff' : '#555' }}>{pred ? formatAmt(pred.amount_used ?? pred.amount_derivative, inv.unit) : '—'}</div>
              <div style={{ ...S.tableColCnpe, color: cnpeColor(cnpeU) }}>{cnpeU != null ? `${cnpeU.toFixed(1)}%` : '—'}</div>
              <div style={{ ...S.tableColNum, color: '#66BB6A' }}>{formatAmt(ev?.after ?? inv.gt_remaining, inv.unit)}</div>
              <div style={{ ...S.tableColNum, color: pred ? '#fff' : '#555' }}>
                {pred
                  ? (pred.amount_remaining != null
                      ? <>{formatAmt(pred.amount_remaining, inv.unit)}{pred.amount_kind === 'computed_remaining' && <span style={{ fontSize: 9, color: '#888', marginLeft: 3 }}>(s−d)</span>}</>
                      : pred.amount_starting != null
                        ? <>{formatAmt(pred.amount_starting, inv.unit)}<span style={{ fontSize: 9, color: '#FFA726', marginLeft: 3 }}>[start]</span></>
                        : '—')
                  : '—'}
              </div>
              <div style={{ ...S.tableColCnpe, color: cnpeColor(cnpeR) }}>{cnpeR != null ? `${cnpeR.toFixed(1)}%` : '—'}</div>
            </div>
          );
        })}
      </div>

      {/* Iterative: swap everything below the predictions table for the
          per-round plan-observe trace. */}
      {isIterative ? (
        <IterativeTrace
          participant={participant}
          session={session}
          selectedTag={selectedTag}
          seekTo={seekTo}
          inventory={inventory}
          selectedItem={selectedItem}
        />
      ) : (<>

      {/* Activity summary (AVP planner) */}
      {sr?.activity_summary && (
        <div style={{ borderBottom: '1px solid #333' }}>
          <div style={S.sectionTitle}>Activity Summary</div>
          <pre style={{ ...S.traceContent, margin: '0 10px 8px 10px', maxHeight: 150 }}>
            {sr.activity_summary}
          </pre>
        </div>
      )}

      {/* Session reasoning (wholevideo thinking trace) */}
      {sr?.thinking && (
        <div style={{ borderBottom: '1px solid #333' }}>
          <div style={S.sectionTitle}>
            Session Reasoning
            {sr.stats?.total_tokens && (
              <span style={{ fontSize: 10, color: '#888', marginLeft: 8 }}>
                ({sr.stats.total_tokens.toLocaleString()} tok{sr.stats.inference_time_s ? `, ${sr.stats.inference_time_s}s` : ''})
              </span>
            )}
          </div>
          <pre style={{ ...S.traceContent, margin: '0 10px 8px 10px', maxHeight: 300 }}>
            {sr.thinking}
          </pre>
        </div>
      )}

      {/* Session prompt (collapsible, any pipeline)
          Auto-expanded when no item is selected (useful for AVP planner prompt). */}
      {sr?.prompt && (() => {
        const showPrompt = !selectedItem || promptExpanded;
        return (
        <div style={{ borderBottom: '1px solid #333' }}>
          <div style={{ ...S.sectionTitle, cursor: 'pointer' }} onClick={() => setPromptExpanded(!promptExpanded)}>
            {showPrompt ? '▼' : '▶'} Session Prompt {!selectedItem && <span style={{ fontSize: 10, color: '#888' }}>(click item to collapse)</span>}
          </div>
          {showPrompt && (
            <pre style={{ ...S.traceContent, margin: '0 10px 8px 10px', maxHeight: 400 }}>
              {sr.prompt}
            </pre>
          )}
        </div>
        );
      })()}

      {/* Item detail */}
      {selectedItem && (
        <div style={S.detailSection}>
          <div style={S.sectionTitle}>
            {inventory.find(i => i.instance_id === selectedItem)?.visual_class || selectedItem}
          </div>

          {/* GT vs Prediction comparison */}
          <div style={S.comparisonCard}>
            <div style={S.compRow}>
              <div style={S.compLabel}></div>
              <div style={S.compHeader}>Ground Truth</div>
              <div style={S.compHeader}>Prediction</div>
              <div style={S.compHeader}>CNPE</div>
            </div>
            <div style={S.compRow}>
              <div style={S.compLabel}>Used</div>
              <div style={S.compVal}>{formatAmt(selectedEval?.used, selectedEval?.unit)}</div>
              <div style={S.compVal}>{formatAmt(selectedPred?.amount_used ?? selectedPred?.amount_derivative, selectedEval?.unit)}</div>
              <CnpeBadge value={selectedEval?.cnpe_used} />
            </div>
            <div style={S.compRow}>
              <div style={S.compLabel}>Remaining</div>
              <div style={S.compVal}>{formatAmt(selectedEval?.after, selectedEval?.unit)}</div>
              <div style={S.compVal}>
                {selectedPred?.amount_remaining != null
                  ? <>{formatAmt(selectedPred.amount_remaining, selectedEval?.unit)}{selectedPred.amount_kind === 'computed_remaining' && <span style={{ fontSize: 10, color: '#888', marginLeft: 4 }}>(s−d)</span>}</>
                  : selectedPred?.amount_starting != null
                    ? <>{formatAmt(selectedPred.amount_starting, selectedEval?.unit)}<span style={{ fontSize: 10, color: '#FFA726', marginLeft: 4 }}>[start]</span></>
                    : '—'}
              </div>
              <CnpeBadge value={selectedEval?.cnpe_rem} />
            </div>
            {/* New journey/dense schema: surface the raw triple so the user
                sees what each independent amount field contained. */}
            {(selectedPred?.amount_starting != null || selectedPred?.amount_remaining_raw != null || selectedPred?.amount_derivative != null) && (
              <div style={{ ...S.compRow, borderTop: '1px solid #2a2a2a', paddingTop: 4, fontSize: 10, color: '#888' }}>
                <div style={S.compLabel}>raw</div>
                <div style={S.compVal}></div>
                <div style={S.compVal}>
                  s={selectedPred.amount_starting ?? '—'} · r={selectedPred.amount_remaining_raw ?? '—'} · d={selectedPred.amount_derivative ?? '—'}
                  {selectedPred.amount_kind && <span style={{ marginLeft: 6, color: '#66BB6A' }}>[{selectedPred.amount_kind}]</span>}
                </div>
                <div style={S.compVal}></div>
              </div>
            )}
          </div>

          {/* Stats */}
          {selectedPred?.stats && (() => {
            const st = selectedPred.stats;
            return (
              <div style={S.statsRow}>
                {st.inference_time_s && <span>{st.inference_time_s}s</span>}
                {st.input_tokens && <span>{st.input_tokens} in</span>}
                {st.num_frames && <span>{st.num_frames} frames</span>}
                {st.planner_inference_time_s && <span>plan: {st.planner_inference_time_s}s</span>}
                {selectedPred.planner_confidence && (
                  <span style={{
                    color: selectedPred.planner_confidence === 'high' ? '#66BB6A'
                      : selectedPred.planner_confidence === 'medium' ? '#FFA726' : '#EF5350'
                  }}>{selectedPred.planner_confidence}</span>
                )}
                {selectedPred.segments?.length > 0 && <span>{selectedPred.segments.length} seg</span>}
              </div>
            );
          })()}

          {/* Planner reasoning */}
          {selectedPred?.planner_reasoning && (
            <div style={S.traceSection}>
              <div style={S.traceTitle}>Planner Reasoning</div>
              <pre style={S.traceContent}>{selectedPred.planner_reasoning}</pre>
            </div>
          )}

          {/* Segments sent to observer (AVP) */}
          {isAvp && selectedPred?.segments?.length > 0 && (
            <div style={S.traceSection}>
              <div style={S.traceTitle}>Planner → Observer Segments</div>
              <div style={{ fontSize: 11, padding: '4px 8px', background: '#1a1a1a', borderRadius: 4, margin: '4px 0' }}>
                {selectedPred.segments.map((seg, i) => (
                  <span
                    key={i}
                    style={{ display: 'inline-block', margin: '2px 4px 2px 0', padding: '1px 6px', background: '#333', borderRadius: 3, cursor: 'pointer', color: '#aaa' }}
                    onClick={() => seekTo(seg.start)}
                    title={`Click to seek to ${formatTime(seg.start)}`}
                  >
                    {formatTime(seg.start)}–{formatTime(seg.end)}
                  </span>
                ))}
                <div style={{ marginTop: 4, color: '#666', fontSize: 10 }}>
                  {selectedPred.segments.length} segments, {selectedPred.segments.reduce((sum, seg) => sum + (seg.end - seg.start), 0).toFixed(0)}s total
                </div>
              </div>
            </div>
          )}

          {/* Thinking trace (per-item, from baseline or item log) */}
          {(itemLog?.thinking || selectedPred?.thinking) && (
            <div style={S.traceSection}>
              <div style={S.traceTitle}>Thinking Trace</div>
              <pre style={S.traceContent}>{itemLog?.thinking || selectedPred.thinking}</pre>
            </div>
          )}

          {/* Reasoning (per-item: AVP observer output or baseline VLM reasoning) */}
          {(itemLog?.reasoning || selectedPred?.reasoning) && (
            <div style={S.traceSection}>
              <div style={S.traceTitle}>Reasoning</div>
              <pre style={S.traceContent}>{itemLog?.reasoning || selectedPred.reasoning}</pre>
            </div>
          )}

          {/* Evidence timestamps */}
          {selectedPred?.evidence_timestamps?.length > 0 && (
            <div style={{ marginTop: 4, padding: '0 8px' }}>
              <span style={{ fontSize: 10, color: '#888' }}>Evidence timestamps: </span>
              {selectedPred.evidence_timestamps.map((ts, i) => (
                <span
                  key={i}
                  style={{ display: 'inline-block', margin: '2px 3px', padding: '1px 5px', background: '#2E7D32', borderRadius: 3, cursor: 'pointer', color: '#fff', fontSize: 10 }}
                  onClick={() => seekTo(ts)}
                  title={`Seek to ${formatTime(ts)}`}
                >
                  {formatTime(ts)}
                </span>
              ))}
            </div>
          )}

          {/* Per-item prompt (collapsible, from item log) */}
          {itemLog?.prompt && (
            <div style={S.traceSection}>
              <div style={{ ...S.traceTitle, cursor: 'pointer' }} onClick={() => setPromptExpanded(!promptExpanded)}>
                {promptExpanded ? '▼' : '▶'} Item Prompt
              </div>
              {promptExpanded && <pre style={S.traceContent}>{itemLog.prompt}</pre>}
            </div>
          )}
        </div>
      )}

      </>)}

      {/* Per-frame detection detail (HOI + DINOv2 + SigLIP + scene tag) */}
      {(isAvp || isIterative) && selectedDetFrame && (
        <DetectionFrameDetail
          frame={selectedDetFrame}
          participant={participant}
          session={session}
          itemColorMap={itemColorMap}
          onDismiss={() => setSelectedDetFrame(null)}
        />
      )}
      </div>
    </div>
  );
}

function DetectionFrameDetail({ frame, participant, session, itemColorMap, onDismiss }) {
  // Collect per-hand DINO top matches (best-per-instance_id) and per-hand
  // SigLIP top matches (best-per-visual_class) from the nested detector output.
  const byHandDino = {}; const byHandSig = {};
  for (const m of (frame.dino_matches || [])) {
    const hs = m.hand_side || 'unknown';
    for (const tm of (m.top_matches || [])) {
      if (!byHandDino[hs]) byHandDino[hs] = {};
      const prev = byHandDino[hs][tm.instance_id];
      if (prev == null || tm.similarity > prev) byHandDino[hs][tm.instance_id] = tm.similarity;
    }
  }
  for (const m of (frame.siglip_matches || [])) {
    const hs = m.hand_side || 'unknown';
    for (const tm of (m.top_matches || [])) {
      if (!byHandSig[hs]) byHandSig[hs] = {};
      const prev = byHandSig[hs][tm.food_name];
      if (prev == null || tm.similarity > prev) byHandSig[hs][tm.food_name] = tm.similarity;
    }
  }

  const handOrder = ['left_hand', 'right_hand'];
  const handLabel = hs => (hs === 'left_hand' ? 'L' : hs === 'right_hand' ? 'R' : hs || 'U');
  const allHands = new Set([
    ...((frame.detections || []).map(d => d.hand_side)),
    ...Object.keys(byHandDino),
    ...Object.keys(byHandSig),
  ]);
  const orderedHands = [
    ...handOrder.filter(h => allHands.has(h)),
    ...[...allHands].filter(h => !handOrder.includes(h)).sort(),
  ];

  const tsLabel = (frame.session_timestamp_s != null ? `${frame.session_timestamp_s.toFixed(2)}s` : '?');

  return (
    <div style={S.detailSection}>
      <div style={S.sectionTitle}>
        Frame @ {tsLabel}
        <span style={{ fontSize: 10, color: '#888', marginLeft: 8 }}>
          {frame.video_id ? `clip ${frame.video_id} · ${frame.clip_timestamp_s?.toFixed(2)}s` : ''}
        </span>
        <span
          style={{ fontSize: 10, color: '#888', marginLeft: 8, cursor: 'pointer' }}
          onClick={onDismiss}
        >(dismiss)</span>
      </div>

      <div style={{ padding: '4px 0', fontSize: 12 }}>
        <div>
          <strong>Scene (OWLv2):</strong>{' '}
          <span style={{ color: '#4FC3F7' }}>{frame.scene_tag || 'unknown'}</span>
        </div>
        <div><strong>Hands detected:</strong> {frame.num_hands ?? (frame.detections?.length || 0)}</div>
      </div>

      {frame.visualization_path && (
        <div style={{ padding: '4px 0' }}>
          <img
            src={`${API}/eval-frame/${encodeURIComponent(participant)}/${encodeURIComponent(session)}/${frame.visualization_path}`}
            alt="hands23 visualization"
            style={{ width: '100%', maxHeight: 200, objectFit: 'contain', border: '1px solid #444', borderRadius: 3 }}
            onError={e => { e.currentTarget.style.display = 'none'; }}
          />
        </div>
      )}

      {orderedHands.map(hs => {
        const det = (frame.detections || []).find(d => d.hand_side === hs) || {};
        const dinoHits = Object.entries(byHandDino[hs] || {}).sort(([,a],[,b])=>b-a).slice(0, 6);
        const sigHits  = Object.entries(byHandSig[hs]  || {}).sort(([,a],[,b])=>b-a).slice(0, 6);
        const grasp = det.grasp;
        const touch = det.obj_touch;
        const contact = det.contact_state;
        return (
          <div key={hs} style={{ padding: '4px 0', fontSize: 12, borderTop: '1px solid #333' }}>
            <div><strong>{handLabel(hs)}</strong>{' '}
              <span style={{ color: contact === 'object_contact' ? '#66BB6A' : '#888' }}>
                {contact || '—'}
              </span>
              {grasp && <span style={{ marginLeft: 6, color: '#CE93D8' }}>grasp={grasp}</span>}
              {touch && <span style={{ marginLeft: 6, color: '#FFA726' }}>touch={touch}</span>}
              {det.hand_score != null && <span style={{ marginLeft: 6, color: '#666' }}>h={det.hand_score.toFixed(2)}</span>}
              {det.obj_score != null && <span style={{ marginLeft: 4, color: '#666' }}>o={det.obj_score.toFixed(2)}</span>}
            </div>

            {dinoHits.length > 0 && (
              <div style={{ marginTop: 2 }}>
                <span style={{ color: '#CE93D8', fontSize: 10 }}>DINOv2 visual</span>
                {dinoHits.map(([iid, s]) => (
                  <div key={iid} style={{ paddingLeft: 8, display: 'flex', gap: 6 }}>
                    <span style={{ color: itemColorMap[iid] || '#ccc', flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                      {iid.replace(/_\d{8}$/, '')}
                    </span>
                    <span style={{ color: '#888', flexShrink: 0 }}>{s.toFixed(3)}</span>
                  </div>
                ))}
              </div>
            )}
            {sigHits.length > 0 && (
              <div style={{ marginTop: 2 }}>
                <span style={{ color: '#FFD54F', fontSize: 10 }}>SigLIP text</span>
                {sigHits.map(([name, s]) => (
                  <div key={name} style={{ paddingLeft: 8, display: 'flex', gap: 6 }}>
                    <span style={{ color: '#ccc', flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                      {name}
                    </span>
                    <span style={{ color: '#888', flexShrink: 0 }}>{s.toFixed(3)}</span>
                  </div>
                ))}
              </div>
            )}
            {dinoHits.length === 0 && sigHits.length === 0 && contact === 'object_contact' && (
              <div style={{ paddingLeft: 8, color: '#666', fontSize: 11 }}>
                contact but no inventory match ≥ 0.15
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}

function CnpeBadge({ value }) {
  const color = value != null
    ? (value < 5 ? '#66BB6A' : value < 15 ? '#FFA726' : '#EF5350')
    : '#888';
  return (
    <div style={{ ...S.compVal, color }}>
      {value != null ? `${value.toFixed(1)}%` : '—'}
    </div>
  );
}

function TimelineBar({ rows, totalDuration, globalTime, itemColorMap, seekTo, onSegmentClick }) {
  const containerRef = React.useRef(null);

  const handleClick = useCallback((e) => {
    if (!containerRef.current || !totalDuration) return;
    const rect = containerRef.current.getBoundingClientRect();
    const pct = (e.clientX - rect.left) / rect.width;
    seekTo(Math.max(0, Math.min(totalDuration, pct * totalDuration)));
  }, [totalDuration, seekTo]);

  const timePct = totalDuration ? (globalTime / totalDuration) * 100 : 0;
  const rowH = 100 / rows.length;

  return (
    <div style={{ ...S.timelineContainer, height: Math.max(60, rows.length * 22) }}>
      <div style={S.timelineLabels}>
        {rows.map(r => (
          <div key={r.label} style={{ ...S.timelineLabel, height: `${rowH}%` }}>{r.label}</div>
        ))}
      </div>
      <div ref={containerRef} style={S.timelineBar} onClick={handleClick}>
        {rows.map((row, ri) => (
          <React.Fragment key={row.label}>
            {row.segments.map((seg, si) => {
              if (!totalDuration) return null;
              const left = (seg.start / totalDuration) * 100;
              const width = Math.max(0.3, ((seg.end - seg.start) / totalDuration) * 100);
              const color = itemColorMap[seg.item] || '#888';
              return (
                <div
                  key={`${row.label}-${si}`}
                  title={`${seg.action || seg.item} [${formatTime(seg.start)}–${formatTime(seg.end)}]`}
                  style={{
                    position: 'absolute',
                    left: `${left}%`, width: `${width}%`,
                    top: `${ri * rowH}%`, height: `${rowH}%`,
                    background: color, opacity: row.opacity,
                    borderRadius: 2, cursor: 'pointer', borderBottom: '1px solid #222',
                  }}
                  onClick={(e) => { e.stopPropagation(); seekTo(seg.start); if (onSegmentClick) onSegmentClick(seg); }}
                />
              );
            })}
          </React.Fragment>
        ))}
        <div style={{
          position: 'absolute', left: `${timePct}%`, top: 0, bottom: 0,
          width: 2, background: '#fff', pointerEvents: 'none', zIndex: 10,
        }} />
        {rows.slice(1).map((_, i) => (
          <div key={`div-${i}`} style={{
            position: 'absolute', left: 0, right: 0,
            top: `${((i + 1) * rowH)}%`,
            height: 1, background: '#444', pointerEvents: 'none',
          }} />
        ))}
      </div>
    </div>
  );
}

const S = {
  empty: { padding: 40, textAlign: 'center', color: '#888' },
  main: { display: 'flex', flex: 1, overflow: 'hidden', color: '#ccc', fontFamily: 'monospace' },
  leftPanel: { flex: 1, display: 'flex', flexDirection: 'column', minWidth: 0 },
  rightPanel: { width: 520, display: 'flex', flexDirection: 'column', borderLeft: '1px solid #444', overflow: 'hidden' },

  // Item filter
  itemFilterBar: { display: 'flex', alignItems: 'center', padding: '4px 8px', borderBottom: '1px solid #333', flexShrink: 0 },
  itemSearchInput: { width: '100%', background: '#222', color: '#ccc', border: '1px solid #444', borderRadius: 3, padding: '3px 22px 3px 6px', fontSize: 11, outline: 'none' },
  itemFilterClear: { position: 'absolute', right: 6, top: '50%', transform: 'translateY(-50%)', cursor: 'pointer', color: '#888', fontSize: 12, lineHeight: 1 },
  itemSearchDropdown: { position: 'absolute', top: '100%', left: 0, right: 0, background: '#2a2a2a', border: '1px solid #555', borderTop: 'none', borderRadius: '0 0 4px 4px', zIndex: 20, maxHeight: 200, overflow: 'auto' },
  itemSearchOption: { padding: '4px 8px', cursor: 'pointer', fontSize: 11, display: 'flex', alignItems: 'baseline' },

  videoWrap: { position: 'relative', background: '#000', flexShrink: 0 },
  video: { width: '100%', display: 'block', maxHeight: '55vh' },
  videoOverlay: { position: 'absolute', bottom: 4, left: 8, fontSize: 11, color: '#aaa', background: 'rgba(0,0,0,0.6)', padding: '2px 6px', borderRadius: 3 },
  transport: { display: 'flex', alignItems: 'center', gap: 8, padding: '6px 12px', background: '#252525' },
  playBtn: { background: 'none', border: 'none', color: '#eee', fontSize: 18, cursor: 'pointer', padding: '2px 8px' },
  timeDisplay: { fontSize: 12, color: '#aaa', minWidth: 50, textAlign: 'center' },
  scrubber: { flex: 1 },
  clipBar: { display: 'flex', height: 16, background: '#333' },
  header: { display: 'flex', flexDirection: 'column', padding: '6px 10px', borderBottom: '1px solid #333', gap: 4 },
  headerLeft: { display: 'flex', alignItems: 'center', gap: 8 },
  label: { fontSize: 11, color: '#888' },
  select: { background: '#333', color: '#ccc', border: '1px solid #555', borderRadius: 3, padding: '2px 6px', fontSize: 11, maxWidth: 300 },
  itemToggles: { display: 'flex', flexWrap: 'wrap', gap: 3 },
  toggleBtn: { border: '1px solid #555', borderRadius: 3, padding: '2px 8px', fontSize: 10, cursor: 'pointer', color: '#ccc', background: '#333' },
  timelineContainer: { display: 'flex', borderBottom: '1px solid #333', position: 'relative' },
  timelineLabels: { width: 50, display: 'flex', flexDirection: 'column', borderRight: '1px solid #444' },
  timelineLabel: { display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 9, color: '#888' },
  timelineBar: { flex: 1, position: 'relative', background: '#1a1a1a', cursor: 'crosshair' },
  inventorySection: { borderBottom: '1px solid #333', overflow: 'auto', flexShrink: 0 },
  sectionTitle: { fontSize: 12, fontWeight: 'bold', padding: '6px 10px', borderBottom: '1px solid #333', color: '#ddd' },
  tableHeader: { display: 'flex', fontSize: 10, color: '#888', padding: '4px 10px', borderBottom: '1px solid #333' },
  tableRow: { display: 'flex', fontSize: 11, padding: '3px 10px', cursor: 'pointer', borderBottom: '1px solid #222' },
  tableColItem: { flex: 2, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' },
  tableColNum: { flex: 1, textAlign: 'right', paddingRight: 4 },
  tableColCnpe: { width: 50, textAlign: 'right', fontSize: 10 },
  detailSection: { padding: '8px 10px', overflow: 'auto', flex: 1 },
  comparisonCard: { background: '#1a1a1a', borderRadius: 6, padding: 8, marginBottom: 8 },
  compRow: { display: 'flex', padding: '2px 0' },
  compLabel: { width: 70, fontSize: 11, color: '#888' },
  compHeader: { flex: 1, fontSize: 10, color: '#666', textAlign: 'center' },
  compVal: { flex: 1, fontSize: 12, textAlign: 'center' },
  statsRow: { display: 'flex', gap: 12, fontSize: 10, color: '#888', padding: '4px 0', flexWrap: 'wrap' },
  traceSection: { marginTop: 8 },
  traceTitle: { fontSize: 11, color: '#888', marginBottom: 4 },
  traceContent: { fontSize: 11, color: '#bbb', whiteSpace: 'pre-wrap', wordBreak: 'break-word', background: '#1a1a1a', borderRadius: 4, padding: 8, maxHeight: 200, overflow: 'auto', margin: 0 },
};

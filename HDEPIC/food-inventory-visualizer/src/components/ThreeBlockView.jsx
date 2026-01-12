import React, { useMemo, useRef, useState, useEffect, useCallback } from 'react';

const styles = {
  container: {
    backgroundColor: '#fafafa',
    borderRadius: '8px',
    padding: '12px',
    marginTop: '20px',
  },
  header: {
    fontSize: '14px',
    fontWeight: 'bold',
    marginBottom: '10px',
    color: '#333',
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  closeBtn: {
    background: 'none',
    border: 'none',
    fontSize: '18px',
    cursor: 'pointer',
    color: '#666',
    padding: '0 5px',
  },
  graphContainer: {
    display: 'flex',
    gap: '60px',
    overflow: 'auto',
    position: 'relative',
    minHeight: '300px',
  },
  snapshotColumn: {
    flex: 1,
    display: 'flex',
    flexDirection: 'column',
    minWidth: '180px',
  },
  snapshotHeader: {
    fontSize: '12px',
    fontWeight: 'bold',
    padding: '8px',
    backgroundColor: '#e0e0e0',
    borderRadius: '4px 4px 0 0',
    textAlign: 'center',
  },
  snapshotHeaderBeforeLast: {
    backgroundColor: '#e1bee7',  // Purple
  },
  snapshotHeaderBefore: {
    backgroundColor: '#ffecb3',  // Yellow
  },
  snapshotHeaderAfter: {
    backgroundColor: '#c8e6c9',  // Green
  },
  nodesContainer: {
    flex: 1,
    backgroundColor: '#fff',
    border: '1px solid #ddd',
    borderTop: 'none',
    borderRadius: '0 0 4px 4px',
    padding: '8px',
    overflowY: 'auto',
    maxHeight: '400px',
  },
  locationGroup: {
    marginBottom: '8px',
    borderRadius: '4px',
    border: '1px solid #ddd',
    overflow: 'hidden',
  },
  locationGroupInvolved: {
    border: '2px solid #ff9800',
  },
  locationHeader: {
    padding: '4px 8px',
    backgroundColor: '#f0f0f0',
    fontSize: '10px',
    fontWeight: 'bold',
    display: 'flex',
    alignItems: 'center',
    gap: '4px',
  },
  locationHeaderInvolved: {
    backgroundColor: '#fff3e0',
  },
  locationFoods: {
    padding: '4px',
    backgroundColor: '#fafafa',
  },
  foodNode: {
    padding: '4px 6px',
    marginBottom: '3px',
    borderRadius: '3px',
    border: '1px solid #2196F3',
    backgroundColor: '#e3f2fd',
    fontSize: '10px',
  },
  foodNodeInvolved: {
    border: '2px solid #ff9800',
    backgroundColor: '#fff3e0',
  },
  foodId: {
    fontWeight: 'bold',
    color: '#1565c0',
    fontSize: '10px',
    wordBreak: 'break-all',
  },
  foodProperty: {
    color: '#666',
    fontSize: '9px',
  },
  noSelection: {
    padding: '30px',
    textAlign: 'center',
    color: '#666',
  },
};

const EDGE_COLORS = {
  split: '#4CAF50',
  merge: '#9c27b0',
  update: '#2196F3',
  identity_transform: '#FF9800',
  transfer: '#607d8b',
  consume: '#f44336',
  default: '#666',
};

const LOCATION_ICONS = {
  pan: '🍳',
  pot: '🍲',
  bowl: '🥣',
  plate: '🍽️',
  cup: '☕',
  mug: '☕',
  glass: '🥛',
  jar: '🫙',
  bottle: '🍾',
  bag: '🛍️',
  blender: '🥤',
  container: '📦',
  box: '📦',
  environment: '',
  default: '📦',
};

function getLocationIcon(location) {
  if (!location) return LOCATION_ICONS.environment;
  const loc = location.toLowerCase();
  for (const [key, icon] of Object.entries(LOCATION_ICONS)) {
    if (key !== 'environment' && key !== 'default' && loc.includes(key)) return icon;
  }
  return LOCATION_ICONS.default;
}

function formatLocationName(location) {
  if (!location) return 'Environment';
  return location.replace(/_\d+$/, '').replace(/_/g, ' ');
}

function FoodNode({ food, isInvolved, nodeRef }) {
  const form = food.form || food.state?.form_state || '';
  const quantity = food.quantity || food.state?.quantity || '';
  const foodId = food.food_id || food.instance_id || 'unknown';

  return (
    <div
      ref={nodeRef}
      style={{
        ...styles.foodNode,
        ...(isInvolved ? styles.foodNodeInvolved : {}),
      }}
    >
      <div style={styles.foodId}>{foodId}</div>
      {(form || quantity) && (
        <div style={styles.foodProperty}>
          {form}{form && quantity ? ' | ' : ''}{quantity}
        </div>
      )}
    </div>
  );
}

function groupFoodsByLocation(foods, involvedFoodIds, showOnlyInvolved = false) {
  if (!foods || foods.length === 0) return [];

  const groups = {};
  const noLocation = [];

  for (const food of foods) {
    const location = food.location || null;
    const foodId = food.food_id || food.instance_id;
    const isInvolved = involvedFoodIds.has(foodId);

    // Filter out non-involved nodes when showOnlyInvolved is true
    if (showOnlyInvolved && !isInvolved) {
      continue;
    }

    if (location) {
      if (!groups[location]) {
        groups[location] = { location, foods: [], hasInvolvedFood: false };
      }
      groups[location].foods.push({ ...food, _isInvolved: isInvolved });
      if (isInvolved) groups[location].hasInvolvedFood = true;
    } else {
      noLocation.push({ ...food, _isInvolved: isInvolved });
    }
  }

  const groupArray = Object.values(groups);
  groupArray.sort((a, b) => {
    if (a.hasInvolvedFood && !b.hasInvolvedFood) return -1;
    if (!a.hasInvolvedFood && b.hasInvolvedFood) return 1;
    return a.location.localeCompare(b.location);
  });

  if (noLocation.length > 0) {
    groupArray.push({
      location: null,
      foods: noLocation,
      hasInvolvedFood: noLocation.some(f => f._isInvolved),
    });
  }

  return groupArray;
}

function LocationGroup({ group, nodeRefs, side }) {
  const { location, foods, hasInvolvedFood } = group;
  const icon = getLocationIcon(location);
  const displayName = location ? formatLocationName(location) : 'Environment';

  return (
    <div style={{ ...styles.locationGroup, ...(hasInvolvedFood ? styles.locationGroupInvolved : {}) }}>
      <div style={{ ...styles.locationHeader, ...(hasInvolvedFood ? styles.locationHeaderInvolved : {}) }}>
        <span>{icon}</span>
        <span>{displayName}</span>
        <span style={{ color: '#888', fontWeight: 'normal' }}>({foods.length})</span>
      </div>
      <div style={styles.locationFoods}>
        {foods.map((food, idx) => {
          const foodId = food.food_id || food.instance_id;
          const refKey = `${side}_${foodId}`;
          return (
            <FoodNode
              key={foodId || idx}
              food={food}
              isInvolved={food._isInvolved}
              nodeRef={nodeRefs ? (el) => { if (el) nodeRefs.current[refKey] = el; } : undefined}
            />
          );
        })}
      </div>
    </div>
  );
}

const convertFoodNodesToArray = (blockGraph) => {
  if (!blockGraph) return [];
  if (blockGraph.foods && Array.isArray(blockGraph.foods)) return blockGraph.foods;
  if (blockGraph.food_nodes && typeof blockGraph.food_nodes === 'object') {
    return Object.values(blockGraph.food_nodes);
  }
  return [];
};

function ThreeBlockView({ graph, selectedEventIndex, events, showOnlyInvolved = false, onClose }) {
  const nodeRefs = useRef({});
  const containerRef = useRef(null);
  const [edgePaths, setEdgePaths] = useState([]);

  // Compute three snapshots and edges
  const { beforeLastSnapshot, beforeSnapshot, afterSnapshot, prevEdges, currEdges, involvedFoodIds } = useMemo(() => {
    if (!graph || selectedEventIndex === null || selectedEventIndex === undefined) {
      return { beforeLastSnapshot: null, beforeSnapshot: null, afterSnapshot: null, prevEdges: [], currEdges: [], involvedFoodIds: new Set() };
    }

    const blockGraphs = graph.block_graphs || [];
    const lineageEdges = graph.lineage_edges || [];
    const inventory = graph.inventory || [];

    const afterIdx = selectedEventIndex;
    const beforeIdx = selectedEventIndex - 1;
    const beforeLastIdx = selectedEventIndex - 2;

    // After snapshot
    let afterSnapshot = null;
    if (afterIdx >= 0 && afterIdx < blockGraphs.length) {
      const block = blockGraphs[afterIdx];
      afterSnapshot = { ...block, foods: convertFoodNodesToArray(block) };
    }

    // Before snapshot
    let beforeSnapshot = null;
    if (beforeIdx >= 0 && beforeIdx < blockGraphs.length) {
      const block = blockGraphs[beforeIdx];
      beforeSnapshot = { ...block, foods: convertFoodNodesToArray(block) };
    } else if (selectedEventIndex === 0) {
      beforeSnapshot = {
        block_idx: -1,
        foods: inventory.map(f => ({
          food_id: f.food_id || f.instance_id,
          form: f.initial_state?.form || f.state?.form_state,
          quantity: f.initial_state?.quantity || f.state?.quantity,
          location: f.initial_state?.location || f.location,
          status: 'active',
        })),
      };
    }

    // Before Last snapshot
    let beforeLastSnapshot = null;
    if (beforeLastIdx >= 0 && beforeLastIdx < blockGraphs.length) {
      const block = blockGraphs[beforeLastIdx];
      beforeLastSnapshot = { ...block, foods: convertFoodNodesToArray(block) };
    } else if (selectedEventIndex === 1) {
      beforeLastSnapshot = {
        block_idx: -1,
        foods: inventory.map(f => ({
          food_id: f.food_id || f.instance_id,
          form: f.initial_state?.form || f.state?.form_state,
          quantity: f.initial_state?.quantity || f.state?.quantity,
          location: f.initial_state?.location || f.location,
          status: 'active',
        })),
      };
    }

    // Edges: before-last → before (previous event)
    const prevEdges = lineageEdges.filter((edge) => edge.target_block === beforeIdx);
    // Edges: before → after (current event)
    const currEdges = lineageEdges.filter((edge) => edge.target_block === afterIdx);

    // Involved food IDs
    const ids = new Set();
    [...prevEdges, ...currEdges].forEach((edge) => {
      if (edge.parent_id) ids.add(edge.parent_id);
      if (edge.parent_instance_id) ids.add(edge.parent_instance_id);
      if (edge.child_id) ids.add(edge.child_id);
      if (edge.child_instance_id) ids.add(edge.child_instance_id);
    });

    return { beforeLastSnapshot, beforeSnapshot, afterSnapshot, prevEdges, currEdges, involvedFoodIds: ids };
  }, [graph, selectedEventIndex]);

  // Calculate SVG edge paths
  const calculateEdgePaths = useCallback(() => {
    if (!containerRef.current) {
      setEdgePaths([]);
      return;
    }

    const containerRect = containerRef.current.getBoundingClientRect();
    const paths = [];

    // Previous event edges: beforeLast → before
    prevEdges.forEach((edge, idx) => {
      const parentId = edge.parent_id || edge.parent_instance_id || '';
      const childId = edge.child_id || edge.child_instance_id || '';
      const fromEl = nodeRefs.current[`beforeLast_${parentId}`];
      const toEl = nodeRefs.current[`before_${childId}`];
      if (!fromEl || !toEl) return;

      const fromRect = fromEl.getBoundingClientRect();
      const toRect = toEl.getBoundingClientRect();
      const fromX = fromRect.right - containerRect.left;
      const fromY = fromRect.top + fromRect.height / 2 - containerRect.top;
      const toX = toRect.left - containerRect.left;
      const toY = toRect.top + toRect.height / 2 - containerRect.top;
      const midX = (fromX + toX) / 2;
      const midY = (fromY + toY) / 2;

      paths.push({
        id: `prev-${parentId}-${childId}-${idx}`,
        d: `M ${fromX} ${fromY} C ${midX} ${fromY}, ${midX} ${toY}, ${toX} ${toY}`,
        color: EDGE_COLORS[edge.derivation_type] || EDGE_COLORS.default,
        type: edge.derivation_type,
        labelX: midX,
        labelY: midY,
      });
    });

    // Current event edges: before → after
    currEdges.forEach((edge, idx) => {
      const parentId = edge.parent_id || edge.parent_instance_id || '';
      const childId = edge.child_id || edge.child_instance_id || '';
      const fromEl = nodeRefs.current[`before_${parentId}`];
      const toEl = nodeRefs.current[`after_${childId}`];
      if (!fromEl || !toEl) return;

      const fromRect = fromEl.getBoundingClientRect();
      const toRect = toEl.getBoundingClientRect();
      const fromX = fromRect.right - containerRect.left;
      const fromY = fromRect.top + fromRect.height / 2 - containerRect.top;
      const toX = toRect.left - containerRect.left;
      const toY = toRect.top + toRect.height / 2 - containerRect.top;
      const midX = (fromX + toX) / 2;
      const midY = (fromY + toY) / 2;

      paths.push({
        id: `curr-${parentId}-${childId}-${idx}`,
        d: `M ${fromX} ${fromY} C ${midX} ${fromY}, ${midX} ${toY}, ${toX} ${toY}`,
        color: EDGE_COLORS[edge.derivation_type] || EDGE_COLORS.default,
        type: edge.derivation_type,
        labelX: midX,
        labelY: midY,
      });
    });

    setEdgePaths(paths);
  }, [prevEdges, currEdges]);

  useEffect(() => {
    const timer = setTimeout(calculateEdgePaths, 100);
    return () => clearTimeout(timer);
  }, [calculateEdgePaths, beforeLastSnapshot, beforeSnapshot, afterSnapshot, showOnlyInvolved]);

  useEffect(() => {
    nodeRefs.current = {};
  }, [selectedEventIndex]);

  if (!graph || selectedEventIndex === null || selectedEventIndex === undefined) {
    return null;
  }

  // Only show when there's a before-last snapshot (event index >= 1)
  if (!beforeLastSnapshot) {
    return null;
  }

  const selectedEvent = events?.[selectedEventIndex];
  const prevEvent = events?.[selectedEventIndex - 1];

  return (
    <div style={styles.container}>
      <div style={styles.header}>
        <span>
          Three-Block View - Events {prevEvent?.event_id || selectedEventIndex} → {selectedEvent?.event_id || selectedEventIndex + 1}
        </span>
        <button style={styles.closeBtn} onClick={onClose} title="Close">×</button>
      </div>

      <div ref={containerRef} style={styles.graphContainer}>
        {/* SVG Edge Overlay */}
        <svg
          style={{
            position: 'absolute',
            top: 0,
            left: 0,
            width: '100%',
            height: '100%',
            pointerEvents: 'none',
            zIndex: 10,
          }}
        >
          <defs>
            <marker id="arrowhead-three" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
              <polygon points="0 0, 8 3, 0 6" fill="#666" />
            </marker>
          </defs>
          {edgePaths.map((path) => (
            <g key={path.id}>
              <path
                d={path.d}
                stroke={path.color}
                strokeWidth={2}
                fill="none"
                markerEnd="url(#arrowhead-three)"
                opacity={0.8}
              />
              <rect x={path.labelX - 20} y={path.labelY - 7} width={40} height={14} fill="white" opacity={0.9} rx={2} />
              <text x={path.labelX} y={path.labelY + 3} textAnchor="middle" fontSize="9" fontWeight="bold" fill={path.color}>
                {path.type}
              </text>
            </g>
          ))}
        </svg>

        {/* Before Last Snapshot */}
        <div style={styles.snapshotColumn}>
          <div style={{ ...styles.snapshotHeader, ...styles.snapshotHeaderBeforeLast }}>
            Before Last (Block {beforeLastSnapshot?.block_id ?? beforeLastSnapshot?.block_idx ?? 'Init'})
          </div>
          <div style={styles.nodesContainer}>
            {beforeLastSnapshot?.foods?.length > 0 ? (
              groupFoodsByLocation(beforeLastSnapshot.foods, involvedFoodIds, showOnlyInvolved).map((group, idx) => (
                <LocationGroup key={group.location || `no-loc-${idx}`} group={group} nodeRefs={nodeRefs} side="beforeLast" />
              ))
            ) : (
              <div style={{ color: '#999', textAlign: 'center', fontSize: '11px' }}>No foods</div>
            )}
          </div>
        </div>

        {/* Before Snapshot */}
        <div style={styles.snapshotColumn}>
          <div style={{ ...styles.snapshotHeader, ...styles.snapshotHeaderBefore }}>
            Before (Block {beforeSnapshot?.block_id ?? beforeSnapshot?.block_idx ?? 'Init'})
          </div>
          <div style={styles.nodesContainer}>
            {beforeSnapshot?.foods?.length > 0 ? (
              groupFoodsByLocation(beforeSnapshot.foods, involvedFoodIds, showOnlyInvolved).map((group, idx) => (
                <LocationGroup key={group.location || `no-loc-${idx}`} group={group} nodeRefs={nodeRefs} side="before" />
              ))
            ) : (
              <div style={{ color: '#999', textAlign: 'center', fontSize: '11px' }}>No foods</div>
            )}
          </div>
        </div>

        {/* After Snapshot */}
        <div style={styles.snapshotColumn}>
          <div style={{ ...styles.snapshotHeader, ...styles.snapshotHeaderAfter }}>
            After (Block {afterSnapshot?.block_id ?? afterSnapshot?.block_idx ?? selectedEventIndex + 1})
          </div>
          <div style={styles.nodesContainer}>
            {afterSnapshot?.foods?.length > 0 ? (
              groupFoodsByLocation(afterSnapshot.foods, involvedFoodIds, showOnlyInvolved).map((group, idx) => (
                <LocationGroup key={group.location || `no-loc-${idx}`} group={group} nodeRefs={nodeRefs} side="after" />
              ))
            ) : (
              <div style={{ color: '#999', textAlign: 'center', fontSize: '11px' }}>No foods</div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default ThreeBlockView;

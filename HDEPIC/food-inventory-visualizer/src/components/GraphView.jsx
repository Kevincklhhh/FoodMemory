import React, { useMemo, useRef, useState, useEffect, useCallback } from 'react';

const styles = {
  container: {
    backgroundColor: '#fafafa',
    borderRadius: '8px',
    padding: '12px',
    height: '100%',
    display: 'flex',
    flexDirection: 'column',
    overflow: 'hidden',
  },
  header: {
    fontSize: '14px',
    fontWeight: 'bold',
    marginBottom: '10px',
    color: '#333',
  },
  graphContainer: {
    flex: 1,
    display: 'flex',
    gap: '80px',  // Increased gap for SVG edge lines
    overflow: 'auto',
  },
  snapshotColumn: {
    flex: 1,
    display: 'flex',
    flexDirection: 'column',
    minWidth: '220px',
  },
  snapshotHeader: {
    fontSize: '12px',
    fontWeight: 'bold',
    padding: '8px',
    backgroundColor: '#e0e0e0',
    borderRadius: '4px 4px 0 0',
    textAlign: 'center',
  },
  snapshotHeaderBefore: {
    backgroundColor: '#ffecb3',
  },
  snapshotHeaderAfter: {
    backgroundColor: '#c8e6c9',
  },
  nodesContainer: {
    flex: 1,
    backgroundColor: '#fff',
    border: '1px solid #ddd',
    borderTop: 'none',
    borderRadius: '0 0 4px 4px',
    padding: '8px',
    overflowY: 'auto',
  },
  // Location group (container) styles
  locationGroup: {
    marginBottom: '10px',
    borderRadius: '6px',
    border: '1px solid #ccc',
    overflow: 'hidden',
  },
  locationGroupInvolved: {
    border: '2px solid #ff9800',
  },
  locationGroupFocused: {
    border: '2px solid #9c27b0',
    boxShadow: '0 0 8px rgba(156, 39, 176, 0.3)',
  },
  locationHeader: {
    padding: '6px 10px',
    backgroundColor: '#e8e8e8',
    fontSize: '11px',
    fontWeight: 'bold',
    display: 'flex',
    alignItems: 'center',
    gap: '6px',
  },
  locationHeaderInvolved: {
    backgroundColor: '#fff3e0',
  },
  locationHeaderFocused: {
    backgroundColor: '#f3e5f5',
  },
  locationHeaderClickable: {
    cursor: 'pointer',
    transition: 'background-color 0.2s',
  },
  locationIcon: {
    fontSize: '12px',
  },
  locationFoods: {
    padding: '6px',
    backgroundColor: '#fafafa',
  },
  // Compact food node styles
  foodNode: {
    padding: '6px 8px',
    marginBottom: '4px',
    borderRadius: '4px',
    border: '1px solid #2196F3',
    backgroundColor: '#e3f2fd',
    fontSize: '11px',
  },
  foodNodeConsumed: {
    border: '1px solid #9e9e9e',
    backgroundColor: '#f5f5f5',
    opacity: 0.7,
  },
  foodNodeInvolved: {
    border: '2px solid #ff9800',
    backgroundColor: '#fff3e0',
  },
  foodNodeFocused: {
    border: '2px solid #9c27b0',
    backgroundColor: '#f3e5f5',
    boxShadow: '0 0 6px rgba(156, 39, 176, 0.4)',
  },
  foodId: {
    fontWeight: 'bold',
    color: '#1565c0',
    marginBottom: '2px',
    wordBreak: 'break-all',
    fontSize: '11px',
  },
  foodProperty: {
    color: '#555',
    marginBottom: '1px',
    fontSize: '10px',
  },
  foodPropertiesRow: {
    display: 'flex',
    gap: '8px',
    flexWrap: 'wrap',
  },
  noSelection: {
    padding: '40px',
    textAlign: 'center',
    color: '#666',
  },
  legend: {
    display: 'flex',
    flexWrap: 'wrap',
    gap: '8px',
    marginTop: '10px',
    padding: '10px',
    backgroundColor: '#f5f5f5',
    borderRadius: '4px',
  },
  legendItem: {
    display: 'flex',
    alignItems: 'center',
    gap: '4px',
    fontSize: '11px',
  },
  legendColor: {
    width: '12px',
    height: '12px',
    borderRadius: '2px',
  },
  edgeSplit: { backgroundColor: '#c8e6c9', color: '#2e7d32' },
  edgeMerge: { backgroundColor: '#e1bee7', color: '#7b1fa2' },
  edgeUpdate: { backgroundColor: '#bbdefb', color: '#1565c0' },
  edgeIdentity: { backgroundColor: '#ffe0b2', color: '#e65100' },
  edgeTransfer: { backgroundColor: '#cfd8dc', color: '#455a64' },
  edgeConsume: { backgroundColor: '#ffcdd2', color: '#c62828' },
};

const EDGE_STYLES = {
  split: styles.edgeSplit,
  merge: styles.edgeMerge,
  update: styles.edgeUpdate,
  identity_transform: styles.edgeIdentity,
  transfer: styles.edgeTransfer,
  consume: styles.edgeConsume,
};

// Edge colors for SVG visualization
const EDGE_COLORS = {
  split: '#4CAF50',      // Green
  merge: '#9c27b0',      // Purple
  update: '#2196F3',     // Blue
  identity_transform: '#FF9800', // Orange
  transfer: '#607d8b',   // Blue-grey
  consume: '#f44336',    // Red
  default: '#666',
};

function FoodNode({ food, isInvolved, showLocation = false, onClick, isFocused, nodeRef }) {
  const isConsumed = food.status === 'consumed';
  // Handle both flat structure and nested state structure
  const form = food.form || food.state?.form_state || 'unknown';
  const quantity = food.quantity || food.state?.quantity || 'unknown';
  const location = food.location || null;
  const foodId = food.food_id || food.instance_id || 'unknown';

  return (
    <div
      ref={nodeRef}
      onClick={() => onClick && onClick(foodId)}
      style={{
        ...styles.foodNode,
        ...(isConsumed ? styles.foodNodeConsumed : {}),
        ...(isInvolved ? styles.foodNodeInvolved : {}),
        ...(isFocused ? styles.foodNodeFocused : {}),
        cursor: onClick ? 'pointer' : 'default',
      }}
      title="Click to trace lineage"
    >
      <div style={styles.foodId}>{foodId}</div>
      <div style={styles.foodPropertiesRow}>
        <span style={styles.foodProperty}>{form}</span>
        <span style={styles.foodProperty}>| {quantity}</span>
        {food.status && food.status !== 'active' && (
          <span style={styles.foodProperty}>| {food.status}</span>
        )}
      </div>
      {showLocation && location && (
        <div style={styles.foodProperty}>
          <strong>loc:</strong> {location}
        </div>
      )}
    </div>
  );
}

// Location icons for containers (V3 container-only model)
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
  if (!location) return 'Unknown';
  return location.replace(/_\d+$/, '').replace(/_/g, ' ');
}

// Group foods by their location
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
        groups[location] = {
          location,
          foods: [],
          hasInvolvedFood: false,
        };
      }
      groups[location].foods.push({ ...food, _isInvolved: isInvolved });
      if (isInvolved) {
        groups[location].hasInvolvedFood = true;
      }
    } else {
      noLocation.push({ ...food, _isInvolved: isInvolved });
    }
  }

  // Convert to array and sort: involved groups first, then alphabetically
  const groupArray = Object.values(groups);
  groupArray.sort((a, b) => {
    if (a.hasInvolvedFood && !b.hasInvolvedFood) return -1;
    if (!a.hasInvolvedFood && b.hasInvolvedFood) return 1;
    return a.location.localeCompare(b.location);
  });

  // Sort foods within each group: involved first
  for (const group of groupArray) {
    group.foods.sort((a, b) => {
      if (a._isInvolved && !b._isInvolved) return -1;
      if (!a._isInvolved && b._isInvolved) return 1;
      const aId = a.food_id || a.instance_id || '';
      const bId = b.food_id || b.instance_id || '';
      return aId.localeCompare(bId);
    });
  }

  // Add "No Location" group if there are items without location
  if (noLocation.length > 0) {
    noLocation.sort((a, b) => {
      if (a._isInvolved && !b._isInvolved) return -1;
      if (!a._isInvolved && b._isInvolved) return 1;
      const aId = a.food_id || a.instance_id || '';
      const bId = b.food_id || b.instance_id || '';
      return aId.localeCompare(bId);
    });
    groupArray.push({
      location: null,
      foods: noLocation,
      hasInvolvedFood: noLocation.some(f => f._isInvolved),
    });
  }

  return groupArray;
}

function LocationGroup({ group, onFoodNodeClick, onContainerClick, focusedNodeId, focusedContainerId, nodeRefs, side }) {
  const { location, foods, hasInvolvedFood } = group;
  const icon = getLocationIcon(location);
  const displayName = location ? formatLocationName(location) : 'Environment (on surface)';
  const isContainerFocused = location && location === focusedContainerId;
  const isClickableContainer = location !== null;

  return (
    <div
      style={{
        ...styles.locationGroup,
        ...(hasInvolvedFood ? styles.locationGroupInvolved : {}),
        ...(isContainerFocused ? styles.locationGroupFocused : {}),
      }}
    >
      <div
        onClick={() => isClickableContainer && onContainerClick && onContainerClick(location)}
        style={{
          ...styles.locationHeader,
          ...(hasInvolvedFood ? styles.locationHeaderInvolved : {}),
          ...(isContainerFocused ? styles.locationHeaderFocused : {}),
          ...(isClickableContainer ? styles.locationHeaderClickable : {}),
        }}
        title={isClickableContainer ? 'Click to view container history' : undefined}
      >
        <span style={styles.locationIcon}>{icon}</span>
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
              showLocation={false}
              onClick={onFoodNodeClick}
              isFocused={focusedNodeId === foodId}
              nodeRef={nodeRefs ? (el) => { if (el) nodeRefs.current[refKey] = el; } : undefined}
            />
          );
        })}
      </div>
    </div>
  );
}

// Helper to convert food_nodes object to foods array
const convertFoodNodesToArray = (blockGraph) => {
  if (!blockGraph) return [];
  if (blockGraph.foods && Array.isArray(blockGraph.foods)) {
    return blockGraph.foods;
  }
  if (blockGraph.food_nodes && typeof blockGraph.food_nodes === 'object') {
    return Object.values(blockGraph.food_nodes);
  }
  return [];
};

function GraphView({ graph, selectedEventIndex, events, onFoodNodeClick, onContainerClick, focusedNodeId, focusedContainerId, showOnlyInvolved = false }) {
  // Compute before/after snapshots and relevant edges
  const { beforeSnapshot, afterSnapshot, relevantEdges } = useMemo(() => {
    if (!graph || selectedEventIndex === null || selectedEventIndex === undefined) {
      return { beforeSnapshot: null, afterSnapshot: null, relevantEdges: [] };
    }

    const blockGraphs = graph.block_graphs || [];
    const lineageEdges = graph.lineage_edges || [];
    const inventory = graph.inventory || [];

    const afterIdx = selectedEventIndex;
    const beforeIdx = selectedEventIndex - 1;

    // After snapshot
    let afterSnapshot = null;
    if (afterIdx >= 0 && afterIdx < blockGraphs.length) {
      const block = blockGraphs[afterIdx];
      afterSnapshot = {
        ...block,
        foods: convertFoodNodesToArray(block),
      };
    }

    // Before snapshot
    let beforeSnapshot = null;
    if (beforeIdx >= 0 && beforeIdx < blockGraphs.length) {
      const block = blockGraphs[beforeIdx];
      beforeSnapshot = {
        ...block,
        foods: convertFoodNodesToArray(block),
      };
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

    const relevantEdges = lineageEdges.filter(
      (edge) => edge.target_block === afterIdx
    );

    return { beforeSnapshot, afterSnapshot, relevantEdges };
  }, [graph, selectedEventIndex]);

  // Get IDs of foods involved in edges
  const involvedFoodIds = useMemo(() => {
    const ids = new Set();
    relevantEdges.forEach((edge) => {
      if (edge.parent_id) ids.add(edge.parent_id);
      if (edge.parent_instance_id) ids.add(edge.parent_instance_id);
      if (edge.child_id) ids.add(edge.child_id);
      if (edge.child_instance_id) ids.add(edge.child_instance_id);
      if (edge.source_id) ids.add(edge.source_id);
      if (edge.target_id) ids.add(edge.target_id);
    });
    return ids;
  }, [relevantEdges]);

  // Refs and state for SVG edge visualization
  const nodeRefs = useRef({});
  const containerRef = useRef(null);
  const [edgePaths, setEdgePaths] = useState([]);

  // Calculate edge paths
  const calculateEdgePaths = useCallback(() => {
    if (!containerRef.current || relevantEdges.length === 0) {
      setEdgePaths([]);
      return;
    }

    const containerRect = containerRef.current.getBoundingClientRect();
    const paths = [];

    relevantEdges.forEach((edge, idx) => {
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
        id: `${parentId}-${childId}-${idx}`,
        d: `M ${fromX} ${fromY} C ${midX} ${fromY}, ${midX} ${toY}, ${toX} ${toY}`,
        color: EDGE_COLORS[edge.derivation_type] || EDGE_COLORS.default,
        type: edge.derivation_type,
        labelX: midX,
        labelY: midY,
      });
    });

    setEdgePaths(paths);
  }, [relevantEdges]);

  // Recalculate paths after render
  useEffect(() => {
    const timer = setTimeout(calculateEdgePaths, 100);
    return () => clearTimeout(timer);
  }, [calculateEdgePaths, beforeSnapshot, afterSnapshot, showOnlyInvolved]);

  // Clear refs when event changes
  useEffect(() => {
    nodeRefs.current = {};
  }, [selectedEventIndex]);

  if (!graph) {
    return (
      <div style={styles.container}>
        <div style={styles.header}>Graph Visualization</div>
        <div style={styles.noSelection}>
          Load data files to view the graph
        </div>
      </div>
    );
  }

  if (selectedEventIndex === null || selectedEventIndex === undefined) {
    return (
      <div style={styles.container}>
        <div style={styles.header}>Graph Visualization</div>
        <div style={styles.noSelection}>
          Select an event from the list to view the graph snapshot
        </div>
      </div>
    );
  }

  const selectedEvent = events?.[selectedEventIndex];

  return (
    <div style={styles.container}>
      <div style={styles.header}>
        Graph Visualization - Event {selectedEvent?.event_id || selectedEventIndex + 1}
        {selectedEvent?.primary_action && `: ${selectedEvent.primary_action}`}
      </div>

      <div ref={containerRef} style={{ ...styles.graphContainer, position: 'relative' }}>
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
            <marker
              id="arrowhead"
              markerWidth="8"
              markerHeight="6"
              refX="7"
              refY="3"
              orient="auto"
            >
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
                markerEnd="url(#arrowhead)"
                opacity={0.8}
              />
              <rect
                x={path.labelX - 24}
                y={path.labelY - 8}
                width={48}
                height={16}
                fill="white"
                opacity={0.9}
                rx={3}
              />
              <text
                x={path.labelX}
                y={path.labelY + 4}
                textAnchor="middle"
                fontSize="10"
                fontWeight="bold"
                fill={path.color}
              >
                {path.type}
              </text>
            </g>
          ))}
        </svg>

        {/* Before Snapshot */}
        <div style={styles.snapshotColumn}>
          <div style={{ ...styles.snapshotHeader, ...styles.snapshotHeaderBefore }}>
            Before (Block {beforeSnapshot?.block_id ?? beforeSnapshot?.block_idx ?? 'Initial'})
          </div>
          <div style={styles.nodesContainer}>
            {beforeSnapshot?.foods?.length > 0 ? (
              groupFoodsByLocation(beforeSnapshot.foods, involvedFoodIds, showOnlyInvolved).map((group, idx) => (
                <LocationGroup
                  key={group.location || `no-loc-${idx}`}
                  group={group}
                  onFoodNodeClick={onFoodNodeClick}
                  onContainerClick={onContainerClick}
                  focusedNodeId={focusedNodeId}
                  focusedContainerId={focusedContainerId}
                  nodeRefs={nodeRefs}
                  side="before"
                />
              ))
            ) : (
              <div style={{ color: '#999', textAlign: 'center' }}>No foods</div>
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
                <LocationGroup
                  key={group.location || `no-loc-${idx}`}
                  group={group}
                  onFoodNodeClick={onFoodNodeClick}
                  onContainerClick={onContainerClick}
                  focusedNodeId={focusedNodeId}
                  focusedContainerId={focusedContainerId}
                  nodeRefs={nodeRefs}
                  side="after"
                />
              ))
            ) : (
              <div style={{ color: '#999', textAlign: 'center' }}>No foods</div>
            )}
          </div>
        </div>
      </div>

      {/* Legend */}
      <div style={styles.legend}>
        <span style={{ fontWeight: 'bold', marginRight: '10px' }}>Edge Types:</span>
        {Object.entries(EDGE_STYLES).map(([type, style]) => (
          <div key={type} style={styles.legendItem}>
            <div style={{ ...styles.legendColor, backgroundColor: style.backgroundColor }} />
            <span>{type}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

export default GraphView;

import React, { useMemo } from 'react';

const styles = {
  container: {
    backgroundColor: '#fafafa',
    borderRadius: '8px',
    padding: '15px',
    height: '100%',
    display: 'flex',
    flexDirection: 'column',
    overflow: 'hidden',
  },
  header: {
    fontSize: '16px',
    fontWeight: 'bold',
    marginBottom: '15px',
    color: '#333',
  },
  graphContainer: {
    flex: 1,
    display: 'flex',
    gap: '20px',
    overflow: 'auto',
  },
  snapshotColumn: {
    flex: 1,
    display: 'flex',
    flexDirection: 'column',
    minWidth: '250px',
  },
  snapshotHeader: {
    fontSize: '14px',
    fontWeight: 'bold',
    padding: '10px',
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
    padding: '10px',
    overflowY: 'auto',
  },
  foodNode: {
    padding: '10px',
    marginBottom: '8px',
    borderRadius: '6px',
    border: '2px solid #2196F3',
    backgroundColor: '#e3f2fd',
    fontSize: '12px',
  },
  foodNodeConsumed: {
    border: '2px solid #9e9e9e',
    backgroundColor: '#f5f5f5',
    opacity: 0.7,
  },
  foodNodeInvolved: {
    border: '2px solid #ff9800',
    backgroundColor: '#fff3e0',
  },
  foodId: {
    fontWeight: 'bold',
    color: '#1565c0',
    marginBottom: '4px',
    wordBreak: 'break-all',
  },
  foodProperty: {
    color: '#555',
    marginBottom: '2px',
  },
  edgesColumn: {
    width: '150px',
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'flex-start',
    paddingTop: '40px',
  },
  edge: {
    padding: '6px 10px',
    marginBottom: '6px',
    borderRadius: '4px',
    fontSize: '11px',
    textAlign: 'center',
    minWidth: '100px',
  },
  edgeSplit: { backgroundColor: '#c8e6c9', color: '#2e7d32' },
  edgeMerge: { backgroundColor: '#e1bee7', color: '#7b1fa2' },
  edgeUpdate: { backgroundColor: '#bbdefb', color: '#1565c0' },
  edgeIdentity: { backgroundColor: '#ffe0b2', color: '#e65100' },
  edgeTransfer: { backgroundColor: '#cfd8dc', color: '#455a64' },
  edgeConsume: { backgroundColor: '#ffcdd2', color: '#c62828' },
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
};

const EDGE_STYLES = {
  split: styles.edgeSplit,
  merge: styles.edgeMerge,
  update: styles.edgeUpdate,
  identity_transform: styles.edgeIdentity,
  transfer: styles.edgeTransfer,
  consume: styles.edgeConsume,
};

function FoodNode({ food, isInvolved }) {
  const isConsumed = food.status === 'consumed';
  // Handle both flat structure and nested state structure
  const form = food.form || food.state?.form_state || 'unknown';
  const quantity = food.quantity || food.state?.quantity || 'unknown';
  const location = food.location || null;
  const foodId = food.food_id || food.instance_id || 'unknown';

  return (
    <div
      style={{
        ...styles.foodNode,
        ...(isConsumed ? styles.foodNodeConsumed : {}),
        ...(isInvolved ? styles.foodNodeInvolved : {}),
      }}
    >
      <div style={styles.foodId}>{foodId}</div>
      <div style={styles.foodProperty}>
        <strong>form:</strong> {form}
      </div>
      <div style={styles.foodProperty}>
        <strong>qty:</strong> {quantity}
      </div>
      {location && (
        <div style={styles.foodProperty}>
          <strong>loc:</strong> {location}
        </div>
      )}
      {food.status && (
        <div style={styles.foodProperty}>
          <strong>status:</strong> {food.status}
        </div>
      )}
    </div>
  );
}

// Helper to convert food_nodes object to foods array
const convertFoodNodesToArray = (blockGraph) => {
  if (!blockGraph) return [];
  // If it already has foods array, use it
  if (blockGraph.foods && Array.isArray(blockGraph.foods)) {
    return blockGraph.foods;
  }
  // Convert food_nodes object to array
  if (blockGraph.food_nodes && typeof blockGraph.food_nodes === 'object') {
    return Object.values(blockGraph.food_nodes);
  }
  return [];
};

function GraphView({ graph, selectedEventIndex, events }) {
  // Compute before/after snapshots and relevant edges
  const { beforeSnapshot, afterSnapshot, relevantEdges } = useMemo(() => {
    if (!graph || selectedEventIndex === null || selectedEventIndex === undefined) {
      return { beforeSnapshot: null, afterSnapshot: null, relevantEdges: [] };
    }

    const blockGraphs = graph.block_graphs || [];
    const lineageEdges = graph.lineage_edges || [];
    const inventory = graph.inventory || [];

    // Event N (1-indexed in state_change.json) corresponds to:
    // - Before: block_graphs[N-2] or inventory if N=1
    // - After: block_graphs[N-1]
    // But selectedEventIndex is 0-indexed in our array

    // After event at index i, the state is block_graphs[i]
    const afterIdx = selectedEventIndex;
    const beforeIdx = selectedEventIndex - 1;

    // After snapshot - convert food_nodes object to array
    let afterSnapshot = null;
    if (afterIdx >= 0 && afterIdx < blockGraphs.length) {
      const block = blockGraphs[afterIdx];
      afterSnapshot = {
        ...block,
        foods: convertFoodNodesToArray(block),
      };
    }

    // Before snapshot: previous block_graph or inventory
    let beforeSnapshot = null;
    if (beforeIdx >= 0 && beforeIdx < blockGraphs.length) {
      const block = blockGraphs[beforeIdx];
      beforeSnapshot = {
        ...block,
        foods: convertFoodNodesToArray(block),
      };
    } else if (selectedEventIndex === 0) {
      // First event: before is the initial inventory
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

    // Find edges that target the selected event's block
    // target_block corresponds to the block index after the event
    const relevantEdges = lineageEdges.filter(
      (edge) => edge.target_block === afterIdx
    );

    return { beforeSnapshot, afterSnapshot, relevantEdges };
  }, [graph, selectedEventIndex]);

  // Get IDs of foods involved in edges
  const involvedFoodIds = useMemo(() => {
    const ids = new Set();
    relevantEdges.forEach((edge) => {
      // Handle multiple naming conventions: parent_id, parent_instance_id, source_id
      if (edge.parent_id) ids.add(edge.parent_id);
      if (edge.parent_instance_id) ids.add(edge.parent_instance_id);
      if (edge.child_id) ids.add(edge.child_id);
      if (edge.child_instance_id) ids.add(edge.child_instance_id);
      if (edge.source_id) ids.add(edge.source_id);
      if (edge.target_id) ids.add(edge.target_id);
    });
    return ids;
  }, [relevantEdges]);

  // Helper to check if a food is involved
  const isFoodInvolved = (food) => {
    const id = food.food_id || food.instance_id;
    return involvedFoodIds.has(id);
  };

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

      <div style={styles.graphContainer}>
        {/* Before Snapshot */}
        <div style={styles.snapshotColumn}>
          <div style={{ ...styles.snapshotHeader, ...styles.snapshotHeaderBefore }}>
            Before (Block {beforeSnapshot?.block_id ?? beforeSnapshot?.block_idx ?? 'Initial'})
          </div>
          <div style={styles.nodesContainer}>
            {beforeSnapshot?.foods?.length > 0 ? (
              beforeSnapshot.foods.map((food, idx) => (
                <FoodNode
                  key={food.food_id || food.instance_id || idx}
                  food={food}
                  isInvolved={isFoodInvolved(food)}
                />
              ))
            ) : (
              <div style={{ color: '#999', textAlign: 'center' }}>No foods</div>
            )}
          </div>
        </div>

        {/* Edges Column */}
        <div style={styles.edgesColumn}>
          <div style={{ fontSize: '12px', fontWeight: 'bold', marginBottom: '10px' }}>
            Edges ({relevantEdges.length})
          </div>
          {relevantEdges.map((edge, idx) => {
            const parentId = edge.parent_id || edge.parent_instance_id || '';
            const childId = edge.child_id || edge.child_instance_id || '';
            return (
              <div
                key={idx}
                style={{
                  ...styles.edge,
                  ...(EDGE_STYLES[edge.derivation_type] || styles.edgeUpdate),
                }}
                title={`${parentId} → ${childId}`}
              >
                <div style={{ fontWeight: 'bold' }}>{edge.derivation_type}</div>
                <div style={{ fontSize: '10px', marginTop: '2px' }}>
                  {parentId?.split('_').slice(-1)[0]} → {childId?.split('_').slice(-1)[0]}
                </div>
              </div>
            );
          })}
          {relevantEdges.length === 0 && (
            <div style={{ fontSize: '11px', color: '#999' }}>No edges</div>
          )}
        </div>

        {/* After Snapshot */}
        <div style={styles.snapshotColumn}>
          <div style={{ ...styles.snapshotHeader, ...styles.snapshotHeaderAfter }}>
            After (Block {afterSnapshot?.block_id ?? afterSnapshot?.block_idx ?? selectedEventIndex + 1})
          </div>
          <div style={styles.nodesContainer}>
            {afterSnapshot?.foods?.length > 0 ? (
              afterSnapshot.foods.map((food, idx) => (
                <FoodNode
                  key={food.food_id || food.instance_id || idx}
                  food={food}
                  isInvolved={isFoodInvolved(food)}
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

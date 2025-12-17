/**
 * Lineage Graph Utilities
 *
 * Pre-computes ancestry and descendant paths for all food nodes
 * from the lineage_edges data structure.
 */

/**
 * Build a lineage graph from lineage_edges for fast traversal.
 *
 * @param {Array} lineageEdges - Array of lineage edge objects
 * @param {Array} blockGraphs - Array of block graph snapshots
 * @returns {Object} Lineage graph with parent/child maps and node info
 */
export function buildLineageGraph(lineageEdges, blockGraphs) {
  // Maps for traversal
  const parentMap = {}; // child_id -> [{parent_id, derivation_type, source_block, target_block}]
  const childMap = {};  // parent_id -> [{child_id, derivation_type, source_block, target_block}]

  // Build adjacency lists
  for (const edge of lineageEdges || []) {
    const { child_instance_id, parent_instance_id, derivation_type, source_block, target_block, timestamp } = edge;

    // Parent map (for tracing ancestors)
    if (!parentMap[child_instance_id]) {
      parentMap[child_instance_id] = [];
    }
    parentMap[child_instance_id].push({
      parent_id: parent_instance_id,
      derivation_type,
      source_block,
      target_block,
      timestamp
    });

    // Child map (for tracing descendants)
    if (!childMap[parent_instance_id]) {
      childMap[parent_instance_id] = [];
    }
    childMap[parent_instance_id].push({
      child_id: child_instance_id,
      derivation_type,
      source_block,
      target_block,
      timestamp
    });
  }

  // Build node info index: instance_id -> {block_idx, node_data}
  // For each node, find the block where it first appears or last appears
  const nodeInfo = {};

  for (let blockIdx = 0; blockIdx < (blockGraphs || []).length; blockIdx++) {
    const block = blockGraphs[blockIdx];
    const foodNodes = block.food_nodes || {};

    for (const [instanceId, nodeData] of Object.entries(foodNodes)) {
      // Store the latest block info for each node
      nodeInfo[instanceId] = {
        block_idx: blockIdx,
        block_id: block.block_id,
        video_id: block.video_id,
        ...nodeData
      };
    }
  }

  return { parentMap, childMap, nodeInfo, blockGraphs };
}

/**
 * Trace the full ancestry path for a given node.
 * Returns an array of path steps from the earliest ancestor to the target node.
 *
 * @param {string} nodeId - The target node instance_id
 * @param {Object} lineageGraph - Pre-built lineage graph
 * @returns {Array} Path array with steps
 */
export function traceAncestry(nodeId, lineageGraph) {
  const { parentMap, nodeInfo } = lineageGraph;
  const visited = new Set();
  const paths = [];

  function dfs(currentId, pathSoFar) {
    if (visited.has(currentId)) return;
    visited.add(currentId);

    const parents = parentMap[currentId] || [];

    // Filter to get "real" derivation (not identity transfers to self)
    const realParents = parents.filter(p =>
      p.parent_id !== currentId ||
      p.derivation_type !== 'transfer'
    );

    if (realParents.length === 0) {
      // This is a root node (from inventory)
      paths.push([...pathSoFar]);
    } else {
      for (const parent of realParents) {
        const parentInfo = nodeInfo[parent.parent_id] || {};

        dfs(parent.parent_id, [
          {
            node: parent.parent_id,
            location: parentInfo.location || null,
            state: parentInfo.state || {},
            status: parentInfo.status,
            food_noun: parentInfo.food_noun,
            action_next: `${parent.derivation_type.toUpperCase()}`,
            target_block: parent.target_block,
            source_block: parent.source_block,
            timestamp: parent.timestamp
          },
          ...pathSoFar
        ]);
      }
    }
  }

  // Start DFS from the target node
  const targetInfo = nodeInfo[nodeId] || {};
  dfs(nodeId, [{
    node: nodeId,
    location: targetInfo.location || null,
    state: targetInfo.state || {},
    status: targetInfo.status,
    food_noun: targetInfo.food_noun,
    action_next: null, // Terminal
    isCurrent: true
  }]);

  // Return the longest path (most complete ancestry)
  // For nodes with multiple parents (merge), we get multiple paths
  return paths.sort((a, b) => b.length - a.length)[0] || [];
}

/**
 * Trace descendants (forward in time) from a given node.
 *
 * @param {string} nodeId - The source node instance_id
 * @param {Object} lineageGraph - Pre-built lineage graph
 * @returns {Array} Array of descendant nodes with derivation info
 */
export function traceDescendants(nodeId, lineageGraph) {
  const { childMap, nodeInfo } = lineageGraph;
  const visited = new Set();
  const descendants = [];

  function bfs(startId) {
    const queue = [startId];

    while (queue.length > 0) {
      const currentId = queue.shift();
      if (visited.has(currentId)) continue;
      visited.add(currentId);

      const children = childMap[currentId] || [];

      for (const child of children) {
        // Skip identity transfers to self
        if (child.child_id === currentId && child.derivation_type === 'transfer') {
          continue;
        }

        const childInfo = nodeInfo[child.child_id] || {};

        descendants.push({
          node: child.child_id,
          from: currentId,
          derivation_type: child.derivation_type,
          location: childInfo.location || null,
          state: childInfo.state || {},
          status: childInfo.status,
          food_noun: childInfo.food_noun,
          target_block: child.target_block,
          timestamp: child.timestamp
        });

        queue.push(child.child_id);
      }
    }
  }

  bfs(nodeId);
  return descendants;
}

/**
 * Get all unique food nodes that appear across all blocks.
 *
 * @param {Object} lineageGraph - Pre-built lineage graph
 * @returns {Array} Array of unique node IDs
 */
export function getAllNodes(lineageGraph) {
  return Object.keys(lineageGraph.nodeInfo || {});
}

/**
 * Pre-compute ancestry paths for all nodes.
 *
 * @param {Object} lineageGraph - Pre-built lineage graph
 * @returns {Object} Map of nodeId -> ancestry path
 */
export function precomputeAllAncestry(lineageGraph) {
  const allNodes = getAllNodes(lineageGraph);
  const ancestryCache = {};

  for (const nodeId of allNodes) {
    ancestryCache[nodeId] = traceAncestry(nodeId, lineageGraph);
  }

  return ancestryCache;
}

/**
 * Trace what foods have been added to a container over time.
 * Scans all block_graphs to find when foods first appeared in the container.
 * Also finds the parent/source for each food via lineage edges.
 *
 * @param {string} containerId - The container ID (e.g., "pan_001", "pot_001")
 * @param {Object} lineageGraph - Pre-built lineage graph
 * @param {Array} events - Array of events with timestamps
 * @returns {Array} Array of addition events sorted by block index
 */
export function traceContainerHistory(containerId, lineageGraph, events) {
  const { blockGraphs, parentMap, nodeInfo } = lineageGraph;
  const history = [];
  const seenFoods = new Set(); // Track which foods we've already recorded

  if (!blockGraphs || blockGraphs.length === 0) {
    return history;
  }

  // Scan each block graph chronologically
  for (let blockIdx = 0; blockIdx < blockGraphs.length; blockIdx++) {
    const block = blockGraphs[blockIdx];
    const foodNodes = block.food_nodes || {};

    // Find foods in this container that we haven't seen before
    for (const [instanceId, nodeData] of Object.entries(foodNodes)) {
      if (nodeData.location === containerId && !seenFoods.has(instanceId)) {
        seenFoods.add(instanceId);

        // Get event info for this block
        const event = events?.[blockIdx];
        const timestamp = event?.timestamp_start || null;

        // Find parent/source via lineage edges
        let sourceInfo = null;
        const parents = parentMap?.[instanceId] || [];
        if (parents.length > 0) {
          // Get the parent that led to this food appearing in this block
          const relevantParent = parents.find(p => p.target_block === blockIdx) || parents[0];
          const parentId = relevantParent.parent_id;
          const parentNodeInfo = nodeInfo?.[parentId] || {};

          // Get parent's location from the block before this one (source_block)
          let parentLocation = parentNodeInfo.location || null;
          if (relevantParent.source_block !== undefined && relevantParent.source_block >= 0) {
            const sourceBlock = blockGraphs[relevantParent.source_block];
            const sourceNodeData = sourceBlock?.food_nodes?.[parentId];
            if (sourceNodeData) {
              parentLocation = sourceNodeData.location || null;
            }
          }

          sourceInfo = {
            parent_id: parentId,
            parent_noun: parentNodeInfo.food_noun || parentId.split('_')[0],
            parent_location: parentLocation,
            derivation_type: relevantParent.derivation_type,
          };
        }

        history.push({
          food_id: instanceId,
          food_noun: nodeData.food_noun || instanceId.split('_')[0],
          state: nodeData.state || {},
          status: nodeData.status || 'active',
          block_idx: blockIdx,
          block_id: block.block_id,
          timestamp: timestamp,
          event_id: event?.event_id || blockIdx + 1,
          primary_action: event?.primary_action || 'Unknown action',
          source: sourceInfo, // Parent/source information
        });
      }
    }
  }

  return history;
}

/**
 * Get all unique container IDs that appear across all blocks.
 *
 * @param {Object} lineageGraph - Pre-built lineage graph
 * @returns {Array} Array of unique container IDs
 */
export function getAllContainers(lineageGraph) {
  const { blockGraphs } = lineageGraph;
  const containers = new Set();

  for (const block of blockGraphs || []) {
    const foodNodes = block.food_nodes || {};
    for (const nodeData of Object.values(foodNodes)) {
      if (nodeData.location) {
        containers.add(nodeData.location);
      }
    }
  }

  return Array.from(containers);
}

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

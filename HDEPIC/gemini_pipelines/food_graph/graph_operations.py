"""
Graph Operations for Spatio-Temporal Food Graph

This module provides operations for:
- Processing blocks to produce BlockGraphs
- Applying transactions (TRANSFER, SPLIT, MERGE, CONSUME, UPDATE)
- Helper functions for graph manipulation

EDGE CONSTRAINTS (strictly enforced):
1. CONTAINMENT EDGES: Within a single block only
   - food_node and container_node must be in the SAME BlockGraph
   - These are spatial relationships at a point in time

2. LINEAGE EDGES: Between ADJACENT blocks only
   - source_block = N, target_block = N+1 (always consecutive)
   - Connect parent in block N to child in block N+1
   - These represent temporal derivation (split, merge_transform)
"""

import logging
from typing import Dict, List, Tuple, Optional, Any
from copy import deepcopy
import re

from .data_structures import (
    FoodNode,
    ContainerNode,
    FoodState,
    BlockGraph,
    ContainmentEdge,
    LineageEdge,
    NodeStatus,
    LocationRegistry,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Local Pool for Just-in-Time Resolution
# ============================================================================

class LocalPool:
    """
    Track items created during current block's transaction processing.

    This enables "Just-in-Time" resolution where VLM can refer to newly
    created items by their food_noun (e.g., "butter") instead of the
    system-assigned UUID (e.g., "butter_005").

    Items are marked 'matched' after being resolved to ensure sequential
    references to the same food_noun target different physical items.
    Items are marked 'unavailable' after being consumed/discarded/split.
    """

    def __init__(self):
        self.items: List[Dict[str, Any]] = []  # [{noun, uuid, available, matched}]

    def add(self, food_noun: str, instance_id: str) -> None:
        """Add a newly created item to the pool"""
        self.items.append({
            'noun': food_noun.lower().strip().replace(' ', '_'),
            'uuid': instance_id,
            'available': True,
            'matched': False  # Track if already resolved in this batch
        })

    def find_and_claim(self, food_noun: str) -> Optional[str]:
        """
        Find the first available AND unmatched item matching the food_noun.

        When found, marks the item as 'matched' to ensure subsequent
        resolutions for the same food_noun target a different physical item.

        Returns the instance_id (UUID) if found, None otherwise.
        """
        normalized = food_noun.lower().strip().replace(' ', '_')
        for item in self.items:
            if item['noun'] == normalized and item['available'] and not item['matched']:
                item['matched'] = True  # Claim this item
                return item['uuid']
        return None

    def find_available(self, food_noun: str) -> Optional[str]:
        """
        Find the first available item matching the food_noun (legacy method).

        Prefer find_and_claim() for sequential resolution.
        """
        normalized = food_noun.lower().strip().replace(' ', '_')
        for item in self.items:
            if item['noun'] == normalized and item['available']:
                return item['uuid']
        return None

    def mark_unavailable(self, instance_id: str) -> None:
        """Mark an item as unavailable (consumed/discarded/split)"""
        for item in self.items:
            if item['uuid'] == instance_id:
                item['available'] = False
                break

    def __repr__(self) -> str:
        available = sum(1 for i in self.items if i['available'])
        matched = sum(1 for i in self.items if i['matched'])
        return f"LocalPool({len(self.items)} items, {available} available, {matched} matched)"


# Global counter for generating instance IDs
_instance_counters: Dict[str, int] = {}


def reset_instance_counters():
    """Reset all instance counters (useful for testing)"""
    global _instance_counters
    _instance_counters = {}


def generate_instance_id(food_noun: str) -> str:
    """
    Generate a unique instance ID for a food item.

    Format: {food_noun}_{counter:03d}
    """
    global _instance_counters
    food_noun_clean = food_noun.lower().strip().replace(' ', '_')

    if food_noun_clean not in _instance_counters:
        _instance_counters[food_noun_clean] = 0

    _instance_counters[food_noun_clean] += 1
    return f"{food_noun_clean}_{_instance_counters[food_noun_clean]:03d}"


def resolve_food_id(
    graph: BlockGraph,
    food_id: str,
    warnings: List[str] = None,
    local_pool: Optional[LocalPool] = None
) -> Optional[str]:
    """
    Resolve a food_id to an actual instance ID in the graph.

    Resolution order:
    1. Direct match in graph.food_nodes
    2. If local_pool provided, check for available items created in this block
    3. Match by food_noun in graph (highest-numbered active instance)

    Args:
        graph: Current BlockGraph
        food_id: ID from VLM (might be instance_id or food_noun)
        warnings: Optional list to collect resolution messages
        local_pool: Optional LocalPool for just-in-time resolution of newly created items

    Returns:
        Resolved instance_id or None if not found
    """
    # Direct match in graph
    if food_id in graph.food_nodes:
        return food_id

    # Normalize the food_id for matching (e.g., "orange half" -> "orange_half")
    normalized = food_id.lower().strip().replace(' ', '_')

    # Priority 1: Check local_pool for newly created items (Just-in-Time resolution)
    # Use find_and_claim to ensure sequential references to same food_noun
    # target different physical items
    if local_pool is not None:
        pool_match = local_pool.find_and_claim(normalized)
        if pool_match:
            if warnings is not None:
                warnings.append(f"RESOLVED (pool): '{food_id}' -> '{pool_match}'")
            return pool_match

    # Priority 2: Match by food_noun in graph
    def find_matches(target_noun):
        matches = []
        for instance_id, node in graph.food_nodes.items():
            if node.status != NodeStatus.ACTIVE:
                continue
            node_noun_normalized = node.food_noun.lower().strip().replace(' ', '_')
            if node_noun_normalized == target_noun:
                matches.append(instance_id)
        return matches

    # Try 1: Match by normalized food_noun
    matches = find_matches(normalized)

    # Try 2: If no match, and input looks like ID (noun_NNN), try stripping suffix
    if not matches:
        # Regex to strip _NNN suffix
        match = re.match(r'^(.*)_\d+$', normalized)
        if match:
            stripped_noun = match.group(1)
            matches = find_matches(stripped_noun)
            if matches and warnings is not None:
                warnings.append(f"RESOLVED: Guessed ID '{food_id}' -> Matched noun '{stripped_noun}'")

    if not matches:
        return None

    # Sort by instance number (extract number from ID like "orange_half_002")
    def get_instance_num(iid: str) -> int:
        match = re.search(r'_(\d+)$', iid)
        return int(match.group(1)) if match else 0

    matches.sort(key=get_instance_num, reverse=True)
    resolved_id = matches[0]

    if warnings is not None:
        warnings.append(f"RESOLVED: '{food_id}' -> '{resolved_id}'")

    return resolved_id


def snapshot_active_state(graph: BlockGraph, block_id: int) -> BlockGraph:
    """
    Create a snapshot of current graph state with only ACTIVE nodes.

    Used to save block state for visualization - consumed/split nodes are excluded.
    """
    snapshot = BlockGraph(
        video_id=graph.video_id,
        block_id=block_id,
        block_start_time=graph.block_start_time,
        block_end_time=graph.block_end_time,
        food_nodes={},
        container_nodes={},
        containment_edges=[]
    )

    # Only copy ACTIVE food nodes
    for instance_id, node in graph.food_nodes.items():
        if node.status == NodeStatus.ACTIVE:
            snapshot.food_nodes[instance_id] = node.copy()

    # Copy all container nodes
    for container_id, node in graph.container_nodes.items():
        snapshot.container_nodes[container_id] = node.copy()

    # Copy containment edges for active food nodes only
    for edge in graph.containment_edges:
        if edge.food_instance_id in snapshot.food_nodes:
            snapshot.containment_edges.append(ContainmentEdge(
                food_instance_id=edge.food_instance_id,
                container_id=edge.container_id
            ))

    return snapshot


def ensure_container_exists(
    graph: BlockGraph,
    container_id: str,
    block: Dict,
    zone: Optional[str] = None
) -> None:
    """
    Ensure a container exists in the graph, creating it if necessary.
    """
    if container_id not in graph.container_nodes:
        graph.container_nodes[container_id] = ContainerNode(
            container_id=container_id,
            zone=zone or "unknown",
            created_at=block.get('block_start_time', 0.0),
            created_in_video=block.get('video_id', ''),
            created_in_block=block.get('block_id', 0)
        )


def get_container_for_food(graph: BlockGraph, food_id: str) -> Optional[str]:
    """Get the container/location for a food item in the current graph.

    Works with both V2 model (FoodNode.location) and V1 model (containment_edges).
    """
    # V2 model: check FoodNode.location directly
    if food_id in graph.food_nodes:
        location = graph.food_nodes[food_id].location
        if location:
            return location

    # V1 model fallback: check containment_edges
    for edge in graph.containment_edges:
        if edge.food_instance_id == food_id:
            return edge.container_id
    return None


def decrease_quantity(current: str) -> str:
    """Decrease quantity level"""
    progression = ["full", "partial", "nearly_empty"]
    if current in progression:
        idx = progression.index(current)
        if idx < len(progression) - 1:
            return progression[idx + 1]
    return current


def increase_quantity(current: str) -> str:
    """Increase quantity level"""
    progression = ["nearly_empty", "partial", "full"]
    if current in progression:
        idx = progression.index(current)
        if idx < len(progression) - 1:
            return progression[idx + 1]
    return current


def create_lineage_edge(
    child_id: str,
    parent_id: str,
    derivation_type: str,
    current_block_idx: int,
    timestamp: float
) -> LineageEdge:
    """
    Create a lineage edge between ADJACENT blocks.

    ENFORCES: source_block = current_block_idx - 1, target_block = current_block_idx
    Lineage edges ALWAYS connect block N-1 to block N (using global block_idx).
    """
    source_block = current_block_idx - 1
    target_block = current_block_idx

    # Sanity check: source must be non-negative
    if source_block < 0:
        raise ValueError(f"Cannot create lineage edge: source_block would be {source_block}")

    return LineageEdge(
        child_instance_id=child_id,
        parent_instance_id=parent_id,
        derivation_type=derivation_type,
        source_block=source_block,
        target_block=target_block,
        timestamp=timestamp
    )


# ============================================================================
# Transaction Handlers
# ============================================================================

def execute_transfer(
    graph: BlockGraph,
    txn: Dict[str, Any],
    block: Dict,
    warnings: List[str] = None,
    local_pool: Optional[LocalPool] = None,
    location_registry: Optional[LocationRegistry] = None
) -> List[LineageEdge]:
    """
    TRANSFER: Move food to different location.

    Creates a lineage edge with derivation_type='transfer' to track location changes.

    V3 Container-Only Model:
    - to_location: "pan_001" → move to container
    - to_location: null → remove from container (place on surface)

    Supports both V2/V3 (to_location) and V1 (to_container + to_zone) formats.
    """
    raw_food_id = txn['food_id']
    current_block_idx = block.get('block_idx', 0)

    # V3/V2 model: to_location (can be null for surfaces); V1 fallback: to_container
    # Check for explicit null value (VLM can send null to indicate surface placement)
    to_location = txn.get('to_location')
    if to_location is None and 'to_location' not in txn:
        # Fall back to V1 format if to_location not specified at all
        to_location = txn.get('to_container')
    to_zone = txn.get('to_zone', 'unknown')

    # V3: null is valid (place on surface), so only error if neither field exists
    if to_location is None and 'to_location' not in txn and 'to_container' not in txn:
        msg = f"TRANSFER: missing to_location/to_container"
        logger.warning(msg)
        if warnings is not None:
            warnings.append(msg)
        return []

    # Resolve food_id (may be food_noun instead of instance_id)
    food_id = resolve_food_id(graph, raw_food_id, warnings, local_pool)
    if food_id is None:
        msg = f"TRANSFER: unknown food_id '{raw_food_id}'"
        logger.warning(msg)
        if warnings is not None:
            warnings.append(msg)
        return []

    food = graph.food_nodes[food_id]
    if food.status != NodeStatus.ACTIVE:
        msg = f"TRANSFER: inactive food_id '{food_id}'"
        logger.warning(msg)
        if warnings is not None:
            warnings.append(msg)
        return []

    lineage_edges = []

    # V3 model: Resolve location via registry (auto-numbers containers, returns None for surfaces)
    if location_registry and to_location is not None:
        to_location = location_registry.get_or_create_location(to_location)

    # Update food's location directly (V2/V3 model)
    # V3: to_location=null means food is on a surface (not in any container)
    food.location = to_location

    # Also update containment edges for backward compatibility
    graph.containment_edges = [
        e for e in graph.containment_edges
        if e.food_instance_id != food_id
    ]

    # Only create containment edge if food is in a container (V3: null = on surface)
    if to_location is not None:
        ensure_container_exists(graph, to_location, block, to_zone)
        graph.containment_edges.append(ContainmentEdge(
            food_instance_id=food_id,
            container_id=to_location
        ))

    # Create lineage edge if food existed in a previous block
    if food.created_in_block < current_block_idx:
        lineage_edges.append(create_lineage_edge(
            child_id=food_id,
            parent_id=food_id,
            derivation_type='transfer',
            current_block_idx=current_block_idx,
            timestamp=block.get('block_start_time', 0.0)
        ))

    return lineage_edges


def execute_split(
    graph: BlockGraph,
    txn: Dict[str, Any],
    block: Dict,
    warnings: List[str] = None,
    local_pool: Optional[LocalPool] = None,
    location_registry: Optional[LocationRegistry] = None
) -> List[LineageEdge]:
    """
    SPLIT: Divide food into portions (supports "Directional Split" pattern).

    - partial: Parent survives with reduced quantity, child(ren) created
    - complete: Parent retired (SPLIT_SOURCE), N children created

    DIRECTIONAL SPLIT: Children can specify a "destination" field for immediate
    placement (e.g., "grind pepper into pan" → child goes directly to pan).
    If no destination, children inherit parent's location.

    Lineage edges are only created if parent existed in a PREVIOUS block.
    If parent was created in the same block, it's a within-block transformation
    and no lineage edge is needed (spatial relationship only).

    Newly created children are added to local_pool for just-in-time resolution
    by subsequent transactions in the same block.
    """
    raw_parent_id = txn['parent_id']
    subtype = txn.get('subtype', 'partial')
    children_specs = txn.get('children', [])
    current_block_idx = block.get('block_idx', 0)  # Use global block_idx

    # Resolve parent_id (may be food_noun instead of instance_id)
    parent_id = resolve_food_id(graph, raw_parent_id, warnings, local_pool)
    if parent_id is None:
        msg = f"SPLIT: unknown parent_id '{raw_parent_id}'"
        logger.warning(msg)
        if warnings is not None:
            warnings.append(msg)
        return []

    parent = graph.food_nodes[parent_id]
    if parent.status != NodeStatus.ACTIVE:
        msg = f"SPLIT: inactive parent_id '{parent_id}'"
        logger.warning(msg)
        if warnings is not None:
            warnings.append(msg)
        return []

    # Get parent's location (V2: from node.location, V1 fallback: containment edge)
    parent_location = parent.location or get_container_for_food(graph, parent_id)
    lineage_edges = []

    # Check if parent was created in a previous block (lineage edge needed)
    # Compare using global block_idx, not per-video block_id
    parent_created_in_previous_block = parent.created_in_block < current_block_idx

    if subtype == 'complete':
        parent.status = NodeStatus.SPLIT_SOURCE
        parent.location = None  # V2: clear parent's location
        # Remove parent from containment (retired) - V1 compat
        graph.containment_edges = [
            e for e in graph.containment_edges
            if e.food_instance_id != parent_id
        ]
        # Mark parent unavailable in pool (if it was created this block)
        if local_pool is not None:
            local_pool.mark_unavailable(parent_id)
    else:  # partial
        parent.state.quantity = decrease_quantity(parent.state.quantity)
        if parent.state.count is not None and parent.state.count > 0:
            parent.state.count -= 1

    # Create children
    # Track created child IDs for return/reference
    created_child_ids = []

    for child_spec in children_specs:
        # Handle case where VLM returns string instead of dict
        if isinstance(child_spec, str):
            child_spec = {'food_noun': child_spec, 'quantity': 'full', 'count': 1}

        # Use child's food_noun if specified, otherwise inherit from parent
        child_food_noun = child_spec.get('food_noun', parent.food_noun)

        # DIRECTIONAL SPLIT: Check for destination field
        # V3 Container-Only Model:
        # - destination: "pan_001" → child goes to container
        # - destination: null → child goes to surface (not in any container)
        # - destination not specified → inherit parent's location
        if 'destination' in child_spec:
            # Explicit destination (can be null for surface placement)
            child_destination = child_spec.get('destination')
            if child_destination is not None and location_registry:
                # Auto-number the destination container (e.g., "pan" -> "pan_001")
                # Returns None if destination is a surface (e.g., "counter")
                child_destination = location_registry.get_or_create_location(child_destination)
            child_location = child_destination  # Could be container_id or None (surface)
        else:
            # No destination specified → inherit parent's location
            child_location = parent_location

        # If count > 1, create multiple separate child nodes (one per physical item)
        # e.g., "pick up 3 oranges" -> create orange_001, orange_002, orange_003
        child_count = child_spec.get('count', 1) or 1

        for _ in range(child_count):
            child_id = generate_instance_id(child_food_noun)
            child = FoodNode(
                instance_id=child_id,
                food_noun=child_food_noun,
                status=NodeStatus.ACTIVE,
                state=FoodState(
                    form_state=child_spec.get('form_state', 'prepared_ingredient'),
                    quantity=child_spec.get('quantity', 'full'),
                    count=1  # Each child is ONE item
                ),
                location=child_location,  # V2: use destination or inherit parent's location
                parent_instance=parent_id,
                created_at=block.get('block_start_time', 0.0),
                created_in_video=block.get('video_id', ''),
                created_in_block=current_block_idx  # Use global block_idx
            )
            graph.food_nodes[child_id] = child
            created_child_ids.append(child_id)

            # Add to local_pool for just-in-time resolution
            if local_pool is not None:
                local_pool.add(child_food_noun, child_id)

            # Add containment edge for V1 compat (V3: only if in a container, not on surface)
            if child_location is not None:
                # Ensure container exists
                ensure_container_exists(graph, child_location, block)
                graph.containment_edges.append(ContainmentEdge(
                    food_instance_id=child_id,
                    container_id=child_location
                ))
            # V3: if child_location is None, food is on a surface (no containment edge)

            # Lineage edge ONLY if parent existed in a previous block
            if parent_created_in_previous_block:
                lineage_edges.append(create_lineage_edge(
                    child_id=child_id,
                    parent_id=parent_id,
                    derivation_type='split',
                    current_block_idx=current_block_idx,  # Use global block_idx
                    timestamp=block.get('block_start_time', 0.0)
                ))

    return lineage_edges


def execute_merge(
    graph: BlockGraph,
    txn: Dict[str, Any],
    block: Dict,
    warnings: List[str] = None,
    local_pool: Optional[LocalPool] = None,
    location_registry: Optional[LocationRegistry] = None
) -> List[LineageEdge]:
    """
    MERGE: Absorb one food into another (Survivor Protocol).

    Subject is consumed into Target (subject dies, target survives).
    If the result should have a new identity, use UPDATE afterward.

    Subtypes:
    - accumulation: Same food_noun, quantities add
    - incorporation: Subject absorbed into target (default)

    Note: 'transformation' subtype removed in favor of Survivor Protocol
    (MERGE incorporation + UPDATE food_noun).

    Creates a lineage edge: subject (absorbed) -> target (survivor)
    """
    raw_subject_id = txn['subject_id']
    raw_target_id = txn['target_id']
    subtype = txn.get('subtype', 'incorporation')
    current_block_idx = block.get('block_idx', 0)

    # Handle legacy 'transformation' subtype - treat as incorporation + warn
    if subtype == 'transformation':
        if warnings is not None:
            new_noun = txn.get('new_food_noun', 'mixture')
            warnings.append(
                f"MERGE transformation deprecated: Use MERGE incorporation + UPDATE food_noun='{new_noun}' instead"
            )
        subtype = 'incorporation'

    # Resolve IDs (may be food_noun instead of instance_id)
    subject_id = resolve_food_id(graph, raw_subject_id, warnings, local_pool)
    if subject_id is None:
        msg = f"MERGE: unknown subject_id '{raw_subject_id}'"
        logger.warning(msg)
        if warnings is not None:
            warnings.append(msg)
        return []

    target_id = resolve_food_id(graph, raw_target_id, warnings, local_pool)
    if target_id is None:
        msg = f"MERGE: unknown target_id '{raw_target_id}'"
        logger.warning(msg)
        if warnings is not None:
            warnings.append(msg)
        return []

    subject = graph.food_nodes[subject_id]
    target = graph.food_nodes[target_id]

    if subject.status != NodeStatus.ACTIVE:
        msg = f"MERGE: inactive subject_id '{subject_id}'"
        logger.warning(msg)
        if warnings is not None:
            warnings.append(msg)
        return []
    if target.status != NodeStatus.ACTIVE:
        msg = f"MERGE: inactive target_id '{target_id}'"
        logger.warning(msg)
        if warnings is not None:
            warnings.append(msg)
        return []

    lineage_edges = []

    # Subject always consumed and removed from spatial graph
    subject.status = NodeStatus.CONSUMED
    subject.location = None  # V2: clear subject's location
    graph.containment_edges = [
        e for e in graph.containment_edges
        if e.food_instance_id != subject_id
    ]
    # Mark subject unavailable in pool
    if local_pool is not None:
        local_pool.mark_unavailable(subject_id)

    if subtype == 'accumulation':
        # Same food type: quantities add up
        target.state.quantity = increase_quantity(target.state.quantity)
        if target.state.count is not None and subject.state.count is not None:
            target.state.count += subject.state.count
    else:
        # incorporation (default): subject absorbed, target unchanged structurally
        pass

    # Create lineage edge: subject (absorbed) -> target (survivor)
    # Only if subject existed in a previous block
    if subject.created_in_block < current_block_idx:
        lineage_edges.append(create_lineage_edge(
            child_id=target_id,      # survivor
            parent_id=subject_id,    # absorbed
            derivation_type='merge',
            current_block_idx=current_block_idx,
            timestamp=block.get('block_start_time', 0.0)
        ))

    return lineage_edges


def execute_consume(
    graph: BlockGraph,
    txn: Dict[str, Any],
    block: Dict,
    warnings: List[str] = None,
    local_pool: Optional[LocalPool] = None,
    location_registry: Optional[LocationRegistry] = None
) -> List[LineageEdge]:
    """
    CONSUME: Mark food as consumed/discarded and remove from spatial graph.

    Creates a lineage edge with derivation_type='consume' to mark the terminal state.

    Consumed items are marked unavailable in local_pool.
    Food's location is cleared (V2 model).
    """
    raw_food_id = txn['food_id']
    consume_type = txn.get('consume_type', 'eaten')
    current_block_idx = block.get('block_idx', 0)

    # Resolve food_id (may be food_noun instead of instance_id)
    food_id = resolve_food_id(graph, raw_food_id, warnings, local_pool)
    if food_id is None:
        msg = f"CONSUME: unknown food_id '{raw_food_id}'"
        logger.warning(msg)
        if warnings is not None:
            warnings.append(msg)
        return []

    food = graph.food_nodes[food_id]
    if food.status != NodeStatus.ACTIVE:
        msg = f"CONSUME: inactive food_id '{food_id}'"
        logger.warning(msg)
        if warnings is not None:
            warnings.append(msg)
        return []

    lineage_edges = []

    food.status = NodeStatus.DISCARDED if consume_type == 'discarded' else NodeStatus.CONSUMED
    food.location = None  # V2: clear food's location

    # Remove from containment - V1 compat
    graph.containment_edges = [
        e for e in graph.containment_edges
        if e.food_instance_id != food_id
    ]

    # Mark unavailable in pool
    if local_pool is not None:
        local_pool.mark_unavailable(food_id)

    # Create lineage edge if food existed in a previous block (terminal state marker)
    if food.created_in_block < current_block_idx:
        lineage_edges.append(create_lineage_edge(
            child_id=food_id,
            parent_id=food_id,
            derivation_type='consume',
            current_block_idx=current_block_idx,
            timestamp=block.get('block_start_time', 0.0)
        ))

    return lineage_edges


def execute_update(
    graph: BlockGraph,
    txn: Dict[str, Any],
    block: Dict,
    warnings: List[str] = None,
    local_pool: Optional[LocalPool] = None,
    location_registry: Optional[LocationRegistry] = None
) -> List[LineageEdge]:
    """
    UPDATE: Modify food state or identity (Survivor Protocol).

    Supports:
    - form_state: Change physical form (whole -> liquid, etc.)
    - quantity: Change amount (full -> partial, etc.)
    - count: Change count
    - food_noun: Change IDENTITY (orange -> orange_juice)

    When food_noun changes, the instance is renamed to prevent VLM confusion:
    e.g., orange_001 with food_noun="orange juice" becomes orange_juice_001

    Creates a lineage edge:
    - "identity_transform" if food_noun changes (butter_slice -> melted_butter)
    - "update" if only state properties change (form, quantity, count)
    """
    raw_food_id = txn['food_id']
    state_changes = txn.get('state_changes', {})
    current_block_idx = block.get('block_idx', 0)

    # Resolve food_id (may be food_noun instead of instance_id)
    food_id = resolve_food_id(graph, raw_food_id, warnings, local_pool)
    if food_id is None:
        msg = f"UPDATE: unknown food_id '{raw_food_id}'"
        logger.warning(msg)
        if warnings is not None:
            warnings.append(msg)
        return []

    food = graph.food_nodes[food_id]
    if food.status != NodeStatus.ACTIVE:
        msg = f"UPDATE: inactive food_id '{food_id}'"
        logger.warning(msg)
        if warnings is not None:
            warnings.append(msg)
        return []

    lineage_edges = []
    original_food_id = food_id  # Track for edge creation
    is_identity_transform = False

    # Update state properties
    if 'form_state' in state_changes:
        food.state.form_state = state_changes['form_state']
    if 'quantity' in state_changes:
        food.state.quantity = state_changes['quantity']
    if 'count' in state_changes:
        food.state.count = state_changes['count']

    # Handle identity change (food_noun) - Survivor Protocol
    if 'food_noun' in state_changes:
        new_food_noun = state_changes['food_noun']
        if new_food_noun and new_food_noun != food.food_noun:
            is_identity_transform = True
            old_food_noun = food.food_noun
            old_id = food_id

            # Update food_noun
            food.food_noun = new_food_noun

            # Rename instance_id to prevent VLM confusion
            new_id = generate_instance_id(new_food_noun)
            food.instance_id = new_id

            # Move in graph dict
            del graph.food_nodes[old_id]
            graph.food_nodes[new_id] = food

            # Update containment edges (V1 compat)
            for edge in graph.containment_edges:
                if edge.food_instance_id == old_id:
                    edge.food_instance_id = new_id

            # Update local_pool: mark old unavailable, add new
            if local_pool is not None:
                local_pool.mark_unavailable(old_id)
                local_pool.add(new_food_noun, new_id)

            if warnings is not None:
                warnings.append(f"IDENTITY: '{old_id}' ({old_food_noun}) -> '{new_id}' ({new_food_noun})")

            # Update food_id for edge creation
            food_id = new_id

    # Create lineage edge if food existed in a previous block
    if food.created_in_block < current_block_idx:
        derivation_type = 'identity_transform' if is_identity_transform else 'update'
        lineage_edges.append(create_lineage_edge(
            child_id=food_id,
            parent_id=original_food_id,
            derivation_type=derivation_type,
            current_block_idx=current_block_idx,
            timestamp=block.get('block_start_time', 0.0)
        ))

    return lineage_edges


# ============================================================================
# Main Transaction Dispatcher
# ============================================================================

def apply_transaction(
    graph: BlockGraph,
    txn: Dict[str, Any],
    block: Dict,
    warnings: List[str] = None,
    local_pool: Optional[LocalPool] = None,
    location_registry: Optional[LocationRegistry] = None
) -> List[LineageEdge]:
    """
    Apply a single transaction to the graph.

    Args:
        graph: BlockGraph to modify
        txn: Transaction dict from VLM
        block: Block metadata
        warnings: Optional list to collect warning messages
        local_pool: Optional LocalPool for just-in-time resolution
        location_registry: Optional LocationRegistry for auto-numbering containers

    Returns any lineage edges created by the transaction.
    """
    txn_type = txn.get('type', '').lower()

    handlers = {
        'transfer': execute_transfer,
        'split': execute_split,
        'merge': execute_merge,
        'consume': execute_consume,
        'update': execute_update,
    }

    handler = handlers.get(txn_type)
    if handler is None:
        msg = f"Unknown transaction type: {txn_type}"
        logger.warning(msg)
        if warnings is not None:
            warnings.append(msg)
        return []

    return handler(graph, txn, block, warnings, local_pool, location_registry)


def apply_transactions_batch(
    graph: BlockGraph,
    transactions: List[Dict[str, Any]],
    block: Dict,
    location_registry: Optional[LocationRegistry] = None
) -> Tuple[List[LineageEdge], List[str]]:
    """
    Apply all transactions with Just-in-Time resolution.

    Creates a LocalPool to track items created during this block's processing.
    VLM can refer to newly created items by food_noun (e.g., "butter")
    and they will be resolved to the correct instance_id (e.g., "butter_005").

    Args:
        graph: BlockGraph to modify
        transactions: List of transaction dicts from VLM
        block: Block metadata
        location_registry: Optional LocationRegistry for auto-numbering containers (V2)

    Returns:
        Tuple of (lineage_edges, warnings)
    """
    lineage_edges = []
    warnings = []

    # Create local pool for Just-in-Time resolution
    local_pool = LocalPool()

    for txn in transactions:
        new_lineage = apply_transaction(graph, txn, block, warnings, local_pool, location_registry)
        lineage_edges.extend(new_lineage)

    return lineage_edges, warnings


def process_new_containers(
    graph: BlockGraph,
    new_containers: List[Dict[str, Any]],
    block: Dict
) -> None:
    """
    Process new container discoveries from VLM.
    """
    for container in new_containers:
        container_id = container.get('container_id')
        zone = container.get('zone', 'unknown')

        if container_id and container_id not in graph.container_nodes:
            graph.container_nodes[container_id] = ContainerNode(
                container_id=container_id,
                zone=zone,
                created_at=block.get('block_start_time', 0.0),
                created_in_video=block.get('video_id', ''),
                created_in_block=block.get('block_id', 0)
            )


def process_block(
    prev_graph: Optional[BlockGraph],
    block: Dict,
    arrivals: List[Dict],
    food_init_states: List[Dict],
    vlm_response: Dict[str, Any]
) -> Tuple[BlockGraph, List[LineageEdge]]:
    """
    Process a block to produce new spatial graph + lineage edges.

    Args:
        prev_graph: Previous block's graph (or None if first block)
        block: Block metadata with video_id, block_id, start/end times, narrations
        arrivals: List of inventory arrivals matched to this block
        food_init_states: VLM-determined initial states for arrivals
        vlm_response: VLM response with new_containers and transactions

    Returns:
        Tuple of (new BlockGraph, list of LineageEdges)
    """
    # Start with copy of previous graph's active state
    new_graph = copy_active_state(prev_graph)
    new_graph.video_id = block.get('video_id', '')
    new_graph.block_id = block.get('block_id', 0)
    new_graph.block_start_time = block.get('block_start_time', 0.0)
    new_graph.block_end_time = block.get('block_end_time', 0.0)

    lineage_edges = []

    # Create nodes from arrivals with VLM-determined initial states
    for arrival, init_state in zip(arrivals, food_init_states):
        semantic_name = arrival.get('semantic_name', '')
        if not semantic_name:
            continue

        node = FoodNode(
            instance_id=generate_instance_id(semantic_name),
            food_noun=semantic_name,
            status=NodeStatus.ACTIVE,
            state=FoodState(
                form_state=init_state.get('state', {}).get('form_state', 'whole'),
                quantity=init_state.get('state', {}).get('quantity', 'full'),
                count=init_state.get('state', {}).get('count')
            ),
            parent_instance=None,
            created_at=arrival.get('timestamp', block.get('block_start_time', 0.0)),
            created_in_video=block.get('video_id', ''),
            created_in_block=block.get('block_id', 0)
        )
        new_graph.food_nodes[node.instance_id] = node

        # Add containment edge if VLM specified initial container
        initial_container = init_state.get('initial_container')
        if initial_container:
            ensure_container_exists(new_graph, initial_container, block)
            new_graph.containment_edges.append(ContainmentEdge(
                food_instance_id=node.instance_id,
                container_id=initial_container
            ))

    # Process new containers from VLM
    new_containers = vlm_response.get('new_containers', [])
    process_new_containers(new_graph, new_containers, block)

    # Apply each transaction
    transactions = vlm_response.get('transactions', [])
    for txn in transactions:
        new_lineage = apply_transaction(new_graph, txn, block)
        lineage_edges.extend(new_lineage)

    return new_graph, lineage_edges

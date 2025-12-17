"""
Data Structures for Spatio-Temporal Food Graph

This module defines the core data structures:
- FoodNode: Represents a food item with state and location
- LocationRegistry: Tracks and auto-numbers containers (container-only model)
- BlockGraph: Spatial snapshot at end of a block
- LineageEdge: Temporal edge connecting nodes across blocks
- SpatioTemporalGraph: Complete graph structure

V3 Model (Container-Only):
- FoodNode.location is either a container ID (pan_001) or None (on surface)
- LocationRegistry uses blocklist + normalization + fallback strategy:
  * SURFACE_BLOCKLIST: counter, table, stove, etc. → return None
  * CONTAINER_SYNONYMS: skillet → pan, saucepan → pot, etc.
  * Fallback: Trust VLM, auto-number any other name as container
- Food in container (location != None): Subject to physics/mixing
- Food on surface (location == None): Stays distinct/rigid
- ContainerNode and ContainmentEdge are deprecated (kept for backward compat)
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Set, ClassVar
from enum import Enum
import json
import re
from datetime import datetime


class NodeStatus(str, Enum):
    """Status of a food node in the graph"""
    ACTIVE = "active"
    CONSUMED = "consumed"
    DISCARDED = "discarded"
    SPLIT_SOURCE = "split_source"


@dataclass
class FoodState:
    """
    2-dimensional state of a food item.

    Attributes:
        form_state: Physical form (whole, prepared_ingredient, cooking_in_progress, cooked_dish, leftover)
        quantity: Amount remaining (full, partial, nearly_empty)
        count: For countable items (eggs, slices, etc.), None otherwise
    """
    form_state: str = "unknown"
    quantity: str = "unknown"
    count: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "form_state": self.form_state,
            "quantity": self.quantity,
            "count": self.count
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FoodState':
        return cls(
            form_state=data.get("form_state", "unknown"),
            quantity=data.get("quantity", "unknown"),
            count=data.get("count")
        )


@dataclass
class FoodNode:
    """
    Represents a food item in the graph.

    Instance ID format: {food_noun}_{counter:03d} (NO state in name)
    Location: Where the food is (zone like "counter" or container like "pan_001")
    """
    instance_id: str
    food_noun: str
    status: NodeStatus
    state: FoodState
    location: Optional[str]  # Unified location: zone or container ID
    parent_instance: Optional[str]
    created_at: float
    created_in_video: str
    created_in_block: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "food_noun": self.food_noun,
            "status": self.status.value if isinstance(self.status, NodeStatus) else self.status,
            "state": self.state.to_dict(),
            "location": self.location,
            "parent_instance": self.parent_instance,
            "created_at": self.created_at,
            "created_in_video": self.created_in_video,
            "created_in_block": self.created_in_block
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FoodNode':
        return cls(
            instance_id=data["instance_id"],
            food_noun=data["food_noun"],
            status=NodeStatus(data["status"]) if isinstance(data["status"], str) else data["status"],
            state=FoodState.from_dict(data["state"]),
            location=data.get("location"),
            parent_instance=data.get("parent_instance"),
            created_at=data["created_at"],
            created_in_video=data["created_in_video"],
            created_in_block=data["created_in_block"]
        )

    def copy(self) -> 'FoodNode':
        """Create a deep copy of this node"""
        return FoodNode(
            instance_id=self.instance_id,
            food_noun=self.food_noun,
            status=self.status,
            state=FoodState(
                form_state=self.state.form_state,
                quantity=self.state.quantity,
                count=self.state.count
            ),
            location=self.location,
            parent_instance=self.parent_instance,
            created_at=self.created_at,
            created_in_video=self.created_in_video,
            created_in_block=self.created_in_block
        )


@dataclass
class LocationRegistry:
    """
    Tracks known containers and auto-numbers container instances.

    CONTAINER-ONLY MODEL (V3):
    - TRACKED: Containers with walls/volume (pan, bowl, cup) → auto-numbered (pan_001)
    - UNTRACKED: Flat surfaces (counter, table, stove) → return None

    Strategy: Blocklist + Normalize + Fallback (trust VLM)
    """

    # BLOCKLIST: Flat open surfaces → return None (food stays distinct)
    SURFACE_BLOCKLIST: ClassVar[Set[str]] = {
        "counter", "countertop", "table", "stove", "stovetop",
        "shelf", "fridge", "freezer", "pantry", "sink",
        "cutting_board", "board", "chopping_board",
        "oven", "microwave", "trash", "prep_surface",
        "storage", "serving", "hand", "unknown"
    }

    # SYNONYM NORMALIZATION: Map variations to canonical container names
    CONTAINER_SYNONYMS: ClassVar[Dict[str, str]] = {
        "skillet": "pan",
        "frying_pan": "pan",
        "saucepan": "pot",
        "cooking_pot": "pot",
        "mixing_bowl": "bowl",
        "salad_bowl": "bowl",
        "coffee_cup": "cup",
        "coffee_mug": "mug",
        "water_glass": "glass",
        "drinking_glass": "glass",
        "food_processor": "blender",
    }

    known_locations: Set[str] = field(default_factory=set)
    container_counters: Dict[str, int] = field(default_factory=dict)

    def get_or_create_location(self, location_name: str) -> Optional[str]:
        """
        Returns a valid container ID or None for surfaces.

        Logic:
        1. BLOCKLIST CHECK: Is it a surface? → return None
        2. ALREADY NUMBERED: pan_001 → return as-is
        3. NORMALIZE: Map synonyms to canonical names
        4. FALLBACK: Trust VLM, auto-number as container
        """
        if not location_name:
            return None

        # Normalize input
        normalized = location_name.lower().strip().replace(' ', '_')

        # 1. BLOCKLIST CHECK: Is it a surface?
        if normalized in self.SURFACE_BLOCKLIST:
            return None  # Untracked surface - food stays distinct

        # 2. Check if already a numbered ID (e.g., "pan_001")
        if re.match(r'.+_\d{3}$', normalized):
            self.known_locations.add(normalized)
            # Update counter to avoid collision
            base = re.sub(r'_\d{3}$', '', normalized)
            num = int(normalized[-3:])
            self.container_counters[base] = max(self.container_counters.get(base, 0), num)
            return normalized

        # 3. NORMALIZE: Map synonyms to canonical names
        canonical = self.CONTAINER_SYNONYMS.get(normalized, normalized)

        # 4. FALLBACK: Trust VLM - auto-number as container
        return self._get_next_container_id(canonical)

    def _get_next_container_id(self, container_type: str) -> str:
        """Create a new numbered container."""
        container_type = container_type.lower().strip().replace(' ', '_')
        if container_type not in self.container_counters:
            self.container_counters[container_type] = 0
        self.container_counters[container_type] += 1
        new_id = f"{container_type}_{self.container_counters[container_type]:03d}"
        self.known_locations.add(new_id)
        return new_id

    def is_surface(self, location_name: str) -> bool:
        """Check if a location name is a surface (would return None)."""
        if not location_name:
            return True
        normalized = location_name.lower().strip().replace(' ', '_')
        return normalized in self.SURFACE_BLOCKLIST

    def to_dict(self) -> Dict[str, Any]:
        return {
            "known_locations": list(self.known_locations),
            "container_counters": self.container_counters
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LocationRegistry':
        return cls(
            known_locations=set(data.get("known_locations", [])),
            container_counters=data.get("container_counters", {})
        )


# =============================================================================
# DEPRECATED: ContainerNode and ContainmentEdge
# Kept for backward compatibility with v1.0 graphs
# New code should use FoodNode.location instead
# =============================================================================

@dataclass
class ContainerNode:
    """
    Represents a container in the graph.

    Container IDs are VLM-assigned (e.g., "bowl_001", "frying_pan_001")
    """
    container_id: str
    zone: str
    created_at: float
    created_in_video: str
    created_in_block: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "container_id": self.container_id,
            "zone": self.zone,
            "created_at": self.created_at,
            "created_in_video": self.created_in_video,
            "created_in_block": self.created_in_block
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ContainerNode':
        return cls(
            container_id=data["container_id"],
            zone=data["zone"],
            created_at=data["created_at"],
            created_in_video=data["created_in_video"],
            created_in_block=data["created_in_block"]
        )

    def copy(self) -> 'ContainerNode':
        """Create a copy of this node"""
        return ContainerNode(
            container_id=self.container_id,
            zone=self.zone,
            created_at=self.created_at,
            created_in_video=self.created_in_video,
            created_in_block=self.created_in_block
        )


@dataclass
class ContainmentEdge:
    """
    Spatial relationship within a block: food IS_IN container.

    These edges are valid only for the block they appear in.
    """
    food_instance_id: str
    container_id: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "food_instance_id": self.food_instance_id,
            "container_id": self.container_id
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ContainmentEdge':
        return cls(
            food_instance_id=data["food_instance_id"],
            container_id=data["container_id"]
        )


@dataclass
class LineageEdge:
    """
    Temporal relationship BETWEEN blocks: food DERIVED_FROM parent.

    Links a food node in snapshot N to its corresponding node in snapshot N+1.

    derivation_type values:
    - "split": parent divided, child is new portion (parent -> new_child)
    - "merge": subject absorbed into target (subject -> target, subject dies)
    - "update": state changed in place (form, quantity, count) (self -> self)
    - "identity_transform": food_noun changed (butter_slice -> melted_butter)
    - "transfer": location changed (self -> self)
    - "consume": food eaten/discarded - terminal state (self -> self)
    """
    child_instance_id: str
    parent_instance_id: str
    derivation_type: str
    source_block: int
    target_block: int
    timestamp: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "child_instance_id": self.child_instance_id,
            "parent_instance_id": self.parent_instance_id,
            "derivation_type": self.derivation_type,
            "source_block": self.source_block,
            "target_block": self.target_block,
            "timestamp": self.timestamp
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LineageEdge':
        return cls(
            child_instance_id=data["child_instance_id"],
            parent_instance_id=data["parent_instance_id"],
            derivation_type=data["derivation_type"],
            source_block=data["source_block"],
            target_block=data["target_block"],
            timestamp=data["timestamp"]
        )


@dataclass
class BlockGraph:
    """
    Spatial snapshot after block events are applied.

    Each block produces one BlockGraph representing the state of all
    food items and containers at the end of that block.
    """
    video_id: str
    block_id: int
    block_start_time: float
    block_end_time: float
    food_nodes: Dict[str, FoodNode] = field(default_factory=dict)
    container_nodes: Dict[str, ContainerNode] = field(default_factory=dict)
    containment_edges: List[ContainmentEdge] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "video_id": self.video_id,
            "block_id": self.block_id,
            "block_start_time": self.block_start_time,
            "block_end_time": self.block_end_time,
            "food_nodes": {k: v.to_dict() for k, v in self.food_nodes.items()},
            "container_nodes": {k: v.to_dict() for k, v in self.container_nodes.items()},
            "containment_edges": [e.to_dict() for e in self.containment_edges]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BlockGraph':
        return cls(
            video_id=data["video_id"],
            block_id=data["block_id"],
            block_start_time=data["block_start_time"],
            block_end_time=data["block_end_time"],
            food_nodes={k: FoodNode.from_dict(v) for k, v in data.get("food_nodes", {}).items()},
            container_nodes={k: ContainerNode.from_dict(v) for k, v in data.get("container_nodes", {}).items()},
            containment_edges=[ContainmentEdge.from_dict(e) for e in data.get("containment_edges", [])]
        )

    def get_active_food_nodes(self) -> Dict[str, FoodNode]:
        """Return only active food nodes (not consumed/discarded/split_source)"""
        return {
            k: v for k, v in self.food_nodes.items()
            if v.status == NodeStatus.ACTIVE
        }

    def get_container_for_food(self, food_id: str) -> Optional[str]:
        """Get the container/location for a food item.

        Works with both v1 (containment_edges) and v2 (FoodNode.location) models.
        """
        # V2 model: check FoodNode.location directly
        if food_id in self.food_nodes and self.food_nodes[food_id].location:
            return self.food_nodes[food_id].location

        # V1 model fallback: check containment_edges
        for edge in self.containment_edges:
            if edge.food_instance_id == food_id:
                return edge.container_id
        return None

    def get_foods_in_container(self, container_id: str) -> List[str]:
        """Get all food IDs in a container/location.

        Works with both v1 (containment_edges) and v2 (FoodNode.location) models.
        """
        # V2 model: check FoodNode.location directly
        foods_v2 = [
            instance_id for instance_id, node in self.food_nodes.items()
            if node.status == NodeStatus.ACTIVE and node.location == container_id
        ]
        if foods_v2:
            return foods_v2

        # V1 model fallback: check containment_edges
        return [
            edge.food_instance_id
            for edge in self.containment_edges
            if edge.container_id == container_id
        ]

    def get_foods_at_location(self, location: str) -> List[str]:
        """Get all active food IDs at a specific location (V2 model)."""
        return [
            instance_id for instance_id, node in self.food_nodes.items()
            if node.status == NodeStatus.ACTIVE and node.location == location
        ]

    def get_all_locations(self) -> Dict[str, List[str]]:
        """Get a mapping of locations to food IDs at that location (V2 model)."""
        from collections import defaultdict
        location_map = defaultdict(list)
        for instance_id, node in self.food_nodes.items():
            if node.status == NodeStatus.ACTIVE and node.location:
                location_map[node.location].append(instance_id)
        return dict(location_map)


@dataclass
class SpatioTemporalGraph:
    """
    Complete graph across all processed blocks.

    Contains:
    - block_graphs: List of spatial snapshots ordered by (video_id, block_id)
    - lineage_edges: Cross-block temporal edges for splits and transformations
    """
    metadata: Dict[str, Any] = field(default_factory=dict)
    block_graphs: List[BlockGraph] = field(default_factory=list)
    lineage_edges: List[LineageEdge] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metadata": self.metadata,
            "block_graphs": [bg.to_dict() for bg in self.block_graphs],
            "lineage_edges": [le.to_dict() for le in self.lineage_edges]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SpatioTemporalGraph':
        return cls(
            metadata=data.get("metadata", {}),
            block_graphs=[BlockGraph.from_dict(bg) for bg in data.get("block_graphs", [])],
            lineage_edges=[LineageEdge.from_dict(le) for le in data.get("lineage_edges", [])]
        )

    def save(self, filepath: str) -> None:
        """Save graph to JSON file"""
        self.metadata["last_updated"] = datetime.now().isoformat()
        self.metadata["total_blocks"] = len(self.block_graphs)

        # Count unique food and container nodes
        all_food_ids = set()
        all_container_ids = set()
        for bg in self.block_graphs:
            all_food_ids.update(bg.food_nodes.keys())
            all_container_ids.update(bg.container_nodes.keys())

        self.metadata["total_food_nodes"] = len(all_food_ids)
        self.metadata["total_container_nodes"] = len(all_container_ids)
        self.metadata["total_lineage_edges"] = len(self.lineage_edges)

        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> 'SpatioTemporalGraph':
        """Load graph from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)

    def get_last_block_graph(self) -> Optional[BlockGraph]:
        """Get the most recent block graph"""
        if self.block_graphs:
            return self.block_graphs[-1]
        return None

    def get_videos_processed(self) -> List[str]:
        """Get list of video IDs that have been processed"""
        videos = []
        for bg in self.block_graphs:
            if bg.video_id not in videos:
                videos.append(bg.video_id)
        return videos

    def get_lineage_for_node(self, instance_id: str) -> Dict[str, Any]:
        """Get the complete lineage (ancestors and descendants) for a food node"""
        ancestors = []
        descendants = []

        # Find ancestors (parents)
        for edge in self.lineage_edges:
            if edge.child_instance_id == instance_id:
                ancestors.append({
                    "parent_id": edge.parent_instance_id,
                    "type": edge.derivation_type,
                    "timestamp": edge.timestamp
                })

        # Find descendants (children)
        for edge in self.lineage_edges:
            if edge.parent_instance_id == instance_id:
                descendants.append({
                    "child_id": edge.child_instance_id,
                    "type": edge.derivation_type,
                    "timestamp": edge.timestamp
                })

        return {
            "instance_id": instance_id,
            "ancestors": ancestors,
            "descendants": descendants
        }

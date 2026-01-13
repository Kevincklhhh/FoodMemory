"""
Food Graph Module - Spatio-Temporal Food Tracking

This module provides data structures and operations for building a spatio-temporal
graph that tracks food items through cooking processes in egocentric videos.

Key concepts:
- FoodNode: Food item with state and location (V2 model)
- LocationRegistry: Auto-numbers containers (pan -> pan_001, bowl -> bowl_002)
- BlockGraph: Spatial snapshot of food at end of each block
- LineageEdge: Temporal edges connecting nodes across blocks (splits, merges)
- SpatioTemporalGraph: Complete graph with all block snapshots + lineage edges
"""

from .data_structures import (
    FoodState,
    FoodNode,
    ContainerNode,  # Deprecated, kept for backward compat
    ContainmentEdge,  # Deprecated, kept for backward compat
    LineageEdge,
    BlockGraph,
    SpatioTemporalGraph,
    NodeStatus,
    LocationRegistry,  # V2: for auto-numbering containers
)

from .graph_operations import (
    process_block,
    snapshot_active_state,
    generate_instance_id,
    apply_transaction,
    apply_transactions_batch,
    reset_instance_counters,
)

from .vlm_prompts import (
    build_food_init_prompt,
    build_transaction_prompt,
    build_transaction_prompt_from_descriptions,
    build_known_locations_table,  # V2: replaces build_containers_table
    parse_vlm_response,
    parse_food_init_response,
    load_state_descriptions,
    FOOD_INIT_SYSTEM_PROMPT,
    TRANSACTION_FROM_DESCRIPTIONS_PROMPT,
    # Gemini pre-annotation pipeline
    GEMINI_TRANSACTION_SYSTEM_PROMPT,
    build_gemini_transaction_prompt,
)

from .gemini_utils import (
    NarrationLookup,
    VideoClipExtractor,
)

__all__ = [
    # Data structures
    'FoodState',
    'FoodNode',
    'ContainerNode',  # Deprecated
    'ContainmentEdge',  # Deprecated
    'LineageEdge',
    'BlockGraph',
    'SpatioTemporalGraph',
    'NodeStatus',
    'LocationRegistry',  # V2
    # Operations
    'process_block',
    'snapshot_active_state',
    'generate_instance_id',
    'apply_transaction',
    'apply_transactions_batch',
    'reset_instance_counters',
    # VLM prompts
    'build_food_init_prompt',
    'build_transaction_prompt',
    'build_transaction_prompt_from_descriptions',
    'build_known_locations_table',  # V2
    'parse_vlm_response',
    'parse_food_init_response',
    'load_state_descriptions',
    'FOOD_INIT_SYSTEM_PROMPT',
    'TRANSACTION_FROM_DESCRIPTIONS_PROMPT',
    # Gemini pre-annotation pipeline
    'GEMINI_TRANSACTION_SYSTEM_PROMPT',
    'build_gemini_transaction_prompt',
    'NarrationLookup',
    'VideoClipExtractor',
]

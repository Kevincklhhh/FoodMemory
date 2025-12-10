#!/usr/bin/env python3
"""
Test script for Just-in-Time (JIT) Resolution in Food Graph

This tests the LocalPool mechanism that allows VLM to refer to
newly created items by their food_noun instead of system-assigned IDs.

Example scenario:
1. SPLIT: orange_001 -> 2x "orange half"  (creates orange_half_001, orange_half_002)
2. TRANSFER: "orange half" to juicer      (should resolve to orange_half_001)
3. CONSUME: "orange half" discarded       (should resolve to orange_half_002)

Without JIT resolution, steps 2 and 3 would fail because VLM doesn't know
the system-assigned IDs (orange_half_001, orange_half_002).
"""

import json
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from food_graph.data_structures import (
    BlockGraph, FoodNode, ContainerNode, FoodState,
    ContainmentEdge, NodeStatus
)
from food_graph.graph_operations import (
    apply_transactions_batch, reset_instance_counters
)


def create_initial_graph() -> BlockGraph:
    """Create initial graph with a single orange."""
    graph = BlockGraph(
        video_id="test_video",
        block_id=0,
        block_start_time=0.0,
        block_end_time=30.0,
        food_nodes={},
        container_nodes={},
        containment_edges=[]
    )

    # Add orange_001
    graph.food_nodes["orange_001"] = FoodNode(
        instance_id="orange_001",
        food_noun="orange",
        status=NodeStatus.ACTIVE,
        state=FoodState(form_state="whole", quantity="full", count=1),
        parent_instance=None,
        created_at=0.0,
        created_in_video="test_video",
        created_in_block=0
    )

    # Add counter container
    graph.container_nodes["counter"] = ContainerNode(
        container_id="counter",
        zone="prep_surface",
        created_at=0.0,
        created_in_video="test_video",
        created_in_block=0
    )

    # Orange is on counter
    graph.containment_edges.append(ContainmentEdge(
        food_instance_id="orange_001",
        container_id="counter"
    ))

    return graph


def test_jit_resolution_basic():
    """Test basic JIT resolution with split -> transfer -> consume."""
    print("=" * 60)
    print("TEST: Basic JIT Resolution")
    print("=" * 60)

    reset_instance_counters()
    # Set initial counter so orange_half starts at 001
    from food_graph import graph_operations
    graph_operations._instance_counters['orange_half'] = 0

    graph = create_initial_graph()

    # Transactions that VLM might output
    # Note: VLM refers to "orange half" by food_noun, not by instance_id
    transactions = [
        # 1. Split orange into two halves
        {
            "type": "split",
            "subtype": "complete",
            "parent_id": "orange_001",
            "children": [
                {"food_noun": "orange half", "form_state": "prepared_ingredient", "quantity": "full", "count": 2}
            ]
        },
        # 2. Transfer one "orange half" to juicer (VLM doesn't know the ID)
        {
            "type": "transfer",
            "food_id": "orange half",  # NOT "orange_half_001" - VLM uses food_noun
            "to_container": "juicer_001",
            "to_zone": "counter"
        },
        # 3. Consume the other "orange half"
        {
            "type": "consume",
            "food_id": "orange half",  # Second one should be resolved
            "consume_type": "discarded"
        }
    ]

    block = {
        "video_id": "test_video",
        "block_id": 1,
        "block_idx": 1,
        "block_start_time": 30.0,
        "block_end_time": 60.0
    }

    print("\n[Initial State]")
    print(f"  Food nodes: {list(graph.food_nodes.keys())}")
    print(f"  Active: {[k for k, v in graph.food_nodes.items() if v.status == NodeStatus.ACTIVE]}")

    print("\n[Transactions]")
    for i, txn in enumerate(transactions, 1):
        print(f"  {i}. {txn['type'].upper()}: {json.dumps(txn, indent=6)[:100]}...")

    # Apply transactions
    lineage_edges, warnings = apply_transactions_batch(graph, transactions, block)

    print("\n[Results]")
    print(f"  Food nodes: {list(graph.food_nodes.keys())}")
    active = [k for k, v in graph.food_nodes.items() if v.status == NodeStatus.ACTIVE]
    consumed = [k for k, v in graph.food_nodes.items() if v.status == NodeStatus.CONSUMED]
    discarded = [k for k, v in graph.food_nodes.items() if v.status == NodeStatus.DISCARDED]
    split_source = [k for k, v in graph.food_nodes.items() if v.status == NodeStatus.SPLIT_SOURCE]

    print(f"  Active: {active}")
    print(f"  Consumed: {consumed}")
    print(f"  Discarded: {discarded}")
    print(f"  Split Source: {split_source}")

    print("\n[Warnings]")
    for w in warnings:
        print(f"  - {w}")

    # Verify results
    print("\n[Verification]")
    success = True

    # Check that orange_half_001 exists and is in juicer
    if "orange_half_001" in graph.food_nodes:
        node = graph.food_nodes["orange_half_001"]
        container = None
        for edge in graph.containment_edges:
            if edge.food_instance_id == "orange_half_001":
                container = edge.container_id
                break

        if node.status == NodeStatus.ACTIVE and container == "juicer_001":
            print("  ✓ orange_half_001 is ACTIVE in juicer_001")
        else:
            print(f"  ✗ orange_half_001 status={node.status}, container={container}")
            success = False
    else:
        print("  ✗ orange_half_001 not found")
        success = False

    # Check that orange_half_002 is discarded
    if "orange_half_002" in graph.food_nodes:
        node = graph.food_nodes["orange_half_002"]
        if node.status == NodeStatus.DISCARDED:
            print("  ✓ orange_half_002 is DISCARDED")
        else:
            print(f"  ✗ orange_half_002 status={node.status} (expected DISCARDED)")
            success = False
    else:
        print("  ✗ orange_half_002 not found")
        success = False

    # Check that orange_001 is SPLIT_SOURCE
    if "orange_001" in graph.food_nodes:
        node = graph.food_nodes["orange_001"]
        if node.status == NodeStatus.SPLIT_SOURCE:
            print("  ✓ orange_001 is SPLIT_SOURCE")
        else:
            print(f"  ✗ orange_001 status={node.status} (expected SPLIT_SOURCE)")
            success = False

    # Check resolution warnings are present
    pool_resolutions = [w for w in warnings if "RESOLVED (pool)" in w]
    if len(pool_resolutions) >= 2:
        print(f"  ✓ Pool resolutions found: {len(pool_resolutions)}")
    else:
        print(f"  ✗ Expected 2+ pool resolutions, found {len(pool_resolutions)}")
        success = False

    return success


def test_block_011_scenario():
    """
    Test with a simplified version of block_011 from VLM logs.

    Scenario:
    - butter_003 and butter_004 exist on counter
    - SPLIT butter_003 -> creates new butter child
    - SPLIT butter_004 -> creates new butter child
    - MERGE the new butter children into butter_002
    """
    print("\n" + "=" * 60)
    print("TEST: Block 011 Simplified Scenario")
    print("=" * 60)

    reset_instance_counters()
    from food_graph import graph_operations
    graph_operations._instance_counters['butter'] = 4  # Start at 005

    # Create initial graph matching block_011 active items
    graph = BlockGraph(
        video_id="P01-20240202-161948",
        block_id=11,
        block_start_time=330.0,
        block_end_time=360.0,
        food_nodes={},
        container_nodes={},
        containment_edges=[]
    )

    # Add butter instances
    for i, (instance_id, form, qty, container) in enumerate([
        ("butter_002", "cooking_in_progress", "partial", "pan_001"),
        ("butter_003", "prepared_ingredient", "partial", "counter"),
        ("butter_004", "prepared_ingredient", "partial", "counter"),
    ]):
        graph.food_nodes[instance_id] = FoodNode(
            instance_id=instance_id,
            food_noun="butter",
            status=NodeStatus.ACTIVE,
            state=FoodState(form_state=form, quantity=qty, count=1),
            parent_instance=None,
            created_at=0.0,
            created_in_video="P01-20240202-161948",
            created_in_block=0  # Created in previous block
        )

    # Add containers
    graph.container_nodes["pan_001"] = ContainerNode(
        container_id="pan_001", zone="stove",
        created_at=0.0, created_in_video="P01-20240202-161948", created_in_block=0
    )
    graph.container_nodes["counter"] = ContainerNode(
        container_id="counter", zone="counter",
        created_at=0.0, created_in_video="P01-20240202-161948", created_in_block=0
    )

    # Add containment edges
    graph.containment_edges = [
        ContainmentEdge(food_instance_id="butter_002", container_id="pan_001"),
        ContainmentEdge(food_instance_id="butter_003", container_id="counter"),
        ContainmentEdge(food_instance_id="butter_004", container_id="counter"),
    ]

    # Simplified transactions: split and merge
    transactions = [
        # Split butter_003 -> new butter portion
        {
            "type": "split",
            "subtype": "partial",
            "parent_id": "butter_003",
            "children": [{"food_noun": "butter", "form_state": "cooking_in_progress", "quantity": "partial", "count": 1}]
        },
        # Split butter_004 -> new butter portion
        {
            "type": "split",
            "subtype": "partial",
            "parent_id": "butter_004",
            "children": [{"food_noun": "butter", "form_state": "cooking_in_progress", "quantity": "partial", "count": 1}]
        },
        # Transfer newly created butter to pan (VLM uses food_noun "butter")
        {
            "type": "transfer",
            "food_id": "butter",  # Should resolve to butter_005 (first created)
            "to_container": "pan_001",
            "to_zone": "stove"
        },
        # Transfer second newly created butter to pan
        {
            "type": "transfer",
            "food_id": "butter",  # Should resolve to butter_006 (second created)
            "to_container": "pan_001",
            "to_zone": "stove"
        },
        # Merge the new butters into butter_002
        {
            "type": "merge",
            "subtype": "accumulation",
            "subject_id": "butter",  # Should resolve to butter_005
            "target_id": "butter_002"
        },
        {
            "type": "merge",
            "subtype": "accumulation",
            "subject_id": "butter",  # Should resolve to butter_006
            "target_id": "butter_002"
        }
    ]

    block = {
        "video_id": "P01-20240202-161948",
        "block_id": 11,
        "block_idx": 20,  # Assume global block index
        "block_start_time": 330.0,
        "block_end_time": 360.0
    }

    print("\n[Initial State]")
    print(f"  Food nodes: {list(graph.food_nodes.keys())}")

    print("\n[Transactions]")
    for i, txn in enumerate(transactions, 1):
        print(f"  {i}. {txn['type'].upper()}")

    # Apply transactions
    lineage_edges, warnings = apply_transactions_batch(graph, transactions, block)

    print("\n[Results]")
    print(f"  Food nodes: {list(graph.food_nodes.keys())}")
    active = [(k, v.food_noun) for k, v in graph.food_nodes.items() if v.status == NodeStatus.ACTIVE]
    consumed = [(k, v.food_noun) for k, v in graph.food_nodes.items() if v.status == NodeStatus.CONSUMED]

    print(f"  Active: {active}")
    print(f"  Consumed: {consumed}")

    print("\n[Warnings]")
    for w in warnings:
        print(f"  - {w}")

    # Count pool resolutions
    pool_resolutions = [w for w in warnings if "RESOLVED (pool)" in w]
    unknown_errors = [w for w in warnings if "unknown" in w.lower()]
    inactive_errors = [w for w in warnings if "inactive" in w.lower()]

    print("\n[Summary]")
    print(f"  Pool resolutions: {len(pool_resolutions)}")
    print(f"  Unknown errors: {len(unknown_errors)}")
    print(f"  Inactive errors: {len(inactive_errors)}")

    success = len(unknown_errors) == 0
    if success:
        print("  ✓ No unknown food_id errors!")
    else:
        print("  ✗ Some food_ids could not be resolved")

    return success


def test_block_011_actual():
    """
    Test with ACTUAL block_011 transactions from VLM logs.

    This uses the exact transactions from:
    HDEPIC/outputs/food_graph/vlm_logs/P01-20240202-161948/block_011_transactions.json
    """
    print("\n" + "=" * 60)
    print("TEST: Actual Block 011 from VLM Logs")
    print("=" * 60)

    # Load actual transactions from VLM log
    vlm_log_path = Path(__file__).parent.parent / "outputs" / "food_graph" / "vlm_logs" / \
                   "P01-20240202-161948" / "block_011_transactions.json"

    if not vlm_log_path.exists():
        # Try alternate path
        vlm_log_path = Path(__file__).parent.parent.parent / "outputs" / "food_graph" / "vlm_logs" / \
                       "P01-20240202-161948" / "block_011_transactions.json"

    if not vlm_log_path.exists():
        print(f"  ✗ VLM log file not found: {vlm_log_path}")
        return False

    with open(vlm_log_path) as f:
        vlm_log = json.load(f)

    transactions = vlm_log["parsed_response"]["transactions"]
    original_warnings = vlm_log.get("execution_warnings", [])

    print(f"\n[VLM Log] {vlm_log_path.name}")
    print(f"  Transactions: {len(transactions)}")
    print(f"  Original warnings: {len(original_warnings)}")

    # Show original warnings
    print("\n[Original Execution Warnings]")
    for w in original_warnings:
        print(f"  - {w}")

    # Reset counters and set up initial state matching block_011
    reset_instance_counters()
    from food_graph import graph_operations
    graph_operations._instance_counters['butter'] = 4  # Next will be butter_005
    graph_operations._instance_counters['pasta_sauce'] = 0

    # Create initial graph matching ACTIVE FOOD ITEMS table from block_011
    graph = BlockGraph(
        video_id="P01-20240202-161948",
        block_id=11,
        block_start_time=330.0,
        block_end_time=360.0,
        food_nodes={},
        container_nodes={},
        containment_edges=[]
    )

    # Add food nodes matching the ACTIVE FOOD ITEMS table
    active_items = [
        ("black_pepper_001", "black pepper", "cooking_in_progress", "full", "pan_001"),
        ("butter_001", "butter", "whole", "nearly_empty", "pan_001"),
        ("butter_002", "butter", "cooking_in_progress", "partial", "pan_001"),
        ("butter_003", "butter", "prepared_ingredient", "partial", "counter"),
        ("butter_004", "butter", "prepared_ingredient", "partial", "counter"),
        ("pasta_001", "pasta", "cooking_in_progress", "full", "pan_001"),
        ("salt_001", "salt", "cooking_in_progress", "full", "pan_001"),
    ]

    for instance_id, food_noun, form_state, quantity, container in active_items:
        graph.food_nodes[instance_id] = FoodNode(
            instance_id=instance_id,
            food_noun=food_noun,
            status=NodeStatus.ACTIVE,
            state=FoodState(form_state=form_state, quantity=quantity, count=1),
            parent_instance=None,
            created_at=0.0,
            created_in_video="P01-20240202-161948",
            created_in_block=10  # Created in previous block
        )

    # Add containers
    containers = [
        ("counter", "counter"),
        ("pan_001", "stove"),
        ("pasta_box_001", "counter"),
        ("plate_001", "counter"),
        ("plate_002", "counter"),
    ]
    for container_id, zone in containers:
        graph.container_nodes[container_id] = ContainerNode(
            container_id=container_id,
            zone=zone,
            created_at=0.0,
            created_in_video="P01-20240202-161948",
            created_in_block=0
        )

    # Add containment edges
    for instance_id, _, _, _, container in active_items:
        graph.containment_edges.append(ContainmentEdge(
            food_instance_id=instance_id,
            container_id=container
        ))

    block = {
        "video_id": "P01-20240202-161948",
        "block_id": 11,
        "block_idx": 20,  # Assume global block index
        "block_start_time": 330.0,
        "block_end_time": 360.0
    }

    print("\n[Initial State]")
    print(f"  Food nodes: {list(graph.food_nodes.keys())}")

    # Apply transactions with JIT resolution
    _, new_warnings = apply_transactions_batch(graph, transactions, block)

    print("\n[New Execution Warnings with JIT Resolution]")
    for w in new_warnings:
        print(f"  - {w}")

    # Count different types of warnings
    original_inactive = sum(1 for w in original_warnings if "inactive" in w.lower())
    original_unknown = sum(1 for w in original_warnings if "unknown" in w.lower())

    new_inactive = sum(1 for w in new_warnings if "inactive" in w.lower())
    new_unknown = sum(1 for w in new_warnings if "unknown" in w.lower())
    new_resolved = sum(1 for w in new_warnings if "RESOLVED" in w)

    print("\n[Comparison]")
    print(f"  Original: {original_inactive} inactive, {original_unknown} unknown errors")
    print(f"  With JIT: {new_inactive} inactive, {new_unknown} unknown errors, {new_resolved} resolutions")

    # Show final state
    print("\n[Final State]")
    active = [(k, v.food_noun) for k, v in graph.food_nodes.items() if v.status == NodeStatus.ACTIVE]
    consumed = [(k, v.food_noun) for k, v in graph.food_nodes.items() if v.status == NodeStatus.CONSUMED]
    print(f"  Active: {active}")
    print(f"  Consumed: {[c[0] for c in consumed]}")
    print(f"  Total nodes: {len(graph.food_nodes)}")

    # The test "passes" if we have fewer or equal unknown errors
    # (Some inactive errors are expected due to VLM's redundant transactions)
    success = new_unknown <= original_unknown
    if success:
        print(f"\n  ✓ JIT resolution working (unknown errors: {original_unknown} -> {new_unknown})")
    else:
        print(f"\n  ✗ More unknown errors than before")

    return success


def test_block_011_with_food_noun_refs():
    """
    Test block_011 scenario with food_noun references (new prompt style).

    This simulates what VLM would output with the updated Rule 5:
    - SPLIT creates new butter portions
    - Subsequent operations reference them by "butter" instead of "butter_005"
    """
    print("\n" + "=" * 60)
    print("TEST: Block 011 with food_noun References (New Prompt)")
    print("=" * 60)

    reset_instance_counters()
    from food_graph import graph_operations
    graph_operations._instance_counters['butter'] = 4  # Next will be butter_005
    graph_operations._instance_counters['pasta_sauce'] = 0

    # Create initial graph matching block_011
    graph = BlockGraph(
        video_id="P01-20240202-161948",
        block_id=11,
        block_start_time=330.0,
        block_end_time=360.0,
        food_nodes={},
        container_nodes={},
        containment_edges=[]
    )

    # Add food nodes
    active_items = [
        ("black_pepper_001", "black pepper", "cooking_in_progress", "full", "pan_001"),
        ("butter_001", "butter", "whole", "nearly_empty", "pan_001"),
        ("butter_002", "butter", "cooking_in_progress", "partial", "pan_001"),
        ("butter_003", "butter", "prepared_ingredient", "partial", "counter"),
        ("butter_004", "butter", "prepared_ingredient", "partial", "counter"),
        ("pasta_001", "pasta", "cooking_in_progress", "full", "pan_001"),
        ("salt_001", "salt", "cooking_in_progress", "full", "pan_001"),
    ]

    for instance_id, food_noun, form_state, quantity, container in active_items:
        graph.food_nodes[instance_id] = FoodNode(
            instance_id=instance_id,
            food_noun=food_noun,
            status=NodeStatus.ACTIVE,
            state=FoodState(form_state=form_state, quantity=quantity, count=1),
            parent_instance=None,
            created_at=0.0,
            created_in_video="P01-20240202-161948",
            created_in_block=10
        )

    # Add containers
    for container_id, zone in [("counter", "counter"), ("pan_001", "stove")]:
        graph.container_nodes[container_id] = ContainerNode(
            container_id=container_id, zone=zone,
            created_at=0.0, created_in_video="P01-20240202-161948", created_in_block=0
        )

    # Add containment edges
    for instance_id, _, _, _, container in active_items:
        graph.containment_edges.append(ContainmentEdge(
            food_instance_id=instance_id, container_id=container
        ))

    # Transactions using food_noun references for newly created items
    # This is what VLM would output with the updated prompt
    transactions = [
        # Split butter_003 -> creates butter_005
        {
            "type": "split",
            "subtype": "partial",
            "parent_id": "butter_003",
            "children": [{"food_noun": "butter", "form_state": "cooking_in_progress", "quantity": "partial", "count": 1}]
        },
        # Split butter_004 -> creates butter_006
        {
            "type": "split",
            "subtype": "partial",
            "parent_id": "butter_004",
            "children": [{"food_noun": "butter", "form_state": "cooking_in_progress", "quantity": "partial", "count": 1}]
        },
        # Transfer newly created butter to pan (uses food_noun "butter")
        {"type": "transfer", "food_id": "butter", "to_container": "pan_001", "to_zone": "stove"},
        # Transfer second newly created butter to pan
        {"type": "transfer", "food_id": "butter", "to_container": "pan_001", "to_zone": "stove"},
        # Merge first new butter into butter_002
        {"type": "merge", "subtype": "incorporation", "subject_id": "butter", "target_id": "butter_002"},
        # Merge second new butter into butter_002
        {"type": "merge", "subtype": "incorporation", "subject_id": "butter", "target_id": "butter_002"},
        # Create pasta_sauce from butter_002 and pasta_001
        {"type": "merge", "subtype": "transformation", "subject_id": "butter_002", "target_id": "pasta_001", "new_food_noun": "pasta_sauce"},
    ]

    block = {
        "video_id": "P01-20240202-161948",
        "block_id": 11,
        "block_idx": 20,
        "block_start_time": 330.0,
        "block_end_time": 360.0
    }

    print("\n[Initial State]")
    print(f"  Food nodes: {list(graph.food_nodes.keys())}")

    print("\n[Transactions with food_noun refs]")
    for i, txn in enumerate(transactions, 1):
        if txn['type'] == 'split':
            print(f"  {i}. SPLIT: {txn['parent_id']} -> butter")
        elif txn['type'] == 'transfer':
            print(f"  {i}. TRANSFER: '{txn['food_id']}' -> {txn['to_container']}")
        elif txn['type'] == 'merge':
            print(f"  {i}. MERGE: '{txn['subject_id']}' + '{txn['target_id']}' ({txn['subtype']})")

    # Apply transactions
    _, warnings = apply_transactions_batch(graph, transactions, block)

    print("\n[Execution Warnings]")
    for w in warnings:
        print(f"  - {w}")

    # Count resolution types
    pool_resolutions = [w for w in warnings if "RESOLVED (pool)" in w]
    graph_resolutions = [w for w in warnings if "RESOLVED:" in w and "(pool)" not in w]
    unknown_errors = [w for w in warnings if "unknown" in w.lower()]
    inactive_errors = [w for w in warnings if "inactive" in w.lower()]

    print("\n[Resolution Summary]")
    print(f"  Pool resolutions (JIT): {len(pool_resolutions)}")
    print(f"  Graph resolutions: {len(graph_resolutions)}")
    print(f"  Unknown errors: {len(unknown_errors)}")
    print(f"  Inactive errors: {len(inactive_errors)}")

    # Show final state
    print("\n[Final State]")
    active = [(k, v.food_noun) for k, v in graph.food_nodes.items() if v.status == NodeStatus.ACTIVE]
    consumed = [(k, v.food_noun) for k, v in graph.food_nodes.items() if v.status == NodeStatus.CONSUMED]
    print(f"  Active: {active}")
    print(f"  Consumed: {[c[0] for c in consumed]}")
    print(f"  Total nodes: {len(graph.food_nodes)}")

    # Verify JIT resolution worked
    # Success = no unknown errors, all food_noun refs resolved (pool + graph)
    total_resolutions = len(pool_resolutions) + len(graph_resolutions)
    success = len(unknown_errors) == 0 and total_resolutions >= 4
    if success:
        print(f"\n  ✓ JIT resolution successfully resolved all food_noun references!")
        print(f"    ({len(pool_resolutions)} via pool, {len(graph_resolutions)} via graph)")
    else:
        print(f"\n  ✗ Expected 4+ resolutions with 0 unknown errors")
        print(f"    Got: {total_resolutions} resolutions, {len(unknown_errors)} unknown")

    return success


def main():
    print("\n" + "#" * 60)
    print("# Just-in-Time Resolution Test Suite")
    print("#" * 60)

    results = []

    results.append(("Basic JIT Resolution", test_jit_resolution_basic()))
    results.append(("Block 011 Simplified", test_block_011_scenario()))
    results.append(("Block 011 Actual VLM", test_block_011_actual()))
    results.append(("Block 011 food_noun Refs", test_block_011_with_food_noun_refs()))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")
        if not passed:
            all_passed = False

    print()
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

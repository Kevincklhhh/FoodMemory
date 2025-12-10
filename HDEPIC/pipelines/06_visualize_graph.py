#!/usr/bin/env python3
"""
Spatio-Temporal Food Graph Visualization using NetworkX

Creates a visual representation showing:
- Each block as a separate spatial graph (column)
- Food nodes positioned at their creation block, grouped by location
- Location nodes (V2 model) or Container nodes (V1 fallback) local to each block
- Lineage edges connecting food nodes across blocks

V2 Model: Uses food_node.location field directly
V1 Fallback: Uses container_nodes + containment_edges

Usage:
    python 06_visualize_graph.py
    python 06_visualize_graph.py --output graph.png
"""

import json
import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Default paths (relative to this script's location in pipelines/)
_SCRIPT_DIR = Path(__file__).parent
_PROJECT_ROOT = _SCRIPT_DIR.parent


def load_graph(graph_file: Path) -> Dict:
    """Load spatio-temporal graph from JSON"""
    with open(graph_file, 'r') as f:
        return json.load(f)


def get_food_location(node: Dict, bg: Dict) -> str:
    """
    Get the location of a food node (V2 model first, V1 fallback).

    V2: node['location'] field directly
    V1: Look up in containment_edges
    """
    # V2 model: location field on node
    if node.get('location'):
        return node['location']

    # V1 fallback: find in containment_edges
    food_id = node.get('instance_id', '')
    for edge in bg.get('containment_edges', []):
        if edge.get('food_instance_id') == food_id:
            return edge.get('container_id', '')

    return None


def create_spatio_temporal_graph(graph_data: Dict) -> Tuple[nx.DiGraph, Dict, Dict, List[Dict]]:
    """
    Create a NetworkX graph representing the spatio-temporal structure.

    Key principle: Each BlockGraph is a complete spatial snapshot.
    - Food nodes: Block-local IDs (food_id@B{block_idx}) - shown at EVERY block where present
    - Location nodes: Block-local IDs (location@B{block_idx}) - V2 model (from food.location)
    - Location edges: Within a single block (food@block → location@block)
    - Lineage edges: Connect food nodes across blocks (parent@source → child@target)

    Returns:
        G: NetworkX DiGraph
        node_positions: Dict of node -> (x, y) positions
        food_colors: Dict of food_noun -> color
        block_info: List of dicts with video_id and block_id per block_idx
    """
    G = nx.DiGraph()

    block_graphs = graph_data.get('block_graphs', [])
    lineage_edges = graph_data.get('lineage_edges', [])

    # Build mapping from (video_id, block_id) -> block_idx
    block_id_to_idx = {}
    block_info = []
    for block_idx, bg in enumerate(block_graphs):
        video_id = bg.get('video_id', '')
        block_id = bg.get('block_id', 0)
        block_id_to_idx[(video_id, block_id)] = block_idx
        block_info.append({'video_id': video_id, 'block_id': block_id})

    # Color palette for food types
    food_colors = {}
    color_palette = [
        '#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6',
        '#1abc9c', '#e91e63', '#00bcd4', '#ff9800', '#795548'
    ]
    color_idx = 0

    # First pass: collect all food types for consistent coloring
    for block_idx, bg in enumerate(block_graphs):
        for food_id, node in bg.get('food_nodes', {}).items():
            food_noun = node.get('food_noun', 'unknown')
            if food_noun not in food_colors:
                food_colors[food_noun] = color_palette[color_idx % len(color_palette)]
                color_idx += 1

    # Calculate positions
    node_positions = {}
    block_width = 4  # Horizontal spacing between blocks

    # Track y offsets per block for food
    y_food_offset = defaultdict(int)

    # Second pass: Add block-local food nodes
    # Also collect locations per block (V2 model)
    block_locations = defaultdict(set)  # block_idx -> set of location names

    for block_idx, bg in enumerate(block_graphs):
        # Add food nodes for THIS block (block-local IDs)
        for food_id, node in bg.get('food_nodes', {}).items():
            local_food_id = f"{food_id}@B{block_idx}"
            food_noun = node.get('food_noun', 'unknown')

            # V2: Get location from node
            location = get_food_location(node, bg)
            if location:
                block_locations[block_idx].add(location)

            x = block_idx * block_width
            y = y_food_offset[block_idx]
            y_food_offset[block_idx] += 1

            node_positions[local_food_id] = (x, y)
            G.add_node(local_food_id,
                       node_type='food',
                       block=block_idx,
                       original_id=food_id,
                       food_noun=food_noun,
                       location=location,  # V2: store location on node
                       status=node.get('status', 'unknown'),
                       state=node.get('state', {}),
                       color=food_colors.get(food_noun, '#999'))

    # Calculate location y offset (below all food nodes)
    max_food_y = max([pos[1] for pos in node_positions.values()], default=0) + 2

    # Add block-local location nodes (V2 model)
    for block_idx, bg in enumerate(block_graphs):
        locations = sorted(block_locations[block_idx])
        n_locations = len(locations)

        for i, location in enumerate(locations):
            local_location_id = f"{location}@B{block_idx}"

            # Spread locations horizontally within block (±1.5 from center)
            x_offset = (i - (n_locations - 1) / 2) * 0.5 if n_locations > 1 else 0
            x = block_idx * block_width + x_offset
            y = -max_food_y - 2  # All locations at same y level

            node_positions[local_location_id] = (x, y)
            G.add_node(local_location_id,
                       node_type='location',
                       block=block_idx,
                       original_id=location)

        # Add location edges (food -> location, within this block)
        for food_id, node in bg.get('food_nodes', {}).items():
            location = get_food_location(node, bg)
            if location:
                local_food_id = f"{food_id}@B{block_idx}"
                local_location_id = f"{location}@B{block_idx}"

                if local_food_id in G.nodes and local_location_id in G.nodes:
                    G.add_edge(local_food_id, local_location_id,
                              edge_type='location',
                              block=block_idx)

    # Add lineage edges (connect food nodes across ADJACENT blocks)
    # Use stored source_block and target_block values (always adjacent: N-1 -> N)
    for edge in lineage_edges:
        parent_id = edge.get('parent_instance_id', '')
        child_id = edge.get('child_instance_id', '')

        # Use the stored block indices directly (not computed from where instances appear)
        source_block_idx = edge.get('source_block')
        target_block_idx = edge.get('target_block')

        if source_block_idx is None or target_block_idx is None:
            continue

        local_parent_id = f"{parent_id}@B{source_block_idx}"
        local_child_id = f"{child_id}@B{target_block_idx}"

        if local_parent_id in G.nodes and local_child_id in G.nodes:
            G.add_edge(local_parent_id, local_child_id,
                      edge_type='lineage',
                      derivation=edge.get('derivation_type', 'unknown'),
                      source_block=source_block_idx,
                      target_block=target_block_idx)

    return G, node_positions, food_colors, block_info


def visualize_graph(G: nx.DiGraph, positions: Dict, food_colors: Dict,
                    block_info: List[Dict], output_file: Path = None, figsize: Tuple = (24, 14)):
    """Create visualization of the spatio-temporal graph"""

    _, ax = plt.subplots(figsize=figsize)
    block_count = len(block_info)

    # Separate nodes by type
    food_nodes = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'food']
    location_nodes = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'location']

    # Separate edges by type
    location_edges = [(u, v) for u, v, d in G.edges(data=True)
                      if d.get('edge_type') == 'location']
    lineage_edges = [(u, v) for u, v, d in G.edges(data=True)
                     if d.get('edge_type') == 'lineage']

    # Track video boundaries for coloring
    video_colors = {}
    video_palette = ['#f0f8ff', '#fff8f0', '#f0fff0', '#fff0f8', '#f8f0ff']
    current_video = None
    video_idx = -1

    # Draw block separators and labels
    block_width = 4
    for block_idx in range(block_count):
        info = block_info[block_idx]
        video_id = info['video_id']
        block_id = info['block_id']

        # Track video changes for background coloring
        if video_id != current_video:
            current_video = video_id
            video_idx += 1
            video_colors[video_id] = video_palette[video_idx % len(video_palette)]

        x = block_idx * block_width - 0.5
        ax.axvline(x=x, color='#e0e0e0', linestyle='-', linewidth=0.5, zorder=0)

        # Short video label (last 6 chars of timestamp)
        short_vid = video_id[-6:] if len(video_id) > 6 else video_id
        label = f'{short_vid}\nB{block_id}'
        ax.text(block_idx * block_width, ax.get_ylim()[1] if ax.get_ylim()[1] != 0 else 5,
                label, ha='center', fontsize=7, color='#666')

    # Draw location edges (light gray, within blocks)
    if location_edges:
        nx.draw_networkx_edges(G, positions, edgelist=location_edges,
                               edge_color='#aaaaaa', alpha=0.6,
                               width=1, style='dotted', arrows=True,
                               arrowsize=8, ax=ax)

    # Draw lineage edges (red, across blocks)
    if lineage_edges:
        nx.draw_networkx_edges(G, positions, edgelist=lineage_edges,
                               edge_color='#e74c3c', alpha=0.8,
                               width=2, arrows=True, arrowsize=12,
                               connectionstyle='arc3,rad=0.1', ax=ax)

    # Draw food nodes
    food_node_colors = [G.nodes[n].get('color', '#999') for n in food_nodes]
    nx.draw_networkx_nodes(G, positions, nodelist=food_nodes,
                           node_color=food_node_colors, node_size=500,
                           node_shape='o', alpha=0.9, ax=ax,
                           edgecolors='black', linewidths=0.5)

    # Draw location nodes (smaller, gray squares)
    if location_nodes:
        nx.draw_networkx_nodes(G, positions, nodelist=location_nodes,
                               node_color='#95a5a6', node_size=300,
                               node_shape='s', alpha=0.7, ax=ax)

    # Draw food labels (shortened)
    food_labels = {}
    for n in food_nodes:
        # Extract original_id (e.g., "orange_001" from "orange_001@B3")
        original_id = G.nodes[n].get('original_id', n.split('@')[0])
        parts = original_id.split('_')
        if len(parts) >= 2:
            food_labels[n] = parts[-1]  # Just the number (e.g., "001")
        else:
            food_labels[n] = original_id[:8]

    nx.draw_networkx_labels(G, positions, labels=food_labels,
                            font_size=6, font_color='black',
                            font_weight='bold', ax=ax)

    # Draw location labels (full name, rotated for readability)
    location_labels = {}
    for n in location_nodes:
        original = G.nodes[n].get('original_id', n)
        # Remove _001/_002 suffix but keep full name
        short = re.sub(r'_\d{3}$', '', original)
        location_labels[n] = short

    # Draw location labels with rotation to avoid overlap
    for node, label in location_labels.items():
        x, y = positions[node]
        ax.annotate(label, (x, y), fontsize=6, ha='center', va='top',
                    rotation=45, color='#333',
                    xytext=(0, -12), textcoords='offset points')

    # Create legend
    legend_elements = []

    # Food type legend
    for food_noun, color in sorted(food_colors.items()):
        short_name = food_noun[:20] + '...' if len(food_noun) > 20 else food_noun
        legend_elements.append(mpatches.Patch(color=color, label=short_name))

    # Node type legend
    legend_elements.append(mpatches.Patch(color='#95a5a6', label='Location (per block)'))

    # Edge type legend
    legend_elements.append(plt.Line2D([0], [0], color='#e74c3c', linewidth=2,
                                       label='Lineage (cross-block)'))
    legend_elements.append(plt.Line2D([0], [0], color='#aaaaaa', linewidth=1,
                                       linestyle='dotted', label='At location (in-block)'))

    ax.legend(handles=legend_elements, loc='upper left', fontsize=8)

    # Title
    ax.set_title('Spatio-Temporal Food Graph (V2 Location Model)\n'
                 'Each block = complete spatial snapshot | '
                 'Lineage=red (cross-block), Location=gray (in-block)',
                 fontsize=12, fontweight='bold')
    ax.axis('off')

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"Graph saved to: {output_file}")
    else:
        plt.show()

    plt.close()


def print_graph_summary(G: nx.DiGraph, graph_data: Dict):
    """Print summary statistics"""

    print("\n" + "=" * 60)
    print("GRAPH SUMMARY (V2 Location Model)")
    print("=" * 60)

    food_nodes = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'food']
    location_nodes = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'location']

    location_edges = [(u, v) for u, v, d in G.edges(data=True)
                      if d.get('edge_type') == 'location']
    lineage_edges = [(u, v) for u, v, d in G.edges(data=True)
                     if d.get('edge_type') == 'lineage']

    block_count = len(graph_data.get('block_graphs', []))

    print(f"\nBlocks: {block_count}")
    print(f"\nNodes:")
    print(f"  Food nodes: {len(food_nodes)} (block-local, shown in each block)")
    print(f"  Location nodes: {len(location_nodes)} (block-local)")
    print(f"  Total: {G.number_of_nodes()}")

    print(f"\nEdges:")
    print(f"  Location: {len(location_edges)} (food->location, within-block)")
    print(f"  Lineage: {len(lineage_edges)} (cross-block)")
    print(f"  Total: {G.number_of_edges()}")

    # Food nodes by type
    food_by_type = defaultdict(list)
    for n in food_nodes:
        food_noun = G.nodes[n].get('food_noun', 'unknown')
        food_by_type[food_noun].append(n)

    print(f"\nFood nodes by type:")
    for food_noun, nodes in sorted(food_by_type.items()):
        print(f"  {food_noun}: {len(nodes)} nodes")

    # Lineage summary
    if lineage_edges:
        print(f"\nLineage edges (cross-block splits/merges): {len(lineage_edges)}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize Spatio-Temporal Food Graph using NetworkX"
    )
    parser.add_argument(
        '--graph-file',
        default=str(_PROJECT_ROOT / "gemini_outputs" / "food_graph_gemini" / "spatio_temporal_graph.json"),
        help='Path to spatio_temporal_graph.json'
    )
    parser.add_argument(
        '--output',
        default=str(_PROJECT_ROOT / "gemini_outputs" / "food_graph_gemini" / "food_graph_visualization.png"),
        help='Output image file path'
    )
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Display instead of saving'
    )
    parser.add_argument(
        '--figsize',
        nargs=2,
        type=int,
        default=[24, 14],
        help='Figure size (width height)'
    )

    args = parser.parse_args()

    graph_file = Path(args.graph_file)
    output_file = None if args.no_save else Path(args.output)

    print("=" * 60)
    print("SPATIO-TEMPORAL FOOD GRAPH VISUALIZATION")
    print("=" * 60)
    print(f"Input: {graph_file}")
    print(f"Output: {output_file if output_file else 'display'}")

    if not graph_file.exists():
        print(f"\nERROR: Graph file not found: {graph_file}")
        return

    print("\n[Step 1] Loading graph...")
    graph_data = load_graph(graph_file)

    print("[Step 2] Building NetworkX graph (V2 Location Model)...")
    print("  - Food nodes: Block-local IDs (food@block) - present in EVERY block")
    print("  - Location nodes: Block-local IDs (location@block)")
    print("  - Location edges: Within-block (food@B → location@B)")
    print("  - Lineage edges: Cross-block (parent@B_src → child@B_tgt)")

    G, positions, food_colors, block_info = create_spatio_temporal_graph(graph_data)

    print_graph_summary(G, graph_data)

    print("\n[Step 3] Creating visualization...")
    visualize_graph(G, positions, food_colors, block_info, output_file,
                    tuple(args.figsize))

    print("\nDone!")


if __name__ == '__main__':
    main()

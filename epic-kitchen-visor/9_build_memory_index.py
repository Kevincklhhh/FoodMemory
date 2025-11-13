#!/usr/bin/env python3
"""
Step 9: Build FAISS index for food memory database.

This script implements the Memory Module from FoodMemory_System_Design.md:
1. Loads visual embeddings from Step 8
2. Normalizes embeddings for cosine similarity
3. Builds FAISS index using autofaiss
4. Creates metadata mapping for retrieval

Usage:
    python3 9_build_memory_index.py
    python3 9_build_memory_index.py --input memory_database/embeddings
"""

import json
import argparse
import numpy as np
from pathlib import Path
from typing import Dict
import pandas as pd

try:
    import faiss
except ImportError as e:
    print(f"Error: Missing dependency - {e}")
    print("Please install: pip install faiss-cpu")
    import sys
    sys.exit(1)


def load_embeddings(embeddings_dir: Path) -> tuple:
    """Load embeddings and metadata from Step 8 output.

    Args:
        embeddings_dir: Directory containing embeddings and metadata

    Returns:
        Tuple of (embeddings array, metadata dict)
    """
    embeddings_path = embeddings_dir / 'food_embeddings.npy'
    metadata_path = embeddings_dir / 'food_metadata.json'

    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embeddings not found: {embeddings_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")

    print(f"Loading embeddings from {embeddings_path}...")
    embeddings = np.load(embeddings_path)
    print(f"✓ Loaded embeddings: {embeddings.shape}")

    print(f"Loading metadata from {metadata_path}...")
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    print(f"✓ Loaded metadata for {metadata['total_frames']} frames")

    return embeddings, metadata


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """L2 normalize embeddings for cosine similarity via inner product.

    Args:
        embeddings: N x D array of embeddings

    Returns:
        L2-normalized embeddings
    """
    print("Normalizing embeddings (L2 norm)...")
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    # Avoid division by zero
    norms[norms == 0] = 1
    normalized = embeddings / norms
    print(f"✓ Normalized {len(normalized)} embeddings")
    return normalized.astype(np.float32)


def build_faiss_index(embeddings: np.ndarray,
                      output_dir: Path) -> str:
    """Build FAISS index directly (flat index for exact search).

    Args:
        embeddings: N x D normalized embeddings
        output_dir: Output directory for index

    Returns:
        Path to created index file
    """
    print("\nBuilding FAISS index...")
    print(f"  Embeddings shape: {embeddings.shape}")
    n, d = embeddings.shape

    output_dir.mkdir(parents=True, exist_ok=True)

    # Build index directly with FAISS
    # Use IndexFlatIP for exact inner product search (cosine similarity on normalized vectors)
    index_path = str(output_dir / "memory_index.faiss")

    print(f"  Index type: Flat (exact search)")
    print(f"  Metric: Inner Product (cosine similarity)")

    # Create flat index for inner product
    index = faiss.IndexFlatIP(d)

    # Add embeddings
    index.add(embeddings)

    print(f"✓ Index built successfully")
    print(f"  Vectors in index: {index.ntotal}")

    # Save index
    faiss.write_index(index, index_path)
    print(f"✓ Index saved to {index_path}")

    # Save index info
    index_info = {
        "index_type": "Flat",
        "metric_type": "inner_product",
        "dimension": d,
        "num_vectors": int(index.ntotal),
        "description": "Exact search using inner product on L2-normalized vectors (cosine similarity)"
    }

    info_path = output_dir / "index_info.json"
    with open(info_path, 'w') as f:
        json.dump(index_info, f, indent=2)
    print(f"✓ Index info saved to {info_path}")

    return index_path


def create_metadata_mapping(metadata: Dict, output_dir: Path) -> None:
    """Create parquet metadata file for fast lookups during retrieval.

    Args:
        metadata: Metadata dict from Step 8
        output_dir: Output directory
    """
    print("\nCreating metadata mapping...")

    # Convert to DataFrame
    frames_data = []
    for i, frame in enumerate(metadata['frames']):
        frames_data.append({
            'index': i,  # Position in embedding array
            'food_class': frame['food_class'],
            'instance_id': frame['instance_id'],
            'frame_id': frame['frame_id'],
            'filename': frame['filename'],
            'semantic_label': frame['semantic_label'],
            'video_id': frame['source_reference']['video_id'],
            'frame_number': frame['source_reference']['frame_number'],
            'participant_id': frame['source_reference']['participant_id'],
            'visor_image_path': frame['source_reference']['visor_image_path']
        })

    df = pd.DataFrame(frames_data)

    # Save as parquet for efficient loading
    metadata_path = output_dir / "memory_metadata.parquet"
    df.to_parquet(metadata_path, index=False)
    print(f"✓ Saved metadata to {metadata_path}")

    # Print statistics
    print("\nMetadata statistics:")
    print(f"  Total frames: {len(df)}")
    print(f"  Total instances: {df['instance_id'].nunique()}")
    print(f"  Food classes: {df['food_class'].nunique()}")
    print(f"  Unique videos: {df['video_id'].nunique()}")


def verify_index(index_path: str, embeddings: np.ndarray) -> None:
    """Verify the created index works correctly.

    Args:
        index_path: Path to FAISS index
        embeddings: Original embeddings
    """
    print("\nVerifying index...")

    # Load index
    index = faiss.read_index(index_path)
    print(f"  Index size: {index.ntotal} vectors")
    print(f"  Index dimension: {index.d}")

    # Test search with first embedding
    if len(embeddings) > 0:
        query = embeddings[0:1]
        distances, indices = index.search(query, k=5)

        print(f"  Test query (first embedding):")
        print(f"    Top-1 index: {indices[0][0]} (should be 0)")
        print(f"    Top-1 similarity: {distances[0][0]:.4f} (should be ~1.0)")

        if indices[0][0] == 0 and distances[0][0] > 0.99:
            print("  ✓ Index verification passed")
        else:
            print("  ⚠ Warning: Index verification failed")


def main():
    parser = argparse.ArgumentParser(
        description="Build FAISS index for food memory database (Step 9)"
    )
    parser.add_argument(
        '--input',
        default='memory_database/embeddings',
        help='Input directory with embeddings (default: memory_database/embeddings)'
    )
    parser.add_argument(
        '--output',
        default='memory_database/index',
        help='Output directory for index (default: memory_database/index)'
    )

    args = parser.parse_args()

    print("=" * 80)
    print("STEP 9: Build FAISS Index for Food Memory Database")
    print("=" * 80)
    print()

    # Load embeddings and metadata
    embeddings, metadata = load_embeddings(Path(args.input))

    # Normalize embeddings
    normalized_embeddings = normalize_embeddings(embeddings)

    # Build FAISS index
    index_path = build_faiss_index(
        embeddings=normalized_embeddings,
        output_dir=Path(args.output)
    )

    # Create metadata mapping
    create_metadata_mapping(metadata, Path(args.output))

    # Verify index
    verify_index(index_path, normalized_embeddings)

    print("\n" + "=" * 80)
    print("✓ Memory index built successfully!")
    print("=" * 80)
    print(f"Index location: {index_path}")
    print(f"Metadata: {Path(args.output) / 'memory_metadata.parquet'}")
    print(f"Index info: {Path(args.output) / 'index_info.json'}")
    print("\nNext step: Use 10_food_retrieval.py to query the index")
    print("=" * 80)


if __name__ == '__main__':
    main()

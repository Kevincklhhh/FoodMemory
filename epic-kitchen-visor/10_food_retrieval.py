#!/usr/bin/env python3
"""
Step 10: Food retrieval interface using visual similarity.

This script implements the Retrieval Module from FoodMemory_System_Design.md:
1. Loads FAISS index and metadata from Step 9
2. Processes query images with masks (same as logging)
3. Performs k-NN search for visual similarity
4. Returns ranked results with metadata

Usage:
    # Interactive query
    python3 10_food_retrieval.py --query /path/to/image.jpg --mask-json mask.json

    # Batch query from benchmark
    python3 10_food_retrieval.py --benchmark yoghurt --k 10
"""

import json
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
import sys

# Add clip-retrieval to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'clip-retrieval'))

try:
    import torch
    import cv2
    import faiss
    import pandas as pd
    from PIL import Image
    from clip_retrieval.clip_inference.mapper import ClipMapper
except ImportError as e:
    print(f"Error: Missing dependency - {e}")
    print("Please install: pip install torch opencv-python pillow faiss-cpu pandas pyarrow open-clip-torch")
    sys.exit(1)


class FoodMemoryRetriever:
    """Food memory retrieval system using visual similarity."""

    def __init__(self,
                 index_dir: Path,
                 clip_model: str = "ViT-B/32"):
        """Initialize retriever.

        Args:
            index_dir: Directory containing FAISS index and metadata
            clip_model: CLIP model name (must match Step 8)
        """
        self.index_dir = index_dir
        self.clip_model = clip_model

        # Load FAISS index
        index_path = index_dir / "memory_index.faiss"
        print(f"Loading FAISS index from {index_path}...")
        self.index = faiss.read_index(str(index_path))
        print(f"✓ Loaded index with {self.index.ntotal} vectors")

        # Load metadata
        metadata_path = index_dir / "memory_metadata.parquet"
        print(f"Loading metadata from {metadata_path}...")
        self.metadata_df = pd.read_parquet(metadata_path)
        print(f"✓ Loaded metadata for {len(self.metadata_df)} frames")

        # Initialize CLIP mapper
        print(f"Initializing CLIP model: {clip_model}...")
        self.clip_mapper = ClipMapper(
            enable_image=True,
            enable_text=False,
            enable_metadata=False,
            use_mclip=False,
            clip_model=clip_model,
            use_jit=False,
            mclip_model='M-CLIP/XLM-Roberta-Large-Vit-L-14'  # Required parameter but not used
        )

        # Get preprocess transform from CLIP model
        import open_clip
        _, _, self.preprocess = open_clip.create_model_and_transforms(
            clip_model.replace('/', '-'),  # Convert 'ViT-B/32' to 'ViT-B-32'
            pretrained='openai'
        )
        print("✓ CLIP model ready")

    def create_mask_from_segments(self,
                                  segments: List[List[List[float]]],
                                  height: int,
                                  width: int) -> np.ndarray:
        """Create binary mask from VISOR polygon segments."""
        mask = np.zeros((height, width), dtype=np.uint8)
        for polygon in segments:
            pts = np.array(polygon, dtype=np.int32)
            cv2.fillPoly(mask, [pts], 255)
        return mask

    def apply_mask(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Apply mask to image (black out background)."""
        masked = image.copy()
        masked[mask == 0] = 0
        return masked

    def extract_query_embedding(self,
                                image_path: str,
                                mask_segments: Optional[List] = None) -> np.ndarray:
        """Extract CLIP embedding from query image.

        Args:
            image_path: Path to query image
            mask_segments: Optional VISOR segments for masking

        Returns:
            Normalized embedding vector (1 x D)
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]

        # Apply mask if provided
        if mask_segments is not None:
            mask = self.create_mask_from_segments(mask_segments, height, width)
            image = self.apply_mask(image, mask)

        # Convert to PIL
        pil_image = Image.fromarray(image)

        # Preprocess image to tensor
        image_tensor = self.preprocess(pil_image).unsqueeze(0)  # Add batch dimension

        # Extract embedding
        result = self.clip_mapper({
            'image_tensor': image_tensor,
            'image_filename': str(image_path)
        })
        embedding = result['image_embs'][0]  # Get first (and only) embedding

        # Normalize
        embedding = embedding / np.linalg.norm(embedding)

        return embedding.reshape(1, -1).astype(np.float32)

    def retrieve(self,
                query_image_path: str,
                query_mask_segments: Optional[List] = None,
                k: int = 10) -> List[Dict]:
        """Retrieve similar food instances.

        Args:
            query_image_path: Path to query image
            query_mask_segments: Optional VISOR segments for masking
            k: Number of results to return

        Returns:
            List of results with metadata and similarity scores
        """
        # Extract query embedding
        query_embedding = self.extract_query_embedding(
            query_image_path,
            query_mask_segments
        )

        # Search index
        distances, indices = self.index.search(query_embedding, k)

        # Build results
        results = []
        for rank, (idx, distance) in enumerate(zip(indices[0], distances[0])):
            # Get metadata
            row = self.metadata_df.iloc[idx]

            result = {
                'rank': rank + 1,
                'similarity': float(distance),  # Cosine similarity
                'instance_id': row['instance_id'],
                'frame_id': row['frame_id'],
                'filename': row['filename'],
                'food_class': row['food_class'],
                'semantic_label': row['semantic_label'],
                'metadata': {
                    'video_id': row['video_id'],
                    'frame_number': int(row['frame_number']),
                    'participant_id': row['participant_id'],
                    'visor_image_path': row['visor_image_path']
                }
            }
            results.append(result)

        return results

    def batch_retrieve(self,
                      benchmark_metadata: Dict,
                      benchmarks_dir: Path,
                      k: int = 10) -> Dict:
        """Retrieve for all frames in a benchmark.

        Args:
            benchmark_metadata: Benchmark instances JSON
            benchmarks_dir: Directory containing benchmark images
            k: Number of results per query

        Returns:
            Dict mapping frame_id to retrieval results
        """
        all_results = {}

        for instance_id, instance_data in benchmark_metadata['instances'].items():
            for frame in instance_data['frames']:
                frame_id = frame['frame_id']
                filename = frame['filename']
                image_path = benchmarks_dir / filename

                if not image_path.exists():
                    print(f"Warning: {image_path} not found, skipping")
                    continue

                print(f"Querying: {filename}")

                # Retrieve using frame's own mask
                results = self.retrieve(
                    query_image_path=str(image_path),
                    query_mask_segments=frame['segments'],
                    k=k
                )

                all_results[frame_id] = {
                    'query_frame': filename,
                    'query_instance': instance_id,
                    'results': results
                }

        return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Food retrieval using visual similarity (Step 10)"
    )
    parser.add_argument(
        '--index-dir',
        default='memory_database/index',
        help='Directory with FAISS index (default: memory_database/index)'
    )
    parser.add_argument(
        '--clip-model',
        default='ViT-B/32',
        help='CLIP model name (default: ViT-B/32)'
    )

    # Query modes
    query_group = parser.add_mutually_exclusive_group(required=True)
    query_group.add_argument(
        '--query',
        help='Path to query image'
    )
    query_group.add_argument(
        '--benchmark',
        help='Benchmark food class (e.g., yoghurt) for batch retrieval'
    )

    parser.add_argument(
        '--mask-json',
        help='JSON file with mask segments (for --query mode)'
    )
    parser.add_argument(
        '--benchmarks-dir',
        default='retrieve_benchmarks',
        help='Benchmarks directory (for --benchmark mode)'
    )
    parser.add_argument(
        '-k',
        type=int,
        default=10,
        help='Number of results to return (default: 10)'
    )
    parser.add_argument(
        '--output',
        help='Output JSON file for batch results'
    )

    args = parser.parse_args()

    print("=" * 80)
    print("STEP 10: Food Retrieval using Visual Similarity")
    print("=" * 80)
    print()

    # Initialize retriever
    retriever = FoodMemoryRetriever(
        index_dir=Path(args.index_dir),
        clip_model=args.clip_model
    )

    # Single query mode
    if args.query:
        print(f"\nQuerying: {args.query}")

        # Load mask if provided
        mask_segments = None
        if args.mask_json:
            with open(args.mask_json, 'r') as f:
                mask_data = json.load(f)
                mask_segments = mask_data.get('segments')
        else:
            # Auto-detect VISOR dataset format and load mask from benchmark metadata
            query_path = Path(args.query)
            query_filename = query_path.name

            # Check if this is from a benchmark directory (retrieve_samples or retrieve_benchmarks)
            if 'retrieve_samples' in str(query_path) or 'retrieve_benchmarks' in str(query_path):
                # Try to find the food class directory
                food_class = None
                for parent in query_path.parents:
                    if parent.name in ['retrieve_samples', 'retrieve_benchmarks']:
                        # Food class is the directory above the image
                        food_class = query_path.parent.name
                        benchmark_dir = parent
                        break

                if food_class:
                    # Try to load benchmark metadata
                    metadata_path = benchmark_dir / f"{food_class}_benchmark_instances.json"
                    if metadata_path.exists():
                        print(f"  ℹ Auto-detected VISOR benchmark image, loading mask from {metadata_path.name}")
                        with open(metadata_path, 'r') as f:
                            benchmark_data = json.load(f)

                        # Find matching frame in benchmark
                        for instance_data in benchmark_data['instances'].values():
                            for frame in instance_data['frames']:
                                if frame['filename'] == query_filename:
                                    mask_segments = frame['segments']
                                    print(f"  ✓ Found mask for {query_filename}")
                                    break
                            if mask_segments:
                                break

        results = retriever.retrieve(
            query_image_path=args.query,
            query_mask_segments=mask_segments,
            k=args.k
        )

        # Print results
        print(f"\nTop {len(results)} results:")
        print("-" * 80)
        for result in results:
            print(f"Rank {result['rank']}: {result['filename']}")
            print(f"  Instance ID: {result['instance_id']}")
            print(f"  Similarity: {result['similarity']:.4f}")
            print(f"  Food class: {result['food_class']}")
            print(f"  Video: {result['metadata']['video_id']}")
            print()

    # Batch benchmark mode
    elif args.benchmark:
        benchmarks_dir = Path(args.benchmarks_dir)
        food_class = args.benchmark

        # Load benchmark metadata
        metadata_path = benchmarks_dir / f"{food_class}_benchmark_instances.json"
        if not metadata_path.exists():
            print(f"Error: Benchmark metadata not found: {metadata_path}")
            return

        with open(metadata_path, 'r') as f:
            benchmark_metadata = json.load(f)

        food_dir = benchmarks_dir / food_class

        print(f"\nBatch retrieval for {food_class} benchmark...")
        print(f"Total frames: {benchmark_metadata['statistics']['total_frames']}")
        print()

        results = retriever.batch_retrieve(
            benchmark_metadata=benchmark_metadata,
            benchmarks_dir=food_dir,
            k=args.k
        )

        # Print summary
        print("\n" + "=" * 80)
        print("BATCH RETRIEVAL SUMMARY")
        print("=" * 80)
        print(f"Total queries: {len(results)}")
        print()

        # Print results for each query
        for frame_id, result_data in results.items():
            print(f"Query: {result_data['query_frame']} (Instance: {result_data['query_instance']})")
            print("Top 5 results:")
            for result in result_data['results'][:5]:
                same_instance = result['instance_id'] == result_data['query_instance']
                marker = "✓" if same_instance else " "
                print(f"  {marker} {result['rank']}. {result['filename']} "
                      f"(sim={result['similarity']:.3f}, instance={result['instance_id']})")
            print()

        # Save to file if requested
        if args.output:
            output_path = Path(args.output)
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to: {output_path}")

    print("=" * 80)


if __name__ == '__main__':
    main()

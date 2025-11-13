#!/usr/bin/env python3
"""
Step 8: Extract visual embeddings and VLM captions for food memory database.

This script implements the Logging Module from FoodMemory_System_Design.md:
1. Loads frames from retrieve_benchmarks/{food}/
2. Applies VISOR segmentation masks (black out background)
3. Extracts CLIP visual embeddings using clip-retrieval
4. Generates VLM captions using GPT-4o
5. Saves embeddings, captions, and metadata for memory indexing

Usage:
    python3 8_extract_food_embeddings.py --food yoghurt
    python3 8_extract_food_embeddings.py --food yoghurt pizza --batch-size 32
    python3 8_extract_food_embeddings.py --food yoghurt --skip-captions  # Skip VLM
"""

import json
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict
import sys
import base64

# Add clip-retrieval to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'clip-retrieval'))

# Add llm-api to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'llm-api'))

try:
    import torch
    import cv2
    from PIL import Image
    from clip_retrieval.clip_inference.mapper import ClipMapper
except ImportError as e:
    print(f"Error: Missing dependency - {e}")
    print("Please install: pip install torch torchvision opencv-python pillow open-clip-torch")
    sys.exit(1)

try:
    from openai_api import OpenAIAPI
except ImportError:
    print("Warning: Could not import openai_api. VLM captions will be disabled.")
    OpenAIAPI = None


def load_benchmark_metadata(benchmark_path: Path) -> Dict:
    """Load benchmark instances JSON file."""
    print(f"Loading benchmark metadata from {benchmark_path}...")
    with open(benchmark_path, 'r') as f:
        data = json.load(f)

    frame_count = sum(inst['frame_count'] for inst in data['instances'].values())
    print(f"✓ Loaded {len(data['instances'])} instances with {frame_count} frames")
    return data


def create_mask_from_visor_segments(segments: List[List[List[float]]],
                                    height: int, width: int) -> np.ndarray:
    """Create binary mask from VISOR polygon segments.

    Args:
        segments: List of polygons, each polygon is list of [x,y] points
        height: Image height
        width: Image width

    Returns:
        Binary mask (height x width) where 255 = food region, 0 = background
    """
    mask = np.zeros((height, width), dtype=np.uint8)

    for polygon in segments:
        # Convert to numpy array of integer coordinates
        pts = np.array(polygon, dtype=np.int32)
        # Fill polygon
        cv2.fillPoly(mask, [pts], 255)

    return mask


def apply_mask_to_image(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Apply mask to image (black out background).

    Args:
        image: RGB image (H x W x 3)
        mask: Binary mask (H x W)

    Returns:
        Masked image with background blacked out
    """
    masked = image.copy()
    masked[mask == 0] = 0  # Black out background
    return masked


def create_mask_overlay_image(image: np.ndarray, mask: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    """Create image with transparent mask overlay for VLM.

    Args:
        image: Original RGB image (H x W x 3)
        mask: Binary mask (H x W)
        alpha: Transparency of mask overlay

    Returns:
        Image with green transparent overlay on masked region
    """
    overlay = image.copy()
    # Create green overlay on masked region
    overlay[mask > 0] = (overlay[mask > 0] * (1 - alpha) +
                         np.array([0, 255, 0]) * alpha).astype(np.uint8)
    return overlay


def image_to_base64(image: np.ndarray) -> str:
    """Convert numpy image to base64 string.

    Args:
        image: RGB image (H x W x 3)

    Returns:
        Base64 encoded JPEG string
    """
    # Convert to PIL Image
    pil_image = Image.fromarray(image)

    # Save to bytes
    import io
    buffer = io.BytesIO()
    pil_image.save(buffer, format='JPEG', quality=85)
    buffer.seek(0)

    # Encode to base64
    img_bytes = buffer.read()
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')

    return f"data:image/jpeg;base64,{img_base64}"


def generate_vlm_caption(image: np.ndarray,
                        mask: np.ndarray,
                        semantic_label: str,
                        openai_api: 'OpenAIAPI') -> str:
    """Generate VLM caption using GPT-4o.

    Args:
        image: Original RGB image
        mask: Binary mask
        semantic_label: Food class label (e.g., "yoghurt")
        openai_api: OpenAI API instance

    Returns:
        Generated caption string
    """
    # Create overlay image
    overlay_image = create_mask_overlay_image(image, mask)

    # Convert to base64
    image_b64 = image_to_base64(overlay_image)

    # Create prompt
    prompt = f"""You are observing a kitchen scene with a highlighted food item.
The green overlay highlights the {semantic_label}.

Describe this {semantic_label} in the context of the kitchen scene. Include:
- The appearance and state of the {semantic_label}
- Its location/context in the scene (e.g., on counter, in fridge, being held)
- Any relevant visual details that would help identify this specific instance

Keep the description concise (2-3 sentences)."""

    # Call GPT-4o
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_b64
                    }
                }
            ]
        }
    ]

    try:
        completion = openai_api.chat_completion(messages, max_tokens=256)
        caption = completion.choices[0].message.content.strip()
        return caption
    except Exception as e:
        print(f"Warning: VLM caption generation failed: {e}")
        return f"[VLM error: {str(e)}]"


def extract_embeddings(food_classes: List[str],
                      benchmarks_dir: Path,
                      output_dir: Path,
                      clip_model: str = "ViT-B/32",
                      batch_size: int = 32,
                      skip_captions: bool = False) -> None:
    """Extract CLIP embeddings and VLM captions for all benchmark frames.

    Args:
        food_classes: List of food classes to process
        benchmarks_dir: Directory containing benchmark subdirectories
        output_dir: Output directory for embeddings and metadata
        clip_model: CLIP model name
        batch_size: Batch size for CLIP inference
        skip_captions: If True, skip VLM caption generation
    """

    # Initialize CLIP mapper
    print(f"\nInitializing CLIP model: {clip_model}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    clip_mapper = ClipMapper(
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
    _, _, preprocess = open_clip.create_model_and_transforms(
        clip_model.replace('/', '-'),  # Convert 'ViT-B/32' to 'ViT-B-32'
        pretrained='openai'
    )
    print(f"✓ CLIP preprocess transform initialized")

    # Initialize OpenAI API for VLM captions
    openai_api = None
    if not skip_captions:
        if OpenAIAPI is None:
            print("Warning: OpenAI API not available, skipping captions")
            skip_captions = True
        else:
            try:
                print("Initializing GPT-4o for VLM captions...")
                openai_api = OpenAIAPI(deployment="gpt-4o")
                print("✓ GPT-4o ready")
            except Exception as e:
                print(f"Warning: Failed to initialize OpenAI API: {e}")
                print("Skipping VLM captions")
                skip_captions = True

    # Collect all frames to process
    all_frames = []

    for food_class in food_classes:
        food_dir = benchmarks_dir / food_class
        metadata_path = benchmarks_dir / f"{food_class}_benchmark_instances.json"

        if not food_dir.exists():
            print(f"Warning: {food_dir} does not exist, skipping")
            continue

        if not metadata_path.exists():
            print(f"Warning: {metadata_path} does not exist, skipping")
            continue

        # Load benchmark metadata
        benchmark_data = load_benchmark_metadata(metadata_path)

        # Process each instance
        for instance_id, instance_data in benchmark_data['instances'].items():
            for frame in instance_data['frames']:
                image_path = food_dir / frame['filename']

                if not image_path.exists():
                    print(f"Warning: {image_path} does not exist, skipping")
                    continue

                all_frames.append({
                    'food_class': food_class,
                    'instance_id': instance_id,
                    'frame_id': frame['frame_id'],
                    'filename': frame['filename'],
                    'image_path': str(image_path),
                    'segments': frame['segments'],
                    'semantic_label': frame['object_name'],
                    'source_reference': {
                        'video_id': frame['video_id'],
                        'frame_number': frame['frame_number'],
                        'visor_image_path': frame['image_path'],
                        'participant_id': frame['participant_id']
                    }
                })

    if not all_frames:
        print("Error: No frames found to process")
        return

    print(f"\n✓ Found {len(all_frames)} frames to process")

    # Extract embeddings and captions
    embeddings = []
    metadata_list = []
    captions = []

    print(f"\nExtracting CLIP embeddings and VLM captions...")
    print(f"CLIP batch size: {batch_size}")
    print(f"VLM captions: {'enabled' if not skip_captions else 'disabled'}")

    for i, frame_data in enumerate(all_frames):
        if i % 10 == 0:
            print(f"Processing frame {i+1}/{len(all_frames)}...")

        # Load image
        image = cv2.imread(frame_data['image_path'])
        if image is None:
            print(f"Warning: Failed to load {frame_data['image_path']}, skipping")
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]

        # Create mask from VISOR segments
        mask = create_mask_from_visor_segments(
            frame_data['segments'],
            height,
            width
        )

        # Apply mask for CLIP
        masked_image = apply_mask_to_image(image, mask)

        # Convert to PIL Image for CLIP
        pil_image = Image.fromarray(masked_image)

        # Preprocess image to tensor
        image_tensor = preprocess(pil_image).unsqueeze(0)  # Add batch dimension

        # Extract CLIP embedding
        # ClipMapper expects dict with 'image_tensor' and 'image_filename' keys
        result = clip_mapper({
            'image_tensor': image_tensor,
            'image_filename': frame_data['filename']
        })
        embedding = result['image_embs'][0]  # Get first (and only) embedding

        # Generate VLM caption if enabled
        caption = None
        if not skip_captions and openai_api is not None:
            caption = generate_vlm_caption(
                image=image,  # Original image, not masked
                mask=mask,
                semantic_label=frame_data['semantic_label'],
                openai_api=openai_api
            )

        # Store embedding, caption, and metadata
        embeddings.append(embedding)
        captions.append({
            'frame_id': frame_data['frame_id'],
            'filename': frame_data['filename'],
            'caption': caption
        })
        metadata_list.append({
            'food_class': frame_data['food_class'],
            'instance_id': frame_data['instance_id'],
            'frame_id': frame_data['frame_id'],
            'filename': frame_data['filename'],
            'semantic_label': frame_data['semantic_label'],
            'caption': caption,
            'source_reference': frame_data['source_reference']
        })

    print(f"✓ Extracted {len(embeddings)} embeddings")
    if not skip_captions:
        captions_generated = sum(1 for c in captions if c['caption'] is not None)
        print(f"✓ Generated {captions_generated} VLM captions")

    # Convert to numpy array
    embeddings_array = np.array(embeddings, dtype=np.float32)
    print(f"Embeddings shape: {embeddings_array.shape}")

    # Save embeddings and metadata
    embeddings_dir = output_dir / 'embeddings'
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    embeddings_path = embeddings_dir / 'food_embeddings.npy'
    metadata_path = embeddings_dir / 'food_metadata.json'

    print(f"\nSaving embeddings to {embeddings_path}...")
    np.save(embeddings_path, embeddings_array)

    print(f"Saving metadata to {metadata_path}...")
    with open(metadata_path, 'w') as f:
        json.dump({
            'total_frames': len(metadata_list),
            'embedding_dim': embeddings_array.shape[1],
            'clip_model': clip_model,
            'frames': metadata_list
        }, f, indent=2)

    # Save captions separately
    if not skip_captions:
        captions_dir = output_dir / 'captions'
        captions_dir.mkdir(parents=True, exist_ok=True)
        captions_path = captions_dir / 'vlm_captions.json'

        print(f"Saving VLM captions to {captions_path}...")
        with open(captions_path, 'w') as f:
            json.dump({
                'total_captions': len(captions),
                'vlm_model': 'gpt-4o',
                'captions': captions
            }, f, indent=2)

    print("\n" + "=" * 80)
    print("✓ Embedding extraction complete!")
    print("=" * 80)
    print(f"Total frames processed: {len(embeddings)}")
    print(f"Embedding dimension: {embeddings_array.shape[1]}")
    if not skip_captions:
        captions_count = sum(1 for c in captions if c['caption'] is not None)
        print(f"VLM captions generated: {captions_count}")
    print(f"\nOutput files:")
    print(f"  - {embeddings_path}")
    print(f"  - {metadata_path}")
    if not skip_captions:
        print(f"  - {captions_dir / 'vlm_captions.json'}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Extract CLIP visual embeddings from food benchmark frames (Step 8)"
    )
    parser.add_argument(
        '--food',
        nargs='+',
        required=True,
        help='Food classes to process (e.g., yoghurt pizza)'
    )
    parser.add_argument(
        '--benchmarks-dir',
        default='retrieve_benchmarks',
        help='Directory containing benchmark subdirectories (default: retrieve_benchmarks)'
    )
    parser.add_argument(
        '--output-dir',
        default='memory_database',
        help='Output directory for embeddings (default: memory_database)'
    )
    parser.add_argument(
        '--clip-model',
        default='ViT-B/32',
        help='CLIP model name (default: ViT-B/32)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for CLIP inference (default: 32)'
    )
    parser.add_argument(
        '--skip-captions',
        action='store_true',
        help='Skip VLM caption generation (default: False)'
    )

    args = parser.parse_args()

    print("=" * 80)
    print("STEP 8: Extract Food Embeddings for Memory Database")
    print("=" * 80)
    print(f"Food classes: {', '.join(args.food)}")
    print(f"CLIP model: {args.clip_model}")
    print()

    extract_embeddings(
        food_classes=args.food,
        benchmarks_dir=Path(args.benchmarks_dir),
        output_dir=Path(args.output_dir),
        clip_model=args.clip_model,
        batch_size=args.batch_size,
        skip_captions=args.skip_captions
    )


if __name__ == '__main__':
    main()

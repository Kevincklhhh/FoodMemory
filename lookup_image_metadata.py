#!/usr/bin/env python3
"""
Helper script to lookup image metadata using the food_images_index.json

Demonstrates how to use the index to find bounding boxes and other metadata
from the original EgoObjects dataset files.
"""

import json
from pathlib import Path

# Paths
SCRIPT_DIR = Path(__file__).parent
INDEX_FILE = SCRIPT_DIR / "food_images_index.json"
TRAIN_METADATA = SCRIPT_DIR / "egoobjects_metadata" / "EgoObjectsV1_unified_train.json"
EVAL_METADATA = SCRIPT_DIR / "egoobjects_metadata" / "EgoObjectsV1_unified_eval.json"

def load_index():
    """Load the food images index"""
    with open(INDEX_FILE, 'r') as f:
        return json.load(f)

def load_metadata():
    """Load both train and eval metadata"""
    print("Loading EgoObjects metadata...")

    with open(TRAIN_METADATA, 'r') as f:
        train_data = json.load(f)

    with open(EVAL_METADATA, 'r') as f:
        eval_data = json.load(f)

    print(f"  Train: {len(train_data['images'])} images, {len(train_data['annotations'])} annotations")
    print(f"  Eval: {len(eval_data['images'])} images, {len(eval_data['annotations'])} annotations")

    return train_data, eval_data

def find_image_metadata(original_filename, train_data, eval_data):
    """
    Find metadata for an image by its original filename.
    Returns (source, image_info, annotations)
    """
    # Try train data first
    for img in train_data['images']:
        if img['url'] == original_filename:
            image_id = img['id']

            # Find all annotations for this image
            annotations = [anno for anno in train_data['annotations'] if anno['image_id'] == image_id]

            return 'train', img, annotations

    # Try eval data
    for img in eval_data['images']:
        if img['url'] == original_filename:
            image_id = img['id']

            # Find all annotations for this image
            annotations = [anno for anno in eval_data['annotations'] if anno['image_id'] == image_id]

            return 'eval', img, annotations

    return None, None, []

def lookup_category(category_id, index):
    """Lookup all images for a category"""
    if category_id not in index['categories']:
        print(f"Category {category_id} not found in index")
        return None

    return index['categories'][category_id]

def demo_lookup():
    """Demonstrate looking up metadata for images"""
    print("="*80)
    print("FOOD IMAGE METADATA LOOKUP DEMO")
    print("="*80)

    # Load data
    index = load_index()
    train_data, eval_data = load_metadata()

    print("\n" + "-"*80)
    print("EXAMPLE 1: Lookup all images for 'apple' category")
    print("-"*80)

    apples = lookup_category("014", index)
    if apples:
        print(f"\nCategory: {apples['category_name']}")
        print(f"Total images: {apples['total_images']}")
        print(f"Unique video instances: {apples['unique_instances']}")

        print("\nVideo Instances:")
        for instance_id, instance_data in apples['video_instances'].items():
            print(f"\n  Instance: {instance_id}")
            print(f"  Image count: {instance_data['image_count']}")

            for img_info in instance_data['images']:
                print(f"\n    File: {img_info['filename']}")
                print(f"    Original: {img_info['original_filename']}")

                # Lookup metadata
                source, image_data, annotations = find_image_metadata(
                    img_info['original_filename'],
                    train_data,
                    eval_data
                )

                if image_data:
                    print(f"    Source: {source}")
                    print(f"    Image ID: {image_data['id']}")
                    print(f"    Annotations found: {len(annotations)}")

                    for anno in annotations:
                        print(f"      - Category: {anno['category_id']}")
                        print(f"        Bbox: {anno['bbox']}")
                        print(f"        Area: {anno.get('area', 'N/A')}")
                        print(f"        Is main: {anno.get('is_main', False)}")
                else:
                    print(f"    ⚠️  Metadata not found!")

    print("\n" + "-"*80)
    print("EXAMPLE 2: Lookup specific image by filename")
    print("-"*80)

    # Get first banana image as example
    bananas = lookup_category("029", index)
    if bananas and bananas['video_instances']:
        first_instance = list(bananas['video_instances'].values())[0]
        if first_instance['images']:
            example_img = first_instance['images'][0]

            print(f"\nLooking up: {example_img['filename']}")
            print(f"Original filename: {example_img['original_filename']}")

            source, image_data, annotations = find_image_metadata(
                example_img['original_filename'],
                train_data,
                eval_data
            )

            if image_data:
                print(f"\n✓ Found in {source} dataset")
                print(f"  Image ID: {image_data['id']}")
                print(f"  Width: {image_data.get('width', 'N/A')}")
                print(f"  Height: {image_data.get('height', 'N/A')}")

                print(f"\n  Annotations ({len(annotations)} total):")
                for i, anno in enumerate(annotations, 1):
                    print(f"    {i}. Category ID: {anno['category_id']}")
                    print(f"       Bbox [x, y, w, h]: {anno['bbox']}")
                    print(f"       Area: {anno.get('area', 'N/A')}")
                    print(f"       Is main object: {anno.get('is_main', False)}")

    print("\n" + "-"*80)
    print("EXAMPLE 3: Statistics by source (train vs eval)")
    print("-"*80)

    train_count = 0
    eval_count = 0
    total_images = 0

    for cat_id, cat_data in index['categories'].items():
        for instance_id, instance_data in cat_data['video_instances'].items():
            for img_info in instance_data['images']:
                total_images += 1

                source, _, _ = find_image_metadata(
                    img_info['original_filename'],
                    train_data,
                    eval_data
                )

                if source == 'train':
                    train_count += 1
                elif source == 'eval':
                    eval_count += 1

    print(f"\nTotal images in index: {total_images}")
    print(f"  From train dataset: {train_count} ({train_count/total_images*100:.1f}%)")
    print(f"  From eval dataset: {eval_count} ({eval_count/total_images*100:.1f}%)")

    print("\n" + "="*80)

if __name__ == "__main__":
    demo_lookup()

#!/usr/bin/env python3
"""
Receipt Parser Module
Parses receipt text files and extracts expected food items for un-bagging videos.
"""

import json
from pathlib import Path
from typing import List, Optional


class ReceiptParser:
    """Parses receipt text files to extract expected food items."""

    @staticmethod
    def parse(receipt_path: str) -> Optional[List[str]]:
        """
        Parse a receipt text file and return list of expected items.

        Args:
            receipt_path: Path to the receipt text file

        Returns:
            List of food item names, or None if file doesn't exist/error
        """
        path = Path(receipt_path)

        if not path.exists():
            print(f"Receipt file not found: {receipt_path}")
            return None

        try:
            with open(path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            # Parse each line as a food item (strip whitespace and skip empty lines)
            items = []
            for line in lines:
                item = line.strip()
                if item:  # Skip empty lines
                    items.append(item)

            print(f"Parsed {len(items)} items from receipt: {receipt_path}")
            return items

        except Exception as e:
            print(f"Error parsing receipt {receipt_path}: {e}")
            return None

    @staticmethod
    def to_json(items: List[str]) -> dict:
        """
        Convert items list to JSON format.

        Args:
            items: List of food item names

        Returns:
            Dictionary with expected_items key
        """
        return {"expected_items": items}

    @staticmethod
    def find_receipt_for_video(video_path: str) -> Optional[str]:
        """
        Find the corresponding receipt file for a video.
        Receipt should have the same base name with .txt extension.

        Args:
            video_path: Path to video file (e.g., "20251029-193737.MOV")

        Returns:
            Path to receipt file if found, None otherwise
        """
        video_file = Path(video_path)
        video_stem = video_file.stem  # e.g., "20251029-193737"

        # Look for receipt with same base name
        receipt_path = video_file.parent / f"receipt_{video_stem}.txt"

        if receipt_path.exists():
            print(f"Found receipt for video: {receipt_path}")
            return str(receipt_path)

        return None


if __name__ == "__main__":
    # Test the ReceiptParser module
    print("Testing ReceiptParser module...\n")

    # Test parsing existing receipt
    test_receipt = "receipt_20251029-193737.txt"

    items = ReceiptParser.parse(test_receipt)
    if items:
        print(f"\nParsed items:")
        for i, item in enumerate(items, 1):
            print(f"  {i}. {item}")

        # Convert to JSON
        json_output = ReceiptParser.to_json(items)
        print(f"\nJSON format:")
        print(json.dumps(json_output, indent=2))

    # Test finding receipt for video
    print("\n" + "=" * 60)
    test_video = "20251029-193737.MOV"
    receipt = ReceiptParser.find_receipt_for_video(test_video)
    if receipt:
        print(f"Receipt found for {test_video}: {receipt}")
    else:
        print(f"No receipt found for {test_video}")

    print("\n✓ ReceiptParser module test complete")

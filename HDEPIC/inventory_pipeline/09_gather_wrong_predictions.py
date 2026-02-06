#!/usr/bin/env python3
"""
Gather Wrong Predictions Across All Participants

Collects all wrong VLM predictions from evaluation reports and outputs
a unified JSON file compatible with the visualizer's VLM Results View.

Usage:
    # Gather wrong predictions for qwen model
    python gather_wrong_predictions.py --tag qwen

    # Gather wrong predictions for qwen_low model
    python gather_wrong_predictions.py --tag qwen_low

    # Include close matches (off by 1)
    python gather_wrong_predictions.py --tag qwen --include-close

Inputs:
    outputs/02_inventory/P*/P*_vlm_*_{tag}_results.json

Outputs:
    outputs/02_inventory/wrong_predictions_{tag}.json
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent / "outputs" / "02_inventory"


def find_vlm_results_files(output_dir: Path, tag: str) -> List[Path]:
    """Find all VLM results files matching the given tag."""
    pattern = f"P*/P*_vlm_*_{tag}_results.json"
    files = list(output_dir.glob(pattern))
    # Filter out eval files
    files = [f for f in files if "_eval" not in f.name]
    return sorted(files)


def extract_wrong_predictions(
    vlm_results: Dict[str, Any],
    participant: str,
    source_file: str,
    include_close: bool = False
) -> List[Dict[str, Any]]:
    """
    Extract items with wrong predictions from VLM results.

    Args:
        vlm_results: Loaded VLM results JSON
        participant: Participant ID
        source_file: Source filename for identification
        include_close: If True, also include 'close' matches (off by 1)

    Returns:
        List of items with wrong predictions, augmented with participant info
    """
    wrong_items = []
    items = vlm_results.get("items", [])

    for item in items:
        segments = item.get("segments", [])
        if not segments:
            continue

        # Check if any segment has wrong prediction
        has_wrong = False
        wrong_segments = []

        for seg in segments:
            match = seg.get("match")
            # Skip segments with errors (no VLM response)
            if isinstance(seg.get("error"), str):
                continue

            if match == "wrong":
                has_wrong = True
                wrong_segments.append(seg)
            elif include_close and match == "close":
                has_wrong = True
                wrong_segments.append(seg)

        if has_wrong:
            # Create a copy of the item with participant info
            # Prefix food_name with participant for clarity in visualizer
            original_food_name = item.get("food_name", "unknown")
            wrong_item = {
                **item,
                "participant": participant,
                "source_file": source_file,
                "food_name": f"[{participant}] {original_food_name}",
                "original_food_name": original_food_name,
                # Keep only the wrong segments for display
                "wrong_segments": wrong_segments,
                # Calculate total error
                "total_error": sum(
                    abs(seg.get("error", 0)) if isinstance(seg.get("error"), (int, float)) else 0
                    for seg in wrong_segments
                ),
            }
            wrong_items.append(wrong_item)

    return wrong_items


def main():
    parser = argparse.ArgumentParser(
        description="Gather wrong VLM predictions across all participants"
    )
    parser.add_argument(
        "--tag",
        required=True,
        help="VLM tag to search for (e.g., qwen, qwen_low, gpt4o)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for results"
    )
    parser.add_argument(
        "--include-close",
        default=True,
        help="Include close matches (off by 1) in addition to wrong"
    )
    parser.add_argument(
        "--filter",
        default=None,
        help="Filter files by pattern (e.g., 'qa' to only include *_qa_* files)"
    )
    parser.add_argument(
        "--exclude",
        default=None,
        help="Exclude files by pattern (e.g., 'baseline' to exclude *baseline* files)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print verbose output"
    )

    args = parser.parse_args()

    # Find all VLM results files
    vlm_files = find_vlm_results_files(args.output_dir, args.tag)

    if not vlm_files:
        print(f"ERROR: No VLM results files found for tag '{args.tag}'")
        print(f"       Looking in: {args.output_dir}")
        print(f"       Pattern: P*/P*_vlm_*_{args.tag}_results.json")
        return

    # Apply filter/exclude patterns
    if args.filter:
        vlm_files = [f for f in vlm_files if args.filter in f.name]
    if args.exclude:
        vlm_files = [f for f in vlm_files if args.exclude not in f.name]

    if not vlm_files:
        print(f"ERROR: No files remaining after filter/exclude")
        return

    print(f"Found {len(vlm_files)} VLM results files for tag '{args.tag}':")
    for f in vlm_files:
        print(f"  - {f.relative_to(args.output_dir)}")

    # Collect wrong predictions from all files
    all_wrong_items = []
    stats_by_participant = {}
    stats_by_file = {}

    for vlm_file in vlm_files:
        # Extract participant from filename
        participant = vlm_file.parent.name
        source_file = vlm_file.name

        with open(vlm_file, "r") as f:
            vlm_results = json.load(f)

        wrong_items = extract_wrong_predictions(
            vlm_results, participant, source_file, args.include_close
        )

        all_wrong_items.extend(wrong_items)

        # Track stats per file
        total_items = len(vlm_results.get("items", []))
        file_stats = {
            "total_items": total_items,
            "wrong_items": len(wrong_items),
            "wrong_segments": sum(len(item.get("wrong_segments", [])) for item in wrong_items),
        }
        stats_by_file[source_file] = file_stats

        # Aggregate stats by participant
        if participant not in stats_by_participant:
            stats_by_participant[participant] = {
                "total_items": 0,
                "wrong_items": 0,
                "wrong_segments": 0,
                "files": [],
            }
        stats_by_participant[participant]["total_items"] += total_items
        stats_by_participant[participant]["wrong_items"] += len(wrong_items)
        stats_by_participant[participant]["wrong_segments"] += sum(
            len(item.get("wrong_segments", [])) for item in wrong_items
        )
        stats_by_participant[participant]["files"].append(source_file)

        if args.verbose:
            print(f"\n{participant} ({source_file}):")
            print(f"  Total items: {total_items}")
            print(f"  Wrong predictions: {len(wrong_items)}")
            for item in wrong_items:
                print(f"    - {item['original_food_name']}: error={item['total_error']}")

    # Sort by total error (worst first)
    all_wrong_items.sort(key=lambda x: -x.get("total_error", 0))

    # Create output in visualizer-compatible format
    output_data = {
        "description": f"Wrong predictions for VLM tag '{args.tag}'",
        "tag": args.tag,
        "include_close": args.include_close,
        "total_wrong_items": len(all_wrong_items),
        "total_wrong_segments": sum(
            len(item.get("wrong_segments", [])) for item in all_wrong_items
        ),
        "stats_by_participant": stats_by_participant,
        "stats_by_file": stats_by_file,
        # Items in visualizer format
        "items": all_wrong_items,
    }

    # Save output
    suffix = args.tag
    if args.filter:
        suffix += f"_{args.filter}"
    if args.exclude:
        suffix += f"_no{args.exclude}"
    pred_analysis_dir = args.output_dir / "prediction_analysis"
    pred_analysis_dir.mkdir(parents=True, exist_ok=True)
    output_file = pred_analysis_dir / f"wrong_predictions_{suffix}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Tag: {args.tag}")
    print(f"Include close matches: {args.include_close}")
    print(f"\nParticipant stats:")
    for participant, stats in sorted(stats_by_participant.items()):
        pct = (stats["wrong_items"] / stats["total_items"] * 100) if stats["total_items"] > 0 else 0
        print(f"  {participant}: {stats['wrong_items']}/{stats['total_items']} wrong ({pct:.1f}%)")

    print(f"\nTotal wrong items: {len(all_wrong_items)}")
    print(f"Total wrong segments: {output_data['total_wrong_segments']}")
    print(f"\nOutput saved to: {output_file.name}")

    # Print top 10 worst predictions
    print(f"\n{'='*70}")
    print(f"TOP 10 WORST PREDICTIONS (by error magnitude)")
    print(f"{'='*70}")
    print(f"{'Participant':<6} {'Food Name':<35} {'GT':<6} {'Pred':<6} {'Error':<8}")
    print("-" * 70)
    for item in all_wrong_items[:10]:
        participant = item.get("participant", "?")
        food_name = (item.get("original_food_name") or "")[:34]
        # Get first wrong segment for display
        wrong_segs = item.get("wrong_segments", [])
        if wrong_segs:
            seg = wrong_segs[0]
            gt = seg.get("ground_truth_count", "?")
            pred = seg.get("predicted_count", "?")
            error = seg.get("error", "?")
            print(f"{participant:<6} {food_name:<35} {gt:<6} {pred:<6} {error:<8}")


if __name__ == "__main__":
    main()
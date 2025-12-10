#!/usr/bin/env python3
"""
Validate Narration Groupings

This script performs two validation checks on narration groupings:

1. Coverage Check: Verify all narration lines from participant_P01_narrations.csv
   are included in some semantic group in narration_grouping/*.json

2. Semantic Fit Check: Use Qwen VL to classify if each individual narration
   belongs to its assigned group (i.e., if it truly describes part of the
   grouped semantic action)

Usage:
    python pipelines/validate_narration_groupings.py [--check-coverage] [--check-fit] [--video VIDEO_ID]

    --check-coverage    Only run coverage check
    --check-fit         Only run semantic fit check via Qwen VL
    --video VIDEO_ID    Process only specific video (e.g., P01-20240202-110250)
    --output DIR        Output directory (default: outputs/grouping_validation)
    --dry-run           Don't call Qwen VL, just show what would be checked
"""

import json
import csv
import sys
import argparse
import requests
from pathlib import Path
from typing import Dict, List, Set, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict

# Paths
HDEPIC_DIR = Path(__file__).parent.parent
NARRATIONS_CSV = HDEPIC_DIR / "participant_P01_narrations.csv"
GROUPINGS_DIR = HDEPIC_DIR / "narration_grouping"
OUTPUT_DIR = HDEPIC_DIR / "outputs" / "grouping_validation"

# Qwen3-VL API endpoint
QWEN3VL_URL = "http://saltyfish.eecs.umich.edu:8000/v1/chat/completions"
QWEN_MODEL = "Qwen/Qwen3-VL-30B-A3B-Instruct"


@dataclass
class NarrationLine:
    """Represents a single narration line from the CSV"""
    unique_id: str
    video_id: str
    narration: str
    start_timestamp: float
    end_timestamp: float


@dataclass
class SemanticGroup:
    """Represents a semantic grouping of narrations"""
    query: str  # The grouped action description
    start: float
    end: float
    merged_ids: List[str]


@dataclass
class MisfitResult:
    """Result of a misfit classification"""
    narration_id: str
    narration_text: str
    group_query: str
    belongs_to_group: bool
    confidence: str
    reasoning: str


def load_narrations() -> Dict[str, NarrationLine]:
    """Load all narrations from CSV into a dict keyed by unique_narration_id"""
    narrations = {}
    with open(NARRATIONS_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            narr = NarrationLine(
                unique_id=row['unique_narration_id'],
                video_id=row['video_id'],
                narration=row['narration'].strip(),
                start_timestamp=float(row['start_timestamp']),
                end_timestamp=float(row['end_timestamp'])
            )
            narrations[narr.unique_id] = narr
    return narrations


def load_groupings() -> Dict[str, List[SemanticGroup]]:
    """Load all semantic groupings from JSON files, keyed by video_id"""
    groupings = {}
    for json_file in sorted(GROUPINGS_DIR.glob("*.json")):
        # Extract video_id from filename (e.g., P01-20240202-110250_anonymized.json)
        video_id = json_file.stem.replace("_anonymized", "")

        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        groups = []
        for item in data:
            group = SemanticGroup(
                query=item['query'],
                start=item['start'],
                end=item['end'],
                merged_ids=item['merged_id']
            )
            groups.append(group)
        groupings[video_id] = groups

    return groupings


def check_coverage(narrations: Dict[str, NarrationLine],
                   groupings: Dict[str, List[SemanticGroup]]) -> Dict:
    """Check if all narrations are covered by some semantic group

    Returns:
        Dict with coverage statistics and lists of missing/covered IDs
    """
    # Get all narration IDs grouped by video
    narrations_by_video = defaultdict(set)
    for narr_id, narr in narrations.items():
        narrations_by_video[narr.video_id].add(narr_id)

    # Get all IDs that are in groupings
    grouped_by_video = defaultdict(set)
    for video_id, groups in groupings.items():
        for group in groups:
            for merged_id in group.merged_ids:
                grouped_by_video[video_id].add(merged_id)

    # Compare
    results = {
        'total_narrations': len(narrations),
        'total_grouped': 0,
        'total_missing': 0,
        'per_video': {}
    }

    all_grouped = set()
    all_missing = set()

    # Process videos in sorted order
    for video_id in sorted(set(narrations_by_video.keys()) | set(grouped_by_video.keys())):
        narr_ids = narrations_by_video.get(video_id, set())
        group_ids = grouped_by_video.get(video_id, set())

        covered = narr_ids & group_ids
        missing = narr_ids - group_ids
        extra = group_ids - narr_ids  # IDs in grouping but not in CSV

        all_grouped.update(covered)
        all_missing.update(missing)

        results['per_video'][video_id] = {
            'total_narrations': len(narr_ids),
            'covered': len(covered),
            'missing': len(missing),
            'missing_ids': sorted(list(missing)),
            'extra_in_grouping': sorted(list(extra)) if extra else [],
            'coverage_pct': round(100 * len(covered) / len(narr_ids), 2) if narr_ids else 100
        }

    results['total_grouped'] = len(all_grouped)
    results['total_missing'] = len(all_missing)
    results['overall_coverage_pct'] = round(100 * len(all_grouped) / len(narrations), 2) if narrations else 100
    results['all_missing_ids'] = sorted(list(all_missing))

    return results


def query_qwen(system_prompt: str, user_prompt: str, max_tokens: int = 500) -> str:
    """Query Qwen3-VL (text-only)"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [{"type": "text", "text": user_prompt}]}
    ]

    data = {
        "model": QWEN_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.1,  # Low temperature for more deterministic classification
    }

    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(QWEN3VL_URL, headers=headers, json=data, timeout=60)
        response.raise_for_status()
        result = response.json()
        if "choices" in result and len(result["choices"]) > 0:
            return result["choices"][0]["message"]["content"]
        return ""
    except Exception as e:
        print(f"  ✗ Qwen API Error: {e}")
        return ""


def classify_narration_fit(
    narration: NarrationLine,
    group: SemanticGroup,
    all_narrations_in_group: List[NarrationLine]
) -> MisfitResult:
    """Use Qwen VL to classify if a narration belongs to its semantic group

    Args:
        narration: The single narration to check
        group: The semantic group it belongs to
        all_narrations_in_group: All narrations in this group (for context)

    Returns:
        MisfitResult with classification and reasoning
    """
    # Build prompt with context
    system_prompt = """You are an expert at analyzing action descriptions in egocentric video narrations.
Your task is to determine if a single narration line semantically belongs to a grouped action description.

A narration BELONGS to a group if it describes:
- A sub-action or step that is part of accomplishing the grouped action
- A preparatory action directly needed for the grouped action
- A cleanup/completion action that concludes the grouped action

A narration does NOT belong if:
- It describes an unrelated action (different objects, different goals)
- It's temporally adjacent but semantically independent
- It's a completely separate task that just happens to occur nearby

Respond in JSON format:
{
    "belongs": true/false,
    "confidence": "high/medium/low",
    "reasoning": "Brief explanation of why this narration does or doesn't fit the grouped action"
}"""

    # Format all narrations in the group for context
    group_narrations_text = "\n".join([
        f"  - [{n.start_timestamp:.2f}s] {n.narration}"
        for n in sorted(all_narrations_in_group, key=lambda x: x.start_timestamp)
    ])

    user_prompt = f"""Grouped Action: "{group.query}"
Time range: {group.start}s - {group.end}s

All narrations assigned to this group:
{group_narrations_text}

Target narration to classify:
ID: {narration.unique_id}
Time: {narration.start_timestamp:.2f}s - {narration.end_timestamp:.2f}s
Text: "{narration.narration}"

Does this specific narration belong to the grouped action "{group.query}"?"""

    response = query_qwen(system_prompt, user_prompt)

    # Parse response
    try:
        # Handle potential markdown code blocks
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0]
        elif "```" in response:
            response = response.split("```")[1].split("```")[0]

        result = json.loads(response.strip())
        return MisfitResult(
            narration_id=narration.unique_id,
            narration_text=narration.narration,
            group_query=group.query,
            belongs_to_group=result.get('belongs', True),
            confidence=result.get('confidence', 'unknown'),
            reasoning=result.get('reasoning', 'Failed to parse response')
        )
    except (json.JSONDecodeError, KeyError) as e:
        return MisfitResult(
            narration_id=narration.unique_id,
            narration_text=narration.narration,
            group_query=group.query,
            belongs_to_group=True,  # Default to belongs if parsing fails
            confidence='error',
            reasoning=f"Failed to parse Qwen response: {response[:200]}..."
        )


def check_semantic_fit(
    narrations: Dict[str, NarrationLine],
    groupings: Dict[str, List[SemanticGroup]],
    video_filter: str = None,
    dry_run: bool = False
) -> Dict:
    """Check semantic fit of all narrations in their groups using Qwen VL

    Args:
        narrations: All narrations keyed by ID
        groupings: All semantic groups keyed by video ID
        video_filter: If set, only process this video
        dry_run: If True, don't call Qwen VL

    Returns:
        Dict with fit results and misfits
    """
    results = {
        'total_checked': 0,
        'total_misfits': 0,
        'misfits': [],
        'per_video': {}
    }

    videos_to_process = [video_filter] if video_filter else sorted(groupings.keys())

    for video_id in videos_to_process:
        if video_id not in groupings:
            print(f"  Warning: No groupings found for video {video_id}")
            continue

        print(f"\n[{video_id}] Processing {len(groupings[video_id])} semantic groups...")
        video_results = {
            'total_checked': 0,
            'misfits': []
        }

        for group_idx, group in enumerate(groupings[video_id]):
            # Get all narration objects in this group
            group_narrations = []
            for narr_id in group.merged_ids:
                if narr_id in narrations:
                    group_narrations.append(narrations[narr_id])
                else:
                    print(f"    Warning: Narration ID {narr_id} not found in CSV")

            if not group_narrations:
                continue

            print(f"  [{group_idx+1}/{len(groupings[video_id])}] Group: \"{group.query}\" ({len(group_narrations)} narrations)")

            # Check each narration in the group
            for narr in group_narrations:
                video_results['total_checked'] += 1
                results['total_checked'] += 1

                if dry_run:
                    print(f"    [DRY-RUN] Would check: {narr.unique_id}")
                    continue

                result = classify_narration_fit(narr, group, group_narrations)

                if not result.belongs_to_group:
                    results['total_misfits'] += 1
                    video_results['misfits'].append(asdict(result))
                    results['misfits'].append(asdict(result))
                    print(f"    ✗ MISFIT: {narr.unique_id}")
                    print(f"      Text: {narr.narration[:80]}...")
                    print(f"      Reason: {result.reasoning}")

        results['per_video'][video_id] = video_results
        print(f"  → Checked {video_results['total_checked']} narrations, found {len(video_results['misfits'])} misfits")

    return results


def print_coverage_report(coverage: Dict):
    """Print a formatted coverage report"""
    print("\n" + "="*70)
    print("COVERAGE REPORT")
    print("="*70)
    print(f"Total narrations in CSV: {coverage['total_narrations']}")
    print(f"Total covered by groups: {coverage['total_grouped']}")
    print(f"Total missing from groups: {coverage['total_missing']}")
    print(f"Overall coverage: {coverage['overall_coverage_pct']}%")

    print("\n" + "-"*70)
    print("Per-Video Breakdown:")
    print("-"*70)

    for video_id, stats in sorted(coverage['per_video'].items()):
        status = "✓" if stats['coverage_pct'] == 100 else "✗"
        print(f"  {status} {video_id}: {stats['covered']}/{stats['total_narrations']} ({stats['coverage_pct']}%)")
        if stats['missing_ids']:
            print(f"      Missing IDs: {stats['missing_ids'][:5]}{'...' if len(stats['missing_ids']) > 5 else ''}")
        if stats['extra_in_grouping']:
            print(f"      Extra in grouping (not in CSV): {stats['extra_in_grouping'][:5]}")

    if coverage['all_missing_ids']:
        print("\n" + "-"*70)
        print(f"All Missing IDs ({len(coverage['all_missing_ids'])}):")
        for mid in coverage['all_missing_ids'][:20]:
            print(f"  - {mid}")
        if len(coverage['all_missing_ids']) > 20:
            print(f"  ... and {len(coverage['all_missing_ids']) - 20} more")


def print_misfit_report(fit_results: Dict):
    """Print a formatted misfit report"""
    print("\n" + "="*70)
    print("SEMANTIC FIT REPORT")
    print("="*70)
    print(f"Total narrations checked: {fit_results['total_checked']}")
    print(f"Total misfits found: {fit_results['total_misfits']}")

    if fit_results['misfits']:
        print("\n" + "-"*70)
        print("Misfits Detail:")
        print("-"*70)
        for misfit in fit_results['misfits']:
            print(f"\n  ID: {misfit['narration_id']}")
            print(f"  Text: {misfit['narration_text'][:100]}...")
            print(f"  Group: \"{misfit['group_query']}\"")
            print(f"  Confidence: {misfit['confidence']}")
            print(f"  Reason: {misfit['reasoning']}")


def main():
    parser = argparse.ArgumentParser(description="Validate narration groupings")
    parser.add_argument('--check-coverage', action='store_true',
                       help='Only run coverage check')
    parser.add_argument('--check-fit', action='store_true',
                       help='Only run semantic fit check')
    parser.add_argument('--video', type=str, default=None,
                       help='Process only specific video ID')
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory')
    parser.add_argument('--dry-run', action='store_true',
                       help="Don't call Qwen VL, just show what would be checked")

    args = parser.parse_args()

    # If neither flag is set, run both
    run_coverage = args.check_coverage or not args.check_fit
    run_fit = args.check_fit or not args.check_coverage

    output_dir = Path(args.output) if args.output else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading narrations from CSV...")
    narrations = load_narrations()
    print(f"  Loaded {len(narrations)} narrations")

    print("\nLoading semantic groupings...")
    groupings = load_groupings()
    print(f"  Loaded groupings for {len(groupings)} videos")

    # Coverage check
    if run_coverage:
        print("\n" + "="*70)
        print("CHECKING COVERAGE...")
        print("="*70)
        coverage = check_coverage(narrations, groupings)
        print_coverage_report(coverage)

        # Save coverage results
        coverage_file = output_dir / "coverage_report.json"
        with open(coverage_file, 'w') as f:
            json.dump(coverage, f, indent=2)
        print(f"\nCoverage report saved to: {coverage_file}")

    # Semantic fit check
    if run_fit:
        print("\n" + "="*70)
        print("CHECKING SEMANTIC FIT (via Qwen VL)...")
        print("="*70)

        if args.dry_run:
            print("[DRY-RUN MODE] - No actual Qwen VL calls will be made")

        fit_results = check_semantic_fit(
            narrations,
            groupings,
            video_filter=args.video,
            dry_run=args.dry_run
        )
        print_misfit_report(fit_results)

        # Save fit results
        if not args.dry_run:
            fit_file = output_dir / "semantic_fit_report.json"
            with open(fit_file, 'w') as f:
                json.dump(fit_results, f, indent=2)
            print(f"\nSemantic fit report saved to: {fit_file}")


if __name__ == "__main__":
    main()

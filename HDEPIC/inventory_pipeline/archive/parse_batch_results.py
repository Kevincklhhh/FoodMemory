#!/usr/bin/env python3
"""
parse_batch_results.py - Convert Gemini batch results JSONL into
per-participant vlm_qa_{tag}_results.json files.

Reads the batch results JSONL, parses each VLM response using the same
functions as 07c_vlm_frame.py, resolves timestamps, evaluates against
ground truth, and writes per-participant output files compatible with
the visualizer.

Usage:
    python parse_batch_results.py \
        --results outputs/02_inventory/batches_..._results.jsonl \
        --tag hybrid_gemini3_batch_low \
        --prompt hybrid
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

# Reuse functions from the main pipeline
from importlib.machinery import SourceFileLoader
_vlm_frame = SourceFileLoader(
    "vlm_frame",
    str(Path(__file__).parent / "07c_vlm_frame.py")
).load_module()

_parse_path_evidence = _vlm_frame._parse_path_evidence
_parse_qa_fields = _vlm_frame._parse_qa_fields
resolve_multipath_timestamps = _vlm_frame.resolve_multipath_timestamps
build_evidence_frames = _vlm_frame.build_evidence_frames
evaluate_result = _vlm_frame.evaluate_result
parse_mmss = _vlm_frame.parse_mmss

OUTPUT_DIR = Path(__file__).parent.parent / "outputs" / "02_inventory"


def extract_response_text(result: dict) -> str:
    """Extract text from batch response candidates."""
    text = ''
    for candidate in result.get('response', {}).get('candidates', []):
        for part in candidate.get('content', {}).get('parts', []):
            text += part.get('text', '')
    return text


def parse_vlm_json(text: str) -> dict | None:
    """Parse JSON from VLM response text (handles markdown fences)."""
    # Try markdown code block first
    m = re.search(r'```(?:json)?\s*\n?(.*?)```', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # Try raw JSON
    m = re.search(r'\{.*\}', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass

    return None


def parse_key(key: str) -> dict:
    """Parse the request key back into components."""
    parts = key.split('|')
    return {
        'participant': parts[0],
        'narration_id': parts[1],
        'food_name': parts[2],
        'seg_key': parts[3],  # e.g. "seg0"
        'video_id': parts[4],
        'start_ts': float(parts[5]),
        'end_ts': float(parts[6]),
        'clip_start': float(parts[7]),
    }


def process_batch_results(
    results_path: Path,
    tag: str,
    prompt_mode: str = 'hybrid',
    model_name: str = 'gemini-3-flash',
):
    """Parse batch JSONL results into per-participant result files."""

    require_transfer = (prompt_mode not in ('hybrid_no_transfer',))

    # Load batch results
    print(f"Loading batch results: {results_path}")
    raw_results = []
    with open(results_path) as f:
        for line in f:
            if line.strip():
                raw_results.append(json.loads(line))
    print(f"  {len(raw_results)} responses")

    # Group by participant -> narration_id -> segments
    by_participant = defaultdict(lambda: defaultdict(list))
    errors = 0

    for r in raw_results:
        key = r.get('key', '')
        if 'error' in r and 'response' not in r:
            errors += 1
            print(f"  ERROR in {key[:60]}: {r['error']}")
            continue

        key_info = parse_key(key)
        participant = key_info['participant']
        text = extract_response_text(r)
        if not text:
            errors += 1
            print(f"  EMPTY response for {key[:60]}")
            continue

        by_participant[participant][key_info['narration_id']].append({
            'key_info': key_info,
            'raw_text': text,
        })

    print(f"  Errors: {errors}")
    print(f"  Participants: {sorted(by_participant.keys())}")

    # Process each participant
    for participant in sorted(by_participant.keys()):
        narration_groups = by_participant[participant]

        # Load timeline for ground truth
        timeline_path = OUTPUT_DIR / participant / f"{participant}_timeline_annotated.json"
        if not timeline_path.exists():
            print(f"\n  SKIP {participant}: no timeline file")
            continue

        with open(timeline_path) as f:
            timeline = json.load(f)
        all_items = timeline.get('items', []) if isinstance(timeline, dict) else timeline

        # Build narration_id -> item lookup
        item_lookup = {}
        for item in all_items:
            nid = item.get('narration_id')
            if nid:
                item_lookup[nid] = item

        print(f"\n{'='*60}")
        print(f"PARTICIPANT: {participant}")
        print(f"{'='*60}")

        results = []

        for narr_id, seg_responses in sorted(narration_groups.items()):
            item = item_lookup.get(narr_id)
            if not item:
                print(f"  WARN: No timeline item for {narr_id}")
                continue

            food_name = item.get('food_name', 'unknown')
            difficulty = item.get('difficulty', 'UNKNOWN')
            video_range = item.get('video_range', [])
            segments = item.get('dispensal_segments', [])

            print(f"\n  {food_name} ({len(seg_responses)} responses)")

            segment_results = []

            for resp in sorted(seg_responses, key=lambda x: x['key_info']['seg_key']):
                ki = resp['key_info']
                seg_idx = int(ki['seg_key'].replace('seg', ''))
                clip_start = ki['clip_start']

                # Get segment info from timeline
                seg_info = segments[seg_idx] if seg_idx < len(segments) else {}
                seg_id = seg_info.get('segment_id', f'seg_batch_{seg_idx}')

                # Parse VLM JSON
                vlm_json = parse_vlm_json(resp['raw_text'])
                if not vlm_json:
                    print(f"    Seg {seg_idx}: PARSE FAILED")
                    segment_results.append({
                        'segment_id': seg_id,
                        'segment_idx': seg_idx,
                        'video_id': ki['video_id'],
                        'start_timestamp': ki['start_ts'],
                        'end_timestamp': ki['end_ts'],
                        'clip_start': clip_start,
                        'error': 'Failed to parse VLM response',
                        'raw_vlm_response': resp['raw_text'],
                    })
                    continue

                # Parse path evidence
                path_parsed = _parse_path_evidence(vlm_json, require_transfer=require_transfer)
                qa_fields = _parse_qa_fields(vlm_json)

                # Resolve timestamps (no frame snapping for video-based models)
                if path_parsed:
                    path_parsed = resolve_multipath_timestamps(
                        path_parsed,
                        clip_start=clip_start,
                        frame_timestamps=None,  # video-based, no frame snapping
                    )
                    evidence_frames = build_evidence_frames(path_parsed, prompt_mode)
                else:
                    evidence_frames = []

                # Evaluate against ground truth
                ground_truth = {
                    'total_count': seg_info.get('count'),
                    'count_unit': seg_info.get('count_unit') or item.get('count_unit'),
                }
                eval_result = evaluate_result(qa_fields, ground_truth)

                # Build segment result
                seg_result = {
                    'segment_id': seg_id,
                    'segment_idx': seg_idx,
                    'video_id': ki['video_id'],
                    'start_timestamp': ki['start_ts'],
                    'end_timestamp': ki['end_ts'],
                    'clip_start': clip_start,
                    'item_name': vlm_json.get('item_name', food_name),
                    'evidence_frames': evidence_frames,
                    'raw_vlm_response': resp['raw_text'],
                    'ground_truth_count': ground_truth['total_count'],
                    'ground_truth_unit': ground_truth['count_unit'],
                    **eval_result,
                }

                # Add paths
                if path_parsed:
                    seg_result['paths'] = {
                        'source': path_parsed.get('path_source', {}),
                        'destination': path_parsed.get('path_destination', {}),
                    }
                    if path_parsed.get('path_transfer'):
                        seg_result['paths']['transfer'] = path_parsed['path_transfer']

                match = eval_result.get('match', '?')
                pred = eval_result.get('predicted_count', '?')
                gt = ground_truth['total_count']
                print(f"    Seg {seg_idx}: pred={pred} gt={gt} match={match}")

                segment_results.append(seg_result)

            # Build item result
            item_result = {
                'narration_id': narr_id,
                'food_name': food_name,
                'difficulty': difficulty,
                'video_range': video_range,
                'total_ground_truth': item.get('total_count'),
                'total_ground_truth_unit': item.get('count_unit'),
                'num_segments': len(segments),
                'segments': segment_results,
            }

            # Total predicted across segments
            total_predicted = sum(
                s.get('predicted_count', 0) or 0
                for s in segment_results
                if s.get('predicted_count') is not None
            )
            item_result['total_predicted'] = total_predicted if total_predicted > 0 else None
            item_result['recipe_amount'] = item.get('matched_ingredient_weight')

            results.append(item_result)

        # Write output file
        task_name = {
            'hybrid': 'hybrid_selection',
            'hybrid_no_transfer': 'hybrid_no_transfer_selection',
            'multipath': 'multipath_selection',
        }.get(prompt_mode, prompt_mode)

        output_data = {
            'participant': participant,
            'model': model_name,
            'tag': tag,
            'prompt_mode': prompt_mode,
            'task': task_name,
            'low_only': True,
            'padding': 2.0,
            'fps': None,
            'max_frames': None,
            'total_items': len(results),
            'items': results,
        }

        output_file = OUTPUT_DIR / participant / f"{participant}_vlm_qa_{tag}_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2)

        # Summary
        total_exact = sum(
            1 for item in results
            for seg in item['segments']
            if seg.get('match') == 'exact'
        )
        total_segs = sum(len(item['segments']) for item in results)
        print(f"\n  Saved: {output_file.name}")
        print(f"  Items: {len(results)}, Segments: {total_segs}, Exact: {total_exact}/{total_segs}")


def main():
    parser = argparse.ArgumentParser(description="Parse Gemini batch results into per-participant JSON")
    parser.add_argument('--results', type=str, required=True,
                        help='Path to batch results JSONL file')
    parser.add_argument('--tag', type=str, required=True,
                        help='Tag for output files (e.g., hybrid_gemini3_batch_low)')
    parser.add_argument('--prompt', type=str, default='hybrid',
                        choices=['hybrid', 'hybrid_no_transfer', 'multipath'],
                        help='Prompt mode used for the batch')
    parser.add_argument('--model', type=str, default='gemini-3-flash',
                        help='Model name for metadata')

    args = parser.parse_args()

    process_batch_results(
        results_path=Path(args.results),
        tag=args.tag,
        prompt_mode=args.prompt,
        model_name=args.model,
    )


if __name__ == '__main__':
    main()

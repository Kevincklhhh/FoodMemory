#!/usr/bin/env python3
"""Build a failure-cases JSON (VLMResultsView-compatible) from 3 count eval reports.

For each segment where at least one model gets the count wrong, emits an item
with segment_id, video_id, timestamps, notes, and tags drawn from the
underlying VLM results files.

Usage:
    python build_failure_cases_3model.py
"""

import json
import os
import sys
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent / "outputs" / "02_inventory"

# ── config ──────────────────────────────────────────────────────────────────
EVAL_REPORTS = {
    "qwen": BASE / "eval_reports" / "vlm_qa_hybrid_no_transfer_qwen_count_eval_report.json",
    "gpt5_v2": BASE / "eval_reports" / "vlm_qa_hybrid_no_transfer_gpt5_v2_count_eval_report.json",
    "gemini3": BASE / "eval_reports" / "vlm_qa_hybrid_gemini3_batch_low_count_eval_report.json",
}

FAILURE_CASES_DIR = BASE / "failure_cases"
OUTPUT = FAILURE_CASES_DIR / "failure_cases_hybrid_no_transfer_3model.json"

# ── helpers ─────────────────────────────────────────────────────────────────

def load_timeline_data(participants):
    """Load tags and GT from timeline_annotated files.

    Returns:
        tag_lookup:  (participant, narration_id, segment_idx) -> [tags]
        gt_lookup:   (participant, narration_id, segment_idx) -> {count, count_unit}
        item_lookup:  (participant, narration_id) -> {food_name, difficulty}
    """
    tag_lookup = {}
    gt_lookup = {}
    item_lookup = {}
    for p in participants:
        timeline_path = BASE / p / f"{p}_timeline_annotated.json"
        if not timeline_path.exists():
            continue
        with open(timeline_path) as f:
            data = json.load(f)
        for item in data.get("items", []):
            narration_id = item.get("narration_id", "")
            item_lookup[(p, narration_id)] = {
                "food_name": item.get("food_name"),
                "difficulty": item.get("difficulty"),
            }
            for idx, seg in enumerate(item.get("dispensal_segments", [])):
                key = (p, narration_id, idx)
                tags = seg.get("tags", [])
                if tags:
                    tag_lookup[key] = tags
                gt_lookup[key] = {
                    "count": seg.get("count"),
                    "count_unit": seg.get("count_unit"),
                }
    return tag_lookup, gt_lookup, item_lookup

def load_eval_segments(report_path):
    """Extract per-segment eval results keyed by (participant, narration_id, segment_idx)."""
    with open(report_path) as f:
        report = json.load(f)
    result = {}
    for detail in report["details"]:
        participant = detail["participant"]
        for item in detail["items"]:
            narration_id = item["narration_id"]
            food_name = item["food_name"]
            for seg in item["segments"]:
                key = (participant, narration_id, seg["segment_idx"])
                result[key] = {
                    "food_name": food_name,
                    "ground_truth": seg["ground_truth"],
                    "predicted": seg["predicted"],
                    "is_correct": seg["is_correct"],
                    "unit": seg.get("unit", ""),
                    "skipped": seg.get("skipped", False),
                }
    return result


def load_vlm_segment_lookup(participant, model_tag):
    """Load VLM results for a participant and build (narration_id, segment_idx) -> segment dict."""
    # Try to find the VLM results file
    p_dir = BASE / participant
    pattern = f"{participant}_vlm_qa_hybrid_no_transfer_{model_tag}_results.json"
    # For gemini3, the filename pattern differs
    alt_patterns = [
        f"{participant}_vlm_qa_hybrid_{model_tag}_results.json",
        f"{participant}_vlm_qa_{model_tag}_results.json",
    ]

    vlm_path = p_dir / pattern
    if not vlm_path.exists():
        for alt in alt_patterns:
            vlm_path = p_dir / alt
            if vlm_path.exists():
                break

    if not vlm_path.exists():
        return {}

    with open(vlm_path) as f:
        vlm_data = json.load(f)

    lookup = {}
    for item in vlm_data.get("items", []):
        narration_id = item.get("narration_id")
        if not narration_id:
            continue
        for seg in item.get("segments", []):
            key = (narration_id, seg.get("segment_idx", 0))
            lookup[key] = seg
    return lookup


def find_vlm_segment(participant, narration_id, segment_idx, vlm_lookups):
    """Try to find segment metadata from any model's VLM results."""
    for model_tag, by_participant in vlm_lookups.items():
        lookup = by_participant.get(participant, {})
        seg = lookup.get((narration_id, segment_idx))
        if seg and seg.get("segment_id"):
            return seg
    return None


# ── main ────────────────────────────────────────────────────────────────────

def main():
    models = list(EVAL_REPORTS.keys())

    # Load eval reports
    eval_data = {}
    for name, path in EVAL_REPORTS.items():
        if not path.exists():
            print(f"ERROR: {path} not found", file=sys.stderr)
            sys.exit(1)
        eval_data[name] = load_eval_segments(path)
        print(f"Loaded {len(eval_data[name])} segments from {name}")

    # Collect all keys
    all_keys = set()
    for segs in eval_data.values():
        all_keys.update(segs.keys())
    print(f"Total unique segments: {len(all_keys)}")

    # Identify failure keys
    failure_keys = []
    for key in sorted(all_keys):
        for m in models:
            seg = eval_data[m].get(key)
            if seg and not seg["is_correct"] and not seg.get("skipped", False):
                failure_keys.append(key)
                break
    print(f"Failure cases (>=1 model wrong): {len(failure_keys)}")

    # Collect participants that appear in failure cases
    failure_participants = sorted(set(k[0] for k in failure_keys))
    print(f"Participants with failures: {failure_participants}")

    # Load VLM results for segment metadata (segment_id, video_id, timestamps)
    # We map model config names to the filename tag used in VLM result files
    vlm_file_tags = {
        "qwen": "qwen",
        "gpt5_v2": "gpt5_v2",
        "gemini3": "gemini3_batch_low",
    }

    vlm_lookups = {}  # model -> { participant -> { (narration_id, seg_idx) -> segment } }
    for model_name, file_tag in vlm_file_tags.items():
        vlm_lookups[model_name] = {}
        for p in failure_participants:
            vlm_lookups[model_name][p] = load_vlm_segment_lookup(p, file_tag)
        n = sum(len(v) for v in vlm_lookups[model_name].values())
        print(f"Loaded {n} VLM segments for {model_name}")

    # Load persistent tags and GT from timeline_annotated
    timeline_tags, timeline_gt, timeline_items = load_timeline_data(failure_participants)
    print(f"Loaded persistent tags for {len(timeline_tags)} segments, GT for {len(timeline_gt)} segments from timeline_annotated")

    # Build items
    items = []
    missing_segment_ids = 0

    for key in failure_keys:
        participant, narration_id, seg_idx = key

        # Get eval info per model
        gt = None
        unit = ""
        note_parts = []
        tags = []
        n_correct = 0
        n_wrong = 0
        model_statuses = {}

        # Get GT from timeline_annotated (preferred), fall back to eval report
        tl_gt = timeline_gt.get(key, {})
        if tl_gt.get("count") is not None:
            gt = tl_gt["count"]
            unit = tl_gt.get("count_unit", "")

        for m in models:
            seg = eval_data[m].get(key)
            if seg is None:
                note_parts.append(f"{m}: missing")
                model_statuses[m] = "missing"
            elif seg.get("skipped", False):
                note_parts.append(f"{m}: skipped")
                model_statuses[m] = "skipped"
            else:
                if gt is None:
                    gt = seg["ground_truth"]
                    unit = seg.get("unit", "")
                ok = seg["is_correct"]
                pred = seg["predicted"]
                sym = "\u2713" if ok else "\u2717"
                note_parts.append(f"{m}: {pred} ({sym})")
                model_statuses[m] = "correct" if ok else "wrong"
                if ok:
                    n_correct += 1
                else:
                    n_wrong += 1

        # Tags (summary only — per-model status is already in notes)
        if n_wrong == len(models):
            tags.append("all-wrong")
        if n_correct == 1:
            tags.append("only-1-correct")
        elif n_correct == 2:
            tags.append("2-correct")

        # Merge persistent tags from timeline_annotated
        persistent_tags = timeline_tags.get(key, [])
        for pt in persistent_tags:
            if pt not in tags:
                tags.append(pt)

        notes_str = f"GT={gt} {unit} | " + "; ".join(note_parts)

        # Find segment metadata from VLM results
        vlm_seg = find_vlm_segment(participant, narration_id, seg_idx, vlm_lookups)
        # Get food_name from timeline (preferred), fall back to eval report
        tl_item = timeline_items.get((participant, narration_id), {})
        food_name = tl_item.get("food_name")
        if not food_name:
            food_name = eval_data[models[0]].get(key, {}).get("food_name")
            if not food_name:
                for m in models:
                    fn = eval_data[m].get(key, {}).get("food_name")
                    if fn:
                        food_name = fn
                        break

        segment_data = {
            "segment_idx": seg_idx,
            "ground_truth_count": gt,
            "ground_truth_unit": unit,
            "notes": notes_str,
            "tags": tags,
            "n_models_correct": n_correct,
            "n_models_wrong": n_wrong,
        }

        if vlm_seg:
            segment_data["segment_id"] = vlm_seg.get("segment_id")
            segment_data["video_id"] = vlm_seg.get("video_id")
            segment_data["start_timestamp"] = vlm_seg.get("start_timestamp")
            segment_data["end_timestamp"] = vlm_seg.get("end_timestamp")
        else:
            missing_segment_ids += 1

        items.append({
            "food_name": f"[{participant}] {food_name}",
            "narration_id": narration_id,
            "participant": participant,
            "segments": [segment_data],
        })

    # Write output
    output = {
        "description": "Failure cases: at least one of qwen/gpt5_v2/gemini3 (hybrid no-transfer) gets the count wrong",
        "source_reports": {k: str(v) for k, v in EVAL_REPORTS.items()},
        "total_failure_cases": len(items),
        "items": items,
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nWrote {len(items)} failure cases to {OUTPUT}")
    if missing_segment_ids:
        print(f"WARNING: {missing_segment_ids} segments missing segment_id (no VLM result found)")

    # Tag summary
    tag_counts = Counter()
    for item in items:
        for tag in item["segments"][0]["tags"]:
            tag_counts[tag] += 1
    print("\nTag distribution:")
    for tag, count in tag_counts.most_common():
        print(f"  {tag}: {count}")


if __name__ == "__main__":
    main()

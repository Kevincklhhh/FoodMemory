#!/usr/bin/env python3
"""Evaluate VLM amount-estimation predictions using CNPE metrics.

Ground truth comes directly from ledger.json usage events.
One benchmark question per usage event, grouped by session.

Metrics:
  CNPE_used = |predicted_used - GT_used| / W0 * 100
  CNPE_rem  = |predicted_remaining - GT_remaining| / W0 * 100

W0 = package_amount from ledger items (labeled capacity from receipt).

Usage:
  # Print benchmark questions (ground truth)
  python evaluate_amount.py --participant kailai --print-gt

  # Print VLM prompt for a specific session
  python evaluate_amount.py --participant kailai --session 20260310-195710 --print-prompt

  # Evaluate predictions
  python evaluate_amount.py --participant kailai --predictions preds.json
"""

import argparse
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from tare import compute_tare_weights, correct_value, tare_status
from utils import load_ledger, load_session_inventory, participant_dir

# The HTML visualizer lives next to the annotation tooling, not in system_design.
# Add it to sys.path so we can import its render_html() and auto-generate the
# HTML view alongside every eval JSON write — see write_report() below.
_ANNOTATING_DIR = Path(__file__).resolve().parent.parent / "annotating"
if str(_ANNOTATING_DIR) not in sys.path:
    sys.path.insert(0, str(_ANNOTATING_DIR))

# Items excluded from all evaluations due to corrupted recording (missing video clips).
# Format: (session, instance_id)
CORRUPTED_ITEMS: set[tuple[str, str]] = {
    ("20260320-095416", "kashi_peanut_butter_cereal_20260317"),
    ("20260320-095416", "whole_milk_gallon_20260318"),
}


# ---------------------------------------------------------------------------
# W0 / package amount parsing
# ---------------------------------------------------------------------------

def parse_package_amount(package_amount: str) -> float | None:
    """Extract numeric value from package_amount string.

    Examples: "724g" -> 724, "60 count" -> 60, "3.7 count" -> 3.7
    """
    if not package_amount:
        return None
    m = re.search(r"([\d.]+)", package_amount)
    return float(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# Ground truth extraction
# ---------------------------------------------------------------------------

def load_skip_items(participant: str) -> set[str]:
    """Load instance_ids to skip from participants/<id>/skip_items.json."""
    path = participant_dir(participant) / "skip_items.json"
    if not path.exists():
        return set()
    data = json.loads(path.read_text())
    return set(data.get("skip", []))


def extract_ground_truth(ledger: dict, exclude: set[tuple[str, str]] | None = None,
                         apply_tare: bool = True,
                         skip_instance_ids: set[str] | None = None) -> list[dict]:
    """Extract benchmark questions from ledger usage events.

    Always excludes CORRUPTED_ITEMS (corrupted recordings). Additional
    exclusions can be passed via `exclude` (session, instance_id pairs)
    or `skip_instance_ids` (instance_ids excluded across all sessions).

    When apply_tare=True, subtracts container weight from before and after
    (raw scale readings) for items with known tare. W0 (package_amount) is
    already a net label value and is NOT tared. The `used` field is unchanged
    (tare cancels).

    Returns list of dicts with: instance_id, session, visual_class, unit, W0,
    package_amount, before, after, used, tare, tare_status.
    """
    items = ledger["items"]
    events = ledger["events"]
    all_exclude = CORRUPTED_ITEMS | (exclude or set())
    skip_iids = skip_instance_ids or set()

    tare_map = compute_tare_weights(ledger) if apply_tare else {}

    gt = []
    for event in events:
        if event["type"] != "usage":
            continue

        iid = event["item"]
        session = event["time"]
        if (session, iid) in all_exclude:
            continue
        if iid in skip_iids:
            continue

        item = items[iid]
        tare = tare_map.get(iid, 0.0) if apply_tare else 0.0
        # package_amount is already a net label value (oil only, not bottle+oil),
        # so it should NOT be tared even when before/after are.
        w0 = parse_package_amount(item.get("package_amount", ""))

        entry = {
            "instance_id": iid,
            "session": session,
            "visual_class": item["visual_class"],
            "unit": item["unit"],
            "W0": w0,
            "package_amount": item.get("package_amount", ""),
            "before": correct_value(event.get("before"), tare),
            "after": correct_value(event.get("after"), tare),
            "used": event.get("used"),  # tare cancels — no correction
            "tare": tare,
            "tare_status": tare_status(ledger, iid, tare_map) if apply_tare else "not_applied",
        }
        gt.append(entry)

    return gt


def gt_by_session(gt: list[dict]) -> dict[str, list[dict]]:
    """Group ground truth entries by session."""
    by_sess: dict[str, list[dict]] = {}
    for e in gt:
        by_sess.setdefault(e["session"], []).append(e)
    return by_sess


# ---------------------------------------------------------------------------
# VLM prompt template
# ---------------------------------------------------------------------------

def build_prompt(participant: str, session: str) -> str:
    """Build the VLM prompt for a session.

    The VLM receives:
    - A list of items currently in the kitchen (with unit and W0 package capacity)
    - The session's video clips
    And must output: which items were used, amount used, amount remaining.
    """
    inventory = load_session_inventory(participant, session)
    if not inventory:
        return f"# No inventory data for session {session}"

    item_lines = []
    for inv in inventory:
        unit_label = "grams" if inv["unit"] == "g" else "count"
        pkg = inv.get("package_amount", "")
        iid = inv.get("instance_id", "")
        if pkg:
            item_lines.append(f"- {iid}: \"{inv['visual_class']}\" ({unit_label}, package size: {pkg})")
        else:
            item_lines.append(f"- {iid}: \"{inv['visual_class']}\" ({unit_label})")
    item_list = "\n".join(item_lines)

    return f"""You are analyzing an egocentric kitchen video recorded with smart glasses.

The following food items are currently in the kitchen. Each line is `<instance_id>: "<display name>" (...)`. Multiple lines may share the same display name when there are several physical instances of the same product — they are different physical items and must be reported separately by `instance_id`.

{item_list}

Watch the video carefully. For each instance that was used (taken out, consumed, cooked with, etc.), estimate:
1. The amount used during this session
2. The amount remaining after this session

Output a JSON array. Only include instances that were actually used (amount_used > 0).
Each entry must have exactly these fields:

```json
[
  {{
    "instance_id": "<instance_id exactly as listed above>",
    "item": "<display name exactly as listed above>",
    "amount_used": <number>,
    "amount_remaining": <number>
  }}
]
```

Rules:
- Use grams for weight items and integer count for discrete items
- `instance_id` is REQUIRED and must match one of the IDs above exactly
- Only report instances you actually see being used in the video
- If unsure about exact amounts, give your best estimate"""


# ---------------------------------------------------------------------------
# Prediction matching
# ---------------------------------------------------------------------------

def _score_pair(pred: dict, gt: dict) -> float | None:
    """Charitable match score for (pred, gt): lower is better.

    Prefers CNPE_rem (what we are trying to estimate); falls back to CNPE_used;
    returns None if neither is computable (e.g. missing W0 or both amount fields
    None). None sorts last in the greedy pairing below.
    """
    w0 = gt.get("W0")
    if not w0 or w0 <= 0:
        return None
    pr = pred.get("amount_remaining")
    gt_rem = gt.get("after")
    if pr is not None and gt_rem is not None:
        return compute_cnpe(pr, gt_rem, w0)
    pu = pred.get("amount_used")
    gt_used = gt.get("used")
    if pu is not None and gt_used is not None:
        return compute_cnpe(pu, gt_used, w0)
    return None


def _build_matched_result(gt_entry: dict, pred: dict, collapsed_vc: bool) -> dict:
    """Construct a matched-entry result dict with CNPE/RPE populated."""
    w0 = gt_entry["W0"]
    result = {
        **gt_entry,
        "status": "matched",
        "predicted_used": pred.get("amount_used"),
        "predicted_remaining": pred.get("amount_remaining"),
        "cnpe_used": None,
        "cnpe_rem": None,
        "rpe_used": None,
        "matched_via_vc_collapse": collapsed_vc,
    }
    if collapsed_vc:
        result["matched_pred_instance_id"] = (pred.get("instance_id") or "").strip() or None
    if w0 and w0 > 0:
        if pred.get("amount_used") is not None and gt_entry["used"] is not None:
            result["cnpe_used"] = compute_cnpe(pred["amount_used"], gt_entry["used"], w0)
        if pred.get("amount_remaining") is not None and gt_entry["after"] is not None:
            result["cnpe_rem"] = compute_cnpe(pred["amount_remaining"], gt_entry["after"], w0)
    if pred.get("amount_used") is not None and gt_entry["used"] is not None:
        result["rpe_used"] = compute_rpe(pred["amount_used"], gt_entry["used"])
    return result


def match_predictions_to_gt(
    session_gt: list[dict],
    session_preds: list[dict],
) -> list[dict]:
    """Match VLM predictions to ground truth entries.

    Two-pass strategy:

    **Pass 1 — exact `instance_id` match.** The canonical key.

    **Pass 2 — visual_class-level bipartite matching.** For any residual GT
    and preds that share a visual_class within the session, build all
    (gt, pred) pairs, score by CNPE_rem (fallback CNPE_used), and greedily
    assign lowest-score-first. This absorbs two failure modes the old
    matcher couldn't handle:

    * single GT + multiple preds of the same class (pick the pred that
      scores closest to GT — "charitable" assignment when the pipeline
      can't disambiguate two purchase instances from visual evidence alone)
    * multiple GT + multiple preds (greedy lowest-CNPE assignment)
    * the pre-existing single-candidate fallback is just a special case.

    Residual GT → `missed`; residual preds → `hallucinated`.

    Every matched entry carries `matched_via_vc_collapse: bool` so the
    aggregator can surface how many matches came from visual_class collapse
    vs direct iid.
    """
    gt_by_iid: dict[str, dict] = {entry["instance_id"]: entry for entry in session_gt}

    results: list[dict] = []
    matched_gt_iids: set[str] = set()
    used_pred_ids: set[int] = set()

    # -- Pass 1: direct iid match ------------------------------------------
    for pred in session_preds:
        pred_iid = (pred.get("instance_id") or "").strip()
        if pred_iid and pred_iid in gt_by_iid and pred_iid not in matched_gt_iids:
            gt_entry = gt_by_iid[pred_iid]
            matched_gt_iids.add(pred_iid)
            used_pred_ids.add(id(pred))
            results.append(_build_matched_result(gt_entry, pred, collapsed_vc=False))

    # -- Pass 2: visual_class-level bipartite for the rest -----------------
    rem_gt_by_vc: dict[str, list[dict]] = defaultdict(list)
    for gt in session_gt:
        if gt["instance_id"] not in matched_gt_iids:
            rem_gt_by_vc[gt["visual_class"].lower()].append(gt)

    rem_preds_by_vc: dict[str, list[dict]] = defaultdict(list)
    for pred in session_preds:
        if id(pred) in used_pred_ids:
            continue
        vc = (pred.get("item") or pred.get("visual_class") or "").strip().lower()
        rem_preds_by_vc[vc].append(pred)

    for vc, gts in rem_gt_by_vc.items():
        preds = rem_preds_by_vc.get(vc, [])
        if not preds:
            continue
        # Build all (score, gt, pred) triples; sort ascending by score.
        # None scores are sorted last via a large sentinel.
        triples = []
        for gt in gts:
            for pred in preds:
                triples.append((_score_pair(pred, gt), gt, pred))
        triples.sort(key=lambda t: (float("inf") if t[0] is None else t[0]))

        for score, gt, pred in triples:
            if gt["instance_id"] in matched_gt_iids:
                continue
            if id(pred) in used_pred_ids:
                continue
            matched_gt_iids.add(gt["instance_id"])
            used_pred_ids.add(id(pred))
            results.append(_build_matched_result(gt, pred, collapsed_vc=True))

    # -- Unmatched preds → hallucinated ------------------------------------
    for pred in session_preds:
        if id(pred) in used_pred_ids:
            continue
        pred_iid = (pred.get("instance_id") or "").strip()
        pred_name = (pred.get("item") or pred.get("visual_class") or "").strip()
        results.append({
            "session": session_gt[0]["session"] if session_gt else pred.get("session", "?"),
            "instance_id": pred_iid,
            "visual_class": pred_name,
            "status": "hallucinated",
            "predicted_used": pred.get("amount_used"),
            "predicted_remaining": pred.get("amount_remaining"),
        })

    # -- Unmatched GT → missed ---------------------------------------------
    for entry in session_gt:
        if entry["instance_id"] not in matched_gt_iids:
            results.append({
                **entry,
                "status": "missed",
                "predicted_used": None,
                "predicted_remaining": None,
                "cnpe_used": None,
                "cnpe_rem": None,
                "rpe_used": None,
            })

    return results


# ---------------------------------------------------------------------------
# CNPE computation
# ---------------------------------------------------------------------------

def compute_cnpe(predicted: float, ground_truth: float, w0: float) -> float:
    """Compute Capacity-Normalized Percentage Error."""
    return abs(predicted - ground_truth) / w0 * 100


def compute_rpe(predicted: float, ground_truth: float) -> float | None:
    """Compute Relative Percentage Error: |pred - GT| / GT * 100.

    Returns None when GT is zero (division undefined).
    """
    if ground_truth == 0:
        return None
    return abs(predicted - ground_truth) / ground_truth * 100


# ---------------------------------------------------------------------------
# Report writer (JSON + auto-generated HTML view)
# ---------------------------------------------------------------------------

def write_report(report: dict, eval_path: Path) -> None:
    """Write the eval JSON to `eval_path` AND auto-generate an HTML view next
    to it via annotating/visualize_vlm_amount.py.

    All eval-writing call sites (this script's main, upperbound_amount.py,
    lowerbound_amount.py) should funnel through here so the HTML never drifts
    out of sync with the JSON. Visualizer failures are non-fatal — they print
    a warning but don't abort the calling run.
    """
    eval_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Evaluation saved to {eval_path}")

    try:
        import visualize_vlm_amount  # imported lazily so import errors don't break eval

        # Mirror the visualizer CLI's output-naming rule:
        #   foo_preds_eval.json -> foo_eval.html
        #   foo_eval.json       -> foo_eval.html
        #   anything else       -> <stem>_eval.html
        stem = eval_path.stem
        if stem.endswith("_preds_eval"):
            html_stem = stem[: -len("_preds_eval")] + "_eval"
        elif stem.endswith("_eval"):
            html_stem = stem
        else:
            html_stem = stem + "_eval"
        html_path = eval_path.with_name(html_stem + ".html")

        visualize_vlm_amount.render_html(report, eval_path, html_path)
    except Exception as e:
        print(f"  WARN: HTML visualizer failed: {e}")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(gt_entries: list[dict], predictions: list[dict],
             skip_instance_ids: set[str] | None = None) -> dict:
    """Score predictions against ground truth.

    Predictions format: list of dicts with {session, item, amount_used, amount_remaining}.
    Matching is by session + item name (visual_class).
    """
    gt_sessions = gt_by_session(gt_entries)

    # Group predictions by session, filtering out skipped items
    pred_sessions: dict[str, list[dict]] = {}
    for p in predictions:
        if skip_instance_ids and p.get("instance_id") in skip_instance_ids:
            continue
        pred_sessions.setdefault(p["session"], []).append(p)

    all_results = []
    for session in sorted(gt_sessions.keys()):
        session_preds = pred_sessions.get(session, [])
        session_results = match_predictions_to_gt(gt_sessions[session], session_preds)
        all_results.extend(session_results)

    return aggregate(all_results)


def aggregate(results: list[dict]) -> dict:
    """Compute summary statistics from per-entry results."""

    def stats(values):
        if not values:
            return {"n": 0}
        return {
            "n": len(values),
            "mean": round(statistics.mean(values), 2),
            "median": round(statistics.median(values), 2),
            "std": round(statistics.stdev(values), 2) if len(values) > 1 else 0,
            "min": round(min(values), 2),
            "max": round(max(values), 2),
        }

    matched = [r for r in results if r.get("status") == "matched"]
    missed = [r for r in results if r.get("status") == "missed"]
    hallucinated = [r for r in results if r.get("status") == "hallucinated"]
    matched_vc_collapsed = [r for r in matched if r.get("matched_via_vc_collapse")]

    cnpe_used_all = [r["cnpe_used"] for r in matched if r["cnpe_used"] is not None]
    cnpe_rem_all = [r["cnpe_rem"] for r in matched if r["cnpe_rem"] is not None]
    rpe_used_all = [r["rpe_used"] for r in matched if r.get("rpe_used") is not None]

    # Per-unit breakdown (matched only)
    by_unit = {}
    for unit in ("g", "count"):
        subset = [r for r in matched if r.get("unit") == unit]
        by_unit[unit] = {
            "cnpe_used": stats([r["cnpe_used"] for r in subset if r["cnpe_used"] is not None]),
            "cnpe_rem": stats([r["cnpe_rem"] for r in subset if r["cnpe_rem"] is not None]),
            "rpe_used": stats([r["rpe_used"] for r in subset if r.get("rpe_used") is not None]),
        }

    # Per-item (visual_class) breakdown, restricted to items with >1 usage events.
    # The event-weighted `overall` mean is dominated by high-frequency items
    # (eggs/milk/yogurt). The macro-averaged number below gives each item equal
    # weight so per-item difficulty shows through.
    by_item: dict[str, dict] = {}
    item_means_used: list[float] = []
    item_means_rem: list[float] = []
    item_means_rpe: list[float] = []
    vcs = sorted({r.get("visual_class") for r in matched if r.get("visual_class")})
    for vc in vcs:
        subset = [r for r in matched if r.get("visual_class") == vc]
        if len(subset) <= 1:
            continue
        cu = [r["cnpe_used"] for r in subset if r["cnpe_used"] is not None]
        cr = [r["cnpe_rem"] for r in subset if r["cnpe_rem"] is not None]
        ru = [r["rpe_used"] for r in subset if r.get("rpe_used") is not None]
        by_item[vc] = {
            "n": len(subset),
            "cnpe_used": stats(cu),
            "cnpe_rem": stats(cr),
            "rpe_used": stats(ru),
        }
        if cu:
            item_means_used.append(statistics.mean(cu))
        if cr:
            item_means_rem.append(statistics.mean(cr))
        if ru:
            item_means_rpe.append(statistics.mean(ru))

    overall_macro = {
        "cnpe_used": stats(item_means_used),
        "cnpe_rem": stats(item_means_rem),
        "rpe_used": stats(item_means_rpe),
        "n_items": len(by_item),
    }

    # Per-session breakdown
    sessions = sorted(set(r["session"] for r in results))
    by_session = {}
    for sess in sessions:
        subset = [r for r in results if r["session"] == sess]
        sess_matched = [r for r in subset if r.get("status") == "matched"]
        sess_missed = [r for r in subset if r.get("status") == "missed"]
        sess_halluc = [r for r in subset if r.get("status") == "hallucinated"]
        by_session[sess] = {
            "cnpe_used": stats([r["cnpe_used"] for r in sess_matched if r["cnpe_used"] is not None]),
            "cnpe_rem": stats([r["cnpe_rem"] for r in sess_matched if r["cnpe_rem"] is not None]),
            "rpe_used": stats([r["rpe_used"] for r in sess_matched if r.get("rpe_used") is not None]),
            "matched": len(sess_matched),
            "missed": len(sess_missed),
            "hallucinated": len(sess_halluc),
        }

    return {
        "entries": results,
        "overall": {
            "cnpe_used": stats(cnpe_used_all),
            "cnpe_rem": stats(cnpe_rem_all),
            "rpe_used": stats(rpe_used_all),
        },
        "overall_macro": overall_macro,
        "by_unit": by_unit,
        "by_item": by_item,
        "by_session": by_session,
        "matched": len(matched),
        "missed": len(missed),
        "hallucinated": len(hallucinated),
        "matched_via_vc_collapse": len(matched_vc_collapsed),
        "total_gt": len(matched) + len(missed),
    }


def compute_token_stats(predictions: list[dict]) -> dict:
    """Aggregate token usage and inference time from prediction stats."""
    planner_input = 0
    planner_output = 0
    planner_time = 0.0
    observer_input = 0
    observer_output = 0
    observer_time = 0.0
    observer_frames = 0
    n_observer = 0
    seen_sessions = set()

    for p in predictions:
        s = p.get("stats", {})
        ps = s.get("planner", {})
        os_ = s.get("observer", {})

        # Planner stats are per-session (shared across items), count once
        sess = p.get("session", "")
        if sess not in seen_sessions:
            seen_sessions.add(sess)
            planner_input += ps.get("input_tokens", 0)
            planner_output += ps.get("output_tokens", 0)
            planner_time += ps.get("inference_time_s", 0)

        observer_input += os_.get("input_tokens", 0)
        observer_output += os_.get("output_tokens", 0)
        observer_time += os_.get("inference_time_s", 0)
        observer_frames += os_.get("num_frames", 0)
        n_observer += 1

    total_input = planner_input + observer_input
    total_output = planner_output + observer_output
    total_time = planner_time + observer_time

    return {
        "planner": {
            "input_tokens": planner_input,
            "output_tokens": planner_output,
            "total_tokens": planner_input + planner_output,
            "inference_time_s": round(planner_time, 1),
            "n_sessions": len(seen_sessions),
        },
        "observer": {
            "input_tokens": observer_input,
            "output_tokens": observer_output,
            "total_tokens": observer_input + observer_output,
            "inference_time_s": round(observer_time, 1),
            "total_frames": observer_frames,
            "n_calls": n_observer,
        },
        "total": {
            "input_tokens": total_input,
            "output_tokens": total_output,
            "total_tokens": total_input + total_output,
            "inference_time_s": round(total_time, 1),
        },
    }


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def print_gt_table(gt_entries: list[dict]):
    """Print ground truth as a human-readable table.

    An entry is "scorable" if W0 (package capacity) is known AND at least one
    of `used` or `after` is known — remaining-only annotations are legitimate
    and yield CNPE_rem even without a `used` value.
    """
    scorable = [
        e for e in gt_entries
        if e["W0"] is not None
        and (e["used"] is not None or e["after"] is not None)
    ]
    skipped = len(gt_entries) - len(scorable)

    hdr = f"{'Session':<20s} {'Item':<40s} {'Unit':>5s} {'W0':>8s} {'Before':>8s} {'After':>8s} {'Used':>8s}"
    print(hdr)
    print("-" * len(hdr))

    for e in scorable:
        w0 = f"{e['W0']:.1f}" if e["W0"] is not None else "-"
        before = f"{e['before']:.1f}" if e["before"] is not None else "-"
        after = f"{e['after']:.1f}" if e["after"] is not None else "-"
        used = f"{e['used']:.1f}" if e["used"] is not None else "-"
        print(f"{e['session']:<20s} {e['visual_class']:<40s} {e['unit']:>5s} {w0:>8s} {before:>8s} {after:>8s} {used:>8s}")

    print(f"\n{len(scorable)} scorable entries ({skipped} skipped: missing W0, or both used and after)")


def print_eval_table(report: dict):
    """Print evaluation results as a human-readable table."""
    entries = report["entries"]

    hdr = (
        f"{'Session':<20s} {'Item':<30s} {'Stat':>6s} {'Unit':>5s} "
        f"{'GT_used':>8s} {'Pr_used':>8s} {'CNPE_u':>7s} "
        f"{'GT_rem':>8s} {'Pr_rem':>8s} {'CNPE_r':>7s}"
    )
    print(hdr)
    print("-" * len(hdr))

    def fmt(v):
        return f"{v:.1f}" if v is not None else "-"
    def fmt_pct(v):
        return f"{v:.1f}%" if v is not None else "-"

    for e in entries:
        status = e.get("status", "?")
        marker = {"matched": "  OK", "missed": "MISS", "hallucinated": "HALL"}.get(status, "?")
        unit = e.get("unit", "-")
        print(
            f"{e['session']:<20s} {e.get('visual_class','?'):<30s} {marker:>6s} {unit:>5s} "
            f"{fmt(e.get('used')):>8s} {fmt(e.get('predicted_used')):>8s} {fmt_pct(e.get('cnpe_used')):>7s} "
            f"{fmt(e.get('after')):>8s} {fmt(e.get('predicted_remaining')):>8s} {fmt_pct(e.get('cnpe_rem')):>7s}"
        )

    # Summary
    print(f"\n--- Item Matching ---")
    print(f"  Matched: {report['matched']}/{report['total_gt']} GT items")
    print(f"  Missed:  {report['missed']}")
    print(f"  Hallucinated: {report['hallucinated']}")
    vc_collapsed = report.get("matched_via_vc_collapse", 0)
    if vc_collapsed:
        print(f"  (of matched, {vc_collapsed} via visual_class collapse — "
              f"pred iid differed from GT iid within a class)")

    print(f"\n--- CNPE (matched items only, event-weighted) ---")
    for metric in ("cnpe_used", "cnpe_rem"):
        s = report["overall"][metric]
        if s["n"] > 0:
            print(f"  {metric}: mean={s['mean']:.1f}%, median={s['median']:.1f}%, std={s['std']:.1f}%, n={s['n']}")

    macro = report.get("overall_macro")
    if macro and macro.get("n_items", 0) > 0:
        print(f"\n--- CNPE (macro-averaged: equal weight per item, items with >1 events) ---")
        print(f"  n_items: {macro['n_items']}")
        for metric in ("cnpe_used", "cnpe_rem"):
            s = macro[metric]
            if s["n"] > 0:
                print(f"  {metric}: mean={s['mean']:.1f}%, median={s['median']:.1f}%, std={s['std']:.1f}%, n_items={s['n']}")

    by_item = report.get("by_item") or {}
    if by_item:
        print(f"\n--- Per item (CNPE mean, items with >1 events, sorted by n desc) ---")
        hdr_i = f"  {'n':>3s}  {'CNPE_u':>7s}  {'CNPE_r':>7s}   visual_class"
        print(hdr_i)
        print("  " + "-" * (len(hdr_i) - 2))
        for vc, data in sorted(by_item.items(), key=lambda kv: (-kv[1]["n"], kv[0])):
            cu = data["cnpe_used"]
            cr = data["cnpe_rem"]
            cu_s = f"{cu['mean']:.1f}%" if cu["n"] > 0 else "-"
            cr_s = f"{cr['mean']:.1f}%" if cr["n"] > 0 else "-"
            print(f"  {data['n']:>3d}  {cu_s:>7s}  {cr_s:>7s}   {vc}")

    print(f"\n--- By unit ---")
    for unit in ("g", "count"):
        for metric in ("cnpe_used", "cnpe_rem"):
            s = report["by_unit"][unit][metric]
            if s["n"] > 0:
                print(f"  {unit}/{metric}: mean={s['mean']:.1f}%, median={s['median']:.1f}%, n={s['n']}")

    print(f"\n--- By session ---")
    for sess, data in sorted(report["by_session"].items()):
        m = data["matched"]
        mi = data["missed"]
        h = data["hallucinated"]
        cu = data["cnpe_used"]
        cr = data["cnpe_rem"]
        cu_str = f"CNPE_u={cu['mean']:.1f}%" if cu["n"] > 0 else "CNPE_u=n/a"
        cr_str = f"CNPE_r={cr['mean']:.1f}%" if cr["n"] > 0 else "CNPE_r=n/a"
        print(f"  {sess}: matched={m} missed={mi} hall={h} {cu_str} {cr_str}")

    if "token_stats" in report:
        ts = report["token_stats"]
        print(f"\n--- Token Usage & Inference Time ---")
        pl = ts["planner"]
        ob = ts["observer"]
        tot = ts["total"]
        print(f"  Planner:  {pl['input_tokens']:,} in + {pl['output_tokens']:,} out = {pl['total_tokens']:,} tokens, {pl['inference_time_s']:.1f}s ({pl['n_sessions']} sessions)")
        print(f"  Observer: {ob['input_tokens']:,} in + {ob['output_tokens']:,} out = {ob['total_tokens']:,} tokens, {ob['inference_time_s']:.1f}s ({ob['n_calls']} calls, {ob['total_frames']} frames)")
        print(f"  Total:    {tot['input_tokens']:,} in + {tot['output_tokens']:,} out = {tot['total_tokens']:,} tokens, {tot['inference_time_s']:.1f}s")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate amount estimation using CNPE metrics")
    parser.add_argument("--participant", required=True, help="Participant ID")
    parser.add_argument("--session", help="Specific session (for --print-prompt)")
    parser.add_argument("--predictions", type=Path, help="Path to predictions JSON file")
    parser.add_argument("--output", type=Path, help="Write evaluation report JSON to file")
    parser.add_argument("--print-gt", action="store_true", help="Print ground truth benchmark questions")
    parser.add_argument("--print-gt-json", action="store_true", help="Print ground truth as JSON")
    parser.add_argument("--print-prompt", action="store_true", help="Print VLM prompt for a session")
    parser.add_argument("--skip-items", action="store_true",
                        help="Exclude items listed in participants/<id>/skip_items.json")
    args = parser.parse_args()

    if args.print_prompt:
        if not args.session:
            parser.error("--session is required with --print-prompt")
        print(build_prompt(args.participant, args.session))
        return

    ledger = load_ledger(args.participant)
    skip_iids = load_skip_items(args.participant) if args.skip_items else set()
    if skip_iids:
        print(f"Skipping {len(skip_iids)} items from skip_items.json")
    gt = extract_ground_truth(ledger, skip_instance_ids=skip_iids)

    if args.print_gt:
        print_gt_table(gt)
        return

    if args.print_gt_json:
        scorable = [
            e for e in gt
            if e["W0"] is not None
            and (e["used"] is not None or e["after"] is not None)
        ]
        print(json.dumps(scorable, indent=2))
        return

    if not args.predictions:
        parser.error("--predictions is required unless using --print-gt or --print-prompt")

    predictions = json.loads(args.predictions.read_text())
    report = evaluate(gt, predictions, skip_instance_ids=skip_iids)

    # Attach token/time stats if predictions contain them
    if predictions and "stats" in predictions[0]:
        preds_for_stats = [p for p in predictions
                           if not skip_iids or p.get("instance_id") not in skip_iids]
        report["token_stats"] = compute_token_stats(preds_for_stats)

    print_eval_table(report)

    if args.output:
        eval_path = args.output
    else:
        # Auto-derive: foo_preds.json -> foo_preds_eval.json
        eval_path = args.predictions.with_name(
            args.predictions.stem + "_eval.json"
        )
    write_report(report, eval_path)


if __name__ == "__main__":
    main()

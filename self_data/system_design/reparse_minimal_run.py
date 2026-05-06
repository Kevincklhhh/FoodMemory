"""Re-parse cached sweep raw_responses with the fixed parser, regenerate
per-session predictions and the merged preds.json + status.json + eval, all
without re-billing the API.

Use this after fixing `parse_sweep_response` if the cached responses contain
items the buggy regex missed (e.g. gemini-3.1's double-```json-block edge
case). The candidate list and inventory are read straight out of the per-
session planner.json so the regen is faithful to the original run.

Usage:
  python reparse_minimal_run.py \
      --participant kailai \
      --tag blocks_burst_late_r2 \
      --model gemini-3.1-pro-preview
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import sys
from pathlib import Path


def _load_minimal_module() -> object:
    """Import 06_avp_round1_remaining_minimal.py despite the leading-digit name."""
    here = Path(__file__).resolve().parent
    src = here / "06_avp_round1_remaining_minimal.py"
    spec = importlib.util.spec_from_file_location("_minimal_mod", src)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_minimal_mod"] = mod
    spec.loader.exec_module(mod)
    return mod


def reparse_session(planner_path: Path, parse_sweep_response) -> tuple[dict, list[dict]]:
    """Reparse one session. Returns (updated_session_log, predictions)."""
    blob = json.loads(planner_path.read_text())
    sess_log = blob["session"]
    session_id = sess_log["session"]

    # Re-parse R1 sweep
    sweep = sess_log.get("sweep") or {}
    r1_raw = sweep.get("raw_response") or ""
    r1_items, r1_per_seg = parse_sweep_response(r1_raw)
    sweep["items"] = r1_items
    sweep["per_segment_observations"] = r1_per_seg
    sess_log["sweep"] = sweep
    sweep_by_iid = {it["instance_id"]: it for it in r1_items if it.get("instance_id")}

    # Re-parse R2 sweep if present
    sweep_r2 = sess_log.get("sweep_r2")
    r2_by_iid: dict[str, dict] = {}
    if sweep_r2:
        r2_raw = sweep_r2.get("raw_response") or ""
        r2_items, r2_per_seg = parse_sweep_response(r2_raw)
        sweep_r2["items"] = r2_items
        sweep_r2["per_segment_observations"] = r2_per_seg
        sess_log["sweep_r2"] = sweep_r2
        r2_by_iid = {it["instance_id"]: it for it in r2_items if it.get("instance_id")}

    # Merge: r2 overrides r1 when present, mirroring the original script.
    final_by_iid: dict[str, dict] = dict(sweep_by_iid)
    item_resolved_round: dict[str, int] = {}
    for iid, r2it in r2_by_iid.items():
        merged = dict(sweep_by_iid.get(iid, {}))
        # Only fields the r2 sweep populates take precedence.
        for k in ("status", "amount_starting", "amount_remaining", "amount_derivative"):
            v = r2it.get(k)
            if v is not None:
                merged[k] = v
        if r2it.get("reasoning"):
            merged["reasoning_r2"] = r2it["reasoning"]
        merged["instance_id"] = iid
        merged["visual_class"] = r2it.get("visual_class") or merged.get("visual_class", "")
        final_by_iid[iid] = merged
        item_resolved_round[iid] = 2

    # Reconstruct candidate list from R1 sweep items (the iids the model saw).
    # If empty, fall back to the planner decisions (where status='observe').
    planner = sess_log.get("planner") or {}
    decisions = planner.get("item_decisions") or []
    iid_to_visual = {it["instance_id"]: it.get("visual_class", "") for it in r1_items if it.get("instance_id")}
    for d in decisions:
        iid = d.get("instance_id")
        if iid and iid not in iid_to_visual:
            iid_to_visual[iid] = d.get("visual_class", "")
    # Ordering: prefer the order in R1 sweep_items, then any extras.
    ordered_iids = [it["instance_id"] for it in r1_items if it.get("instance_id")]
    for iid in iid_to_visual:
        if iid not in ordered_iids:
            ordered_iids.append(iid)

    # Build the predictions list using the same rules as process_session.
    predictions: list[dict] = []
    for iid in ordered_iids:
        sw = final_by_iid.get(iid)
        if not sw:
            continue
        status = sw.get("status")
        if status not in {"used", "not_used"}:
            continue
        if status == "not_used":
            continue
        amt_s = sw.get("amount_starting")
        amt_r = sw.get("amount_remaining")
        amt_d = sw.get("amount_derivative")
        if amt_r is not None:
            amount, kind = amt_r, "remaining"
        elif amt_s is not None and amt_d is not None:
            amount, kind = max(0.0, amt_s - amt_d), "computed_remaining"
        elif amt_d is not None:
            amount, kind = amt_d, "derivative"
        elif amt_s is not None:
            amount, kind = amt_s, "starting_only"
        else:
            continue

        round_src = item_resolved_round.get(iid, 1)
        # Stats block — preserve original where available.
        stats_block = {
            "planner": planner.get("stats"),
            "sweep": sweep.get("stats"),
        }
        if sess_log.get("planner_r2"):
            stats_block["planner_r2"] = sess_log["planner_r2"].get("stats")
        if sess_log.get("sweep_r2"):
            stats_block["sweep_r2"] = sess_log["sweep_r2"].get("stats")

        predictions.append({
            "session": session_id,
            "item": iid_to_visual.get(iid, ""),
            "instance_id": iid,
            "amount_starting": amt_s,
            "amount_remaining": amount if kind in {"remaining", "computed_remaining"} else None,
            "amount_derivative": amt_d,
            "amount_remaining_raw": amt_r,
            "amount_kind": kind,
            "status": status,
            "round_source": f"r{round_src}",
            "reasoning": sw.get("reasoning", ""),
            "reasoning_r2": sw.get("reasoning_r2", ""),
            "stats": stats_block,
        })

    return sess_log, predictions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--participant", default="kailai")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--sessions",
                    help="Comma-separated session list. If omitted, reprocess every "
                         "session that has a planner.json under this tag.")
    args = ap.parse_args()

    minimal = _load_minimal_module()
    parse_sweep_response = minimal.parse_sweep_response

    model_tag = args.model.replace("/", "_")
    run_tag = args.tag.replace("/", "_")
    file_stem = f"avp_minimal_remaining_{model_tag}_{run_tag}"

    base = Path("/home/kailaic/NeuroTrace/kitchen/self_data/participants") / args.participant
    out_dir = base / "outputs"

    # Discover sessions with a planner.json under this tag, then optionally filter.
    found = sorted(out_dir.glob(f"*/{file_stem}_planner.json"))
    found_sessions = [p.parent.name for p in found]
    if args.sessions:
        wanted = {s.strip() for s in args.sessions.split(",") if s.strip()}
        found = [p for p, s in zip(found, found_sessions) if s in wanted]
    print(f"Reparsing {len(found)} session(s) under tag '{args.tag}'")

    all_logs: list[dict] = []
    all_preds: list[dict] = []
    for planner_path in found:
        session = planner_path.parent.name
        sess_log, preds = reparse_session(planner_path, parse_sweep_response)
        all_logs.append(sess_log)
        all_preds.extend(preds)

        # Rewrite per-session planner.json (with re-parsed items)
        full_blob = json.loads(planner_path.read_text())
        full_blob["session"] = sess_log
        planner_path.write_text(json.dumps(full_blob, indent=2, ensure_ascii=False) + "\n")

        # Rewrite per-session preds.json
        per_sess_preds_path = planner_path.with_name(f"{file_stem}_preds.json")
        per_sess_preds_path.write_text(json.dumps(preds, indent=2, ensure_ascii=False) + "\n")

        print(f"  {session}: {len(preds)} preds")

    # Rewrite merged planner + preds + status (top-level outputs/ files)
    merged_planner = out_dir / f"{file_stem}_planner.json"
    merged_planner.write_text(json.dumps({
        "participant": args.participant,
        "model": args.model,
        "tag": args.tag,
        "sessions": all_logs,
    }, indent=2, ensure_ascii=False) + "\n")

    merged_preds = out_dir / f"{file_stem}_preds.json"
    merged_preds.write_text(json.dumps(all_preds, indent=2, ensure_ascii=False) + "\n")

    status_path = out_dir / f"{file_stem}_status.json"
    if status_path.exists():
        status = json.loads(status_path.read_text())
    else:
        status = {}
    status["completed_sessions"] = sorted({sl["session"] for sl in all_logs})
    status["reparsed"] = True
    status_path.write_text(json.dumps(status, indent=2, ensure_ascii=False) + "\n")

    # Eval
    from importlib import import_module
    ev = import_module("evaluate_amount")
    ledger_path = base / "ledger.json"
    ledger = json.loads(ledger_path.read_text())
    gt = ev.extract_ground_truth(ledger)
    report = ev.evaluate(gt, all_preds)
    eval_path = out_dir / f"{file_stem}_preds_eval.json"
    ev.write_report(report, eval_path)
    print(f"\nMerged: {merged_preds}")
    print(f"Eval:   {eval_path}")
    print(f"Total predictions: {len(all_preds)}")


if __name__ == "__main__":
    main()

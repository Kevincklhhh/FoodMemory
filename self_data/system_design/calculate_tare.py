#!/usr/bin/env python3
"""calculate_tare.py - Suggest tare_weight values from depletion events.

This is the old auto-detection logic that used to live in `tare.py`. It's now
a standalone audit tool: it computes a candidate tare value for each item from
the events log and reports the diff against the static `items[iid].tare_weight`
field in `ledger.json`.

Usage:
    python calculate_tare.py --participant kailai
    python calculate_tare.py --participant kailai --apply   # write detected values back

Heuristic (priority order):
  1. items[iid].tare_weight (manual override) — never overwritten
  2. In-session depletion: a `depletion` event whose time matches the last
     `usage` event's time → that usage event's `after` field IS the tare
     (the empty container weight on the scale).

Items that have a known limitation (e.g. transferred to a different container
before depletion) should be left unset in the ledger or set manually; the
detection from depletion events will be flagged as a difference here so you
can review and decide.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running this script directly from system_design/
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import participant_dir


def detect_tare_from_events(ledger: dict) -> dict[str, float]:
    """Detect tare weight from in-session depletion events.

    Returns {instance_id: detected_tare_g}. Only includes gram items where
    a depletion event time matches the last usage event time and after > 0.
    """
    items = ledger["items"]
    events = ledger.get("events", [])

    depletions: dict[str, str] = {}
    for ev in events:
        if ev.get("type") == "depletion":
            depletions[ev["item"]] = ev["time"]

    last_usage: dict[str, dict] = {}
    for ev in events:
        if ev.get("type") == "usage":
            last_usage[ev["item"]] = ev

    detected: dict[str, float] = {}
    for iid, meta in items.items():
        if meta.get("unit") != "g":
            continue
        if iid not in depletions or iid not in last_usage:
            continue
        if last_usage[iid]["time"] != depletions[iid]:
            continue  # between-session depletion → unreliable
        after = last_usage[iid].get("after", 0)
        if after and after > 0:
            detected[iid] = float(after)
    return detected


def diff_tare(ledger: dict, detected: dict[str, float]) -> list[dict]:
    """Compare detected values against ledger items[iid].tare_weight.

    Returns one row per item that has either a recorded value, a detected
    value, or both. Each row includes:
      - iid, recorded, detected, status
    where status is one of:
      - 'match'        : recorded == detected (rounded to 1g)
      - 'differ'       : both set, values differ
      - 'recorded_only': recorded set, no detection (kept as-is by ledger)
      - 'detected_only': detection found new value, ledger has none
    """
    items = ledger["items"]
    rows = []
    iids = set(detected) | {iid for iid, m in items.items()
                            if m.get("tare_weight") is not None and m.get("unit") == "g"}
    for iid in sorted(iids):
        meta = items.get(iid, {})
        recorded = meta.get("tare_weight")
        det = detected.get(iid)
        if recorded is None and det is None:
            continue
        if recorded is not None and det is not None:
            status = "match" if abs(float(recorded) - det) < 0.5 else "differ"
        elif recorded is not None:
            status = "recorded_only"
        else:
            status = "detected_only"
        rows.append({
            "iid": iid,
            "recorded": float(recorded) if recorded is not None else None,
            "detected": det,
            "status": status,
            "visual_class": meta.get("visual_class", iid),
            "package_amount": meta.get("package_amount", ""),
        })
    return rows


def print_report(rows: list[dict]) -> None:
    """Print the diff report grouped by status."""
    print()
    print(f"{'STATUS':<14s} {'INSTANCE_ID':<42s} {'RECORDED':>10s} {'DETECTED':>10s}  PACKAGE")
    print("-" * 110)
    by_status = {}
    for r in rows:
        by_status.setdefault(r["status"], []).append(r)
    order = ["differ", "detected_only", "recorded_only", "match"]
    for status in order:
        for r in by_status.get(status, []):
            rec = f"{r['recorded']:.0f}" if r["recorded"] is not None else "—"
            det = f"{r['detected']:.0f}" if r["detected"] is not None else "—"
            print(f"{status:<14s} {r['iid']:<42s} {rec:>10s} {det:>10s}  {r['package_amount']}")
    print()
    counts = {s: len(by_status.get(s, [])) for s in order}
    print(f"Summary: {counts['match']} match, {counts['differ']} differ, "
          f"{counts['detected_only']} detected_only, {counts['recorded_only']} recorded_only")


def apply_changes(ledger_path: Path, ledger: dict, rows: list[dict]) -> int:
    """Apply detected_only and differ rows: set items[iid].tare_weight = detected.

    Returns the number of items modified. Skips 'match' and 'recorded_only'.
    """
    n = 0
    for r in rows:
        if r["status"] in ("detected_only", "differ"):
            ledger["items"][r["iid"]]["tare_weight"] = r["detected"]
            n += 1
    if n > 0:
        ledger_path.write_text(json.dumps(ledger, indent=2) + "\n")
    return n


def main():
    parser = argparse.ArgumentParser(
        description="Audit detected vs recorded tare weights in ledger.json")
    parser.add_argument("--participant", required=True)
    parser.add_argument("--apply", action="store_true",
                        help="Overwrite items[iid].tare_weight with detected values "
                             "for 'differ' and 'detected_only' rows")
    args = parser.parse_args()

    ledger_path = participant_dir(args.participant) / "ledger.json"
    ledger = json.loads(ledger_path.read_text())

    detected = detect_tare_from_events(ledger)
    rows = diff_tare(ledger, detected)
    print_report(rows)

    if args.apply:
        n = apply_changes(ledger_path, ledger, rows)
        print(f"\nApplied {n} change(s) to {ledger_path}")
    else:
        differs = [r for r in rows if r["status"] in ("differ", "detected_only")]
        if differs:
            print(f"\n{len(differs)} item(s) would be changed by --apply.")
        else:
            print("\nNo changes needed.")


if __name__ == "__main__":
    main()

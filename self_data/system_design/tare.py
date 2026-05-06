"""tare.py - Container weight (tare) correction for scale measurements.

Tare values are stored as a static field on each item in `ledger.json`:
    items[iid]["tare_weight"] = <float grams>  (or null if not set)

This module just reads that field. To recompute tare from depletion-event
heuristics and compare against the ledger, use `calculate_tare.py`.

Why a static field instead of auto-detection: depletion-event auto-detection
breaks for items that get transferred to a different container before being
finished (e.g. tofu placed in a glass storage box — the "remaining" reading
at depletion is the new container, not the original packaging). The static
field gives the user a single place to override these cases.
"""

from __future__ import annotations

from typing import Optional


def compute_tare_weights(ledger: dict) -> dict[str, float]:
    """Return {instance_id: tare_weight} from items[iid].tare_weight.

    Only includes gram-based items with a positive tare_weight value.
    Items where tare_weight is null/missing/0 are excluded.
    """
    out: dict[str, float] = {}
    for iid, meta in ledger.get("items", {}).items():
        if meta.get("unit") != "g":
            continue
        t = meta.get("tare_weight")
        if t is None:
            continue
        try:
            t = float(t)
        except (TypeError, ValueError):
            continue
        if t > 0:
            out[iid] = t
    return out


def correct_value(value: Optional[float], tare: float) -> Optional[float]:
    """Subtract tare from a measurement, clamping to 0."""
    if value is None:
        return None
    return max(0.0, value - tare)


def tare_status(ledger: dict, instance_id: str,
                tare_map: dict[str, float] | None = None) -> str:
    """Return tare correction status for an item.

    Returns:
        "manual"          - tare_weight set in ledger and applied
        "not_applicable"  - unit is count (no tare to apply)
        "not_set"         - gram-based, no tare_weight in ledger
    """
    meta = ledger["items"].get(instance_id, {})
    if meta.get("unit") != "g":
        return "not_applicable"

    if tare_map is None:
        tare_map = compute_tare_weights(ledger)
    if instance_id in tare_map:
        return "manual"
    return "not_set"

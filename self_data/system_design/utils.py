#!/usr/bin/env python3
"""
utils.py - Shared helpers for self_data kitchen pipeline scripts.

Provides common data loading and path resolution functions used by
01_extract_and_detect_hands.py, 02_siglip_food_matching.py, etc.
"""

import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_SCRIPT_DIR = Path(__file__).parent
_SELF_DATA = _SCRIPT_DIR.parent
_KITCHEN_DIR = _SELF_DATA.parent


def participant_dir(participant: str) -> Path:
    return _SELF_DATA / "participants" / participant


def load_ledger(participant: str) -> dict:
    """Load ledger.json for participant."""
    ledger_path = participant_dir(participant) / "ledger.json"
    if not ledger_path.exists():
        raise FileNotFoundError(f"Ledger not found: {ledger_path}")
    return json.loads(ledger_path.read_text())


def load_food_items(participant: str) -> List[str]:
    """Get sorted unique food item visual_class names from ledger.json."""
    ledger = load_ledger(participant)
    return sorted(set(item["visual_class"] for item in ledger["items"].values()))


def _session_item_ids(ledger: dict, session: str) -> List[str]:
    """Return instance_ids present during a session (from snapshot). Empty = all."""
    snap = ledger.get("snapshots", {}).get(session)
    if not snap:
        return list(ledger.get("items", {}).keys())
    present = []
    for iid, state in snap.items():
        starting = state.get("starting") or state.get("last_remaining") or 0
        if starting is None:
            starting = 0
        used = state.get("used") or 0
        if used is None:
            used = 0
        if starting > 0 or used > 0:
            present.append(iid)
    return present if present else list(ledger.get("items", {}).keys())


def load_session_food_items(participant: str, session: str) -> List[str]:
    """Get food items present during a specific session (from ledger snapshots).

    Falls back to all items if session has no snapshot.
    """
    ledger = load_ledger(participant)
    iids = _session_item_ids(ledger, session)
    names = sorted(set(
        ledger["items"][iid]["visual_class"]
        for iid in iids if iid in ledger["items"]
    ))
    return names if names else load_food_items(participant)


def _full_item_ids(ledger: dict, session: str) -> List[str]:
    """Return instance_ids that are in stock at the start of a session.

    "In stock" = item has at least one `purchase` event before `session` AND no
    `depletion` event before `session`. This replays the ledger's event log and
    covers ALL items in the kitchen at that time, not just the GT-annotated
    subset for that particular cooking session.
    """
    purchased: set[str] = set()
    depleted: set[str] = set()
    for event in ledger.get("events", []):
        if event["time"] >= session:
            break
        iid = event.get("item")
        if not iid:
            continue
        if event["type"] == "purchase":
            purchased.add(iid)
        elif event["type"] == "depletion":
            depleted.add(iid)
    in_stock = purchased - depleted
    # Preserve any items with just a snapshot entry (safety net for legacy data)
    snap = ledger.get("snapshots", {}).get(session, {})
    in_stock.update(iid for iid in snap.keys() if iid in ledger.get("items", {}))
    return sorted(in_stock)


def _resolve_item_ids(ledger: dict, session: str, scope: str) -> List[str]:
    """Dispatch scope -> instance_id list. scope in {"full", "session"}."""
    if scope == "full":
        ids = _full_item_ids(ledger, session)
        return ids if ids else _session_item_ids(ledger, session)
    if scope == "session":
        return _session_item_ids(ledger, session)
    raise ValueError(f"unknown inventory scope: {scope!r} (expected 'full' or 'session')")


def load_session_siglip_labels(participant: str, session: str,
                               scope: str = "full") -> List[dict]:
    """Get SigLIP label config for items relevant to a session.

    scope:
      "full"    — every item in stock at session time (purchase/depletion replay).
                  Default for realistic runtime behavior.
      "session" — only items in the session's snapshot with starting > 0
                  (the GT-annotated subset).

    Returns list of dicts:
        {
            "display_name": str,     # visual_class — used in output/UI
            "labels": List[str],     # text strings for SigLIP embedding
        }

    If an item has "siglip_labels" in ledger, those are used as labels.
    Otherwise, visual_class is used (with _expand_name handling "food (variant)" syntax).
    Multiple instance_ids with the same visual_class are deduplicated.
    """
    ledger = load_ledger(participant)
    iids = _resolve_item_ids(ledger, session, scope)

    seen = {}  # visual_class -> labels
    for iid in iids:
        item = ledger["items"].get(iid)
        if not item:
            continue
        vc = item["visual_class"]
        if vc in seen:
            # Merge siglip_labels from multiple instances of same visual_class
            existing = seen[vc]
            extra = item.get("siglip_labels", [])
            for lbl in extra:
                if lbl not in existing:
                    existing.append(lbl)
            continue
        custom = item.get("siglip_labels")
        if custom:
            seen[vc] = list(custom)
        else:
            seen[vc] = [vc]

    result = [{"display_name": vc, "labels": lbls} for vc, lbls in sorted(seen.items())]
    return result


def load_session_inventory(participant: str, session: str,
                           apply_tare: bool = True) -> List[dict]:
    """Get full inventory state at the start of a session.

    Returns list of dicts with: instance_id, visual_class, unit, package_amount,
    starting_amount (amount at start of session).

    When apply_tare=True, subtracts container weight from starting_amount
    for items with known tare (see tare.py).

    Only includes items that are present (starting > 0) at this session.
    """
    from tare import compute_tare_weights, correct_value

    ledger = load_ledger(participant)
    tare_map = compute_tare_weights(ledger) if apply_tare else {}
    snap = ledger.get("snapshots", {}).get(session)
    if not snap:
        return []
    inventory = []
    for iid, state in snap.items():
        starting = state.get("starting")
        if starting is None or starting <= 0:
            continue
        item = ledger["items"].get(iid, {})
        tare = tare_map.get(iid, 0.0) if apply_tare else 0.0
        corrected = correct_value(starting, tare)
        inventory.append({
            "instance_id": iid,
            "visual_class": item.get("visual_class", iid),
            "unit": item.get("unit", "g"),
            "package_amount": item.get("package_amount", ""),
            "starting_amount": corrected,
            "visible_during_interaction": item.get("visible_during_interaction", True),
        })
    return inventory


def load_full_inventory(participant: str, session: str) -> List[dict]:
    """Get ALL non-depleted items present in the kitchen at the start of a session.

    Tracks presence by event replay: an item is "in stock" if it has at least
    one `purchase` event before `session` and no `depletion` event before
    `session`. We deliberately do NOT compute remaining grams from
    package_amount — package_amount is a free-form reference label, not a
    quantity to do math on.

    Returns list of dicts with: instance_id, visual_class, unit, package_amount.
    """
    ledger = load_ledger(participant)
    items = ledger["items"]
    in_stock = _full_item_ids(ledger, session)
    result = []
    for iid in in_stock:
        item = items.get(iid, {})
        result.append({
            "instance_id": iid,
            "visual_class": item.get("visual_class", iid),
            "unit": item.get("unit", "g"),
            "package_amount": item.get("package_amount", ""),
        })
    return result


def load_inventory(participant: str, session: str,
                   scope: str = "full",
                   apply_tare: bool = True) -> List[dict]:
    """Unified inventory loader for planner prompts and detection targets.

    scope:
      "full"    — all items in stock at session time (purchase/depletion replay).
                  Default. Matches what a real home-deployment would see.
      "session" — only items listed in the session snapshot with starting > 0
                  (the GT-annotated subset used for per-session evaluation).

    Returns list of dicts with: instance_id, visual_class, unit, package_amount,
    starting_amount (None when item is not in the session snapshot),
    visible_during_interaction, in_session_snapshot (bool).

    When apply_tare=True and starting_amount is available, the snapshot value
    is tare-corrected (container weight subtracted) the same way
    `load_session_inventory` corrects it.
    """
    from tare import compute_tare_weights, correct_value

    ledger = load_ledger(participant)
    items = ledger["items"]
    snap = ledger.get("snapshots", {}).get(session, {}) or {}
    tare_map = compute_tare_weights(ledger) if apply_tare else {}
    iids = _resolve_item_ids(ledger, session, scope)

    inventory = []
    for iid in iids:
        item = items.get(iid, {})
        if not item:
            continue
        state = snap.get(iid) or {}
        starting_raw = state.get("starting")
        tare = tare_map.get(iid, 0.0) if apply_tare else 0.0
        starting_amount = None
        if starting_raw is not None and starting_raw > 0:
            starting_amount = correct_value(starting_raw, tare)
        elif scope == "session":
            # In session scope we rely on snapshot; skip items without a valid starting.
            continue
        inventory.append({
            "instance_id": iid,
            "visual_class": item.get("visual_class", iid),
            "unit": item.get("unit", "g"),
            "package_amount": item.get("package_amount", ""),
            "starting_amount": starting_amount,
            "visible_during_interaction": item.get("visible_during_interaction", True),
            "in_session_snapshot": iid in snap,
        })
    inventory.sort(key=lambda x: x["visual_class"].lower())
    return inventory


def instance_id_to_visual_class(ledger: dict) -> Dict[str, str]:
    """Return {instance_id: visual_class} mapping from ledger."""
    return {iid: item["visual_class"] for iid, item in ledger["items"].items()}


def load_actions(participant: str, session: str) -> List[dict]:
    """Load actions.json from annotations/{session}/."""
    path = participant_dir(participant) / "annotations" / session / "actions.json"
    if not path.exists():
        raise FileNotFoundError(f"actions.json not found: {path}")
    return json.loads(path.read_text())


def get_video_duration(video_path: Path) -> float:
    """Return video duration in seconds via ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return float(result.stdout.strip())
    except Exception:
        return 0.0


def get_session_clips(participant: str, session: str) -> List[Tuple[str, Path, float]]:
    """Return sorted [(filename, path, duration)] for a session's .mp4 clips."""
    session_dir = participant_dir(participant) / "videos" / session
    if not session_dir.exists():
        return []
    clips = []
    for mp4 in sorted(session_dir.glob("*.mp4")):
        dur = get_video_duration(mp4)
        if dur > 0:
            clips.append((mp4.name, mp4, dur))
    return clips


def get_sessions(participant: str) -> List[str]:
    """Return sorted list of session IDs for a participant."""
    videos_dir = participant_dir(participant) / "videos"
    if not videos_dir.exists():
        return []
    return sorted(d.name for d in videos_dir.iterdir() if d.is_dir())


def outputs_dir(participant: str, session: str) -> Path:
    """Return participants/{P}/outputs/{session}/."""
    return participant_dir(participant) / "outputs" / session


def hands23_dir(participant: str, session: str) -> Path:
    """Return path to hands23_detection output directory."""
    return outputs_dir(participant, session) / "hands23_detection"


def interact_dir(participant: str, session: str) -> Path:
    """Return path to interaction_detect output directory."""
    return outputs_dir(participant, session) / "interaction_detect"

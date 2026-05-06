#!/usr/bin/env python3
"""AVP Round 1 (remaining-only, CandList_HOI_PerItem): per-item evidence + journey-aware sampling.

Branch of 06_avp_round1_remaining_CandList_HOI.py with two core changes:

1. **Per-item evidence format.** The planner sees detections grouped by
   visual_class (not by timestamp). Each inventory class with ≥1
   detection gets a chronological list of sightings across the
   session, annotated with scene, hand, grasp, obj_touch, DINO/SigLIP
   scores, and the other hand's contents at the same moment. This makes
   it easy for the planner to narrate an item's journey: retrieval →
   transit → dispensing → return.

2. **Journey-aware observation plan.** For each observed item the
   planner emits TWO kinds of samples:
     - `journey_samples`: individual timestamps across the journey
       (retrieval, staging, leaving-package, return). Sparse context.
     - `dense_windows`: short continuous windows (≤ 10 s) centred on
       moments the item leaves the original package. Dense zoom-in.
   The observer sees frames from both and estimates the remaining
   amount with full journey context.

Per-item section format:
    ### "Whole Milk Gallon" (iids: whole_milk_gallon_20260310; [opaque]; pkg=3780g)
       12.0s  storage   L[Pow-Pris container_held]  dino=0.41
       45.2s  stove     R[Pre-Pris container_held]  dino=0.38 sig=0.22   (other: L="Ceramic Bowl"=0.22)
      180.5s  storage   L[Pow-Pris container_held]  dino=0.40

Usage:
  python system_design/06_avp_round1_remaining_CandList_HOI_PerItem.py \
      --participant kailai --tag PerItem_v1
  python system_design/06_avp_round1_remaining_CandList_HOI_PerItem.py \
      --participant kailai --session 20260329-000145 --tag PerItem_v1
"""

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))
from frame_sampling import (
    extract_segments_frames as _extract_segments_frames,
)
from utils import (
    get_session_clips,
    get_sessions,
    hands23_dir,
    load_inventory,
    load_ledger,
    outputs_dir,
    participant_dir,
)

_KITCHEN_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(_KITCHEN_DIR / ".env")

MAX_IMAGES = 50

CACHE_DIR = Path(__file__).resolve().parent.parent / "cache" / "avp_remaining_CandList_HOI_PerItem"

QWEN_URL = "http://saltyfish.eecs.umich.edu:8000/v1/chat/completions"
QWEN_MODEL_DEFAULT = "Qwen/Qwen3-VL-30B-A3B-Instruct"

VLLM_ENDPOINTS = {
    "qwen": ("http://saltyfish.eecs.umich.edu:8000/v1/chat/completions", "Qwen/Qwen3-VL-30B-A3B-Instruct"),
    "gemma": ("http://saltyfish.eecs.umich.edu:8000/v1/chat/completions", "google/gemma-4-31B-it"),
}

OUTPUT_PREFIX = "avp_CandList_HOI_PerItem_remaining"


# ---------------------------------------------------------------------------
# Clients
# ---------------------------------------------------------------------------

def make_client(model: str = "gpt-5.4"):
    from openai import AzureOpenAI
    api_key = os.getenv("AZURE_OPENAI_API_KEY_2") or os.getenv("AZURE_OPENAI_API_KEY")
    endpoint = (
        os.getenv("AZURE_OPENAI_ENDPOINT_2")
        or os.getenv("AZURE_OPENAI_ENDPOINT")
        or ""
    ).strip()
    if not api_key or not endpoint:
        raise ValueError("Missing Azure OpenAI API credentials")
    return AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version="2025-03-01-preview",
    )


# ---------------------------------------------------------------------------
# Per-frame data loading (bypasses 05b)
# ---------------------------------------------------------------------------

def load_hoi_timestamps(participant: str, session: str) -> tuple[list[float], set[float]]:
    """Return (sorted_all_timestamps, set_of_hoi_contact_timestamps)."""
    path = hands23_dir(participant, session) / f"{participant}_{session}_hands23_results.json"
    d = json.loads(path.read_text())
    all_ts: set[float] = set()
    hoi_ts: set[float] = set()
    for v in d["videos"]:
        for f in v["frames"]:
            t = round(f["session_timestamp_s"], 2)
            all_ts.add(t)
            for det in f["detections"]:
                if det.get("contact_state") == "object_contact":
                    hoi_ts.add(t)
                    break
    return sorted(all_ts), hoi_ts


def load_siglip_by_t(participant: str, session: str) -> dict[float, dict[str, dict[str, float]]]:
    """timestamp -> {hand_side: {visual_class: max_similarity}} (HOI-contact only)."""
    path = hands23_dir(participant, session) / f"{participant}_{session}_siglip_matches.json"
    if not path.exists():
        return {}
    d = json.loads(path.read_text())
    out: dict[float, dict[str, dict[str, float]]] = {}
    for v in d["videos"]:
        for m in v["matches"]:
            if m.get("contact_state") != "object_contact":
                continue
            t = round(m["timestamp"], 2)
            hs = m.get("hand_side") or "unknown"
            per_hand = out.setdefault(t, {}).setdefault(hs, {})
            for tm in m.get("top_matches") or []:
                name = tm["food_name"]
                sim = float(tm["similarity"])
                per_hand[name] = max(per_hand.get(name, 0.0), sim)
    return out


def load_dino_by_t(participant: str, session: str) -> dict[float, dict[str, dict[str, float]]]:
    """timestamp -> {hand_side: {instance_id: max_similarity}} (HOI-contact only)."""
    path = hands23_dir(participant, session) / f"{participant}_{session}_dino_matches.json"
    if not path.exists():
        return {}
    d = json.loads(path.read_text())
    out: dict[float, dict[str, dict[str, float]]] = {}
    for v in d["videos"]:
        for m in v["matches"]:
            if m.get("contact_state") != "object_contact":
                continue
            t = round(m["timestamp"], 2)
            hs = m.get("hand_side") or "unknown"
            per_hand = out.setdefault(t, {}).setdefault(hs, {})
            for tm in m.get("top_matches") or []:
                iid = tm["instance_id"]
                sim = float(tm["similarity"])
                per_hand[iid] = max(per_hand.get(iid, 0.0), sim)
    return out


def load_hoi_details_by_t(participant: str, session: str) -> dict[float, dict[str, dict]]:
    """timestamp -> {hand_side: {grasp, obj_touch, contact_state}} (HOI-contact only)."""
    path = hands23_dir(participant, session) / f"{participant}_{session}_hands23_results.json"
    if not path.exists():
        return {}
    d = json.loads(path.read_text())
    out: dict[float, dict[str, dict]] = {}
    for v in d["videos"]:
        for f in v["frames"]:
            t = round(f["session_timestamp_s"], 2)
            for det in f["detections"]:
                if det.get("contact_state") != "object_contact":
                    continue
                hs = det.get("hand_side") or "unknown"
                out.setdefault(t, {})[hs] = {
                    "grasp": det.get("grasp"),
                    "obj_touch": det.get("obj_touch"),
                    "contact_state": det.get("contact_state"),
                }
    return out


def load_owlv2_scene_by_t(participant: str, session: str) -> dict[float, str]:
    """timestamp -> scene_tag (one of storage|sink|stove|unknown). Empty if missing."""
    path = outputs_dir(participant, session) / "scene_tags_owlv2.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}
    out: dict[float, str] = {}
    for v in (data.get("frames") or {}).values():
        t = round(float(v.get("timestamp", 0.0)), 2)
        out[t] = str(v.get("scene") or "unknown")
    return out


def load_transparency_profile(participant: str) -> dict:
    """Load confusable_profile.json; return {instance_id: is_transparent_package}."""
    path = participant_dir(participant) / "confusable_profile.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}
    out: dict = {}
    for iid, entry in (data.get("items") or {}).items():
        if isinstance(entry, dict) and "is_transparent_package" in entry:
            out[iid] = bool(entry["is_transparent_package"])
    return out


# ---------------------------------------------------------------------------
# Per-item evidence formatter
# ---------------------------------------------------------------------------

_HAND_LABEL = {"left_hand": "L", "right_hand": "R"}
_HIDDEN_OBJ_TOUCH = {"neither_held", "neither_touched", None, ""}


def format_per_item_evidence(
    hoi_ts_sorted: list[float],
    siglip_by_t: dict[float, dict[str, dict[str, float]]],
    dino_by_t: dict[float, dict[str, dict[str, float]]],
    scene_by_t: dict[float, str],
    hoi_details_by_t: dict[float, dict[str, dict]],
    inventory: list[dict],
    min_score: float = 0.15,
    transparency_by_iid: dict | None = None,
) -> tuple[str, dict]:
    """Group detections by visual_class and render a chronological per-item block.

    Output shape (one section per inventory visual_class with ≥1 hit):

        ### "<visual_class>" (iids: ...; [opaque|transparent]; pkg=<capacity><unit>)
           <t>s  <scene>  <hand>[<grasp> <touch>]  dino=<v>[ sig=<v>]   (other: <side>="<other_vc>"=<v>)
           ...

    Each row is one (timestamp, hand_side) at which that visual_class was
    observed. Rows are sorted by timestamp. The "(other: ...)" suffix
    records the primary inventory class seen in the OTHER hand's crop at
    the same timestamp, which helps disambiguate dispensing moments (e.g.
    milk in R, target bowl in L). Classes with no detections at/above
    `min_score` are omitted entirely — the planner sees only active items.
    """
    transparency_by_iid = transparency_by_iid or {}

    inv_iids = {inv["instance_id"] for inv in inventory}
    inv_vcs = {inv["visual_class"] for inv in inventory}
    iid_to_vc: dict[str, str] = {inv["instance_id"]: inv["visual_class"] for inv in inventory}
    vc_to_iids: dict[str, list[str]] = defaultdict(list)
    inv_by_iid = {inv["instance_id"]: inv for inv in inventory}
    ordered_vcs: list[str] = []
    seen_vc: set[str] = set()
    for inv in inventory:
        vc = inv["visual_class"]
        vc_to_iids[vc].append(inv["instance_id"])
        if vc not in seen_vc:
            seen_vc.add(vc)
            ordered_vcs.append(vc)

    # First pass: for each (t, hand), resolve (dino_vc_scores, sig_vc_scores,
    # primary_vc_for_other_hand_reference). For each hand in HOI contact,
    # also remember grasp+obj_touch so the OTHER-hand field can reference
    # hands that were in contact with untracked items (tool, bowl, etc.).
    rows_by_vc: dict[str, list[dict]] = defaultdict(list)
    t_hand_primary: dict[float, dict[str, tuple[str, float]]] = {}
    t_hand_hoi: dict[float, dict[str, tuple[str | None, str | None]]] = {}

    for t in hoi_ts_sorted:
        dino_by_hand = dino_by_t.get(t) or {}
        sig_by_hand = siglip_by_t.get(t) or {}
        hands_detail = hoi_details_by_t.get(t) or {}
        scene = scene_by_t.get(t, "unknown") or "unknown"

        # Record HOI metadata for every hand in object_contact (tracked OR not).
        for hs, detail in hands_detail.items():
            g = detail.get("grasp")
            ot = detail.get("obj_touch")
            if ot in _HIDDEN_OBJ_TOUCH:
                ot = None
            t_hand_hoi.setdefault(t, {})[hs] = (g, ot)

        all_hands = set(dino_by_hand) | set(sig_by_hand) | set(hands_detail)
        for hs in all_hands:
            dino_vc: dict[str, float] = {}
            for iid, s in (dino_by_hand.get(hs) or {}).items():
                if iid not in inv_iids or s < min_score:
                    continue
                vc = iid_to_vc.get(iid)
                if not vc:
                    continue
                dino_vc[vc] = max(dino_vc.get(vc, 0.0), s)
            sig_vc: dict[str, float] = {
                vc: s for vc, s in (sig_by_hand.get(hs) or {}).items()
                if vc in inv_vcs and s >= min_score
            }
            if not dino_vc and not sig_vc:
                continue

            if dino_vc:
                primary_vc = max(dino_vc, key=lambda k: dino_vc[k])
                primary_score = dino_vc[primary_vc]
            else:
                primary_vc = max(sig_vc, key=lambda k: sig_vc[k])
                primary_score = sig_vc[primary_vc]
            t_hand_primary.setdefault(t, {})[hs] = (primary_vc, primary_score)

            detail = hands_detail.get(hs) or {}
            grasp = detail.get("grasp")
            obj_touch = detail.get("obj_touch")
            if obj_touch in _HIDDEN_OBJ_TOUCH:
                obj_touch = None

            for vc in set(dino_vc) | set(sig_vc):
                rows_by_vc[vc].append({
                    "t": t,
                    "hand": hs,
                    "scene": scene,
                    "grasp": grasp,
                    "touch": obj_touch,
                    "dino": dino_vc.get(vc),
                    "sig": sig_vc.get(vc),
                })

    # Render.
    lines: list[str] = []
    n_items_emitted = 0
    n_rows_emitted = 0

    for vc in ordered_vcs:
        rows = rows_by_vc.get(vc) or []
        if not rows:
            continue
        n_items_emitted += 1
        iids = vc_to_iids[vc]

        # Transparency: prefer explicit profile; majority of iids.
        n_transp = 0
        n_opaque = 0
        for iid in iids:
            if iid in transparency_by_iid:
                if transparency_by_iid[iid]:
                    n_transp += 1
                else:
                    n_opaque += 1
            else:
                vis = inv_by_iid[iid].get("visible_during_interaction", True)
                if vis:
                    n_transp += 1
                else:
                    n_opaque += 1
        tp_tag = "[transparent]" if n_transp >= n_opaque else "[opaque]"

        caps = {str(inv_by_iid[iid].get("package_amount", "?")) for iid in iids}
        pkg_str = next(iter(caps)) if len(caps) == 1 else "varies"
        unit = inv_by_iid[iids[0]].get("unit", "g")
        unit_label = "g" if unit == "g" else "ct"

        lines.append(
            f'### "{vc}" (iids: {", ".join(iids)}; {tp_tag}; pkg={pkg_str}, unit={unit_label})'
        )
        for r in sorted(rows, key=lambda x: x["t"]):
            label = _HAND_LABEL.get(r["hand"], "U")
            hdr_parts: list[str] = []
            if r["grasp"]:
                hdr_parts.append(r["grasp"])
            if r["touch"]:
                hdr_parts.append(r["touch"])
            hand_hdr = f"{label}[{' '.join(hdr_parts)}]" if hdr_parts else f"{label}[]"

            score_bits: list[str] = []
            if r["dino"] is not None:
                score_bits.append(f"dino={r['dino']:.2f}")
            if r["sig"] is not None:
                score_bits.append(f"sig={r['sig']:.2f}")
            scores_str = " ".join(score_bits)

            other_str = ""
            # Prefer another inventory-matched hand on the same timestamp.
            for oh, (ovc, osc) in (t_hand_primary.get(r["t"]) or {}).items():
                if oh == r["hand"] or ovc == vc:
                    continue
                olabel = _HAND_LABEL.get(oh, "U")
                ohoi = (t_hand_hoi.get(r["t"]) or {}).get(oh) or (None, None)
                ohoi_parts = [p for p in (ohoi[0], ohoi[1]) if p]
                ohoi_str = f"[{' '.join(ohoi_parts)}] " if ohoi_parts else ""
                other_str = f'   (other: {olabel}{ohoi_str}"{ovc}"={osc:.2f})'
                break
            # Fallback: other hand was in HOI contact with no inventory match.
            # Surface its grasp/obj_touch so the planner sees tool/container cues.
            if not other_str:
                for oh, (g, ot) in (t_hand_hoi.get(r["t"]) or {}).items():
                    if oh == r["hand"]:
                        continue
                    if (t_hand_primary.get(r["t"]) or {}).get(oh):
                        continue  # already covered above
                    parts = [p for p in (g, ot) if p]
                    if not parts:
                        continue
                    olabel = _HAND_LABEL.get(oh, "U")
                    other_str = f'   (other: {olabel}[{" ".join(parts)}] no_inv_match)'
                    break

            lines.append(
                f'  {r["t"]:7.1f}s  {r["scene"]:<8} {hand_hdr:<28} {scores_str}{other_str}'
            )
            n_rows_emitted += 1
        lines.append("")

    stats = {
        "n_hoi_frames_total": len(hoi_ts_sorted),
        "n_items_emitted": n_items_emitted,
        "n_rows_emitted": n_rows_emitted,
        "min_score": min_score,
    }
    return "\n".join(lines), stats


def format_inventory_for_prompt(
    inventory: list[dict],
    transparency_by_iid: dict | None = None,
) -> str:
    transparency_by_iid = transparency_by_iid or {}
    lines = []
    for inv in inventory:
        unit_label = "grams" if inv["unit"] == "g" else "count"
        iid = inv["instance_id"]
        if iid in transparency_by_iid:
            tag = "[transparent]" if transparency_by_iid[iid] else "[opaque]"
        else:
            visible = inv.get("visible_during_interaction", True)
            tag = "[transparent]" if visible else "[opaque]"
        lines.append(
            f"- {iid}: \"{inv['visual_class']}\" "
            f"({unit_label}, package={inv['package_amount']}, {tag})"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Step 1: Planner prompt (per-item evidence + journey-aware plan)
# ---------------------------------------------------------------------------

PLANNER_PROMPT = """\
You are an expert kitchen activity analyst. For every inventory \
visual_class that shows evidence in the per-item detections below, \
narrate the item's journey through the session and decide how a vision \
model should observe it to estimate the **remaining amount** in the \
stock container at the end of the session.

## What you are looking at

Evidence is grouped by visual_class. Each section lists every \
hand-object-contact timestamp where that class was observed:

    ### "visual_class" (iids: ...; [opaque|transparent]; pkg=<capacity>)
       <t>s  <scene>   <hand>[<grasp> <touch>]  dino=<v>[ sig=<v>]   (other: <side>="<other_vc>"=<v>)

Fields:
- **timestamp** — seconds from session start.
- **scene** — OWLv2 tag: `storage` (fridge / cabinet), `sink`, \
`stove`, or `unknown` (counter-side prep or unmatched).
- **hand** — `L` / `R` = left_hand / right_hand. The bracketed block \
summarises the hand-object interaction reported by the hands23 HOI \
detector for that hand at this timestamp.
    - `grasp=<code>` — hand posture (always shown when known).
    - `<obj_touch>` — qualitative object descriptor. Shown only when \
informative; omitted when the hand was in contact but the detector saw \
neither a tool nor a container.
- **dino** — DINO image-to-image similarity to this product's reference \
photo (max across all purchase instances of the class), computed on \
that hand's HOI crop. Usually the strongest signal.
- **sig** — SigLIP text-to-image similarity (only printed when available).
- **(other: ...)** — what the OTHER hand was doing at the same \
timestamp. Two forms:
    - `(other: L[<grasp> <obj_touch>] "<other_vc>"=<v>)` — the other \
hand's crop matched an inventory class. Useful when both hands hold \
tracked items simultaneously.
    - `(other: L[<grasp> <obj_touch>] no_inv_match)` — the other hand \
was in HOI contact but its crop did NOT match any tracked inventory \
item. The grasp / obj_touch are still informative: e.g. \
`R[Pre-Pris tool_used] no_inv_match` alongside a `container_held` on \
this item strongly suggests dispensing (this item's stock container is \
being spooned/poured with an untracked utensil); `L[Pow-Pris \
container_held] no_inv_match` next to this item's `Pre-Pris \
container_held` suggests the other hand holds a target bowl / pan.
  If the other hand had no contact or no informative HOI label, the \
suffix is omitted.

Only scores ≥ {min_score} are printed. Classes with NO detections are \
not listed at all — the planner sees only active items.

### Grasp codes (hands23 taxonomy)

The grasp tag has two parts joined by a hyphen. First = **contact \
strength**: `Pow` (power — full-hand wrap), `Pre` (precision — \
fingertip grip), `NP` (non-prehensile — poke / press). Second = \
**object shape** being contacted: `Pris` (prismatic — cartons, bottles, \
handles), `Circ` (circular — jars, cans, eggs, round fruit), `Fin` \
(finger-only), `Palm` (palm-only).

- `Pow-Pris` — wrapping a prism (carton, bottle, handle). Typical of \
retrieving / carrying a stock container.
- `Pow-Circ` — wrapping a round object (can, jar).
- `Pre-Pris` — fingertip grip on a prism (knife handle, carton edge \
while pouring, cracking an egg). Typical of dispensing, cutting.
- `Pre-Circ` — fingertip grip on a round object (egg, cap, berry).
- `NP-Fin` / `NP-Palm` — poking / pressing / wiping without enclosing.

### Object-touch codes

- `tool_held` — the hand holds a tool (knife, spoon, spatula, tongs).
- `tool_used` — held tool is actively contacting a target.
- `container_held` — the hand holds a food container (carton, bottle, \
bag, tub, pan). Strong signal that this block's visual_class IS the \
stock container.
- `container_touched` — touching but not holding (e.g. steadying a pan).

## How to reason about the journey

Walk each active class's timestamps in order and identify phases:

1. **Retrieval.** First appearance, typically in `storage` with a \
power grasp and `container_held`. The item was taken out of the fridge / \
cabinet.
2. **Transit / staging.** May appear in `unknown` scenes as the person \
carries the container to the counter.
3. **Leaving the original package (dispensing).** The critical moment. \
Signs: precision grasp (`Pre-Pris` / `Pre-Circ`), `container_held`. Short bursts rather than long holds. Portions \
physically leave the stock container here.
4. **Return.** Final appearance, often back to `storage` with a power \
grasp. Reading the container's fill level at or just before return gives \
the remaining amount.

If no dispensing phase is visible, the item was likely retrieved and \
returned without being used.

## Observation plan — sparse journey + dense dispensing

For each actively-used class, produce TWO kinds of samples:

- **journey_samples**: 3-6 individual timestamps that cover the phases \
above (one per phase you identified). These give the observer context \
about where the item came from and where it ended up. Draw from \
timestamps appearing in the evidence (or nearby).
- **dense_windows**: 0-2 short continuous windows (≤ 10 s each) centred \
on the moments the item LEAVES the original package. This is where the \
observer reads the stock container's fill level. For transparent \
packages, also include a dense window at the final return / last \
sighting to read the fill directly.

Budget: the observer has ~50 frames total across all observed classes. \
Spend the budget on classes and moments where it will change the \
remaining-amount answer. For items with no dispensing phase, \
`dense_windows` can be empty and `journey_samples` can be just the \
retrieval + return timestamps.

## Duplicate instance_ids of the same visual_class

When two or more inventory entries share the same `visual_class`, list \
ALL of its iids in `candidate_instance_ids`. The observer will decide \
which physical package is being handled and report per-instance \
remaining amounts for the others.

## Session Inventory
{inventory}

## Per-Item Detections (grouped by visual_class; chronological per item; HOI contact only)
{evidence}

## What to return

For every visual_class that appears in the per-item detections above, \
emit an `observations` entry. For every inventory instance_id whose \
class shows no coherent usage (e.g. only transient cross-talk hits), add a \
`skipped_items` entry with one-sentence reasoning. Every inventory iid \
MUST appear either as a candidate in at least one observation OR in \
`skipped_items`.

## Output — JSON only, no other text
```json
{{
  "observations": [
    {{
      "visual_class": "<visual_class exactly as in inventory>",
      "candidate_instance_ids": ["<iid1>", "<iid2>", ...],
      "journey_summary": "<2-3 sentences: where it came from, how it \
was used (or not), where it ended up — cite timestamps and grasps/scene>",
      "journey_samples": [<float>, ...],
      "dense_windows": [[<start_s>, <end_s>], ...],
      "reasoning": "<why these samples/windows — what you expect the \
observer to read from each>",
      "confidence": "high" | "medium" | "low"
    }}
  ],
  "skipped_items": [
    {{
      "instance_id": "<iid>",
      "reasoning": "<why this iid is being skipped>"
    }}
  ]
}}
```

Rules:
- Use EXACT `visual_class` and `instance_id` strings from the inventory.
- `candidate_instance_ids` MUST include EVERY inventory iid whose \
visual_class matches this observation.
- `journey_samples` SHOULD use timestamps that appear in the evidence \
(or are within a few seconds of one). Keep it to 3-6 samples per item.
- `dense_windows` may be empty. Each window should be ≤ 10 seconds and \
centred on a "leaving the package" moment.
- `confidence` refers to the observation decision.
"""


def _is_refusal(text: str) -> bool:
    low = text.strip().lower()
    return low.startswith("i'm sorry") or low.startswith("i cannot") or "cannot assist" in low


def run_planner(
    client,
    hoi_ts_sorted: list[float],
    siglip_by_t: dict,
    dino_by_t: dict,
    scene_by_t: dict,
    hoi_details_by_t: dict,
    inventory: list[dict],
    model: str,
    min_score: float,
    max_retries: int = 5,
    prompt_save_path: Path | None = None,
    transparency_by_iid: dict | None = None,
) -> tuple[str, str, dict, dict]:
    evidence_text, ev_stats = format_per_item_evidence(
        hoi_ts_sorted, siglip_by_t, dino_by_t, scene_by_t, hoi_details_by_t,
        inventory,
        min_score=min_score,
        transparency_by_iid=transparency_by_iid,
    )
    inventory_text = format_inventory_for_prompt(inventory, transparency_by_iid)

    prompt = PLANNER_PROMPT.format(
        inventory=inventory_text,
        evidence=evidence_text,
        min_score=f"{min_score:.2f}",
    )

    if prompt_save_path is not None:
        prompt_save_path.parent.mkdir(parents=True, exist_ok=True)
        prompt_save_path.write_text(prompt)

    for attempt in range(max_retries):
        t0 = time.time()
        try:
            response = client.responses.create(
                model=model,
                input=[{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
                reasoning={"effort": "medium"},
            )
            response_text = response.output_text or ""
            usage = response.usage
            stats = {
                "inference_time_s": round(time.time() - t0, 2),
                "model": model,
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "attempt": attempt + 1,
            }
        except Exception as e:
            print(f"  Planner ERROR (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(5 * (attempt + 1))
                continue
            return "", prompt, {"error": str(e), "inference_time_s": round(time.time() - t0, 2)}, ev_stats

        if _is_refusal(response_text) or (
            response_text and "observations" not in response_text
        ):
            filter_info = ""
            try:
                if hasattr(response, "model_dump"):
                    dump = response.model_dump()
                    for key in ("content_filter_results", "incomplete_details", "status"):
                        if key in dump and dump[key]:
                            filter_info += f" {key}={dump[key]}"
            except Exception:
                pass
            print(f"  Planner refusal/truncated (attempt {attempt + 1}), retrying...{filter_info}")
            if attempt < max_retries - 1:
                time.sleep(5 * (attempt + 1))
                continue

        return response_text, prompt, stats, ev_stats

    # Fallback: Qwen
    print(f"  Planner: GPT failed {max_retries} times, falling back to Qwen...")
    import requests
    try:
        t0 = time.time()
        resp = requests.post(QWEN_URL, json={
            "model": QWEN_MODEL_DEFAULT,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 16384,
            "temperature": 0.3,
        }, timeout=600)
        resp.raise_for_status()
        result = resp.json()
        response_text = result["choices"][0]["message"]["content"]
        usage = result.get("usage", {})
        stats = {
            "inference_time_s": round(time.time() - t0, 2),
            "model": f"qwen_fallback:{QWEN_MODEL_DEFAULT}",
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        }
        return response_text, prompt, stats, ev_stats
    except Exception as e:
        print(f"  Qwen fallback also failed: {e}")
        return "", prompt, {"error": f"all backends failed: {e}", "inference_time_s": 0}, ev_stats


def parse_planner_response(response_text: str) -> tuple[list[dict], list[dict]]:
    """Parse planner JSON.

    Returns (observations, skipped_items). `observations` is the list of
    per-item plans; each entry has visual_class, candidate_instance_ids,
    journey_summary, journey_samples (list[float]), dense_windows
    (list[[start, end]]), reasoning, confidence.
    """
    parsed = None

    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response_text, re.DOTALL)
    if fence:
        try:
            parsed = json.loads(fence.group(1))
        except json.JSONDecodeError:
            parsed = None

    if parsed is None:
        obj_match = re.search(r"\{.*\"observations\".*\}", response_text, re.DOTALL)
        if obj_match:
            try:
                parsed = json.loads(obj_match.group())
            except json.JSONDecodeError:
                parsed = None

    if not isinstance(parsed, dict):
        return [], []

    observations_raw = parsed.get("observations") or []
    skipped = parsed.get("skipped_items") or []
    if not isinstance(observations_raw, list):
        observations_raw = []
    if not isinstance(skipped, list):
        skipped = []

    observations: list[dict] = []
    for obs in observations_raw:
        if not isinstance(obs, dict):
            continue
        cand_iids = obs.get("candidate_instance_ids") or []
        if not isinstance(cand_iids, list) or not cand_iids:
            continue
        cand_iids = [str(c) for c in cand_iids]

        samples_raw = obs.get("journey_samples") or []
        samples: list[float] = []
        if isinstance(samples_raw, list):
            for s in samples_raw:
                try:
                    samples.append(float(s))
                except (TypeError, ValueError):
                    continue

        windows_raw = obs.get("dense_windows") or []
        windows: list[list[float]] = []
        if isinstance(windows_raw, list):
            for w in windows_raw:
                if not isinstance(w, (list, tuple)) or len(w) < 2:
                    continue
                try:
                    start = float(w[0])
                    end = float(w[1])
                except (TypeError, ValueError):
                    continue
                if end > start:
                    windows.append([start, end])

        observations.append({
            "visual_class": obs.get("visual_class"),
            "candidate_instance_ids": cand_iids,
            "journey_summary": obs.get("journey_summary", ""),
            "journey_samples": sorted(samples),
            "dense_windows": windows,
            "reasoning": obs.get("reasoning", ""),
            "confidence": obs.get("confidence"),
        })

    return observations, skipped


# ---------------------------------------------------------------------------
# Step 2: Observer
# ---------------------------------------------------------------------------

OBSERVER_PROMPT = """\
You are analyzing frames from an egocentric kitchen video recorded with smart glasses.

These {n_frames} frames are extracted from **multiple separate moments** within \
a cooking session — they are NOT continuous footage. Each frame is labeled with its \
session timestamp. There may be large time gaps between frames.

Frame timestamps: {frame_timestamps}

Frames were sampled in two ways (both are intermixed below, ordered by timestamp):
- **Journey samples** — sparse individual frames spanning the item's journey \
(retrieval from storage, transit, leaving the original package, return). These \
give CONTEXT about where the item came from and where it ended up.
- **Dense windows** — short continuous bursts of frames centred on moments when \
the item left its original package. These are where you read the stock \
container's fill level.

## Target Item
You are looking for: "{visual_class}"
- Unit: {unit_label}
- Package capacity: {package_capacity}

## Tracked purchase instances (candidates)
The inventory tracks {n_candidates} purchase instance(s) of this product. \
Identical-looking packages of the same product are tracked as separate \
instances; your job is to decide which physical package is being handled \
and to report a remaining amount for EACH candidate.

{candidate_table}

## Context from the planner
{segment_descriptions}

## Task
1. Confirm whether "{visual_class}" is visible in the frames.
2. Track the item across ALL frames. Distinguish between:
   - The **stock container/package** (carton, bottle, bag, block, tub, etc.) \
that holds the remaining inventory of this item.
   - **Portions that have been taken out** for use in this session (e.g., \
eggs cracked into a bowl, cheese grated onto a plate, oil poured into a \
pan, vegetables chopped onto a cutting board).
   Portions taken out are NOT remaining — they have already been consumed \
or are about to be consumed in this session.
3. **Assign each candidate a `handling_status`:**
   - `handled` — this specific instance is the physical package being \
retrieved, opened, dispensed from, or otherwise used in this session. At \
most one candidate is `handled` per session, unless two distinct packages \
are clearly handled separately.
   - `visible_untouched` — this candidate is visible (e.g. a second \
identical tub sitting in the fridge) but is NOT handled this session. \
Used ONLY for duplicate-instance disambiguation so the `handled` label \
gets attached to the right physical package. Always report \
`remaining: null` — the ledger's carry-forward value is the answer.
   - `not_visible` — this candidate is not visible in the frames; report \
`remaining: null` (the ledger will carry forward). Do NOT guess.
4. Estimate the **remaining amount** ONLY for the `handled` candidate.
   - Prefer the **dense-window frames** to read fill level during / just \
after dispensing, and use the **return / last sighting** (often a journey \
sample) to confirm the final state. If the container disappears from view \
before the end of the session, carry forward its last observed fill.
   - For `visible_untouched` and `not_visible` candidates, ALWAYS set \
`remaining: null`. The ledger already knows the carry-forward value; a \
fresh reading from the frames adds no information and introduces noise.
   - Do NOT add loose portions on plates/bowls/pans to any remaining amount; \
those are used, not remaining.
5. Cite which frame timestamps support each candidate's estimate (prefer \
frames showing that stock container).

Think step by step:
- How many distinct physical packages of this product are visible across \
the frames? If you see only one, mark one candidate `handled` and the rest \
`not_visible` — do NOT double-count by assigning the same physical package \
to two candidates.
- For the handled package: when is its stock container last visible? What \
portion appears to remain inside in its last visible frame?
- Are portions already taken out onto plates/bowls/pans/cutting boards? \
Note them as used, not remaining.
- If the container disappears from view before the end of the session, \
your estimate is its last observed fill — not what sits on the counter \
afterwards.

Output ONLY JSON:
```json
{{
  "per_instance": [
    {{
      "instance_id": "<iid exactly as listed in the candidates>",
      "handling_status": "handled" | "visible_untouched" | "not_visible",
      "remaining": <number or null>,
      "reasoning": "<step-by-step reasoning specific to this instance>",
      "evidence_frames": [<list of timestamp values of key frames \
supporting your estimate>]
    }}
  ]
}}
```"""


def extract_frames_for_observation(
    journey_samples: list[float],
    dense_windows: list[list[float]],
    video_durations: list[tuple[Path, float]],
    fps: float = 1.0,
    max_frames: int = MAX_IMAGES,
    sample_half_width: float = 0.5,
) -> tuple[list[str], list[float]]:
    """Combine sparse journey samples with dense windows into one frame set.

    Journey samples are expanded into tiny windows [t - w, t + w] and fetched
    with padding=0 so each one yields ~1 frame at fps=1. Dense windows are
    fetched with the default padding. The two results are merged and
    re-sorted by timestamp.
    """
    frames: list[str] = []
    times: list[float] = []

    if journey_samples:
        sample_segs = [[max(0.0, t - sample_half_width), t + sample_half_width] for t in journey_samples]
        jf, jt = _extract_segments_frames(
            [(s[0], s[1]) for s in sample_segs],
            video_durations,
            padding=0.0,
            max_frames=max_frames,
            target_fps=fps,
        )
        frames.extend(jf)
        times.extend(jt)

    if dense_windows:
        remaining_budget = max(1, max_frames - len(frames))
        df, dt = _extract_segments_frames(
            [(w[0], w[1]) for w in dense_windows],
            video_durations,
            padding=2.0,
            max_frames=remaining_budget,
            target_fps=fps,
        )
        frames.extend(df)
        times.extend(dt)

    if not frames:
        return [], []

    order = sorted(range(len(times)), key=lambda i: times[i])
    frames = [frames[i] for i in order]
    times = [times[i] for i in order]

    if len(frames) > max_frames:
        frames = frames[:max_frames]
        times = times[:max_frames]

    return frames, times


def _build_observer_prompt(
    n_frames: int,
    timestamps: list[float],
    candidates: list[dict],
    observation: dict,
) -> str:
    visual_class = observation.get("visual_class") or candidates[0]["visual_class"]
    unit_label = "grams" if candidates[0]["unit"] == "g" else "count"
    caps = {str(c["package_amount"]) for c in candidates}
    package_capacity = caps.pop() if len(caps) == 1 else (
        "varies by instance (see candidate table)"
    )

    candidate_lines = []
    for c in candidates:
        iid = c["instance_id"]
        m = re.search(r"_(\d{8})$", iid)
        purchase = m.group(1) if m else "unknown"
        cap = c.get("package_amount", "?")
        candidate_lines.append(
            f"- `{iid}` — purchased {purchase} — package {cap}"
        )
    candidate_table = "\n".join(candidate_lines)

    descs = []
    if observation.get("journey_summary"):
        descs.append(f"Journey: {observation['journey_summary']}")
    if observation.get("reasoning"):
        descs.append(f"Planner note: {observation['reasoning']}")
    js = observation.get("journey_samples") or []
    if js:
        descs.append(
            "Journey samples at: " + ", ".join(f"{t:.1f}s" for t in js)
        )
    dw = observation.get("dense_windows") or []
    if dw:
        descs.append(
            "Dense windows at: " + ", ".join(f"{w[0]:.1f}–{w[1]:.1f}s" for w in dw)
        )
    segment_descriptions = "\n".join(f"- {d}" for d in descs) if descs else "- (no planner context)"

    frame_ts_str = ", ".join(f"{t:.1f}s" for t in timestamps)

    return OBSERVER_PROMPT.format(
        n_frames=n_frames,
        frame_timestamps=frame_ts_str,
        visual_class=visual_class,
        unit_label=unit_label,
        package_capacity=package_capacity,
        n_candidates=len(candidates),
        candidate_table=candidate_table,
        segment_descriptions=segment_descriptions,
    )


def run_observer(
    client,
    frames_b64: list[str],
    timestamps: list[float],
    candidates: list[dict],
    observation: dict,
    model: str,
    prompt_save_path: Path | None = None,
) -> tuple[str, str, dict]:
    prompt = _build_observer_prompt(len(frames_b64), timestamps, candidates, observation)

    if prompt_save_path is not None:
        prompt_save_path.parent.mkdir(parents=True, exist_ok=True)
        prompt_save_path.write_text(prompt)

    content: list[dict] = [{"type": "input_text", "text": prompt}]
    for i, (fb64, ts) in enumerate(zip(frames_b64, timestamps)):
        content.append({"type": "input_text", "text": f"[Frame {i+1}, t={ts:.1f}s]"})
        content.append({
            "type": "input_image",
            "image_url": f"data:image/jpeg;base64,{fb64}",
            "detail": "high",
        })

    max_retries = 5
    for attempt in range(max_retries):
        t0 = time.time()
        try:
            response = client.responses.create(
                model=model,
                input=[{"role": "user", "content": content}],
                reasoning={"effort": "medium"},
            )
            response_text = response.output_text or ""
            usage = response.usage
            stats = {
                "inference_time_s": round(time.time() - t0, 2),
                "model": model,
                "num_frames": len(frames_b64),
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "attempt": attempt + 1,
            }
        except Exception as e:
            err_str = str(e)
            transient = any(m in err_str for m in (
                "503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED",
                "500", "INTERNAL", "deadline", "timeout", "connection", "Connection",
            ))
            if "content_policy_violation" in err_str and attempt < max_retries - 1:
                print(f" content filter (attempt {attempt + 1}), retrying...", end="", flush=True)
                time.sleep(5 * (attempt + 1))
                continue
            if transient and attempt < max_retries - 1:
                wait = 4 * (2 ** attempt)
                print(f" transient API error (attempt {attempt + 1}), retrying in {wait}s...", end="", flush=True)
                time.sleep(wait)
                continue
            vc = observation.get("visual_class") or (candidates[0]["visual_class"] if candidates else "?")
            print(f"  Observer ERROR ({vc}): {e}")
            return "", prompt, {"error": err_str, "inference_time_s": round(time.time() - t0, 2)}

        if _is_refusal(response_text) and attempt < max_retries - 1:
            print(f" refusal (attempt {attempt + 1}), retrying...", end="", flush=True)
            time.sleep(5)
            continue

        return response_text, prompt, stats

    return "", prompt, {"error": "max retries exceeded"}


def run_observer_qwen(
    frames_b64: list[str],
    timestamps: list[float],
    candidates: list[dict],
    observation: dict,
    qwen_url: str = QWEN_URL,
    qwen_model: str = QWEN_MODEL_DEFAULT,
    prompt_save_path: Path | None = None,
) -> tuple[str, str, dict]:
    import requests

    prompt = _build_observer_prompt(len(frames_b64), timestamps, candidates, observation)

    if prompt_save_path is not None:
        prompt_save_path.parent.mkdir(parents=True, exist_ok=True)
        prompt_save_path.write_text(prompt)

    content: list[dict] = [{"type": "text", "text": prompt}]
    for i, (fb64, ts) in enumerate(zip(frames_b64, timestamps), 1):
        content.append({"type": "text", "text": f"[Frame {i}, t={ts:.1f}s]"})
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{fb64}"},
        })

    messages = [{"role": "user", "content": content}]

    t0 = time.time()
    max_retries = 4
    for attempt in range(max_retries):
        try:
            resp = requests.post(qwen_url, json={
                "model": qwen_model,
                "messages": messages,
                "max_tokens": 4096,
                "temperature": 0.3,
            }, timeout=600)
            resp.raise_for_status()
            result = resp.json()
            response_text = result["choices"][0]["message"]["content"]
            usage = result.get("usage", {})
            stats = {
                "inference_time_s": round(time.time() - t0, 2),
                "model": qwen_model,
                "num_frames": len(frames_b64),
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
                "attempt": attempt + 1,
            }
            break
        except Exception as e:
            last_err = str(e)
            transient = any(m in last_err for m in (
                "503", "UNAVAILABLE", "429", "500", "502", "504",
                "deadline", "timeout", "connection", "Connection",
            ))
            if transient and attempt < max_retries - 1:
                wait = 4 * (2 ** attempt)
                print(f" Qwen transient error (attempt {attempt + 1}), retrying in {wait}s...", end="", flush=True)
                time.sleep(wait)
                continue
            print(f"  Qwen ERROR: {e}")
            return "", prompt, {"error": last_err, "inference_time_s": round(time.time() - t0, 2)}

    return response_text, prompt, stats


def parse_observer_response(response_text: str) -> list[dict]:
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response_text, re.DOTALL)
    text_to_parse = fence.group(1) if fence else response_text
    obj_match = re.search(r"\{.*\"per_instance\".*\}", text_to_parse, re.DOTALL)
    if not obj_match:
        return []
    try:
        parsed = json.loads(obj_match.group())
    except json.JSONDecodeError:
        return []
    entries = parsed.get("per_instance") if isinstance(parsed, dict) else None
    if not isinstance(entries, list):
        return []

    result = []
    for e in entries:
        if not isinstance(e, dict):
            continue
        iid = e.get("instance_id")
        if not iid:
            continue
        rem = e.get("remaining")
        try:
            rem_val = float(rem) if rem is not None else None
        except (TypeError, ValueError):
            rem_val = None
        result.append({
            "instance_id": iid,
            "handling_status": e.get("handling_status"),
            "remaining": rem_val,
            "reasoning": e.get("reasoning", ""),
            "evidence_frames": e.get("evidence_frames", []),
        })
    return result


# ---------------------------------------------------------------------------
# Session pipeline
# ---------------------------------------------------------------------------

def process_session(
    participant: str,
    session: str,
    client,
    ledger: dict,
    model: str,
    fps: float,
    max_frames: int,
    min_score: float,
    planner_only: bool = False,
    observer_only: bool = False,
    verbose: bool = False,
    model_tag: str = "default",
    run_tag: str = "default",
    inventory_scope: str = "full",
    transparency_by_iid: dict | None = None,
) -> tuple[list[dict], dict]:
    session_log: dict = {"session": session, "planner": {}, "observer": []}

    cache_dir = CACHE_DIR / participant / session / model_tag / run_tag
    cache_dir.mkdir(parents=True, exist_ok=True)

    inventory = load_inventory(participant, session, scope=inventory_scope)
    if not inventory:
        print(f"  {session}: no inventory for scope={inventory_scope}")
        return [], session_log

    try:
        all_ts, hoi_ts = load_hoi_timestamps(participant, session)
    except FileNotFoundError as e:
        print(f"  {session}: missing hands23 results ({e}) — skipping")
        return [], session_log
    if not hoi_ts:
        print(f"  {session}: no HOI-contact frames — skipping")
        return [], session_log

    siglip_by_t = load_siglip_by_t(participant, session)
    dino_by_t = load_dino_by_t(participant, session)
    scene_by_t = load_owlv2_scene_by_t(participant, session)
    hoi_details_by_t = load_hoi_details_by_t(participant, session)

    hoi_sorted = sorted(hoi_ts)
    print(f"  {session}: {len(all_ts)} frames, {len(hoi_sorted)} HOI-contact, "
          f"{len(inventory)} inventory items ({inventory_scope} scope), "
          f"{len(scene_by_t)} OWLv2 scene tags")

    inv_by_iid = {inv["instance_id"]: inv for inv in inventory}
    planner_model = "gpt-5.4"

    if observer_only:
        planner_sess_path = (
            outputs_dir(participant, session)
            / f"{OUTPUT_PREFIX}_{model_tag}_{run_tag}_planner.json"
        )
        if not planner_sess_path.exists():
            print(f"  {session}: --observer-only but no saved plan at "
                  f"{planner_sess_path} — skipping")
            return [], session_log
        try:
            saved = json.loads(planner_sess_path.read_text())
        except json.JSONDecodeError as e:
            print(f"  {session}: failed to read saved plan ({e}) — skipping")
            return [], session_log
        saved_log = saved.get("session", {})
        saved_planner = saved_log.get("planner", {}) or {}
        observations = saved_planner.get("observations") or []
        skipped_items = saved_planner.get("skipped_items") or []
        session_log["planner"] = saved_planner
        print(f"  Step 1 (Planner): LOADED {len(observations)} observations "
              f"from saved plan (observer-only mode)")
        planner_stats = saved_planner.get("stats", {})
    else:
        print(f"  Step 1 (Planner): sending per-item evidence to {planner_model}...")
        planner_text, planner_prompt, planner_stats, ev_stats = run_planner(
            client, hoi_sorted, siglip_by_t, dino_by_t, scene_by_t,
            hoi_details_by_t, inventory, planner_model,
            min_score=min_score,
            prompt_save_path=cache_dir / "planner_prompt.txt",
            transparency_by_iid=transparency_by_iid,
        )
        (cache_dir / "planner_response.txt").write_text(planner_text or "")
        observations, skipped_items = parse_planner_response(planner_text)

        valid: list[dict] = []
        for obs in observations:
            valid_iids = [c for c in obs["candidate_instance_ids"] if c in inv_by_iid]
            if not valid_iids:
                if verbose:
                    print(f"    DROP observation: no candidate iids in inventory "
                          f"({obs['candidate_instance_ids']})")
                continue
            obs["candidate_instance_ids"] = valid_iids
            valid.append(obs)
        observations = valid

        n_obs = len(observations)
        n_skipped = len(skipped_items)
        total_samples = sum(len(o.get("journey_samples") or []) for o in observations)
        total_windows = sum(len(o.get("dense_windows") or []) for o in observations)
        print(f"  Evidence: {ev_stats['n_items_emitted']} items with hits "
              f"({ev_stats['n_rows_emitted']} detection rows across "
              f"{ev_stats['n_hoi_frames_total']} HOI frames)")
        print(f"  Planner: {n_obs} observations "
              f"({total_samples} journey samples + {total_windows} dense windows); "
              f"{n_skipped} skipped_items")
        if n_skipped:
            for s in skipped_items:
                print(f"    SKIP {s.get('instance_id')}: {s.get('reasoning', '')[:120]}")
        # Derive a flat `observation_plan` (one entry per candidate iid, with
        # a plain [start, end] segments list) so the annotator / video-server,
        # which still expects the legacy AVP planner schema, can render
        # observer-trace segments for items whose observer verdict produced no
        # prediction (e.g. not_visible).
        observation_plan_flat = []
        for obs in observations:
            segs = []
            for t in (obs.get("journey_samples") or []):
                segs.append([max(0.0, float(t) - 0.5), float(t) + 0.5])
            for w in (obs.get("dense_windows") or []):
                segs.append([float(w[0]), float(w[1])])
            for iid in obs.get("candidate_instance_ids") or []:
                observation_plan_flat.append({
                    "instance_id": iid,
                    "visual_class": obs.get("visual_class"),
                    "segments": segs,
                    "reasoning": obs.get("reasoning", ""),
                    "confidence": obs.get("confidence"),
                    "journey_summary": obs.get("journey_summary", ""),
                })
        session_log["planner"] = {
            "n_hoi_frames": len(hoi_sorted),
            "evidence_stats": ev_stats,
            "n_observations": n_obs,
            "n_skipped_items": n_skipped,
            "observations": observations,
            "observation_plan": observation_plan_flat,
            "skipped_items": skipped_items,
            "stats": planner_stats,
            "prompt": planner_prompt,
            "raw_response": planner_text,
        }

    if not observations:
        print(f"  Planner returned no observations")
        return [], session_log

    items_str = ", ".join(
        f"{o.get('visual_class', '?')}"
        f"({'+'.join(o['candidate_instance_ids'])})"
        f"[{o.get('confidence', '?')}]"
        for o in observations
    )
    print(f"  Planner identified {len(observations)} observations: {items_str}")

    if planner_only:
        return [], session_log

    clips = get_session_clips(participant, session)
    if not clips:
        print(f"  {session}: no video clips found")
        return [], session_log
    video_durations = [(path, dur) for _, path, dur in clips]

    predictions = []
    for obs in observations:
        cand_iids = obs["candidate_instance_ids"]
        candidates = [inv_by_iid[c] for c in cand_iids if c in inv_by_iid]
        if not candidates:
            continue

        journey_samples = obs.get("journey_samples") or []
        dense_windows = obs.get("dense_windows") or []
        if not journey_samples and not dense_windows:
            print(f"    SKIP {obs.get('visual_class')}: no samples/windows")
            continue

        vc = obs.get("visual_class") or candidates[0]["visual_class"]
        cand_str = ",".join(cand_iids)
        js_str = ("+".join(f"{t:.0f}" for t in journey_samples)) or "—"
        dw_str = ("+".join(f"{w[0]:.0f}-{w[1]:.0f}" for w in dense_windows)) or "—"
        print(f"    Observer: {vc} journey=[{js_str}]s dense=[{dw_str}]s "
              f"candidates=[{cand_str}]...", end="", flush=True)

        vllm_endpoint = VLLM_ENDPOINTS.get(model.lower())
        use_vllm = (
            vllm_endpoint is not None
            or model.lower().startswith("qwen")
            or model.lower().startswith("gemma")
        )

        frames, frame_ts = extract_frames_for_observation(
            journey_samples, dense_windows, video_durations,
            fps=fps, max_frames=max_frames,
        )
        if not frames:
            print(f" no frames")
            continue

        vc_slug = re.sub(r"[^A-Za-z0-9]+", "_", vc).strip("_").lower() or "obs"
        obs_prompt_path = cache_dir / f"{vc_slug}_observer_prompt.txt"
        obs_response_path = cache_dir / f"{vc_slug}_observer_response.txt"

        if use_vllm:
            if vllm_endpoint:
                obs_url, obs_model = vllm_endpoint
            else:
                obs_url, obs_model = QWEN_URL, model
            obs_text, obs_prompt, obs_stats = run_observer_qwen(
                frames, frame_ts, candidates, obs,
                qwen_url=obs_url,
                qwen_model=obs_model,
                prompt_save_path=obs_prompt_path,
            )
        else:
            obs_text, obs_prompt, obs_stats = run_observer(
                client, frames, frame_ts, candidates, obs, model,
                prompt_save_path=obs_prompt_path,
            )
        obs_response_path.write_text(obs_text or "")
        per_instance = parse_observer_response(obs_text)

        obs_log = {
            "visual_class": vc,
            "candidate_instance_ids": cand_iids,
            "journey_samples": journey_samples,
            "dense_windows": dense_windows,
            "n_frames": len(frame_ts) if frame_ts else 0,
            "per_instance": per_instance,
            "stats": obs_stats,
            "raw_response": obs_text,
        }
        session_log["observer"].append(obs_log)

        if not per_instance:
            if _is_refusal(obs_text):
                print(f" REFUSAL (all retries exhausted)")
            else:
                print(f" parse failed / empty per_instance")
            continue

        emitted = []
        for inst in per_instance:
            iid = inst["instance_id"]
            if iid not in inv_by_iid:
                continue
            status = inst.get("handling_status")
            rem = inst.get("remaining")
            # visible_untouched is treated like not_visible: the ledger's
            # carry-forward is the answer, so we drop any fresh reading the
            # observer produced. Only 'handled' verdicts emit predictions.
            if status == "visible_untouched":
                emitted.append(f"{iid}:visible_untouched=null")
                continue
            if rem is None:
                emitted.append(f"{iid}:{status}=null")
                continue
            emitted.append(f"{iid}:{status}={rem}")
            # Flatten journey_samples + dense_windows into a single `segments`
            # list so downstream consumers (annotator / video-server) that
            # expect the legacy AVP schema can draw VLM-input bands without
            # having to understand the new per-item fields.
            seg_list: list[list[float]] = []
            for t in journey_samples:
                seg_list.append([max(0.0, t - 0.5), t + 0.5])
            for w in dense_windows:
                seg_list.append([w[0], w[1]])
            predictions.append({
                "session": session,
                "item": inv_by_iid[iid]["visual_class"],
                "instance_id": iid,
                "amount_remaining": rem,
                "handling_status": status,
                "reasoning": inst.get("reasoning", ""),
                "evidence_frames": inst.get("evidence_frames", []),
                "planner_reasoning": obs.get("reasoning", ""),
                "planner_confidence": obs.get("confidence", ""),
                "journey_samples": journey_samples,
                "dense_windows": dense_windows,
                "segments": seg_list,
                "stats": {
                    "planner": planner_stats,
                    "observer": obs_stats,
                },
            })
        print(f" {' '.join(emitted)}")

    print(f"  {session}: {len(predictions)} predictions from "
          f"{len(observations)} observations")
    return predictions, session_log


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="AVP Round 1 (remaining-only, CandList_HOI_PerItem): "
                    "branch of CandList_HOI with per-item detection evidence "
                    "+ journey-aware observation plan (sparse journey "
                    "samples + dense dispensing windows)."
    )
    parser.add_argument("--participant", default="kailai")
    parser.add_argument("--tag", required=True,
                        help="Short label for this run (e.g. 'PerItem_v1').")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--session", help="Single session")
    group.add_argument("--all", action="store_true", help="All sessions")
    parser.add_argument("--model", default="gpt-5.4", help="Observer model (planner always uses gpt-5.4)")
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--max-frames", type=int, default=MAX_IMAGES)
    parser.add_argument("--min-score", type=float, default=0.15,
                        help="Only include per-frame detections with DINO or SigLIP >= this threshold.")
    parser.add_argument("--planner-only", action="store_true")
    parser.add_argument("--observer-only", action="store_true",
                        help="Skip planner; load saved plan (same --tag) and run observer only.")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", action="store_true",
                        help="Resume a previously-interrupted run with the same --tag.")
    parser.add_argument("--inventory-scope", choices=["full", "session"], default="full",
                        help="Which items to list in the planner's inventory.")
    args = parser.parse_args()

    sessions = (
        [args.session] if args.session
        else get_sessions(args.participant) if args.all
        else get_sessions(args.participant)
    )

    model_tag = args.model.replace("/", "_")
    run_tag = args.tag.replace("/", "_")

    print(f"{'=' * 70}")
    print(f"AVP Round 1 (Remaining-Only, CandList_HOI_PerItem)")
    print(f"{'=' * 70}")
    print(f"Participant:  {args.participant}")
    print(f"Model:        {args.model}")
    print(f"Tag:          {run_tag}")
    print(f"FPS:          {args.fps}")
    print(f"Max frames:   {args.max_frames}")
    print(f"Min score:    {args.min_score}")
    print(f"Planner only: {args.planner_only}")
    print(f"Observer only:{args.observer_only}")
    if args.planner_only and args.observer_only:
        parser.error("--planner-only and --observer-only are mutually exclusive.")
    print(f"Sessions:     {len(sessions)}")
    print()

    client = make_client()
    ledger = load_ledger(args.participant)
    transparency_by_iid = load_transparency_profile(args.participant)
    print(f"Transparency profile: {len(transparency_by_iid)} items tagged "
          f"(from confusable_profile.json)")

    all_predictions: list[dict] = []
    all_logs: list[dict] = []

    out_dir = participant_dir(args.participant) / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    preds_path = args.output or out_dir / f"{OUTPUT_PREFIX}_{model_tag}_{run_tag}_preds.json"
    status_path = out_dir / f"{OUTPUT_PREFIX}_{model_tag}_{run_tag}_status.json"

    status: dict = {"completed_sessions": [], "failed_sessions": []}
    if args.observer_only:
        print("\nOBSERVER-ONLY: loading saved plans and re-running observer "
              "for all requested sessions; prior preds/status will be overwritten.")
    elif args.resume and preds_path.exists() and status_path.exists():
        all_predictions = json.loads(preds_path.read_text())
        status = json.loads(status_path.read_text())
        for s in status.get("completed_sessions", []):
            sess_log_path = outputs_dir(args.participant, s) / f"{OUTPUT_PREFIX}_{model_tag}_{run_tag}_planner.json"
            if sess_log_path.exists():
                try:
                    sess_data = json.loads(sess_log_path.read_text())
                    if "session" in sess_data:
                        all_logs.append(sess_data["session"])
                except json.JSONDecodeError:
                    pass
        completed = set(status.get("completed_sessions", []))
        pending = [s for s in sessions if s not in completed]
        print(f"\nRESUME: {len(completed)} session(s) already complete, "
              f"{len(pending)} pending (of {len(sessions)} total).")
        sessions = pending
    elif args.resume:
        print("\nRESUME requested but no existing predictions/status found — starting fresh.")

    failed_sessions: list[tuple[str, str]] = []
    for session in sessions:
        session_failed = False
        try:
            preds, log = process_session(
                participant=args.participant,
                session=session,
                client=client,
                ledger=ledger,
                model=args.model,
                fps=args.fps,
                max_frames=args.max_frames,
                min_score=args.min_score,
                planner_only=args.planner_only,
                observer_only=args.observer_only,
                verbose=args.verbose,
                model_tag=model_tag,
                run_tag=run_tag,
                inventory_scope=args.inventory_scope,
                transparency_by_iid=transparency_by_iid,
            )
        except KeyboardInterrupt:
            raise
        except Exception as e:
            import traceback
            print(f"\n  ERROR in session {session}: {e}")
            traceback.print_exc()
            failed_sessions.append((session, str(e)[:200]))
            preds, log = [], {"session": session, "planner": {}, "observer": [], "error": str(e)[:500]}
            session_failed = True
            status["failed_sessions"] = [
                f for f in status["failed_sessions"] if f.get("session") != session
            ] + [{"session": session, "error": str(e)[:200]}]
            status_path.write_text(json.dumps(status, indent=2) + "\n")
            all_logs.append(log)
            if not args.planner_only:
                with open(preds_path, "w", encoding="utf-8") as f:
                    json.dump(all_predictions, f, indent=2)
            print(f"\n  HALTED at session {session}. Fix the issue and re-run with --resume.")
            break

        all_predictions.extend(preds)
        all_logs.append(log)

        if not session_failed:
            if session not in status["completed_sessions"]:
                status["completed_sessions"].append(session)
            status["failed_sessions"] = [
                f for f in status["failed_sessions"] if f.get("session") != session
            ]
            status_path.write_text(json.dumps(status, indent=2) + "\n")

        if not args.planner_only:
            with open(preds_path, "w", encoding="utf-8") as f:
                json.dump(all_predictions, f, indent=2)

        sess_out = outputs_dir(args.participant, session)
        sess_out.mkdir(parents=True, exist_ok=True)

        planner_sess_path = sess_out / f"{OUTPUT_PREFIX}_{model_tag}_{run_tag}_planner.json"
        with open(planner_sess_path, "w", encoding="utf-8") as f:
            json.dump({
                "participant": args.participant,
                "timestamp": datetime.now().isoformat(),
                "model": args.model,
                "tag": run_tag,
                "session": log,
            }, f, indent=2)

        if not args.planner_only:
            preds_sess_path = sess_out / f"{OUTPUT_PREFIX}_{model_tag}_{run_tag}_preds.json"
            with open(preds_sess_path, "w", encoding="utf-8") as f:
                json.dump(preds, f, indent=2)
            print(f"  Saved: {preds_sess_path}")

        print()

    planner_path = out_dir / f"{OUTPUT_PREFIX}_{model_tag}_{run_tag}_planner.json"
    planner_output = {
        "participant": args.participant,
        "timestamp": datetime.now().isoformat(),
        "model": args.model,
        "tag": run_tag,
        "sessions": all_logs,
    }
    with open(planner_path, "w", encoding="utf-8") as f:
        json.dump(planner_output, f, indent=2)
    print(f"Saved: {planner_path}")

    if not args.planner_only:
        print(f"Saved: {preds_path}")
    print(f"Status saved to {status_path}")

    if not args.planner_only and all_predictions:
        print(f"\n{'=' * 70}")
        print(f"Predictions ({len(all_predictions)} items):")
        for p in all_predictions:
            unit = "count" if ledger["items"][p["instance_id"]]["unit"] == "count" else "g"
            print(f"  [{p['session']}] {p['item']}: "
                  f"remaining={p['amount_remaining']} {unit} "
                  f"({p['planner_confidence']})")

    if failed_sessions:
        print(f"\n{len(failed_sessions)} session(s) failed in this run:")
        for s, err in failed_sessions:
            print(f"  {s}: {err}")
        print(f"\nRe-run with --resume to retry the failed sessions only.")


if __name__ == "__main__":
    main()

"""Cross-session evidence store for the observer.

Persists, per (instance_id, session), the frame timestamps the sweep observer
cited as backing its `amount_starting` and `amount_remaining` estimates, plus
the JPEG snapshots of those frames. On subsequent sessions the retrieval
helper can return the most recent N priors for a given item so the caller
can splice them into the next observer prompt.

Storage layout (under `participants/<p>/outputs/`):

  observer_evidence_{model_tag}_{run_tag}.json
      Cumulative index keyed by instance_id:
      {
        "<instance_id>": [
          {
            "session": "<YYYYMMDD-HHMMSS>",
            "visual_class": "...",
            "starting_amount": 510.0,        # may be null
            "starting_frame": 254.3,         # session-cumulative seconds
            "starting_image_path": "observer_evidence/<run_tag>/<iid>/<sess>_starting.jpg",
            "remaining_amount": 445.0,
            "remaining_frame": 266.6,
            "remaining_image_path": "observer_evidence/<run_tag>/<iid>/<sess>_remaining.jpg",
            "round_starting": 1,             # which round produced each frame
            "round_remaining": 1,
          },
          ...
        ]
      }

  observer_evidence/<run_tag>/<instance_id>/<session>_{starting,remaining}.jpg
      Snapshotted JPEG at the cited timestamp. JPEG quality 85 — same as the
      live sampler so prior frames look identical to current-session frames.

Storage is **unconditional**: the main script calls `save()` after every
prediction whether or not `--use-evidence` is on, so a no-evidence run still
populates the store and a later evidence-on run can consume it.

Retrieval is **gated by the caller** (see `--use-evidence` in the main
script). `load_priors()` returns only entries strictly before the current
session, sorted oldest-first, capped at the most-recent K.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Iterable

from frame_sampling import extract_single_frame


def _index_path(participant_dir: Path, model_tag: str, run_tag: str) -> Path:
    return participant_dir / "outputs" / f"observer_evidence_{model_tag}_{run_tag}.json"


def _image_dir(participant_dir: Path, run_tag: str, instance_id: str) -> Path:
    return participant_dir / "outputs" / "observer_evidence" / run_tag / instance_id


def load_index(participant_dir: Path, model_tag: str, run_tag: str) -> dict[str, list[dict]]:
    """Read the cumulative index from disk. Empty dict if none."""
    p = _index_path(participant_dir, model_tag, run_tag)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def write_index(
    participant_dir: Path, model_tag: str, run_tag: str,
    index: dict[str, list[dict]],
) -> None:
    p = _index_path(participant_dir, model_tag, run_tag)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(index, indent=2) + "\n")


def save(
    *,
    participant_dir: Path,
    model_tag: str,
    run_tag: str,
    instance_id: str,
    visual_class: str,
    session: str,
    starting_amount: float | None,
    starting_frame: float | None,
    starting_round: int | None,
    remaining_amount: float | None,
    remaining_frame: float | None,
    remaining_round: int | None,
    video_durations: list[tuple[Path, float]],
) -> dict | None:
    """Persist evidence for one (instance_id, session) and return the entry.

    Snapshots up to two JPEGs (one for starting, one for remaining) at the
    cited timestamps. Skips a side if its frame timestamp is None — e.g.,
    `computed_remaining` predictions don't ground `amount_remaining` in a
    single frame, so they get a starting snapshot but not a remaining one.

    Re-saving the same (instance_id, session) replaces the prior entry — so
    a re-run with `--resume` updates rather than dupes.

    Returns the dict that was written into the index, or None if neither
    side could be snapshotted (e.g., both frame timestamps null).
    """
    img_dir = _image_dir(participant_dir, run_tag, instance_id)
    img_dir.mkdir(parents=True, exist_ok=True)

    entry: dict = {
        "session": session,
        "visual_class": visual_class,
        "starting_amount": starting_amount,
        "starting_frame": starting_frame,
        "starting_image_path": None,
        "round_starting": starting_round,
        "remaining_amount": remaining_amount,
        "remaining_frame": remaining_frame,
        "remaining_image_path": None,
        "round_remaining": remaining_round,
    }

    snapshotted = False
    for side, ts in (("starting", starting_frame), ("remaining", remaining_frame)):
        if ts is None:
            continue
        snap = extract_single_frame(float(ts), video_durations)
        if snap is None:
            continue
        b64, _, _ = snap
        rel = Path("observer_evidence") / run_tag / instance_id / f"{session}_{side}.jpg"
        abs_path = participant_dir / "outputs" / rel
        abs_path.write_bytes(base64.b64decode(b64))
        entry[f"{side}_image_path"] = str(rel)
        snapshotted = True

    if not snapshotted:
        return None

    # Update index: replace any prior entry for the same session.
    index = load_index(participant_dir, model_tag, run_tag)
    bucket = index.setdefault(instance_id, [])
    bucket[:] = [e for e in bucket if e.get("session") != session]
    bucket.append(entry)
    bucket.sort(key=lambda e: e.get("session", ""))
    write_index(participant_dir, model_tag, run_tag, index)
    return entry


def load_priors(
    *,
    participant_dir: Path,
    model_tag: str,
    run_tag: str,
    instance_id: str,
    visual_class: str,
    before_session: str,
    k: int = 3,
    match_key: str = "instance_id",
    hydrate_images: bool = True,
) -> list[dict]:
    """Return up to K most recent prior evidence entries.

    Args:
        instance_id: target item's instance_id.
        visual_class: target item's visual_class — used when match_key is
            `visual_class` to pull priors from sibling instances of the
            same visual class (different purchase dates).
        before_session: only entries with session < this string are
            returned. Compared lexicographically — works because session
            ids are YYYYMMDD-HHMMSS.
        k: cap. 0 means unlimited (debug only — caller should normally cap).
        match_key: `instance_id` (default, strict same-physical-item priors)
            or `visual_class` (cross-instance priors of the same product).
        hydrate_images: when True, attach `starting_image_b64` and
            `remaining_image_b64` keys with the JPEG bytes loaded from disk.
            False is faster for callers that just want the metadata.

    Returns entries in CHRONOLOGICAL order, oldest first. Entries whose
    image files are missing on disk are dropped.
    """
    index = load_index(participant_dir, model_tag, run_tag)

    candidates: list[dict] = []
    if match_key == "visual_class":
        for iid, entries in index.items():
            for e in entries:
                if e.get("visual_class") == visual_class:
                    candidates.append({**e, "_source_iid": iid})
    else:
        for e in index.get(instance_id, []):
            candidates.append({**e, "_source_iid": instance_id})

    candidates = [e for e in candidates if e.get("session", "") < before_session]
    candidates.sort(key=lambda e: e.get("session", ""))
    if k > 0:
        candidates = candidates[-k:]

    if not hydrate_images:
        return candidates

    out: list[dict] = []
    for e in candidates:
        loaded = dict(e)
        any_image = False
        for side in ("starting", "remaining"):
            rel = e.get(f"{side}_image_path")
            if not rel:
                loaded[f"{side}_image_b64"] = None
                continue
            abs_path = participant_dir / "outputs" / rel
            if not abs_path.exists():
                loaded[f"{side}_image_b64"] = None
                continue
            loaded[f"{side}_image_b64"] = base64.b64encode(abs_path.read_bytes()).decode()
            any_image = True
        if any_image:
            out.append(loaded)
    return out


def format_priors_for_prompt(
    priors: list[dict],
    *,
    include_amounts: bool = False,
    unit_label: str = "g",
    images_per_prior: int = 1,
    prefer_side: str = "remaining",
    item_label: str = "this item",
) -> tuple[str, list[tuple[str, str]]]:
    """Build a human-readable prior-evidence text block + parallel image list.

    Args:
        priors: list of evidence entries (from `load_priors(...)`),
            chronological oldest-first.
        include_amounts: when True, embed the numeric `starting_amount` /
            `remaining_amount` values in the text block. Default False
            keeps the prompt visual-only so the model isn't anchored to
            prior numeric estimates.
        unit_label: 'g' or '' (for count items). Only used when
            include_amounts is True.
        images_per_prior: 0, 1 (default), or 2. With 0, no images are
            emitted — bullets are numeric-only (requires include_amounts
            or the bullets carry no information; caller is responsible).
            With 1, each prior contributes exactly one image —
            `prefer_side` (remaining/starting) is tried first, fallback
            to the other side. With 2, both sides are emitted when
            available.
        prefer_side: 'remaining' (default) or 'starting'. Only meaningful
            when images_per_prior == 1.
        item_label: short item name used in the section header for clarity
            (e.g. "Sharp Cheddar Cheese").

    Returns:
        (text_block, image_pairs)
        - text_block: Markdown header + one bullet per prior session.
          Empty string if `priors` is empty or no images survived.
        - image_pairs: list of (label, b64) ready for the caller to inject
          as image content blocks, in the same order as the bullets.

    The returned text references each image by its label so a downstream
    observer can correlate the bullet with the image. The caller is
    responsible for actually inserting the images into the request payload.
    """
    if not priors:
        return "", []

    images: list[tuple[str, str]] = []
    bullet_lines: list[str] = []
    n = len(priors)
    other_side = "starting" if prefer_side == "remaining" else "remaining"

    for i, p in enumerate(priors, 1):
        sess = p.get("session", "?")
        sides_emitted: list[str] = []

        if images_per_prior == 0:
            # Numbers-only mode: do not emit images. Track which sides have
            # numeric data so the bullet still reports what's known.
            for side in ("starting", "remaining"):
                if p.get(f"{side}_amount") is not None:
                    sides_emitted.append(side)
        elif images_per_prior == 1:
            order = [prefer_side, other_side]
            for side in order:
                b64 = p.get(f"{side}_image_b64")
                if b64:
                    images.append((f"Prior {i}/{n} · {sess} · {side}", b64))
                    sides_emitted.append(side)
                    break
        else:
            for side in ("starting", "remaining"):
                b64 = p.get(f"{side}_image_b64")
                if b64:
                    images.append((f"Prior {i}/{n} · {sess} · {side}", b64))
                    sides_emitted.append(side)

        if not sides_emitted:
            continue

        # In numbers-only mode the side label is meaningless (no image to
        # match it to); leave it off the bullet header. Image modes keep
        # the side label so the model can pair the image with the bullet.
        if images_per_prior == 0:
            bits = [f"Prior {i}/{n} · session {sess}"]
        else:
            bits = [f"Prior {i}/{n} · session {sess} · {'+'.join(sides_emitted)}"]
        if include_amounts:
            amt_s = p.get("starting_amount")
            amt_r = p.get("remaining_amount")
            if amt_s is not None:
                bits.append(f"start={amt_s:g}{unit_label}")
            if amt_r is not None:
                bits.append(f"remain={amt_r:g}{unit_label}")
        bullet_lines.append("- " + " · ".join(bits))

    # Image modes need at least one surviving image; numbers-only mode
    # needs at least one bullet (which requires include_amounts to carry
    # any signal — if the caller forgot to pass it, the bullets degenerate
    # to just `Prior i/n · session …`, which is uninformative but harmless).
    if images_per_prior == 0:
        if not bullet_lines:
            return "", []
    elif not images:
        return "", []

    # The most-recent prior is the LAST observed state of this item before
    # today; framing it that way explicitly is what licenses the observer
    # to use it as a previous-state anchor instead of decoration.
    if images_per_prior == 0:
        # Numbers-only header: a single most-recent prior, no images. The
        # prior is the previous session's CLOSING amount — i.e., what this
        # observer recorded as remaining at the end of last session, which
        # is by definition this session's STARTING amount (modulo any
        # unrecorded handling between sessions).
        last = priors[-1]
        last_sess = last.get("session", "?")
        last_remain = last.get("remaining_amount")
        last_start = last.get("starting_amount")
        if last_remain is not None:
            anchor_str = f"{last_remain:g}{unit_label}"
            anchor_field = "amount_remaining"
        elif last_start is not None:
            anchor_str = f"{last_start:g}{unit_label}"
            anchor_field = "amount_starting"
        else:
            anchor_str = "(no value)"
            anchor_field = "amount_remaining"
        header = [
            f"## Previous-Session Anchor for {item_label}",
            (f"Last session ({last_sess}) the observer recorded "
             f"`{anchor_field} = {anchor_str}` at session end. Treat this "
             f"as the BEST AVAILABLE estimate of how much of this item "
             f"was in the stock container at the START of today's "
             f"session — i.e., today's `amount_starting` should equal "
             f"this number unless today's frames clearly show a fresh / "
             f"replacement container, or you can see consumption/spoilage "
             f"that occurred off-camera between sessions."),
            "",
            (f"This anchor is itself an observer estimate, not ground "
             f"truth — it could be off. Use it as a starting hypothesis "
             f"for today's `amount_starting` and a sanity-check ceiling "
             f"for today's `amount_remaining` (which cannot exceed the "
             f"anchor without a fresh container). Rely on today's frames "
             f"for the actual numeric reading."),
        ]
        # In numbers-only mode the anchor number is already inlined in the
        # header sentence; per-prior bullets would duplicate it. Return
        # the header alone.
        return "\n".join(header) + "\n", []
    else:
        header = [
            f"## Prior Observations for {item_label}",
            ("The image(s) below are saved snapshots of this item's STOCK "
             "CONTAINER from EARLIER sessions, oldest first. The LAST bullet "
             "is the most recent — it shows the container the LAST TIME this "
             "item was handled, and is the previous-state anchor for today."),
            "",
            ("How to use the most-recent prior:"),
            ("- Today's `amount_starting` should be at-or-below the prior "
             "`remaining` view (the stock cannot grow between sessions unless "
             "today's frames clearly show a fresh / replacement container)."),
            ("- Use the prior as a reference for what the container looked "
             "like at the end of the previous session. Compare it against "
             "today's frames to inform `amount_starting` and `amount_remaining`, "
             "but rely on today's frames for the actual reading — the prior "
             "is context, not ground truth."),
            ("- Earlier priors (if any) are context only — they show how this "
             "container's appearance has evolved across sessions (lighting, "
             "transparency, packaging quirks)."),
        ]
    return "\n".join(header) + "\n" + "\n".join(bullet_lines) + "\n", images

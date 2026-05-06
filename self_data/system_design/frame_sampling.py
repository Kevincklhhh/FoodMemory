"""Shared frame-sampling utilities for VLM observer/baseline scripts.

The same logic was duplicated in upperbound_amount.py and 06_avp_round1.py.
Both call sites now import from here. Each segment is anchored to the source
clip containing its raw start so padding can't bleed across clip boundaries
and pull frames from the wrong clip.
"""
from __future__ import annotations

import base64
import subprocess
from pathlib import Path
from typing import Iterable


def get_video_durations(paths: list[Path]) -> list[tuple[Path, float]]:
    """Return [(path, duration_s)] for each video file via ffprobe."""
    out = []
    for p in paths:
        res = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(p)],
            capture_output=True, text=True,
        )
        try:
            out.append((p, float(res.stdout.strip())))
        except (ValueError, AttributeError):
            out.append((p, 0.0))
    return out


def cumulative_to_video_offset(
    cumulative_time: float,
    video_durations: list[tuple[Path, float]],
) -> tuple[Path, float] | None:
    """Map a session-cumulative timestamp to (video_path, offset_within_clip).

    If `cumulative_time` falls past the last clip, returns the last clip with
    offset clamped just before EOF (rather than None). This makes downstream
    seeking safe even when callers pass slightly-out-of-range timestamps.
    """
    elapsed = 0.0
    for video_path, dur in video_durations:
        if cumulative_time < elapsed + dur:
            return video_path, cumulative_time - elapsed
        elapsed += dur
    if video_durations:
        last_path, last_dur = video_durations[-1]
        return last_path, max(0.0, last_dur - 0.1)
    return None


def extract_segments_frames(
    segments: Iterable[tuple[float, float]],
    video_durations: list[tuple[Path, float]],
    padding: float = 2.0,
    max_frames: int = 50,
    target_fps: float = 1.0,
) -> tuple[list[str], list[float]]:
    """Sample frames from a list of (start, end) session-time segments.

    Per-clip clamping: each segment is anchored to the source clip containing
    its RAW (unpadded) start; the padded window is then clamped to that clip's
    bounds. Per-clip overlapping intervals are merged.

    The total frame budget is `min(max_frames, round(target_fps * total_padded_dur))`,
    so short items don't burn the whole `max_frames` cap on a few seconds of
    video. The budget is allocated proportionally across merged intervals;
    each interval gets at least 1 frame. Samples are read directly from each
    source clip via cv2 seek (no concatenation).

    Returns (frames_b64, session_timestamps) as parallel lists in chronological
    session-time order. Frames are JPEG (quality 85), base64-encoded.
    """
    import cv2

    segments = list(segments)
    if not segments:
        return [], []

    # Cumulative offset of each clip's start in session time
    cum_starts: list[float] = []
    el = 0.0
    for _, d in video_durations:
        cum_starts.append(el)
        el += d

    # Build per-clip padded intervals with clamping
    intervals_per_clip: dict[str, list[tuple[float, float]]] = {}
    for raw_s, raw_e in segments:
        rs = cumulative_to_video_offset(raw_s, video_durations)
        if rs is None:
            continue
        vp, ros = rs
        vi = next(i for i, (p, _) in enumerate(video_durations) if p == vp)
        vd = video_durations[vi][1]
        offset_start = max(0.0, ros - padding)
        re_ = cumulative_to_video_offset(raw_e, video_durations)
        if re_ is not None and re_[0] == vp:
            offset_end = min(vd, re_[1] + padding)
        else:
            offset_end = vd  # raw segment spans clips → take to EOF of this clip
        intervals_per_clip.setdefault(str(vp), []).append((offset_start, offset_end))

    if not intervals_per_clip:
        return [], []

    # Merge per clip, then flatten with session-time metadata
    path_by_str = {str(p): p for p, _ in video_durations}
    merged_intervals: list[tuple[Path, float, float, float]] = []
    # (clip_path, clip_offset_start, clip_offset_end, session_cum_start)
    for vk, intervals in intervals_per_clip.items():
        intervals.sort()
        merged: list[tuple[float, float]] = []
        for s, e in intervals:
            if merged and s <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append((s, e))
        clip_path = path_by_str[vk]
        vi = next(i for i, (p, _) in enumerate(video_durations) if p == clip_path)
        clip_cum = cum_starts[vi]
        for cs, ce in merged:
            merged_intervals.append((clip_path, cs, ce, clip_cum + cs))

    merged_intervals.sort(key=lambda x: x[3])

    # Allocate frames: cap by target_fps so short items don't burn the budget.
    total_dur = sum(ce - cs for _, cs, ce, _ in merged_intervals)
    if total_dur <= 0:
        return [], []

    # Effective total budget for THIS item
    n_total = min(max_frames, max(1, round(target_fps * total_dur)))

    sample_specs: list[tuple[Path, float, float]] = []  # (clip_path, clip_offset, session_t)
    for clip_path, cs, ce, sess_start in merged_intervals:
        seg_dur = ce - cs
        n = max(1, round(n_total * seg_dur / total_dur))
        step = seg_dur / n
        for i in range(n):
            sample_specs.append((clip_path, cs + i * step, sess_start + i * step))
            if len(sample_specs) >= n_total:
                break
        if len(sample_specs) >= n_total:
            break

    # Read frames directly from each source clip
    frames_b64: list[str] = []
    timestamps: list[float] = []
    open_caps: dict[str, object] = {}
    try:
        for clip_path, offset, sess_t in sample_specs:
            key = str(clip_path)
            if key not in open_caps:
                cap = cv2.VideoCapture(key)
                if not cap.isOpened():
                    continue
                open_caps[key] = cap
            cap = open_caps[key]
            cap.set(cv2.CAP_PROP_POS_MSEC, offset * 1000)
            ret, frame = cap.read()
            if not ret:
                continue
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            frames_b64.append(base64.b64encode(buf).decode())
            timestamps.append(round(sess_t, 1))
    finally:
        for cap in open_caps.values():
            cap.release()

    return frames_b64, timestamps


def extract_single_frame(
    session_timestamp: float,
    video_durations: list[tuple[Path, float]],
) -> tuple[str, Path, float] | None:
    """Read a single JPEG at the given session-cumulative timestamp.

    Returns (base64_jpeg, clip_path, offset_in_clip) or None if the timestamp
    maps outside all clips or the read fails. Same JPEG quality (85) as
    `extract_segments_frames` so evidence frames look identical to live frames.
    """
    import cv2

    mapped = cumulative_to_video_offset(session_timestamp, video_durations)
    if mapped is None:
        return None
    clip_path, offset = mapped

    cap = cv2.VideoCapture(str(clip_path))
    try:
        if not cap.isOpened():
            return None
        cap.set(cv2.CAP_PROP_POS_MSEC, offset * 1000)
        ret, frame = cap.read()
        if not ret:
            return None
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    finally:
        cap.release()

    return base64.b64encode(buf).decode(), clip_path, offset

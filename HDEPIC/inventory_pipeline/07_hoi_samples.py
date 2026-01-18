#!/usr/bin/env python3
"""
07_hoi_samples.py - Extract and display sample HOI tracklet results with timestamps

Usage:
    python 07_hoi_samples.py --hoi_json /path/to/HOI_AMEGO/video.json --fps 30
    python 07_hoi_samples.py --video_id P01-20240203-123350 --root outputs/03_inventory_results --fps 30
    python 07_hoi_samples.py --video_id P01-20240203-123350 --root outputs/03_inventory_results --fps 30 --n 20

    # Query by time range
    python 07_hoi_samples.py --video_id P01-20240203-123350 --root outputs/03_inventory_results --start_time 00:02:00 --end_time 00:03:00
    python 07_hoi_samples.py --video_id P01-20240203-123350 --root outputs/03_inventory_results --start_time 120 --end_time 180
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict


def frame_to_timestamp(frame: int, fps: float) -> str:
    """Convert frame number to HH:MM:SS.mmm timestamp."""
    total_seconds = frame / fps
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"


def frame_to_seconds(frame: int, fps: float) -> float:
    """Convert frame number to seconds."""
    return frame / fps


def parse_timestamp(timestamp: str) -> float:
    """
    Parse a timestamp string to seconds.

    Supports formats:
    - "HH:MM:SS.mmm" or "HH:MM:SS" (e.g., "00:02:30.500")
    - "MM:SS.mmm" or "MM:SS" (e.g., "02:30")
    - Plain seconds as string (e.g., "150.5")

    Returns:
        Time in seconds as float
    """
    timestamp = timestamp.strip()

    # Try plain seconds first
    try:
        return float(timestamp)
    except ValueError:
        pass

    # Parse HH:MM:SS or MM:SS format
    parts = timestamp.split(":")
    if len(parts) == 3:
        # HH:MM:SS format
        hours, minutes, seconds = parts
        return float(hours) * 3600 + float(minutes) * 60 + float(seconds)
    elif len(parts) == 2:
        # MM:SS format
        minutes, seconds = parts
        return float(minutes) * 60 + float(seconds)
    else:
        raise ValueError(f"Invalid timestamp format: {timestamp}")


def query_tracklets_by_time(
    tracklets: List[Dict[str, Any]],
    fps: float,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    overlap_mode: str = "any"
) -> List[Dict[str, Any]]:
    """
    Query tracklets that fall within a specified time range.

    Args:
        tracklets: List of raw tracklet data from HOI JSON
        fps: Video frames per second
        start_time: Start time in seconds (None = from beginning)
        end_time: End time in seconds (None = to end)
        overlap_mode: How to match tracklets:
            - "any": Tracklet overlaps with time range at all (default)
            - "start": Tracklet starts within time range
            - "end": Tracklet ends within time range
            - "contain": Tracklet is fully contained within time range

    Returns:
        List of analyzed tracklets within the time range
    """
    results = []

    for tracklet in tracklets:
        frames = tracklet["num_frame"]
        tracklet_start = min(frames) / fps
        tracklet_end = max(frames) / fps

        # Default bounds
        query_start = start_time if start_time is not None else 0
        query_end = end_time if end_time is not None else float('inf')

        # Check overlap based on mode
        include = False
        if overlap_mode == "any":
            # Any overlap: tracklet overlaps with query range
            include = tracklet_start <= query_end and tracklet_end >= query_start
        elif overlap_mode == "start":
            # Tracklet starts within range
            include = query_start <= tracklet_start <= query_end
        elif overlap_mode == "end":
            # Tracklet ends within range
            include = query_start <= tracklet_end <= query_end
        elif overlap_mode == "contain":
            # Tracklet fully contained in range
            include = tracklet_start >= query_start and tracklet_end <= query_end

        if include:
            results.append(analyze_tracklet(tracklet, fps))

    return results


def get_tracklets_in_range(
    video_id: str,
    root: Path,
    start_time: str,
    end_time: str,
    fps: float = 30.0,
    overlap_mode: str = "any"
) -> List[Dict[str, Any]]:
    """
    Convenience function to get tracklets within a time range for a video.

    Args:
        video_id: Video identifier
        root: Root directory containing video results
        start_time: Start timestamp (supports "HH:MM:SS", "MM:SS", or seconds)
        end_time: End timestamp (supports "HH:MM:SS", "MM:SS", or seconds)
        fps: Video frames per second (default: 30)
        overlap_mode: "any", "start", "end", or "contain"

    Returns:
        List of analyzed tracklets in the time range

    Example:
        >>> tracklets = get_tracklets_in_range(
        ...     "P01-20240203-123350",
        ...     Path("outputs/03_inventory_results"),
        ...     "00:02:00",
        ...     "00:03:00"
        ... )
    """
    hoi_json_path = root / video_id / "HOI_AMEGO" / f"{video_id}.json"

    if not hoi_json_path.exists():
        raise FileNotFoundError(f"HOI JSON not found: {hoi_json_path}")

    # Load data
    with open(hoi_json_path, 'r') as f:
        tracklets = json.load(f)

    # Parse timestamps
    start_seconds = parse_timestamp(start_time)
    end_seconds = parse_timestamp(end_time)

    # Query
    return query_tracklets_by_time(
        tracklets, fps, start_seconds, end_seconds, overlap_mode
    )


def analyze_tracklet(tracklet: Dict[str, Any], fps: float) -> Dict[str, Any]:
    """Analyze a single tracklet and extract key information."""
    frames = tracklet["num_frame"]
    start_frame = min(frames)
    end_frame = max(frames)

    # Calculate duration
    duration_frames = end_frame - start_frame + 1
    duration_seconds = duration_frames / fps

    # Get bounding box statistics
    bboxes = tracklet["obj_bbox"]
    avg_x = sum(b[0] for b in bboxes) / len(bboxes)
    avg_y = sum(b[1] for b in bboxes) / len(bboxes)
    avg_w = sum(b[2] for b in bboxes) / len(bboxes)
    avg_h = sum(b[3] for b in bboxes) / len(bboxes)

    # Determine hand side
    sides = tracklet.get("side", [])
    if sides:
        left_count = sides.count("left")
        right_count = sides.count("right")
        dominant_side = "left" if left_count > right_count else "right"
    else:
        dominant_side = "unknown"

    return {
        "track_id": tracklet["track_id"],
        "cluster": tracklet.get("cluster", -1),
        "start_frame": start_frame,
        "end_frame": end_frame,
        "start_time": frame_to_timestamp(start_frame, fps),
        "end_time": frame_to_timestamp(end_frame, fps),
        "start_seconds": frame_to_seconds(start_frame, fps),
        "end_seconds": frame_to_seconds(end_frame, fps),
        "duration_seconds": duration_seconds,
        "num_detections": len(frames),
        "hand_side": dominant_side,
        "avg_bbox": {
            "x": round(avg_x, 1),
            "y": round(avg_y, 1),
            "w": round(avg_w, 1),
            "h": round(avg_h, 1)
        }
    }


def load_hoi_data(hoi_json_path: Path) -> List[Dict[str, Any]]:
    """Load HOI tracklet data from JSON file."""
    print(f"Loading HOI data from: {hoi_json_path}")
    with open(hoi_json_path, 'r') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} tracklets")
    return data


def get_summary_stats(analyzed: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Get summary statistics for all tracklets."""
    if not analyzed:
        return {}

    durations = [t["duration_seconds"] for t in analyzed]
    detections = [t["num_detections"] for t in analyzed]

    # Cluster distribution
    cluster_counts = defaultdict(int)
    for t in analyzed:
        cluster_counts[t["cluster"]] += 1

    # Hand side distribution
    side_counts = defaultdict(int)
    for t in analyzed:
        side_counts[t["hand_side"]] += 1

    return {
        "total_tracklets": len(analyzed),
        "duration_stats": {
            "min": round(min(durations), 2),
            "max": round(max(durations), 2),
            "avg": round(sum(durations) / len(durations), 2)
        },
        "detection_stats": {
            "min": min(detections),
            "max": max(detections),
            "avg": round(sum(detections) / len(detections), 1)
        },
        "clusters": dict(sorted(cluster_counts.items())),
        "hand_sides": dict(side_counts),
        "time_range": {
            "start": min(t["start_time"] for t in analyzed),
            "end": max(t["end_time"] for t in analyzed)
        }
    }


def print_tracklet(t: Dict[str, Any], idx: int) -> None:
    """Print a single tracklet in readable format."""
    print(f"\n{'─'*60}")
    print(f"Tracklet #{idx + 1} (ID: {t['track_id']}, Cluster: {t['cluster']})")
    print(f"{'─'*60}")
    print(f"  Time:     {t['start_time']} → {t['end_time']}")
    print(f"  Seconds:  {t['start_seconds']:.2f}s → {t['end_seconds']:.2f}s")
    print(f"  Duration: {t['duration_seconds']:.2f}s ({t['num_detections']} detections)")
    print(f"  Hand:     {t['hand_side']}")
    print(f"  Avg BBox: x={t['avg_bbox']['x']}, y={t['avg_bbox']['y']}, "
          f"w={t['avg_bbox']['w']}, h={t['avg_bbox']['h']}")


def export_tracklets(
    analyzed: List[Dict[str, Any]],
    output_path: Path,
    video_id: str,
    fps: float,
    query_start: Optional[float] = None,
    query_end: Optional[float] = None,
    overlap_mode: str = "any"
) -> Path:
    """
    Export tracklets to JSON file for downstream processing (e.g., VLM).

    Args:
        analyzed: List of analyzed tracklet dictionaries
        output_path: Path to output JSON file
        video_id: Video identifier
        fps: Video frames per second
        query_start: Query start time in seconds (if filtered)
        query_end: Query end time in seconds (if filtered)
        overlap_mode: Overlap mode used for filtering

    Returns:
        Path to the exported file
    """
    export_data = {
        "video_id": video_id,
        "fps": fps,
        "query": {
            "start_seconds": query_start,
            "end_seconds": query_end,
            "start_time": frame_to_timestamp(int(query_start * fps), fps) if query_start else None,
            "end_time": frame_to_timestamp(int(query_end * fps), fps) if query_end else None,
            "overlap_mode": overlap_mode
        },
        "summary": get_summary_stats(analyzed) if analyzed else {},
        "tracklets": analyzed
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(export_data, f, indent=2)

    print(f"\nExported {len(analyzed)} tracklets to: {output_path}")
    return output_path


def generate_output_filename(
    video_id: str,
    start_time: Optional[float],
    end_time: Optional[float],
    output_dir: Optional[Path] = None
) -> Path:
    """Generate a descriptive output filename based on query parameters."""
    if start_time is not None or end_time is not None:
        start_str = f"{int(start_time)}s" if start_time else "0s"
        end_str = f"{int(end_time)}s" if end_time else "end"
        filename = f"{video_id}_tracklets_{start_str}-{end_str}.json"
    else:
        filename = f"{video_id}_tracklets_all.json"

    if output_dir:
        return output_dir / filename
    return Path(filename)


# Keep old function name for backward compatibility
def export_samples(analyzed: List[Dict[str, Any]], output_path: Path,
                   video_id: str, fps: float) -> None:
    """Export sample results to JSON file (deprecated, use export_tracklets)."""
    export_tracklets(analyzed, output_path, video_id, fps)


def main():
    parser = argparse.ArgumentParser(
        description="Extract and display sample HOI tracklet results with timestamps"
    )

    # Input options
    parser.add_argument("--hoi_json", type=Path, help="Direct path to HOI JSON file")
    parser.add_argument("--video_id", type=str, help="Video ID")
    parser.add_argument("--root", type=Path, help="Root directory containing video results")

    # Processing options
    parser.add_argument("--fps", type=float, default=30.0, help="Video FPS (default: 30)")
    parser.add_argument("-n", "--num_samples", type=int, default=10,
                        help="Number of sample tracklets to display (default: 10)")
    parser.add_argument("--sort", type=str, default="duration",
                        choices=["duration", "start_time", "detections", "cluster"],
                        help="Sort tracklets by this field (default: duration)")
    parser.add_argument("--descending", action="store_true",
                        help="Sort in descending order")
    parser.add_argument("--cluster", type=int, help="Filter by specific cluster ID")
    parser.add_argument("--min_duration", type=float, default=0,
                        help="Minimum duration in seconds (default: 0)")

    # Time range filtering
    parser.add_argument("--start_time", type=str,
                        help="Start time (HH:MM:SS, MM:SS, or seconds)")
    parser.add_argument("--end_time", type=str,
                        help="End time (HH:MM:SS, MM:SS, or seconds)")
    parser.add_argument("--overlap_mode", type=str, default="any",
                        choices=["any", "start", "end", "contain"],
                        help="How to match tracklets to time range (default: any)")

    # Output options
    parser.add_argument("-o", "--output", type=Path,
                        help="Output file path for tracklets JSON (auto-generated if not specified)")
    parser.add_argument("--output_dir", type=Path,
                        help="Output directory (default: same as input)")
    parser.add_argument("--no_export", action="store_true",
                        help="Don't export results (display only)")
    parser.add_argument("--export", type=Path, help="(deprecated) Alias for --output")
    parser.add_argument("--all", action="store_true", help="Include all matching tracklets")
    parser.add_argument("--summary", action="store_true", help="Show only summary statistics")

    args = parser.parse_args()

    # Resolve HOI JSON path
    if args.hoi_json:
        hoi_json_path = args.hoi_json
        video_id = hoi_json_path.stem
    elif args.video_id and args.root:
        hoi_json_path = args.root / args.video_id / "HOI_AMEGO" / f"{args.video_id}.json"
        video_id = args.video_id
    else:
        parser.error("Must specify either --hoi_json or both --video_id and --root")

    if not hoi_json_path.exists():
        print(f"ERROR: HOI JSON file not found: {hoi_json_path}")
        return 1

    # Load and analyze data
    tracklets = load_hoi_data(hoi_json_path)

    # Parse time range (keep for export metadata)
    start_seconds = parse_timestamp(args.start_time) if args.start_time else None
    end_seconds = parse_timestamp(args.end_time) if args.end_time else None

    # Apply time range filter if specified
    if args.start_time or args.end_time:
        analyzed = query_tracklets_by_time(
            tracklets, args.fps, start_seconds, end_seconds, args.overlap_mode
        )
        time_range_str = f"{args.start_time or '0'} → {args.end_time or 'end'}"
        print(f"Filtered by time range ({args.overlap_mode}): {time_range_str}")
        print(f"  Found {len(analyzed)} tracklets in range")
    else:
        analyzed = [analyze_tracklet(t, args.fps) for t in tracklets]

    # Filter by cluster if specified
    if args.cluster is not None:
        analyzed = [t for t in analyzed if t["cluster"] == args.cluster]
        print(f"Filtered to cluster {args.cluster}: {len(analyzed)} tracklets")

    # Filter by minimum duration
    if args.min_duration > 0:
        analyzed = [t for t in analyzed if t["duration_seconds"] >= args.min_duration]
        print(f"Filtered by min duration {args.min_duration}s: {len(analyzed)} tracklets")

    # Sort tracklets
    sort_key = {
        "duration": lambda x: x["duration_seconds"],
        "start_time": lambda x: x["start_frame"],
        "detections": lambda x: x["num_detections"],
        "cluster": lambda x: (x["cluster"], -x["duration_seconds"])
    }[args.sort]

    analyzed.sort(key=sort_key, reverse=args.descending or args.sort == "duration")

    # Print summary
    print(f"\n{'='*60}")
    print(f"HOI Tracklet Analysis: {video_id}")
    print(f"{'='*60}")

    stats = get_summary_stats(analyzed)
    print(f"\nSummary Statistics:")
    print(f"  Total tracklets: {stats['total_tracklets']}")
    print(f"  Time range: {stats['time_range']['start']} → {stats['time_range']['end']}")
    print(f"  Duration: min={stats['duration_stats']['min']}s, "
          f"max={stats['duration_stats']['max']}s, avg={stats['duration_stats']['avg']}s")
    print(f"  Detections per tracklet: min={stats['detection_stats']['min']}, "
          f"max={stats['detection_stats']['max']}, avg={stats['detection_stats']['avg']}")
    print(f"  Hand sides: {stats['hand_sides']}")
    print(f"  Unique clusters: {len(stats['clusters'])}")

    # Show cluster distribution (top 10)
    sorted_clusters = sorted(stats['clusters'].items(), key=lambda x: -x[1])[:10]
    print(f"\n  Top clusters: {dict(sorted_clusters)}")

    if args.summary:
        return 0

    # Print sample tracklets
    num_to_show = len(analyzed) if args.all else min(args.num_samples, len(analyzed))
    print(f"\n{'='*60}")
    print(f"Sample Tracklets (showing {num_to_show} of {len(analyzed)}, sorted by {args.sort})")
    print(f"{'='*60}")

    for i, t in enumerate(analyzed[:num_to_show]):
        print_tracklet(t, i)

    # Determine if we should export
    # Auto-export when time range is specified (unless --no_export)
    should_export = not args.no_export and (
        args.output or args.export or args.output_dir or
        (args.start_time or args.end_time)  # Auto-export for time queries
    )

    if should_export:
        # Determine output path
        output_path = args.output or args.export
        if output_path is None:
            # Auto-generate filename
            output_dir = args.output_dir or (args.root / video_id / "HOI_AMEGO" if args.root else Path("."))
            output_path = generate_output_filename(
                video_id, start_seconds, end_seconds, output_dir
            )

        # Export all matching tracklets (not just displayed samples)
        export_tracklets(
            analyzed,
            output_path,
            video_id,
            args.fps,
            query_start=start_seconds,
            query_end=end_seconds,
            overlap_mode=args.overlap_mode
        )

    return 0


if __name__ == "__main__":
    exit(main())

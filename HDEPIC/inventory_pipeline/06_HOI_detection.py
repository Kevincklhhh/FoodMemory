#!/usr/bin/env python3
"""
06_HOI_detection.py - Run AMEGO pipeline for Hand-Object Interaction detection

This script orchestrates the AMEGO pipeline stages:
1. Frame extraction (ffmpeg)
2. Optical flow extraction (Flowformer)
3. Hand-object bounding box detection (Faster R-CNN)
4. HOI tracklet extraction (HOI_AMEGO)
5. Location segment extraction (LS_AMEGO)

Usage:
    python 06_HOI_detection.py --video_path /path/to/video.mp4 --fps 30 --output_dir /path/to/output
    python 06_HOI_detection.py --video_id P01-20240203-123350 --root /path/to/outputs --fps 30  # Resume existing
    python 06_HOI_detection.py --check --video_id P01-20240203-123350 --root /path/to/outputs  # Check status only
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any

# AMEGO model directory
AMEGO_DIR = Path(__file__).parent.parent / "models" / "AMEGO"


def check_pipeline_status(video_dir: Path, video_id: str) -> Dict[str, Any]:
    """Check the completion status of each pipeline stage."""
    status = {
        "video_id": video_id,
        "video_dir": str(video_dir),
        "stages": {}
    }

    # Stage 1: RGB Frames
    rgb_frames_dir = video_dir / "rgb_frames"
    if rgb_frames_dir.exists():
        frame_count = len(list(rgb_frames_dir.glob("frame_*.jpg")))
        status["stages"]["rgb_frames"] = {
            "complete": frame_count > 0,
            "count": frame_count
        }
    else:
        status["stages"]["rgb_frames"] = {"complete": False, "count": 0}

    # Stage 2: Flowformer
    flowformer_dir = video_dir / "flowformer"
    if flowformer_dir.exists():
        flow_count = len(list(flowformer_dir.glob("flow_*.pth")))
        status["stages"]["flowformer"] = {
            "complete": flow_count > 0,
            "count": flow_count
        }
    else:
        status["stages"]["flowformer"] = {"complete": False, "count": 0}

    # Stage 3: Hand-Object Detection
    hand_objects_dir = video_dir / "hand-objects"
    pkl_file = hand_objects_dir / f"{video_id}.pkl" if hand_objects_dir.exists() else None
    status["stages"]["hand_objects"] = {
        "complete": pkl_file is not None and pkl_file.exists(),
        "pkl_exists": pkl_file is not None and pkl_file.exists()
    }

    # Stage 4: HOI_AMEGO
    hoi_dir = video_dir / "HOI_AMEGO"
    if hoi_dir.exists():
        hoi_files = list(hoi_dir.glob("*.json"))
        status["stages"]["hoi_amego"] = {
            "complete": len(hoi_files) > 0,
            "files": [f.name for f in hoi_files]
        }
    else:
        status["stages"]["hoi_amego"] = {"complete": False, "files": []}

    # Stage 5: LS_AMEGO
    ls_dir = video_dir / "LS_AMEGO"
    if ls_dir.exists():
        ls_files = list(ls_dir.glob("*.json"))
        status["stages"]["ls_amego"] = {
            "complete": len(ls_files) > 0,
            "files": [f.name for f in ls_files]
        }
    else:
        status["stages"]["ls_amego"] = {"complete": False, "files": []}

    return status


def print_status(status: Dict[str, Any]) -> None:
    """Print pipeline status in a readable format."""
    print(f"\n{'='*60}")
    print(f"AMEGO Pipeline Status: {status['video_id']}")
    print(f"Directory: {status['video_dir']}")
    print(f"{'='*60}\n")

    stage_names = {
        "rgb_frames": "1. RGB Frame Extraction",
        "flowformer": "2. Optical Flow (Flowformer)",
        "hand_objects": "3. Hand-Object Detection",
        "hoi_amego": "4. HOI Tracklet Extraction",
        "ls_amego": "5. Location Segment Extraction"
    }

    for stage_key, stage_name in stage_names.items():
        stage_info = status["stages"].get(stage_key, {})
        complete = stage_info.get("complete", False)
        status_icon = "[OK]" if complete else "[--]"

        details = ""
        if "count" in stage_info:
            details = f" ({stage_info['count']} files)"
        elif "files" in stage_info and stage_info["files"]:
            details = f" ({len(stage_info['files'])} files)"

        print(f"  {status_icon} {stage_name}{details}")

    print()


def run_command(cmd: list, cwd: Optional[Path] = None, env: Optional[dict] = None) -> int:
    """Run a command and return the exit code."""
    print(f"\n[CMD] {' '.join(cmd)}")
    if cwd:
        print(f"[CWD] {cwd}")

    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)

    result = subprocess.run(cmd, cwd=cwd, env=merged_env)
    return result.returncode


def extract_frames(video_path: Path, output_dir: Path, fps: float) -> int:
    """Extract frames from video using ffmpeg."""
    rgb_frames_dir = output_dir / "rgb_frames"
    rgb_frames_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vf", f"fps={fps},scale=456:256",
        str(rgb_frames_dir / "frame_%010d.jpg")
    ]

    return run_command(cmd)


def extract_flowformer(video_id: str, root: Path, fps: float, conda_env: str = "amego") -> int:
    """Extract optical flow using Flowformer."""
    cmd = [
        "conda", "run", "-n", conda_env, "--no-capture-output",
        "python", "-m", "tools.generate_flowformer_flow",
        "--root", str(root),
        "--v_id", video_id,
        "--dset", "video",
        "--models_root", "submodules/flowformer/models",
        "--model", "sintel",
        "--video_fps", str(fps)
    ]

    return run_command(cmd, cwd=AMEGO_DIR)


def extract_hand_objects(video_id: str, root: Path, conda_env: str = "handobj") -> int:
    """Extract hand-object bounding boxes using Faster R-CNN."""
    # Use absolute paths to avoid issues with cwd
    root_abs = root.resolve()
    rgb_frames_dir = root_abs / video_id / "rgb_frames"
    hand_objects_dir = root_abs / video_id / "hand-objects"
    hand_objects_dir.mkdir(parents=True, exist_ok=True)

    pb_file = f"{video_id}.pb2"
    pb_path = AMEGO_DIR.resolve() / pb_file

    # Run detection with absolute path
    cmd = [
        "conda", "run", "-n", conda_env, "--no-capture-output",
        "python", "-m", "tools.extract_bboxes",
        "--image_dir", str(rgb_frames_dir),
        "--cuda", "--mGPUs",
        "--checksession", "1",
        "--checkepoch", "8",
        "--checkpoint", "132028",
        "--bs", "32",
        "--detections_pb", str(pb_path)
    ]

    ret = run_command(cmd, cwd=AMEGO_DIR)
    if ret != 0:
        return ret

    # Get frame dimensions (assuming 456x256 from prepare_video.sh)
    frame_height = 256
    frame_width = 456

    pkl_path = hand_objects_dir / f"{video_id}.pkl"

    convert_cmd = [
        "conda", "run", "-n", conda_env, "--no-capture-output",
        "python", "-m", "submodules.epic-kitchens-100-hand-object-bboxes.src.scripts.convert_raw_to_releasable_detections",
        str(pb_path),
        str(pkl_path),
        "--frame-height", str(frame_height),
        "--frame-width", str(frame_width)
    ]

    ret = run_command(convert_cmd, cwd=AMEGO_DIR)
    if ret != 0:
        return ret

    # Move pb2 file to hand-objects directory
    if pb_path.exists():
        import shutil
        shutil.move(str(pb_path), str(hand_objects_dir / pb_file))

    return 0


def run_hoi_amego(video_id: str, root: Path, fps: float, conda_env: str = "amego") -> int:
    """Run HOI_AMEGO to extract interaction tracklets."""
    root_abs = root.resolve()
    cmd = [
        "conda", "run", "-n", conda_env, "--no-capture-output",
        "python", "HOI_AMEGO.py",
        "--dset", "video",
        "--v_id", video_id,
        "--root", str(root_abs),
        "--output_dir", str(root_abs),
        "--video_fps", str(fps)
    ]

    return run_command(cmd, cwd=AMEGO_DIR)


def run_ls_amego(video_id: str, root: Path, fps: float, conda_env: str = "amego") -> int:
    """Run LS_AMEGO to extract location segments."""
    root_abs = root.resolve()
    cmd = [
        "conda", "run", "-n", conda_env, "--no-capture-output",
        "python", "LS_AMEGO.py",
        "--dset", "video",
        "--v_id", video_id,
        "--root", str(root_abs),
        "--output_dir", str(root_abs),
        "--video_fps", str(fps)
    ]

    return run_command(cmd, cwd=AMEGO_DIR)


def run_full_pipeline(
    video_path: Optional[Path],
    video_id: str,
    root: Path,
    fps: float,
    start_stage: int = 1,
    end_stage: int = 5,
    amego_env: str = "amego",
    handobj_env: str = "handobj"
) -> int:
    """Run the full AMEGO pipeline or a subset of stages."""

    video_dir = root / video_id
    video_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Running AMEGO Pipeline")
    print(f"Video ID: {video_id}")
    print(f"Root: {root}")
    print(f"FPS: {fps}")
    print(f"Stages: {start_stage} to {end_stage}")
    print(f"{'='*60}")

    # Stage 1: Frame extraction
    if start_stage <= 1 <= end_stage:
        print("\n[Stage 1/5] Extracting RGB frames...")
        rgb_frames_dir = video_dir / "rgb_frames"
        if rgb_frames_dir.exists() and len(list(rgb_frames_dir.glob("frame_*.jpg"))) > 0:
            print("  Skipping - frames already exist")
        else:
            if video_path is None:
                print("  ERROR: video_path required for frame extraction")
                return 1
            ret = extract_frames(video_path, video_dir, fps)
            if ret != 0:
                print(f"  FAILED with exit code {ret}")
                return ret
            print("  Done!")

    # Stage 2: Optical flow
    if start_stage <= 2 <= end_stage:
        print("\n[Stage 2/5] Extracting optical flow (Flowformer)...")
        flowformer_dir = video_dir / "flowformer"
        rgb_frames_dir = video_dir / "rgb_frames"
        expected_flows = len(list(rgb_frames_dir.glob("frame_*.jpg"))) - 1 if rgb_frames_dir.exists() else 0

        if flowformer_dir.exists() and len(list(flowformer_dir.glob("flow_*.pth"))) >= expected_flows * 0.99:
            print(f"  Skipping - flow files already exist ({len(list(flowformer_dir.glob('flow_*.pth')))} files)")
        else:
            ret = extract_flowformer(video_id, root, fps, amego_env)
            if ret != 0:
                print(f"  FAILED with exit code {ret}")
                return ret
            print("  Done!")

    # Stage 3: Hand-object detection
    if start_stage <= 3 <= end_stage:
        print("\n[Stage 3/5] Extracting hand-object detections...")
        pkl_file = video_dir / "hand-objects" / f"{video_id}.pkl"
        if pkl_file.exists():
            print("  Skipping - detections already exist")
        else:
            ret = extract_hand_objects(video_id, root, handobj_env)
            if ret != 0:
                print(f"  FAILED with exit code {ret}")
                return ret
            print("  Done!")

    # Stage 4: HOI_AMEGO
    if start_stage <= 4 <= end_stage:
        print("\n[Stage 4/5] Extracting HOI tracklets (HOI_AMEGO)...")
        hoi_dir = video_dir / "HOI_AMEGO"
        if hoi_dir.exists() and len(list(hoi_dir.glob("*.json"))) > 0:
            print("  Skipping - HOI tracklets already exist")
        else:
            ret = run_hoi_amego(video_id, root, fps, amego_env)
            if ret != 0:
                print(f"  FAILED with exit code {ret}")
                return ret
            print("  Done!")

    # Stage 5: LS_AMEGO
    if start_stage <= 5 <= end_stage:
        print("\n[Stage 5/5] Extracting location segments (LS_AMEGO)...")
        ls_dir = video_dir / "LS_AMEGO"
        if ls_dir.exists() and len(list(ls_dir.glob("*.json"))) > 0:
            print("  Skipping - location segments already exist")
        else:
            ret = run_ls_amego(video_id, root, fps, amego_env)
            if ret != 0:
                print(f"  FAILED with exit code {ret}")
                return ret
            print("  Done!")

    print(f"\n{'='*60}")
    print("Pipeline completed successfully!")
    print(f"{'='*60}\n")

    # Print final status
    status = check_pipeline_status(video_dir, video_id)
    print_status(status)

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Run AMEGO pipeline for Hand-Object Interaction detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline on a new video
  python 06_HOI_detection.py --video_path /path/to/video.mp4 --fps 30 --output_dir ./outputs

  # Resume pipeline for existing video (skips completed stages)
  python 06_HOI_detection.py --video_id P01-20240203-123350 --root ./outputs/03_inventory_results --fps 30

  # Check pipeline status only
  python 06_HOI_detection.py --check --video_id P01-20240203-123350 --root ./outputs/03_inventory_results

  # Run specific stages (e.g., only hand-object detection)
  python 06_HOI_detection.py --video_id P01-20240203-123350 --root ./outputs --fps 30 --start_stage 3 --end_stage 3
        """
    )

    # Input options
    parser.add_argument("--video_path", type=Path, help="Path to input video file")
    parser.add_argument("--video_id", type=str, help="Video ID (directory name)")
    parser.add_argument("--root", type=Path, help="Root directory for outputs")
    parser.add_argument("--output_dir", type=Path, help="Alias for --root")
    parser.add_argument("--fps", type=float, default=30.0, help="Video FPS (default: 30)")

    # Stage control
    parser.add_argument("--start_stage", type=int, default=1, choices=[1,2,3,4,5],
                        help="Start from this stage (default: 1)")
    parser.add_argument("--end_stage", type=int, default=5, choices=[1,2,3,4,5],
                        help="End at this stage (default: 5)")

    # Environment options
    parser.add_argument("--amego_env", type=str, default="amego",
                        help="Conda environment for AMEGO (default: amego)")
    parser.add_argument("--handobj_env", type=str, default="handobj",
                        help="Conda environment for hand-object detection (default: handobj)")

    # Status check
    parser.add_argument("--check", action="store_true",
                        help="Only check pipeline status, don't run anything")

    args = parser.parse_args()

    # Resolve root directory
    root = args.root or args.output_dir
    if root is None:
        if args.video_path:
            root = args.video_path.parent
        else:
            print("ERROR: Must specify --root or --output_dir")
            sys.exit(1)

    # Resolve video_id
    video_id = args.video_id
    if video_id is None:
        if args.video_path:
            video_id = args.video_path.stem
        else:
            print("ERROR: Must specify --video_id or --video_path")
            sys.exit(1)

    video_dir = root / video_id

    # Check status only
    if args.check:
        if not video_dir.exists():
            print(f"ERROR: Video directory does not exist: {video_dir}")
            sys.exit(1)
        status = check_pipeline_status(video_dir, video_id)
        print_status(status)
        sys.exit(0)

    # Run pipeline
    ret = run_full_pipeline(
        video_path=args.video_path,
        video_id=video_id,
        root=root,
        fps=args.fps,
        start_stage=args.start_stage,
        end_stage=args.end_stage,
        amego_env=args.amego_env,
        handobj_env=args.handobj_env
    )

    sys.exit(ret)


if __name__ == "__main__":
    main()

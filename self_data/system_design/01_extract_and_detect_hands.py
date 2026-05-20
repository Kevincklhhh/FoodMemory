#!/usr/bin/env python3
"""
01_extract_and_detect_hands.py - Extract frames from session clips and run hands23 detector.

Processes all clips in a session: extracts frames at specified FPS, then runs
the hands23 HOI detector on each frame. Tracks cumulative session-relative
timestamps across clips.

Usage:
    # Activate hands23 environment first:
    conda activate hands23

    python system_design/01_extract_and_detect_hands.py --participant kailai --session 20260310-195710
    python system_design/01_extract_and_detect_hands.py --participant kailai --session 20260310-195710 --fps 1.0
    python system_design/01_extract_and_detect_hands.py --participant kailai --session 20260310-195710 --skip-detection

Prerequisites:
    - Session clips in participants/{P}/videos/{session}/*.mp4
    - hands23 model weights at kitchen/HDEPIC/models/hands23_detector/model_weights/model_hands23.pth
    - conda environment 'hands23' with detectron2 installed
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
try:
    import torch  # noqa: F401  (used for peak GPU memory reporting)
except ImportError:
    torch = None
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

# Project paths
_SCRIPT_DIR = Path(__file__).parent
_SELF_DATA = _SCRIPT_DIR.parent
_KITCHEN_DIR = _SELF_DATA.parent
_HDEPIC_DIR = _KITCHEN_DIR / "HDEPIC"
_HANDS23_DIR = _HDEPIC_DIR / "models" / "hands23_detector"

sys.path.insert(0, str(_HANDS23_DIR))

from utils import (  # noqa: E402
    get_session_clips,
    get_sessions,
    hands23_dir,
    outputs_dir,
    participant_dir,
)

HANDS23_CONFIG = _HANDS23_DIR / "faster_rcnn_X_101_32x8d_FPN_3x_Hands23.yaml"
HANDS23_WEIGHTS = _HANDS23_DIR / "model_weights" / "model_hands23.pth"
FONT_PATH = _HANDS23_DIR / "utils" / "times_b.ttf"

# Visualization colors (RGB)
HAND_COLORS = {
    "left_hand": (0, 90, 181),
    "right_hand": (220, 50, 32),
}
FIRST_OBJ_COLOR = (255, 194, 10)
SECOND_OBJ_COLOR = (0, 159, 115)


# =============================================================================
# FRAME EXTRACTION
# =============================================================================

def extract_frames(
    video_path: Path,
    output_dir: Path,
    fps: float = 1.0,
) -> List[Path]:
    """Extract frames from video at given fps. Returns list of saved frame paths."""
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  ERROR: Could not open {video_path}")
        return []

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / video_fps if video_fps > 0 else 0
    print(f"  {video_fps:.1f} fps, {total_frames} frames, {duration:.1f}s")

    interval = max(1, int(video_fps / fps)) if fps > 0 else 1
    extracted = []
    frame_idx = 0
    pbar = tqdm(total=total_frames // interval, desc="  Extracting", leave=False)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % interval == 0:
            ts = frame_idx / video_fps
            name = f"frame_{frame_idx:08d}_t{ts:.2f}s.jpg"
            path = output_dir / name
            cv2.imwrite(str(path), frame)
            extracted.append(path)
            pbar.update(1)
        frame_idx += 1

    pbar.close()
    cap.release()
    return extracted


# =============================================================================
# HANDS23 DETECTION
# =============================================================================

def load_hands23_predictor(
    hand_thresh: float = 0.7,
    first_obj_thresh: float = 0.5,
    second_obj_thresh: float = 0.3,
    hand_rela: float = 0.3,
    obj_rela: float = 0.7,
):
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor
    from hodetector.data import register_ho_pascal_voc, hoMapper  # noqa: F401
    from hodetector.modeling import roi_heads  # noqa: F401

    cfg = get_cfg()
    cfg.merge_from_file(str(HANDS23_CONFIG))
    cfg.MODEL.WEIGHTS = str(HANDS23_WEIGHTS)
    cfg.HAND = hand_thresh
    cfg.FIRSTOBJ = first_obj_thresh
    cfg.SECONDOBJ = second_obj_thresh
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = min(hand_thresh, first_obj_thresh, second_obj_thresh)
    cfg.HAND_RELA = hand_rela
    cfg.OBJ_RELA = obj_rela
    cfg.freeze()
    return DefaultPredictor(cfg)


def _parse_contact(v: int) -> str:
    return {0: "no_contact", 1: "other_person_contact", 2: "self_contact",
            3: "object_contact", 4: "obj_to_obj_contact"}.get(v, f"unknown_{v}")


def _parse_grasp(v: int) -> str:
    return {0: "NP-Palm", 1: "NP-Fin", 2: "Pow-Pris", 3: "Pre-Pris",
            4: "Pow-Circ", 5: "Pre-Circ", 6: "Later", 7: "Other"}.get(v, f"unknown_{v}")


def _parse_touch(v: int) -> str:
    return {0: "tool_touched", 1: "tool_held", 2: "tool_used",
            3: "container_touched", 4: "container_held",
            5: "neither_touched", 6: "neither_held"}.get(v, f"unknown_{v}")


def run_hands23_detection(predictor, image: np.ndarray) -> List[Dict[str, Any]]:
    """Run hands23 on a BGR image. Returns list of hand detection dicts."""
    outputs = predictor(image)
    inst = outputs["instances"]
    pred_boxes = inst.get("pred_boxes").tensor.cpu().numpy()
    pred_dz = inst.get("pred_dz").cpu().numpy()
    pred_classes = inst.get("pred_classes").cpu().numpy()
    pred_scores = inst.get("scores").cpu().numpy()
    pred_masks = inst.get("pred_masks").cpu().numpy()

    interaction = pred_dz[:, 4]
    hand_side = pred_dz[:, 5]
    grasp = pred_dz[:, 6]
    touch_type = pred_dz[:, 7]
    contact_state = pred_dz[:, 8]

    hands = []
    for i in range(len(pred_classes)):
        if pred_classes[i] != 0:  # 0 = hand class
            continue
        info = {
            "hand_bbox": pred_boxes[i].tolist(),
            "hand_side": "right_hand" if hand_side[i] == 1 else "left_hand",
            "contact_state": _parse_contact(int(contact_state[i])),
            "grasp": _parse_grasp(int(grasp[i])),
            "hand_score": float(pred_scores[i]),
            "obj_bbox": None,
            "obj_touch": None,
            "obj_score": None,
            "second_obj_bbox": None,
            "second_obj_score": None,
        }
        if interaction[i] >= 0:
            oid = int(interaction[i])
            info["obj_bbox"] = pred_boxes[oid].tolist()
            info["obj_touch"] = _parse_touch(int(touch_type[oid]))
            info["obj_score"] = float(pred_scores[oid])
            if interaction[oid] >= 0:
                soid = int(interaction[oid])
                info["second_obj_bbox"] = pred_boxes[soid].tolist()
                info["second_obj_score"] = float(pred_scores[soid])
        hands.append(info)

    return hands


# =============================================================================
# VISUALIZATION
# =============================================================================

def _draw_bbox(draw, bbox, color, width=3):
    draw.rectangle(bbox, outline=color, width=width)


def _draw_text_bg(draw, pos, text, font, text_color, bg_color):
    bb = draw.textbbox(pos, text, font=font)
    p = 4
    draw.rectangle([bb[0]-p, bb[1]-p, bb[2]+p, bb[3]+p], fill=bg_color)
    draw.text(pos, text, font=font, fill=text_color)


def visualize_detections(image: np.ndarray, detections: List[Dict], output_path: Path):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    im = Image.fromarray(image_rgb).convert("RGBA")
    draw = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype(str(FONT_PATH), size=20)
        small = ImageFont.truetype(str(FONT_PATH), size=16)
    except Exception:
        font = small = ImageFont.load_default()

    for hand in detections:
        hb = hand["hand_bbox"]
        hcolor = HAND_COLORS.get(hand["hand_side"], (128, 128, 128))
        _draw_bbox(draw, hb, hcolor)
        side = "L" if hand["hand_side"] == "left_hand" else "R"
        label = f"{side}: {hand['contact_state']}"
        _draw_text_bg(draw, (hb[0], max(0, hb[1]-25)), label, small, (0,0,0), (255,255,255,200))
        if hand["obj_bbox"]:
            ob = hand["obj_bbox"]
            _draw_bbox(draw, ob, FIRST_OBJ_COLOR)
            hc = ((hb[0]+hb[2])/2, (hb[1]+hb[3])/2)
            oc = ((ob[0]+ob[2])/2, (ob[1]+ob[3])/2)
            draw.line([hc, oc], fill=hcolor, width=2)
            _draw_text_bg(draw, (ob[0], max(0, ob[1]-25)), hand.get("obj_touch","obj"), small, (0,0,0), (255,255,255,200))
            if hand["second_obj_bbox"]:
                sb = hand["second_obj_bbox"]
                _draw_bbox(draw, sb, SECOND_OBJ_COLOR)
                sc = ((sb[0]+sb[2])/2, (sb[1]+sb[3])/2)
                draw.line([oc, sc], fill=FIRST_OBJ_COLOR, width=2)

    im.convert("RGB").save(output_path)


# =============================================================================
# MAIN
# =============================================================================

def run_session(
    participant: str,
    session: str,
    fps: float = 1.0,
    skip_detection: bool = False,
    no_visualization: bool = False,
) -> int:
    clips = get_session_clips(participant, session)
    if not clips:
        print(f"  No clips found for session {session}")
        return 1

    h23_dir = hands23_dir(participant, session)
    h23_dir.mkdir(parents=True, exist_ok=True)

    predictor = None
    if not skip_detection:
        print("Loading hands23 detector...")
        if not HANDS23_WEIGHTS.exists():
            print(f"ERROR: Weights not found at {HANDS23_WEIGHTS}")
            return 1
        try:
            predictor = load_hands23_predictor()
            print("Hands23 loaded OK")
        except Exception as e:
            print(f"ERROR loading hands23: {e}")
            print("  conda activate hands23")
            return 1

    print(f"\n{'='*70}")
    print(f"SESSION {session} | participant={participant} | {len(clips)} clips | {fps} fps")
    print(f"{'='*70}")

    all_videos = []
    total_frames = 0
    total_hands = 0
    clip_offset = 0.0
    timing = {
        "frame_extraction_s": 0.0,
        "detection_s": 0.0,
        "visualization_s": 0.0,
        "image_read_s": 0.0,
    }
    if torch is not None and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    for clip_idx, (filename, clip_path, dur) in enumerate(clips):
        clip_stem = Path(filename).stem
        print(f"\n[{clip_idx+1}/{len(clips)}] {filename} ({dur:.1f}s, offset {clip_offset:.1f}s)")

        frames_dir = h23_dir / clip_stem / "frames"
        vis_dir = h23_dir / clip_stem / "visualizations"

        # Extract frames
        t_ex = time.time()
        frame_paths = extract_frames(clip_path, frames_dir, fps=fps)
        timing["frame_extraction_s"] += time.time() - t_ex
        print(f"  Extracted {len(frame_paths)} frames")
        total_frames += len(frame_paths)

        video_result = {
            "video_id": clip_stem,
            "clip_offset_s": round(clip_offset, 3),
            "num_frames": len(frame_paths),
            "total_hands": 0,
            "frames": [],
        }

        if predictor is not None and frame_paths:
            if not no_visualization:
                vis_dir.mkdir(parents=True, exist_ok=True)

            hands_in_clip = 0
            for frame_path in tqdm(frame_paths, desc="  Detecting", leave=False):
                t_io = time.time()
                image = cv2.imread(str(frame_path))
                timing["image_read_s"] += time.time() - t_io
                if image is None:
                    continue

                t_det = time.time()
                detections = run_hands23_detection(predictor, image)
                timing["detection_s"] += time.time() - t_det
                num_hands = len(detections)
                hands_in_clip += num_hands

                # Parse timestamp from filename: frame_XXXXXXXX_tN.NNs.jpg
                clip_ts = float(frame_path.stem.split("_t")[1].rstrip("s"))
                session_ts = round(clip_offset + clip_ts, 3)

                # Serialize detections (drop mask arrays)
                serial_dets = [{k: v for k, v in d.items() if not k.endswith("_mask")}
                               for d in detections]

                frame_rec = {
                    "frame_path": str(frame_path.relative_to(h23_dir)),
                    "clip_timestamp_s": round(clip_ts, 3),
                    "session_timestamp_s": session_ts,
                    "num_hands": num_hands,
                    "detections": serial_dets,
                }

                if not no_visualization and detections:
                    vis_path = vis_dir / f"vis_{frame_path.name}"
                    t_vis = time.time()
                    visualize_detections(image, detections, vis_path)
                    timing["visualization_s"] += time.time() - t_vis
                    frame_rec["visualization_path"] = str(vis_path.relative_to(h23_dir))

                video_result["frames"].append(frame_rec)

            print(f"  Detected {hands_in_clip} hands")
            total_hands += hands_in_clip
            video_result["total_hands"] = hands_in_clip

        all_videos.append(video_result)
        clip_offset += dur

    # Save results
    timing_out = {k: round(v, 2) for k, v in timing.items()}
    timing_out["counts"] = {
        "num_clips": len(clips),
        "num_frames_extracted": total_frames,
        "num_hands_detected": total_hands,
        "total_video_duration_s": round(clip_offset, 2),
    }
    if total_frames:
        timing_out["detection_s_per_frame"] = round(timing["detection_s"] / total_frames, 4)
    if torch is not None and torch.cuda.is_available():
        timing_out["peak_gpu_mem_allocated_gb"] = round(
            torch.cuda.max_memory_allocated() / 1024**3, 3)
        timing_out["peak_gpu_mem_reserved_gb"] = round(
            torch.cuda.max_memory_reserved() / 1024**3, 3)
        timing_out["gpu_device_name"] = torch.cuda.get_device_name(0)
    results = {
        "participant": participant,
        "session": session,
        "fps": fps,
        "timestamp": datetime.now().isoformat(),
        "total_frames": total_frames,
        "total_hands": total_hands,
        "timing": timing_out,
        "videos": all_videos,
    }

    out_file = h23_dir / f"{participant}_{session}_hands23_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*70}")
    print(f"DONE: {session}")
    print(f"  Clips:   {len(clips)}")
    print(f"  Frames:  {total_frames}")
    if predictor is not None:
        print(f"  Hands:   {total_hands}")
    print(f"  Timing:  extract={timing['frame_extraction_s']:.1f}s  "
          f"read={timing['image_read_s']:.1f}s  "
          f"detect={timing['detection_s']:.1f}s  "
          f"vis={timing['visualization_s']:.1f}s")
    print(f"  Saved:   {out_file}")

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Extract frames from session clips and run hands23 HOI detector"
    )
    parser.add_argument("--participant", required=True, help="Participant ID (e.g., kailai)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--session", help="Session ID (e.g., 20260310-195710)")
    group.add_argument("--all", action="store_true", help="Process all sessions")
    parser.add_argument("--fps", type=float, default=1.0, help="Frame extraction rate (default: 1.0)")
    parser.add_argument("--skip-detection", action="store_true", help="Extract frames only, skip detection")
    parser.add_argument("--no-visualization", action="store_true", help="Skip visualization output")
    parser.add_argument("--resume", action="store_true", help="Skip sessions with existing results")

    args = parser.parse_args()

    if args.all:
        sessions = get_sessions(args.participant)
        print(f"Processing all {len(sessions)} sessions for {args.participant}")
        for i, session in enumerate(sessions):
            print(f"\n{'#'*70}")
            print(f"# SESSION {i+1}/{len(sessions)}: {session}")
            print(f"{'#'*70}")

            if args.resume:
                result_file = hands23_dir(args.participant, session) / f"{args.participant}_{session}_hands23_results.json"
                if result_file.exists():
                    print(f"  SKIPPED (results exist: {result_file.name})")
                    continue

            run_session(
                participant=args.participant,
                session=session,
                fps=args.fps,
                skip_detection=args.skip_detection,
                no_visualization=args.no_visualization,
            )
    else:
        sys.exit(run_session(
            participant=args.participant,
            session=args.session,
            fps=args.fps,
            skip_detection=args.skip_detection,
            no_visualization=args.no_visualization,
        ))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
00_run_adatad.py — Run AdaTAD temporal action detection on session videos.

Thin wrapper that invokes the AdaTAD scripts in models/OpenTAD/ (which depend
on the opentad package and must run from that directory).

Produces: participants/<P>/outputs/<session>/adatad_detections.json
    (consumed by 04_evaluate_adatad_segments.py and 05a_adatad_item_label.py)

Usage:
    python system_design/00_run_adatad.py --participant kailai
    python system_design/00_run_adatad.py --participant kailai --session 20260310-195710
    python system_design/00_run_adatad.py --participant kailai --resume
    python system_design/00_run_adatad.py --participant kailai --device cuda:7
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


OPENTAD_DIR = Path(__file__).resolve().parent.parent / "models" / "OpenTAD"
SCRIPT = "run_adatad_all_sessions.py"


def main():
    parser = argparse.ArgumentParser(
        description="Run AdaTAD inference (wrapper around models/OpenTAD/)"
    )
    parser.add_argument("--participant", required=True)
    parser.add_argument("--session", default=None, help="Single session (default: all)")
    parser.add_argument("--resume", action="store_true", help="Skip sessions with existing results")
    parser.add_argument("--device", default="cuda:0", help="GPU device (default: cuda:0)")
    parser.add_argument("--tasks", nargs="+", choices=["noun", "verb"], default=["verb"],
                        help="Which AdaTAD heads to run (default: verb only — AVP doesn't use noun).")
    args = parser.parse_args()

    script_path = OPENTAD_DIR / SCRIPT
    if not script_path.exists():
        print(f"ERROR: {script_path} not found")
        sys.exit(1)

    cmd = [
        sys.executable, str(script_path),
        "--participant", args.participant,
        "--device", args.device,
        "--tasks", *args.tasks,
    ]
    if args.session:
        cmd += ["--session", args.session]
    if args.resume:
        cmd += ["--resume"]

    print(f"Running AdaTAD from {OPENTAD_DIR}")
    print(f"  {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(OPENTAD_DIR))
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()

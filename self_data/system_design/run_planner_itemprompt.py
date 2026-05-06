"""Run the item-based planner prompt (build_planner_prompt) on a single
session, evidence-mode=blocks, planner-only. Saves a planner.json compatible
with the analyze_burst_variants.py recall/frames analysis (translates per-item
segments → dense_windows).

Reuses everything from 06_avp_round1_remaining_minimal.py.
"""
import argparse
import importlib.util
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

_HERE = Path(__file__).resolve().parent
_KITCHEN_DIR = _HERE.parent
load_dotenv(_KITCHEN_DIR / ".env")
sys.path.insert(0, str(_HERE))

# Import the minimal module by file path (filename starts with a digit).
_spec = importlib.util.spec_from_file_location(
    "minimal_avp", _HERE / "06_avp_round1_remaining_minimal.py"
)
_min = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_min)

OUTPUT_PREFIX = "avp_minimal_remaining"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--participant", default="kailai")
    p.add_argument("--session", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--model", default="gpt-5.4")
    p.add_argument("--evidence-mode", default="blocks",
                   choices=["blocks", "segments", "chrono", "per_frame"])
    p.add_argument("--max-frames", type=int, default=100)
    p.add_argument("--min-score", type=float, default=0.15)
    p.add_argument("--inventory-scope", default="full")
    p.add_argument("--seg-tau-dino", type=float, default=0.15)
    p.add_argument("--seg-gap-close", type=float, default=2.0)
    p.add_argument("--seg-min-duration", type=float, default=1.5)
    p.add_argument("--flicker-min-score", type=float, default=0.15)
    p.add_argument("--flicker-min-hits", type=int, default=2)
    p.add_argument("--flicker-peak-score", type=float, default=0.25)
    p.add_argument("--block-gap-close", type=float, default=0.0)
    p.add_argument("--max-block-s", type=float, default=None)
    p.add_argument("--crosstalk-dino-ratio", type=float, default=0.4)
    p.add_argument("--crosstalk-cov-floor", type=float, default=0.7)
    args = p.parse_args()

    participant = args.participant
    session = args.session
    model = args.model

    inventory = _min.load_inventory(participant, session,
                                    scope=args.inventory_scope)
    if not inventory:
        print(f"  {session}: no inventory")
        return
    all_ts, hoi_ts = _min.load_hoi_timestamps(participant, session)
    if not hoi_ts:
        print(f"  {session}: no HOI frames")
        return
    dino_by_t = _min.load_dino_by_t(participant, session)
    scene_by_t = _min.load_owlv2_scene_by_t(participant, session)
    hoi_details_by_t = _min.load_hoi_details_by_t(participant, session)
    transparency_by_iid = _min.load_transparency_profile(participant)

    prompt, ev_stats, active_vcs, active_iids = _min.build_planner_prompt(
        participant=participant,
        session=session,
        inventory=inventory,
        transparency_by_iid=transparency_by_iid,
        dino_by_t=dino_by_t,
        scene_by_t=scene_by_t,
        hoi_details_by_t=hoi_details_by_t,
        hoi_sorted=sorted(hoi_ts),
        evidence_mode=args.evidence_mode,
        min_score=args.min_score,
        max_frames=args.max_frames,
        inventory_scope=args.inventory_scope,
        seg_tau_dino=args.seg_tau_dino,
        seg_gap_close=args.seg_gap_close,
        seg_min_duration=args.seg_min_duration,
        flicker_min_score=args.flicker_min_score,
        flicker_min_hits=args.flicker_min_hits,
        flicker_peak_score=args.flicker_peak_score,
        block_gap_close=args.block_gap_close,
        max_block_s=args.max_block_s,
        crosstalk_dino_ratio=args.crosstalk_dino_ratio,
        crosstalk_cov_floor=args.crosstalk_cov_floor,
    )

    cache_dir = _min.CACHE_DIR / participant / session / model.replace("/", "_") / args.tag
    cache_dir.mkdir(parents=True, exist_ok=True)

    client = _min.make_client()
    print(f"  {session}: evidence_stats {ev_stats}")
    print(f"  Planner (item-based) → {model}")
    response_text, stats = _min.run_planner(
        client, prompt, model,
        prompt_save_path=cache_dir / "planner_prompt.txt",
    )
    (cache_dir / "planner_response.txt").write_text(response_text or "")

    item_decisions, observation_plan = _min.parse_planner_response(response_text)
    n_observe = sum(1 for d in item_decisions if d.get("decision") == "observe")
    n_skip = sum(1 for d in item_decisions if d.get("decision") == "no_observation")
    print(f"  Decisions: {len(item_decisions)} (observe={n_observe}, "
          f"no_observation={n_skip})  observation_plan={len(observation_plan)} items")

    # Translate per-item segments → dense_windows shape so the same analysis
    # script (analyze_burst_variants) can score it directly.
    dense_windows = []
    for op in observation_plan:
        iid = op.get("instance_id")
        if not iid:
            continue
        for seg in op.get("segments", []):
            if not seg or len(seg) < 2:
                continue
            try:
                s, e = float(seg[0]), float(seg[1])
            except (TypeError, ValueError):
                continue
            if e <= s:
                continue
            dense_windows.append({
                "start": round(s, 2), "end": round(e, 2),
                "target_items": [iid],
            })
    dense_windows.sort(key=lambda d: d["start"])

    planner_log = {
        "n_decisions_total": len(item_decisions),
        "n_decisions_observe": n_observe,
        "n_decisions_no_observation": n_skip,
        "evidence_mode": args.evidence_mode,
        "evidence_stats": ev_stats,
        "item_decisions": item_decisions,
        "journey_samples": [],
        "dense_windows": dense_windows,
        "observation_plan": observation_plan,
        "stats": stats,
        "prompt": prompt,
        "raw_response": response_text,
    }

    sess_out = _min.outputs_dir(participant, session)
    sess_out.mkdir(parents=True, exist_ok=True)
    out_path = sess_out / f"{OUTPUT_PREFIX}_{model.replace('/', '_')}_{args.tag}_planner.json"
    out_path.write_text(json.dumps({
        "participant": participant,
        "timestamp": datetime.now().isoformat(),
        "model": model,
        "tag": args.tag,
        "session": {
            "session": session,
            "planner": planner_log,
            "sweep": {}, "planner_r2": {}, "sweep_r2": {}, "observer": [],
        },
    }, indent=2))
    print(f"  Saved: {out_path}")


if __name__ == "__main__":
    main()

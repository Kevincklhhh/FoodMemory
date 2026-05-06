# AVP Round 1 — duplicate-iid investigation & pipeline fixes

_Session notes, 2026-04-22_

This doc captures everything we learned while comparing the `noTAD_v3` and
`SceneTransp_v1` runs on kailai, chasing down why items were missed, and the
design sketches for the next iteration.  Pick this back up when you want to
resume.

---

## 1. Benchmark comparison: noTAD_v3 vs SceneTransp_v1 (kailai, 69 sessions)

| Metric | noTAD_v3 | SceneTransp_v1 | Δ |
|---|---:|---:|---:|
| CNPE_rem mean | **26.4 %** | 28.5 % | +2.1 pp |
| CNPE_rem median | 21.0 % | 21.1 % | +0.1 pp |
| CNPE_rem std | 23.4 | 25.5 | +2.1 |
| Matched / Missed / Hallucinated | **283** / 37 / 54 | 270 / 50 / 54 | −13 / +13 / 0 |
| Observer in+out tokens | 8.39 M | **6.00 M** | −28.5 % |
| Observer frames | 7 152 | **4 890** | −31.6 % |
| Total tokens | 8.80 M | **6.44 M** | −26.8 % |
| Total wall time | 28 031 s | 27 195 s | −3.0 % |

**Takeaway.** SceneTransp buys a ~30 % observer-token cut but loses recall
(−13 matches) and widens the error tail (+2 pp std).  Median unchanged.
Paired per-item: 98 SceneTransp wins, 106 noTAD wins, 57 ties — a different
operating point, not a strict improvement.  Big SceneTransp regressions are
0→full-package flips (3 × 0→100 % CNPE on Ground Beef, Chicken Wings,
Cheddar).

Reports:
* `participants/kailai/outputs/avp_noTAD_remaining_gpt-5.4_noTAD_v3_eval.html`
* `participants/kailai/outputs/avp_SceneTransp_remaining_gpt-5.4_SceneTransp_v1_eval.html`

---

## 2. Where the 50 SceneTransp misses actually go

Traced every missed item through `per_item_segments.json` and the
SceneTransp planner output:

| Root cause | Count |
|---|---:|
| No DINO reference signal at all (upstream data gap in 03) | 10 |
| Filtered by discontinuity in 05b (`gap_close`/`min_duration`) | 6 |
| **Dict-keyed-by-visual-class overwrite bug in 05b** | 5 |
| Planner saw item, said `no_observation` (pruned as cross-talk / twin) | 12 |
| Planner said `observe` but observer produced no prediction | 16 |
| Planner had segments but omitted item from decision list | 1 |

SigLIP peak similarity is **0.000 for all 21 "never reached planner" items**,
so DINO is the only signal doing useful work.

---

## 3. `05b_per_item_segments.py` — three filtering mechanisms

From line 106 and 111–148:

1. **`active(t)` gate.**  `(t in hoi_ts) AND (siglip >= τ OR dino >= τ)`.
   DINO hits outside hand-object-contact frames are ignored.
2. **`gap_close` merge.**  Runs separated by more than `gap_close` (default
   2.0 s) stay independent.
3. **`min_duration` open.**  Runs whose span < `min_duration` (default 1.5 s)
   are dropped.

Empirical consequence: scattered bursts (1-frame hits separated by 6 s) get
killed even when the DINO signal is strong.  Example: Orange Juice at
`20260319-093205` had **5 hot frames, peak dino 0.70, max gap 6 s** — all
runs died at `min_duration`.  Relaxing to `gap_close=5 s, min_duration=0.5 s`
would have recovered 4 of 6 discontinuity-filtered items.

---

## 4. The visual_class dict-overwrite bug (now **fixed**)

**Symptom.**  Sessions with two still-active purchase instances of the same
product (two egg cartons, two milk gallons) only emitted segments for *one*
of the iids.  Whichever iid was iterated second overwrote the first in
`per_item` (a dict keyed by `visual_class`), so `per_item_segments.json`
contained a single iid per class.

**Empirically verified** on `20260403-102128`: both
`large_white_eggs_20260310` and `large_white_eggs_20260403` received
**byte-identical** segments from `morphological_segments` (53 active frames,
same peak_dino 0.36, same windows), but only the newer iid survived into the
JSON.

**Fix (committed in the file).**
* `build_session_segments` (line ~285) — key `per_item` by `instance_id`
  instead of `visual_class`; sort key changed to `(visual_class.lower(),
  instance_id)` for deterministic ordering.
* `write_segments_json` (line ~321) — loop var renamed, `"visual_class"`
  value now read from `inv["visual_class"]` instead of the former dict key.
* `print_report` (line ~395) — same key-by-iid change.
* `print_chronological_timeline` (line ~243) — loop var renamed.

**Verified fix.**  Regenerated `20260403-102128` (42 vs 38 segments; both egg
iids present) and `20260404-105251` (38 vs 35 segments).

**To fully deploy:**
```bash
python system_design/05b_per_item_segments.py --participant kailai --all --write
python system_design/05b_per_item_segments.py --participant giraffe --all --write
```

---

## 5. What the planner actually does with duplicate iids

Ran `dup_check_v1` (planner-only) on two regenerated sessions.  GPT-5.4's
behavior is *not* what I initially predicted:

### SceneTransp planner (episodes printed without iid)

* **20260403-102128** — `large_white_eggs_20260310` → `no_observation`
  (*"older entry is most plausibly the same physical carton cross-matched
  twice"*); `large_white_eggs_20260403` → `observe`.  Same pattern for
  `olive_oil_20260310` vs `extra_virgin_olive_oil_20260402`.
* **20260404-105251** — only `large_white_eggs_20260403` emitted a
  decision; the older iid was silently dropped from the output.

Dedup direction is **always newer-iid-wins**.  Since the ledger assigns GT
consumption to the **older** carton by FIFO, the planner's choice is the
wrong one for evaluation — eval will still record MISS for the older iid and
HALL for the newer iid.

### PerFrame planner (each line has `iid_A=0.31 iid_B=0.31`)

* **20260403-102128** — *both* egg iids and *both* oil iids returned
  `no_observation`.  Reasoning: *"perfectly time-locked with <sibling> at
  identical scores… evidence cannot be assigned to this specific instance."*

The symmetric per-frame layout actively paralyzes the planner.  Worse
outcome than SceneTransp.

### Plain bottom line
* **No duplicate observer calls ever happen** — my earlier "two observer
  calls" guess was wrong.
* **Neither variant is biased toward FIFO**, so a clean 05b fix alone won't
  recover the GT iid in eval.
* **PerFrame can refuse both iids** when evidence is symmetric, giving a
  worse recall outcome than SceneTransp.

---

## 6. Experiment set: sessions with duplicate iids in inventory

Saved to `participants/kailai/outputs/duplicate_iid_sessions.json` — **29 of
74 kailai sessions**, 0 of 20 giraffe sessions.  Three distinct collisions
drive all 29:

| Visual class | Colliding iids | # sessions |
|---|---|---:|
| Whole Milk Gallon | `_20260310`, `_20260318` | 14 |
| Whole Milk Gallon | `_20260318`, `_20260406` | 3 |
| Chobani Vanilla Greek Yogurt | `_20260318`, `_20260328` | 3 |
| Large White Eggs | `_20260310`, `_20260403` | 9 |

This is the dup-iid benchmark set for future variants.

---

## 7. `06_avp_round1_remaining_PerFrame.py` — new variant (written, not run)

Created to bypass 05b entirely.  Sends raw per-frame HOI + SigLIP + DINO +
OWLv2 scene tags directly to the planner.  Each line = one HOI-contact
frame, e.g.
```
[   52.3s]  stove    organic_firm_tofu_20260310=0.72
[   71.3s]  unknown  napa_cabbage_20260310=0.75  kimchi_20260310=0.51
[  645.6s]  storage  orange_juice_20260310=0.61  whole_milk_gallon_20260310=0.29
```

**All non-planner code is identical to the noTAD version** — observer
prompt, observer invocation loop, parse_observer_response, cache layout,
CLI/resume/observer-only scaffolding all copied verbatim.  Deltas:
* planner input source (raw per-frame JSONs vs `per_item_segments.json`)
* planner prompt (per-frame timeline + burst/scene/transparency heuristics)
* new `--min-score` flag (default 0.15)
* `OUTPUT_PREFIX = "avp_PerFrame_remaining"`, new cache dir
* transparency tags in inventory listing (inherited from SceneTransp)

**Token projection (kailai):**
* Mean 11-min session → ~6 K planner tokens
* Longest (72 min, session 20260404-192616) → ~33 K planner tokens
* All within any LLM budget.

**Known issue discovered post-creation:** on symmetric duplicate-iid
evidence, PerFrame refuses both iids (see §5).  Needs the candidate-list
refactor below before running at scale.

---

## 8. Pending: window-centric planner + candidate-aware observer

The user's proposed redesign, not yet implemented.  Key idea: make the
planner emit **observation windows** (not item decisions), each window
carrying a list of candidate iids that match that segment's visual
evidence.  The observer then picks one candidate per window.

### New planner output schema (sketch)
```json
{
  "observation_windows": [
    {
      "start": 37.0, "end": 41.0,
      "candidates": [
        {"item": "Large White Eggs", "instance_id": "large_white_eggs_20260310"},
        {"item": "Large White Eggs", "instance_id": "large_white_eggs_20260403"}
      ],
      "reasoning": "Storage retrieval of an egg carton; identical dino scores on both iids — observer must decide.",
      "confidence": "high"
    },
    {
      "start": 122.0, "end": 126.0,
      "candidates": [{"item": "Whole Milk Gallon", "instance_id": "whole_milk_gallon_20260318"}],
      "reasoning": "Single unambiguous match.",
      "confidence": "high"
    }
  ],
  "skipped_items": [
    {"instance_id": "beef_top_sirloin_steak_20260403", "reasoning": "cross-talk only"}
  ]
}
```

### New observer contract
Input: frames + candidate list + session inventory context.
Output:
```json
{
  "item_confirmed": true,
  "chosen_instance_id": "large_white_eggs_20260310",
  "chosen_reasoning": "Identical packaging; applying FIFO tiebreaker to oldest iid.",
  "reasoning": "<original step-by-step>",
  "evidence_frames": [...],
  "amount_remaining": 11
}
```

### Tie-break rule (open question)
Default proposal: when visually indistinguishable, pick the oldest trailing
`YYYYMMDD` in the iid (FIFO — matches ledger consumption convention).
Alternatives to consider: newest, random, observer chooses freely.

### Open decisions before coding
1. Build as a **new file** `06_avp_round1_remaining_PerFrameCand.py`, or
   modify PerFrame in place (it's only smoke-tested, not yet run fully)?
2. When a real iid appears in two non-overlapping windows (storage
   retrieval + sink-side use), pick the **later** window's prediction
   for the final remaining amount?  Or average?  Or keep per-window
   predictions and merge in eval?
3. How does this interact with the `confusable_profile.json` / transparency
   tag logic?  Probably just passes through unchanged.

---

## 9. Files changed / created in this session

* `system_design/05b_per_item_segments.py` — visual_class → instance_id
  keying fix (§4).  **Not yet re-run on `--all` after edit.**
* `system_design/06_avp_round1_remaining_PerFrame.py` — new variant (§7),
  syntax-checked and smoke-tested on one session, not run at scale.
* `participants/kailai/outputs/avp_SceneTransp_remaining_gpt-5.4_dup_check_v1_*`
  — planner-only exploratory run on 2 sessions.
* `participants/kailai/outputs/avp_PerFrame_remaining_gpt-5.4_dup_check_v1_*`
  — planner-only exploratory run on 1 session.
* `participants/kailai/outputs/20260403-102128/per_item_segments.json`,
  `participants/kailai/outputs/20260404-105251/per_item_segments.json` —
  regenerated with the fix.
* `participants/kailai/outputs/duplicate_iid_sessions.json` — experiment
  set (§6).
* This note: `system_design/NOTES_avp_duplicate_iids.md`.

---

## 10. Suggested next steps (for when we resume)

1. **Regenerate 05b for all kailai sessions** to get the fix into every
   `per_item_segments.json`:
   ```
   python system_design/05b_per_item_segments.py --participant kailai --all --write
   ```
2. **Rerun SceneTransp & noTAD on the 29-session dup-iid benchmark only**
   to quantify how much the 05b fix moves the needle on its own (before
   adding any candidate-list redesign).
3. **Decide on the candidate-list variant** (new file vs modify PerFrame;
   FIFO vs observer-chooses).  Implement it.  Run on the 29-session dup-iid
   set first, then the full 69-session kailai set if promising.
4. **Upstream DINO data gaps** — 10 items never produce scores because their
   reference embeddings are missing (Organic Carrots, Noodles, Whole
   Branzini, Raw Shrimp, Beef Chuck Roast, Organic Yellow Potato, Vine Ripe
   Tomato).  Need to regenerate DINO reference crops in step 03.

# Eval runs log

One line per run: eval filename + what changed vs. the previous run in its
branch. Metrics (sessions, CNPE, tokens) live in the eval JSONs themselves.

## Main AVP branch (TAD + planner + observer)

- `avp_gpt-5.4_default_preds_eval.json` — baseline.
- `avp_gpt-5.4_refs_v2_preds_eval.json` — added DINO reference anchors (?).
- `avp_remaining_gpt-5.4_remaining_v1_preds_eval.json` — switched to remaining-only task.
- `avp_remaining_gpt-5.4_remaining_v1_cutoff331_preds_eval.json` — same as v1 with sessions capped at 2026-03-31.
- `avp_remaining_gpt-5.4_remaining_v2_preds_eval.json` — planner/observer prompt revision.
- `avp_remaining_gpt-5.4_remaining_v2_detailhigh_preds_eval.json` — observer frames at `image_detail=high` (?).

## No-TAD branch

- `avp_noTAD_remaining_gpt-5.4_noTAD_v1_preds_eval.json` — baseline no-AdaTAD, uses 05b per-item segments.
- `avp_noTAD_remaining_gpt-5.4_noTAD_v2_preds_eval.json` — planner re-tune + session set 60→66.
- `avp_noTAD_remaining_gpt-5.4_noTAD_v3_preds_eval.json` — observer prompt rewritten (stock-container vs taken-out-portion distinction); 05b τ_dino 0.10→0.15; planner display filter 0.10→0.15.

## No-planner branch

- `avp_noplanner_remaining_gpt-5.4_noplanner_v1_preds_eval.json` — baseline with planner disabled; every inventory item reaches the observer.

## Lowerbound (whole-video e2e VLM)

- `lowerbound_gemini-2.5-flash_preds_eval.json` — baseline Flash.
- `lowerbound_gemini-2.5-pro_gt_items_eval.json` — switched to Pro + prior is GT-used items only.
- `lowerbound_gemini-2.5-pro_noisyprior_v1_preds_eval.json` — prior broadened from GT items to full inventory.
- `lowerbound_remaining_gemini-2.5-pro_remaining_batch_v1_preds_eval.json` — remaining-only, batched across sessions.
- `lowerbound_remaining_gemini-2.5-pro_remaining_batch_v2_preds_eval.json` — v2 iteration of the batched remaining run.

## Upperbound (observer on GT segments)

- `upperbound_gpt-5.4_preds_eval.json` — pilot GPT upperbound.
- `upperbound_gpt-5.4_noisyprior_v1_preds_eval.json` — added noisy prior (full inventory) and expanded coverage.
- `upperbound_gemini-2.5-pro_preds_eval.json` — swapped GPT for Gemini Pro.
- `upperbound_gemini-2.5-pro_noisyprior_v1_preds_eval.json` — Gemini Pro + noisy prior.
- `upperbound_video_gemini-2.5-pro_noisyprior_video_preds_eval.json` — frames → video input mode.
- `upperbound_gemma_preds_eval.json` — swapped model to local Gemma.

## PerFrame branch (raw per-frame HOI+DINO+SigLIP+OWLv2; no 05b)

- `avp_PerFrame_remaining_gpt-5.4_PerFrame_v1_preds_eval.json` — baseline PerFrame. Per-frame evidence with iid-keyed DINO tokens (sibling iids print side-by-side at identical scores); planner uses FIFO-pick for duplicate iids; observer prompt identical to SceneTransp (single visual_class target, can sum phantom siblings).

## SceneTransp branch

- `avp_SceneTransp_remaining_gpt-5.4_SceneTransp_v1_preds_eval.json` — planner reads 05b per-item episodes + OWLv2 scene anchors + transparency tags; FIFO-pick for duplicate iids; same single-target observer.

## CandList branch (candidate-aware, `06_avp_round1_remaining_CandList.py`)

- `avp_CandList_remaining_gpt-5.4_CandList_dup29_preds_eval.json` — first CandList run, restricted to the 29-session dup-iid benchmark (`duplicate_iid_sessions.json`). Three changes vs PerFrame_v1: (1) per-frame evidence tokens collapsed by `visual_class` (max DINO across sibling iids — siblings never appear side-by-side); (2) no FIFO in planner — it emits `observation_windows` each with `candidate_instance_ids` listing every iid of that class; (3) observer is candidate-aware — returns a `per_instance` array with `handling_status` ∈ {handled, visible_untouched, not_visible} and a `remaining` per candidate, so sibling tubs are assigned explicitly instead of being summed or dropped. Also fixed planner retry-sentinel (`item_decisions` → `observation_windows`) so planner no longer does 5 spurious retries per call.

## Minimal sweep + R2 gap-fill branch (`06_avp_round1_remaining_minimal*.py`)

Sweep-only variant (one planner call → one observer call) using the new
4-field amount schema (`amount_starting`, `amount_remaining`,
`amount_derivative`) plus a journey/dense planner output. R2 is a single
gap-fill replan triggered when an item is `used` but lacks direct
`amount_remaining` AND can't be derived from `starting − derivative`.

All runs in this branch share: gemini-3.1-pro-preview, evidence-mode blocks,
media_resolution_low (280 tok/img), thinking-level high, sweep-budget 12/8,
rounds=2, full inventory scope. Only the script (parent vs lazyPSO vs r2free)
and `--burst-coverage` flag vary across rows.

- `avp_minimal_remaining_gemini-3.1-pro-preview_blocks_burst_late_r2_mr280_tlhigh_preds_eval.json` — **baseline**, `06_avp_round1_remaining_minimal.py`, `--burst-coverage late`. R1 observer returns one item per candidate plus a **top-level flat `per_segment_observations` (PSO) covering every segment 1..N**. R2 planner re-uses the blocks-evidence formatter with R1 PSO inlined under each DINO block as `↳ R1 Seg N [kind]: …`; prompt forbids R2 windows from overlapping R1 segments + post-hoc script-side filter drops any R2 journey/dense entry whose t-range overlaps an R1 segment; planner is told to "find UNSAMPLED bursts". R2 observer is told status is fixed at `used` and to fill only the missing field(s); per-field merge keeps R1's value for any non-null field. **10 sessions: matched=57/354, missed=297, hall=4, CNPE_rem micro mean=28.4%, macro mean=27.2%; cost $2.15 (487k tokens), R2 fired 3 sessions, 6 windows total.**
- `avp_minimal_remaining_gemini-3.1-pro-preview_lazyPSO_smoke_101158_preds_eval.json` — lazyPSO variant (1-session smoke), `06_avp_round1_remaining_minimal_lazyPSO.py`. R1 observer prompt change: PSO is **nested under each item** and emitted ONLY when ≥2 of 3 amount fields are null (i.e. items that will become R2 candidates). Items the observer pinned down omit PSO entirely. R2 planner's block-evidence inlining tags each note with `(instance_id)` so multiple unresolved items can each describe the same segment without collision. Reduces R1 observer output bulk; downstream R2 plumbing unchanged.
- `avp_minimal_remaining_gemini-3.1-pro-preview_r2free_mr280_tlhigh_preds_eval.json` — r2free, `06_avp_round1_remaining_minimal_r2free.py`, **`--burst-coverage all`** (default). Two coupled R2 changes vs baseline: (1) **overlap restriction dropped** — prompt rule "MUST NOT overlap any R1 segment" + post-hoc filter both removed; R2 may freely revisit any moment, including inside an R1 sampled range. (2) **R1 inlined notes show actual sampled t-range** (`↳ R1 Seg 353.1-363.1s [kind]: …`) instead of segment index + planner-declared window. Also pruned redundancies + biased steering ("earliest preferred", "EARLIER" qualifier, "find UNSAMPLED bursts" guidance, the "30s window with 5 frames" illustration). **10 sessions: matched=58, missed=296, hall=2, CNPE_rem micro mean=29.4%, macro mean=27.2%; cost $2.31 (507k tokens), R2 fired 6 sessions, 41 windows.** Caveat: `--burst-coverage all` flag flip vs baseline confounds the comparison — broader R1 sampling (early bursts included) made R1 leave 13 items unresolved (vs 4 in baseline), inflating R2 firing.
- `avp_minimal_remaining_gemini-3.1-pro-preview_r2free_late_mr280_tlhigh_preds_eval.json` — r2free apples-to-apples, same script as above but **`--burst-coverage late`** to match baseline's R1 regime. Plus a bug fix: per-item history block in R2 planner prompt now renders R1 segments by t-range (`3.5s [journey], 485.5-493.5s [dense]`) instead of integer indices, so labels match the inlined `↳ R1 Seg <ts>-<te>s` lines in the evidence pool below. **10 sessions: matched=57, missed=297, hall=3, CNPE_rem micro mean=30.1%, macro mean=28.5%, macro median=24.5% (slightly better than baseline's 25.9%); cost $2.04 (423k tokens, −5.4% vs baseline), R2 fired 2 sessions, 13 windows.** With burst-coverage controlled, the r2free changes are **roughly neutral** on bottom-line CNPE (recall tied, hall slightly down, mean CNPE slightly worse, cost slightly down). R2-firing pattern is disjoint from baseline (different sessions had unresolved items each run — pure R1 stochasticity).
- `avp_minimal_remaining_gemini-3.1-pro-preview_r2free_late_mr280_tlhigh_jdedup_preds_eval.json` — **current proposed approach (2026-04-28)**, same script + flags as the prior `r2free_late` row (`06_avp_round1_remaining_minimal_r2free.py`, `--burst-coverage late`, mr280, tlhigh). One change: **journey-sample dedup moved from prompt to script**. The planner-prompt sentence "Do NOT place a journey sample for an item inside any dense window that already targets that same item" was deleted; instead a post-hoc `drop_journey_inside_dense()` pass (script lines ~2447, called at ~3523 for R1 and ~3762 for R2) removes a (journey_sample, item) pair when its timestamp falls inside any dense window already targeting that item. Same dedup is applied to R2 journey/dense. **10 sessions: matched=59/354, missed=295, hall=4, CNPE_rem micro mean=27.47%, median=21.47%, macro mean (n=17)=25.62%, macro median=27.96%; R2 fired 4 sessions.**

### evframe series — schema + prompt iteration on `06_avp_round1_remaining_minimal_r2free_evidence_v2.py`

Reference baseline for this sub-branch is `_r2free_no_evidence.py` (the canonical `r2free_late_mr280_tlhigh_jdedup` script, frozen as a backup). All evframe runs use identical Gemini params + 10-session kailai set; only the script's prompt schema changes. Tag prefix `r2free_late_mr280_tlhigh_jdedup_evframe…`.

- `avp_minimal_remaining_gemini-3.1-pro-preview_r2free_late_mr280_tlhigh_jdedup_evframe_preds_eval.json` — **evframe v1**: schema gains three `*_evidence` string fields (`amount_starting_evidence`, `amount_remaining_evidence`, `amount_derivative_evidence`) parallel to the amount triple. Each evidence string cites `t=<sec>s` timestamps + ≤6-word caption; non-null iff the corresponding amount is non-null. R2 prompt + parser + R2-merge + predictions record updated end-to-end. Annotator wired to parse evidence strings and surface clickable `@123.4s` badges in AmountView/PriorView (video-server forwards new keys via normalizePrediction). **10 sessions: matched=57/374, hall=3, CNPE event-weighted mean=30.0% (n=55), macro=29.4% (n_items=15); planner 39,818 in / 72,780 out tokens.** Regressed ~3pp on CNPE — the "Most important — fill whenever possible" pressure on `amount_remaining` skewed estimates toward capacity-anchored guesses.
- `avp_minimal_remaining_gemini-3.1-pro-preview_r2free_late_mr280_tlhigh_jdedup_evframe2_preds_eval.json` — **evframe v2**: prompt cleanup (no schema change). Removed the "Most important" tag on `amount_remaining`; reframed all four fields (status + three amounts) as INDEPENDENT visual readings ("do NOT compute one field from the others"); softened R2 status-lock from "Treat status as fixed at `used`" to "Inherit R1's `status: used` unless..."; collapsed iff invariant to one mention; replaced R2 evidence-format paragraph with one-line back-reference to R1; dropped `visual_class` from observer JSON and `item` from planner `item_decisions[]` (~50-100 output tokens saved per call). **10 sessions: matched=58/379, hall=3, CNPE event-weighted=26.9%/24.4% (n=54), macro=25.8%/26.9% (n_items=16); planner 39,648 in / 68,541 out tokens, 1189s.** Recovered to canonical territory; observer now leaves nulls when fill is unreadable instead of confabulating.
- `avp_minimal_remaining_gemini-3.1-pro-preview_r2free_late_mr280_tlhigh_jdedup_evframe3_preds_eval.json` — **evframe v3 (best so far)**: three additions on top of v2. (1) **Numeric granularity floor**: `unit=grams` rounds to nearest 10g, `unit=count` is integer. (2) **Tare clarification**: `original_package_capacity` is contents only (no tare). (3) **Journey-as-window**: journey samples are now first-class `[start, end]` ranges (~1s wide) parallel to dense windows; planner emits `start/end` for both kinds, parser accepts new shape with backward-compat `t` fallback, frame extractor + segment renderers updated. **10 sessions: matched=57/380, hall=3, CNPE event-weighted=27.4%/21.8% (n=57), macro=23.3%/26.0% (n_items=16); planner 40,368 in / 70,098 out tokens, 1263s.** Macro CNPE dropped 2.5pp vs v2 — granularity floor pulled odd-precision estimates into alignment with GT (Tofu went from 16.8% → 0.8%). Per-item olive oil regressed (50%), traced to commitment pressure on opaque/dark containers.
- `avp_minimal_remaining_gemini-3.1-pro-preview_r2free_late_mr280_tlhigh_jdedup_evframe4_preds_eval.json` — **evframe v4 (rolled back)**: added volume × density CoT on top of v3 (derivative bullet asks for `volume × density`; starting/remaining gets dual-route fill-fraction-vs-volume-mediated procedure based on container shape). Plus 11-item redundancy/bias cleanup: dropped olive-oil example (was inventory-overfit), dropped `× 0.9` density anchor, collapsed triple-stated 10g-rounding to one statement, dropped triple-stated loose-portions backstop, removed "if container looks visibly full / unopened" sharp-edge clause, dropped volume reference list (teaspoon/bowl/etc), trimmed R2 candidate-block prelude to one-line back-reference, removed redundant R2 status-inheritance footer, kept single `do not set` independence example. **10 sessions: matched=56/380, hall=3, CNPE event=30.0%/26.3% (n=56), macro=27.8%/27.1% (n_items=17); planner 40,368 in / 70,968 out tokens, 1158s.** Macro regressed 4.5pp vs v3 — the volume×density CoT *hurt* items where Gemini's prior was already well-calibrated (Tofu 0.8% → 19.4%, Baby Carrots 16.1% → 33.9%, Garden Salad Bag 15.5% → 31.0%); modest gain on tapered-bottle case (olive oil 50% → 40%) didn't compensate. **CoT block rolled back; the v4 redundancy/bias cleanup is retained.** Current state of `_evidence_v2.py` = evframe v3 (granularity + tare + journey-as-window) + the v4 cleanup minus the CoT block.

## Giraffe (controlled-consumption participant)

- `avp_remaining_gpt-5.4_controlled_remaining_v1_preds_eval.json` — full AVP on controlled sessions.

---

Rows marked **(?)** are my guess; please correct if wrong.

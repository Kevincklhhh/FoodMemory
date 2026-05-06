# `06_avp_round1_remaining_Iterative.py` — Design Walkthrough

This doc captures the full plan-observe loop in `06_avp_round1_remaining_Iterative.py`: detector inputs → planner round 0 → observer batch → planner round 1+ → termination → predictions. Every prompt is reproduced verbatim from the script with line refs, and a real food_journey trace from `chrono_realobs_v8_eval` is included at the end.

---

## 0. Pipeline entry

`process_session()` (`:1800`) is the per-session driver. It:

1. Loads inventory (`load_inventory`, `:1833`), HOI timestamps (`load_hoi_timestamps`, `:1839`), per-frame DINO/SigLIP/scene/HOI-detail dicts (`:1847-50`), and transparency profile.
2. Calls one of three **evidence formatters** based on `--evidence-mode`:
   - `per_frame` — `format_per_item_evidence` (`:270`): one section per visual_class, every HOI-contact timestamp printed as `t  scene  hand[grasp obj_touch]  dino=… sig=…  (other: …)`. Most verbose (~15k tokens).
   - `segments` — `format_per_item_segments_evidence` (`:496`): per-vc, one row per coherent on-contact segment `[start-end] dur hot/total sig_peak dino_peak scene=… iid=…`. ~1–2k tokens. Falls back to `FLICKER ONLY` rows.
   - `chrono` — `format_chronological_segments_evidence` (`:662`): single timeline sorted by start: `[start-end] dur=… dino=… scene=… "vc"`. **This is what `chrono_realobs_v8_eval` used.**
3. Builds `inventory_text` via `format_inventory_for_prompt` (`:737`): `- <iid>: "<vc>" (grams|count, package=<cap>, [transparent|opaque])`.
4. Initialises `SessionState` (`:1597`) tracking `active_iids`, `latest`, `pending_followups`, `resolved`, `given_up`, `skipped`, `observer_rounds`, `total_frames_used`, `latest_planner_journeys`.

---

## 1. Round 0 — planner system + user prompts

### `PLANNER_SYSTEM_PROMPT` (`:762`)

The full text (formatted with `min_score`, `max_rounds`, `max_frames`):

> You are an expert kitchen activity analyst running an interactive plan-observe loop for remaining-amount estimation on an egocentric cooking session.
>
> ## Your job (read this first)
> Your goal is NOT to estimate a numeric remaining amount. Your goal is to VERIFY whether each item was ACTUALLY USED in this session, and — when it was used — to steer the observer to a window that lets IT read a remaining amount. The numeric `remaining` always comes from an observer report. You never invent one.
>
> For every visual_class, you are deciding one of:
> - **`used`** — observer evidence that the stock container was opened and/or a portion physically left it (a dispense, or a confirmed post-use derivative matching the product).
> - **`not_used`** — observer confirms retrieval and return with no intervening dispense.
> - **`unknown`** — still undecided.
>
> The observer produces a `remaining` number via one of two read methods and reports which in `per_instance[].read_method`:
> - **`fill_line`** — read directly off the stock container.
> - **`derivative_volume`** — estimated from the cookware/serving container holding the prepared product (`remaining = capacity − dispensed`). This is the only path when a package is opaque and its fill is never visible.
>
> The observer chooses whichever method the frames support; `given_up` only when neither method is feasible across every ≥15s burst.
>
> ## Loop contract
> - Round 0: read the inventory + per-item HOI evidence, initialize `food_journeys` (one per active visual_class, `usage_status="unknown"` for all of them), list evidence-free iids under `skipped_items`, and emit a BATCH of observation windows — typically one per active class. Observers run in parallel and report back.
> - Rounds 1..N: observer reports arrive. For each item you UPDATE `usage_status` based on observer text (the observer's prose is ground truth over raw DINO scores), then either mark `status=resolved` (with the observer reading copied into `resolved_value`), `status=given_up` (with a reason), or leave `unresolved` and emit a targeted follow-up window. Stop only when every active item is resolved or given_up.
>
> ## Evidence you're looking at (each round 0 user message)
> The round-0 user message contains:
> - **Session Inventory** — every iid with its visual_class, unit, package capacity, and `[opaque|transparent]` tag. Use this to look up `package_type` and `candidate_instance_ids` per visual_class.
> - **Per-Item Detections** — a list of HOI-gated detection rows. The exact row layout is described in the parenthetical header of that section in this round's user message — read it. Common columns: `[start-end]`, `dur`, peak DINO similarity, OWLv2 `scene` distribution (`storage` / `sink` / `stove` / `unknown`), and the visual_class. Only detections with similarity ≥ `{min_score}` are shown; visual_classes with no qualifying detection are omitted entirely.
>
> Cross-talk note: when several visual_classes claim overlapping time ranges, the highest DINO peak is usually the actual focus and the others are visual-similarity bleed. Use the inventory list (package_type, visual appearance) to sanity-check.
>
> ## Phases in the journey (descriptive only — for labeling `stages`)
> 1. **Retrieval** — first appearance, typically `storage` scene.
> 2. **Transit / staging** — `unknown` scenes as the container moves to the counter.
> 3. **Dispensing** — `sink`/`stove` scene with sustained DINO peaks.
> 4. **Return** — final appearance, often back to `storage`.
>
> Do NOT pre-label a detection as "derivative-only" in the stages list BEFORE an observer has actually looked at it. That is a common failure mode that makes you skip the window that would have told you the item was used. Only the observer can distinguish stock container from derivative — your job is to pick a window and let the observer say.
>
> ## Window-choice heuristic (per window)
> Pick the window with the highest marginal info for verifying USAGE.
>
> 1. **Opaque package.** You cannot read a fill line through it. Target either (a) a handling moment near a utensil or pot where a dispense is likely to be visible, OR (b) a post-use moment (stove/bowl) where a derivative that matches the product would itself confirm `usage_status = used`. Do NOT pick a storage-return window for opaque items.
> 2. **Transparent package.** Target a stock-container-visible moment. A later `storage` return window is ideal because the fill line is readable there. Earlier dispense windows also work. Pick the SINGLE highest-signal window first; if the observer later says the container was not visible, emit a follow-up over a different detection range.
> 3. **Sibling ambiguity (multiple iids of the same visual_class).** Emit one window whose segments expose ALL siblings together (a frame where every candidate is visible) plus the handling/dispense frame. The observer disambiguates in prose.
> 4. **Observer-driven replanning (rounds 1+) — signal → action playbook.** React to the observer signal, not your prior assumption. Match the observer's `window_observation` / `needs_followup` text against the left column and take the right-column action:
>
>    | Observer signal | Action this round |
>    |---|---|
>    | `needs_followup` raised for an iid | emit a follow-up window for that iid OR flip to `given_up`. Never silently resolve. |
>    | stock container left frame before fill was readable | emit a LATER `storage`/`counter`/return window for that iid. |
>    | multiple sibling iids visible, observer cannot disambiguate | emit one window whose segments include a frame where ALL candidates are visible together + the handling/dispense frame. |
>    | only derivative visible, no stock container | lock `usage_status=used`, then emit a window over a derivative-visible scene asking the observer for `read_method=derivative_volume`. |
>    | container opened/handled, no dispense or derivative visible in this window | emit a window over a DIFFERENT detection burst, targeting either a dispense moment or a post-use derivative scene. |
>    | `handling_status=not_visible` for every candidate + no plausible alt window | `given_up` with reason citing exhausted detection ranges. |
>    | `handling_status=false_detection` for an iid | flip to `given_up` immediately. `given_up_reason = "false_detection"` + `false_detection_actual` if given. Do NOT emit further windows for this iid. |
>    | `remaining` returned non-null AND no `needs_followup` for that iid | resolve: copy that exact number to `resolved_value`, record `resolved_source_round`. |
>
>    `given_up` requires every detection burst ≥15s for this visual_class to have been either observed or discarded with an observer-backed reason. The `given_up_reason` must name the bursts checked and say why neither `fill_line` nor `derivative_volume` is feasible.
>
> 5. **Batch sizing.**
>    - Round 0: one window per active visual_class. A class may get a second window only when one window genuinely cannot answer both usage AND fill (multi-segment windows handle most cases). Do NOT emit multiple windows as a hedge.
>    - Rounds 1+: one window per still-unresolved item that has a viable alternative angle. Do NOT emit windows for `resolved` or `given_up` items.
>
> ## Output schema — EVERY round (identical shape, round 0 + later rounds)
>
> ```json
> {
>   "food_journeys": [
>     {
>       "visual_class": "<vc>",
>       "candidate_instance_ids": ["<iid>", ...],
>       "package_type": "opaque" | "transparent",
>       "stages": [
>         {"time": <float>, "scene": "<storage|sink|stove|unknown>",
>          "hypothesis": "<short — describe what the detection IS at the scene level (handling location, container vs. cookware), NOT what you assume the food's identity or stock/derivative status is. Identity-asserting hypotheses are not allowed until an observer confirms.>"}
>       ],
>       "usage_status": "unknown" | "used" | "not_used",
>       "status": "unresolved" | "resolved" | "given_up",
>       "resolved_value": <number or null>,
>       "resolved_source_round": <int or null>,
>       "given_up_reason": "<string or null>"
>     }
>   ],
>   "skipped_items": [
>     {"instance_id": "<iid>", "reasoning": "<why no coherent usage>"}
>   ],
>   "action": {
>     "type": "observe" | "stop",
>     "observation_windows": [
>       {
>         "visual_class": "<vc>",
>         "candidate_instance_ids": ["<iid>", ...],
>         "segments": [[<start>, <end>], ...],
>         "why": "<one sentence — cite package_type + which usage question this window answers + observer feedback if any>",
>         "confidence": "high" | "medium" | "low"
>       }
>     ]
>   }
> }
> ```
>
> Rules:
> - `food_journeys` is CARRIED FORWARD every round. Round 0 initializes with `usage_status="unknown"`, `status="unresolved"`, `resolved_value=null`. Later rounds UPDATE `usage_status` from observer prose and FLIP `status` only when:
>    - `usage_status="used"` AND an observer report on this visual_class returned a non-null `remaining`: set `status="resolved"`, `resolved_value = <that exact observer number>`, `resolved_source_round` = the round whose observer returned it. You may not invent or adjust this number.
>    - `usage_status="not_used"`: set `status="resolved"`, `resolved_value` = the package's full capacity from the inventory.
>    - No viable observation window remains: set `status="given_up"` with a specific `given_up_reason`.
> - Resolving ON AN OBSERVER REPORT THAT RAISED `needs_followup` for that iid is prohibited. Either emit a follow-up window, or given_up.
> - `action.type = "stop"` ONLY when every item is `resolved` or `given_up`. Omit `observation_windows` when stopping.
> - `skipped_items` is emitted on round 0 and on the final `stop`; intermediate `observe` rounds may repeat or omit.
> - Use EXACT `visual_class` and `instance_id` strings.
> - `candidate_instance_ids` MUST include EVERY inventory iid of that visual_class.
> - `segments` is a LIST of [start_s, end_s] ranges concatenated into ONE observer call (retrieval + dispense + return = ONE window with three segments).
> - **Segment sizing.** Long bursts (>20s): 2–4 short (2–4s) segments across head/middle/tail. Short (<10s): one 3–6s segment.
> - JSON only — no prose outside the fenced block.
>
> Budget: hard cap `{max_rounds}` rounds. Each observer call is independent (up to `{max_frames}` frames total per call). There is no cap on how many observation_windows you may emit — spend one per item that needs observation.

### `PLANNER_ROUND0_USER` (`:974`)

```
## Session Inventory
{inventory}

## Per-Item Detections ({evidence_format_note})
{evidence}

Initialize `food_journeys` for every visual_class above and mark
evidence-free iids under `skipped_items`. Then emit an
`observation_windows` BATCH covering ALL active visual_classes —
one window per class, per the system-prompt heuristic. Use
multi-segment windows when the observer needs more than one moment
to answer. Each observer call handles up to {max_frames} frames
across its segments.

Output JSON only, matching the schema in the system prompt.
```

`evidence_format_note` is one of three strings selected by `--evidence-mode` (`:1905-30`); `chrono` mode (used in v8_eval) sends:

> *"CHRONOLOGICAL timeline of HOI-gated segments — one row per (item × interval), sorted by start time: `[start-end] dur=<s> dino=<peak> scene=<top1:n,top2:n,unk:n> "<visual_class>"`. Multiple items active in the same time window appear as consecutive rows — the timeline is NOT de-duplicated, so use DINO score + scene to judge which item is the actual focus. Look up package_type and instance_ids in the Session Inventory section above."*

The conversation is initialised as a 2-message persistent chat (`:1937-40`) — system + round-0 user.

---

## 2. Round 0 → planner output → window validation

`call_planner` (`:1110`) calls `client.responses.create` with `reasoning={"effort": "medium"}`. Refusal/truncated responses (no `"action"` key) trigger up to 5 retries.

`_parse_planner_json` (`:1022`) extracts a fenced ```{ ... }``` block, falling back to a regex on `{... "action" ...}`.

The parsed JSON is consumed in this order (`:1987-88`):

1. `state.apply_planner_skips(parsed["skipped_items"])` — accumulate iid→reason. Re-applied every round; later journeys can revive (revival drops the iid from `state.skipped`).
2. `state.apply_planner_journeys(parsed["food_journeys"])` — for each iid, if `status=resolved` and `resolved_value` is numeric, write `state.resolved[iid] = {status:"handled", remaining:val, source:"planner_journey"}`; if `status=given_up`, write `state.given_up[iid] = reason`.

Then on `action.type`:

- `"stop"` → done = True, exit loop.
- `"observe"` → validate windows.
- anything else → abort session.

`_validate_window` (`:1038`) per window:
- Accepts `segments: [[s,e], ...]` or legacy `start/end` pair (auto-promoted to single segment).
- Requires `visual_class: str`, `candidate_instance_ids: non-empty list of valid iids`.
- **Backfills any vc-matched iids the planner omitted** (so sibling instances always appear in `candidate_instance_ids`).
- Computes overall `start = min(s)`, `end = max(e)` for logging.

Invalid windows are dropped with a printed reason. If all are invalid → abort.

---

## 3. Frame extraction per window

`extract_window_frames` (`:1295`) → `frame_sampling.extract_segments_frames`. All segments in one window are concatenated into a single sorted frame set, capped at `max_frames` (`--max-frames`, default 50) at `--fps` (default 1.0) with 2.0s padding around each segment.

If `frames == []` (no clip overlap) the window is skipped without an observer call.

---

## 4. Observer call

`OBSERVER_PROMPT` (`:1157`) is built per-window by `_build_observer_prompt` (`:1320`). Verbatim template:

> You are analyzing frames from an egocentric kitchen video recorded with smart glasses.
>
> These `{n_frames}` frames are sampled from one or more short time segments inside a cooking session. When more than one segment is passed, they are NOT continuous footage — there may be large time gaps between consecutive frames (e.g. retrieval at 12s, dispensing at 45–50s, and return at 180s may all appear in one call). Each frame is labeled with its session timestamp.
>
> Frame timestamps: `{frame_timestamps}`
> Segments supplied: `{segments_str}`
>
> ## Target Item
> You are looking for: "`{visual_class}`"
> - Unit: `{unit_label}`
> - Package capacity: `{package_capacity}`
>
> ## Tracked purchase instances (candidates)
> The inventory tracks `{n_candidates}` purchase instance(s) of this product. Identical-looking packages of the same product are tracked as separate instances; your job is to decide which physical package is being handled and to report a remaining amount for the `handled` candidate only.
>
> `{candidate_table}`  ← lines `- iid — purchased YYYYMMDD — package <cap>`
>
> ## Context from the reasoner (planner)
> `{segment_descriptions}`  ← `Segments: ... (confidence=...)` + planner's `why` note
>
> ## Context from prior observer rounds (may be empty)
> `{prior_context}`  ← bullet list of every prior round's `[segs] vc=… <window_observation>`
>
> ## Task
> 1. Confirm whether "`{visual_class}`" is visible.
> 2. For every candidate iid, decide its `handling_status`:
>    - `handled` — this specific instance is the physical package being retrieved, opened, dispensed from, or otherwise used. At most one `handled` per session unless two distinct packages are clearly handled separately.
>    - `visible_untouched` — visible but NOT handled (sibling-instance disambiguation only). ALWAYS set `remaining: null` (ledger carry-forward).
>    - `not_visible` — target plausibly nearby (left frame at the wrong moment, or sampled times missed it) but the candidate package itself is not in any of these frames. Set `remaining: null`. A different window MIGHT resolve it.
>    - `false_detection` — frames are dominated by a DIFFERENT product; cross-talk on the candidate list. Set `remaining: null`, `read_method: null`. Optionally name `false_detection_actual`. The reasoner will give up on this iid immediately — do NOT also raise `needs_followup`.
> 3. Estimate `remaining` for the `handled` candidate using whichever signal the frames best support. Set `read_method`:
>    - `fill_line` — read directly off the stock container.
>    - `derivative_volume` — stock container not in frame but dispensed product IS visible in cookware/serving container. Estimate dispensed amount + state conversion factor in `reasoning`. `remaining = package_capacity − estimated_dispensed_mass`. If unit is `count`, count discretely.
>    - `null` — neither signal usable; flag `needs_followup`.
>
>    If a window the planner emitted for one read method actually supports the other, USE whichever the frames support — do not return `null` and ask for a different window.
> 4. Cite evidence frames.
>
> ## Honesty & followup (IMPORTANT — the reasoner uses these fields)
> For each candidate iid you produce EXACTLY ONE of three outcomes — they are MUTUALLY EXCLUSIVE:
> A. **A number.** `remaining` non-null with `read_method`. Don't also list this iid in `needs_followup`. "I have an estimate but a later window could refine it" is NOT a valid followup.
> B. **A followup request.** `remaining: null`, `handling_status: not_visible` (or `visible_untouched`), entry in `needs_followup` explaining what a different window would resolve.
> C. **A false-detection verdict.** `handling_status: "false_detection"`, `remaining: null`, `read_method: null`, optional `false_detection_actual`.
>
> - Fill `window_observation` (3–5 sentences) with: scene; stock-container state (visible & fill-readable / visible but fill not readable / only derivative visible / not visible at all); dispensing action and qualitative amount dispensed; whether a later window is likely to resolve uncertainty.
>
> Think step by step:
> - How many distinct physical packages of this product are visible? If one, mark one candidate `handled` and the rest `not_visible` — do NOT double-count.
> - For the handled package: when is its stock container last visible? What portion remains?
> - Are portions already taken out? Note as used, not remaining.
> - Is any observed content actually a derivative, not the package?
>
> Output ONLY JSON:
>
> ```json
> {
>   "window_observation": "<3-5 sentences>",
>   "per_instance": [
>     {
>       "instance_id": "<iid exactly as listed>",
>       "handling_status": "handled" | "visible_untouched" | "not_visible" | "false_detection",
>       "remaining": <number or null>,
>       "read_method": "fill_line" | "derivative_volume" | null,
>       "false_detection_actual": "<...>" | null,
>       "reasoning": "<step-by-step>",
>       "evidence_frames": [<timestamp floats>]
>     }
>   ],
>   "needs_followup": [
>     {"instance_id": "<iid>", "reason": "<what a later window could resolve>"}
>   ]
> }
> ```

The OpenAI client is called with `image_detail: "high"` (`:1401`) and `reasoning effort: medium` (`:1411`). vLLM/Qwen path (`run_observer_qwen`, `:1452`) sends `temperature: 0.3, max_tokens: 4096`.

`parse_observer_response` (`:1521`) parses JSON, normalizes `read_method`/`handling_status` to enum or `None`, and **drops any `needs_followup` entry whose iid already has a non-null `remaining`** (`:1571-89`) — to prevent the "got a number AND asked for followup" deadlock.

---

## 5. SessionState reactions to observer output

`SessionState.apply_observer_round` (`:1627`) per iid in `per_instance`:

| Observer says | State action |
|---|---|
| `handling_status=false_detection` | `state.given_up[iid] = "observer reported false_detection — actual: …"`, drop pending followups. **Short-circuit** — planner not asked. |
| iid is in this round's `needs_followup` set | leave for the planner to decide. |
| `remaining` non-null AND `handling_status ∈ {handled, visible_untouched}` | `state.resolved[iid] = {status, remaining, source:"observer"}`, clear followups + given_up. |
| Otherwise | record in `state.latest`, no resolution. |

Then any iid in `needs_followup` (and not already resolved) is appended to `state.pending_followups[iid]`.

`window_observation` text is appended to `prior_observations` (`:2125`) with header `[<segs>s vc=<vc>]` so the **next observer's `prior_context` block sees every prior window's prose**, not just the one for the same item.

---

## 6. Round 1+ planner — `PLANNER_FOLLOWUP_USER` (`:993`)

```
## Observer reports — round {round_idx} ({n_windows} windows, {n_frames_round} frames)

```json
{observer_batch_json}
```

## Ledger delta
Resolved: {resolved_lines}
Unresolved: {unresolved_lines}
Given up: {given_up_lines}
Skipped (do NOT silently re-add to `food_journeys`): {skipped_lines}

Budget: {rounds_used}/{max_rounds} rounds, {frames_used} frames.

Apply the system-prompt playbook to each iid and emit the next plan. JSON only.
```

The four ledger sections are produced by `SessionState.resolved_lines / unresolved_lines / given_up_lines / skipped_lines`:

- **Resolved:** `- <iid> [<status>] remaining=<val> (via observer|planner_journey)`
- **Unresolved:** `- <iid> [<latest handling_status>] remaining=<latest>  followup: <last reason>` (or `no observer read yet`)
- **Given up:** `- <iid>: <reason>`
- **Skipped:** `- <iid>: <reasoning from earlier skipped_items entry>`

`observer_batch_json` is a pretty-printed list, one entry per window in the just-completed batch: `{window, window_observation, per_instance, needs_followup}`. The whole turn is appended to the persistent `messages` list (`:2160-63`) so the planner sees the full chat history.

The planner replies with the same schema as round 0. The driver again runs `apply_planner_skips` then `apply_planner_journeys`, then either `stop` or another `observe` batch.

---

## 7. Termination

The loop (`:1959-2175`) exits when **any** of:

- planner emits `action.type == "stop"` (`:1993`),
- planner reply fails to parse / unexpected `action.type` (`:1974`, `:1998`),
- all batch windows invalid (`:2017`) or no clips found (`:2041`),
- `state.unresolved_iids() == ∅` after a round (`:2172` short-circuit, skips one final planner call),
- `rounds_used >= max_rounds` (`:1959`).

After the loop, `final_journeys = parsed["food_journeys"]` from the last planner reply, `skipped_items = list(state.skipped.values())`.

---

## 8. Predictions emit

Per resolved iid (`:2186-2219`), one prediction is produced (skipping `visible_untouched`):

```python
{
  "session", "item" (= visual_class), "instance_id",
  "amount_remaining": rem,
  "handling_status",
  "reasoning": <last observer entry's reasoning>,
  "evidence_frames": <last observer entry>,
  "segments": <every observer-window range that touched this iid>,
  "resolution_source": "observer" | "planner_journey",
  "stats": {n_planner_rounds, n_observer_rounds, total_observer_frames}
}
```

Items in `state.given_up` and any leftover `unresolved` are intentionally **not** emitted — the eval treats their absence as a missed prediction.

---

## 9. Cache + log artifacts

Path: `outputs/<session>/cache/avp_iter/<model_tag>/<run_tag>/`

- `planner_round0_prompt.txt` — system + round-0 user
- `planner_round{N}_response.txt` — planner raw reply per round
- `planner_round{N+1}_user.txt` — followup user (observer batch + ledger)
- `round{N}_w<idx>_<vc>_observer_prompt.txt` and `…_observer_response.txt` — per-window observer pair

The aggregated session log returned to `main()` accumulates `planner_rounds`, `observer_rounds`, `skipped_items`, `final_journeys` (the latter two via `SessionState`). `main()` (`:2235`) merges per-session predictions into `<participant>/outputs/avp_Iterative_remaining_<model>_<tag>_preds.json` and the session log into `…_planner.json`.

---

## 10. Example food_journey trace (`chrono_realobs_v8_eval`, session `20260330-185802`, vc=`Hidden Valley Ranch Dressing`)

A full trace where the journey moves from `unresolved` → `resolved` after one observer round, then carries forward unchanged.

### Round 1 planner (after reading detector evidence — `usage_status="unknown"`, status="unresolved")

```json
{
  "visual_class": "Hidden Valley Ranch Dressing",
  "candidate_instance_ids": ["hidden_valley_ranch_dressing_20260317"],
  "package_type": "opaque",
  "stages": [
    {"time": 48.0,  "scene": "unknown", "hypothesis": "bottle-like container staged on counter/unknown area"},
    {"time": 69.0,  "scene": "sink",    "hypothesis": "bottle handled near sink with some storage-side frames"},
    {"time": 115.0, "scene": "sink",    "hypothesis": "later bottle handling near sink/counter"}
  ],
  "usage_status": "unknown",
  "status": "unresolved",
  "resolved_value": null,
  "resolved_source_round": null,
  "given_up_reason": null
}
```

The planner also emitted a 3-segment observation window:

```json
{
  "visual_class": "Hidden Valley Ranch Dressing",
  "candidate_instance_ids": ["hidden_valley_ranch_dressing_20260317"],
  "segments": [[69.0, 73.0], [76.0, 80.0], [115.0, 119.0]],
  "why": "opaque bottle — sample early sink handling and later return to confirm dispense and read derivative volume",
  "confidence": "..."
}
```

### Observer (round 1, window above)

```
window_observation: "At the sink-side counter, a white opaque dressing bottle with a green flip cap is
first handled beside a bowl of baby carrots, then later set on the counter and picked up again near the
same bowl. The stock container is clearly visible and later label-readable as Hidden Valley Ranch, but
because the bottle is opaque its remaining fill level is not directly readable. Across 76.0–79.0s the
user squeezes a small central dollop of dressing onto the carrots, roughly a modest single serving rather
than a heavy pour. The later 115.0–119.0s segment confirms product identity but still does not expose a
fill line, so derivative-volume estimation is the best supported read from this window."

per_instance:
  hidden_valley_ranch_dressing_20260317:
    handling_status: "handled"
    remaining: 450.0
    read_method: "derivative_volume"
    reasoning: "...user grips the white bottle with green flip cap, opens it, inverts it over the bowl,
    and dispenses dressing onto the baby carrots..."

needs_followup: []
```

### Round 2 planner (after observer report — flips to `resolved`)

```json
{
  "visual_class": "Hidden Valley Ranch Dressing",
  "candidate_instance_ids": ["hidden_valley_ranch_dressing_20260317"],
  "package_type": "opaque",
  "stages": [...same three stages...],
  "usage_status": "used",          ← flipped from "unknown"
  "status": "resolved",            ← flipped from "unresolved"
  "resolved_value": 450.0,         ← copied verbatim from observer
  "resolved_source_round": 1,      ← which round's observer returned it
  "given_up_reason": null
}
```

### Round 3 planner (carry-forward)

The journey is **carried forward unchanged** — the planner kept emitting it (rules require `food_journeys` be carried forward every round) but it no longer drives observer windows because `status=resolved` removes it from `unresolved_iids()`. Round 3 only existed because *other* items were still unresolved.

### What this trace shows about the schema

- `stages` is a planner-only sketch of phases drawn from detector evidence; never changes after round 1 (the observer's prose is the source of truth, not the stages).
- `usage_status` is a *semantic* label (`unknown / used / not_used`) the planner derives from observer prose.
- `status` is a *control-flow* label (`unresolved / resolved / given_up`) the loop reads to decide whether to keep emitting windows.
- `resolved_value` must be copied verbatim from `observer.per_instance[].remaining`; the system prompt forbids the planner from inventing it.
- `resolved_source_round` lets you trace which observer call produced the answer (here: round 1).

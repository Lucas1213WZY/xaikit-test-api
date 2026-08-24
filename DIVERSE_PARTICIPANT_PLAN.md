# `mode="diverse_participant"` — implementation plan

**Status: implemented 2026-08-24.** Eight commits, `ff4f711`..`abf4600`, with
tests in `tests/test_participant_pools.py`,
`tests/test_{coax,coxam,sim2real}_diverse_participant.py`,
`tests/test_api_diverse_participant.py` and
`tests/test_server_diverse_participant.py`. Sections 1-3 are the reasoning
behind it and still describe what was built; section 4 describes the code as it
now stands. Section 9 records what was measured and what is left.

Draft, 2026-08-24. Adds a simulation mode that gives every virtual participant
its **own** cognitive parameters, drawn from the pool of parameters fitted to
the real study participants, filtered by the condition the virtual participant
is assigned to.

---

## 1. The problem, measured

Every simulation mode today runs one parameter set for the whole study, so all
N participants inside a condition are the same person taking N sessions. The
two internal-test notebooks show the two failure shapes this produces.

### CoXAM tutorial — inflated, not degenerate

`pairwise_condition_tests` prints `p_value = 0.0000` only because of
`.round(4)`. Back-computed from the reported t statistics (paired, df = 23):

| statistic | actual p |
| --- | --- |
| −10.5632 | 2.69e−10 |
| −9.3488  | 2.68e−09 |
| −11.7206 | 3.53e−11 |
| 0.9405   | 0.3567 (matches the printed value exactly — the t's are genuine) |

Within-condition SD is non-zero (`0.0859 / 0.0707 / 0.0827` for
decision_tree / hybrid / logistic_regression in the notebook's stored
`analyze_iv_dv` output). The only variance source is **which instances each
participant happened to see** — counterbalancing plus trial randomisation. No
individual differences at all, so the denominator of every t is far too small.

### Sim2Real tutorial — genuinely degenerate

Stored notebook output:

```
xai_property     count    mean       std
faithful            12  0.689655  0.000000
robust              12  0.908046  0.047265
sparse              12  0.758621  0.000000
sparse_robust       12  0.931034  0.000000

statistic = -1.456811e+15  ...  RuntimeWarning: catastrophic cancellation
```

Here every participant sees the *same* 29 corpus test instances
(`num_testing=len(test_instance_ids)`) and the model is deterministic, so three
of four conditions have literally zero variance and p underflows to 0. `robust`
is non-zero only because it is the one condition with a lapse rng draw.

### Why the fix is fitted sampling, not injected noise

The codebase already says so, in `coax_study_runner.FITTED_COAX_PARAMS`:

> The raw per-participant fits are bimodal (41% pinned at the sensitivity floor
> of 1.0, the rest spread 19–100) rather than clustered near the mean […] The
> more correct fix is per-participant sampling from this real distribution
> instead of one shared constant.

So `diverse_participant` is expected to improve two things at once: realistic
between-participant variance **and** better calibration against real human
accuracy, since the fitted distributions are bimodal and their means are not
representative of anyone.

---

## 2. Mode semantics

```python
simulated_results = study.run_experiment(
    mode="diverse_participant",
    participant_id=None,
    explanation_pool=study.combined_explanations,
)
```

* Trial selection is identical to `whole_experiment` — every generated trial runs.
* Before running, each `participantId` is assigned one row from the fitted
  parameter pool, filtered to that participant's own condition cell.
* One draw per participant **per condition cell they occupy**: a between-subjects
  IV means one draw per participant; a within-subject block IV means the
  participant keeps one identity but is served the pool row matching each block's
  condition, so parameters never contradict the condition being simulated.
* The assignment is stable for the whole session (memory/accumulation still work)
  and is recorded on every output row for provenance.
* Deterministic given `sampling_seed` (default 0).

New runner kwargs, agent-only (the generic executor path already raises
`TypeError` on unexpected kwargs, which is the behaviour we want for baselines):

| kwarg | default | meaning |
| --- | --- | --- |
| `sampling_seed` | `0` | seeds both pool assignment and the per-participant simulation RNGs |
| `sampling_replace` | `None` | `None` = without replacement while the pool lasts, then with; `True`/`False` force |
| `parameter_pool` | `None` | a `DataFrame` to use instead of the shipped pool (own fits, sensitivity analysis) |

Modes are otherwise untouched; `whole_experiment` keeps today's behaviour so no
existing run or test changes.

---

## 3. Which pool each framework samples

All four pools are the anonymised, open-source-safe copies under
`assets/human_data/` — never the originals under `src/cognitive_models/`
(`assets/build_human_data.py` renumbers participants and drops raw ids).

### 3.1 CoAX → `assets/human_data/CoAX/coax_fitted_strategies.csv`

1,133 rows / 330 participants. Column names already match the CoAX strategy
constructors exactly, so no name mapping is needed.

* Filter on: `dataId` (study dataset), `expMethod` (= `xai_method`, lime/shap),
  `XAIType` (`Attribution` / `Importance` / blank = none),
  `Tested w/ XAI` (`w/ XAI` / `w/o XAI`), `Strategy` (the strategy the runner
  picked for that cell — see `PREFERRED_COAX_STRATEGY_BY_XAI_TYPE`).
  Ignore `Session`.
* Draw: `sensitivity`, `k`, `retrieval_threshold`, `scaling_factor`
  (`scaling_factor` only for AttributionSum, `sensitivity` only for the other
  three — `coax_params_for_strategy` already filters per strategy).
* `decay_param` is not in the file; keep the reference value 0.5.
* **No clipping.** `COAX_PARAM_BOUNDS` are UI slider limits, not bounds on the
  population — the existing docstring says this explicitly.
* Coverage caveat: only `adult`, `forest_cover`, `wine_quality` were fitted.
  `mushrooms` has no rows → must fall back with a clear warning.

### 3.2 CoXAM counterfactual → `assets/human_data/CoXAM/coxam_counterfactual_replay.csv`

**This is the right file, not `coxam_counterfactual_fit.csv`** — the latter is a
50-participant subset of exactly this replay (see `build_human_data.py`'s note),
and several of its condition × complexity cells hold only 1–3 participants.
The replay carries 270 participants covering all 3 conditions × 2 complexities
× 2 datasets, with 14–33 participants per cell.

Its three fitted columns map onto the counterfactual env's parameters, and the
fitted ranges match the trained ranges in `counterfactual_env.DEFAULT_COGNITIVE_PARAMS`
essentially exactly — strong confirmation the mapping is right:

| replay column | env parameter | fitted range | trained range |
| --- | --- | --- | --- |
| `Retrieval threshold, κ` | `memory_recall_threshold` | −1.998 … 0.497 | [−2.0, 0.5] |
| `Margin, ε` | `counterfactual_overshoot_fraction` | 0.051 … 0.499 | [0.0, 0.5] |
| `Opportunity cost, γ` | `time_penalty_weight` | 0.0001 … 0.0200 | [0.0, 0.02] |

* Filter on: `dataId` (`Mushrooms` / `Wine Quality` — needs case/space
  normalisation), `condition` (`Decision Tree` / `Linear Regression` / `Hybrid`),
  `complexity`. Dedupe to one row per `Participant Id` (the file is per-trial).
* `random_response_rate` was not fitted → keep the current default (0.3).

### 3.3 CoXAM forward → per dataset

This is the only pool whose column names do **not** line up with the runner's
parameters, because the forward RL meta-policy exposes exactly three free
parameters (`decision_noise`, `memory_recall_threshold`, `opportunity_cost` —
see `CombinedStrategyPolicyEnv._sample_episode_params`) while the forward fits
are ACT-R/DDM shaped.

* wine_quality → `coxam_forward_params_wine_quality_dt.csv` (45 participants,
  DT condition) and `coxam_forward_params_wine_quality_lr.csv` (45 participants
  × calculation/heuristic, LR condition). Filter on `dataId`, `Model`,
  `Complexity`, `Variant`, and `Strategy` for LR.
* mushrooms → `coxam_forward_fit_mushrooms.csv` deduped to one row per
  participant (137 participants, per-trial file). Filter on `Condition`,
  `Complexity`, `dataId`, `Model`.
* `hybrid` condition: pool the DT and LR files (the real study's hybrid
  participants were fitted per displayed family, not as a third family).

Mapping, with an explicit per-parameter policy:

| fitted column | env parameter | trained range | policy |
| --- | --- | --- | --- |
| `retrieval_threshold` (−2.0 … 1.5) | `memory_recall_threshold` | [−1, 2] in the loaded checkpoint's config.json | clip |
| `chi_value` (0.0 … 0.02, mushrooms only) | `opportunity_cost` | [0, 0.02] | direct |
| `ddm_s` (0.2 … 1.5) | `decision_noise` | [0.3, 0.7] | clip (see below) |
| `Strategy` (`dt`/`lr_calc`/`lr_heur`) | `policy_override` | — | **optional**, off by default |

Notes / decisions to confirm:

* `ddm_s → decision_noise` is the weakest link: both are the noise scale of the
  decision process, but the ranges only partly overlap and clipping compresses
  the distribution. Alternatives are (a) rank-map the fitted distribution onto
  the trained range (preserves individual ordering, loses the fitted units) or
  (b) don't map it and leave `decision_noise` at its midpoint. Recommend
  starting with **clip**, and exposing `param_map_policy="clip"|"rescale"|"skip"`
  so a sensitivity check is one kwarg away.
* wine_quality's forward fits carry `compute_sf`, not `chi_value`, so there is
  no fitted `opportunity_cost` for that dataset. Fallback order: matched pool →
  the counterfactual replay's γ for the same dataset/condition (same construct,
  same trained range) → current default. Whichever is used gets recorded.
* The mushrooms file's per-participant `Strategy` maps onto the meta-policy's
  own sub-strategy slots. Feeding it as `policy_override` would make virtual
  participants differ in *strategy*, not just parameters — a real second source
  of individual differences, but it bypasses the meta-policy. Keep it behind an
  opt-in flag (`use_fitted_strategy=False` by default).

### 3.4 Sim2Real → `assets/human_data/Sim2Real/sim2real_participant_fits.csv`

46 participants, one row each, 11–12 per `exp_property` — a near 1:1 match for
the notebook's 12 participants per condition, so sampling without replacement is
almost a permutation of the real population. Column names already match the
model constructors.

* Filter on: `exp_property` (`faithful` / `sparse` / `robust` / `sparse_robust`).
* Draw: `strategy`, `aggregation`, `confidence_scale`, `confidence_intercept`,
  `comparison_scale`, `comparison_intercept`, `comparison_C`,
  `max_features_attended`, `use_exemplar_memory`, `memory_sensitivity`,
  `memory_decay`, `retrieval_threshold`, `k`, `sensitivity`,
  `always_attend_changed`, `guess_bias`, `lapse_rate`.
* Because this file's fit **selected the strategy per participant**
  (`sparse`: 6 sensitive_features / 3 attribution_sum / 3 salient_features), the
  strategy varies within a condition too. This is the correct behaviour and
  supersedes the flat `SIM2REAL_STRATEGY_BY_PROPERTY` table in diverse mode.
* Baseline / unoptimised condition (`exp_property = None`) has no fitted
  participants → falls back to `NEUTRAL_SIM2REAL_PARAMS`, warned once.

---

## 4. Code changes

### New: `src/virtual_experiment_executor/participant_pools.py`

One framework-agnostic module holding the pool specs and the sampler. Mirrors
`src/result_visualizer/study_comparisons.py`'s `HUMAN_DATA = REPO_ROOT / "assets" / "human_data"`
convention and its "pool not built yet" error message.

```python
@dataclass(frozen=True)
class PoolSpec:
    path: Path
    filters: Mapping[str, str]        # trial/condition key -> CSV column
    parameters: Mapping[str, str]     # CSV column -> runner parameter name
    ranges: Mapping[str, tuple]       # runner parameter -> trained range (clip)
    dedupe_on: str | None             # "Participant Id" for per-trial files

def load_pool(spec) -> pd.DataFrame
def sample_participant_parameters(
    pool, *, condition, n, seed, replace=None
) -> list[dict]        # one dict per virtual participant, + _pool_participant_id
```

Sampling rule: shuffle the matched pool rows with `default_rng(seed)`, deal them
out without replacement; if there are more virtual participants than pool rows,
keep dealing reshuffled passes (so the empirical distribution is preserved and
duplicates only appear once unavoidable). `replace=True` forces i.i.d. draws.

Failure modes, all explicit rather than silent:
* pool file missing → `FileNotFoundError` naming `assets/build_human_data.py`;
* condition cell empty (e.g. CoAX + mushrooms) → warn, fall back to the existing
  fitted means, and stamp `parameter_source="fitted_mean_fallback"`.

### `src/experiment_planner/config.py`

Add `diverse_participant` to `select_trial_rows` as an alias of
`whole_experiment` (same row selection) and to the error message. Runners read
the raw `mode` string to decide whether to sample, so direct runner calls
(`run_sim2real_study(study, mode="diverse_participant")`, which the Sim2Real
notebook uses) work without going through `study.run_experiment`.

### `src/api.py`

* `run_experiment` docstring: document the mode and the new kwargs.
* Reject `diverse_participant` for baselines / custom `cognitive_model` with a
  message saying it needs a research agent — silently running the old behaviour
  under a new name is exactly the bug this fixes.
* `_run_agent_experiment`: pass the sampling kwargs through unchanged (the
  existing `_COGNITIVE_PARAM_KEYWORDS` routing still applies — an explicit
  `cognitive_params` should be treated as *overrides on top of* each drawn row,
  not a replacement; document that precedence).

### `src/virtual_experiment_executor/experiment_simualtion/CoAX/`

* `coax_trial_executor.run_coax_experiment_executor`: allow `cognitive_model` to
  be a **callable** `(participant_id) -> model | mapping`, resolved inside the
  existing per-participant loop (which already deep-copies a model per
  participant and gives it its own `SimulationClock`). Minimal diff, and mapping
  / single-model inputs keep working.
* `coax_study_runner.run_coax_study`: when `mode == "diverse_participant"`,
  build that callable from the pool — for each participant, per condition key,
  `make_coax_model(strategy, **fitted_coax_params(strategy, xai_type, **drawn))`.

### `src/virtual_experiment_executor/experiment_simualtion/CoXAM/`

* `coxam_study_runner.run_coxam_study`: the forward path already groups by
  `(participantId, block)` and passes `fixed_eval_params` per episode — inject
  the drawn row there. Also derive the episode seed per participant
  (`dataclasses.replace(config, seed=base_seed + index)`); today every episode
  is built with the same `config.seed = 123`, so two participants with the same
  trial order draw identical noise.
* `coxam_counterfactual_runner.run_coxam_counterfactual_study`: same injection
  into its existing per-group loop (`cognitive_params=` and the already-present
  `seed + offset`).

### `src/virtual_experiment_executor/experiment_simualtion/Sim2Real/`

* `sim2real_trial_executor.run_sim2real_experiment_executor`: change the model
  cache key from `exp_property` to `(participantId, exp_property)` and take an
  optional `params_by_participant`. Also derive the lapse RNG per participant so
  `robust`'s lapse draws differ between people.
* `sim2real_study_runner.run_sim2real_study`: draw the pool and pass it down.

### Server / docs

* `server/schemas.py`: add the mode to the `mode` description, plus the new
  optional sampling fields on `SimulationRequest`.
* `server/pipeline.py`: extend the `run_simulation_stage` docstring's mode list;
  see section 4.1 for the two changes that are not just documentation.
* `docs/user_guide.md`: document the mode next to the existing four; regenerate
  `site/` from it.

### 4.1 Server impact — what actually changes

Nothing changes for existing runs. `SimulationRequest.mode` defaults to
`whole_experiment`, and the UI never sends `mode` at all (no occurrence in
`UI/`), so every UI-driven `/simulate` keeps today's behaviour exactly. The mode
is opt-in from an API caller or a notebook.

Five real touch points, two of which are more than documentation:

1. **`mode` is an unvalidated `str`, not a `Literal`.** So the server already
   *accepts* `diverse_participant` today and fails deep inside
   `select_trial_rows` with "mode must be one of: ...". After this change it
   simply works. Optional hardening: tighten the field to a `Literal` so a typo
   fails at the request boundary rather than mid-run.

2. **The mlProxyBaselines loop reuses `request.mode`.** In
   `run_simulation_stage`, a design that names `mlProxyBaselines` alongside a
   research agent re-runs the whole study once per baseline with
   `mode=request.mode`. If `run_experiment` *raises* for baselines under
   `diverse_participant` (section 4, `src/api.py`), that combination 500s in
   diverse mode. Rule to implement instead:
   * raise only when the **primary** runner is a baseline (the user asked for
     fitted diversity from a model that has no fitted humans — worth an error);
   * in the secondary baseline loop, pass `mode="whole_experiment"` explicitly
     and report it in the payload (e.g. `baseline_mode`), so the proxy
     comparison still runs and the difference is visible rather than silent.

3. **`save_results` must keep returning exactly two paths.**
   `pipeline.py:1255` unpacks `csv_path, json_path = study.save_results(...)`.
   So the participant-parameters table (section 5) ships as a *side* artifact
   written by the same call and exposed as `study.participant_parameters_path`
   / a new `files.participant_parameters` key in the stage payload — **not** as
   a third tuple element.

4. **New result columns flow into the payload automatically.** `preview` uses
   `frame_records`, which maps every column through `jsonable`, and
   `frame_payload` derives its `columns` list from the frame — so
   `fitted_participant_id`, `parameter_source` and the `sampled_*` columns
   appear in `/simulate` previews and in the results table without any schema
   change. Check `jsonable` on the fallback rows' NaNs, and consider whether the
   UI's results table wants the `sampled_*` columns hidden by default.

5. **Output directory is keyed by mode.** `app.py` submits the job with
   `output_subdir=f"simulated_results/{request.mode}"`, so diverse runs land in
   `simulated_results/diverse_participant/` — a safe path segment, no collision
   with existing `whole_experiment/` output.

Two server-shaped correctness notes for the implementation:

* **CoXAM's two tasks are two `/simulate` calls** merged by
  `_merge_coxam_task_results`. Forward and counterfactual draw from *different*
  pools, so one virtual participant legitimately carries two different
  `fitted_participant_id`s in the merged table. Add a `parameter_pool` /
  `pool_task` column so the merged frame is self-explanatory instead of looking
  inconsistent.
* **Multi-dataset studies** loop `_run_agent_experiment` once per dataset level
  (`_run_multi_dataset_experiment`). Seed the draw on `(dataset level,
  participantId)` so each level's participants — which are disjoint by
  construction, per that function's own comment — get a stable, reproducible
  assignment rather than the same first-N pool rows in every level.
* **Cache the pool files.** `/simulate` runs per job on a long-lived study
  session; `coxam_counterfactual_replay.csv` is 10,758 × 47 and
  `coxam_forward_fit_mushrooms.csv` is 71k rows. Load through an `lru_cache`d
  reader so repeated simulations don't re-parse them.

---

## 5. Provenance columns on the results

Every output row gains (so a run can be audited and the tutorials can *show* the
sampled distribution):

| column | meaning |
| --- | --- |
| `fitted_participant_id` | the pool row's anonymised participant id |
| `parameter_source` | `pool` / `fitted_mean_fallback` / `override` |
| `sampled_<param>` | one column per drawn parameter |

Plus a `study.participant_parameters` DataFrame (one row per virtual
participant) stored alongside `simulated_results`, written by `save_results` as
a side artifact — the return value stays a 2-tuple, since `server/pipeline.py`
unpacks exactly `csv_path, json_path` (section 4.1).

---

## 6. Tests

New `tests/test_diverse_participant.py`:

1. **Pool loading** — each spec's file exists, filters resolve, every mapped
   column is present, and each condition cell is non-empty for the datasets the
   framework actually fitted (guards against a `build_human_data.py` rename).
2. **Determinism** — same `sampling_seed` twice → identical
   `participant_parameters`; different seed → different assignment.
3. **Variance** — the property the whole change exists for: on a small synthetic
   design, `whole_experiment` yields within-condition SD ≈ 0 (Sim2Real) while
   `diverse_participant` yields SD > 0, and no participant's parameter row is
   shared with another when the pool is larger than N.
4. **Assignment stability** — one participant has exactly one parameter row
   across all their trials within a condition cell.
5. **Range safety** — every value handed to a CoXAM env sits inside that env's
   trained range after clipping.
6. **Fallback** — CoAX + `mushrooms` (never fitted) warns and falls back rather
   than raising or silently producing clones.
7. **Mode rejection** — `diverse_participant` with a baseline model raises with a
   message naming the research agents.

Existing mode tests (`tests/test_knn_baseline.py`, `tests/test_multi_dataset_api.py`)
must keep passing untouched.

---

## 7. Notebook / doc follow-up

* Re-run both `tutorials/bug-fix/*.ipynb` with the new mode and refresh the
  stored outputs. Expect Sim2Real's `std = 0.0000` columns to become non-zero and
  its `1e15` t-statistics to become finite; expect CoXAM's `2.7e−10` p-values to
  move into a plausible range.
* Add a short cell after `run_experiment` displaying `study.participant_parameters`
  — it makes the mode's meaning obvious and doubles as the sanity check that the
  pool was actually filtered by condition.
* Consider making `diverse_participant` the tutorials' default and keeping
  `whole_experiment` as the "one canonical participant" debugging mode.

---

## 8. Open decisions

1. `ddm_s → decision_noise` for CoXAM forward: clip (recommended), rank-map, or
   skip?
2. wine_quality forward `opportunity_cost`: borrow γ from the counterfactual
   replay (recommended), or leave at default?
3. Fitted per-participant `Strategy` as `policy_override` for CoXAM forward /
   the already-per-participant `strategy` column for Sim2Real: Sim2Real should
   use it (it is part of that fit); CoXAM forward stays opt-in — confirm.
4. Should `diverse_participant` become the default mode for the tutorials, or
   stay opt-in?


---

## 9. What landed, and what it measured

### Commits

| commit | what |
| --- | --- |
| `ff4f711` | `src/virtual_experiment_executor/participant_pools.py` — the four pools, matching, and the sampler |
| `5641be1` | `select_trial_rows` accepts the mode |
| `8d7eb37` | condition relaxation for cells the fit never covered |
| `64b1521` | Sim2Real: per-participant model, strategy and lapse RNG |
| `6deecf8` | CoAX: per-participant model templates via `participant_models` |
| `99abefb` | CoXAM: per-participant parameters plus a per-episode env seed |
| `013b3fc` | `study.run_experiment` / `save_results` |
| `abf4600` | `/simulate`, including the mlProxyBaselines fix |

### Measured

**Sim2Real, the tutorial's own design** (12 participants per condition, the full
published corpus, `pairwise_condition_tests` with Holm):

| | `whole_experiment` | `diverse_participant` |
| --- | --- | --- |
| SD, faithful / sparse / sparse_robust | **0.0000 / 0.0000 / 0.0000** | 0.1925 / 0.1718 / 0.1177 |
| largest \|t\| | 7.21e+15 | 2.72 |
| smallest p (raw) | 2.5e-320 | 0.0149 |
| smallest p (Holm) | 1.5e-319 | 0.0894 |

**CoAX** (6 participants per condition, corpus-backed, training + testing
blocks): within-condition SD 0.0000 -> 0.098 (attribution) and 0.132
(importance).

**CoXAM forward** (8 participants per condition, wine_quality, surrogates
fitted against a freshly trained MLP), within-condition SD:

| condition | `whole_experiment` | `diverse_participant` |
| --- | --- | --- |
| decision_tree | 0.0554 | 0.1567 |
| hybrid | 0.0522 | 0.0668 |
| logistic_regression | 0.0579 | 0.0999 |

Note what this does *not* claim. CoXAM's shared-parameter SD was never zero --
its participants see different instances -- so the mode does not rescue it from
a divide-by-nothing; it replaces item-sampling noise with variance that is
mostly real individual differences. Nor do p-values move in one direction: in
this run the strongest comparison got *stronger* (p 0.021 -> 0.009), because
both the means and the spreads move. The MLP is retrained per run and is not
seeded, so these numbers vary run to run; the SD increase is the stable part.

Two limits worth knowing. The logistic_regression and hybrid conditions move
least, because the LR surrogate reproduces the AI on nearly every instance --
near that ceiling no parameter has leverage, so individual differences have
little to express. And wine_quality has no fitted `opportunity_cost`
(section 3.3), so only two of the three parameters vary there.

### Left to do

1. Re-run both `tutorials/bug-fix/*.ipynb` under the new mode and refresh their
   stored outputs.
2. Decide the `ddm_s -> decision_noise` policy (section 8, question 1). It is
   currently clipped; the pool spec's `ranges` is the single place to change it.
3. wine_quality's missing forward `opportunity_cost` (section 8, question 2) is
   still unfilled -- the runner keeps its default there.
4. Whether the tutorials should default to the new mode (section 8, question 4).

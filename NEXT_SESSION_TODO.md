# Where things stand — 2026-08-09

Untracked working note. Companions: `COXAM_INTEGRATION_HANDOFF.md`,
`SIM2REAL_ATTRIBUTION_SUM_HANDOFF.md`.

Run this first:

```bash
nice -n 15 env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE \
  conda run -n xaik-api-dev python -m pytest tests -q
```
Expect **221 passed**.

---

## ⚠️ UNCOMMITTED — do this first or it is lost

`git status` is clean apart from these. Everything else was pushed to main
(`cd330a4` CoXAM integration, `72d660c` sim2real workstream).

- **`src/api.py`** — the `run_experiment()` agent dispatch (`_AGENT_RUNNERS`,
  `_run_agent_experiment`). This is finished and verified, just not committed.
- **`tests/test_run_experiment_dispatch.py`** — 12 tests, all passing.
- Left uncommitted on purpose: the three `*.md` handoffs, `experiment_output/*`,
  and 36 dirty tracked `.pyc`.

**No `Co-Authored-By` trailer in commits** — this repo is going open source.

---

## Correction to carry forward

`faithful | sparse | robust | sparse_robust` are **`xai_properties`**, NOT
`xai_type`. sim2real's `xai_type` is `attribution`, and the support matrix
already declares `cognitive_models.sim2real.xai_properties` with exactly those
four. I conflated the two and reported a "wrong declared vocabulary" that was
not wrong — the support matrix needs no change here.

Three spellings exist for the same concept; unifying them is part of the work:

| layer | spelling |
|---|---|
| `support_matrix.json` (`cognitive_models.sim2real`) | `xai_properties` |
| `fit_sim2real_attribution_sum_to_participants.py` | `exp_property` |
| `src/experiment_planner/preview.py` | `explanation_property` |

---

## 1. UI→backend design converter (`src/experiment_planner/design_export.py`)

Parsing itself is fine and vocabulary-agnostic: coax, coxam forward, and coxam
counterfactual exports all parse with **zero warnings**, and
`counterfactual_sim → counterfactual_accuracy` aliasing works. The gaps are all
downstream of parsing.

**Open questions — answer before coding:**
- What does the UI call the sim2real condition factor, and does
  `IV_FACTOR_ALIASES` need an entry mapping it to `xai_properties`?
- Is `xai_properties` a *declarable IV* (i.e. in the matrix's top-level `ivs`),
  or only a capability listed under `cognitive_models.sim2real`? I was checking
  this when the session ended — **unresolved**.

**Gaps, most severe first:**

1. **`apply_design_export` never calls `set_cognitive_model`.** It parses
   `model_framework` from `userModel` and drops it. On the notebook path
   `cognitive_model_id` stays `"placeholder"`, so with the new dispatch
   `run_experiment()` silently runs the placeholder. Same failure mode as the
   bug the dispatch just fixed, re-entering through the export path. The
   *server* path is unaffected — it routes via
   `pipeline.participant_runner(study)` reading `design_export.model_framework`.
2. **`user_task` is derived nowhere** — not the converter, not `pipeline.py`,
   not `schemas.py`. A CoXAM design whose DV is `counterfactual_sim` **silently
   runs forward simulation**. Counterfactual is a separate trained agent with a
   different DV, so this produces confidently wrong results. The DV name is the
   only signal the export carries.
3. **`cognitive_config` is parsed and never applied.** UI cognitive params never
   reach the study.
4. **No level-vs-vocabulary validation at parse time.** Any levels are accepted
   against any `userModel`.

**Separator collision (needs re-verification).** `LEVEL_SEPARATOR` splits on
`|`, `,`, `/`, `vs`, `versus` — the same characters someone would use to write a
combined condition. Measured under the *wrong* factor name (`xai_type`), so
**re-run as `xai_properties` before trusting this**; `split_levels` branches on
`name=`, so the result may differ:

| UI phrasing | parsed | |
|---|---|---|
| `Sparse + Robust` / `Sparse & Robust` / `Sparse-Robust` | `sparse_robust` | ok |
| `Sparse and Robust` | `sparse_and_robust` | wrong slug |
| `Sparse, Robust` / `Sparse/Robust` | `['sparse','robust']` | **condition silently vanishes** |

Validating levels against the declared vocabulary would catch every bad row.

5. **sim2real has no runner.** `COAX_FRAMEWORKS = {"coax"}`,
   `COXAM_FRAMEWORKS = {"coxam"}` — a sim2real design falls through to
   `"baseline"` and runs the placeholder, even though
   `sim2real_fitted_attribution_sum.py` exists. Bigger piece of work; keep
   separate from 1–4.

## 2. `explanations()` can silently produce nothing

With trials already generated **and** no `xai_method` IV,
`explanations(methods=["shap"])` returns **zero explanation rows and raises
nothing**. `_trial_ids_requiring_explanations_by_method` keys its id map by
`xai_method` if that column exists, else by `xai_type` — so the keys are
`xai_type` values (`attribution`) while `methods=` asks for a method name
(`shap`). No overlap → empty id list → no rows. `_iv_config_for_explanations`
compounds it by overwriting `xai_type`'s levels with `["shap"]`.

Ordering alone is **not** the problem — 3 of 4 tested configurations work.
`generate_trials()` does work directly from the design.

Suggested fix: raise when the resolved method yields an empty id set while
trials exist and the design expects explanations.

## 3. Train/test split — design decision, no code yet

The dataset split is created once in `prepare_dataset(test_size=0.2)` and does
double duty (verified empirically, 1279/320):

- participant **training** phase ⊆ model **TRAIN** split
- participant **testing** phase ⊆ model **TEST** split, zero overlap
- `trials.py:189-190` is where this is decided

So the split *must* precede trial generation — it *is* the two pools — and it
already does. It cannot change after `train_AI_model()` without leakage, which
is why `reencode_prepared_dataset` preserves it and no re-split function exists.

Real question: should participant phases be tied to the model's split at all?
- **(a) status quo** — participants' training phase is AI-training data
- **(b) both phases sub-split from the TEST pool** — most defensible; the
  "decide later / auto-fill" shape; only needs `trials.py:189-190` to change.
  **My recommendation.** Open sub-question: phase disjointness is currently free
  (different splits) and would need explicit enforcement.
- **(c) explicit `phase_split=` over the whole dataset** — reopens leakage

Also note `max_trial_instances=300` caps the **testing** pool only.

## 4. CoXAM — unresolved after the 7,680-trial evaluation

- **DT path lags and is unexplained.** Scored the sweep's way (surrogate flip)
  on a balanced pool, `cf_dt_read` = 0.242 vs the sweep's 0.857. LR matches
  (0.506 vs 0.420; 0.476 vs 0.558), so the LR port is faithful. Candidates,
  neither verified: the forced arm pinned `counterfactual_tree_depth=1` while
  the sweep sweeps 0/1/2; and this study's DT surrogate is fit over a skewed
  1,599-row split rather than the corpus's balanced 400. `cf_dt_read` flips the
  **AI** more often than the **surrogate** (0.351 vs 0.242) — a low-fidelity
  tree. **Chase this before trusting DT counterfactual rates.**
- **The CF policy is at chance.** Balanced pool: DT 0.221 vs random 0.287, LR
  0.325 vs 0.308, hybrid 0.342 vs 0.279. Zero illegal actions and a sensible
  strategy mix, but no better than uniform selection on this data.
- Use `coxam_balanced_instance_ids(study)` for trial generation — balancing
  roughly triples success (corpus is 200/200; a raw split is ~9% class 1).

## 5. sim2real workstream (paused, see its own handoff)

- **`candidate_grid()`'s in-code defaults are still the trimmed ones that made
  NLL worse in all four conditions.** Revert before any refit — top priority.
- Delta standardisation, and low `comparison_C` vs `lapse_rate=1.0`, still open.

## 6. Housekeeping

- **713 real participant IDs** across 8 files, all `.gitignore`d; **0 in git
  history**; 0 in either commit. Verified against the actual ID set, not a
  regex — a 24-hex pattern gives false positives (26-char notebook cell IDs).
  Anonymization plan when wanted: stable `P001…P713` map in an ignored
  `participant_id_map.csv`, after which the CSVs are safe to track.
- 36 tracked `.pyc` files are dirty on every diff — `git rm --cached` + a
  `.gitignore` entry would stop that.
- CoAX's `default_cognitive_params("coax")` returns
  `cog_retrieval_threshold=-0.3`, outside its own declared `[-2.3, -1.5]`.
  Same class of bug CoXAM's had; left unfixed deliberately.

# Strategy Parameter Effects

This document describes the lean timing model, the live strategy parameters, and the algorithmic flow in each source file after the parameter rename.

Fixed assumptions:

- Memory retrieval is always `0.5` seconds.
- Reading one displayed value, coefficient, threshold, or factor is always `1.0` second.
- One mental calculation is always `0.0` seconds.
- DDM non-decision time is fixed inside the DDM helper at `0.5` seconds.
- DDM gain is fixed at `1.0`.
- Memory decay is fixed at `0.5`.
- Displayed significant figures are fixed at `2` for the current RL and sweep scripts.
- Parameter sweeps fix `memory_recall_noise = 0.5`, `retrieval_candidate_count = 3`, `simulation_sample_count = 16`, and `depth_choice_temperature = 1.0`.
- Counterfactual strategies do not include DDM decision time; they only use read, retrieval, and mental-calculation time.

## Live Parameters

| Parameter | Used by | Effect on time | Effect on accuracy or success |
|---|---|---|---|
| `memory_recall_threshold` | Memory-backed LR, DT, and recall strategies | Can shorten failed retrieval paths when set high. | Higher values increase recall failures; lower values allow more chunks to influence a response. |
| `memory_recall_noise` | Memory retrieval | No direct time effect. | Fixed to `0.5` in the parameter sweep; higher values make retrieved chunks less stable. |
| `memory_mismatch_penalty` | Memory retrieval | No direct time effect. | Penalizes partially matching chunks; stronger negative values reduce wrong matches. |
| `cue_association_strength` | Memory retrieval | No direct time effect. | Increases activation from matching cues. |
| `working_memory_capacity` | `CombinedMemory` | No direct time effect under constant retrieval time. | Larger values keep more recent chunks available as exact working-memory hits. |
| `retrieval_candidate_count` | Top-k memory retrieval, LR/DT stochastic retrieval | No direct cognitive time effect; runtime compute can increase. | Fixed to `3` in the parameter sweep; more candidates can recover useful alternatives but may add confusion. |
| `retrieved_combo_count` | `recall_change_lr`, `recall_change_dt` | Retrieval remains `0.5` seconds. | More recalled combo chunks can improve aggregate recall or mix conflicting memories. |
| `memory_refresh_probability` | Number/chunk refresh after retrieval | No immediate time effect. | Strengthens chunks for later trials. |
| `max_memory_refresh_probability` | Stochastic DT refresh | No immediate time effect. | Caps how strongly sampled path chunks are refreshed. |
| `explanation_access_mode` | LR calculation, DT traverse, DT counterfactual | `read` pays read costs; `retrieve` pays memory retrieval costs. | `read` uses displayed explanations; `retrieve` depends on recalled chunks. |
| `simulation_sample_count` | Stochastic LR/DT retrieval estimates | No cognitive time effect; script runtime increases. | Fixed to `16` in the parameter sweep; higher values stabilize estimated probabilities. |
| `selected_feature_indices` | LR calculation, LR heuristic, LR counterfactuals | Fewer selected features reduce reads and calculations. | Can hurt accuracy/success if important features are excluded. |
| `decision_boundary` | Forward DDM decisions only | Larger values usually increase DDM decision time. | Usually makes choices more decisive when evidence is reliable. |
| `decision_noise` | Forward DDM decisions only | Changes DDM decision time through the DDM formula. | Higher values weaken evidence and usually lower accuracy. |
| `evidence_scaling` | LR evidence scaling before DDM | No direct time effect. | Changes evidence scale before the DDM. Usually keep fixed. |
| `counterfactual_tree_depth` | DT path counterfactual | Deeper choices require more prior path work. | Deeper edits are more local; shallow edits can be broader. |
| `depth_choice_temperature` | DT path counterfactual | No direct time effect except via depth choice distribution. | Fixed to `1.0` in the parameter sweep; higher values spread probability across more depths. |
| `counterfactual_overshoot_fraction` | Counterfactual evaluation wrapper/scripts | No reasoning-time effect. | Pushes edits farther past the computed boundary, often increasing flip success. |
| `preferred_change_direction` | LR recall counterfactual | No direct time effect. | Aligns recalled LR edits with increasing or decreasing the LR score. |
| `initial_belief_variance` | LR heuristic memory initialization | No direct time effect. | Controls initial uncertainty in remembered coefficients. |
| `min_learning_curvature` | LR heuristic memory refresh | No direct time effect. | Stabilizes the Bayesian update after feedback. |
| `feasibility_leeway` | LR heuristic counterfactual | No direct time effect. | Allows larger normalized edits before marking a change infeasible. |
| `time_penalty_weight` | RL reward functions | Does not change raw strategy time. | Changes the policy reward tradeoff between correctness/success and time. |

## `src/memory.py`

### `Chunk`

Algorithm:

1. Store chunk name, slot dictionary, exact retrieval times, and probabilistic refresh events.
2. `base_level_activation` sums fixed-decay traces from prior retrievals and probabilistic refreshes.
3. `similarity_to` requires hard matches on type/kind slots and applies `memory_mismatch_penalty` to other mismatches.
4. `activation` adds base activation, similarity, and cue association strength.

### `DeclarativeMemory`

Algorithm:

1. Keep all chunks and a simple memory clock.
2. `add_chunk` stores a new chunk and optionally records retrieval at the current clock time.
3. `retrieve` computes activation for every chunk, adds optional `memory_recall_noise`, and returns the best chunk above `memory_recall_threshold`.
4. Retrieval always returns `0.5` seconds.
5. `retrieval_success_prob` converts top activation into a deterministic or logistic success probability.

### `WorkingMemoryQueue` and `CombinedMemory`

Algorithm:

1. Check working memory first for exact slot matches.
2. If found, return the chunk in `0.5` seconds.
3. If not found, query declarative memory.
4. Successful declarative matches are inserted into working memory.
5. `topk_retrievals_with_prob_refresh` returns a probability distribution over top retrieved chunks plus a no-recall probability.
6. Optional refresh events strengthen likely chunks for future trials.

### Number Storage Helpers

Algorithm:

1. `remember_number_to_sf` breaks a number into sign, exponent, and significant digits.
2. The metadata chunk stores sign/exponent.
3. One digit chunk is stored per significant digit.
4. `build_number_profile` retrieves the metadata and digit distributions and computes expected retrieval time as a chain of fixed `0.5` second retrievals.

## `src/cognitive.py`

### Shared Timing And Decision Helpers

Algorithm:

1. Store fixed timing constants: `READ_TIME = 1.0`, `MENTAL_CALCULATION_TIME = 0.0`, and `DDM_NON_DECISION_TIME = 0.5`.
2. `round_to_sf` rounds displayed values to a requested number of significant figures.
3. `lr_evidence` scales LR terms with `evidence_scaling`.
4. `drift_diffusion_decision` maps evidence to choice probability and DDM response time with fixed `Tnd = 0.5` and gain `1.0` in the strategy calls.
5. `slider_step` and `snap_to_step` provide shared numeric counterfactual step logic.

## `src/lr_memory.py`

### `lr_calculation`

Algorithm:

1. Filter coefficients by `selected_feature_indices` when provided.
2. In `read` mode, read the intercept, each coefficient, and each feature value.
3. Add `0.0` seconds for each nonzero coefficient-value multiplication.
4. Scale the summed evidence with `evidence_scaling`.
5. Run the DDM and add its response time.
6. In `retrieve` mode, retrieve digit profiles for the intercept and coefficients.
7. Sample reconstructed numbers for `simulation_sample_count` runs.
8. Average predicted probabilities and time over runs.

### `refresh_lr_calculation_in_memory`

Algorithm:

1. Refresh remembered intercept metadata and digits.
2. Refresh selected coefficient metadata and digits.
3. Advance memory time after each successful refresh.

### `cf_lr_calculation`

Algorithm:

1. Compute the LR score using raw coefficients and raw feature values.
2. Round the displayed score.
3. For each candidate feature, compute the required change to cancel the displayed score.
4. Test feasibility against bounds.
5. Weight feasible changes by coefficient magnitude.
6. Return feature-selection probabilities, mean deltas, and basic read/calculation time.
7. Optionally save directional LR counterfactual combo chunks for later recall.

### `recall_change_lr`

Algorithm:

1. Try to retrieve a directional LR counterfactual combo chunk.
2. If found, return its feature probabilities and deltas.
3. If no combo is found, fall back to older per-feature change chunks.
4. Mean time is combo retrieval time plus one feature read.

## `src/heuristic_lr_model.py`

### `add_lr_heuristic_to_memory`

Algorithm:

1. Store an intercept belief chunk with mean and variance.
2. Store one coefficient belief chunk per LR coefficient.
3. `initial_belief_variance` controls initial uncertainty.

### `lr_heuristic`

Algorithm:

1. Retrieve the intercept belief and each selected coefficient belief.
2. Read selected feature values once.
3. For `simulation_sample_count` runs, sample intercept and coefficients from recalled beliefs.
4. Build LR evidence and run the DDM.
5. Return averaged probabilities and total time: retrieval + reads + mental calculations + average DDM time.

### `refresh_lr_heuristic_in_memory`

Algorithm:

1. Use feedback label and predicted probability to update recalled beliefs.
2. Apply a diagonal Bayesian logistic update.
3. Clamp curvature by `min_learning_curvature`.
4. Update only selected features when `selected_feature_indices` is provided.

### `cf_lr_heuristic`

Algorithm:

1. Retrieve coefficient beliefs and read normalized feature values.
2. Estimate the current LR score from remembered beliefs.
3. Compute feasible normalized feature changes that would cross the LR boundary.
4. Convert numeric deltas back to original units.
5. Weight feasible edits by coefficient magnitude and return probabilities, deltas, and basic time.

## `src/dt_memory.py`

### `add_dt_to_memory`

Algorithm:

1. Traverse the explanation tree.
2. Store node type, feature key, threshold pointer, threshold digits, child pointers, and leaf class chunks.
3. Numeric thresholds are stored through the shared number-memory helper.

### `dt_traverse`

Algorithm:

1. Start at the root node.
2. In `read` mode, read the displayed feature and threshold directly.
3. In `retrieve` mode, retrieve feature chunks, threshold pointer chunks, and threshold number profiles.
4. Read the instance value at each visited node.
5. Categorical branch decisions are exact.
6. Numeric branch decisions use the DDM.
7. Stop at a leaf and return class probabilities and expected time.
8. Refresh retrieved path chunks after stochastic retrieval.

### `refresh_dt_path_in_memory`

Algorithm:

1. Traverse the literal tree path for the current instance.
2. Refresh the feature, threshold, child-pointer, and leaf chunks used on that path.

### `cf_change_path_dt`

Algorithm:

1. Build the literal or recalled path to a decision depth.
2. Select an edit depth using `counterfactual_tree_depth` and `depth_choice_temperature`.
3. Compute the minimal feature change needed to flip the selected node test.
4. Count only retrieval, read, and mental-calculation time.
5. Aggregate feature probabilities, deltas, and conditional mean times.
6. In retrieve mode, save a DT counterfactual combo chunk for later recall.

### `recall_change_dt`

Algorithm:

1. Retrieve stored DT counterfactual combo chunks.
2. Merge feature probabilities across recalled chunks.
3. Recompute the current signed delta needed to cross the stored threshold.
4. Return feature probabilities, deltas, and retrieval-plus-read time.

## `src/utils.py`

### `LogisticRegressionInterpreter`

Algorithm:

1. Load a local LR explanation row for an app/model.
2. Convert normalized coefficients into raw input space when metadata bounds allow.
3. Collapse simple binary categorical coefficients into one raw feature where possible.
4. `apply_to_instance` computes the raw LR score.

### `DecisionTreeInterpreter`

Algorithm:

1. Load a local decision-tree explanation row at a requested tree depth.
2. Parse the stored tree structure.
3. `apply_to_instance` traverses the tree and returns the predicted class.
4. `print_tree` returns a string representation instead of printing.

### `AIDatasetLoader`

Algorithm:

1. Load raw or normalized feature vectors by instance id.
2. Load the corresponding AI predictions.
3. Provide app-specific numeric bounds and category metadata.

## `scripts/parameter_effect_sweep.py`

Algorithm:

1. Load datasets, predictions, LR explanations, and DT explanations from `datasets/`.
2. Build evaluation bundles for each available app/model pair.
3. For each parameter grid value, hold all other parameters fixed.
4. For each dataset, draw `--max-instances` randomized instances for each of `--repeats` runs.
5. Run forward strategies: `lr_calculation_read`, `lr_calculation_retrieve`, `lr_heuristic`, `dt_read`, and `dt_retrieve`.
6. Sample the output prediction from the returned probability vector, then record accuracy, target probability, prediction time, repeat id, and instance id.
7. Run counterfactual strategies: `cf_lr_calculation`, `cf_lr_heuristic`, `cf_dt_read`, `cf_dt_retrieve`, `recall_change_dt`, and `recall_change_lr`.
8. Record expected counterfactual flip success, expected time, repeat id, and instance id.
9. Write CSV files and PNG plots with side-by-side dataset panels and 95% CI bars.

Run:

```bash
python scripts/parameter_effect_sweep.py --task both --max-instances 20 --repeats 5
```

Optional narrower run:

```bash
python scripts/parameter_effect_sweep.py --task forward --apps wine_quality mushrooms --max-instances 10 --repeats 3 --seed 123
```

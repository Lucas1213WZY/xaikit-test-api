"""Run a xaikitTest study's counterfactual trials through CoXAM's CF policy.

Counterfactual simulation is a separate agent from the forward meta-policy in
``coxam_study_runner.py``: a different trained checkpoint, a different
observation space, and a different DV (did the named change flip the AI's
prediction, rather than was the AI's prediction reproduced). This module is its
runner; ``coxam_study_runner.run_coxam_study`` dispatches here for a
counterfactual design.

The environment itself is ported in
``src/cognitive_models/cognitive_models/coxam/counterfactual_env.py`` -- read
its module docstring first, since two of its behaviours (retrieve-mode-only
combo saving, and forward-trial priming being required before
``recall_change_*`` can retrieve anything) are load-bearing.
"""

from __future__ import annotations

import os
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from src.cognitive_models.cognitive_models.coxam.counterfactual_env import (  # noqa: E402
    STRATEGIES,
    XAI_TYPES,
    CounterfactualPolicyEnv,
)
from src.experiment_planner import select_trial_rows  # noqa: E402

from .coxam_trial_executor import (  # noqa: E402
    TRAINED_POLICIES_ROOT,
    _canonical_coxam_condition,
)

#: The CF policy that ships with the repo.
COUNTERFACTUAL_POLICY_PATH = (
    TRAINED_POLICIES_ROOT / "counterfactual" / "models" / "simple_chi_model.zip"
)

#: Carries study trial order through the per-participant episodes.
ORDER_COLUMN = "__coxam_cf_trial_order"

#: The design's condition vocabulary -> the CF environment's own.
_CONDITION_TO_XAI_TYPE = {
    "decision_tree": "DT",
    "linear_regression": "LR",
    "hybrid": "DT+LR",
}


def load_counterfactual_policy(path: Optional[Any] = None):
    """Load the trained counterfactual policy (plain ``PPO``, not maskable)."""
    from stable_baselines3 import PPO

    return PPO.load(path or COUNTERFACTUAL_POLICY_PATH, device="cpu")


def _condition_index(xai_type: Any) -> int:
    """Design ``xai_type`` -> the CF env's condition index."""
    label = _CONDITION_TO_XAI_TYPE[_canonical_coxam_condition(xai_type)]
    return next(key for key, value in XAI_TYPES.items() if value == label)


def _shown_label(trial: Mapping[str, Any], condition_label: str) -> str:
    """Which family this trial displays, as the CF env labels it."""
    if condition_label != "DT+LR":
        return condition_label
    shown = trial.get("shown_xai_type", trial.get("xai_type"))
    return _CONDITION_TO_XAI_TYPE[_canonical_coxam_condition(shown)]


def _with_xai(trial: Mapping[str, Any]) -> int:
    value = trial.get("tested_w_xai", False)
    if isinstance(value, str):
        value = value.strip().lower() in {"true", "1", "yes", "y"}
    return int(bool(value))


class TrialDrivenCounterfactualEnv(CounterfactualPolicyEnv):
    """Forces a study's own trials instead of sampling an episode.

    The base env samples its forward and counterfactual instance ids, its
    condition, and its with-XAI schedule. Here all four come from the study's
    generated trials, so the simulation answers the trials the design actually
    specifies.
    """

    def reset(self, seed=None, options=None):
        options = dict(options or {})
        required = {"condition_index", "xai_schedule", "with_xai_schedule",
                    "counterfactual_instance_ids", "forward_instance_ids"}
        missing = required - set(options)
        if missing:
            raise ValueError(
                f"TrialDrivenCounterfactualEnv.reset needs {sorted(missing)} in options; "
                "it never samples an episode itself."
            )
        return super().reset(seed=seed, options=options)


def _result_row(
    trial: Mapping[str, Any],
    step: int,
    info: Mapping[str, Any],
    dvs: Optional[Mapping[str, Any]],
) -> dict[str, Any]:
    """One output row per trial, matching the forward runner's shape.

    The counterfactual DV is ``success``: whether the change the participant
    named actually flipped the AI's prediction. It fills any DV whose name
    mentions accuracy, the same convention the forward runner and
    ``run_experiment_executor`` use.
    """
    success = info.get("success")
    dv_columns = {
        name: int(success)
        for name in (dvs or {})
        if ("accuracy" in name.lower() or "success" in name.lower()) and success is not None
    }
    return {
        **dv_columns,
        **trial,
        "phase": trial.get("phase"),
        "step": step,
        "condition_name": info.get("condition"),
        "shown_xai_type": info.get("shown_xai_type"),
        "selected_strategy": info.get("strategy"),
        "counterfactual_tree_depth": info.get("depth"),
        "feature_changed": info.get("feature_changed"),
        "delta": info.get("delta"),
        "ai_prediction": info.get("ai_prediction_original"),
        "ai_prediction_counterfactual": info.get("ai_prediction_counterfactual"),
        "counterfactual_success": success,
        "invalid_under_condition": info.get("invalid_under_condition"),
        "pred_time": info.get("time"),
    }


def run_coxam_counterfactual_episode(
    trials: pd.DataFrame,
    *,
    bundle: Any,
    predict_fn: Any,
    policy: Any,
    cognitive_params: Optional[Mapping[str, float]] = None,
    forward_instance_ids: Optional[Sequence[int]] = None,
    dvs: Optional[Mapping[str, Any]] = None,
    seed: int = 0,
) -> pd.DataFrame:
    """Run one participant's counterfactual trials as a single forced episode.

    ``forward_instance_ids`` primes memory with a forward phase before the
    counterfactual phase, which is what makes the ``recall_change_*`` strategies
    able to retrieve anything at all. It defaults to the same instances as the
    counterfactual trials.
    """
    if trials.empty:
        return trials

    records = trials.to_dict("records")
    condition_index = _condition_index(records[0].get("xai_type", records[0].get("xai_method")))
    condition_label = XAI_TYPES[condition_index]

    env = TrialDrivenCounterfactualEnv(
        loader=bundle.loader,
        lr_exp=bundle.lr_sparse,
        dt_exp=bundle.dt_depth2,
        bounds=bundle.loader.get_bounds_for_app(app_id=bundle.app_id),
        predict_fn=predict_fn,
        instance_ids=list(bundle.instance_ids),
        instances_per_episode=len(records),
    )
    counterfactual_ids = [int(row["instanceId"]) for row in records]
    obs, _ = env.reset(
        seed=seed,
        options={
            "condition_index": condition_index,
            "xai_schedule": [_shown_label(row, condition_label) for row in records],
            "with_xai_schedule": [_with_xai(row) for row in records],
            "counterfactual_instance_ids": counterfactual_ids,
            "forward_instance_ids": list(forward_instance_ids or counterfactual_ids),
            "cognitive_params_fixed": dict(cognitive_params) if cognitive_params else None,
        },
    )

    rows: list[dict[str, Any]] = []
    for step, trial in enumerate(records):
        action, _ = policy.predict(obs, deterministic=True)
        obs, _reward, _terminated, _truncated, info = env.step(action)
        rows.append(_result_row(trial, step, info, dvs))
    return pd.DataFrame(rows)


def run_coxam_counterfactual_study(
    study: Any,
    *,
    bundle: Any,
    predict_fn: Any,
    mode: str = "whole_experiment",
    participant_id: Optional[int] = None,
    condition_filter: Optional[dict[str, Any]] = None,
    trials: Optional[pd.DataFrame | Sequence[dict[str, Any]]] = None,
    dvs: Optional[Mapping[str, Any]] = None,
    cognitive_params: Optional[Mapping[str, float]] = None,
    seed: int = 0,
    store: bool = True,
) -> pd.DataFrame:
    """Execute a study's trials as counterfactual simulation.

    ``bundle`` and ``predict_fn`` are built by ``run_coxam_study``, which owns
    surrogate fitting and the instance-coverage preflight; this function only
    adds the counterfactual policy and environment on top of them.
    """
    trials_df = pd.DataFrame(study.trials if trials is None else trials).copy()
    if trials_df.empty:
        raise RuntimeError("No generated trials to run. Call study.generate_trials(...) first.")

    selected = select_trial_rows(
        trials_df, mode, participant_id=participant_id, condition_filter=condition_filter,
    ).copy()
    if selected.empty:
        return selected
    selected[ORDER_COLUMN] = range(len(selected))

    policy = load_counterfactual_policy()

    group_columns = [
        column for column in ("participantId", "block") if column in selected.columns
    ]
    groups = (
        selected.groupby(group_columns, sort=False, dropna=False)
        if group_columns
        else [(None, selected)]
    )

    runs = []
    for offset, (_key, rows) in enumerate(groups):
        ordered = rows.sort_values(ORDER_COLUMN, kind="stable")
        runs.append(
            run_coxam_counterfactual_episode(
                ordered,
                bundle=bundle,
                predict_fn=predict_fn,
                policy=policy,
                cognitive_params=cognitive_params,
                dvs=study.DVs if dvs is None else dvs,
                seed=seed + offset,
            )
        )

    results = pd.concat(runs, ignore_index=True, sort=False)
    if ORDER_COLUMN in results.columns:
        results = (
            results.sort_values(ORDER_COLUMN, kind="stable")
            .drop(columns=[ORDER_COLUMN])
            .reset_index(drop=True)
        )
    if store:
        study.simulated_results = results
    return results


__all__ = [
    "COUNTERFACTUAL_POLICY_PATH",
    "ORDER_COLUMN",
    "STRATEGIES",
    "TrialDrivenCounterfactualEnv",
    "load_counterfactual_policy",
    "run_coxam_counterfactual_episode",
    "run_coxam_counterfactual_study",
]

"""Default cognitive-model helpers used by tutorial simulations."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from src.workflow_standard import (
    EXPLANATION_METHOD_COL,
    INSTANCE_ID_COL,
    PREDICTION_COL,
    PREDICTION_ONLY_METHOD,
)


def get_trial_instance_attributes(
    trial_info: dict[str, Any],
    raw_dataset: pd.DataFrame,
    *,
    label_column: str,
) -> dict[str, float]:
    """Extract raw feature values for the trial's original dataset row id.

    Args:
        trial_info: The trial row, supplying the instance id.
        raw_dataset: Raw feature values to look the instance up in.
        label_column: Target column name, excluded from the result.
    """
    instance_id = int(trial_info[INSTANCE_ID_COL])
    row = raw_dataset.iloc[instance_id]
    feature_row = row.drop(labels=[label_column], errors="ignore")
    return {k: float(v) for k, v in feature_row.items()}


def get_trial_ai_prediction(
    trial_info: dict[str, Any],
    explanation_pool: pd.DataFrame,
) -> Optional[int]:
    """Return the trained AI model prediction stored in the explanation CSV.

    Args:
        trial_info: The trial row, supplying the instance id.
        explanation_pool: Table holding the stored predictions.

    Returns:
        The predicted label, or None when the instance has no stored prediction.
    """
    if PREDICTION_COL not in explanation_pool.columns:
        return None

    instance_id = int(trial_info[INSTANCE_ID_COL])
    matches = explanation_pool[explanation_pool[INSTANCE_ID_COL].astype(int) == instance_id]
    if matches.empty:
        return None
    if EXPLANATION_METHOD_COL in matches:
        xai_method = str(
            trial_info.get("xai_method", trial_info.get("xai_type", "none"))
        ).lower()
        method_matches = matches[
            matches[EXPLANATION_METHOD_COL].astype(str).str.lower() == xai_method
        ]
        if not method_matches.empty:
            matches = method_matches
        explanation_matches = matches[matches[EXPLANATION_METHOD_COL].astype(str) != PREDICTION_ONLY_METHOD]
        if not explanation_matches.empty:
            matches = explanation_matches
    return int(matches.iloc[0][PREDICTION_COL])


def get_trial_instance_explanation(
    trial_info: dict[str, Any],
    explanation_pool: pd.DataFrame,
) -> dict[str, float]:
    """Select explanation values matching the trial's XAI method and instance.

    Args:
        trial_info: The trial row, supplying the instance id and XAI method.
        explanation_pool: Explanations to draw from.

    Returns:
        The explanation values, empty when the trial shows no explanation.
    """
    xai_method = str(trial_info.get("xai_method", trial_info.get("xai_type", "none"))).lower()
    if xai_method in {"none", "no_xai", "control"}:
        return {}
    phase = str(trial_info.get("phase", "testing")).lower()
    tested_w_xai = trial_info.get("tested_w_xai", True)
    if isinstance(tested_w_xai, str):
        tested_w_xai = tested_w_xai.strip().lower() in {"true", "1", "yes", "y"}
    if phase != "training" and not bool(tested_w_xai):
        return {}

    instance_id = int(trial_info[INSTANCE_ID_COL])
    matches = explanation_pool[
        (explanation_pool[INSTANCE_ID_COL].astype(int) == instance_id)
        & (explanation_pool[EXPLANATION_METHOD_COL].astype(str).str.lower() == xai_method)
    ]
    if matches.empty:
        return {}

    row = matches.iloc[0]
    explanation_cols = [c for c in explanation_pool.columns if c.startswith("a") and c.endswith("_i")]
    explanation = {c: float(row[c]) for c in explanation_cols}

    for optional_col in [PREDICTION_COL, "i_max", "intercept"]:
        if optional_col in row:
            explanation[optional_col] = float(row[optional_col])

    return explanation


def build_single_trial_cognitive_input(
    trial_info: dict[str, Any],
    raw_dataset: pd.DataFrame,
    explanation_pool: pd.DataFrame,
    *,
    label_column: str,
) -> dict[str, Any]:
    """Combine trial metadata, raw instance attributes, AI pred, and XAI values.

    Args:
        trial_info: The trial row being run.
        raw_dataset: Raw feature values, for the attributes shown.
        explanation_pool: Explanations and AI predictions to draw from.
        label_column: Target column name, excluded from the attributes.

    Returns:
        Everything a cognitive model needs to respond to one trial.
    """
    return {
        "trial_info": dict(trial_info),
        "instance_attributes": get_trial_instance_attributes(
            trial_info,
            raw_dataset,
            label_column=label_column,
        ),
        "instance_explanation": get_trial_instance_explanation(
            trial_info,
            explanation_pool,
        ),
        "ai_prediction": get_trial_ai_prediction(trial_info, explanation_pool),
    }


def dummy_cognitive_model(
    cognitive_params: dict[str, float],
    dvs: dict[str, list[Any]],
    trial_data: dict[str, Any],
) -> dict[str, Any]:
    """Placeholder cognitive model using current CoXAM-style parameter names.

    Args:
        cognitive_params: Agent parameters; accepted but not used by this stub.
        dvs: Dependent variables the response must supply.
        trial_data: The trial input from ``build_single_trial_cognitive_input``.

    Returns:
        One simulated response per DV.
    """
    trial_info = trial_data.get("trial_info", {})
    explanation = trial_data.get("instance_explanation", {}) or {}

    has_xai = bool(explanation) and str(trial_info.get("xai_method", "none")).lower() != "none"
    attr_values = [abs(v) for k, v in explanation.items() if k.startswith("a") and k.endswith("_i")]
    explanation_strength = float(np.mean(attr_values)) if attr_values else 0.0

    retrieval_threshold = float(cognitive_params.get(
        "retrieval_threshold",
        cognitive_params.get("cog_retrieval_threshold", -0.3),
    ))
    exemplar_distance_sensitivity = float(cognitive_params.get("exemplar_distance_sensitivity", 1.0))
    attended_features = float(cognitive_params.get("attended_features", 5.0))
    feature_class_sensitivity = float(cognitive_params.get(
        "feature_class_sensitivity",
        cognitive_params.get("cog_chi", 0.001),
    ))
    chi_value = 0.001 * feature_class_sensitivity
    ddm_a = float(cognitive_params.get("cog_ddm_a", 0.8))
    ddm_s = float(cognitive_params.get("cog_ddm_s", 1.0))
    lapse = float(cognitive_params.get("lapse", 0.005))
    latency_factor = float(cognitive_params.get("cog_latency_factor", 0.2))
    t_enc = float(cognitive_params.get("cog_T_enc", 1.5))
    t_op = float(cognitive_params.get("cog_T_op", 0.5))

    attention_bonus = 0.01 * np.clip(attended_features, 1, 5)
    distance_penalty = 0.004 * np.clip(exemplar_distance_sensitivity, 1, 10)
    accuracy_probability = (
        0.5
        + (0.08 * ddm_a)
        + (0.05 * ddm_s)
        + attention_bonus
        - distance_penalty
        - (0.03 * abs(retrieval_threshold))
    )
    if has_xai:
        accuracy_probability += chi_value + (0.05 * explanation_strength)
    accuracy_probability = float(np.clip(accuracy_probability, lapse, 1.0 - lapse))

    pred_time = t_enc + t_op + latency_factor * (1.0 + abs(retrieval_threshold))
    if has_xai:
        pred_time += ddm_a * max(ddm_s, 0.0) + explanation_strength
    pred_time = float(max(pred_time, 0.0))

    outputs: dict[str, Any] = {}
    for dv_name in dvs:
        dv_key = dv_name.lower()
        if "accuracy" in dv_key or "prob" in dv_key or "correct" in dv_key:
            outputs[dv_name] = accuracy_probability
        elif "time" in dv_key or "rt" in dv_key or "duration" in dv_key:
            outputs[dv_name] = pred_time
        else:
            outputs[dv_name] = accuracy_probability

    outputs["prob_correct"] = accuracy_probability
    outputs["pred_time"] = pred_time
    outputs["agent_prediction"] = bool(accuracy_probability > 0.5)
    outputs["ai_prediction"] = trial_data.get("ai_prediction")
    return outputs


#: CoXAM's own parameter names and operating values. The names and ranges come
#: from ``CombinedPolicyConfig`` (``src/cognitive_models/cognitive_models/coxam/
#: rl_agents/meta_policy_strategy.py``) and are corroborated by the published
#: parameter sweeps in ``src/cognitive_models/CoXAM/outputs/
#: parameter_effect_sweeps/{forward,counterfactual}_results.csv``, whose swept
#: values sit exactly on those bounds.
#:
#: These share no names with the CoAX-shaped defaults below except a
#: conceptually-similar recall threshold, which uses a different name *and* a
#: different range -- so CoAX's defaults are invalid for CoXAM and vice versa.
#: CoXAM's free cognitive parameters, exactly as the reference driver exposes
#: them (``CoXAM/rl_agents/meta_policy_strategy_dashboard.py``'s
#: ``fixed_eval_params``) and as ``CombinedStrategyPolicyEnv._sample_episode_params``
#: samples them. Values are the dashboard's own defaults.
#:
#: These three are the *only* parameters free at evaluation time. Every other
#: knob is either chosen by the sub-policy's own RL action
#: (``decision_boundary`` via ``_decision_boundary_from_bin``,
#: ``selected_feature_count`` via ``_parse_feature_action``) or is structural
#: run configuration (``instances_per_episode``, ``explanation_shown_ratio``,
#: ``target_xai_fidelity``, ``simulation_sample_count``,
#: ``retrieval_candidate_count``, ``seed``) rather than a cognitive parameter.
#:
#: Ranges come from the run config of the checkpoint actually loaded
#: (``outputs/combined_strategy_meta_policy/
#: meta_policy_strategy_masked_explanation_history_demo/mixed/config.json``),
#: not from ``CombinedPolicyConfig``'s dataclass defaults -- no run used those
#: (``memory_recall_threshold`` is [-1.0, 2.0] in every run config vs. [-5.0,
#: 2.0] in the dataclass).
#:
#: Scope note: the meta-policy and its dashboard are **forward-simulation
#: only** -- neither mentions counterfactual strategies. The counterfactual
#: strategies (``cf_lr_calculation``, ``recall_change_dt``, ...) and their
#: parameters (``counterfactual_overshoot_fraction``,
#: ``counterfactual_tree_depth``, ``max_memory_refresh_probability``) live in
#: ``CoXAM/scripts/parameter_effect_sweep.py``, which drives strategies
#: directly and bypasses the meta-policy. They are therefore not reachable
#: through this runner and are deliberately absent here.
COXAM_FORWARD_PARAMS: dict[str, float] = {
    "decision_noise": 0.4,
    "memory_recall_threshold": 0.5,
    "opportunity_cost": 0.01,
}

#: CoXAM's counterfactual policy is a separate trained agent with its own
#: parameters, taken from the ranges its checkpoint was trained on (see
#: ``coxam/counterfactual_env.py``). ``time_penalty_weight`` uses the
#: checkpoint's ``[0, 0.02]``, not the notebook's later ``[0, 0.05]``.
COXAM_COUNTERFACTUAL_PARAMS: dict[str, float] = {
    "memory_recall_threshold": -0.75,
    "random_response_rate": 0.3,
    "counterfactual_overshoot_fraction": 0.25,
    "time_penalty_weight": 0.01,
}

#: Every CoXAM parameter, for a design that runs both tasks.
COXAM_COGNITIVE_PARAMS: dict[str, float] = {
    **COXAM_FORWARD_PARAMS,
    **COXAM_COUNTERFACTUAL_PARAMS,
}


def default_cognitive_params(
    cognitive_model_id: Optional[str] = None,
    *,
    tasks: Optional[Any] = None,
) -> dict[str, float]:
    """Return default parameters for a cognitive agent.

    Args:
        cognitive_model_id: Agent the defaults are for. ``coxam`` returns
            CoXAM's own parameters; anything else (including ``None``) returns
            the CoAX-shaped placeholder set, which is what the placeholder/CoAX
            paths have always used.
        tasks: User tasks the design runs, e.g. ``["forward_simulation"]``.
            CoXAM runs forward and counterfactual simulation as two separate
            trained agents with different parameters, so this trims the result
            to the ones that apply. Omit for both sets.

    CoAX and CoXAM take disjoint parameters, so one shared default dict cannot
    be valid for both -- the CoAX-shaped ``cog_retrieval_threshold=-0.3`` sits
    outside CoXAM's supported range and used to make every coxam design report
    a validation error.
    """
    if str(cognitive_model_id or "").strip().lower().replace("-", "_") == "coxam":
        requested = {str(task).strip().lower() for task in (tasks or [])}
        if not requested:
            return dict(COXAM_COGNITIVE_PARAMS)
        params: dict[str, float] = {}
        if any("forward" in task for task in requested):
            params.update(COXAM_FORWARD_PARAMS)
        if any("counterfactual" in task for task in requested):
            params.update(COXAM_COUNTERFACTUAL_PARAMS)
        return params or dict(COXAM_COGNITIVE_PARAMS)
    return {
        "cog_retrieval_threshold": -0.3,
        "cog_latency_factor": 0.2,
        "cog_T_enc": 1.5,
        "cog_T_op": 0.5,
        "cog_ddm_a": 0.8,
        "cog_ddm_s": 1.0,
        "cog_chi": 0.001,
        "lapse": 0.005,
    }

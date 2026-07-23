"""Lightweight virtual-participant models for the tutorial study designs.

These functions implement the callable contract used by ``XAIKitTest``.  They
are calibrated study simulators for checking trial generation, execution, and
analysis end to end; they do not replace the full fitted cognitive models.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np


def _stable_rng(model_id: str, trial_info: dict[str, Any], seed: int) -> np.random.Generator:
    key = "|".join(
        [
            model_id,
            str(seed),
            str(trial_info.get("participantId", 0)),
            str(trial_info.get("trialId", trial_info.get("task_trial", 0))),
            str(trial_info.get("task", "")),
        ]
    )
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def _finish_trial(
    probability_correct: float,
    cognitive_params: dict[str, float],
    dvs: dict[str, list[Any]],
    trial_data: dict[str, Any],
    *,
    model_id: str,
    response_time: float,
) -> dict[str, Any]:
    trial_info = trial_data.get("trial_info", {})
    seed = int(cognitive_params.get("seed", 42))
    rng = _stable_rng(model_id, trial_info, seed)
    probability_correct = float(np.clip(probability_correct, 0.02, 0.98))
    is_correct = bool(rng.random() < probability_correct)

    ai_prediction = trial_data.get("ai_prediction")
    if ai_prediction is None:
        ai_prediction = int(rng.random() >= 0.5)
    ai_prediction = int(ai_prediction)
    agent_prediction = ai_prediction if is_correct else 1 - ai_prediction

    outputs: dict[str, Any] = {}
    for dv_name in dvs:
        key = dv_name.lower()
        if "time" in key or "duration" in key or key.endswith("_rt"):
            outputs[dv_name] = float(response_time)
        elif "accuracy" in key or "correct" in key:
            outputs[dv_name] = int(is_correct)
        else:
            outputs[dv_name] = int(is_correct)

    outputs.update(
        {
            "prob_correct": probability_correct,
            "pred_time": float(response_time),
            "agent_prediction": int(agent_prediction),
            "ai_prediction": ai_prediction,
        }
    )
    return outputs


def explanation_property_simulator(cognitive_params, dvs, trial_data):
    """Virtual participant for the four explanation-property conditions."""
    trial = trial_data.get("trial_info", {})
    condition = str(trial.get("validation_condition", ""))
    property_name = str(trial.get("explanation_property", "faithful"))

    probability = 0.58
    preferred = {
        "feature_check": {"faithful": 0.80, "sparse": 0.65, "robust": 0.58, "sparse_robust": 0.62},
        "change_prediction": {"faithful": 0.55, "sparse": 0.53, "robust": 0.76, "sparse_robust": 0.70},
        "predict_without_timer": {"faithful": 0.76, "sparse": 0.75, "robust": 0.58, "sparse_robust": 0.64},
        "predict_with_timer": {"faithful": 0.62, "sparse": 0.78, "robust": 0.54, "sparse_robust": 0.68},
    }
    probability = preferred.get(condition, {}).get(property_name, probability)
    response_time = 5.0 if condition == "predict_with_timer" else 8.0
    return _finish_trial(
        probability, cognitive_params, dvs, trial_data,
        model_id="explanation_property", response_time=response_time,
    )


def feature_explanation_simulator(cognitive_params, dvs, trial_data):
    """Virtual participant for none/importance/attribution prediction studies."""
    trial = trial_data.get("trial_info", {})
    xai_type = str(trial.get("xai_type", "none")).lower()
    tested_with_xai = bool(trial.get("tested_w_xai", False))
    dataset = str(trial.get("dataset", trial.get("dataset_assignment", "wine_quality")))

    probability = {
        "none": 0.57,
        "importance": 0.66 if not tested_with_xai else 0.70,
        "attribution": 0.68 if not tested_with_xai else 0.80,
    }.get(xai_type, 0.57)
    probability += {"adult": 0.02, "forest_cover": -0.02}.get(dataset, 0.0)
    response_time = 6.5 + (1.5 if tested_with_xai else 0.0)
    return _finish_trial(
        probability, cognitive_params, dvs, trial_data,
        model_id="feature_explanation", response_time=response_time,
    )


def rules_weights_simulator(cognitive_params, dvs, trial_data):
    """Virtual participant for Rules, Weights, and Hybrid study conditions."""
    trial = trial_data.get("trial_info", {})
    xai_type = str(trial.get("xai_type", "decision_tree")).lower()
    shown_type = str(trial.get("shown_xai_type", xai_type)).lower()
    tested_with_xai = bool(trial.get("tested_w_xai", False))
    task = str(trial.get("task", "forward_simulation"))
    dataset = str(trial.get("dataset", "wine_quality"))
    complexity = str(trial.get("complexity", "low"))

    if task == "counterfactual_simulation":
        probability = 0.28
        if xai_type == "logistic_regression":
            probability += 0.14 if tested_with_xai else 0.08
        elif xai_type == "decision_tree":
            probability += 0.02 if tested_with_xai else -0.02
        else:
            probability += 0.10 if tested_with_xai else 0.04
        if dataset == "mushrooms" and shown_type == "decision_tree":
            probability += 0.08
        response_time = 14.0
    else:
        probability = 0.63
        if xai_type == "decision_tree":
            probability += 0.14 if tested_with_xai else 0.00
        elif xai_type == "logistic_regression":
            probability += 0.07 if tested_with_xai else 0.08
        else:
            probability += 0.10 if tested_with_xai else 0.05
        if dataset == "mushrooms" and shown_type == "decision_tree" and tested_with_xai:
            probability += 0.04
        response_time = 9.0

    if complexity == "high":
        probability -= 0.02
        response_time += 1.5
    return _finish_trial(
        probability, cognitive_params, dvs, trial_data,
        model_id="rules_weights", response_time=response_time,
    )


__all__ = [
    "explanation_property_simulator",
    "feature_explanation_simulator",
    "rules_weights_simulator",
]

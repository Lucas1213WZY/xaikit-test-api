"""Run cognitive agents over generated experiment trials."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import inspect
import os
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from src.cognitive_models import build_single_trial_cognitive_input, dummy_cognitive_model
from src.experiment_planner import select_trial_rows


def run_experiment_executor(
    trials: list[dict[str, Any]] | pd.DataFrame,
    cognitive_params: dict[str, float],
    dvs: dict[str, list[Any]],
    raw_dataset: pd.DataFrame,
    explanation_pool: pd.DataFrame,
    *,
    mode: str = "whole_experiment",
    participant_id: Optional[int] = None,
    condition_filter: Optional[dict[str, Any]] = None,
    condition_columns: Optional[list[str]] = None,
    cognitive_model=dummy_cognitive_model,
    label_column: str,
    explanation_prefix: str = "exp_",
    cognitive_param_prefix: str = "cog_param_",
) -> pd.DataFrame:
    """Run cognitive agents over selected trials and append results."""
    trials_df = pd.DataFrame(trials).copy()
    selected = select_trial_rows(
        trials_df,
        mode,
        participant_id=participant_id,
        condition_filter=condition_filter,
    )

    if "phase" not in selected:
        selected["phase"] = "testing"
    selected["__executor_order"] = range(len(selected))

    executed_rows = []
    model_template = deepcopy(cognitive_model)
    execution_group_columns = [
        column
        for column in ["participantId", *(condition_columns or [])]
        if column in selected and column not in {"phase", "tested_w_xai"}
    ]
    participant_groups = (
        selected.groupby(execution_group_columns, sort=False, dropna=False)
        if execution_group_columns
        else [(None, selected)]
    )
    for participant_position, (_, participant_rows) in enumerate(participant_groups):
        participant_model = (
            cognitive_model if participant_position == 0 else deepcopy(model_template)
        )
        training_rows = participant_rows[participant_rows["phase"] == "training"]
        testing_rows = participant_rows[participant_rows["phase"] != "training"]

        training_contexts = [
            build_single_trial_cognitive_input(
                row.to_dict(),
                raw_dataset,
                explanation_pool,
                label_column=label_column,
            )
            for _, row in training_rows.iterrows()
        ]
        if training_contexts and hasattr(participant_model, "fit"):
            missing_targets = [
                context["trial_info"].get("instanceId")
                for context in training_contexts
                if context.get("ai_prediction") is None
            ]
            if missing_targets:
                raise ValueError(
                    "Training instances are missing AI predictions: "
                    f"{missing_targets}."
                )
            training_instances = pd.DataFrame(
                context["instance_attributes"] for context in training_contexts
            )
            training_predictions = [
                context["ai_prediction"] for context in training_contexts
            ]
            attribution_rows = [
                _attribution_values(context) for context in training_contexts
            ]
            expects_xai = any(
                _trial_expects_xai(context["trial_info"], training=True)
                for context in training_contexts
            )
            if expects_xai:
                missing_xai = [
                    context["trial_info"].get("instanceId")
                    for context, attribution in zip(
                        training_contexts,
                        attribution_rows,
                    )
                    if not attribution
                ]
                if missing_xai:
                    raise ValueError(
                        "XAI training instances are missing explanation vectors: "
                        f"{missing_xai}."
                    )

            fit_parameters = inspect.signature(participant_model.fit).parameters
            if expects_xai and "explanations" in fit_parameters:
                participant_model.fit(
                    training_instances,
                    training_predictions,
                    explanations=pd.DataFrame(attribution_rows),
                )
            else:
                participant_model.fit(training_instances, training_predictions)

        for context in training_contexts:
            executed_rows.append(_training_result_row(
                context,
                cognitive_params=cognitive_params,
                dvs=dvs,
                explanation_prefix=explanation_prefix,
                cognitive_param_prefix=cognitive_param_prefix,
            ))

        for _, trial_row in testing_rows.iterrows():
            trial_info = trial_row.to_dict()
            trial_context = build_single_trial_cognitive_input(
                trial_info,
                raw_dataset,
                explanation_pool,
                label_column=label_column,
            )
            model_outputs = participant_model(cognitive_params, dvs, trial_context)

            explanation_cols = {
                f"{explanation_prefix}{key}": value
                for key, value in trial_context["instance_explanation"].items()
            }
            cognitive_param_cols = {
                f"{cognitive_param_prefix}{key}": value
                for key, value in cognitive_params.items()
            }

            ai_prediction = model_outputs.get("ai_prediction")
            agent_prediction = model_outputs.get("agent_prediction")
            cognitive_correct_vs_ai = (
                None
                if ai_prediction is None or agent_prediction is None
                else bool(int(agent_prediction) == int(ai_prediction))
            )
            for dv_name in dvs:
                if "accuracy" in dv_name.lower() and cognitive_correct_vs_ai is not None:
                    model_outputs[dv_name] = int(cognitive_correct_vs_ai)

            executed_rows.append({
                **trial_info,
                **explanation_cols,
                **cognitive_param_cols,
                **model_outputs,
                "cognitive_correct_vs_ai": cognitive_correct_vs_ai,
            })

    results = pd.DataFrame(executed_rows)
    if "__executor_order" in results:
        results = (
            results.sort_values("__executor_order", kind="stable")
            .drop(columns="__executor_order")
            .reset_index(drop=True)
        )
    return results


def _attribution_values(trial_context: dict[str, Any]) -> dict[str, float]:
    """Return only attribution-vector values from one cognitive input."""
    return {
        key: value
        for key, value in (
            trial_context.get("instance_explanation", {}) or {}
        ).items()
        if key.startswith("a") and key.endswith("_i")
    }


def _trial_expects_xai(
    trial_info: dict[str, Any],
    *,
    training: bool = False,
) -> bool:
    method = str(
        trial_info.get("xai_method", trial_info.get("xai_type", "none"))
    ).lower()
    if method in {"none", "no_xai", "control"}:
        return False
    if training or str(trial_info.get("phase", "")).lower() == "training":
        return True
    tested_w_xai = trial_info.get("tested_w_xai", True)
    if isinstance(tested_w_xai, str):
        return tested_w_xai.strip().lower() in {"true", "1", "yes", "y"}
    return bool(tested_w_xai)


def _training_result_row(
    trial_context: dict[str, Any],
    *,
    cognitive_params: dict[str, float],
    dvs: dict[str, list[Any]],
    explanation_prefix: str,
    cognitive_param_prefix: str,
) -> dict[str, Any]:
    """Record a shown training exemplar without treating it as an inference."""
    return {
        **trial_context["trial_info"],
        **{
            f"{explanation_prefix}{key}": value
            for key, value in trial_context["instance_explanation"].items()
        },
        **{
            f"{cognitive_param_prefix}{key}": value
            for key, value in cognitive_params.items()
        },
        **{dv_name: None for dv_name in dvs},
        "agent_prediction": None,
        "ai_prediction": trial_context.get("ai_prediction"),
        "prob_correct": None,
        "pred_time": 0.0,
        "cognitive_correct_vs_ai": None,
    }


def save_simulated_results(
    simulated_results: pd.DataFrame,
    *,
    out_dir: str | Path = "experiment_output",
) -> tuple[str, str]:
    """Save simulated experiment results as CSV and JSON."""
    os.makedirs(out_dir, exist_ok=True)
    csv_path = str(Path(out_dir) / "simulated_results.csv")
    json_path = str(Path(out_dir) / "simulated_results.json")

    simulated_results.to_csv(csv_path, index=False)
    simulated_results.to_json(json_path, orient="records", indent=2)
    return csv_path, json_path


@dataclass
class VirtualExperimentResult:
    """Executed participant-condition responses and their saved files."""

    responses: pd.DataFrame
    csv_path: str
    json_path: str


def run_virtual_experiment(
    participant_arrangement: Any,
    cognitive_params: dict[str, float],
    dvs: dict[str, list[Any]],
    raw_dataset: pd.DataFrame,
    explanation_pool: pd.DataFrame,
    *,
    cognitive_model: Any,
    label_column: str,
    condition_columns: Optional[list[str]] = None,
    output_dir: str | Path = "experiment_output",
    mode: str = "whole_experiment",
    participant_id: Optional[int] = None,
    condition_filter: Optional[dict[str, Any]] = None,
) -> VirtualExperimentResult:
    """Run and save a baseline separately for every participant-condition."""
    trials = getattr(participant_arrangement, "trials", participant_arrangement)
    responses = run_experiment_executor(
        trials=trials,
        cognitive_params=cognitive_params,
        dvs=dvs,
        raw_dataset=raw_dataset,
        explanation_pool=explanation_pool,
        mode=mode,
        participant_id=participant_id,
        condition_filter=condition_filter,
        condition_columns=condition_columns,
        cognitive_model=cognitive_model,
        label_column=label_column,
    )
    csv_path, json_path = save_simulated_results(responses, out_dir=output_dir)
    return VirtualExperimentResult(
        responses=responses,
        csv_path=csv_path,
        json_path=json_path,
    )

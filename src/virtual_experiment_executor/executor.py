"""Run cognitive agents over generated experiment trials."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from src.cognitive_models import build_single_trial_cognitive_input, dummy_cognitive_model
from src.experiment_design import select_trial_rows


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

    executed_rows = []
    for _, trial_row in selected.iterrows():
        trial_info = trial_row.to_dict()
        trial_context = build_single_trial_cognitive_input(
            trial_info,
            raw_dataset,
            explanation_pool,
            label_column=label_column,
        )
        model_outputs = cognitive_model(cognitive_params, dvs, trial_context)

        explanation_cols = {
            f"{explanation_prefix}{k}": v
            for k, v in trial_context["instance_explanation"].items()
        }
        cognitive_param_cols = {
            f"{cognitive_param_prefix}{k}": v
            for k, v in cognitive_params.items()
        }

        ai_prediction = model_outputs.get("ai_prediction")
        agent_prediction = model_outputs.get("agent_prediction")
        cognitive_correct_vs_ai = (
            None if ai_prediction is None else bool(int(agent_prediction) == int(ai_prediction))
        )

        executed_rows.append({
            **trial_info,
            **explanation_cols,
            **cognitive_param_cols,
            **model_outputs,
            "cognitive_correct_vs_ai": cognitive_correct_vs_ai,
        })

    return pd.DataFrame(executed_rows)


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

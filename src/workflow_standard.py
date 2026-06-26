"""Shared workflow column names and prediction/explanation alignment helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


DATA_ID_COL = "dataId"
INSTANCE_ID_COL = "instanceId"
MODEL_NAME_COL = "modelName"
EXPLANATION_METHOD_COL = "expMethod"
PREDICTION_COL = "pred"
PREDICTION_ONLY_METHOD = "__prediction_only__"
DEFAULT_EXPLANATION_INSTANCE_LIMIT = 300


def prediction_labels(raw_predictions: Any) -> np.ndarray:
    """Convert model labels/probabilities to one integer label per row."""
    predictions = np.asarray(raw_predictions)
    if predictions.ndim > 1:
        return np.argmax(predictions, axis=1).astype(int)
    return predictions.reshape(-1).astype(int)


def ensure_prediction_coverage(
    explanation_pool: pd.DataFrame,
    *,
    trials: list[dict[str, Any]] | pd.DataFrame,
    data: Any,
    trained_engine: Any,
    model_name: str = "model",
    show: bool = True,
) -> pd.DataFrame:
    """
    Ensure every trial instance has an AI prediction available by instanceId.

    Explanations may intentionally cover only a subset of test instances. This
    helper keeps execution standardized by adding prediction-only rows for trial
    instances missing from the explanation table.
    """
    aligned_pool = explanation_pool.copy()
    if PREDICTION_COL not in aligned_pool.columns:
        aligned_pool[PREDICTION_COL] = np.nan

    trials_df = pd.DataFrame(trials)
    if INSTANCE_ID_COL not in trials_df:
        return aligned_pool

    trial_instance_ids = {
        int(instance_id)
        for instance_id in trials_df[INSTANCE_ID_COL].dropna().astype(int).tolist()
    }
    covered_instance_ids = set()
    if INSTANCE_ID_COL in aligned_pool.columns:
        covered_instance_ids = {
            int(instance_id)
            for instance_id in aligned_pool[INSTANCE_ID_COL].dropna().astype(int).tolist()
        }

    missing_instance_ids = sorted(trial_instance_ids - covered_instance_ids)
    if not missing_instance_ids:
        return aligned_pool

    test_id_to_position = {
        int(instance_id): position
        for position, instance_id in enumerate(np.asarray(data.test_instance_ids))
    }
    prediction_rows = []
    for instance_id in missing_instance_ids:
        position = test_id_to_position.get(instance_id)
        if position is None:
            continue
        pred = prediction_labels(trained_engine.predict(data.X_test[position:position + 1]))[0]
        prediction_rows.append({
            DATA_ID_COL: data.dataset_id,
            MODEL_NAME_COL: model_name,
            EXPLANATION_METHOD_COL: PREDICTION_ONLY_METHOD,
            INSTANCE_ID_COL: instance_id,
            PREDICTION_COL: int(pred),
        })

    if not prediction_rows:
        return aligned_pool

    if show:
        print(
            "Added AI predictions for "
            f"{len(prediction_rows)} trial instance(s) that did not have generated explanations."
        )
    return pd.concat([aligned_pool, pd.DataFrame(prediction_rows)], ignore_index=True)

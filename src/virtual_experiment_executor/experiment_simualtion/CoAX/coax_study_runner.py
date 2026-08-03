"""Run a xaikitTest study's generated trials through the CoAX cognitive models.

This is section 5 of ``feature_explanation_user_study_from_design_export_coax``
as a callable. The notebook and the API server both drive it, so the "one
repository per XAI method" procedure exists once.

Why the per-method loop: a design can carry two XAI factors. ``xai_type``
decides *how* an explanation is displayed and is what CoAX dispatches on, while
``xai_method`` decides *which* vector is displayed. ``CoAXAssetRepository`` keys
explanations by ``xai_type``, so the executor runs once per method against a
repository built from that method's table -- attribution is the signed vector,
importance its absolute value.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import pandas as pd

from src.experiment_planner import select_trial_rows

from .coax_trial_executor import (
    COAX_STRATEGIES_BY_XAI_TYPE,
    CoAXAssetRepository,
    default_coax_strategy,
    make_coax_model,
    make_coax_models,
    run_coax_experiment_executor,
    _canonical_tested_condition,
    _canonical_xai_type,
    _normalize_model_key,
)


#: ``expMethod`` marker of the shared AI-prediction rows in a combined table.
PREDICTION_ONLY_METHOD = "__prediction_only__"

#: Fitted-population parameter file behind :data:`FITTED_COAX_PARAMS`.
FITTED_PARAMETER_FILE = (
    Path(__file__).resolve().parents[4]
    / "src" / "cognitive_models" / "CoAX" / "results" / "pop_em_subset"
    / "refit_all_assignments_detlocal-Jan-10.csv"
)

#: Per-strategy means of the parameters fitted to the original CoAX study
#: participants, computed from :data:`FITTED_PARAMETER_FILE` (one row per
#: participant x session x tested-condition assignment).
#:
#: These replace the reference CLI's round numbers as the runner's defaults, so
#: an unconfigured run starts from the fitted population rather than from
#: arbitrary values. ``k`` indexes a top-k slice, so its mean is rounded to an
#: integer. ``decay_param`` is not fitted in that file and keeps the reference
#: value. Note that several means sit outside ``COAX_PARAM_BOUNDS`` -- those are
#: the simulator's slider limits, not bounds on the fitted population.
FITTED_COAX_PARAMS: dict[str, dict[str, Any]] = {
    # n = 398 assignments, mean k 2.7161
    "SensitiveFeatures": {
        "decay_param": 0.5,
        "sensitivity": 21.7171,
        "k": 3,
        "retrieval_threshold": -2.4247,
    },
    # n = 210 assignments, mean k 1.8381
    "SalientFeatures": {
        "decay_param": 0.5,
        "sensitivity": 19.6077,
        "k": 2,
        "retrieval_threshold": -2.9775,
    },
    # n = 14 assignments only -- the thinnest cell in the fitted table, so this
    # mean is the least stable of the four.
    "ImportanceCategorization": {
        "decay_param": 0.5,
        "sensitivity": 39.9953,
        "k": 4,
        "retrieval_threshold": -3.5771,
    },
    # n = 511 assignments, mean k 4.2035. AttributionSum is fitted with
    # scaling_factor instead of sensitivity, so sensitivity keeps its class
    # default.
    "AttributionSum": {
        "decay_param": 0.5,
        "scaling_factor": 3.4464,
        "k": 4,
        "retrieval_threshold": -2.6399,
    },
}

#: Carries study trial order through the per-method runs so the concatenated
#: result reads in experiment order rather than grouped by XAI method.
ORDER_COLUMN = "__coax_trial_order"


def coax_feature_table(data: Any) -> pd.DataFrame:
    """Build the CoAX-schema feature table from ``study.data``.

    The repository expects ``dataId``, ``instanceId`` and ``x0..xn`` columns;
    a ``PreparedDataset`` holds the same values as the model matrix plus the
    raw instance ids it was built from.
    """
    split = data.split
    features = pd.DataFrame(
        split.X_model,
        columns=[f"x{index}" for index in range(split.X_model.shape[1])],
    )
    features.insert(0, "instanceId", [int(value) for value in split.raw_instance_ids])
    features.insert(0, "dataId", data.dataset_id)
    return features


def explanation_value_columns(explanations: pd.DataFrame) -> list[str]:
    """Attribution columns of a CoAX-schema explanation table (``a0_i``, ...)."""
    return [
        column
        for column in explanations.columns
        if column.startswith("a") and column.endswith("_i")
    ]


def build_coax_repository(
    features: pd.DataFrame,
    predictions: pd.DataFrame,
    method_explanations: Optional[pd.DataFrame] = None,
) -> CoAXAssetRepository:
    """Repository for one XAI method: signed vectors and their absolute value.

    ``method_explanations`` is one method's slice of the combined explanation
    table. Pass ``None`` (or an empty frame) for a run whose trials only use
    ``xai_type='none'``, where no explanation is ever displayed.
    """
    tables: dict[str, pd.DataFrame] = {}
    if method_explanations is not None and not method_explanations.empty:
        attribution = method_explanations.copy()
        importance = attribution.copy()
        value_columns = explanation_value_columns(attribution)
        importance[value_columns] = importance[value_columns].abs()
        tables = {"attribution": attribution, "importance": importance}
    return CoAXAssetRepository.from_tables(
        features=features,
        predictions=predictions,
        explanations=tables,
    )


def fitted_coax_params(
    strategy_name: str,
    xai_type: Any,
    **overrides: Any,
) -> dict[str, Any]:
    """Constructor kwargs for one strategy, from the fitted population means.

    Falls back to the class defaults for anything the fitted table does not
    carry, and keeps ``explanation_type`` wired to the condition the way
    ``default_coax_params`` does.
    """
    params = dict(FITTED_COAX_PARAMS.get(strategy_name, {}))
    if strategy_name == "AttributionSum":
        params["explanation_type"] = _canonical_xai_type(xai_type)
    params.update(overrides)
    return params


def coax_models_for_trials(
    trials: pd.DataFrame,
    *,
    strategies: Optional[Mapping[Any, str]] = None,
    params: Optional[Mapping[Any, Mapping[str, Any]]] = None,
    by_tested_condition: bool = False,
    use_fitted_params: bool = True,
) -> dict[Any, Any]:
    """One CoAX strategy per condition, checked against the conditions in ``trials``.

    With ``use_fitted_params`` (the default) each strategy starts from the
    fitted-population means in :data:`FITTED_COAX_PARAMS`; set it to ``False``
    for the reference CLI's round defaults via ``make_coax_models``. ``params``
    overrides either, keyed the same way as the returned mapping.

    Coverage is verified before the run rather than at the first uncovered
    trial, so a design level with no registered strategy fails immediately.
    """
    if not use_fitted_params:
        models = make_coax_models(
            strategies=strategies,
            params=params,
            by_tested_condition=by_tested_condition,
        )
        _assert_models_cover(trials, models)
        return models

    strategy_overrides = {
        _normalize_model_key(key): name for key, name in (strategies or {}).items()
    }
    param_overrides = {
        _normalize_model_key(key): dict(value) for key, value in (params or {}).items()
    }

    keys: list[Any] = (
        [
            (xai_type, tested)
            for xai_type in COAX_STRATEGIES_BY_XAI_TYPE
            for tested in (False, True)
        ]
        if by_tested_condition
        else list(COAX_STRATEGIES_BY_XAI_TYPE)
    )

    models: dict[Any, Any] = {}
    for key in keys:
        xai_type, tested = key if isinstance(key, tuple) else (key, None)
        strategy_name = strategy_overrides.get(key) or default_coax_strategy(xai_type, tested)
        models[key] = make_coax_model(
            strategy_name,
            **fitted_coax_params(strategy_name, xai_type, **param_overrides.get(key, {})),
        )
    _assert_models_cover(trials, models)
    return models


def run_coax_study(
    study: Any,
    *,
    mode: str = "whole_experiment",
    participant_id: Optional[int] = None,
    condition_filter: Optional[dict[str, Any]] = None,
    cognitive_model: Any = None,
    trials: Optional[pd.DataFrame | Sequence[dict[str, Any]]] = None,
    explanation_pool: Optional[pd.DataFrame] = None,
    data: Any = None,
    dvs: Optional[Mapping[str, Any]] = None,
    train_with_explanation: bool = True,
    store: bool = True,
) -> pd.DataFrame:
    """Execute a study's trials with CoAX participants and return the step rows.

    Everything the run needs is read off the study object -- generated trials,
    the combined explanation table, the prepared dataset and the DVs -- so the
    call sits directly after ``study.explanations(...)``. Each argument can be
    overridden to run a subset without touching the stored study state.

    ``mode`` takes the same values as ``study.run_experiment``:
    ``trial_by_trial``, ``participant_by_participant``, ``whole_condition``
    (with ``condition_filter``) or ``whole_experiment``. Selection is applied
    once, before trials are grouped by XAI method, so ``trial_by_trial`` yields
    one trial and not one trial per method.

    With ``store=True`` the result is written back to
    ``study.simulated_results``, which is what ``save_results``,
    ``analyze_iv_dv`` and the plotting helpers read.
    """
    trials_df = pd.DataFrame(study.trials if trials is None else trials).copy()
    if trials_df.empty:
        raise RuntimeError(
            "No generated trials to run. Call study.generate_trials(...) first."
        )

    pool = explanation_pool if explanation_pool is not None else study.combined_explanations
    if pool is None:
        raise RuntimeError(
            "No explanation table available. Call study.explanations(...) first."
        )

    dataset = data if data is not None else study.data
    if dataset is None:
        raise RuntimeError(
            "No prepared dataset available. Call study.prepare_dataset(...) first."
        )

    selected = select_trial_rows(
        trials_df,
        mode,
        participant_id=participant_id,
        condition_filter=condition_filter,
    ).copy()
    if selected.empty:
        return selected
    selected[ORDER_COLUMN] = range(len(selected))

    models = (
        coax_models_for_trials(selected)
        if cognitive_model is None
        else cognitive_model
    )
    if isinstance(models, Mapping):
        _assert_models_cover(selected, models)

    features = coax_feature_table(dataset)
    predictions = pool[pool["expMethod"].astype(str) == PREDICTION_ONLY_METHOD]
    if predictions.empty:
        raise ValueError(
            "The explanation table carries no AI-prediction rows "
            f"(expMethod == {PREDICTION_ONLY_METHOD!r}), so CoAX has nothing to "
            "be scored against."
        )

    runs = []
    for method, method_trials in _group_by_xai_method(selected, pool):
        method_explanations = pool[pool["expMethod"].astype(str) == str(method)]
        if method_explanations.empty and _needs_explanations(method_trials):
            raise ValueError(
                f"No explanation rows for xai_method={method!r}, but its trials "
                "include an xai_type that displays one. Generate explanations for "
                "every method the design uses."
            )
        repository = build_coax_repository(features, predictions, method_explanations)
        runs.append(
            run_coax_experiment_executor(
                method_trials,
                models,
                # Selection already happened above, on the whole trial table.
                mode="whole_experiment",
                data_repository=repository,
                train_with_explanation=train_with_explanation,
                dvs=study.DVs if dvs is None else dvs,
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


def _group_by_xai_method(
    trials: pd.DataFrame,
    explanation_pool: pd.DataFrame,
) -> list[tuple[Any, pd.DataFrame]]:
    """Split trials by the XAI method whose vectors they display."""
    generated = sorted(
        method
        for method in explanation_pool["expMethod"].astype(str).unique()
        if method != PREDICTION_ONLY_METHOD
    )
    if "xai_method" in trials.columns:
        return [
            (method, rows)
            for method, rows in trials.groupby(
                trials["xai_method"].astype(str), sort=False
            )
        ]
    if len(generated) != 1:
        raise ValueError(
            "Trials carry no 'xai_method' column, so the method cannot be "
            f"resolved per trial, and the explanation table holds {len(generated)} "
            f"methods: {generated}. Generate explanations for exactly one method, "
            "or add xai_method to the design."
        )
    return [(generated[0], trials)]


def _needs_explanations(trials: pd.DataFrame) -> bool:
    """Whether any of these trials would display an explanation vector."""
    if "xai_type" not in trials.columns:
        return False
    return any(
        _canonical_xai_type(value) != "none"
        for value in trials["xai_type"].dropna().unique()
    )


def _assert_models_cover(trials: pd.DataFrame, models: Mapping[Any, Any]) -> None:
    """Fail before the run if a trial condition has no registered strategy."""
    if "xai_type" not in trials.columns:
        return
    condition_columns = [
        column for column in ("xai_type", "tested_w_xai") if column in trials.columns
    ]
    conditions = trials[condition_columns].drop_duplicates()

    registered = set(models)
    uncovered: set[str] = set()
    for _, row in conditions.iterrows():
        xai_type = _canonical_xai_type(row["xai_type"])
        tested = _canonical_tested_condition(row.get("tested_w_xai", False))
        if (xai_type, tested) not in registered and xai_type not in registered:
            uncovered.add(xai_type)
    if uncovered:
        raise ValueError(
            f"No CoAX model registered for xai_type(s): {sorted(uncovered)}. "
            f"Registered keys: {sorted(registered, key=repr)}."
        )


__all__ = [
    "FITTED_COAX_PARAMS",
    "FITTED_PARAMETER_FILE",
    "ORDER_COLUMN",
    "PREDICTION_ONLY_METHOD",
    "build_coax_repository",
    "coax_feature_table",
    "coax_models_for_trials",
    "explanation_value_columns",
    "fitted_coax_params",
    "run_coax_study",
]

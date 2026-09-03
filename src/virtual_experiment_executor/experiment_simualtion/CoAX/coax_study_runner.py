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

import warnings
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import pandas as pd

from src.experiment_planner import select_trial_rows
from src.virtual_experiment_executor.participant_pools import (
    draw_participant_parameters,
    draw_participant_units,
    draws_to_frame,
    is_diverse_mode,
    is_fitted_population_mode,
    samples_participants,
)

from .coax_trial_executor import (
    COAX_STRATEGIES_BY_XAI_TYPE,
    CoAXAssetRepository,
    coax_available_strategies,
    coax_params_for_strategy,
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
    # n = 398 assignments, mean k 2.7161. The raw per-participant fits are
    # bimodal (41% pinned at the sensitivity floor of 1.0, the rest spread
    # 19-100) rather than clustered near the mean -- the population mean
    # (21.7171) made simulated "none"-condition forward_accuracy run well
    # above real humans on the same instances (0.886 vs 0.671). sensitivity
    # is not linear here: 9 barely moved it (0.894), 1 undershot (0.588), 4.5
    # overshot (0.832), 2.2 landed at 0.732 (+0.061). 1.9 (interpolated one
    # step further toward the 1/2.2 data points) is the closest single-value
    # match found so far -- still an approximation of a population that is
    # not well summarized by any one value. The more correct fix is
    # per-participant sampling from this real distribution instead of one
    # shared constant.
    "SensitiveFeatures": {
        "decay_param": 0.5,
        "sensitivity": 1.9,
        "k": 3,
        "retrieval_threshold": -2.4247,
    },
    # n = 210 assignments, mean k 1.8381. Same bimodal issue as
    # SensitiveFeatures, more pronounced: 62% of real participants (130/210)
    # are pinned at the sensitivity floor of 1.0. The population mean
    # (19.6077) made simulated "importance"-condition accuracy run at 0.916
    # against a real 0.735 on the same instances. 2.2 (reusing the value
    # that closed the equivalent gap for SensitiveFeatures) landed almost
    # exactly on target: 0.742 vs 0.735 (+0.007).
    "SalientFeatures": {
        "decay_param": 0.5,
        "sensitivity": 2.2,
        "k": 2,
        "retrieval_threshold": -2.9775,
    },
    # n = 14 assignments only -- the thinnest cell in the fitted table, so this
    # mean is the least stable of the four. No longer the default strategy for
    # the "importance" condition (see PREFERRED_COAX_STRATEGY_BY_XAI_TYPE --
    # SalientFeatures fits that condition much better), but still a legal
    # fallback, so its parameters stay the real fitted values.
    "ImportanceCategorization": {
        "decay_param": 0.5,
        "sensitivity": 39.9953,
        "k": 4,
        "retrieval_threshold": -3.5771,
    },
    # n = 511 assignments, mean k 4.2035. AttributionSum is fitted with
    # scaling_factor instead of sensitivity, so sensitivity keeps its class
    # default. scaling_factor turned out not to be the lever here (1.5 and
    # 0.1, the real floor, both left accuracy unchanged at 0.978) -- k was:
    # k=5 -> 0.984, k=4/rt=-4.0 -> 0.98, k=4/rt=-1.64 -> 0.838, k=2 -> 0.89.
    # k=2 (itself the real distribution's floor value, 22/511 = 4.3% of
    # participants -- the same "minority floor behavior matches the
    # population average" pattern seen in SensitiveFeatures/SalientFeatures)
    # landed almost exactly on the real "attribution"-condition accuracy on
    # the same instances: 0.89 vs 0.883 (+0.007).
    "AttributionSum": {
        "decay_param": 0.5,
        "scaling_factor": 3.4464,
        "k": 2,
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
    params.update(coax_params_for_strategy(strategy_name, overrides))
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
    source: str = "study",
    data_id: Optional[str] = None,
    dvs: Optional[Mapping[str, Any]] = None,
    train_with_explanation: bool = True,
    sampling_seed: int = 0,
    sampling_replace: Optional[bool] = None,
    parameter_pool: Optional[pd.DataFrame] = None,
    store: bool = True,
) -> pd.DataFrame:
    """Execute a study's trials with CoAX participants and return the step rows.

    Everything the run needs is read off the study object -- generated trials,
    the combined explanation table, the prepared dataset and the DVs -- so the
    call sits directly after ``study.explanations(...)``. Each argument can be
    overridden to run a subset without touching the stored study state.

    ``mode`` takes the same values as ``study.run_experiment``:
    ``trial_by_trial``, ``participant_by_participant``, ``whole_condition``
    (with ``condition_filter``), ``whole_experiment`` or
    ``diverse_participant``. Selection is applied once, before trials are
    grouped by XAI method, so ``trial_by_trial`` yields one trial and not one
    trial per method.

    ``diverse_participant`` runs the same trials as ``whole_experiment`` but
    deals every participant its own parameters, drawn per condition cell from
    the 330 participants in
    ``assets/human_data/CoAX/coax_fitted_strategies.csv`` rather than running
    :data:`FITTED_COAX_PARAMS`'s population means for everybody. That table's own
    docstring explains why this is the better default: the fitted distributions
    are bimodal (41% of SensitiveFeatures participants sit at the sensitivity
    floor of 1.0, the rest spread 19-100), so no single value describes the
    population, and the means had to be hand-corrected against real accuracy to
    be usable at all.

    ``source`` picks where the instances, AI predictions and explanation vectors
    come from:

    * ``"study"`` (the default) uses this study's own prepared dataset and the
      table ``study.explanations(...)`` produced -- so it needs a trained AI
      model and a completed explanation stage.
    * ``"corpus"`` uses the published CoAX user-study corpus instead, needing
      neither. ``data_id`` names the dataset (``adult``, ``forest_cover``,
      ``mushrooms`` or ``wine_quality``); it falls back to the trials' own
      ``dataId``/``dataset`` column and then to ``study.data.dataset_id``.
      Pair it with ``coax_corpus_instance_ids(data_id)`` when generating trials
      so every trial references an instance the corpus carries. This is the
      mode that makes a simulation directly comparable with
      ``run_coax_human_replay``: both then serve the exact vectors the human
      participants were shown.

    With ``store=True`` the result is written back to
    ``study.simulated_results``, which is what ``save_results``,
    ``analyze_iv_dv`` and the plotting helpers read.
    """
    if source not in {"study", "corpus"}:
        raise ValueError(f"source must be 'study' or 'corpus', not {source!r}.")
    trials_df = pd.DataFrame(study.trials if trials is None else trials).copy()
    if trials_df.empty:
        raise RuntimeError(
            "No generated trials to run. Call study.generate_trials(...) first."
        )

    # The corpus already carries instances, predictions and explanation vectors,
    # so neither the explanation stage nor a prepared dataset is required.
    pool = explanation_pool if explanation_pool is not None else getattr(study, "combined_explanations", None)
    if pool is None and source == "study":
        raise RuntimeError(
            "No explanation table available. Call study.explanations(...) first, "
            "or pass source='corpus' to run against the published CoAX corpus."
        )

    dataset = data if data is not None else getattr(study, "data", None)
    if dataset is None and source == "study":
        raise RuntimeError(
            "No prepared dataset available. Call study.prepare_dataset(...) first, "
            "or pass source='corpus' to run against the published CoAX corpus."
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
        # fitted_population draws a strategy per tested half, so it needs one
        # model per half to put them in; every other mode keeps the single
        # model per xai_type it has always built.
        coax_models_for_trials(
            selected, by_tested_condition=is_fitted_population_mode(mode)
        )
        if cognitive_model is None
        else cognitive_model
    )
    if isinstance(models, Mapping):
        _assert_models_cover(selected, models)

    participant_models = participant_draws = None
    if samples_participants(mode):
        if not isinstance(models, Mapping):
            raise ValueError(
                f"mode={mode!r} needs the per-condition model mapping "
                "coax_models_for_trials() builds, so each condition's strategy is "
                "known before its parameters are drawn. Pass cognitive_model as a "
                "{xai_type: model} mapping, or leave it unset."
            )
        draw_models = (
            coax_population_models
            if is_fitted_population_mode(mode)
            else coax_participant_models
        )
        participant_models, participant_draws = draw_models(
            selected,
            models,
            data_id=_pool_data_id(data_id, selected, dataset),
            seed=sampling_seed,
            replace=sampling_replace,
            pool=parameter_pool,
        )
        if store:
            study.participant_parameters = draws_to_frame(participant_draws)

        if is_fitted_population_mode(mode):
            # The trials follow the draw: each virtual participant runs the
            # instances its own fitted human saw, which is the sequence those
            # parameters were fitted against. See coax_population_trials.
            selected = coax_population_trials(participant_draws, selected)
            selected[ORDER_COLUMN] = range(len(selected))
            # Those instance ids index the published corpus, not a freshly
            # trained model's predictions, so the run has to read the corpus.
            source = "corpus"
            data_id = _pool_data_id(data_id, selected, dataset)

    if source == "corpus":
        return _run_coax_study_from_corpus(
            study,
            selected,
            models,
            data_id=data_id,
            dataset=dataset,
            dvs=dvs,
            train_with_explanation=train_with_explanation,
            participant_models=participant_models,
            participant_draws=participant_draws,
            share_training_memory=is_fitted_population_mode(mode),
            store=store,
        )

    share_training_memory = is_fitted_population_mode(mode)
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
                participant_models=participant_models,
                participant_draws=participant_draws,
                share_training_memory=share_training_memory,
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


def _pool_data_id(
    data_id: Optional[str], trials: pd.DataFrame, dataset: Any
) -> Optional[str]:
    """Which dataset the fitted pool should be filtered to, or None if unclear.

    The soft counterpart of :func:`_resolve_corpus_data_id`: a corpus run must
    know the dataset because an instance id means different things in each, but
    a *parameter* draw can simply widen across datasets rather than fail.
    """
    if data_id is not None:
        return str(data_id)
    for column in ("dataId", "dataset", "dataset_id", "appId"):
        if column in trials.columns:
            names = {str(value) for value in trials[column].dropna().unique()}
            if len(names) == 1:
                return next(iter(names))
    dataset_id = getattr(dataset, "dataset_id", None)
    return None if dataset_id is None else str(dataset_id)


def coax_participant_models(
    trials: pd.DataFrame,
    models: Mapping[Any, Any],
    *,
    data_id: Optional[str] = None,
    seed: int = 0,
    replace: Optional[bool] = None,
    pool: Optional[pd.DataFrame] = None,
) -> tuple[Any, dict[Any, Any]]:
    """One CoAX model per participant per condition, from the fitted pool.

    Returns ``(participant_models, draws)``: a callable the executor calls once
    per participant to get that participant's condition -> model mapping, and
    the draws behind it keyed by ``(participantId, model_key)`` for provenance.

    The strategy is *not* drawn -- it is read off the model the runner already
    built for that condition (``coax_models_for_trials`` picks it from
    ``PREFERRED_COAX_STRATEGY_BY_XAI_TYPE``, and an explicit mapping overrides
    that). Only the parameters vary per participant, and they are filtered to
    rows fitted for that same strategy, so an AttributionSum participant is
    never handed a SensitiveFeatures participant's ``sensitivity``.

    A model key that carries ``tested_w_xai`` narrows the draw to that tested
    condition; a plain ``xai_type`` key draws across both, since one model then
    serves both halves of that participant's session.
    """
    condition_columns = [
        column for column in ("xai_type", "tested_w_xai") if column in trials.columns
    ]
    if "participantId" not in trials.columns or not condition_columns:
        return None, {}

    draws: dict[Any, Any] = {}
    parameters_by_participant: dict[Any, dict[Any, dict[str, Any]]] = {}
    strategy_by_key: dict[Any, str] = {}

    cells = trials[["participantId", *condition_columns]].drop_duplicates()
    cells = cells.assign(
        __xai_type=[_canonical_xai_type(value) for value in cells["xai_type"]]
        if "xai_type" in cells.columns
        else "none",
        __tested=[
            _canonical_tested_condition(value)
            for value in cells.get("tested_w_xai", pd.Series([False] * len(cells)))
        ],
    )
    cells["__key"] = [
        _select_pool_model_key(models, row["__xai_type"], row["__tested"])
        for _, row in cells.iterrows()
    ]

    xai_method = None
    if "xai_method" in trials.columns:
        methods = {str(value) for value in trials["xai_method"].dropna().unique()}
        xai_method = next(iter(methods)) if len(methods) == 1 else None

    for offset, (key, rows) in enumerate(cells.groupby("__key", sort=False)):
        strategy = type(models[key]).__name__
        strategy_by_key[key] = strategy
        condition: dict[str, Any] = {"strategy": strategy}
        if data_id is not None:
            condition["dataId"] = data_id
        if xai_method is not None:
            condition["xai_method"] = xai_method
        condition["xai_type"] = rows["__xai_type"].iloc[0]
        if isinstance(key, tuple):
            # The key distinguishes tested conditions, so the draw should too.
            condition["tested_w_xai"] = rows["__tested"].iloc[0]

        participants = sorted(rows["participantId"].dropna().unique().tolist(), key=str)
        cell_draws = draw_participant_parameters(
            "coax",
            condition=condition,
            participants=participants,
            seed=int(seed) + offset,
            replace=replace,
            # xai_method is nearly redundant with dataId in the CoAX corpus (one
            # method per dataset), tested_w_xai halves an already narrow cell,
            # and dropping dataId borrows another dataset's fits -- which is
            # still a real human, and the only option for mushrooms, which was
            # never fitted.
            relax=("xai_method", "tested_w_xai", "dataId"),
            pool=pool,
        )
        for participant, draw in cell_draws.items():
            draws[(participant, key)] = draw
            parameters_by_participant.setdefault(participant, {})[key] = draw.parameters

    def participant_models(participant_id: Any) -> dict[Any, Any]:
        """This participant's models, falling back to the shared one per key."""
        drawn = parameters_by_participant.get(participant_id, {})
        built = dict(models)
        for key, parameters in drawn.items():
            if not parameters:
                continue
            xai_type = key[0] if isinstance(key, tuple) else key
            built[key] = make_coax_model(
                strategy_by_key[key],
                **fitted_coax_params(strategy_by_key[key], xai_type, **parameters),
            )
        return built

    return participant_models, draws


def coax_population_models(
    trials: pd.DataFrame,
    models: Mapping[Any, Any],
    *,
    data_id: Optional[str] = None,
    seed: int = 0,
    replace: Optional[bool] = None,
    pool: Optional[pd.DataFrame] = None,
) -> tuple[Any, dict[Any, Any]]:
    """One CoAX model per participant per tested half, drawn as whole people.

    The counterpart of :func:`coax_participant_models`, and the difference is the
    strategy. That function reads the strategy off the model the runner already
    built for the condition and draws only parameters *fitted to that same
    strategy*; every participant in a condition therefore runs the one strategy
    :data:`PREFERRED_COAX_STRATEGY_BY_XAI_TYPE` names. The fitted population is
    not like that. Its ``importance`` cells split four ways (AttributionSum 240
    rows, SalientFeatures 210, SensitiveFeatures 192, ImportanceCategorization
    14) and its ``attribution`` w/o XAI cell is majority SensitiveFeatures -- so
    running one strategy reproduces neither the mixture nor the accuracy it
    averages to.

    Here the *person* is drawn: one ``(Participant ID, Session)`` unit per
    virtual participant, contributing that human's own strategy and parameters to
    each tested half they were fitted in. A participant's two halves are then the
    same human, which is what makes the within-subject Attribution effect mean
    anything.

    Requires ``models`` to be keyed by ``(xai_type, tested_w_xai)``: with one
    model per ``xai_type`` the two halves share an instance, and only one of the
    person's two strategies could be applied.

    Returns ``(participant_models, draws)`` with the same contract
    :func:`coax_participant_models` returns, so the executor needs no new wiring.
    """
    if "participantId" not in trials.columns or "xai_type" not in trials.columns:
        return None, {}

    cells = trials[["participantId", "xai_type"]].drop_duplicates()
    cells = cells.assign(
        __xai_type=[_canonical_xai_type(value) for value in cells["xai_type"]]
    )

    xai_method = None
    if "xai_method" in trials.columns:
        methods = {str(value) for value in trials["xai_method"].dropna().unique()}
        xai_method = next(iter(methods)) if len(methods) == 1 else None

    draws: dict[Any, Any] = {}
    drawn_by_participant: dict[Any, dict[Any, Any]] = {}

    for offset, (xai_type, rows) in enumerate(cells.groupby("__xai_type", sort=False)):
        _assert_tuple_keyed(models, xai_type)

        condition: dict[str, Any] = {"xai_type": xai_type}
        if data_id is not None:
            condition["dataId"] = data_id
        if xai_method is not None:
            condition["xai_method"] = xai_method

        participants = sorted(rows["participantId"].dropna().unique().tolist(), key=str)
        units = draw_participant_units(
            "coax",
            condition=condition,
            participants=participants,
            seed=int(seed) + offset,
            replace=replace,
            # Same reasoning as coax_participant_models: xai_method is nearly
            # redundant with dataId in this corpus, and borrowing another
            # dataset's fits is still a real human -- and the only option for
            # mushrooms, which was never fitted. tested_w_xai is *not* relaxable
            # here; it is what distinguishes a unit's two halves.
            relax=("xai_method", "dataId"),
            pool=pool,
        )

        for participant, by_half in units.items():
            for half, draw in by_half.items():
                tested = half == "true"
                key = _select_pool_model_key(models, xai_type, tested)
                _warn_on_illegal_strategy(draw, xai_type, tested)
                draws[(participant, key)] = draw
                drawn_by_participant.setdefault(participant, {})[key] = draw

    def participant_models(participant_id: Any) -> dict[Any, Any]:
        """This participant's models, falling back to the shared one per key."""
        built = dict(models)
        for key, draw in drawn_by_participant.get(participant_id, {}).items():
            strategy = draw.attributes.get("strategy")
            if not strategy:
                # An unmatched cell: keep the condition's shared model rather
                # than inventing a strategy for it.
                continue
            xai_type = key[0] if isinstance(key, tuple) else key
            built[key] = make_coax_model(
                strategy,
                **fitted_coax_params(strategy, xai_type, **draw.parameters),
            )
        return built

    return participant_models, draws


def coax_population_trials(
    draws: Mapping[Any, Any],
    selected: pd.DataFrame,
) -> pd.DataFrame:
    """Replace the generated trials with the ones each drawn human actually saw.

    ``fitted_population`` deals every virtual participant a real
    ``(Participant ID, Session)``, and the parameters it deals were fitted by
    maximum likelihood against that person's responses **on that person's own
    trial sequence**. Scoring the resulting model on a different sequence
    measures something else: in the published corpus each participant saw 46 of
    the ~92 instances and no two participants overlap at all, so a generated
    table typically shares only 6-14 of 40 trials with the person the parameters
    came from, and CoAX is an exemplar-memory model whose answers depend on which
    exemplars are in memory and how long ago they were stored.

    So the trials follow the draw: each virtual participant runs its own human's
    logged instances, in their logged order, with their training/testing split
    and their tested halves. The design's trial table still decides *who* exists
    and in which condition -- it just no longer decides which instances they see.

    Read from the anonymised ``assets/human_data`` copy of the log, whose
    participant ids are the ones the parameter pool uses; the raw copy under
    ``src/cognitive_models/`` keys on un-anonymised session codes and does not
    join.
    """
    # Lazy, like _run_coax_study_from_corpus: coax_human_replay imports this
    # module's fitted-population table, so a module-level import is circular.
    from .coax_human_replay import (
        ANONYMISED_HUMAN_TRIALS_FILE,
        load_coax_human_trials,
    )

    units: dict[Any, tuple[Any, Any]] = {}
    for (participant, _key), draw in draws.items():
        fitted_id = getattr(draw, "fitted_participant_id", None)
        session = (getattr(draw, "attributes", None) or {}).get("fitted_session")
        if fitted_id is None or session is None:
            continue
        units[participant] = (fitted_id, session)

    if not units:
        raise ValueError(
            "mode='fitted_population' could not identify the fitted people behind "
            "this draw, so it cannot replay their trials. This happens when the "
            "pool matched nothing and every participant fell back to the fitted "
            "population means."
        )

    log = load_coax_human_trials(path=ANONYMISED_HUMAN_TRIALS_FILE)
    by_unit = {key: rows for key, rows in log.groupby(["participantId", "session"], sort=False)}

    frames: list[pd.DataFrame] = []
    missing: list[tuple[Any, Any]] = []
    for participant, (fitted_id, session) in sorted(units.items(), key=lambda item: str(item[0])):
        rows = by_unit.get((fitted_id, int(session)))
        if rows is None or rows.empty:
            missing.append((fitted_id, session))
            continue
        rows = rows.copy()
        # The virtual participant keeps its own id so the design's conditions,
        # provenance and per-participant aggregation all still line up.
        rows["participantId"] = participant
        frames.append(rows)

    if missing:
        warnings.warn(
            f"{len(missing)} drawn person-session(s) have fitted parameters but no "
            f"logged trials to replay (e.g. {missing[:3]}); those participants are "
            "dropped from the run.",
            stacklevel=2,
        )
    if not frames:
        raise ValueError(
            "None of the drawn people have logged trials, so mode="
            "'fitted_population' has nothing to run."
        )

    trials = pd.concat(frames, ignore_index=True, sort=False)
    # The executor sorts on these, and the study-level order column is rebuilt
    # by the caller against the new table.
    trials = trials.sort_values(["participantId", "trialId"], kind="stable")
    return trials.reset_index(drop=True)


def _assert_tuple_keyed(models: Mapping[Any, Any], xai_type: str) -> None:
    """Refuse an xai_type served by one model for both tested halves."""
    if any(isinstance(key, tuple) and key[0] == xai_type for key in models):
        return
    raise ValueError(
        f"mode='fitted_population' needs one model per tested half, but xai_type "
        f"{xai_type!r} is registered under a plain key, so both halves would share "
        "one instance and only one of the drawn person's two strategies could be "
        "applied. Build the mapping with "
        "coax_models_for_trials(trials, by_tested_condition=True). Registered "
        f"keys: {sorted(models, key=repr)}."
    )


def _warn_on_illegal_strategy(draw: Any, xai_type: str, tested: bool) -> None:
    """Say so when a fitted human used a strategy the generator would not pick.

    The drawn strategy wins regardless: ``COAX_STRATEGY_EXCLUSIONS`` is the
    reference pipeline's rule for *generating* a condition, while this is what a
    real participant was actually fitted as. A mismatch is worth surfacing
    because it usually means the pool was relaxed onto a neighbouring cell.
    """
    strategy = draw.attributes.get("strategy")
    if not strategy:
        return
    allowed = coax_available_strategies(xai_type, tested)
    if strategy not in allowed:
        warnings.warn(
            f"Fitted participant {draw.fitted_participant_id} used {strategy!r} in "
            f"xai_type={xai_type!r} tested_w_xai={tested!r}, which "
            f"coax_available_strategies() excludes (allows {allowed}). Keeping the "
            "fitted strategy -- the fit is what the human did. "
            f"parameter_source={draw.parameter_source!r}.",
            stacklevel=2,
        )


def _select_pool_model_key(models: Mapping[Any, Any], xai_type: str, tested: bool) -> Any:
    """The key the executor will select for this condition, resolved up front."""
    for key in ((xai_type, tested), xai_type):
        if key in models:
            return key
    raise ValueError(
        f"No CoAX model registered for xai_type={xai_type!r} tested_w_xai={tested!r}; "
        f"registered keys: {sorted(models, key=repr)}."
    )


def _resolve_corpus_data_id(
    data_id: Optional[str], trials: pd.DataFrame, dataset: Any
) -> str:
    """Which corpus dataset a corpus-backed run should serve.

    An explicit ``data_id`` wins; otherwise the trials name it, and finally the
    prepared dataset does. Trials spanning two datasets are rejected rather than
    silently served from one, because a corpus instance id means a different
    instance in each dataset.
    """
    if data_id is not None:
        return str(data_id)
    for column in ("dataId", "dataset", "dataset_id", "appId"):
        if column in trials.columns:
            names = {str(value) for value in trials[column].dropna().unique()}
            if len(names) > 1:
                raise ValueError(
                    f"Trials span more than one dataset ({sorted(names)}); a corpus "
                    "instance id names a different instance in each, so run one "
                    "dataset at a time via data_id= or condition_filter."
                )
            if names:
                return next(iter(names))
    dataset_id = getattr(dataset, "dataset_id", None)
    if dataset_id is not None:
        return str(dataset_id)
    raise ValueError(
        "source='corpus' needs a dataset name. Pass data_id=, or give the trials "
        "a 'dataId' column."
    )


def _run_coax_study_from_corpus(
    study: Any,
    selected: pd.DataFrame,
    models: Any,
    *,
    data_id: Optional[str],
    dataset: Any,
    dvs: Optional[Mapping[str, Any]],
    train_with_explanation: bool,
    participant_models: Optional[Any] = None,
    participant_draws: Optional[Mapping[Any, Any]] = None,
    share_training_memory: bool = False,
    store: bool,
) -> pd.DataFrame:
    """Run selected trials against the published CoAX user-study corpus.

    Imported lazily: ``coax_human_replay`` reads this module's fitted-population
    table, so a module-level import would be circular.
    """
    from .coax_human_replay import build_coax_study_repository, load_coax_corpus_tables

    resolved_data_id = _resolve_corpus_data_id(data_id, selected, dataset)
    tables = load_coax_corpus_tables()

    # One repository per XAI method: the corpus keeps lime and shap rows for the
    # same instance in one table, and the repository has no notion of a method.
    methods = (
        sorted({str(value) for value in selected["xai_method"].dropna().unique()})
        if "xai_method" in selected.columns
        else [None]
    ) or [None]

    runs = []
    for method in methods:
        method_trials = (
            selected
            if method is None
            else selected[selected["xai_method"].astype(str) == method]
        )
        if method_trials.empty:
            continue
        # "none"/"no_xai"/"control" is not a real corpus method -- it's what
        # _stamp_resolved_xai_method_on_trials writes for a trial whose
        # xai_type displays no explanation at all. get_trial_payload already
        # skips the explanation lookup for those via xai_type, not xai_method,
        # so no method filter is needed (or servable: the corpus never
        # collected an explanation for a condition that never showed one).
        # Treated the same as the "no xai_method column at all" case below.
        repository_method = (
            None if str(method).strip().lower() in {"none", "no_xai", "control"} else method
        )
        runs.append(
            run_coax_experiment_executor(
                method_trials,
                models,
                mode="whole_experiment",
                data_repository=build_coax_study_repository(
                    resolved_data_id, repository_method, tables=tables
                ),
                train_with_explanation=train_with_explanation,
                dvs=getattr(study, "DVs", None) if dvs is None else dvs,
                participant_models=participant_models,
                participant_draws=participant_draws,
                share_training_memory=share_training_memory,
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
    "coax_participant_models",
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

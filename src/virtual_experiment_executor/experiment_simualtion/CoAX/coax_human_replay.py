"""Replay the CoAX user study with the strategies fitted to its own participants.

Nothing here is generated. ``run_coax_study`` simulates a *new* design: it reads
trials from ``generate_trials()``, explanations from ``study.explanations()``,
and parameters from the fitted-population means. This module does the opposite
on all three counts -- the instances, conditions and trial order come from the
study's own trial log, the explanations come from the study's own published
corpus, and the strategy and its parameters come from the published
per-participant fit. So no AI model is trained and no explanation is computed:
each simulated participant sees exactly the instances and vectors the human of
the same id saw, answering with the strategy that human was fitted to.

Three facts about the source data shape everything below.

*Only three of the four datasets were fitted.* The trial log carries ``adult``,
``forest_cover``, ``mushrooms`` and ``wine_quality``; the refit carries only the
first three (see :data:`FITTED_DATA_IDS`). Mushrooms trials therefore have no
fitted strategy and are skipped unless a fallback is asked for.

*The fit is a hard assignment.* Despite the ``Assignment Prob`` column, the
deterministic-local refit leaves exactly one row per
``(participant, session, XAIType, tested)`` and every probability is 1.0, so
there is no mixture to sample from -- the strategy is simply looked up.

*The corpus holds four datasets and two XAI methods in one file.* ``appId`` is
the corpus files' own spelling of ``dataId`` -- the same thing, the dataset's
name -- so it is mapped onto ``dataId`` on load and every argument here is a
``data_id``. What no spelling fixes is the XAI method: ``attribution.csv``
carries a lime row *and* a shap row for every instance, and the repository has
no notion of a method, so an unscoped lookup returns whichever comes first.
:func:`build_coax_study_repository` therefore slices to one dataset and one
method before building, rather than filtering at lookup time.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import numpy as np
import pandas as pd

from .coax_study_runner import FITTED_COAX_PARAMS
from .coax_trial_executor import (
    DEFAULT_COAX_PARAMS,
    _canonical_tested_condition,
    _canonical_xai_type,
    make_coax_model,
    run_coax_experiment_executor,
    CoAXAssetRepository,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
COAX_STUDY_ROOT = REPO_ROOT / "src" / "cognitive_models" / "CoAX"

#: The study's own published corpus, as shipped with the CoAX sources.
STUDY_CORPUS_DIR = COAX_STUDY_ROOT / "data" / "datasets" / "standard set"

#: Backwards-compatible alias for :data:`STUDY_CORPUS_DIR`.
CORPUS_DIR = STUDY_CORPUS_DIR

#: Where the corpus is read from by default. ``assets/`` is the repository's
#: canonical copy -- kept in step with the user-study apparatus by
#: ``assets/sync_apparatus_corpora.py`` and verified equal to
#: :data:`STUDY_CORPUS_DIR` for every row it carries -- and is what the CoAX and
#: CoXAM trial executors already read, so all three now agree on one source.
ASSET_TABLE_FILES: dict[str, Path] = {
    "features": REPO_ROOT / "assets" / "ai_dataset" / "CoAX" / "values.csv",
    "predictions": REPO_ROOT / "assets" / "explanations" / "CoAX" / "none.csv",
    "attribution": REPO_ROOT / "assets" / "explanations" / "CoAX" / "attribution.csv",
    "importance": REPO_ROOT / "assets" / "explanations" / "CoAX" / "importance.csv",
}

#: The same four tables in the study's own single-directory layout.
STUDY_TABLE_FILES: dict[str, Path] = {
    "features": STUDY_CORPUS_DIR / "values.csv",
    "predictions": STUDY_CORPUS_DIR / "none.csv",
    "attribution": STUDY_CORPUS_DIR / "attribution.csv",
    "importance": STUDY_CORPUS_DIR / "importance.csv",
}

#: One row per human trial step, with the participant's response.
HUMAN_TRIALS_FILE = (
    COAX_STUDY_ROOT / "data" / "user study results" / "3-datasets-jan-09-2026-trials.csv"
)

#: One row per (participant, session, XAIType, tested) with the fitted strategy
#: and its parameters.
FITTED_PARAMS_FILE = (
    COAX_STUDY_ROOT / "results" / "pop_em_subset" / "refit_all_assignments_detlocal-Jan-10.csv"
)

#: Datasets the refit actually covers. The trial log also carries ``mushrooms``,
#: which was collected but never fitted.
FITTED_DATA_IDS: tuple[str, ...] = ("adult", "forest_cover", "wine_quality")

#: The refit's strategy labels, which are prose rather than the class names
#: ``make_coax_model`` resolves. Only "Attribution Sum" survives that function's
#: own normalization, so all four are mapped here explicitly.
FITTED_STRATEGY_ALIASES: dict[str, str] = {
    "attribution sum": "AttributionSum",
    "importance categorization": "ImportanceCategorization",
    "salient-features categorization": "SalientFeatures",
    "sensitive-features categorization": "SensitiveFeatures",
}

#: Trial-log column -> executor column. The executor reads snake_case names off
#: each trial row; the study log uses its own display headers.
_TRIAL_COLUMN_MAP = {
    "Participant ID": "participantId",
    "Instance Index": "instanceId",
    "Trial Index": "trialId",
    "appId": "dataId",
    "expMethod": "xai_method",
    "Session": "session",
}

#: Human columns carried through to the result so a replayed row sits next to
#: the response it is meant to be compared against.
_HUMAN_COLUMN_MAP = {
    "Response": "human_response",
    "Correct": "human_correct",
    "Timing": "human_time",
    "AI Prediction": "human_ai_prediction",
}


def _canonical_data_id(value: Any) -> str:
    """Normalize the display names the published results file uses.

    ``Human and CoAX and Baselines.csv`` renames two datasets for presentation
    (``forest_cover`` -> ``covertype``, ``wine_quality`` -> ``wine quality``);
    the trial log and the corpus both use the underscored ids.
    """
    text = str(value).strip().lower().replace(" ", "_")
    return {"covertype": "forest_cover"}.get(text, text)


def load_coax_corpus_tables(source: str = "assets") -> dict[str, pd.DataFrame]:
    """The published corpus, keyed by ``dataId``.

    ``source="assets"`` (the default) reads the repository's own
    ``assets/`` copy, which is what every other CoAX and CoXAM runner reads.
    ``source="study"`` reads the CoAX sources' ``standard set`` directly; the
    two carry identical values, so this is for verifying that rather than for
    getting different numbers.

    ``appId`` is the study files' spelling of the dataset name and is mapped
    onto ``dataId``; the asset copies already use ``dataId``. Values are
    normalized either way, so a caller can pass the published results file's
    display names.
    """
    files = {"assets": ASSET_TABLE_FILES, "study": STUDY_TABLE_FILES}.get(source)
    if files is None:
        raise ValueError(f"source must be 'assets' or 'study', not {source!r}.")

    tables: dict[str, pd.DataFrame] = {}
    for name, path in files.items():
        table = pd.read_csv(path, low_memory=False)
        table = table.rename(columns={"appId": "dataId"})
        table["dataId"] = table["dataId"].map(_canonical_data_id)
        tables[name] = table
    return tables


def build_coax_study_repository(
    data_id: str,
    xai_method: Optional[str] = None,
    *,
    tables: Optional[Mapping[str, pd.DataFrame]] = None,
) -> CoAXAssetRepository:
    """A repository serving exactly one dataset and one XAI method.

    Scoping happens here rather than being left to the repository, which cannot
    do it (see the module docstring). Passing ``tables`` avoids re-reading the
    corpus once per participant.
    """
    source = dict(tables) if tables is not None else load_coax_corpus_tables()
    name = _canonical_data_id(data_id)

    def scope(table: pd.DataFrame, by_method: bool = False) -> pd.DataFrame:
        rows = table[table["dataId"].astype(str) == name]
        if by_method and xai_method is not None and "expMethod" in rows.columns:
            rows = rows[rows["expMethod"].astype(str) == str(xai_method)]
        return rows.copy()

    features = scope(source["features"])
    if features.empty:
        raise ValueError(
            f"No corpus instances for dataId={name!r}. "
            f"Available: {sorted(source['features']['dataId'].astype(str).unique())}."
        )
    explanations = {
        xai_type: scope(source[xai_type], by_method=True)
        for xai_type in ("attribution", "importance")
    }
    explanations = {key: table for key, table in explanations.items() if not table.empty}
    if xai_method is not None and not explanations:
        # A design can declare xai_method as its own IV with levels the
        # published corpus never collected (it only ever generated lime/shap,
        # one method per dataset) -- without this, the caller gets a
        # repository that looks valid but silently has no explanation table
        # at all, and the failure only surfaces much later as "No explanation
        # table for xai_type=... Available: []" with no mention of the
        # dataset or method actually at fault.
        available = sorted(
            scope(source["attribution"])["expMethod"].astype(str).unique().tolist()
        )
        raise ValueError(
            f"The published CoAX corpus has no {name!r} explanations for "
            f"xai_method={xai_method!r}. Available for this dataset: {available!r}. "
            "Restrict the design's xai_method IV to one of these, or drop xai_method "
            "as a factor and let CoAX use its own default method per xai_type."
        )
    return CoAXAssetRepository.from_tables(
        features=features,
        predictions=scope(source["predictions"]),
        explanations=explanations,
    )


def coax_corpus_instance_ids(
    data_id: str, *, tables: Optional[Mapping[str, pd.DataFrame]] = None
) -> list[int]:
    """Instance ids the published corpus carries for one dataset.

    Pass to ``study.generate_trials(allowed_instance_ids=...)`` so a
    ``source='corpus'`` run never references an instance the corpus lacks. The
    id spaces differ sharply per dataset -- wine_quality has 122 instances,
    mushrooms 3,939 -- and an id means a different instance in each.
    """
    source = dict(tables) if tables is not None else load_coax_corpus_tables()
    name = _canonical_data_id(data_id)
    rows = source["features"][source["features"]["dataId"].astype(str) == name]
    if rows.empty:
        raise ValueError(
            f"No corpus instances for dataId={name!r}. "
            f"Available: {sorted(source['features']['dataId'].astype(str).unique())}."
        )
    return sorted(int(value) for value in rows["instanceId"].unique())


def coax_human_instance_ids(
    data_id: str, *, phase: Optional[str] = None, trials: Optional[pd.DataFrame] = None
) -> list[int]:
    """The instance ids the human participants actually saw for one dataset.

    A tighter constraint than :func:`coax_corpus_instance_ids`: the study drew
    only 92 of each dataset's instances -- 20 for training and a disjoint 72 for
    testing -- and used the same ones for every participant. ``phase`` selects
    ``"training"`` or ``"testing"``; omitting it returns both.
    """
    log = load_coax_human_trials(data_ids=[data_id]) if trials is None else trials
    if phase is not None:
        log = log[log["phase"].astype(str) == str(phase)]
    return sorted(int(value) for value in log["instanceId"].unique())


def load_coax_human_trials(
    data_ids: Optional[Iterable[str]] = None,
    participant_ids: Optional[Iterable[Any]] = None,
    *,
    path: Path = HUMAN_TRIALS_FILE,
) -> pd.DataFrame:
    """The human trial log, in the schema ``run_coax_experiment_executor`` reads.

    Only ``Step == 'Infer'`` rows are kept: those are the trials the participant
    answered. The log's separate ``Feedback`` rows carry no response, and the
    executor generates its own feedback step for training trials, so keeping
    them would double-count.
    """
    log = pd.read_csv(path, low_memory=False)
    log = log[log["Step"].astype(str).str.strip().str.lower() == "infer"].copy()
    log["appId"] = log["appId"].map(_canonical_data_id)

    if data_ids is not None:
        wanted = {_canonical_data_id(value) for value in data_ids}
        log = log[log["appId"].isin(wanted)]
    if participant_ids is not None:
        log = log[log["Participant ID"].isin(set(participant_ids))]

    trials = log.rename(columns={**_TRIAL_COLUMN_MAP, **_HUMAN_COLUMN_MAP})
    # XAIType is blank for the no-explanation control, which is the executor's
    # "none" condition rather than missing data.
    trials["xai_type"] = [
        _canonical_xai_type("none" if pd.isna(value) else value) for value in log["XAIType"]
    ]
    trials["tested_w_xai"] = [
        _canonical_tested_condition(value) for value in log["Tested w/ XAI"]
    ]
    trials["phase"] = np.where(
        log["trialType"].astype(str).str.strip().str.lower() == "train", "training", "testing"
    )
    trials["instanceId"] = trials["instanceId"].astype(int)
    return trials.reset_index(drop=True)


def load_coax_fitted_params(*, path: Path = FITTED_PARAMS_FILE) -> pd.DataFrame:
    """The per-participant refit, keyed to match :func:`load_coax_human_trials`."""
    fits = pd.read_csv(path)
    fits["appId"] = fits["appId"].map(_canonical_data_id)
    fits["xai_type"] = [
        _canonical_xai_type("none" if pd.isna(value) else value) for value in fits["XAIType"]
    ]
    fits["tested_w_xai"] = [
        _canonical_tested_condition(value) for value in fits["Tested w/ XAI"]
    ]
    return fits


def coax_params_from_fit(fit_row: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    """One refit row -> the strategy name and constructor kwargs it describes.

    The fitted values are used as they stand, deliberately unclamped:
    ``COAX_PARAM_BOUNDS`` are the simulator's slider limits, and this population
    ranges well outside them (sensitivity to 100, retrieval_threshold to -4.0,
    scaling_factor to 8.0). Clamping would silently replace a fitted participant
    with a different one.

    ``decay_param`` is not fitted in that file, so it keeps the reference value
    the way :data:`FITTED_COAX_PARAMS` does.
    """
    label = str(fit_row["Strategy"]).strip().lower()
    try:
        strategy_name = FITTED_STRATEGY_ALIASES[label]
    except KeyError as exc:
        raise ValueError(
            f"Unknown fitted strategy {fit_row['Strategy']!r}. "
            f"Known: {sorted(FITTED_STRATEGY_ALIASES)}."
        ) from exc

    params: dict[str, Any] = {
        "decay_param": DEFAULT_COAX_PARAMS["decay_param"],
        "k": int(fit_row["k"]),
        "retrieval_threshold": float(fit_row["retrieval_threshold"]),
    }
    if strategy_name == "AttributionSum":
        params["scaling_factor"] = float(fit_row["scaling_factor"])
        params["explanation_type"] = _canonical_xai_type(fit_row["xai_type"])
    else:
        params["sensitivity"] = float(fit_row["sensitivity"])
    return strategy_name, params


def coax_models_from_fits(fit_rows: pd.DataFrame) -> dict[tuple[str, bool], Any]:
    """The ``(xai_type, tested_w_xai)`` -> strategy map for one participant-session."""
    models: dict[tuple[str, bool], Any] = {}
    for _, row in fit_rows.iterrows():
        strategy_name, params = coax_params_from_fit(row)
        models[(row["xai_type"], bool(row["tested_w_xai"]))] = make_coax_model(
            strategy_name, **params
        )
    return models


def _population_fallback_model(xai_type: str, tested_w_xai: bool) -> Any:
    """A fitted-population model for a condition the refit does not cover."""
    from .coax_trial_executor import default_coax_strategy

    strategy_name = default_coax_strategy(xai_type, tested_w_xai)
    params = dict(FITTED_COAX_PARAMS.get(strategy_name, {}))
    if strategy_name == "AttributionSum":
        params["explanation_type"] = xai_type
    return make_coax_model(strategy_name, **params)


def _human_comparable_step(phase: str, tested_w_xai: bool) -> str:
    """Which generated step the human's single logged response corresponds to.

    A training trial is answered before the explanation is revealed, so the
    human's response is the pre-explanation guess; a testing trial is answered
    once, with the explanation only if that participant was tested with it.
    """
    if phase == "training":
        return "infer_no_explanation"
    return "infer_with_explanation" if tested_w_xai else "infer_no_explanation"


def run_coax_human_replay(
    *,
    data_ids: Optional[Iterable[str]] = None,
    participant_ids: Optional[Iterable[Any]] = None,
    trials: Optional[pd.DataFrame] = None,
    fits: Optional[pd.DataFrame] = None,
    on_missing_fit: str = "skip",
    dvs: Optional[Mapping[str, Any]] = None,
    train_with_explanation: bool = True,
) -> pd.DataFrame:
    """Re-run every logged human trial with that human's own fitted strategy.

    Returns one row per inference step, in the shape
    ``run_coax_experiment_executor`` produces, plus the human's own response
    (``human_response``, ``human_correct``, ``human_time``) and a
    ``human_comparable`` flag marking the single step the logged response should
    be compared against -- a training trial produces two simulated steps but the
    human answered once.

    ``on_missing_fit`` decides what happens to a participant-condition the refit
    does not cover (all of ``mushrooms``, and the 61 participants who were
    logged but not fitted): ``"skip"`` drops them and reports the count,
    ``"population"`` falls back to the fitted-population means in
    :data:`FITTED_COAX_PARAMS`, and ``"raise"`` fails.
    """
    if on_missing_fit not in {"skip", "population", "raise"}:
        raise ValueError(
            f"on_missing_fit must be 'skip', 'population' or 'raise', not {on_missing_fit!r}."
        )

    trials_df = (
        load_coax_human_trials(data_ids=data_ids, participant_ids=participant_ids)
        if trials is None
        else pd.DataFrame(trials).copy()
    )
    if trials_df.empty:
        raise RuntimeError("No human trials selected. Check data_ids/participant_ids.")

    fits_df = load_coax_fitted_params() if fits is None else pd.DataFrame(fits).copy()
    fit_index = {
        key: rows
        for key, rows in fits_df.groupby(["Participant ID", "Session"], sort=False, dropna=False)
    }

    corpus = load_coax_corpus_tables()
    repositories: dict[tuple[str, Any], CoAXAssetRepository] = {}

    runs: list[pd.DataFrame] = []
    skipped_trials = 0
    skipped_keys: set[tuple[Any, ...]] = set()

    group_columns = ["participantId", "session", "dataId", "xai_method"]
    for key, group in trials_df.groupby(group_columns, sort=False, dropna=False):
        participant, session, data_id, xai_method = key
        fit_rows = fit_index.get((participant, session))

        models: dict[Any, Any] = {} if fit_rows is None else coax_models_from_fits(fit_rows)
        needed = {
            (row["xai_type"], bool(row["tested_w_xai"]))
            for _, row in group[["xai_type", "tested_w_xai"]].drop_duplicates().iterrows()
        }
        missing = needed - set(models)
        if missing:
            if on_missing_fit == "raise":
                raise ValueError(
                    f"No fitted strategy for participant {participant!r} session "
                    f"{session!r} conditions {sorted(missing)}."
                )
            if on_missing_fit == "population":
                for condition in missing:
                    models[condition] = _population_fallback_model(*condition)
            else:
                covered = group.apply(
                    lambda row: (row["xai_type"], bool(row["tested_w_xai"])) not in missing,
                    axis=1,
                )
                skipped_trials += int((~covered).sum())
                skipped_keys.add((participant, session, data_id))
                group = group[covered]
                if group.empty:
                    continue

        repository_key = (_canonical_data_id(data_id), xai_method)
        if repository_key not in repositories:
            repositories[repository_key] = build_coax_study_repository(
                data_id, xai_method, tables=corpus
            )

        runs.append(
            run_coax_experiment_executor(
                group,
                models,
                mode="whole_experiment",
                data_repository=repositories[repository_key],
                train_with_explanation=train_with_explanation,
                dvs=dvs,
            )
        )

    if not runs:
        raise RuntimeError(
            "Every selected trial lacked a fitted strategy. "
            f"{FITTED_DATA_IDS} are the only fitted datasets; pass "
            "on_missing_fit='population' to simulate the rest."
        )

    results = pd.concat(runs, ignore_index=True, sort=False)
    results["human_comparable"] = [
        row["step"] == _human_comparable_step(row["phase"], bool(row.get("tested_w_xai") or False))
        for _, row in results.iterrows()
    ]
    if skipped_trials:
        warnings.warn(
            f"Skipped {skipped_trials} trial(s) across {len(skipped_keys)} "
            "participant-session(s) with no fitted strategy "
            f"(the refit covers only {list(FITTED_DATA_IDS)}). "
            "Pass on_missing_fit='population' to simulate them from the "
            "fitted-population means instead.",
            stacklevel=2,
        )
    return results


def coax_replay_agreement(results: pd.DataFrame) -> pd.DataFrame:
    """Agreement between each replayed participant and the human they replay.

    Scored on ``human_comparable`` rows only, so a training trial contributes
    the one response the human actually gave rather than both simulated steps.
    """
    rows = results[results["human_comparable"]].copy()
    rows = rows[rows["human_response"].notna() & rows["agent_prediction"].notna()]
    rows["agrees_with_human"] = (
        rows["human_response"].astype(float).astype(int)
        == rows["agent_prediction"].astype(float).astype(int)
    )
    grouped = rows.groupby(["dataId", "xai_type", "tested_w_xai", "phase"], dropna=False)
    return grouped.agg(
        trials=("agrees_with_human", "size"),
        agreement_with_human=("agrees_with_human", "mean"),
        agent_matches_ai=("cognitive_correct_vs_ai", "mean"),
        human_matches_ai=("human_correct", "mean"),
    ).reset_index()


__all__ = [
    "ASSET_TABLE_FILES",
    "CORPUS_DIR",
    "STUDY_CORPUS_DIR",
    "FITTED_DATA_IDS",
    "FITTED_PARAMS_FILE",
    "FITTED_STRATEGY_ALIASES",
    "HUMAN_TRIALS_FILE",
    "build_coax_study_repository",
    "coax_corpus_instance_ids",
    "coax_human_instance_ids",
    "coax_models_from_fits",
    "coax_params_from_fit",
    "coax_replay_agreement",
    "load_coax_corpus_tables",
    "load_coax_fitted_params",
    "load_coax_human_trials",
    "run_coax_human_replay",
]

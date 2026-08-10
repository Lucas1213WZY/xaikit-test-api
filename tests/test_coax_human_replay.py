"""Replaying the CoAX study must serve each human their own data and strategy.

These run against the real published corpus and the real refit -- both are fixed
files a few MB in size, and the whole point of the module is that it reads them
rather than anything generated, so a mocked corpus would test nothing.
"""

from types import SimpleNamespace

import pandas as pd
import pytest

from src.virtual_experiment_executor.experiment_simualtion.CoAX.coax_human_replay import (
    FITTED_DATA_IDS,
    FITTED_STRATEGY_ALIASES,
    _canonical_data_id,
    build_coax_study_repository,
    coax_corpus_instance_ids,
    coax_human_instance_ids,
    coax_params_from_fit,
    coax_replay_agreement,
    load_coax_corpus_tables,
    load_coax_fitted_params,
    load_coax_human_trials,
    run_coax_human_replay,
)
from src.virtual_experiment_executor.experiment_simualtion.CoAX.coax_study_runner import (
    run_coax_study,
)


@pytest.fixture(scope="module")
def corpus():
    return load_coax_corpus_tables()


@pytest.fixture(scope="module")
def fits():
    return load_coax_fitted_params()


# -- the assets are the corpus ---------------------------------------------


def test_assets_carry_the_same_values_as_the_study_corpus(corpus):
    """assets/ is synced from the apparatus, which equals the CoAX standard set.

    Guards the failure that motivated the sync: assets/explanations/CoAX held
    LIME/SHAP re-run under a different seed, so every attribution differed from
    what the participants were shown by ~1-2%.
    """
    study = load_coax_corpus_tables("study")
    for name, asset_table in corpus.items():
        keys = [
            key
            for key in ("dataId", "instanceId", "expMethod", "modelName")
            if key in asset_table.columns and key in study[name].columns
        ]
        left, right = asset_table.copy(), study[name].copy()
        for key in keys:
            left[key] = left[key].astype(str)
            right[key] = right[key].astype(str)
        merged = left.merge(right, on=keys, suffixes=("_a", "_s"))
        assert len(merged) == len(left), f"{name}: assets hold rows the study corpus lacks"
        for column in (c for c in left.columns if c in right.columns and c not in keys):
            a_col, s_col = merged[f"{column}_a"], merged[f"{column}_s"]
            if pd.api.types.is_numeric_dtype(a_col) and pd.api.types.is_numeric_dtype(s_col):
                assert (
                    a_col.fillna(-9e12).round(9).equals(s_col.fillna(-9e12).round(9))
                ), f"{name}.{column}"
            else:
                assert (a_col.astype(str) == s_col.astype(str)).all(), f"{name}.{column}"


def test_assets_cover_every_instance_the_fitted_study_used(corpus):
    """A replay fails outright if an instance a participant saw is absent."""
    log = load_coax_human_trials(data_ids=FITTED_DATA_IDS)
    for data_id, rows in log.groupby("dataId"):
        available = set(coax_corpus_instance_ids(data_id, tables=corpus))
        assert set(rows["instanceId"]) <= available, data_id


def test_corpus_source_is_validated():
    with pytest.raises(ValueError, match="source must be"):
        load_coax_corpus_tables("apparatus")


@pytest.fixture(scope="module")
def trials():
    return load_coax_human_trials()


@pytest.fixture(scope="module")
def sample_participants(fits):
    return sorted(fits["Participant ID"].unique())[:3]


# -- dataset scoping -------------------------------------------------------


def test_each_dataset_serves_its_own_instance(corpus):
    """The bug this guards: appId/dataId disagreed, so the filter never applied
    and every dataset returned whichever row for that id came first."""
    served = {}
    for data_id in FITTED_DATA_IDS:
        repository = build_coax_study_repository(data_id, tables=corpus)
        features, _pred, _exp = repository.get_trial_payload(5, "none", data_id=data_id)
        served[data_id] = tuple(features)
    assert len(set(served.values())) == len(served), (
        f"datasets collapsed onto one another: {served}"
    )


def test_instance_id_spaces_differ_per_dataset(corpus):
    counts = {
        data_id: len(coax_corpus_instance_ids(data_id, tables=corpus))
        for data_id in FITTED_DATA_IDS
    }
    assert counts == {"wine_quality": 122, "adult": 300, "forest_cover": 300}


def test_assets_carry_one_xai_method_per_dataset(corpus):
    """The study showed each dataset under a single method, and so do the assets.

    Scoping by method still matters -- the study corpus holds both -- but the
    canonical assets deliberately carry only the one that dataset was shown
    with, so a replay cannot silently serve the other method's vectors.
    """
    study_method = {"adult": "lime", "forest_cover": "shap", "wine_quality": "lime"}
    for data_id, expected in study_method.items():
        methods = set(
            corpus["attribution"]
            .loc[corpus["attribution"]["dataId"].astype(str) == data_id, "expMethod"]
            .astype(str)
        )
        assert methods == {expected}, (data_id, methods)


def test_explanations_are_scoped_to_one_xai_method():
    """The study corpus holds a lime row and a shap row for every instance."""
    study = load_coax_corpus_tables("study")
    vectors = {}
    for method in ("lime", "shap"):
        repository = build_coax_study_repository("wine_quality", method, tables=study)
        _f, _p, explanation = repository.get_trial_payload(
            0, "attribution", data_id="wine_quality"
        )
        vectors[method] = tuple(explanation)
    assert vectors["lime"] != vectors["shap"]


def test_unknown_dataset_is_rejected(corpus):
    with pytest.raises(ValueError, match="No corpus instances"):
        build_coax_study_repository("not_a_dataset", tables=corpus)


@pytest.mark.parametrize(
    ("given", "expected"),
    [("covertype", "forest_cover"), ("wine quality", "wine_quality"), ("adult", "adult")],
)
def test_published_display_names_are_normalized(given, expected):
    assert _canonical_data_id(given) == expected


# -- the human trial log ---------------------------------------------------


def test_only_answered_trials_are_kept(trials):
    """Feedback rows carry no response; the executor makes its own feedback step."""
    assert trials["human_response"].notna().all()


def test_phases_map_onto_the_executors_vocabulary(trials):
    assert set(trials["phase"].unique()) == {"training", "testing"}


def test_training_and_testing_instances_are_disjoint():
    """The study drew 20 training and a separate 72 testing instances per dataset."""
    for data_id in FITTED_DATA_IDS:
        training = set(coax_human_instance_ids(data_id, phase="training"))
        testing = set(coax_human_instance_ids(data_id, phase="testing"))
        assert len(training) == 20 and len(testing) == 72, data_id
        assert not training & testing, data_id


def test_blank_xai_type_is_the_none_condition(trials):
    """The control condition shows no explanation; it is not missing data."""
    assert "none" in set(trials["xai_type"].unique())
    assert trials["xai_type"].notna().all()


def test_every_human_instance_exists_in_the_study_corpus(trials):
    """True for all four datasets against the study's own corpus.

    The canonical assets cover only the three the refit fitted; that narrower
    guarantee is pinned by
    ``test_assets_cover_every_instance_the_fitted_study_used``.
    """
    study = load_coax_corpus_tables("study")
    for data_id, rows in trials.groupby("dataId"):
        available = set(coax_corpus_instance_ids(data_id, tables=study))
        assert set(rows["instanceId"]) <= available, data_id


# -- fitted parameters -----------------------------------------------------


def test_every_fitted_strategy_label_is_known(fits):
    labels = {str(value).strip().lower() for value in fits["Strategy"].dropna().unique()}
    assert labels <= set(FITTED_STRATEGY_ALIASES), labels - set(FITTED_STRATEGY_ALIASES)


def test_the_refit_covers_only_three_datasets(fits):
    assert set(fits["appId"].unique()) == set(FITTED_DATA_IDS)


def test_the_fit_is_a_hard_assignment(fits):
    """One row per participant-session-condition, so there is no mixture."""
    key = ["Participant ID", "Session", "XAIType", "Tested w/ XAI"]
    assert fits.dropna(subset=key).groupby(key).size().max() == 1


def test_attribution_sum_takes_scaling_factor_not_sensitivity(fits):
    row = fits[fits["Strategy"] == "Attribution Sum"].iloc[0]
    strategy, params = coax_params_from_fit(row)
    assert strategy == "AttributionSum"
    assert "scaling_factor" in params and "sensitivity" not in params
    assert params["explanation_type"] == row["xai_type"]


def test_other_strategies_take_sensitivity_not_scaling_factor(fits):
    row = fits[fits["Strategy"] == "Sensitive-features categorization"].iloc[0]
    strategy, params = coax_params_from_fit(row)
    assert strategy == "SensitiveFeatures"
    assert "sensitivity" in params and "scaling_factor" not in params


def test_fitted_values_are_not_clamped_to_the_slider_bounds(fits):
    """COAX_PARAM_BOUNDS caps sensitivity at 20; this population reaches 100."""
    row = fits.loc[fits["sensitivity"].idxmax()]
    _strategy, params = coax_params_from_fit(row)
    assert params["sensitivity"] == pytest.approx(float(row["sensitivity"]))
    assert params["sensitivity"] > 20.0


def test_unknown_strategy_label_is_rejected():
    with pytest.raises(ValueError, match="Unknown fitted strategy"):
        coax_params_from_fit(
            {"Strategy": "Telepathy", "k": 3, "retrieval_threshold": -2.0, "xai_type": "none"}
        )


# -- replaying -------------------------------------------------------------


def test_replay_uses_each_participants_own_fitted_strategy(sample_participants, fits):
    results = run_coax_human_replay(participant_ids=sample_participants)
    expected = {
        FITTED_STRATEGY_ALIASES[str(label).strip().lower()]
        for label in fits[fits["Participant ID"].isin(sample_participants)]["Strategy"]
    }
    assert set(results["cognitive_model_strategy"].unique()) <= expected


def test_replay_marks_one_comparable_step_per_human_response(sample_participants):
    """A training trial simulates two steps; the human answered once."""
    results = run_coax_human_replay(participant_ids=sample_participants)
    training = results[results["phase"] == "training"]
    assert (training["step"] == "infer_no_explanation").sum() == int(
        training["human_comparable"].sum()
    )
    tested = results[(results["phase"] == "testing") & (results["tested_w_xai"] == True)]  # noqa: E712
    assert set(tested[tested["human_comparable"]]["step"].unique()) == {
        "infer_with_explanation"
    }


def test_replay_carries_the_human_response_alongside(sample_participants):
    results = run_coax_human_replay(participant_ids=sample_participants)
    for column in ("human_response", "human_correct", "human_time"):
        assert column in results.columns
    assert results[results["human_comparable"]]["human_response"].notna().all()


def test_unfitted_participants_are_skipped_with_a_warning(trials, fits):
    """mushrooms was collected but never fitted."""
    mushroom_only = set(trials[trials["dataId"] == "mushrooms"]["participantId"]) - set(
        fits["Participant ID"]
    )
    with pytest.raises(RuntimeError, match="on_missing_fit='population'"):
        run_coax_human_replay(participant_ids=sorted(mushroom_only)[:2])


def test_population_fallback_simulates_the_unfitted(trials, fits):
    """61 logged participants were never fitted, on fitted datasets too."""
    unfitted = sorted(
        set(trials[trials["dataId"].isin(FITTED_DATA_IDS)]["participantId"])
        - set(fits["Participant ID"])
    )[:2]
    assert unfitted, "expected some logged-but-unfitted participants"

    results = run_coax_human_replay(
        data_ids=FITTED_DATA_IDS, participant_ids=unfitted, on_missing_fit="population"
    )

    assert not results.empty
    assert results["agent_prediction"].notna().any()


def test_bad_missing_fit_policy_is_rejected():
    with pytest.raises(ValueError, match="on_missing_fit"):
        run_coax_human_replay(participant_ids=["x"], on_missing_fit="guess")


def test_agreement_is_scored_on_comparable_rows_only(sample_participants):
    results = run_coax_human_replay(participant_ids=sample_participants)
    agreement = coax_replay_agreement(results)
    assert agreement["trials"].sum() == int(
        results[results["human_comparable"]]["human_response"].notna().sum()
    )
    assert agreement["agreement_with_human"].between(0, 1).all()


# -- corpus-backed simulation ----------------------------------------------


def _corpus_study(trials_df):
    return SimpleNamespace(
        trials=trials_df.to_dict("records"),
        DVs={"accuracy": [0, 1]},
        combined_explanations=None,
        data=None,
        simulated_results=None,
    )


def test_corpus_source_needs_no_trained_model_or_explanations(trials, sample_participants):
    """The whole point: the same instances the humans saw, without a training run."""
    selected = trials[
        (trials["dataId"] == "wine_quality")
        & (trials["participantId"].isin(sample_participants))
    ]
    study = _corpus_study(selected)

    results = run_coax_study(study, source="corpus", mode="whole_experiment")

    assert not results.empty
    assert study.simulated_results is not None
    assert results["ai_prediction"].notna().all()


def test_corpus_source_rejects_trials_spanning_two_datasets(trials, sample_participants):
    selected = trials[trials["participantId"].isin(sample_participants)]
    assert selected["dataId"].nunique() > 1
    with pytest.raises(ValueError, match="span more than one dataset"):
        run_coax_study(_corpus_study(selected), source="corpus")


def test_unknown_source_is_rejected(trials):
    with pytest.raises(ValueError, match="source must be"):
        run_coax_study(_corpus_study(trials.head(4)), source="assets")


def test_study_source_still_requires_its_inputs(trials):
    with pytest.raises(RuntimeError, match="explanation table"):
        run_coax_study(_corpus_study(trials.head(4)))

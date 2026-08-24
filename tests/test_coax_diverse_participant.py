"""CoAX under ``mode="diverse_participant"``.

Runs against the published CoAX corpus rather than a freshly trained model, so
the test needs neither ``prepare_dataset`` nor ``train_AI_model``: the point
under test is which parameters each participant gets, not where the vectors come
from.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.virtual_experiment_executor.experiment_simualtion.CoAX.coax_study_runner import (
    coax_models_for_trials,
    coax_participant_models,
)
from src.virtual_experiment_executor.experiment_simualtion.CoAX.coax_trial_executor import (
    coax_available_instance_ids,
    run_coax_experiment_executor,
)

DATA_ID = "wine_quality"
XAI_METHOD = "lime"
PARTICIPANTS_PER_CONDITION = 6
TRAINING_TRIALS = 10
TESTING_TRIALS = 30


@pytest.fixture(scope="module")
def repository():
    from src.virtual_experiment_executor.experiment_simualtion.CoAX.coax_human_replay import (
        build_coax_study_repository,
    )

    return build_coax_study_repository(DATA_ID, XAI_METHOD)


@pytest.fixture(scope="module")
def trials():
    """A CoAX session: a training block, then a tested block.

    The training block is not optional decoration. CoAX's exemplar strategies
    *are* their memory, and memory is only written by ``feedback`` during
    training -- with a testing-only table the retrieval parameters have nothing
    to retrieve and every participant answers identically whatever their
    ``sensitivity`` and ``k`` say.
    """
    instance_ids = coax_available_instance_ids(DATA_ID)[: TRAINING_TRIALS + TESTING_TRIALS]
    training, testing = instance_ids[:TRAINING_TRIALS], instance_ids[TRAINING_TRIALS:]
    rows = []
    participant = 0
    for xai_type in ("attribution", "importance"):
        for _ in range(PARTICIPANTS_PER_CONDITION):
            participant += 1
            for position, instance_id in enumerate(training):
                rows.append(
                    {
                        "participantId": participant,
                        "trialId": position,
                        "instanceId": int(instance_id),
                        "dataId": DATA_ID,
                        "xai_method": XAI_METHOD,
                        "xai_type": xai_type,
                        "tested_w_xai": None,
                        "phase": "training",
                    }
                )
            for position, instance_id in enumerate(testing):
                rows.append(
                    {
                        "participantId": participant,
                        "trialId": 100 + position,
                        "instanceId": int(instance_id),
                        "dataId": DATA_ID,
                        "xai_method": XAI_METHOD,
                        "xai_type": xai_type,
                        "tested_w_xai": position % 2 == 0,
                        "phase": "testing",
                    }
                )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def models(trials):
    return coax_models_for_trials(trials)


@pytest.fixture(scope="module")
def drawn(trials, models):
    return coax_participant_models(trials, models, data_id=DATA_ID, seed=0)


def _run(trials, repository, models, participant_models=None, draws=None):
    return run_coax_experiment_executor(
        trials,
        models,
        data_repository=repository,
        dvs={"forward_accuracy": ["continuous"]},
        participant_models=participant_models,
        participant_draws=draws,
    )


def test_draws_are_filtered_to_the_condition_strategy(trials, models, drawn):
    """An AttributionSum participant must not get a SensitiveFeatures fit."""
    _participant_models, draws = drawn
    assert draws
    for (_participant, key), draw in draws.items():
        strategy = type(models[key]).__name__
        if strategy == "AttributionSum":
            assert "scaling_factor" in draw.parameters
            assert "sensitivity" not in draw.parameters
        else:
            assert "sensitivity" in draw.parameters


def test_every_participant_gets_its_own_parameters(trials, drawn):
    participant_models, _draws = drawn
    fingerprints = set()
    for participant in trials["participantId"].unique():
        built = participant_models(participant)
        fingerprints.add(
            tuple(
                (str(key), getattr(model, "retrieval_threshold", None), getattr(model, "k", None))
                for key, model in sorted(built.items(), key=lambda item: str(item[0]))
            )
        )
    assert len(fingerprints) > 1


def _spread(results: pd.DataFrame) -> pd.Series:
    per_participant = (
        results[results["phase"] == "testing"]
        .groupby(["xai_type", "participantId"])["forward_accuracy"]
        .mean()
    )
    return per_participant.groupby("xai_type").std()


def test_shared_parameters_give_every_participant_the_same_score(
    trials, repository, models
):
    """The bug, pinned: identical trials plus identical parameters, zero spread."""
    assert (_spread(_run(trials, repository, models)) == 0.0).all()


def test_diverse_run_produces_within_condition_variance(trials, repository, models, drawn):
    participant_models, draws = drawn
    spread = _spread(_run(trials, repository, models, participant_models, draws))
    assert (spread > 0).all(), spread.to_dict()


def test_results_carry_provenance(trials, repository, models, drawn):
    participant_models, draws = drawn
    results = _run(trials, repository, models, participant_models, draws)
    assert {"fitted_participant_id", "parameter_source", "parameter_pool"} <= set(
        results.columns
    )
    assert (results["parameter_pool"] == "coax").all()
    assert results["fitted_participant_id"].nunique() > 1


def test_shared_run_is_unchanged_without_draws(trials, repository, models):
    baseline = _run(trials, repository, models)
    again = _run(trials, repository, models, None, None)
    pd.testing.assert_frame_equal(baseline, again)
    assert "fitted_participant_id" not in baseline.columns


def test_draw_is_reproducible(trials, models):
    first = coax_participant_models(trials, models, data_id=DATA_ID, seed=3)[1]
    again = coax_participant_models(trials, models, data_id=DATA_ID, seed=3)[1]
    assert {k: v.fitted_participant_id for k, v in first.items()} == {
        k: v.fitted_participant_id for k, v in again.items()
    }


def test_unfitted_dataset_relaxes_rather_than_failing(trials, models):
    """mushrooms was never fitted; borrowing another dataset beats a global mean."""
    mushroom_trials = trials.assign(dataId="mushrooms")
    with pytest.warns(UserWarning, match="relaxed"):
        _participant_models, draws = coax_participant_models(
            mushroom_trials, models, data_id="mushrooms", seed=0
        )
    assert draws
    assert all(
        draw.parameter_source.startswith("pool_relaxed") for draw in draws.values()
    )

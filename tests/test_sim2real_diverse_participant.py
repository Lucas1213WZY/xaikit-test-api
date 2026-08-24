"""Sim2Real under ``mode="diverse_participant"``.

The degenerate case the mode exists for: every Sim2Real participant sees the
same corpus test instances and the model is deterministic, so a condition's
participants return byte-identical accuracies. The notebook's stored output
records a within-condition SD of exactly 0.0 in three of four conditions and
t statistics of 1e15.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.virtual_experiment_executor.experiment_simualtion.Sim2Real.sim2real_study_runner import (
    _draw_sim2real_participants,
)
from src.virtual_experiment_executor.experiment_simualtion.Sim2Real.sim2real_trial_executor import (
    build_sim2real_projector,
    run_sim2real_experiment_executor,
    sim2real_available_instance_ids,
)

# 8 rather than the notebook's 12: enough to measure a spread, and every
# executor run here fits one exemplar model per participant.
PARTICIPANTS_PER_CONDITION = 8
PROPERTIES = ("faithful", "sparse", "robust", "sparse_robust")


@pytest.fixture(scope="module")
def projector():
    return build_sim2real_projector()


@pytest.fixture(scope="module")
def trials(projector):
    """A Sim2Real testing block: every participant sees every test instance."""
    instance_ids = sim2real_available_instance_ids(projector, split="test")
    rows = []
    participant = 0
    for exp_property in PROPERTIES:
        for _ in range(PARTICIPANTS_PER_CONDITION):
            participant += 1
            for instance_id in instance_ids:
                rows.append(
                    {
                        "participantId": participant,
                        "instanceId": int(instance_id),
                        "xai_property": exp_property,
                        "phase": "testing",
                    }
                )
    return pd.DataFrame(rows)


def _accuracy_by_participant(results: pd.DataFrame) -> pd.DataFrame:
    return (
        results.groupby(["xai_property", "participantId"])["counterfactual_accuracy"]
        .mean()
        .reset_index()
    )


def _run(trials, projector, draws=None):
    return run_sim2real_experiment_executor(
        trials,
        projector=projector,
        dvs={"counterfactual_accuracy": ["continuous"]},
        participant_draws=draws,
    )


@pytest.fixture(scope="module")
def shared_results(trials, projector):
    """Today's behaviour: one parameter set for the whole study."""
    return _run(trials, projector)


@pytest.fixture(scope="module")
def diverse_draws(trials):
    return _draw_sim2real_participants(trials, seed=0, replace=None, pool=None)


@pytest.fixture(scope="module")
def diverse_results(trials, projector, diverse_draws):
    return _run(trials, projector, diverse_draws)


def test_shared_parameters_give_every_participant_the_same_score(shared_results):
    """The bug, pinned: without draws a condition has zero variance."""
    per_participant = _accuracy_by_participant(shared_results)
    spread = per_participant.groupby("xai_property")["counterfactual_accuracy"].std()
    # robust is the one condition with a lapse, so it alone varies today.
    assert spread.drop("robust").max() == 0.0


def test_diverse_participants_have_real_within_condition_variance(diverse_results):
    per_participant = _accuracy_by_participant(diverse_results)
    spread = per_participant.groupby("xai_property")["counterfactual_accuracy"].std()
    assert (spread > 0).all(), f"still degenerate: {spread.to_dict()}"


def test_every_participant_gets_a_distinct_fitted_human(trials, diverse_draws):
    draws = diverse_draws
    assert len(draws) == PARTICIPANTS_PER_CONDITION * len(PROPERTIES)
    assert all(draw.parameter_source == "pool" for draw in draws.values())
    # 11-12 fitted participants per property against 12 virtual ones: at most
    # one repeat per condition, never one person cloned twelve times.
    by_property: dict[str, list] = {}
    for participant, draw in draws.items():
        exp_property = trials.loc[
            trials["participantId"] == participant, "xai_property"
        ].iloc[0]
        by_property.setdefault(exp_property, []).append(draw.fitted_participant_id)
    for exp_property, assigned in by_property.items():
        assert len(set(assigned)) >= len(assigned) - 1, exp_property


def test_draw_is_filtered_to_the_participants_own_condition(trials, diverse_draws):
    """A faithful participant must not receive a robust participant's fit."""
    from src.virtual_experiment_executor.participant_pools import load_pool

    pool = load_pool("sim2real")
    draws = diverse_draws
    for participant, draw in draws.items():
        expected = trials.loc[
            trials["participantId"] == participant, "xai_property"
        ].iloc[0]
        fitted_row = pool[pool["participant_id"] == draw.fitted_participant_id]
        assert set(fitted_row["exp_property"]) == {expected}


def test_results_carry_provenance_columns(diverse_results):
    results = diverse_results
    assert {"fitted_participant_id", "parameter_source", "parameter_pool"} <= set(
        results.columns
    )
    assert (results["parameter_source"] == "pool").all()
    assert results["fitted_participant_id"].nunique() > 1
    assert "sampled_comparison_scale" in results.columns


def test_diverse_run_is_reproducible(trials, projector, diverse_results, diverse_draws):
    """Same seed, same assignment, same responses."""
    redrawn = _draw_sim2real_participants(trials, seed=0, replace=None, pool=None)
    assert [d.fitted_participant_id for d in redrawn.values()] == [
        d.fitted_participant_id for d in diverse_draws.values()
    ]
    again = _run(trials, projector, redrawn)
    pd.testing.assert_series_equal(
        diverse_results["counterfactual_accuracy"], again["counterfactual_accuracy"]
    )


def test_shared_run_is_unchanged_by_the_new_argument(trials, projector, shared_results):
    """No draws means today's numbers, exactly."""
    pd.testing.assert_frame_equal(shared_results, _run(trials, projector, None))

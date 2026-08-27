"""CoXAM under ``mode="diverse_participant"``.

Two halves. The draw itself is unit-tested against the real fitted tables (that
is where a column rename or a condition-spelling mismatch would bite), and one
integration test drives the real meta-policy environment to confirm the drawn
parameters actually change what a participant answers rather than being carried
along and ignored.
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import pytest

from src.virtual_experiment_executor.experiment_simualtion.CoXAM.coxam_study_runner import (
    _episode_participant,
    _trial_complexity,
    draw_coxam_participants,
    run_coxam_episode,
)
from src.virtual_experiment_executor.experiment_simualtion.CoXAM.coxam_trial_executor import (
    build_coxam_bundle,
    default_coxam_config,
    fit_coxam_surrogates,
    load_coxam_meta_policy,
    load_coxam_sub_policies,
)


def _trials(condition="decision_tree", participants=6, trials_each=8, complexity=None):
    rows = []
    for participant in range(1, participants + 1):
        for position in range(trials_each):
            row = {
                "participantId": participant,
                "instanceId": position,
                "xai_type": condition,
                "tested_w_xai": position % 2 == 0,
                "phase": "testing",
            }
            if complexity is not None:
                row["complexity"] = complexity
            rows.append(row)
    return pd.DataFrame(rows)


# -- the draw -------------------------------------------------------------


def test_forward_draw_is_filtered_to_the_condition(monkeypatch):
    from src.virtual_experiment_executor.participant_pools import load_pool

    draws = draw_coxam_participants(
        _trials("decision_tree"), pool_name="coxam_forward", app_id="wine_quality"
    )
    pool = load_pool("coxam_forward")
    assert len(draws) == 6
    for draw in draws.values():
        assert draw.parameter_source == "pool"
        row = pool[pool["pool_participant_id"] == draw.fitted_participant_id]
        assert "decision_tree" in set(row["condition"])


def test_forward_draw_stays_inside_the_trained_ranges():
    draws = draw_coxam_participants(
        _trials("linear_regression"), pool_name="coxam_forward", app_id="wine_quality"
    )
    for draw in draws.values():
        assert -1.0 <= draw.parameters["memory_recall_threshold"] <= 2.0
        assert 0.3 <= draw.parameters["decision_noise"] <= 0.7


def test_wine_quality_has_no_forward_opportunity_cost():
    """Only the mushrooms fit swept chi_value; wine keeps the runner's default."""
    wine = draw_coxam_participants(
        _trials("decision_tree"), pool_name="coxam_forward", app_id="wine_quality"
    )
    assert all("opportunity_cost" not in d.parameters for d in wine.values())

    mushrooms = draw_coxam_participants(
        _trials("decision_tree"), pool_name="coxam_forward", app_id="mushrooms"
    )
    assert any("opportunity_cost" in d.parameters for d in mushrooms.values())


def test_hybrid_wine_relaxes_to_that_datasets_other_fits():
    """The wine forward fits were run per family, so hybrid has no cell."""
    with pytest.warns(UserWarning, match="relaxed"):
        draws = draw_coxam_participants(
            _trials("hybrid"), pool_name="coxam_forward", app_id="wine_quality"
        )
    assert draws
    assert all(d.parameter_source.startswith("pool_relaxed") for d in draws.values())
    assert all(d.parameters for d in draws.values())


def test_counterfactual_draw_uses_the_replay_pool():
    draws = draw_coxam_participants(
        _trials("hybrid", complexity="high"),
        pool_name="coxam_counterfactual",
        app_id="wine_quality",
    )
    assert len(draws) == 6
    for draw in draws.values():
        assert draw.parameter_source == "pool"
        assert set(draw.parameters) == {
            "memory_recall_threshold",
            "counterfactual_overshoot_fraction",
            "time_penalty_weight",
        }


def test_draw_is_deterministic_and_seed_sensitive():
    trials = _trials("decision_tree")
    first = draw_coxam_participants(trials, pool_name="coxam_forward", app_id="mushrooms", seed=1)
    again = draw_coxam_participants(trials, pool_name="coxam_forward", app_id="mushrooms", seed=1)
    other = draw_coxam_participants(trials, pool_name="coxam_forward", app_id="mushrooms", seed=2)
    assert [d.fitted_participant_id for d in first.values()] == [
        d.fitted_participant_id for d in again.values()
    ]
    assert [d.fitted_participant_id for d in first.values()] != [
        d.fitted_participant_id for d in other.values()
    ]


def test_complexity_is_read_off_the_trials_when_the_design_declares_it():
    assert _trial_complexity(_trials(complexity="low")) == "low"
    assert _trial_complexity(_trials()) is None


def test_episode_participant_handles_both_grouping_shapes():
    rows = _trials(participants=1)
    assert _episode_participant((3, "block_a"), rows) == 3
    assert _episode_participant(7, rows) == 7
    assert _episode_participant(None, rows) == 1


# -- the environment actually reads them ----------------------------------


class _StubAIModel:
    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return (X[:, 0] > np.median(X[:, 0])).astype(int)


@pytest.fixture(scope="module")
def bundle():
    rng = np.random.default_rng(0)
    n_rows, n_features = 60, 4
    X = rng.normal(size=(n_rows, n_features))
    instance_ids = np.arange(n_rows)
    feature_names = [f"f{index}" for index in range(n_features)]
    surrogates = fit_coxam_surrogates(
        app_id="stub",
        model_name="mlp",
        X=X,
        instance_ids=instance_ids,
        trained_ai_model=_StubAIModel(),
        ai_model_input=X,
        feature_names=feature_names,
    )
    features = pd.DataFrame({"dataId": "stub", "instanceId": instance_ids})
    for index in range(n_features):
        features[f"x{index}"] = X[:, index]
    return build_coxam_bundle(
        app_id="stub",
        model_name="mlp",
        features=features,
        predictions=surrogates["predictions"],
        lr_explanations=surrogates["lr_explanations"],
        dt_explanations=surrogates["dt_explanations"],
        metadata=surrogates["metadata"],
    )


def _episode(bundle, params, seed=None):
    config = default_coxam_config()
    meta_policy, uses_action_masks = load_coxam_meta_policy()
    trials = _trials(participants=1, trials_each=12).assign(instanceId=range(12))
    return run_coxam_episode(
        trials,
        bundle=bundle,
        condition_name="decision_tree",
        meta_policy=meta_policy,
        uses_action_masks=uses_action_masks,
        sub_policies=load_coxam_sub_policies(config),
        config=config,
        fixed_eval_params=params,
        dvs={"forward_accuracy": ["continuous"]},
        seed=seed,
    )


def test_drawn_parameters_change_what_the_episode_answers(bundle):
    """Two real fitted participants, run on identical trials, must diverge.

    Without this the whole mode could be wired correctly end to end and still
    change nothing, because the parameters would reach an environment that never
    reads them.
    """
    low = _episode(
        bundle,
        {"memory_recall_threshold": -1.0, "decision_noise": 0.3, "opportunity_cost": 0.0},
        seed=1,
    )
    high = _episode(
        bundle,
        {"memory_recall_threshold": 2.0, "decision_noise": 0.7, "opportunity_cost": 0.02},
        seed=1,
    )
    assert not low["pred_time"].equals(high["pred_time"]) or not low[
        "agent_prediction"
    ].equals(high["agent_prediction"])


def test_episode_seed_separates_two_identical_participants(bundle):
    """Same parameters, same trials, different seed -> different noise draws."""
    params = {"memory_recall_threshold": 0.5, "decision_noise": 0.5, "opportunity_cost": 0.01}
    first = _episode(bundle, params, seed=1)
    second = _episode(bundle, params, seed=2)
    same_seed = _episode(bundle, params, seed=1)
    pd.testing.assert_series_equal(first["prob_correct"], same_seed["prob_correct"])
    assert not first["prob_correct"].equals(second["prob_correct"])


# -- the counterfactual env needs a COMPLETE parameter dict ----------------


def test_partial_cognitive_params_are_completed_for_the_counterfactual_env():
    """The env replaces its parameter dict rather than overlaying it.

    ``resolve_cognitive_params`` stores exactly the fixed values it is handed,
    and ``build_observation`` then reads every parameter the policy observes --
    so a partial dict (the fitted replay supplies three of the four; it never
    fitted ``random_response_rate``) made the env raise KeyError while building
    its observation. Caught only by running a real episode, which is why the
    draw-level tests above missed it.
    """
    from src.cognitive_models.cognitive_models.CoXAM.counterfactual_env import (
        DEFAULT_COGNITIVE_PARAMS,
    )
    from src.virtual_experiment_executor.experiment_simualtion.CoXAM.coxam_counterfactual_runner import (
        _complete_cognitive_params,
    )

    drawn = {
        "memory_recall_threshold": -1.0,
        "counterfactual_overshoot_fraction": 0.3,
        "time_penalty_weight": 0.01,
    }
    completed = _complete_cognitive_params(drawn)

    # Every parameter the env observes must be present...
    assert set(DEFAULT_COGNITIVE_PARAMS) <= set(completed)
    assert "random_response_rate" in completed
    # ...and the drawn values must survive the merge.
    for key, value in drawn.items():
        assert completed[key] == value
    assert _complete_cognitive_params(None) is None
    assert _complete_cognitive_params({}) is None


def test_a_drawn_counterfactual_row_is_complete_enough_for_the_env():
    """End of the chain: what the pool deals must be env-ready once completed."""
    from src.cognitive_models.cognitive_models.CoXAM.counterfactual_env import (
        DEFAULT_COGNITIVE_PARAMS,
    )
    from src.virtual_experiment_executor.experiment_simualtion.CoXAM.coxam_counterfactual_runner import (
        _complete_cognitive_params,
    )

    draws = draw_coxam_participants(
        _trials("decision_tree", complexity="high"),
        pool_name="coxam_counterfactual",
        app_id="wine_quality",
    )
    for draw in draws.values():
        completed = _complete_cognitive_params(draw.parameters)
        assert set(DEFAULT_COGNITIVE_PARAMS) <= set(completed)

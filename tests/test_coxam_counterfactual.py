"""CoXAM's counterfactual simulation is a separate trained agent.

Its environment was ported out of ``counterfactual_simulation_v0.3.ipynb``
(cells 9, 11, 12, 13, 16, 18), so these tests pin the port against the shipped
checkpoint: if the observation encoding, the action space or the strategy
legality rules drift, the policy would be fed inputs it was never trained on and
would silently produce meaningless choices.
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pytest

from src.cognitive_models.cognitive_models.CoXAM.counterfactual_env import (
    DEFAULT_TIME_PENALTY_WEIGHT_RANGE,
    STRATEGIES,
    XAI_TYPES,
    apply_change_to_feature,
    sample_from_probs,
    smooth_probs_with_random_response,
    strategy_is_legal,
)
from src.virtual_experiment_executor.experiment_simualtion.CoXAM.coxam_counterfactual_runner import (
    COUNTERFACTUAL_POLICY_PATH,
    load_counterfactual_policy,
)


def test_strategy_and_condition_vocabularies_match_the_notebook():
    assert STRATEGIES == {
        0: "change_path_dt",
        1: "zero_out_lr_heuristic",
        2: "zero_out_lr_displayed",
        3: "recall_change_dt",
        4: "recall_change_lr",
    }
    assert XAI_TYPES == {0: "DT", 1: "LR", 2: "DT+LR"}


def test_checkpoint_is_bundled_and_loads():
    assert COUNTERFACTUAL_POLICY_PATH.is_file()
    policy = load_counterfactual_policy()
    assert policy.observation_space.shape == (23,)
    assert list(policy.action_space.nvec) == [5, 3]


def test_time_penalty_range_matches_the_checkpoint_not_the_notebook():
    """The shipped weights were trained on [0, 0.02].

    The notebook's current ``training_cog_params`` says [0.0, 0.05], which
    post-dates the checkpoint. Using it would put the first observation
    dimension outside the policy's own ``observation_space``.
    """
    policy = load_counterfactual_policy()
    assert DEFAULT_TIME_PENALTY_WEIGHT_RANGE == (0.0, 0.02)
    assert float(policy.observation_space.low[0]) == pytest.approx(0.0)
    assert float(policy.observation_space.high[0]) == pytest.approx(0.02)


@pytest.mark.parametrize(
    "condition, strategy, with_xai, expected",
    [
        # A DT condition cannot run LR strategies, and vice versa.
        ("DT", "change_path_dt", 1, True),
        ("DT", "recall_change_dt", 0, True),
        ("DT", "zero_out_lr_heuristic", 1, False),
        ("DT", "recall_change_lr", 1, False),
        ("LR", "zero_out_lr_heuristic", 0, True),
        ("LR", "change_path_dt", 1, False),
        ("LR", "recall_change_dt", 1, False),
        # The "displayed" LR strategy needs the explanation actually on screen.
        ("LR", "zero_out_lr_displayed", 1, True),
        ("LR", "zero_out_lr_displayed", 0, False),
        # Hybrid shows both families, so everything is legal.
        ("DT+LR", "change_path_dt", 1, True),
        ("DT+LR", "recall_change_lr", 1, True),
    ],
)
def test_strategy_legality_rules(condition, strategy, with_xai, expected):
    assert strategy_is_legal(strategy, condition, with_xai) is expected


def test_sample_from_probs_returns_nothing_when_no_feature_is_feasible():
    """A recall strategy with an empty memory must not fabricate a change."""
    empty = {"a0": {"p_selected": 0.0, "mean_delta": 0.0},
             "a1": {"p_selected": 0.0, "mean_delta": 0.0}}
    feature, delta = sample_from_probs(empty, np.random.default_rng(0))
    assert feature is None
    assert delta == 0.0


def test_sample_from_probs_picks_the_only_feasible_feature():
    probs = {"a0": {"p_selected": 0.0, "mean_delta": 1.0},
             "a1": {"p_selected": 0.9, "mean_delta": -2.5},
             "expected_time": 3.0}
    feature, delta = sample_from_probs(probs, np.random.default_rng(0))
    assert feature == "a1"
    assert delta == -2.5


def test_random_response_smoothing_mixes_toward_uniform():
    probs = {"a0": {"p_selected": 1.0, "mean_delta": 0.0},
             "a1": {"p_selected": 0.0, "mean_delta": 0.0}}
    smoothed = smooth_probs_with_random_response(probs, random_response_rate=0.6, num_features=2)

    # 0.4*1.0 + 0.6/2 = 0.7 ; 0.4*0.0 + 0.6/2 = 0.3
    assert smoothed["a0"]["p_selected"] == pytest.approx(0.7)
    assert smoothed["a1"]["p_selected"] == pytest.approx(0.3)
    assert probs["a0"]["p_selected"] == 1.0, "must not mutate the caller's dict"


def test_applying_a_change_overshoots_then_clamps_to_bounds():
    bounds = {"a0": (0.0, 10.0)}
    x = np.array([5.0, 1.0])

    # +2 delta, plus 10% of the 10-wide range as overshoot -> 8.0
    changed = apply_change_to_feature(x, "a0", bounds, 2.0, 0.1)
    assert changed[0] == pytest.approx(8.0)
    assert changed[1] == 1.0, "other features untouched"

    # Overshoot pushes past the upper bound, so it clamps.
    clamped = apply_change_to_feature(x, "a0", bounds, 100.0, 0.1)
    assert clamped[0] == pytest.approx(10.0)

    # A negative delta overshoots downward.
    down = apply_change_to_feature(x, "a0", bounds, -2.0, 0.1)
    assert down[0] == pytest.approx(2.0)


def test_counterfactual_and_forward_are_distinct_agents():
    """They must not share a checkpoint -- different obs spaces entirely."""
    from src.virtual_experiment_executor.experiment_simualtion.CoXAM.coxam_trial_executor import (
        load_coxam_meta_policy,
    )

    forward, _uses_masks = load_coxam_meta_policy()
    counterfactual = load_counterfactual_policy()

    assert forward.observation_space.shape != counterfactual.observation_space.shape
    assert str(forward.action_space) != str(counterfactual.action_space)

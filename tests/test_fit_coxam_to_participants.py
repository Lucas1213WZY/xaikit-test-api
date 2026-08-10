"""Tests for the CoXAM forward fit to study participants."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.cognitive_models.cognitive_models.fit_coxam_to_participants import (
    CONDITION_ALIASES,
    LAPSE_RATE_VALUES,
    XAI_TYPE_ALIASES,
    CoxamCandidate,
    _model_probability_class_one,
    build_agent_comparison,
    candidate_grid,
    count_free_parameters,
    load_coxam_human_forward_trials,
    negative_log_likelihood,
)


def _human_row(**overrides):
    row = {
        "Participant Id": 1,
        "Phase": "forward",
        "Exclude": 0,
        "Response": -60.0,
        "AI prediction": 0,
        "DT prediction": 0,
        "LR prediction": 1,
        "Condition": "DT",
        "XAIType": "DT",
        "Tested w/ XAI": "w/ XAI",
        "Trial Index": 1,
        "Instance Id": 7,
        "dataId": "mushrooms",
        "Complexity": "high",
    }
    row.update(overrides)
    return row


# -- the response encoding ------------------------------------------------


def test_response_sign_is_the_chosen_class(tmp_path):
    """``Response`` is a signed confidence; only its sign names the class."""
    path = tmp_path / "trials.csv"
    pd.DataFrame(
        [
            _human_row(Response=-100.0),
            _human_row(Response=-20.0, **{"Trial Index": 2}),
            _human_row(Response=20.0, **{"Trial Index": 3}),
            _human_row(Response=100.0, **{"Trial Index": 4}),
        ]
    ).to_csv(path, index=False)

    trials = load_coxam_human_forward_trials(path=path)
    assert trials["human_label"].tolist() == [0, 0, 1, 1]


def test_a_zero_response_is_rejected_rather_than_guessed(tmp_path):
    """A 0 response has no sign, so silently calling it class 0 would be wrong."""
    path = tmp_path / "trials.csv"
    pd.DataFrame([_human_row(Response=0.0)]).to_csv(path, index=False)

    with pytest.raises(ValueError, match="no sign"):
        load_coxam_human_forward_trials(path=path)


def test_excluded_and_counterfactual_rows_are_dropped(tmp_path):
    path = tmp_path / "trials.csv"
    pd.DataFrame(
        [
            _human_row(),
            _human_row(Exclude=1, **{"Trial Index": 2}),
            _human_row(Phase="counterfactual", **{"Trial Index": 3}),
        ]
    ).to_csv(path, index=False)

    trials = load_coxam_human_forward_trials(path=path)
    assert len(trials) == 1
    assert trials["Trial Index"].tolist() == [1]


def test_an_unmapped_condition_is_rejected(tmp_path):
    path = tmp_path / "trials.csv"
    pd.DataFrame([_human_row(Condition="Ensemble")]).to_csv(path, index=False)

    with pytest.raises(ValueError, match="Unmapped CoXAM condition"):
        load_coxam_human_forward_trials(path=path)


# -- the two vocabularies -------------------------------------------------


def test_condition_and_explanation_family_use_different_lr_spellings():
    """CoXAM names the LR *condition* and the LR *explanation family* differently.

    Feeding either vocabulary's spelling to the other raises, so this pins the
    distinction rather than leaving it to be rediscovered at runtime.
    """
    assert CONDITION_ALIASES["LR"] == "linear_regression"
    assert XAI_TYPE_ALIASES["LR"] == "logistic_regression"
    assert CONDITION_ALIASES["LR"] != XAI_TYPE_ALIASES["LR"]


def test_the_explanation_family_spelling_is_one_coxam_accepts():
    """Every value we send as ``shown_xai_type`` must resolve, and the condition
    vocabulary's ``linear_regression`` must not -- feeding it here raises, which
    is exactly the mix-up this pair of constants exists to prevent."""
    from src.virtual_experiment_executor.experiment_simualtion.CoXAM.coxam_trial_executor import (
        _canonical_coxam_explanation_type,
    )

    resolved = {
        label: _canonical_coxam_explanation_type(label, shown=True)
        for label in XAI_TYPE_ALIASES.values()
    }
    assert resolved == {"decision_tree": "dt", "logistic_regression": "lr"}

    with pytest.raises(ValueError, match="Unknown CoXAM xai_method"):
        _canonical_coxam_explanation_type(CONDITION_ALIASES["LR"], shown=True)


# -- probability recovery -------------------------------------------------


def test_probability_is_inverted_when_the_ai_predicted_class_zero():
    """``prob_correct`` is P(the AI's class), not P(class 1)."""
    episode = pd.DataFrame({"prob_correct": [0.8, 0.8], "ai_label": [1, 0]})
    probability = _model_probability_class_one(episode)
    assert probability.tolist() == [0.8, pytest.approx(0.2)]


def test_a_missing_distribution_falls_back_to_chance():
    episode = pd.DataFrame({"prob_correct": [np.nan], "ai_label": [1]})
    assert _model_probability_class_one(episode).tolist() == [0.5]


# -- the likelihood -------------------------------------------------------


def test_nll_scores_the_participants_response_not_correctness():
    """A model certain of the AI's class still scores badly against a human who
    disagreed with the AI -- which is the whole point of fitting to people."""
    probability = np.array([1.0, 1.0])
    agreeing = negative_log_likelihood(probability, np.array([1, 1]), lapse_rate=0.1)
    disagreeing = negative_log_likelihood(probability, np.array([0, 0]), lapse_rate=0.1)
    assert agreeing < disagreeing


def test_lapse_keeps_a_degenerate_prediction_finite():
    """Without a lapse a [1, 0] distribution makes a disagreement infinite."""
    probability = np.array([1.0])
    assert np.isinf(negative_log_likelihood(probability, np.array([0]), lapse_rate=0.0))
    assert np.isfinite(negative_log_likelihood(probability, np.array([0]), lapse_rate=0.01))


def test_zero_is_not_searched_as_a_lapse_rate():
    """It can only ever lose, because it leaves those trials at infinity."""
    assert 0.0 not in LAPSE_RATE_VALUES
    assert min(LAPSE_RATE_VALUES) > 0.0


def test_chance_predictions_give_the_log_two_baseline():
    probability = np.full(8, 0.5)
    nll = negative_log_likelihood(probability, np.array([0, 1] * 4), lapse_rate=0.05)
    assert nll == pytest.approx(np.log(2))


# -- parameter counting ---------------------------------------------------


def test_pinning_a_parameter_stops_bic_charging_for_it():
    pinned = candidate_grid(decision_noises=(0.4,), memory_recall_thresholds=(0.5,), opportunity_costs=(0.0,))
    assert count_free_parameters(pinned, [0.05]) == 0

    one_free = candidate_grid(decision_noises=(0.3, 0.7), memory_recall_thresholds=(0.5,), opportunity_costs=(0.0,))
    assert count_free_parameters(one_free, [0.05]) == 1
    assert count_free_parameters(one_free, [0.05, 0.1]) == 2


def test_the_full_grid_charges_all_four_parameters():
    assert count_free_parameters(candidate_grid(), LAPSE_RATE_VALUES) == 4


def test_grid_covers_every_combination():
    grid = candidate_grid(decision_noises=(0.3, 0.7), memory_recall_thresholds=(-1.0, 2.0), opportunity_costs=(0.0,))
    assert len(grid) == 4
    assert len(set(grid)) == 4


def test_candidate_exposes_only_coxam_evaluation_parameters():
    """CoXAM's models must not be changed, so the fit may only set the three
    parameters the env leaves free at evaluation time."""
    params = CoxamCandidate(0.4, 0.5, 0.01).eval_params()
    assert set(params) == {"decision_noise", "memory_recall_threshold", "opportunity_cost"}


# -- the CoAX-shaped comparison table -------------------------------------


def test_agent_comparison_scores_every_agent_against_the_ai():
    predictions = pd.DataFrame(
        {
            "participant_id": [1, 1],
            "dataId": "mushrooms",
            "condition": "DT",
            "trial_index": [1, 2],
            "instanceId": [7, 8],
            "human_label": [1, 0],
            "model_label": [1, 1],
            "dt_label": [0, 0],
            "lr_label": [1, 1],
            "ai_label": [1, 0],
        }
    )
    comparison = build_agent_comparison(predictions)

    assert set(comparison["agent"]) == {"Human", "CoXAM", "DT", "LR"}
    assert len(comparison) == 8
    by_agent = comparison.groupby("agent")["Correct"].mean().to_dict()
    assert by_agent["Human"] == 1.0  # matched the AI on both trials
    assert by_agent["CoXAM"] == 0.5  # matched on trial 1 only
    assert by_agent["DT"] == 0.5  # matched on trial 2 only
    assert by_agent["LR"] == 0.5

import pytest

from src.cognitive_models import (
    explanation_property_simulator,
    feature_explanation_simulator,
    rules_weights_simulator,
)


def _trial(**trial_info):
    return {"trial_info": {"participantId": 1, "trialId": 1, **trial_info}, "ai_prediction": 1}


@pytest.mark.parametrize(
    "simulator,trial_data",
    [
        (explanation_property_simulator, _trial(validation_condition="feature_check", explanation_property="faithful")),
        (feature_explanation_simulator, _trial(xai_type="attribution", tested_w_xai=True)),
        (rules_weights_simulator, _trial(xai_type="decision_tree", tested_w_xai=True, task="forward_simulation")),
    ],
)
def test_study_simulator_contract(simulator, trial_data):
    result = simulator({"seed": 42}, {"forward_accuracy": ["continuous"]}, trial_data)
    assert result["agent_prediction"] in (0, 1)
    assert result["ai_prediction"] == 1
    assert result["forward_accuracy"] in (0, 1)
    assert 0 <= result["prob_correct"] <= 1
    assert result["pred_time"] > 0


def test_study_simulators_are_deterministic_per_trial():
    trial_data = _trial(xai_type="attribution", tested_w_xai=True)
    first = feature_explanation_simulator({"seed": 7}, {"forward_accuracy": ["continuous"]}, trial_data)
    second = feature_explanation_simulator({"seed": 7}, {"forward_accuracy": ["continuous"]}, trial_data)
    assert first == second

from types import SimpleNamespace

import pandas as pd

from src.cognitive_models import KNNBaseline
from src.statistical_analyst import analyze_iv_dv
from src.virtual_experiment_executor import (
    run_experiment_executor,
    run_virtual_experiment,
)


def _condition_experiment_inputs():
    raw_data = pd.DataFrame({
        "x": [0.0, 1.0, 0.1],
        "target": [0, 1, 0],
    })
    trials = [
        {"participantId": 1, "trialId": 1, "condition": "A", "phase": "training", "instanceId": 0},
        {"participantId": 1, "trialId": 2, "condition": "A", "phase": "testing", "instanceId": 2},
        {"participantId": 1, "trialId": 3, "condition": "B", "phase": "training", "instanceId": 1},
        {"participantId": 1, "trialId": 4, "condition": "B", "phase": "testing", "instanceId": 2},
    ]
    predictions = pd.DataFrame({
        "instanceId": [0, 1, 2],
        "pred": [0, 1, 0],
    })
    return raw_data, trials, predictions


def test_executor_fits_a_fresh_knn_for_each_participant_condition():
    raw_data, trials, predictions = _condition_experiment_inputs()

    responses = run_experiment_executor(
        trials,
        cognitive_params={},
        dvs={"forward_accuracy": ["continuous"]},
        raw_dataset=raw_data,
        explanation_pool=predictions,
        condition_columns=["condition"],
        cognitive_model=KNNBaseline(n_neighbors=1),
        label_column="target",
    )

    testing = responses.query("phase == 'testing'")
    assert testing["condition"].tolist() == ["A", "B"]
    assert testing["agent_prediction"].tolist() == [0, 1]


def test_virtual_executor_accepts_arrangement_and_saves_responses(tmp_path):
    raw_data, trials, predictions = _condition_experiment_inputs()

    result = run_virtual_experiment(
        SimpleNamespace(trials=trials),
        cognitive_params={},
        dvs={"forward_accuracy": ["continuous"]},
        raw_dataset=raw_data,
        explanation_pool=predictions,
        condition_columns=["condition"],
        cognitive_model=KNNBaseline(n_neighbors=1),
        label_column="target",
        output_dir=tmp_path,
    )

    assert len(result.responses) == 4
    assert (tmp_path / "simulated_results.csv").exists()
    assert (tmp_path / "simulated_results.json").exists()


def test_one_iv_one_dv_analysis_uses_testing_participant_means():
    responses = pd.DataFrame({
        "participantId": [1, 1, 2, 2, 3, 3, 4, 4, 1],
        "condition": ["A", "A", "A", "A", "B", "B", "B", "B", "A"],
        "phase": ["testing"] * 8 + ["training"],
        "accuracy": [1, 1, 1, 0, 0, 0, 0, 1, 99],
    })

    result = analyze_iv_dv(responses, iv="condition", dv="accuracy")

    assert result.method == "one_way_anova"
    assert len(result.participant_level_data) == 4
    assert result.descriptives.set_index("condition").loc["A", "mean"] == 0.75
    assert result.descriptives.set_index("condition").loc["B", "mean"] == 0.25

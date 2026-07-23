import numpy as np
import pandas as pd
import pytest
from types import SimpleNamespace

from src.cognitive_models import KNNBaseline, build_single_trial_cognitive_input
from src.api import xaikitTest
from src.experiment_planner import export_trials_csv, export_trials_json
from src.experiment_planner.trials import (
    _add_training_and_testing_phases,
    _balance_phase_instances_by_ai_prediction,
)
from src.virtual_experiment_executor import run_experiment_executor
from src.xai_adapter import ExplanationRunConfig, generate_ai_prediction_table


def test_knn_baseline_save_load_and_test(tmp_path):
    training = pd.DataFrame(
        {"age": [20, 22, 70], "income": [25, 28, 90]},
    )
    ai_predictions = [0, 0, 1]

    baseline = KNNBaseline(n_neighbors=5).fit(training, ai_predictions)
    model_path = baseline.save(tmp_path / "knn_proxy.joblib")

    loaded = KNNBaseline.load(model_path)
    predictions = loaded.predict(pd.DataFrame({"income": [26, 88], "age": [21, 72]}))

    np.testing.assert_array_equal(predictions, [0, 0])


def test_knn_baseline_accepts_one_training_instance():
    baseline = KNNBaseline(n_neighbors=5).fit([[1.0, 2.0]], [1])

    np.testing.assert_array_equal(baseline.predict([[100.0, -10.0]]), [1])
    assert baseline.score([[3.0, 4.0]], [1]) == 1.0


def test_knn_trains_feature_only_and_feature_plus_xai_models():
    baseline = KNNBaseline(n_neighbors=1).fit(
        pd.DataFrame({"feature": [0.0, 0.0]}),
        [0, 1],
        explanations=pd.DataFrame({"a0_i": [0.0, 1.0]}),
    )

    assert baseline.predict({"feature": 0.0})[0] == 0
    assert baseline.predict(
        {"feature": 0.0},
        explanations={"a0_i": 1.0},
    )[0] == 1

    without_xai = baseline(
        {},
        {"forward_accuracy": ["continuous"]},
        {
            "trial_info": {
                "phase": "testing",
                "xai_method": "shap",
                "tested_w_xai": False,
            },
            "instance_attributes": {"feature": 0.0},
            "instance_explanation": {"a0_i": 1.0},
            "ai_prediction": 1,
        },
    )
    with_xai = baseline(
        {},
        {"forward_accuracy": ["continuous"]},
        {
            "trial_info": {
                "phase": "testing",
                "xai_method": "shap",
                "tested_w_xai": True,
            },
            "instance_attributes": {"feature": 0.0},
            "instance_explanation": {"a0_i": 1.0},
            "ai_prediction": 1,
        },
    )

    assert without_xai["agent_prediction"] == 0
    assert with_xai["agent_prediction"] == 1


def test_explanation_ids_include_training_and_only_xai_visible_testing():
    study = xaikitTest("explanation_ids")
    study.trials = [
        {"instanceId": 1, "phase": "training", "xai_method": "shap", "tested_w_xai": False},
        {"instanceId": 2, "phase": "testing", "xai_method": "shap", "tested_w_xai": True},
        {"instanceId": 3, "phase": "testing", "xai_method": "shap", "tested_w_xai": False},
        {"instanceId": 4, "phase": "training", "xai_method": "none", "tested_w_xai": True},
        {"instanceId": 5, "phase": "training", "xai_method": "lime"},
        {"instanceId": 6, "phase": "testing", "xai_method": "lime", "tested_w_xai": True},
    ]

    assert study._trial_ids_requiring_explanations() == [1, 2, 5, 6]
    assert study._trial_ids_requiring_explanations_by_method() == {
        "shap": [1, 2],
        "lime": [5, 6],
    }


def test_knn_is_available_through_cognitive_model_api():
    study = xaikitTest("knn_api")

    study.set_cognitive_model(
        cognitive_model_id="knn",
        model_kwargs={"n_neighbors": 3},
    )

    assert isinstance(study.cognitive_model, KNNBaseline)
    assert study.cognitive_model.n_neighbors == 3
    assert study.cognitive_model_id == "knn"
    assert study.cognitive_params == {}


def test_knn_baseline_cognitive_model_adapter():
    baseline = KNNBaseline(n_neighbors=1).fit(
        pd.DataFrame({"x1": [0.0, 1.0], "x2": [0.0, 1.0]}),
        [0, 1],
    )

    result = baseline(
        {},
        {"forward_accuracy": ["continuous"]},
        {
            "instance_attributes": {"x2": 0.9, "x1": 0.9},
            "ai_prediction": 1,
        },
    )

    assert result["agent_prediction"] == 1
    assert result["ai_prediction"] == 1
    assert result["forward_accuracy"] == 1
    assert result["prob_correct"] == 1.0


def test_trial_phases_keep_training_and_testing_instances_separate():
    trials = [
        {"participantId": 1, "trialId": 1, "instanceId": "2", "dataId": "demo"},
        {"participantId": 1, "trialId": 2, "instanceId": "3", "dataId": "demo"},
    ]

    phased = _add_training_and_testing_phases(
        trials,
        train_instance_ids=[0, 1],
        dataset_id="demo",
        num_training=1,
        condition_columns=[],
        seed=42,
    )

    assert [trial["phase"] for trial in phased] == ["training", "testing", "testing"]
    assert int(phased[0]["instanceId"]) in {0, 1}
    assert {int(trial["instanceId"]) for trial in phased[1:]} == {2, 3}


def test_all_training_precedes_testing_and_test_iv_is_test_only():
    trials = [
        {
            "participantId": 1,
            "trialId": 1,
            "instanceId": "2",
            "dataId": "demo",
            "xai_method": "shap",
            "tested_w_xai": True,
        },
        {
            "participantId": 1,
            "trialId": 2,
            "instanceId": "3",
            "dataId": "demo",
            "xai_method": "shap",
            "tested_w_xai": False,
        },
        {
            "participantId": 1,
            "trialId": 3,
            "instanceId": "4",
            "dataId": "demo",
            "xai_method": "lime",
            "tested_w_xai": False,
        },
        {
            "participantId": 1,
            "trialId": 4,
            "instanceId": "5",
            "dataId": "demo",
            "xai_method": "lime",
            "tested_w_xai": True,
        },
    ]

    phased = _add_training_and_testing_phases(
        trials,
        train_instance_ids=[0, 1],
        dataset_id="demo",
        num_training=2,
        condition_columns=["xai_method"],
        test_only_columns=["tested_w_xai"],
        seed=42,
    )

    assert [row["phase"] for row in phased] == [
        "training",
        "training",
        "testing",
        "testing",
        "testing",
        "testing",
    ]
    assert [row["trialId"] for row in phased] == list(range(1, 7))
    assert all("tested_w_xai" not in row for row in phased[:2])
    assert [row["tested_w_xai"] for row in phased[2:]] == [
        True,
        False,
        False,
        True,
    ]


def test_prediction_balancing_is_half_half_and_randomized_within_each_phase():
    trials = []
    for participant_id in (1, 2):
        for trial_id in range(1, 5):
            trials.append({
                "participantId": participant_id,
                "trialId": trial_id,
                "phase": "training",
                "instanceId": "0",
            })
        for trial_id in range(5, 11):
            trials.append({
                "participantId": participant_id,
                "trialId": trial_id,
                "phase": "testing",
                "instanceId": "4",
                "tested_w_xai": bool(trial_id % 2),
            })

    predictions = {
        0: 0, 1: 0, 2: 1, 3: 1,
        4: 0, 5: 0, 6: 0, 7: 1, 8: 1, 9: 1,
    }
    balanced = _balance_phase_instances_by_ai_prediction(
        trials,
        train_instance_ids=[0, 1, 2, 3],
        test_instance_ids=[4, 5, 6, 7, 8, 9],
        predictions_by_instance=predictions,
        seed=42,
    )
    balanced_df = pd.DataFrame(balanced)

    counts = (
        balanced_df.groupby(["participantId", "phase"])
        ["sampled_ai_prediction"]
        .value_counts()
    )
    assert (counts.loc[(slice(None), "training", slice(None))] == 2).all()
    assert (counts.loc[(slice(None), "testing", slice(None))] == 3).all()
    assert set(
        balanced_df.query("phase == 'training'")["instanceId"].astype(int)
    ) <= {0, 1, 2, 3}
    assert set(
        balanced_df.query("phase == 'testing'")["instanceId"].astype(int)
    ) <= {4, 5, 6, 7, 8, 9}

    phase_sequences = balanced_df.groupby(
        ["participantId", "phase"], sort=False
    )["sampled_ai_prediction"].apply(list)
    assert any(sequence != sorted(sequence) for sequence in phase_sequences)


def test_trial_csv_supports_test_only_columns(tmp_path):
    trials = [
        {"participantId": 1, "trialId": 1, "phase": "training"},
        {
            "participantId": 1,
            "trialId": 2,
            "phase": "testing",
            "tested_w_xai": True,
        },
    ]

    csv_path = export_trials_csv(trials, tmp_path / "trials.csv")
    exported = pd.read_csv(csv_path)

    assert "tested_w_xai" in exported
    assert pd.isna(exported.loc[0, "tested_w_xai"])
    assert bool(exported.loc[1, "tested_w_xai"]) is True


def test_trial_json_converts_numpy_prediction_labels(tmp_path):
    trials = [{
        "participantId": 1,
        "instanceId": "7",
        "sampled_ai_prediction": np.int64(1),
    }]

    json_path = export_trials_json(trials, tmp_path / "trials.json")

    assert json_path.read_text().count('"sampled_ai_prediction": 1') == 1


def test_complete_prediction_table_is_saved_before_execution(tmp_path):
    class ThresholdModel:
        def predict(self, rows):
            return (np.asarray(rows)[:, 0] >= 0.5).astype(int)

    data = SimpleNamespace(
        dataset_id="demo",
        train_instance_ids=np.asarray([0, 2]),
        test_instance_ids=np.asarray([1, 3]),
        split=SimpleNamespace(
            X_model=np.asarray([[0.1], [0.8], [0.2], [0.9]])
        ),
    )
    config = ExplanationRunConfig(
        data=data,
        iv_config={"xai_method": {"levels": ["none"]}},
        trained_ai_model=ThresholdModel(),
        model_name="threshold",
        output_dir=tmp_path,
    )

    prediction_path, prediction_table = generate_ai_prediction_table(config)

    assert prediction_path.exists()
    assert prediction_table["instanceId"].tolist() == [0, 2, 1, 3]
    assert prediction_table["pred"].tolist() == [0, 0, 1, 1]
    assert prediction_table["expMethod"].eq("__prediction_only__").all()


def test_prediction_table_reuses_sampling_predictions_without_repredicting(tmp_path):
    class ModelThatMustNotRun:
        def predict(self, rows):
            raise AssertionError("Predictions used for sampling must be reused.")

    data = SimpleNamespace(
        dataset_id="demo",
        train_instance_ids=np.asarray([0, 2]),
        test_instance_ids=np.asarray([1, 3]),
        split=SimpleNamespace(
            X_model=np.asarray([[0.1], [0.8], [0.2], [0.9]])
        ),
    )
    config = ExplanationRunConfig(
        data=data,
        iv_config={"xai_method": {"levels": ["none"]}},
        trained_ai_model=ModelThatMustNotRun(),
        model_name="frozen_predictions",
        output_dir=tmp_path,
        predictions_by_instance={0: 1, 1: 0, 2: 1, 3: 0},
    )

    _, prediction_table = generate_ai_prediction_table(config)

    assert prediction_table["instanceId"].tolist() == [0, 2, 1, 3]
    assert prediction_table["pred"].tolist() == [1, 1, 0, 0]


def test_plot_explanation_without_instance_uses_generated_method_row(monkeypatch):
    study = xaikitTest("plot_generated_explanation")
    study.data = SimpleNamespace(raw_feature_names=["feature"])
    study.combined_explanations = pd.DataFrame({
        "expMethod": ["__prediction_only__", "shap"],
        "instanceId": [1, 42],
        "a0_i": [np.nan, 0.75],
    })

    monkeypatch.setattr(
        "src.xai_adapter.plot_explanation_visual",
        lambda explanation_df, data, **kwargs: (
            kwargs["instance_id"],
            kwargs["show_ai_prediction"],
        ),
    )

    selected_id, prediction_visible = study.plot_explanation(
        method="SHAP",
        instance_id=None,
        phase="testing",
    )

    assert selected_id == 42
    assert prediction_visible is False


def test_hidden_test_xai_is_removed_but_training_xai_is_kept():
    raw_dataset = pd.DataFrame({"feature": [0.25], "label": [1]})
    explanation_pool = pd.DataFrame({
        "instanceId": [0],
        "expMethod": ["shap"],
        "pred": [1],
        "a0_i": [0.75],
    })

    hidden_test = build_single_trial_cognitive_input(
        {
            "instanceId": 0,
            "phase": "testing",
            "xai_method": "shap",
            "tested_w_xai": False,
        },
        raw_dataset,
        explanation_pool,
        label_column="label",
    )
    training = build_single_trial_cognitive_input(
        {
            "instanceId": 0,
            "phase": "training",
            "xai_method": "shap",
            "tested_w_xai": False,
        },
        raw_dataset,
        explanation_pool,
        label_column="label",
    )

    assert hidden_test["instance_explanation"] == {}
    assert training["instance_explanation"]["a0_i"] == pytest.approx(0.75)


def test_trial_prediction_prefers_the_matching_explanation_method():
    raw_dataset = pd.DataFrame({"feature": [0.25], "label": [1]})
    explanation_pool = pd.DataFrame({
        "instanceId": [0, 0],
        "expMethod": ["shap", "lime"],
        "pred": [1, 0],
        "a0_i": [0.75, -0.25],
    })

    context = build_single_trial_cognitive_input(
        {
            "instanceId": 0,
            "phase": "training",
            "xai_method": "lime",
        },
        raw_dataset,
        explanation_pool,
        label_column="label",
    )

    assert context["ai_prediction"] == 0
    assert context["instance_explanation"]["a0_i"] == pytest.approx(-0.25)


def test_executor_fits_on_training_phase_then_infers_on_testing_phase():
    raw_data = pd.DataFrame({
        "x1": [0.0, 0.1, 0.05, 0.95],
        "x2": [0.0, 0.1, 0.05, 0.95],
        "target": [0, 1, 0, 1],
    })
    trials = [
        {"participantId": 1, "trialId": 1, "phase": "training", "instanceId": 0},
        {"participantId": 1, "trialId": 2, "phase": "training", "instanceId": 1},
        {"participantId": 1, "trialId": 3, "phase": "testing", "instanceId": 2},
        {"participantId": 1, "trialId": 4, "phase": "testing", "instanceId": 3},
    ]
    explanation_pool = pd.DataFrame({
        "instanceId": [0, 1, 2, 3],
        "pred": [0, 1, 0, 1],
    })

    results = run_experiment_executor(
        trials,
        cognitive_params={},
        dvs={"forward_accuracy": ["continuous"]},
        raw_dataset=raw_data,
        explanation_pool=explanation_pool,
        participant_id=1,
        mode="participant_by_participant",
        cognitive_model=KNNBaseline(n_neighbors=1),
        label_column="target",
    )

    assert results["phase"].tolist() == ["training", "training", "testing", "testing"]
    assert results.loc[:1, "agent_prediction"].isna().all()
    assert results.loc[2:, "agent_prediction"].tolist() == [0, 1]
    assert results.loc[2:, "cognitive_correct_vs_ai"].tolist() == [True, True]

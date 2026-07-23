import numpy as np
import pandas as pd
import pytest

from src.api import xaikitTest
from src.cognitive_models import (
    DecisionTreeBaseline,
    LogisticRegressionBaseline,
    MLPBaseline,
    create_baseline_model,
)
from src.virtual_experiment_executor import run_experiment_executor
from src.experiment_planner import load_support_matrix


BASELINE_CASES = [
    ("decision_tree", DecisionTreeBaseline, {}),
    ("logistic_regression", LogisticRegressionBaseline, {}),
    ("mlp", MLPBaseline, {"max_iter": 1000}),
]


def test_additional_baselines_are_registered_for_validation():
    registered = load_support_matrix()["cognitive_models"]

    assert {
        "decision_tree",
        "logistic_regression",
        "mlp_baseline",
    } <= set(registered)


def test_logistic_baseline_enforces_requested_coefficient_budget():
    baseline = LogisticRegressionBaseline(k=1).fit(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
        [0, 0, 1, 1],
    )

    logistic = baseline._model.named_steps["logisticregression"]
    assert np.count_nonzero(logistic.coef_) <= 1


def test_mlp_baseline_accepts_reference_style_parameter_names():
    baseline = MLPBaseline(
        hidden_dim=4,
        epochs=20,
        learning_rate=0.01,
    )

    assert baseline.hidden_layer_sizes == (4,)
    assert baseline.max_iter == 20
    assert baseline.learning_rate_init == pytest.approx(0.01)


@pytest.mark.parametrize("model_id,expected_class,kwargs", BASELINE_CASES)
def test_additional_baselines_fit_feature_and_xai_models(
    model_id,
    expected_class,
    kwargs,
):
    features = pd.DataFrame({"feature": [0.0, 0.0, 0.0, 0.0]})
    explanations = pd.DataFrame({"a0_i": [0.0, 0.0, 1.0, 1.0]})
    targets = [0, 0, 1, 1]

    baseline = create_baseline_model(model_id, **kwargs).fit(
        features,
        targets,
        explanations=explanations,
    )

    assert isinstance(baseline, expected_class)
    assert baseline.predict(
        {"feature": 0.0},
        explanations={"a0_i": 1.0},
    )[0] == 1
    assert baseline.predict_proba(
        {"feature": 0.0},
        explanations={"a0_i": 1.0},
    ).shape == (1, 2)


@pytest.mark.parametrize("model_id,expected_class,kwargs", BASELINE_CASES)
def test_additional_baselines_save_and_load(
    tmp_path,
    model_id,
    expected_class,
    kwargs,
):
    baseline = create_baseline_model(model_id, **kwargs).fit(
        [[0.0], [1.0]],
        [0, 1],
    )

    model_path = baseline.save(tmp_path / f"{model_id}.joblib")
    loaded = expected_class.load(model_path)

    np.testing.assert_array_equal(loaded.predict([[1.0]]), baseline.predict([[1.0]]))


@pytest.mark.parametrize(
    "requested_id,canonical_id,expected_class",
    [
        ("decision_tree", "decision_tree", DecisionTreeBaseline),
        ("lr", "logistic_regression", LogisticRegressionBaseline),
        ("mlp", "mlp_baseline", MLPBaseline),
    ],
)
def test_additional_baselines_are_available_through_cognitive_api(
    requested_id,
    canonical_id,
    expected_class,
):
    study = xaikitTest("baseline_api")

    study.set_cognitive_model(
        cognitive_model_id=requested_id,
        model_kwargs={"max_iter": 20} if requested_id == "mlp" else {},
    )

    assert isinstance(study.cognitive_model, expected_class)
    assert study.cognitive_model_id == canonical_id
    assert study.cognitive_params == {}


@pytest.mark.parametrize("model_id,_,kwargs", BASELINE_CASES)
def test_executor_trains_and_tests_additional_baselines(model_id, _, kwargs):
    raw_data = pd.DataFrame({
        "feature": [0.0, 0.0, 0.0, 0.0],
        "target": [0, 1, 0, 1],
    })
    trials = [
        {
            "participantId": 1,
            "trialId": 1,
            "phase": "training",
            "instanceId": 0,
            "xai_method": "shap",
        },
        {
            "participantId": 1,
            "trialId": 2,
            "phase": "training",
            "instanceId": 1,
            "xai_method": "shap",
        },
        {
            "participantId": 1,
            "trialId": 3,
            "phase": "testing",
            "instanceId": 2,
            "xai_method": "shap",
            "tested_w_xai": False,
        },
        {
            "participantId": 1,
            "trialId": 4,
            "phase": "testing",
            "instanceId": 3,
            "xai_method": "shap",
            "tested_w_xai": True,
        },
    ]
    explanation_pool = pd.DataFrame({
        "instanceId": [0, 1, 2, 3],
        "expMethod": ["shap"] * 4,
        "pred": [0, 1, 0, 1],
        "a0_i": [0.0, 1.0, 0.0, 1.0],
    })

    results = run_experiment_executor(
        trials,
        cognitive_params={},
        dvs={"forward_accuracy": ["continuous"]},
        raw_dataset=raw_data,
        explanation_pool=explanation_pool,
        cognitive_model=create_baseline_model(model_id, **kwargs),
        label_column="target",
    )

    testing = results.query("phase == 'testing'")
    assert testing["agent_prediction"].notna().all()
    assert testing["forward_accuracy"].notna().all()

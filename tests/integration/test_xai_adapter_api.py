"""Smoke tests for the XAI adapter API layer."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from src.data_loaders import XAIDatasetParser
from src.xai_adapter import (
    create_custom_xai_method,
    create_custom_surrogate_method,
    create_xai_method,
    create_xai_method_from_engine,
    generate_surrogate_xai_methods,
    get_adapter_registry,
    make_surrogate,
    register_xai_method,
)
from src.xai_adapter.attribution import KernelShap, LimeTabular, LeaveOneFeatureOut


def _predict_proba(x):
    x = np.asarray(x, dtype=float)
    p1 = 0.5 + 0.2 * x[:, 0] + 0.1 * x[:, 1]
    return np.column_stack([1.0 - p1, p1])


def _identity(x):
    return np.asarray(x, dtype=float)


def test_registry_exposes_expected_adapter_aliases():
    available = get_adapter_registry().list_available()

    assert "lofo" in available
    assert "shap" in available
    assert "lime" in available
    assert "deeplift" in available
    assert "integrated_gradients" in available
    assert "decision_tree" in available
    assert "logistic_regression" in available
    assert "rules" in available
    assert "weights" in available


def test_lofo_xai_method_returns_normalized_result():
    adapter = create_xai_method(
        "lofo",
        predict_fn=_predict_proba,
        background_data=np.array([[0.0, 0.0], [1.0, 1.0]]),
    )

    result = adapter.explain(np.array([[1.0, 0.0], [0.0, 1.0]]))

    assert result.method == "lofo"
    assert result.values.shape == (2, 2)
    assert result.base_values.shape == (2,)
    np.testing.assert_allclose(result.attributions, result.values)
    np.testing.assert_allclose(result.importances, np.abs(result.values))


def test_attribution_namespace_exposes_library_backed_methods():
    assert KernelShap.__name__ == "KernelShap"
    assert LimeTabular.__name__ == "LimeTabular"
    assert LeaveOneFeatureOut.__name__ == "LeaveOneFeatureOut"


def test_custom_callable_can_be_wrapped_as_xai_method():
    def custom_algorithm(x):
        return np.asarray(x, dtype=float) * 2.0

    adapter = create_custom_xai_method(custom_algorithm, method_name="double")
    result = adapter.explain(np.array([[1.0, -2.0]]))

    assert result.method == "double"
    np.testing.assert_allclose(result.values, np.array([[2.0, -4.0]]))
    np.testing.assert_allclose(result.importances, np.array([[2.0, 4.0]]))


def test_custom_callable_can_be_registered_in_global_registry():
    def custom_algorithm(x):
        return np.ones_like(np.asarray(x, dtype=float))

    register_xai_method("unit_test_custom", custom_algorithm, "unit_test_custom_alias")
    adapter = create_xai_method("unit_test_custom_alias")
    result = adapter.explain(np.array([[3.0, 4.0]]))

    assert result.method == "unit_test_custom"
    np.testing.assert_allclose(result.values, np.array([[1.0, 1.0]]))


def test_custom_surrogate_callable_pair_can_be_wrapped():
    fitted = {}

    def fit_fn(X, y, **kwargs):
        fitted["mean"] = float(np.mean(y))
        fitted["kwargs"] = kwargs

    def explain_fn(instances):
        x = np.asarray(instances, dtype=float)
        return x * fitted["mean"], np.array([fitted["mean"]])

    surrogate = make_surrogate(fit_fn, explain_fn, name="unit_test_surrogate")

    result = surrogate.fit(np.array([[1.0, 2.0]]), np.array([2.0]), alpha=0.5).explain(
        np.array([[3.0, 4.0]])
    )

    assert surrogate.is_fitted
    assert fitted["kwargs"] == {"alpha": 0.5}
    assert result.method == "unit_test_surrogate"
    np.testing.assert_allclose(result.values, np.array([[6.0, 8.0]]))
    np.testing.assert_allclose(result.base_values, np.array([2.0]))


def test_custom_surrogate_convenience_constructor_uses_surrogate_api():
    def fit_fn(X, y, **kwargs):
        return None

    def explain_fn(instances):
        return np.asarray(instances, dtype=float) + 1.0

    surrogate = create_custom_surrogate_method(fit_fn, explain_fn, method_name="plus_one_surrogate")
    result = surrogate.fit(np.array([[0.0]]), np.array([1.0])).explain(np.array([[2.0]]))

    assert result.method == "plus_one_surrogate"
    np.testing.assert_allclose(result.values, np.array([[3.0]]))


def test_create_xai_method_from_engine_supports_coax_style_inputs():
    engine = SimpleNamespace(predict=_predict_proba)
    train_data = SimpleNamespace(X=np.array([[0.0, 0.0], [1.0, 1.0]]))

    adapter = create_xai_method_from_engine(
        "lofo",
        engine=engine,
        train_data=train_data,
        preprocessing_fn=_identity,
    )

    result = adapter.explain(np.array([[1.0, 0.0]]))
    assert result.values.shape == (1, 2)


def test_xai_dataset_parser_and_precomputed_csv_method_do_not_require_intercept():
    import pandas as pd

    dataset = XAIDatasetParser.from_dataframe(
        dataframe=pd.DataFrame(
            [
                {
                    "instanceId": 0,
                    "pred": 1,
                    "v0": 0.2,
                    "v1": 0.7,
                    "v2": 0.1,
                    "a0_i": 0.4,
                    "a1_i": 0.5,
                    "a2_i": 0.1,
                }
            ]
        ),
    )
    adapter = create_xai_method("csv", dataset=dataset)

    record = dataset.get_records([0])[0]
    result = adapter.explain([0])

    assert record.intercept == 0.0
    assert record.ai_prediction == 1
    np.testing.assert_allclose(record.features, np.array([0.2, 0.7, 0.1]))
    np.testing.assert_allclose(record.explanation, np.array([0.4, 0.5, 0.1]))
    np.testing.assert_allclose(result.base_values, np.array([0.0]))


def test_generate_surrogate_xai_methods_from_prediction_dataframe():
    import pytest

    pytest.importorskip("pandas")
    pytest.importorskip("sklearn")
    import pandas as pd

    generated = generate_surrogate_xai_methods(
        dataframe=pd.DataFrame(
            [
                {"instanceId": 0, "pred": 0, "v0": 0.0, "v1": 0.0, "v2": 0.0},
                {"instanceId": 1, "pred": 0, "v0": 0.2, "v1": 0.1, "v2": 0.0},
                {"instanceId": 2, "pred": 1, "v0": 0.9, "v1": 0.8, "v2": 1.0},
                {"instanceId": 3, "pred": 1, "v0": 1.0, "v1": 0.9, "v2": 1.0},
            ]
        ),
        app_id="toy",
        model_name="external",
        top_k=2,
    )

    assert {"rules", "weights"} <= set(generated.methods)
    assert generated.decision_tree_df is not None
    assert generated.logistic_regression_df is not None
    assert generated.metadata_df.iloc[0]["appId"] == "toy"

    X = np.array([[0.95, 0.85, 1.0]])
    assert generated.methods["rules"].explain(X).values.shape == (1, 3)
    assert generated.methods["weights"].explain(X).values.shape == (1, 3)


def test_generate_surrogate_xai_methods_accepts_xai_dataset_parser():
    import pytest

    pytest.importorskip("pandas")
    pytest.importorskip("sklearn")
    import pandas as pd

    dataset = XAIDatasetParser.from_dataframe(
        pd.DataFrame(
            [
                {"instanceId": 0, "pred": 0, "v0": 0.0, "v1": 0.0, "v2": 0.0},
                {"instanceId": 1, "pred": 0, "v0": 0.2, "v1": 0.1, "v2": 0.0},
                {"instanceId": 2, "pred": 1, "v0": 0.9, "v1": 0.8, "v2": 1.0},
                {"instanceId": 3, "pred": 1, "v0": 1.0, "v1": 0.9, "v2": 1.0},
            ]
        ),
        missing_explanation_strategy="zeros",
    )

    generated = generate_surrogate_xai_methods(
        dataset=dataset,
        instance_ids=[0, 1, 2, 3],
        app_id="toy_parser",
        model_name="external",
        top_k=2,
    )

    X = dataset.get_features([0, 1, 2, 3])
    assert generated.decision_tree_df.iloc[0]["appId"] == "toy_parser"
    assert generated.logistic_regression_df.iloc[0]["appId"] == "toy_parser"
    assert generated.methods["rules"].explain(X).values.shape == X.shape
    assert generated.methods["weights"].explain(X).values.shape == X.shape


def test_surrogate_methods_fit_generated_data_with_sklearn_style_api():
    import pytest

    pytest.importorskip("sklearn")

    X = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.2, 0.1, 0.0],
            [0.9, 0.8, 1.0],
            [1.0, 0.9, 1.0],
        ],
        dtype=float,
    )
    y = np.array([0, 0, 1, 1])

    rules = create_xai_method("rules", app_id="toy", model_name="external", depth=2).fit(X, y)
    weights = create_xai_method(
        "weights",
        app_id="toy",
        model_name="external",
        variant="sparse",
        top_k=2,
    ).fit(X, y)

    assert rules.explain(X).values.shape == X.shape
    assert weights.explain(X).values.shape == X.shape
    assert "tree_structure" in rules.to_explanation_table().columns
    assert any(col.startswith("coef_") for col in weights.to_explanation_table().columns)

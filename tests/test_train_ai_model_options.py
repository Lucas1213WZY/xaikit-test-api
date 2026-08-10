"""Tests for train_AI_model's named parameters and the prediction contract."""

from __future__ import annotations

import numpy as np
import pytest

from src.ai_models import predicted_labels, predicted_probabilities
from src.api import xaikitTest


def _study() -> xaikitTest:
    return xaikitTest("options")


# -- routing the named parameters ----------------------------------------


def test_named_parameters_route_to_constructor_or_trainer():
    build, train = _study()._split_training_options(
        "mlp", epochs=50, hidden_dimension=64, dropout_rate=0.1
    )
    assert build == {"hidden_dimension": 64, "dropout_rate": 0.1}
    assert train == {"epochs": 50}


def test_unset_parameters_are_left_out_rather_than_passed_as_none():
    """Forwarding None would override the engine's own default with nothing."""
    build, train = _study()._split_training_options("mlp", epochs=None, hidden_dimension=None)
    assert build == {} and train == {}


@pytest.mark.parametrize(
    "model_type, option, owner",
    [
        ("xgboost", "epochs", "mlp"),
        ("mlp", "learning_rate", "xgboost"),
        ("sim2real", "hidden_dimension", "mlp"),
        ("mlp", "function_name", "sim2real"),
    ],
)
def test_a_parameter_meant_for_another_model_raises(model_type, option, owner):
    """Silently dropping it would leave the caller thinking it took effect."""
    with pytest.raises(TypeError, match=option):
        _study()._split_training_options(model_type, **{option: 1})


def test_the_error_names_the_model_type_that_owns_the_parameter():
    with pytest.raises(TypeError, match="belongs to"):
        _study()._split_training_options("xgboost", epochs=10)


def test_an_unknown_model_type_lists_the_known_ones():
    with pytest.raises(TypeError, match="mlp, mlp_tf, sim2real, xgboost"):
        _study()._split_training_options("resnet", epochs=1)


def test_every_model_type_declares_both_groups():
    for model_type, groups in xaikitTest._TRAINING_OPTIONS.items():
        assert set(groups) == {"build", "train"}, model_type


# -- the prediction contract ---------------------------------------------


def test_labels_from_a_two_column_probability_matrix():
    assert predicted_labels(np.array([[0.9, 0.1], [0.2, 0.8]])).tolist() == [0, 1]


def test_labels_from_a_single_column_of_probabilities():
    assert predicted_labels(np.array([0.9, 0.1])).tolist() == [1, 0]


def test_labels_from_hard_integer_predictions():
    assert predicted_labels(np.array([1, 0, 1])).tolist() == [1, 0, 1]


def test_every_shape_gives_the_same_label_type():
    """The whole point: one return type, so no caller branches on ndim."""
    shapes = [
        np.array([[0.9, 0.1], [0.2, 0.8]]),
        np.array([0.1, 0.8]),
        np.array([0, 1]),
    ]
    results = [predicted_labels(value) for value in shapes]
    assert all(result.ndim == 1 for result in results)
    assert all(np.issubdtype(result.dtype, np.integer) for result in results)
    assert all(result.tolist() == [0, 1] for result in results)


def test_the_threshold_moves_the_single_column_cut():
    scores = np.array([0.6])
    assert predicted_labels(scores, threshold=0.5).tolist() == [1]
    assert predicted_labels(scores, threshold=0.7).tolist() == [0]


def test_probabilities_widen_a_single_column_to_two():
    widened = predicted_probabilities(np.array([0.9, 0.1]))
    assert widened.shape == (2, 2)
    assert widened[0].tolist() == pytest.approx([0.1, 0.9])


def test_probabilities_pass_a_matrix_through():
    matrix = np.array([[0.9, 0.1], [0.2, 0.8]])
    assert predicted_probabilities(matrix).tolist() == matrix.tolist()


def test_hard_labels_have_no_probabilities_to_return():
    with pytest.raises(ValueError, match="hard class labels"):
        predicted_probabilities(np.array([1, 0]))


def test_continuous_regression_output_is_not_treated_as_probability():
    """The sim2real analytical functions include regressors, not just classifiers."""
    with pytest.raises(ValueError, match="not"):
        predicted_probabilities(np.array([3.7, -12.5]))


def test_predict_helpers_are_exported_from_the_package():
    import src.ai_models as ai_models

    assert hasattr(ai_models, "predicted_labels")
    assert hasattr(ai_models, "predicted_probabilities")

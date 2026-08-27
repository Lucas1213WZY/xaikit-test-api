"""The dataset stage must not accept a constant, majority-class classifier.

Regression: a CoXAM counterfactual design on wine_quality trained an MLP with
the server's default ``target_score=0.85``. wine_quality is 86.4% one class, so
a model predicting that class for every instance scores 0.8644 -- the target was
reached at 20 epochs with ``reached_target: True`` and a model that had learned
nothing. The failure surfaced only much later, inside CoXAM's surrogate fitting
("the trained AI model predicts only class [0]"), advising "train the model
longer", which is not the cause and does not fix it.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from server.pipeline import (
    _effective_target_score,
    _majority_class_rate,
    _reject_single_class_model,
)
from server.schemas import DatasetStageRequest

# 86.4% class 0, like wine_quality's training split (829 / 130).
IMBALANCED = np.array([0] * 829 + [1] * 130)
BALANCED = np.array([0] * 500 + [1] * 500)


def test_majority_class_rate_is_the_accuracy_a_constant_model_scores():
    assert _majority_class_rate(IMBALANCED) == pytest.approx(0.8644, abs=1e-4)
    assert _majority_class_rate(BALANCED) == pytest.approx(0.5)
    assert _majority_class_rate(np.array([])) is None


def test_a_target_below_the_majority_rate_is_raised_above_it():
    request = DatasetStageRequest()  # accuracy, 0.85
    score, note = _effective_target_score(request, IMBALANCED)

    assert score > _majority_class_rate(IMBALANCED)
    assert score == pytest.approx(0.8983, abs=1e-4)
    assert note and "majority-class rate" in note


def test_a_target_that_already_clears_the_floor_is_left_alone():
    request = DatasetStageRequest(target_score=0.95)
    score, note = _effective_target_score(request, IMBALANCED)
    assert score == 0.95
    assert note is None


def test_a_balanced_dataset_is_left_alone():
    """0.85 is a real target when chance is 0.5 -- do not touch it."""
    score, note = _effective_target_score(DatasetStageRequest(), BALANCED)
    assert score == 0.85
    assert note is None


def test_balanced_accuracy_is_left_alone_even_on_imbalanced_data():
    """A constant predictor scores 0.5 on it, so the target is not satisfiable."""
    request = DatasetStageRequest(target_metric="balanced_accuracy", target_score=0.65)
    score, note = _effective_target_score(request, IMBALANCED)
    assert score == 0.65
    assert note is None


def test_no_target_at_all_is_left_alone():
    score, note = _effective_target_score(DatasetStageRequest(target_score=None), IMBALANCED)
    assert score is None
    assert note is None


class _ConstantModel:
    def predict(self, X):
        return np.zeros(len(X), dtype=int)


class _RealModel:
    def predict(self, X):
        return (np.arange(len(X)) % 2).astype(int)


def _study(model):
    return SimpleNamespace(
        data=SimpleNamespace(
            split=SimpleNamespace(X_model=np.zeros((20, 3)), y_train=IMBALANCED)
        ),
        trained_ai_model=model,
    )


def test_a_constant_model_is_rejected_where_it_was_trained():
    with pytest.raises(ValueError) as excinfo:
        _reject_single_class_model(_study(_ConstantModel()))
    message = str(excinfo.value)
    # The message has to name the cause and a way out, not just the symptom.
    assert "predicts only class [0]" in message
    assert "86.4%" in message
    assert "xgboost" in message and "balanced_accuracy" in message


def test_a_model_that_separates_the_classes_passes():
    _reject_single_class_model(_study(_RealModel()))


def test_the_rejection_names_the_dataset_in_a_multi_dataset_study():
    with pytest.raises(ValueError, match="for dataset 'wine_quality'"):
        _reject_single_class_model(_study(_ConstantModel()), "wine_quality")


def test_nothing_to_check_is_not_an_error():
    """A corpus-covered dataset trains no model; that is not a failure."""
    _reject_single_class_model(SimpleNamespace(data=None, trained_ai_model=None))
    _reject_single_class_model(_study(None))

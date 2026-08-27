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
    # A quarter of the headroom is 0.8983, but a lifted target never goes below
    # the 0.90 floor.
    assert score == pytest.approx(0.90)
    assert note and "majority-class rate" in note


def test_a_lifted_target_never_drops_below_the_floor_constant():
    """0.90 is the minimum; a barely-cleared target still accepts a weak model."""
    from server.pipeline import _MINIMUM_LIFTED_TARGET

    for labels in (IMBALANCED, np.array([0] * 88 + [1] * 12)):
        score, _ = _effective_target_score(DatasetStageRequest(), labels)
        assert score >= _MINIMUM_LIFTED_TARGET


def test_a_floor_above_the_minimum_wins_over_it():
    """95% one class needs more than 0.90 to be meaningful."""
    labels = np.array([0] * 950 + [1] * 50)
    score, _ = _effective_target_score(DatasetStageRequest(target_score=0.85), labels)
    assert score > _majority_class_rate(labels)
    assert score == pytest.approx(0.9625, abs=1e-4)


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


# -- CoXAM trains on the corpus's own feature set and order ----------------


def _selection(framework="coxam", datasets=("wine_quality",), **kwargs):
    from server.pipeline import _corpus_feature_selection

    return _corpus_feature_selection(DatasetStageRequest(**kwargs), framework, list(datasets))


def test_coxam_defaults_to_the_corpus_feature_set_and_order():
    """The corpus's a0..a5 are positional, so order matters as much as spelling."""
    from src.virtual_experiment_executor.experiment_simualtion.CoXAM.coxam_trial_executor import (
        coxam_loader_feature_cols,
    )

    cols, rank, note = _selection()
    assert cols == coxam_loader_feature_cols("wine_quality")
    # Ranking would reorder the six by target correlation, losing the positions.
    assert rank is False
    assert note and "positional" in note


def test_the_default_selection_would_otherwise_drop_a_corpus_feature():
    """prepare_dataset's default picks 5 of the corpus's 6 for wine_quality."""
    cols, _rank, _note = _selection()
    assert len(cols) == 6
    assert "chlorides" in cols


def test_an_explicit_feature_choice_is_never_overridden():
    for kwargs in ({"feature_cols": ["Alcohol", "pH"]}, {"num_features": 4}):
        cols, rank, note = _selection(**kwargs)
        assert cols == kwargs.get("feature_cols")
        assert note is None
        assert rank is DatasetStageRequest(**kwargs).rank_features_by_target


@pytest.mark.parametrize("framework", ["coax", "sim2real", "baseline"])
def test_other_frameworks_are_left_alone(framework):
    cols, _rank, note = _selection(framework=framework)
    assert cols is None
    assert note is None


def test_a_dataset_the_corpus_does_not_cover_is_left_alone():
    cols, _rank, note = _selection(datasets=("breast_cancer",))
    assert cols is None
    assert note is None


def test_two_datasets_with_different_corpus_features_are_left_alone():
    """One feature_cols cannot serve two different corpus feature sets."""
    cols, _rank, note = _selection(datasets=("wine_quality", "mushrooms"))
    assert cols is None
    assert note is None

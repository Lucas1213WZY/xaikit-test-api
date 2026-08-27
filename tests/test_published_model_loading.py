"""The dataset stage loads the published AI rather than training a new one.

A model trained fresh on the raw dataset is not the AI the human participants
faced. wine_quality is 86.4% one class, so a fresh MLP predicts the minority
class for ~7% of instances where the corpus AI predicts it for ~50% -- and
counterfactual simulation, which asks whether a proposed change flips the
prediction, then has almost nothing to flip (measured: 0.05-0.15 accuracy
against 0.45-0.65 with the published weights).
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from types import SimpleNamespace

import numpy as np
import pytest

from server.pipeline import (
    _load_published_model,
    _published_weights_path,
    _should_try_published_model,
)
from server.schemas import DatasetStageRequest


# -- which runs prefer the published model --------------------------------


@pytest.mark.parametrize(
    "dataset_id", ["wine_quality", "mushrooms", "forest_cover", "adult"]
)
def test_coxam_prefers_the_published_model_on_its_own_datasets(dataset_id):
    assert _should_try_published_model(DatasetStageRequest(), "coxam", dataset_id) is True


@pytest.mark.parametrize("dataset_id", ["prima_diabetes", "german_credit", "breast_cancer"])
def test_a_dataset_outside_the_published_study_still_trains(dataset_id):
    """Weights are shipped for these, but no CoXAM study ever ran them.

    Loading is justified by "this is the AI the human participants faced",
    which is only true for the corpus's own datasets. For anything else the
    checkpoint is simply some other model, and the design wants its own.
    """
    assert _should_try_published_model(DatasetStageRequest(), "coxam", dataset_id) is False


@pytest.mark.parametrize("framework", ["coax", "sim2real", "baseline"])
def test_other_frameworks_train_as_before(framework):
    """CoAX's corpus path already avoids training; nothing else changes."""
    assert _should_try_published_model(DatasetStageRequest(), framework, "wine_quality") is False


@pytest.mark.parametrize("framework", ["coxam", "coax"])
def test_an_explicit_choice_wins_either_way(framework):
    assert _should_try_published_model(
        DatasetStageRequest(use_published_model=True), framework, "prima_diabetes"
    ) is True
    assert _should_try_published_model(
        DatasetStageRequest(use_published_model=False), framework, "wine_quality"
    ) is False


# -- locating the checkpoint ----------------------------------------------


def test_the_shipped_coxam_checkpoint_is_found():
    path = _published_weights_path("mlp", "coxam", "wine_quality")
    assert path is not None and path.is_file()
    assert path.name == "wine_quality_model_weights.pth"


def test_a_dataset_with_no_published_weights_returns_none():
    assert _published_weights_path("mlp", "coxam", "not_a_dataset") is None


# -- refusing to load against the wrong features --------------------------


def _study_with_width(width: int):
    return SimpleNamespace(
        data=SimpleNamespace(
            dataset_id="wine_quality",
            split=SimpleNamespace(X_model=np.zeros((10, width)), y_train=np.array([0, 1])),
        ),
        trained_ai_model=None,
    )


def test_a_feature_width_mismatch_trains_instead_and_says_why():
    """The weights are positional: the wrong feature set is silent nonsense.

    Measured directly -- feeding the published wine_quality weights a
    differently-ordered or differently-sized feature set agreed with the
    corpus's own predictions 50-51% of the time, i.e. chance for a binary task,
    with no error raised anywhere.
    """
    # The shipped wine_quality checkpoint expects 6 features.
    note = _load_published_model(_study_with_width(5), "wine_quality", "mlp", "coxam")
    assert note is not None
    assert "expect 6 features" in note and "5" in note
    assert "positional" in note


def test_no_weights_means_no_note_and_no_load():
    study = _study_with_width(6)
    assert _load_published_model(study, "not_a_dataset", "mlp", "coxam") is None
    assert study.trained_ai_model is None


# -- the real thing -------------------------------------------------------


@pytest.fixture(scope="module")
def loaded_study():
    from src.api import xaikitTest

    study = xaikitTest(output_dir=os.environ.get("PYTEST_TMPDIR", "/tmp/xaikit_published"))
    study.prepare_dataset(
        dataset_id="wine_quality",
        cognitive_model_id="coxam",  # routes to the corpus's 6-feature set
        show_available=False,
        show_summary=False,
    )
    note = _load_published_model(study, "wine_quality", "mlp", "coxam")
    return study, note


def test_loading_sets_the_same_state_training_would(loaded_study):
    study, note = loaded_study
    assert note and "Loaded the published" in note
    # Every slot train_AI_model fills, so later stages cannot tell the difference.
    assert study.trained_ai_model is not None
    assert study.model is not None
    assert study.model_manager is not None
    assert study.model_name == "mlp"
    assert study.model_source == "coxam"
    assert study.training_info["trained"] is False


def test_the_loaded_model_predicts_both_classes_far_more_evenly(loaded_study):
    """The point of the whole exercise: a model with something to flip."""
    from src.ai_models.model_api import _labels_and_scores_from_predictions

    study, _note = loaded_study
    labels, _ = _labels_and_scores_from_predictions(
        study.trained_ai_model.predict(study.data.split.X_model)
    )
    labels = np.asarray(labels)
    assert set(labels.tolist()) == {0, 1}
    minority = min(np.mean(labels == c) for c in (0, 1))
    # A freshly trained MLP sits at 2-7% here; the corpus AI is ~50% on its own
    # curated instances and ~29% across the full dataset.
    assert minority > 0.15, f"minority share {minority:.1%} is back to degenerate"

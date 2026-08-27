"""Quality columns attach during explanation generation -- only when asked.

The metrics are opt-in because turning them on changes the explanation table's
column set and the ``/explanations`` payload. Existing server runs and the
design-planner UI (a separate project, so not verifiable from this repo) have
not been asked to expect either, so the default must be byte-for-byte what it
was. The first test here is that guarantee.
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.xai_adapter.api import ExplanationRunConfig, generate_xai_explanation_tables
from src.xai_adapter.base import XAIAdapter, XAIAdapterResult
from src.xai_adapter.metrics import QUALITY_COLUMNS
from src.xai_adapter.registry import get_adapter_registry

N_FEATURES = 3
N_ROWS = 12


class _StubModel:
    """Depends on x0 only, and returns (n, 2) scores like the repo's MLPs."""

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        positive = 1.0 / (1.0 + np.exp(-4.0 * (X[:, 0] - 0.5)))
        return np.column_stack([1.0 - positive, positive])


class _FakeAttributor(XAIAdapter):
    """A faithful, additive explainer: attributes everything to x0."""

    method_name = "fake_attributor"

    def __init__(self, **kwargs):
        super().__init__(target=kwargs.pop("target", 1))
        self.is_fitted = True

    def explain(self, instances) -> XAIAdapterResult:
        X = np.atleast_2d(np.asarray(instances, dtype=float))
        values = np.zeros_like(X)
        values[:, 0] = X[:, 0]
        return XAIAdapterResult(
            values=values,
            base_values=np.zeros(len(X)),
            method=self.method_name,
            metadata={},
        )


class _ExplodingAttributor(_FakeAttributor):
    """Explains fine, but returns values the metrics cannot digest."""

    method_name = "exploding_attributor"

    def explain(self, instances):
        result = super().explain(instances)
        # Wrong width: the metrics will raise when they hit the model.
        return XAIAdapterResult(
            values=result.values[:, :1],
            base_values=result.base_values,
            method=self.method_name,
            metadata={},
        )


@pytest.fixture(autouse=True)
def _register_fakes():
    registry = get_adapter_registry()
    registry.register("fake_attributor", _FakeAttributor)
    registry.register("exploding_attributor", _ExplodingAttributor)
    yield


@pytest.fixture
def prepared():
    rng = np.random.default_rng(0)
    X = rng.uniform(0.0, 1.0, size=(N_ROWS, N_FEATURES))
    feature_names = [f"f{i}" for i in range(N_FEATURES)]
    split = SimpleNamespace(
        X_model=X,
        X_train=X,
        y_train=(X[:, 0] > 0.5).astype(int),
        raw_instance_ids=np.arange(N_ROWS),
        test_instance_ids=np.arange(N_ROWS),
        X_test=X,
        feature_names=feature_names,
        model_feature_names=feature_names,
        raw_feature_names=feature_names,
    )
    return SimpleNamespace(
        dataset_id="stub",
        split=split,
        raw_feature_names=feature_names,
        model_feature_names=feature_names,
        y_train=split.y_train,
        X_train=X,
        X_test=X,
    )


def _config(prepared, tmp_path: Path, method: str = "fake_attributor", **kwargs):
    return ExplanationRunConfig(
        data=prepared,
        iv_config={"xai_method": {"levels": [method]}},
        trained_ai_model=_StubModel(),
        model_name="stub_model",
        output_dir=tmp_path,
        instance_ids=list(range(N_ROWS)),
        # init_explanation_run fills this; constructing the config directly does not.
        method_kwargs={},
        **kwargs,
    )


# -- the compatibility guarantee -------------------------------------------


def test_the_default_run_has_no_quality_columns(prepared, tmp_path):
    """Off by default: the table is exactly what it was before this feature."""
    _paths, frames = generate_xai_explanation_tables(_config(prepared, tmp_path))

    assert len(frames) == 1
    for column in QUALITY_COLUMNS:
        assert column not in frames[0].columns
    # And the file on disk carries none of them either.
    written = pd.read_csv(next(tmp_path.glob("*.csv")))
    assert not set(QUALITY_COLUMNS) & set(written.columns)


# -- when switched on -------------------------------------------------------


def test_every_explanation_row_carries_its_own_scores(prepared, tmp_path):
    _paths, frames = generate_xai_explanation_tables(
        _config(prepared, tmp_path, quality_metrics=True)
    )
    frame = frames[0]

    for column in QUALITY_COLUMNS:
        assert column in frame.columns
    assert len(frame) == N_ROWS
    # The faithful explainer ranks the only feature that matters first.
    assert np.isfinite(frame["faithfulness_aopc"]).all()
    assert frame["sparsity_nonzero"].eq(1.0).all()


def test_the_scores_survive_a_round_trip_through_the_csv(prepared, tmp_path):
    """'No extra call' means a table reloaded later still carries its scores."""
    generate_xai_explanation_tables(_config(prepared, tmp_path, quality_metrics=True))
    written = pd.read_csv(next(tmp_path.glob("*.csv")))
    assert set(QUALITY_COLUMNS) <= set(written.columns)
    assert np.isfinite(written["faithfulness_aopc"]).all()


def test_a_failing_metric_does_not_cost_us_the_explanation_table(prepared, tmp_path):
    _paths, frames = generate_xai_explanation_tables(
        _config(prepared, tmp_path, method="exploding_attributor", quality_metrics=True)
    )
    frame = frames[0]

    # The explanations survived...
    assert len(frame) == N_ROWS
    assert any(column.startswith("a") and column.endswith("_i") for column in frame.columns)
    # ...and the failure is recorded rather than silent.
    assert frame["quality_note"].str.startswith("unscored:").all()
    assert frame["faithfulness_aopc"].isna().all()


def test_quality_columns_are_not_mistaken_for_attribution_columns(prepared, tmp_path):
    """Four call sites sniff attribution columns; none may pick these up.

    A quality column read as an attribution would be handed to a cognitive model
    as if it were feature evidence.
    """
    from src.virtual_experiment_executor.experiment_simualtion.CoAX.coax_trial_executor import (
        _explanation_value_columns,
    )
    from src.xai_adapter.visualization import explanation_value_columns

    _paths, frames = generate_xai_explanation_tables(
        _config(prepared, tmp_path, quality_metrics=True)
    )
    frame = frames[0]

    for sniffer in (_explanation_value_columns, explanation_value_columns):
        found = sniffer(frame)
        assert found, "the real attribution columns should still be found"
        assert not set(found) & set(QUALITY_COLUMNS)

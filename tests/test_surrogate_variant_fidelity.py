"""Each surrogate variant must report its own fidelity.

``generate_logistic_regression_table`` fitted one dense logistic regression and
copied that fit's accuracy onto every variant row, so the sparse variant -- which
keeps only the top-k coefficients -- claimed the dense model's agreement with the
AI. That number is not internal bookkeeping: CoXAM renders it to the simulated
participant as ``Fidelity: {x:.4f}``, so a three-feature explanation was
advertised as matching the AI exactly as well as the six-feature one.

The published corpus's own generator (``fit_logistic_row`` in
``CoXAM/dataset_generator/generate_datasets.py``) refits on the kept columns and
scores that fit, which is what this now matches.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.xai_adapter.surrogate.generator import generate_logistic_regression_table


def _separable_data(n=400, n_features=6, seed=0):
    """Two informative features plus noise, so dropping features must cost accuracy."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, n_features))
    logit = 3.0 * X[:, 0] + 2.0 * X[:, 1] + 0.6 * X[:, 2] + 0.4 * X[:, 3]
    y = (logit + rng.normal(scale=0.5, size=n) > 0).astype(int)
    return X, y


def _table(**kwargs):
    X, y = _separable_data()
    return generate_logistic_regression_table(
        X, y, app_id="synthetic", model_name="stub", top_k=2, **kwargs
    ).set_index("variant")


def test_the_sparse_variant_no_longer_borrows_the_dense_fidelity():
    table = _table()
    assert table.loc["sparse", "fidelity"] != pytest.approx(table.loc["dense", "fidelity"])


def test_dropping_informative_features_cannot_improve_agreement_with_the_ai():
    table = _table()
    assert table.loc["sparse", "fidelity"] <= table.loc["dense", "fidelity"]


def test_the_dense_fidelity_is_unchanged():
    """The dense row is the pre-existing behaviour and must stay put."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score

    X, y = _separable_data()
    X_model = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    reference = LogisticRegression(C=1.0, random_state=0, max_iter=1000).fit(X_model, y)
    expected = accuracy_score(y, reference.predict(X_model))
    assert _table().loc["dense", "fidelity"] == pytest.approx(expected)


def test_the_reported_fidelity_is_the_accuracy_of_the_reported_coefficients():
    """The row must describe one model: its own coefficients, its own score."""
    from sklearn.metrics import accuracy_score

    X, y = _separable_data()
    X_model = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    table = generate_logistic_regression_table(
        X, y, app_id="synthetic", model_name="stub", top_k=2
    ).set_index("variant")

    for variant in ("dense", "sparse"):
        row = table.loc[variant]
        kept = [
            int(name.removeprefix("coef_a"))
            for name in table.columns
            if name.startswith("coef_a") and not np.isnan(row[name])
        ]
        margin = row["intercept"] + X_model[:, kept] @ np.array([row[f"coef_a{i}"] for i in kept])
        assert accuracy_score(y, (margin > 0).astype(int)) == pytest.approx(row["fidelity"])


def test_the_sparse_row_still_keeps_top_k_features():
    table = _table()
    assert table.loc["sparse", "nnz"] == 2
    assert len(json.loads(table.loc["sparse", "kept_groups"])) == 2

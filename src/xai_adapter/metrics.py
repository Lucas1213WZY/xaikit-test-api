"""Property metrics for feature-attribution explanations."""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np


def ensure_2d(values) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        return array.reshape(1, -1)
    return array


def sparsity_loss(W, *, atol: float = 1e-12) -> np.ndarray:
    """Count non-zero attribution weights per instance."""
    weights = ensure_2d(W)
    return np.sum(np.abs(weights) > atol, axis=1).astype(float)


def _predict(model_or_fn, X: np.ndarray) -> np.ndarray:
    if callable(model_or_fn) and not hasattr(model_or_fn, "predict"):
        return np.asarray(model_or_fn(X))
    return np.asarray(model_or_fn.predict(X))


def fidelity_regression_loss(
    model_or_fn,
    X,
    W,
    *,
    base_values: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Squared reconstruction error between f(x) and base + x^T w."""
    X_arr = ensure_2d(X)
    W_arr = ensure_2d(W)
    base = np.zeros(X_arr.shape[0], dtype=float) if base_values is None else np.asarray(base_values, dtype=float)
    true_values = _predict(model_or_fn, X_arr).reshape(-1)
    reconstructed = base.reshape(-1) + np.sum(X_arr * W_arr, axis=1)
    return (true_values - reconstructed) ** 2


def fidelity_classification_loss(
    model_or_fn,
    X,
    W,
    *,
    base_values: Optional[np.ndarray] = None,
) -> np.ndarray:
    """0/1 reconstruction error between f(x) and I(base + x^T w > 0)."""
    X_arr = ensure_2d(X)
    W_arr = ensure_2d(W)
    base = np.zeros(X_arr.shape[0], dtype=float) if base_values is None else np.asarray(base_values, dtype=float)
    true_labels = _predict(model_or_fn, X_arr).reshape(-1).astype(int)
    reconstructed = (base.reshape(-1) + np.sum(X_arr * W_arr, axis=1) > 0.0).astype(int)
    return (true_labels != reconstructed).astype(float)


def robustness_loss(X, W, *, radius: float) -> np.ndarray:
    """Local stability loss from nearby instance-explanation pairs.

    Returns one loss per instance, using the maximum ratio among neighbors
    within ``radius``. Instances with no neighbors receive 0.
    """
    X_arr = ensure_2d(X)
    W_arr = ensure_2d(W)
    losses = np.zeros(X_arr.shape[0], dtype=float)
    for i in range(X_arr.shape[0]):
        diffs = X_arr - X_arr[i]
        distances = np.linalg.norm(diffs, axis=1)
        mask = (distances > 0.0) & (distances <= radius)
        if not np.any(mask):
            continue
        weight_distances = np.linalg.norm(W_arr[mask] - W_arr[i], axis=1)
        losses[i] = float(np.max(weight_distances / distances[mask]))
    return losses


def summarize_property_metrics(
    model_or_fn,
    X,
    W,
    *,
    output_type: str,
    base_values: Optional[np.ndarray] = None,
    radius: float = 1.0,
) -> dict[str, float]:
    """Return mean sparsity, faithfulness loss, and robustness loss."""
    if output_type == "regression":
        faithfulness = fidelity_regression_loss(model_or_fn, X, W, base_values=base_values)
    else:
        faithfulness = fidelity_classification_loss(model_or_fn, X, W, base_values=base_values)
    return {
        "faithfulness_loss": float(np.mean(faithfulness)),
        "sparsity_loss": float(np.mean(sparsity_loss(W))),
        "robustness_loss": float(np.mean(robustness_loss(X, W, radius=radius))),
    }

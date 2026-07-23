"""Synthetic data-generating functions with known ground-truth attributions.

These are closed-form, deterministic analytical functions rather than trained
estimators: because the function is known exactly, so are its per-instance
feature attributions. That makes them the reference point for benchmarking XAI
methods against truth. They implement the same public shape as the other
engines where practical, and additionally expose ground-truth local weights
used by property-optimized explanations.

The concrete functions implemented here are taken from the XAIsim2real paper.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


def _ensure_2d(values) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        return array.reshape(1, -1)
    return array


@dataclass(frozen=True)
class Sim2RealSpec:
    """Metadata for a deterministic sim2real function."""

    function_name: str
    input_dim: int
    output_type: str
    feature_names: tuple[str, ...]


class BaseSim2RealFunction:
    """Base class for deterministic sim2real functions."""

    spec: Sim2RealSpec

    @property
    def function_name(self) -> str:
        return self.spec.function_name

    @property
    def input_dim(self) -> int:
        return self.spec.input_dim

    @property
    def output_type(self) -> str:
        return self.spec.output_type

    @property
    def feature_names(self) -> tuple[str, ...]:
        return self.spec.feature_names

    def predict(self, X) -> np.ndarray:
        raise NotImplementedError

    def ground_truth_weights(self, X) -> np.ndarray:
        raise NotImplementedError(f"{self.function_name} does not expose ground-truth weights")

    def ground_truth_base_values(self, X) -> np.ndarray:
        return np.zeros(_ensure_2d(X).shape[0], dtype=float)

    def trend_weights(self) -> np.ndarray:
        raise NotImplementedError(f"{self.function_name} does not expose global trend weights")

    def evaluate(self, X, y) -> float:
        preds = self.predict(X)
        y_arr = np.asarray(y)
        if self.output_type == "regression":
            return float(np.mean((preds.reshape(-1) - y_arr.reshape(-1)) ** 2))
        return float(np.mean(preds.reshape(-1).astype(int) == y_arr.reshape(-1).astype(int)))

    def get_info(self) -> Dict:
        return {
            "model_type": "sim2real",
            "function_name": self.function_name,
            "input_dim": self.input_dim,
            "output_type": self.output_type,
            "feature_names": list(self.feature_names),
            "is_trained": True,
        }


class SparseFunction(BaseSim2RealFunction):
    """Paper function f_sparse/fbox with UI-facing four-feature inputs.

    Appendix prose defines a 3D function, but the user-study UI and Table 1
    show four measurements/weights. The fourth feature is kept as a displayed
    measurement but does not select the active decision feature.
    """

    spec = Sim2RealSpec(
        "sparse",
        4,
        "classification",
        ("core_temperature", "glow_level", "antenna_length", "hearing_score"),
    )

    def _active_feature_indices(self, X: np.ndarray) -> np.ndarray:
        X = _ensure_2d(X)
        x3 = X[:, 2]
        use_x2 = (x3 <= 0.25) | ((x3 > 0.5) & (x3 <= 0.75))
        return np.where(use_x2, 1, 0)

    def predict(self, X) -> np.ndarray:
        X = _ensure_2d(X)
        active = self._active_feature_indices(X)
        values = X[np.arange(X.shape[0]), active]
        return (values > 0.5).astype(int)

    def ground_truth_weights(self, X) -> np.ndarray:
        X = _ensure_2d(X)
        weights = np.zeros((X.shape[0], self.input_dim), dtype=float)
        active = self._active_feature_indices(X)
        weights[np.arange(X.shape[0]), active] = 1.0
        return weights

    def ground_truth_base_values(self, X) -> np.ndarray:
        return np.full(_ensure_2d(X).shape[0], -0.5, dtype=float)


class TrendWiggleFunction(BaseSim2RealFunction):
    """Paper function f_trend+wiggle: linear trend plus sinusoidal wiggles.

    The appendix text is internally inconsistent, but the user-study UI for the
    counterfactual task shows seven measurements/weights. We use seven
    UI-facing dimensions and pad the printed trend weights with zeros.
    """

    spec = Sim2RealSpec(
        "trend_wiggle",
        7,
        "regression",
        (
            "core_temperature",
            "pulse_rate",
            "antenna_length",
            "glow",
            "hearing_score",
            "skin_moisture",
            "eye_reflex",
        ),
    )
    _trend = np.array([20.0, -1.0, -20.0, 1.0, 0.0, 0.0, 0.0], dtype=float)

    def predict(self, X) -> np.ndarray:
        X = _ensure_2d(X)
        return np.sum(5.0 * np.sin(20.0 * X) + self._trend * X, axis=1)

    def trend_weights(self) -> np.ndarray:
        return self._trend.copy()

    def ground_truth_weights(self, X) -> np.ndarray:
        X = _ensure_2d(X)
        return 100.0 * np.cos(20.0 * X) + self._trend


class WiggleFunction(BaseSim2RealFunction):
    """Paper function f_wiggle/fpiece with UI-facing eleven-feature inputs."""

    spec = Sim2RealSpec(
        "wiggle",
        11,
        "classification",
        (
            "core_temperature",
            "pulse_rate",
            "antenna_length",
            "glow",
            "hearing_score",
            "skin_moisture",
            "eye_reflex",
            "limb_flexibility",
            "tentacle_reflex",
            "brainwave_activity",
            "neural_sync",
        ),
    )
    weight_matrix = np.array(
        [
            [0.0, 1.0, -1.0, 0.0, 1.0, -0.1, 0.1, -0.1, 0.1, -0.1, -0.7],
            [0.0, -0.8, -0.2, 0.2, 0.1, -0.9, -0.1, -0.1, 0.1, -0.2, 1.0],
            [0.0, -0.8, -0.2, 0.0, 0.1, -0.9, -0.1, -0.1, 0.1, -0.2, 1.0],
            [0.0, -0.05, 1.0, -0.8, -0.1, 0.1, 0.9, -0.2, 0.1, 0.8, -1.0],
        ],
        dtype=float,
    )

    def _row_indices(self, X: np.ndarray) -> np.ndarray:
        X = _ensure_2d(X)
        x1 = X[:, 0]
        return np.select(
            [x1 <= 0.25, (x1 > 0.25) & (x1 <= 0.5), (x1 > 0.5) & (x1 <= 0.75)],
            [0, 1, 2],
            default=3,
        )

    def predict(self, X) -> np.ndarray:
        X = _ensure_2d(X)
        weights = self.ground_truth_weights(X)
        return (np.sum(X * weights, axis=1) > 0.0).astype(int)

    def ground_truth_weights(self, X) -> np.ndarray:
        rows = self._row_indices(_ensure_2d(X))
        return self.weight_matrix[rows].copy()


_FUNCTIONS = {
    "box": SparseFunction,
    "fbox": SparseFunction,
    "sparse": SparseFunction,
    "fsparse": SparseFunction,
    "trend_wiggle": TrendWiggleFunction,
    "trend+wiggle": TrendWiggleFunction,
    "ftrend+wiggle": TrendWiggleFunction,
    "piece": WiggleFunction,
    "piecewise": WiggleFunction,
    "fpiece": WiggleFunction,
    "wiggle": WiggleFunction,
    "fwiggle": WiggleFunction,
}


def create_sim2real_function(function_name: str) -> BaseSim2RealFunction:
    """Create a deterministic sim2real function by name."""
    key = function_name.lower().strip()
    if key not in _FUNCTIONS:
        raise ValueError(f"Unknown sim2real function '{function_name}'. Choose from {sorted(_FUNCTIONS)}")
    return _FUNCTIONS[key]()

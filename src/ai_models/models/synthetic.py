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


# Fixed Adult one-hot schema used by the local sim2real corpus.  The order is
# part of the model contract: attribution/importance columns a0-a66 and value
# columns v0-v66 use these exact dimensions.
ADULT_SIM2REAL_FEATURE_NAMES = (
    "age",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "education_10th",
    "education_11th",
    "education_12th",
    "education_1st-4th",
    "education_5th-6th",
    "education_7th-8th",
    "education_9th",
    "education_Assoc-acdm",
    "education_Assoc-voc",
    "education_Bachelors",
    "education_Doctorate",
    "education_HS-grad",
    "education_Masters",
    "education_Preschool",
    "education_Prof-school",
    "education_Some-college",
    "marital_Divorced",
    "marital_Married-AF-spouse",
    "marital_Married-civ-spouse",
    "marital_Married-spouse-absent",
    "marital_Never-married",
    "marital_Separated",
    "marital_Widowed",
    "native-country_United-States",
    "native-country_other",
    "native-country_unknown",
    "occupation_Adm-clerical",
    "occupation_Armed-Forces",
    "occupation_Craft-repair",
    "occupation_Exec-managerial",
    "occupation_Farming-fishing",
    "occupation_Handlers-cleaners",
    "occupation_Machine-op-inspct",
    "occupation_Other-service",
    "occupation_Priv-house-serv",
    "occupation_Prof-specialty",
    "occupation_Protective-serv",
    "occupation_Sales",
    "occupation_Tech-support",
    "occupation_Transport-moving",
    "occupation_unknown",
    "race_Amer-Indian-Eskimo",
    "race_Asian-Pac-Islander",
    "race_Black",
    "race_Other",
    "race_White",
    "relationship_Husband",
    "relationship_Not-in-family",
    "relationship_Other-relative",
    "relationship_Own-child",
    "relationship_Unmarried",
    "relationship_Wife",
    "sex_Female",
    "sex_Male",
    "workclass_Federal-gov",
    "workclass_Local-gov",
    "workclass_Never-worked",
    "workclass_Private",
    "workclass_Self-emp-inc",
    "workclass_Self-emp-not-inc",
    "workclass_State-gov",
    "workclass_Without-pay",
    "workclass_unknown",
)
ADULT_SIM2REAL_NUMERIC_FEATURE_INDICES = (0, 1, 2, 3)
ADULT_SIM2REAL_CATEGORICAL_FEATURE_INDICES = tuple(range(4, 67))


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
    model_name = "synthetic_ai"

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

    def _validated_input(self, X) -> np.ndarray:
        array = _ensure_2d(X)
        if array.ndim != 2:
            raise ValueError(
                f"{self.__class__.__name__} expects a 1D row or 2D matrix; "
                f"received shape {array.shape}"
            )
        if array.shape[1] != self.input_dim:
            raise ValueError(
                f"{self.__class__.__name__} expects {self.input_dim} features "
                f"in its declared order; received {array.shape[1]}"
            )
        return array

    def predict(self, X) -> np.ndarray:
        raise NotImplementedError

    def ground_truth_weights(self, X) -> np.ndarray:
        raise NotImplementedError(f"{self.function_name} does not expose ground-truth weights")

    def ground_truth_base_values(self, X) -> np.ndarray:
        return np.zeros(self._validated_input(X).shape[0], dtype=float)

    def predict_proba(self, X) -> np.ndarray:
        """Return two-class probabilities for LIME-compatible classifiers.

        These deterministic classifiers produce hard probabilities because the
        paper functions are closed-form decision rules, not calibrated models.
        """
        if self.output_type != "classification":
            raise ValueError(f"{self.function_name} is a regression function")
        labels = self.predict(X).reshape(-1).astype(int)
        return np.column_stack((1 - labels, labels)).astype(float)

    def trend_weights(self) -> np.ndarray:
        raise NotImplementedError(f"{self.function_name} does not expose global trend weights")

    def evaluate(self, X, y) -> float:
        preds = self.predict(X)
        y_arr = np.asarray(y)
        if self.output_type == "regression":
            return float(np.mean((preds.reshape(-1) - y_arr.reshape(-1)) ** 2))
        return float(np.mean(preds.reshape(-1).astype(int) == y_arr.reshape(-1).astype(int)))

    def get_info(self) -> Dict:
        info = {
            "model_type": "sim2real",
            "model_name": self.model_name,
            "function_name": self.function_name,
            "input_dim": self.input_dim,
            "output_type": self.output_type,
            "feature_names": list(self.feature_names),
            "is_trained": True,
        }
        if hasattr(self, "variant_name"):
            info["variant_name"] = self.variant_name
        if hasattr(self, "active_feature_indices"):
            active = list(self.active_feature_indices)
            info["active_feature_indices"] = active
            info["active_feature_names"] = [self.feature_names[index] for index in active]
        if hasattr(self, "numeric_feature_indices"):
            info["numeric_feature_indices"] = list(self.numeric_feature_indices)
        if hasattr(self, "categorical_feature_indices"):
            info["categorical_feature_indices"] = list(self.categorical_feature_indices)
        return info


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
        X = self._validated_input(X)
        x3 = X[:, 2]
        use_x2 = (x3 <= 0.25) | ((x3 > 0.5) & (x3 <= 0.75))
        return np.where(use_x2, 1, 0)

    def predict(self, X) -> np.ndarray:
        X = self._validated_input(X)
        active = self._active_feature_indices(X)
        values = X[np.arange(X.shape[0]), active]
        return (values > 0.5).astype(int)

    def ground_truth_weights(self, X) -> np.ndarray:
        X = self._validated_input(X)
        weights = np.zeros((X.shape[0], self.input_dim), dtype=float)
        active = self._active_feature_indices(X)
        weights[np.arange(X.shape[0]), active] = 1.0
        return weights

    def ground_truth_base_values(self, X) -> np.ndarray:
        return np.full(self._validated_input(X).shape[0], -0.5, dtype=float)


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
        X = self._validated_input(X)
        return np.sum(5.0 * np.sin(20.0 * X) + self._trend * X, axis=1)

    def trend_weights(self) -> np.ndarray:
        return self._trend.copy()

    def ground_truth_weights(self, X) -> np.ndarray:
        X = self._validated_input(X)
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
        X = self._validated_input(X)
        x1 = X[:, 0]
        return np.select(
            [x1 <= 0.25, (x1 > 0.25) & (x1 <= 0.5), (x1 > 0.5) & (x1 <= 0.75)],
            [0, 1, 2],
            default=3,
        )

    def predict(self, X) -> np.ndarray:
        X = self._validated_input(X)
        weights = self.ground_truth_weights(X)
        return (np.sum(X * weights, axis=1) > 0.0).astype(int)

    def ground_truth_weights(self, X) -> np.ndarray:
        rows = self._row_indices(self._validated_input(X))
        return self.weight_matrix[rows].copy()


class AdultSparseFunction(SparseFunction):
    """Sparse paper function embedded in the 67D Adult sim2real schema.

    The original rule is preserved on the first three normalized numeric
    dimensions: capital-loss selects whether age or capital-gain is active.
    All one-hot dimensions are valid model inputs and receive zero analytical
    weight when inactive.
    """

    spec = Sim2RealSpec(
        "sparse",
        len(ADULT_SIM2REAL_FEATURE_NAMES),
        "classification",
        ADULT_SIM2REAL_FEATURE_NAMES,
    )
    variant_name = "adult_income"
    active_feature_indices = (0, 1, 2)
    numeric_feature_indices = ADULT_SIM2REAL_NUMERIC_FEATURE_INDICES
    categorical_feature_indices = ADULT_SIM2REAL_CATEGORICAL_FEATURE_INDICES


class AdultTrendWiggleFunction(TrendWiggleFunction):
    """Trend-and-wiggle function embedded in selected Adult dimensions."""

    spec = Sim2RealSpec(
        "trend_wiggle",
        len(ADULT_SIM2REAL_FEATURE_NAMES),
        "regression",
        ADULT_SIM2REAL_FEATURE_NAMES,
    )
    variant_name = "adult_income"
    active_feature_indices = (0, 1, 2, 3, 13, 22, 56)
    numeric_feature_indices = ADULT_SIM2REAL_NUMERIC_FEATURE_INDICES
    categorical_feature_indices = ADULT_SIM2REAL_CATEGORICAL_FEATURE_INDICES

    def predict(self, X) -> np.ndarray:
        X = self._validated_input(X)
        active_values = X[:, self.active_feature_indices]
        return np.sum(5.0 * np.sin(20.0 * active_values) + self._trend * active_values, axis=1)

    def trend_weights(self) -> np.ndarray:
        trend = np.zeros(self.input_dim, dtype=float)
        trend[list(self.active_feature_indices)] = self._trend
        return trend

    def ground_truth_weights(self, X) -> np.ndarray:
        X = self._validated_input(X)
        active_values = X[:, self.active_feature_indices]
        weights = np.zeros((X.shape[0], self.input_dim), dtype=float)
        weights[:, self.active_feature_indices] = 100.0 * np.cos(20.0 * active_values) + self._trend
        return weights


class AdultWiggleFunction(WiggleFunction):
    """Piecewise wiggle classifier embedded in selected Adult dimensions."""

    spec = Sim2RealSpec(
        "wiggle",
        len(ADULT_SIM2REAL_FEATURE_NAMES),
        "classification",
        ADULT_SIM2REAL_FEATURE_NAMES,
    )
    variant_name = "adult_income"
    active_feature_indices = (0, 1, 2, 3, 13, 15, 22, 33, 39, 49, 56)
    numeric_feature_indices = ADULT_SIM2REAL_NUMERIC_FEATURE_INDICES
    categorical_feature_indices = ADULT_SIM2REAL_CATEGORICAL_FEATURE_INDICES

    def predict(self, X) -> np.ndarray:
        X = self._validated_input(X)
        weights = self.ground_truth_weights(X)
        return (np.sum(X * weights, axis=1) > 0.0).astype(int)

    def ground_truth_weights(self, X) -> np.ndarray:
        X = self._validated_input(X)
        rows = self._row_indices(X)
        weights = np.zeros((X.shape[0], self.input_dim), dtype=float)
        weights[:, self.active_feature_indices] = self.weight_matrix[rows]
        return weights


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
    "adult": AdultSparseFunction,
    "adult_income": AdultSparseFunction,
    "adult_sparse": AdultSparseFunction,
    "adult_trend_wiggle": AdultTrendWiggleFunction,
    "adult_trend+wiggle": AdultTrendWiggleFunction,
    "adult_wiggle": AdultWiggleFunction,
    "adult_piecewise": AdultWiggleFunction,
}


def create_sim2real_function(function_name: str) -> BaseSim2RealFunction:
    """Create a deterministic sim2real function by name.

    Args:
        function_name: Which synthetic function to build, e.g. ``sparse``,
            ``wiggle``, ``trend_wiggle`` or their 67D Adult variants
            ``adult_sparse``, ``adult_wiggle`` and ``adult_trend_wiggle``.
    """
    key = function_name.lower().strip()
    if key not in _FUNCTIONS:
        raise ValueError(f"Unknown sim2real function '{function_name}'. Choose from {sorted(_FUNCTIONS)}")
    return _FUNCTIONS[key]()

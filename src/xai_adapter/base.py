"""Base interfaces for XAI library adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

import numpy as np


ArrayLike = Any
PreprocessFn = Callable[[ArrayLike], np.ndarray]
PostprocessFn = Callable[[ArrayLike, np.ndarray], np.ndarray]


def identity_preprocess(x: ArrayLike) -> np.ndarray:
    """Default preprocessing: coerce input to a numpy array."""
    return np.asarray(x)


def identity_postprocess(_raw_instances: ArrayLike, attributions: np.ndarray) -> np.ndarray:
    """Default postprocessing: leave attributions unchanged."""
    return np.asarray(attributions)


def ensure_2d(values: ArrayLike) -> np.ndarray:
    """Return values as a 2D numpy array."""
    array = np.asarray(values)
    if array.ndim == 1:
        return array.reshape(1, -1)
    return array


def select_target(values: ArrayLike, target: Optional[int] = 1) -> np.ndarray:
    """
    Select a target column from model outputs.

    If outputs are already 1D, they are returned as-is. For 2D outputs and a
    target, the selected column is returned.
    """
    array = np.asarray(values)
    if array.ndim == 2 and target is not None:
        return array[:, target]
    return array.reshape(-1)


def baseline_from_data(data: ArrayLike, baseline: str = "mean") -> np.ndarray:
    """Compute a baseline vector from reference data."""
    data_array = ensure_2d(data)
    if baseline == "mean":
        return np.mean(data_array, axis=0)
    if baseline == "median":
        return np.median(data_array, axis=0)
    if baseline == "zeros":
        return np.zeros(data_array.shape[1], dtype=float)
    raise ValueError(f"Unknown baseline: {baseline}")


@dataclass
class XAIAdapterResult:
    """
    Normalized adapter output.

    Attributes:
        values: Attribution array with shape (n_instances, n_features).
        base_values: Per-instance intercept/base values.
        method: Adapter method name.
        metadata: Optional method-specific payload.
    """

    values: np.ndarray
    base_values: np.ndarray
    method: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def attributions(self) -> np.ndarray:
        """Signed per-feature attribution values."""
        return self.values

    @property
    def importances(self) -> np.ndarray:
        """Absolute per-feature importance magnitudes."""
        return np.abs(self.values)

    @property
    def feature_importance(self) -> np.ndarray:
        """Backward-compatible alias for signed attribution values."""
        return self.values


class XAIAdapter(ABC):
    """Common interface for external XAI library adapters."""

    method_name = "xai_adapter"

    def __init__(
        self,
        *,
        target: int = 1,
        preprocessing_fn: Optional[PreprocessFn] = None,
        postprocessing_fn: Optional[PostprocessFn] = None,
    ):
        self.target = target
        self.preprocessing_fn = preprocessing_fn or identity_preprocess
        self.postprocessing_fn = postprocessing_fn or identity_postprocess
        self.is_fitted = False

    def fit(self, X: ArrayLike = None, y: ArrayLike = None, **kwargs):
        """Fit or initialize method state. Subclasses override when needed."""
        self.is_fitted = True
        return self

    def _require_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError(f"{self.__class__.__name__} must be fitted before explain()")

    @abstractmethod
    def explain(self, instances: ArrayLike) -> XAIAdapterResult:
        """Explain one or more instances."""

    def attribute(self, instances: ArrayLike, **kwargs) -> XAIAdapterResult:
        """Backward-compatible alias for `explain(...)`."""
        return self.explain(instances, **kwargs)

    def _postprocess_values(self, raw_instances: ArrayLike, attributions: np.ndarray) -> np.ndarray:
        """Apply attribution postprocessing and ensure batch-major shape."""
        processed = np.asarray(self.postprocessing_fn(raw_instances, attributions))
        return ensure_2d(processed)

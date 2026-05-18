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

    def to_explanation_df(
        self,
        instance_ids: ArrayLike,
        predictions: ArrayLike,
        dataset_id: str,
        model_name: str,
        importance: bool = False,
    ):
        """Convert to the standard project explanation DataFrame.

        Schema: dataId, modelName, expMethod, instanceId, pred,
                i_max, a0_i … aN_i, intercept

        - ``aN_i``     — attribution (signed) or importance (absolute) normalized
                         by ``i_max``, depending on the ``importance`` flag
        - ``i_max``    — max absolute raw attribution for this instance
        - ``intercept``— SHAP / adapter base value (expected model output)

        Args:
            instance_ids: Array-like of instance identifiers, length n.
            predictions:  Array-like of model predictions (class labels), length n.
            dataset_id:   Value written to the ``dataId`` column.
            model_name:   Value written to the ``modelName`` column.
            importance:   If True, ``aN_i`` values are absolute (non-negative),
                          matching ``assets/explanations/coax/importance.csv``.
                          If False (default), values are signed, matching
                          ``assets/explanations/coax/attribution.csv``.

        Returns:
            pandas.DataFrame in the standard explanation schema.
        """
        import pandas as pd

        ids   = np.asarray(instance_ids)
        preds = np.asarray(predictions)
        n, n_features = self.values.shape
        rows = []
        for i in range(n):
            raw   = self.values[i]
            attrs = np.abs(raw) if importance else raw
            i_max = float(np.max(np.abs(raw))) if np.any(raw != 0) else 0.0
            row = {
                "dataId":     dataset_id,
                "modelName":  model_name,
                "expMethod":  self.method,
                "instanceId": int(ids[i]),
                "pred":       int(preds[i]),
                "i_max":      i_max,
            }
            for k in range(n_features):
                row[f"a{k}_i"] = attrs[k] / i_max if i_max != 0 else 0.0
            row["intercept"] = float(self.base_values[i])
            rows.append(row)
        return pd.DataFrame(rows)

    def save_csv(
        self,
        path,
        instance_ids: ArrayLike,
        predictions: ArrayLike,
        dataset_id: str,
        model_name: str,
        importance: bool = False,
    ) -> None:
        """Save explanation to a CSV file in the standard project format.

        Equivalent to ``to_explanation_df(..., importance=importance).to_csv(path, index=False)``.
        Parent directories are created automatically.

        Args:
            path:         Destination file path (str or Path).
            instance_ids: Array-like of instance identifiers, length n.
            predictions:  Array-like of model predictions (class labels), length n.
            dataset_id:   Value written to the ``dataId`` column.
            model_name:   Value written to the ``modelName`` column.
            importance:   If True, save absolute attribution values (importance format).
        """
        from pathlib import Path
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        self.to_explanation_df(
            instance_ids, predictions, dataset_id, model_name, importance=importance
        ).to_csv(dest, index=False)


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

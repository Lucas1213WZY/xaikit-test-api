"""Perturbation-based attribution methods."""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from ..base import (
    ArrayLike,
    PostprocessFn,
    PreprocessFn,
    XAIAdapterResult,
    baseline_from_data,
    ensure_2d,
    select_target,
)
from .base import LocalAttribution


class LeaveOneFeatureOut(LocalAttribution):
    """Local leave-one-feature-out attribution."""

    method_name = "lofo"

    def __init__(
        self,
        *,
        predict_fn: Callable[[np.ndarray], np.ndarray],
        background_data: Optional[ArrayLike] = None,
        baseline: str = "mean",
        target: int = 1,
        preprocessing_fn: Optional[PreprocessFn] = None,
        postprocessing_fn: Optional[PostprocessFn] = None,
    ):
        super().__init__(
            target=target,
            preprocessing_fn=preprocessing_fn,
            postprocessing_fn=postprocessing_fn,
        )
        self.predict_fn = predict_fn
        self.baseline = baseline
        self.baseline_vec = None
        self.baseline_value = 0.0
        if background_data is not None:
            self.fit(background_data)

    def fit(self, X: ArrayLike, y: ArrayLike = None, **kwargs):
        """Fit the replacement baseline from background data."""
        self.baseline_vec = baseline_from_data(self.preprocessing_fn(X), self.baseline)
        self.baseline_value = float(select_target(self.predict_fn(self.baseline_vec.reshape(1, -1)), self.target)[0])
        self.is_fitted = True
        return self

    def explain(self, instances: ArrayLike) -> XAIAdapterResult:
        self._require_fitted()
        raw_instances = ensure_2d(instances)
        x = ensure_2d(self.preprocessing_fn(raw_instances))
        base_probs = select_target(self.predict_fn(x), self.target)

        attributions = np.zeros_like(x, dtype=float)
        for feature_idx in range(x.shape[1]):
            masked = x.copy()
            masked[:, feature_idx] = self.baseline_vec[feature_idx]
            masked_probs = select_target(self.predict_fn(masked), self.target)
            attributions[:, feature_idx] = base_probs - masked_probs

        values = self._postprocess_values(raw_instances, attributions)
        return XAIAdapterResult(
            values=values,
            base_values=np.full(values.shape[0], self.baseline_value, dtype=float),
            method=self.method_name,
            metadata={"baseline": self.baseline_vec},
        )


__all__ = [
    "LeaveOneFeatureOut",
]

"""Base interfaces for feature attribution adapters."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from ..base import (
    ArrayLike,
    PostprocessFn,
    PreprocessFn,
    XAIAdapter,
    XAIAdapterResult,
    ensure_2d,
)


class Attribution(XAIAdapter):
    """Base class for local feature attribution methods."""

    def attribute(self, instances: ArrayLike, **kwargs) -> XAIAdapterResult:
        """Backward-compatible alias for explaining instances."""
        return self.explain(instances, **kwargs)


class LocalAttribution(Attribution):
    """Marker base class for local, instance-level attribution methods."""


class GlobalImportance:
    """Base mixin for methods that expose global feature importance."""

    def explain_global(self):
        raise NotImplementedError


class CustomAttribution(LocalAttribution):
    """
    Adapter for user-provided attribution functions or objects.

    The wrapped implementation can be a callable, or an object exposing
    `fit`, `explain`, or a legacy `attribute`. Outputs can be an
    `XAIAdapterResult`, a raw attribution array, or a `(values, base_values)`
    tuple.
    """

    method_name = "custom"

    def __init__(
        self,
        algorithm: Any,
        *,
        method_name: str = "custom",
        target: int = 1,
        preprocessing_fn: Optional[PreprocessFn] = None,
        postprocessing_fn: Optional[PostprocessFn] = None,
        auto_fit: bool = False,
        **algorithm_kwargs,
    ):
        super().__init__(
            target=target,
            preprocessing_fn=preprocessing_fn,
            postprocessing_fn=postprocessing_fn,
        )
        self.algorithm = algorithm
        self.method_name = method_name
        self.algorithm_kwargs = dict(algorithm_kwargs)
        self.is_fitted = not hasattr(algorithm, "fit")
        if auto_fit and hasattr(algorithm, "fit"):
            self.fit(**self.algorithm_kwargs)

    def fit(self, X: ArrayLike = None, y: ArrayLike = None, **kwargs):
        """Fit the wrapped algorithm if it exposes `fit`."""
        if hasattr(self.algorithm, "fit"):
            fit_kwargs = {**self.algorithm_kwargs, **kwargs}
            self.algorithm.fit(X, y, **fit_kwargs)
        self.is_fitted = True
        return self

    def explain(self, instances: ArrayLike, **kwargs) -> XAIAdapterResult:
        self._require_fitted()
        raw_instances = ensure_2d(instances)
        x = ensure_2d(self.preprocessing_fn(raw_instances))

        output = self._call_algorithm(raw_instances, x, **kwargs)
        return self._coerce_result(raw_instances, output)

    def _call_algorithm(self, raw_instances: np.ndarray, processed_instances: np.ndarray, **kwargs):
        call_kwargs = {**self.algorithm_kwargs, **kwargs}
        if hasattr(self.algorithm, "explain"):
            return self.algorithm.explain(processed_instances, **call_kwargs)
        if hasattr(self.algorithm, "attribute"):
            return self.algorithm.attribute(processed_instances, **call_kwargs)
        if callable(self.algorithm):
            return self.algorithm(processed_instances, **call_kwargs)
        raise TypeError("Custom attribution algorithm must be callable or expose explain()")

    def _coerce_result(self, raw_instances: np.ndarray, output: Any) -> XAIAdapterResult:
        if isinstance(output, XAIAdapterResult):
            return output

        metadata = {}
        if isinstance(output, tuple):
            if len(output) == 2:
                values, base_values = output
            elif len(output) == 3:
                values, base_values, metadata = output
            else:
                raise ValueError("Custom attribution tuple output must be (values, base_values[, metadata])")
        else:
            values = output
            base_values = np.zeros(ensure_2d(values).shape[0], dtype=float)

        values = self._postprocess_values(raw_instances, np.asarray(values, dtype=float))
        base_values = np.asarray(base_values, dtype=float).reshape(-1)
        if base_values.size == 1 and values.shape[0] > 1:
            base_values = np.full(values.shape[0], float(base_values[0]), dtype=float)

        return XAIAdapterResult(
            values=values,
            base_values=base_values,
            method=self.method_name,
            metadata=dict(metadata or {}),
        )


def make_attribution(
    algorithm: Any,
    *,
    method_name: str = "custom",
    **kwargs,
) -> CustomAttribution:
    """Create a normalized adapter from a user-defined attribution algorithm."""
    return CustomAttribution(algorithm, method_name=method_name, **kwargs)

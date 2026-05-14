"""Base interfaces for surrogate XAI methods."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from ..base import ArrayLike, XAIAdapter, XAIAdapterResult, ensure_2d


class SurrogateMethod(XAIAdapter):
    """Base class for fitted surrogate explainers."""

    def predict(self, instances: ArrayLike):
        raise NotImplementedError

    def apply(self, raw_input: Any):
        raise NotImplementedError

    def apply_batch(self, instances):
        return [self.apply(instance) for instance in instances]


class CustomSurrogate(SurrogateMethod):
    """
    Adapter for user-provided surrogate fit/explain callables.

    This mirrors the custom-attribution wrapper for global surrogate methods.
    `fit_fn` is called from `fit(...)`; its return value is kept on
    `fit_result` for explain callables that close over or inspect fitted state.
    """

    method_name = "custom"

    def __init__(
        self,
        fit_fn: Callable[..., Any],
        explain_fn: Callable[..., Any],
        *,
        method_name: str = "custom",
    ):
        super().__init__()
        self.fit_fn = fit_fn
        self.explain_fn = explain_fn
        self.method_name = method_name
        self.fit_result = None

    def fit(self, X: ArrayLike = None, y: ArrayLike = None, **kwargs):
        """Fit the wrapped surrogate callable."""
        self.fit_result = self.fit_fn(X, y, **kwargs)
        self.is_fitted = True
        return self

    def explain(self, instances: ArrayLike) -> XAIAdapterResult:
        """Explain instances with the wrapped surrogate callable."""
        self._require_fitted()
        output = self.explain_fn(instances)
        return self._coerce_result(instances, output)

    def _coerce_result(self, instances: ArrayLike, output: Any) -> XAIAdapterResult:
        if isinstance(output, XAIAdapterResult):
            return output

        metadata = {}
        if isinstance(output, tuple):
            if len(output) == 2:
                values, base_values = output
            elif len(output) == 3:
                values, base_values, metadata = output
            else:
                raise ValueError("Custom surrogate tuple output must be (values, base_values[, metadata])")
        else:
            values = output
            base_values = np.zeros(ensure_2d(values).shape[0], dtype=float)

        values = ensure_2d(values).astype(float, copy=False)
        base_values = np.asarray(base_values, dtype=float).reshape(-1)
        if base_values.size == 1 and values.shape[0] > 1:
            base_values = np.full(values.shape[0], float(base_values[0]), dtype=float)

        return XAIAdapterResult(
            values=values,
            base_values=base_values,
            method=self.method_name,
            metadata=dict(metadata or {}),
        )


def make_surrogate(
    fit_fn: Callable[..., Any],
    explain_fn: Callable[..., Any],
    name: str = "custom",
) -> CustomSurrogate:
    """Wrap any surrogate-like callable pair in the SurrogateMethod interface."""
    return CustomSurrogate(fit_fn, explain_fn, method_name=name)

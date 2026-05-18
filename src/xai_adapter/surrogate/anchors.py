"""Anchor rule-based local explanations via alibi."""

from __future__ import annotations

from typing import Callable, List, Optional

import numpy as np

from ..base import (
    ArrayLike,
    XAIAdapterResult,
    ensure_2d,
)
from .base import SurrogateMethod


class AnchorsAdapter(SurrogateMethod):
    """Rule-based local explanations (Anchors) via alibi.AnchorTabular.

    Anchors find a minimal set of feature conditions (an IF-THEN rule) that
    locally fixes the model's prediction with high precision.  This is a
    rule-based surrogate — structurally equivalent to a local decision rule —
    so it lives alongside the other rule-based surrogates.

    ``explain()`` returns an ``XAIAdapterResult`` where:
      - ``values``: ``(n_instances, n_features)`` binary mask — 1 for each
        feature that appears in the anchor rule, 0 otherwise.
      - ``base_values``: zeros.
      - ``metadata["rules"]``: list of human-readable rule strings.
      - ``metadata["precision"]``: per-instance anchor precision.
      - ``metadata["coverage"]``: per-instance anchor coverage.

    Parameters
    ----------
    predict_fn : callable
        Prediction function returning integer class labels
        ``f(X_np) -> (n,)`` int array.  AnchorTabular requires discrete
        class outputs, not probabilities.
    training_data : array-like
        Background data for beam-search perturbation sampling.
    feature_names : list of str
        Human-readable feature names (must match column order of X).
    categorical_names : dict, optional
        ``{feature_index: [category_str, ...]}`` for categorical features.
    threshold : float
        Minimum anchor precision required (default 0.95).
    target : int
        Kept for API consistency (not used internally).
    """

    method_name = "anchors"

    def __init__(
        self,
        *,
        predict_fn: Callable[[np.ndarray], np.ndarray],
        training_data: ArrayLike,
        feature_names: List[str],
        categorical_names: Optional[dict] = None,
        threshold: float = 0.95,
        target: int = 1,
        preprocessing_fn: Optional[Callable] = None,
    ):
        super().__init__()
        try:
            from alibi.explainers import AnchorTabular  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "alibi is required for AnchorsAdapter. "
                "Install with: pip install alibi"
            ) from exc

        self.target = target
        self.preprocessing_fn = preprocessing_fn or (lambda x: x)
        self.predict_fn = predict_fn
        self.feature_names = list(feature_names)
        self.threshold = threshold

        X_bg = ensure_2d(self.preprocessing_fn(training_data))
        cat_names = categorical_names or {}

        from alibi.explainers import AnchorTabular
        self._explainer = AnchorTabular(
            predictor=lambda x: self.predict_fn(self.preprocessing_fn(x)).astype(int),
            feature_names=self.feature_names,
            categorical_names=cat_names,
        )
        self._explainer.fit(X_bg)
        self.is_fitted = True

    def fit(self, X: ArrayLike = None, y: ArrayLike = None, **kwargs):
        if X is not None:
            x = ensure_2d(self.preprocessing_fn(X))
            self._explainer.fit(x)
        self.is_fitted = True
        return self

    def predict(self, instances: ArrayLike):
        return self.explain(instances)

    def apply(self, instance):
        x = ensure_2d(self.preprocessing_fn(instance))
        exp = self._explainer.explain(x[0], threshold=self.threshold)
        return " AND ".join(exp.anchor) if exp.anchor else "(no anchor)"

    def explain(self, instances: ArrayLike) -> XAIAdapterResult:
        self._require_fitted()
        raw = ensure_2d(instances)
        x = ensure_2d(self.preprocessing_fn(raw))
        n, n_features = x.shape

        masks = np.zeros((n, n_features), dtype=float)
        rules: List[str] = []
        precisions = np.zeros(n, dtype=float)
        coverages = np.zeros(n, dtype=float)

        for i in range(n):
            exp = self._explainer.explain(x[i], threshold=self.threshold)
            anchor_conditions = exp.anchor
            rules.append(" AND ".join(anchor_conditions) if anchor_conditions else "(no anchor)")
            precisions[i] = float(exp.precision)
            coverages[i] = float(exp.coverage)

            for cond in anchor_conditions:
                for j, name in enumerate(self.feature_names):
                    if name in cond:
                        masks[i, j] = 1.0
                        break

        return XAIAdapterResult(
            values=masks,
            base_values=np.zeros(n, dtype=float),
            method=self.method_name,
            metadata={
                "rules": rules,
                "precision": precisions,
                "coverage": coverages,
            },
        )


__all__ = ["AnchorsAdapter"]

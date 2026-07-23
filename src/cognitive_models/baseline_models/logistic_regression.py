"""Logistic-regression machine-proxy baseline."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .base import DualInputClassifierBaseline


class LogisticRegressionBaseline(DualInputClassifierBaseline):
    """Regularized logistic proxy trained on participant examples."""

    def __init__(
        self,
        *,
        penalty: str = "l2",
        C: float = 1.0,
        k: int | None = None,
        smoothing_factor: float = 0.01,
        max_iter: int = 1000,
        random_state: int = 42,
    ) -> None:
        super().__init__(smoothing_factor=smoothing_factor)
        if penalty not in {"l1", "l2"}:
            raise ValueError("penalty must be 'l1' or 'l2'.")
        if C <= 0:
            raise ValueError("C must be positive.")
        if k is not None and k < 1:
            raise ValueError("k must be at least 1 or None.")
        if max_iter < 1:
            raise ValueError("max_iter must be at least 1.")
        self.penalty = penalty
        self.C = float(C)
        self.k = int(k) if k is not None else None
        self.max_iter = int(max_iter)
        self.random_state = int(random_state)

    def _make_estimator(self) -> Any:
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                penalty=self.penalty,
                C=self.C,
                solver="liblinear",
                max_iter=self.max_iter,
                random_state=self.random_state,
            ),
        )

    def _fit_estimator(self, x: np.ndarray, y: np.ndarray) -> Any:
        estimator = super()._fit_estimator(x, y)
        if self.k is None or not hasattr(estimator, "named_steps"):
            return estimator

        logistic = estimator.named_steps["logisticregression"]
        coefficients = logistic.coef_
        if coefficients.size > self.k:
            keep = np.argpartition(
                np.abs(coefficients).reshape(-1),
                -self.k,
            )[-self.k:]
            mask = np.zeros(coefficients.size, dtype=bool)
            mask[keep] = True
            logistic.coef_ = np.where(
                mask.reshape(coefficients.shape),
                coefficients,
                0.0,
            )
        return estimator

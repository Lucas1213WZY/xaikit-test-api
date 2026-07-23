"""Decision-tree machine-proxy baseline."""

from __future__ import annotations

from typing import Any, Optional

from sklearn.tree import DecisionTreeClassifier

from .base import DualInputClassifierBaseline


class DecisionTreeBaseline(DualInputClassifierBaseline):
    """Decision tree trained to reproduce AI predictions."""

    def __init__(
        self,
        *,
        max_depth: Optional[int] = None,
        min_samples_leaf: int = 1,
        smoothing_factor: float = 0.01,
        random_state: int = 42,
    ) -> None:
        super().__init__(smoothing_factor=smoothing_factor)
        if max_depth is not None and max_depth < 1:
            raise ValueError("max_depth must be at least 1 or None.")
        if min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be at least 1.")
        self.max_depth = max_depth
        self.min_samples_leaf = int(min_samples_leaf)
        self.random_state = int(random_state)

    def _make_estimator(self) -> Any:
        return DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
        )

"""MLP machine-proxy baseline."""

from __future__ import annotations

from typing import Any, Optional, Sequence

from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .base import DualInputClassifierBaseline


class MLPBaseline(DualInputClassifierBaseline):
    """Small feed-forward proxy trained on participant examples."""

    def __init__(
        self,
        *,
        hidden_layer_sizes: Optional[Sequence[int]] = None,
        hidden_dim: int = 32,
        smoothing_factor: float = 0.01,
        epochs: int = 300,
        learning_rate: float = 0.001,
        max_iter: Optional[int] = None,
        learning_rate_init: Optional[float] = None,
        random_state: int = 42,
    ) -> None:
        super().__init__(smoothing_factor=smoothing_factor)
        hidden_layers = tuple(
            int(size)
            for size in (
                hidden_layer_sizes
                if hidden_layer_sizes is not None
                else (hidden_dim,)
            )
        )
        resolved_iterations = int(max_iter if max_iter is not None else epochs)
        resolved_learning_rate = float(
            learning_rate_init
            if learning_rate_init is not None
            else learning_rate
        )
        if not hidden_layers or any(size < 1 for size in hidden_layers):
            raise ValueError("hidden_layer_sizes must contain positive integers.")
        if resolved_iterations < 1:
            raise ValueError("epochs/max_iter must be at least 1.")
        if resolved_learning_rate <= 0:
            raise ValueError("learning_rate/learning_rate_init must be positive.")
        self.hidden_layer_sizes = hidden_layers
        self.hidden_dim = int(hidden_dim)
        self.epochs = int(epochs)
        self.learning_rate = float(learning_rate)
        self.max_iter = resolved_iterations
        self.learning_rate_init = resolved_learning_rate
        self.random_state = int(random_state)

    def _make_estimator(self) -> Any:
        return make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=self.hidden_layer_sizes,
                max_iter=self.max_iter,
                learning_rate_init=self.learning_rate_init,
                random_state=self.random_state,
            ),
        )

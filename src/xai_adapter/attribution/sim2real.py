"""Property-optimized explanations for XAIsim2real functions."""

from __future__ import annotations

from typing import Optional

import numpy as np

from src.ai_models.models.synthetic import create_sim2real_function

from ..base import ArrayLike, PreprocessFn, XAIAdapterResult, ensure_2d
from ..metrics import (
    fidelity_classification_loss,
    fidelity_regression_loss,
    summarize_property_metrics,
)
from .base import LocalAttribution


_PROPERTY_ALIASES = {
    "faithful": "faithful",
    "faithfulness": "faithful",
    "sparse": "sparse",
    "sparsity": "sparse",
    "robust": "robust",
    "robustness": "robust",
    "sparse_robust": "sparse_robust",
    "sparse+robust": "sparse_robust",
    "sparse_and_robust": "sparse_robust",
    "robust_and_sparse": "sparse_robust",
    "robust_sparse": "sparse_robust",
    "robust+sparse": "sparse_robust",
}


class Sim2RealPropertyAttribution(LocalAttribution):
    """Generate paper-style property-optimized attribution matrices.

    The paper denotes the random-search sample count as ``S`` but does not
    provide a concrete value. ``n_candidate_explanations`` is therefore an
    implementation default, not a reproduction claim.
    """

    method_name = "sim2real_property"

    def __init__(
        self,
        *,
        model=None,
        function_name: Optional[str] = None,
        property_name: str = "faithful",
        top_k: int = 2,
        n_candidate_explanations: int = 1000,
        n_local_samples: int = 200,
        local_radius: float = 0.05,
        robustness_radius: float = 1.0,
        confound_scale: float = 4.0,
        random_state: Optional[int] = 0,
        target: int = 1,
        preprocessing_fn: Optional[PreprocessFn] = None,
        postprocessing_fn=None,
    ):
        super().__init__(
            target=target,
            preprocessing_fn=preprocessing_fn,
            postprocessing_fn=postprocessing_fn,
        )
        if model is None:
            if function_name is None:
                raise ValueError("Pass either model or function_name.")
            model = create_sim2real_function(function_name)
        self.model = getattr(model, "engine", model)
        self.function_name = function_name or getattr(self.model, "function_name", None)
        if self.function_name is None:
            raise ValueError("Could not infer sim2real function_name from model.")
        self.function_name = self.function_name.lower()
        self.property_name = _PROPERTY_ALIASES.get(property_name.lower(), property_name.lower())
        if self.property_name not in set(_PROPERTY_ALIASES.values()):
            raise ValueError(f"Unknown property_name '{property_name}'.")
        self.top_k = int(top_k)
        self.n_candidate_explanations = int(n_candidate_explanations)
        self.n_local_samples = int(n_local_samples)
        self.local_radius = float(local_radius)
        self.robustness_radius = float(robustness_radius)
        self.confound_scale = float(confound_scale)
        self.rng = np.random.default_rng(random_state)
        self.is_fitted = True

    def explain(self, instances: ArrayLike) -> XAIAdapterResult:
        raw_instances = ensure_2d(instances)
        X = ensure_2d(self.preprocessing_fn(raw_instances))
        values, base_values, strategy = self._generate_values(X)
        values = self._postprocess_values(raw_instances, values)
        metadata = {
            "function_name": self.function_name,
            "property_name": self.property_name,
            "strategy": strategy,
            "feature_names": list(getattr(self.model, "feature_names", [])),
            "top_k": self.top_k,
            "n_candidate_explanations": self.n_candidate_explanations,
            "n_local_samples": self.n_local_samples,
            "local_radius": self.local_radius,
            "paper_note": (
                "The paper defines random-search and local-neighbor sample counts as S "
                "but does not specify concrete values."
            ),
            "property_metrics": summarize_property_metrics(
                self.model,
                X,
                values,
                output_type=getattr(self.model, "output_type", "classification"),
                base_values=base_values,
                radius=self.robustness_radius,
            ),
        }
        return XAIAdapterResult(
            values=values,
            base_values=base_values,
            method=f"{self.method_name}_{self.property_name}",
            metadata=metadata,
        )

    def _generate_values(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, str]:
        if self.function_name in {"sparse", "fsparse"}:
            return self._generate_sparse_function_values(X)
        if self.function_name in {"trend_wiggle", "trend+wiggle", "ftrend+wiggle"}:
            return self._generate_trend_wiggle_values(X)
        if self.function_name in {"wiggle", "fwiggle"}:
            return self._generate_wiggle_values(X)
        raise ValueError(f"Unknown sim2real function '{self.function_name}'.")

    def _generate_sparse_function_values(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, str]:
        faithful = self.model.ground_truth_weights(X)
        base = self.model.ground_truth_base_values(X)
        if self.property_name == "faithful":
            return self._make_sparse_faithful_less_sparse(faithful), base, "ground_truth_weights_confound_adjusted"
        if self.property_name == "sparse":
            return faithful, base, "ground_truth_weights"
        if self.property_name == "robust":
            return self._sample_global_explanation(X, sparse=False), np.zeros(X.shape[0]), "sample_global_explanations"
        return self._sample_global_explanation(X, sparse=True), np.zeros(X.shape[0]), "sample_global_explanations_only_top_k"

    def _generate_trend_wiggle_values(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, str]:
        faithful = self._fit_locally_faithful_line(X)
        base = np.zeros(X.shape[0], dtype=float)
        if self.property_name == "faithful":
            return faithful, base, "fit_locally_faithful_line"
        if self.property_name == "sparse":
            sparse = self._top_k_from_faithful(faithful)
            return self._reduce_robustness(sparse), base, "top_k_from_faithful_confound_adjusted"
        if self.property_name == "robust":
            trend = np.tile(self.model.trend_weights(), (X.shape[0], 1))
            return trend, base, "ground_truth_trend_weights"
        return self._sample_global_explanation(X, sparse=True), base, "sample_global_explanations_only_top_k"

    def _generate_wiggle_values(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, str]:
        faithful = self.model.ground_truth_weights(X)
        base = np.zeros(X.shape[0], dtype=float)
        if self.property_name == "faithful":
            return faithful, base, "ground_truth_weights"
        if self.property_name == "sparse":
            sparse = self._top_k_from_faithful(faithful)
            return self._reduce_robustness(sparse), base, "top_k_from_faithful_confound_adjusted"
        if self.property_name == "robust":
            return self._sample_global_explanation(X, sparse=False), base, "sample_global_explanations"
        return self._sample_global_explanation(X, sparse=True), base, "sample_global_explanations_only_top_k"

    def _make_sparse_faithful_less_sparse(self, W: np.ndarray) -> np.ndarray:
        adjusted = W.copy()
        for row in adjusted:
            zero_indices = np.flatnonzero(row == 0.0)
            if len(zero_indices) == 1:
                row[zero_indices[0]] += 0.6
            elif len(zero_indices) >= 2:
                row[zero_indices[0]] -= 0.6
                row[zero_indices[1]] += 0.6
        return adjusted

    def _top_k_from_faithful(self, W: np.ndarray) -> np.ndarray:
        if self.top_k >= W.shape[1]:
            return W.copy()
        sparse = np.zeros_like(W)
        indices = np.argpartition(np.abs(W), -self.top_k, axis=1)[:, -self.top_k:]
        rows = np.arange(W.shape[0])[:, None]
        sparse[rows, indices] = W[rows, indices]
        return sparse

    def _reduce_robustness(self, W: np.ndarray) -> np.ndarray:
        adjusted = W.copy()
        if adjusted.shape[0] == 0:
            return adjusted
        row_mask = self.rng.random(adjusted.shape[0]) < 0.5
        if not np.any(row_mask):
            row_mask[self.rng.integers(0, adjusted.shape[0])] = True
        adjusted[row_mask] *= self.confound_scale
        return adjusted

    def _fit_locally_faithful_line(self, X: np.ndarray) -> np.ndarray:
        weights = np.zeros_like(X, dtype=float)
        for i, x in enumerate(X):
            noise = self.rng.normal(loc=0.0, scale=self.local_radius, size=(self.n_local_samples, X.shape[1]))
            neighbors = x + noise
            neighbors[0] = x
            y = np.asarray(self.model.predict(neighbors), dtype=float).reshape(-1)
            weights[i] = np.linalg.lstsq(neighbors, y, rcond=None)[0]
        return weights

    def _sample_global_explanation(self, X: np.ndarray, *, sparse: bool) -> np.ndarray:
        n, d = X.shape
        best_row = None
        best_loss = np.inf
        output_type = getattr(self.model, "output_type", "classification")

        for _ in range(self.n_candidate_explanations):
            row = self.rng.uniform(0.0, 1.0, size=d)
            if sparse:
                mask = np.zeros(d, dtype=bool)
                keep = self.rng.choice(d, size=min(self.top_k, d), replace=False)
                mask[keep] = True
                row = row * mask
            candidate = np.tile(row, (n, 1))
            if output_type == "regression":
                loss = float(np.mean(fidelity_regression_loss(self.model, X, candidate)))
            else:
                loss = float(np.mean(fidelity_classification_loss(self.model, X, candidate)))
            if loss < best_loss:
                best_loss = loss
                best_row = row

        return np.tile(best_row, (n, 1))

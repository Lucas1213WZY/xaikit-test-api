"""Prototype and criticism selection via MMD-Critic."""

from __future__ import annotations

from typing import Callable, List, Optional

import numpy as np

from ..base import (
    ArrayLike,
    XAIAdapterResult,
    ensure_2d,
)
from .base import ExampleBasedAdapter


def _rbf_kernel(X: np.ndarray, Y: np.ndarray, bandwidth: float) -> np.ndarray:
    """Compute RBF (Gaussian) kernel matrix K(X, Y)."""
    diff = X[:, None, :] - Y[None, :, :]
    return np.exp(-np.sum(diff ** 2, axis=-1) / (2.0 * bandwidth ** 2))


def _mmd_critic_select(
    X: np.ndarray,
    n_prototypes: int,
    n_criticisms: int,
    bandwidth: float,
    regularizer: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Greedy prototype and criticism selection (MMD-Critic algorithm).

    Returns
    -------
    proto_idx : (n_prototypes,) int array
    crit_idx  : (n_criticisms,) int array
    """
    n = X.shape[0]
    K = _rbf_kernel(X, X, bandwidth)

    # Greedy prototype selection: maximise witness function coverage
    proto_idx: List[int] = []
    remaining = list(range(n))
    for _ in range(min(n_prototypes, n)):
        best, best_score = -1, -np.inf
        for j in remaining:
            tentative = proto_idx + [j]
            score = (
                np.sum(K[tentative, :]) / (len(tentative) * n)
                - np.sum(K[tentative][:, tentative]) / (2 * len(tentative) ** 2)
            )
            if score > best_score:
                best, best_score = j, score
        proto_idx.append(best)
        remaining.remove(best)

    # Greedy criticism selection: maximise witness function value
    crit_idx: List[int] = []
    proto_arr = np.array(proto_idx)
    non_proto = [i for i in range(n) if i not in set(proto_idx)]
    candidate_pool = non_proto if non_proto else list(range(n))

    for _ in range(min(n_criticisms, len(candidate_pool))):
        witness = (
            np.mean(K[candidate_pool][:, proto_arr], axis=1)
            - np.mean(K[candidate_pool][:, :], axis=1)
        )
        if regularizer == "logdet" and len(crit_idx) > 0:
            crit_arr = np.array(crit_idx)
            reg = np.log(np.abs(np.linalg.det(K[np.ix_(crit_arr, crit_arr)]) + 1e-8))
            witness = np.abs(witness) + reg
        else:
            witness = np.abs(witness)
        best_local = int(np.argmax(witness))
        best_global = candidate_pool[best_local]
        crit_idx.append(best_global)
        candidate_pool.pop(best_local)

    return np.array(proto_idx, dtype=int), np.array(crit_idx, dtype=int)


class PrototypesAdapter(ExampleBasedAdapter):
    """Prototype and criticism selection using MMD-Critic (Kim et al., 2016).

    Selects a small set of *prototypes* (representative examples) and
    *criticisms* (atypical examples) from background data using a
    Maximum Mean Discrepancy greedy algorithm.

    After ``fit()``, ``explain()`` returns for each query instance:
      - ``values``: shape ``(n_instances, n_features)`` — the feature vector
        of the *nearest prototype* to that instance.
      - ``base_values``: zeros.
      - ``metadata["proto_idx"]``: indices into background data.
      - ``metadata["crit_idx"]``: indices into background data.
      - ``metadata["proto_X"]``: prototype feature vectors ``(n_prototypes, n_features)``.
      - ``metadata["crit_X"]``: criticism feature vectors ``(n_criticisms, n_features)``.
      - ``metadata["nearest_proto_idx"]``: per-instance index of the nearest prototype.
      - ``metadata["nearest_proto_dist"]``: per-instance Euclidean distance.

    Parameters
    ----------
    n_prototypes : int
        Number of prototypes to select (default 5).
    n_criticisms : int
        Number of criticisms to select (default 5).
    bandwidth : float
        RBF kernel bandwidth.  If None (default), set to median pairwise
        distance of background data.
    regularizer : str
        Criticism selection regularizer: 'logdet' or 'iterative' (default
        'iterative' = greedy without regularization).
    target : int
        Kept for API consistency (not used in prototype selection).
    """

    method_name = "prototypes"

    def __init__(
        self,
        *,
        n_prototypes: int = 5,
        n_criticisms: int = 5,
        bandwidth: Optional[float] = None,
        regularizer: str = "iterative",
        target: int = 1,
        preprocessing_fn: Optional[Callable] = None,
        postprocessing_fn: Optional[Callable] = None,
    ):
        super().__init__(
            target=target,
            preprocessing_fn=preprocessing_fn,
            postprocessing_fn=postprocessing_fn,
        )
        self.n_prototypes = int(n_prototypes)
        self.n_criticisms = int(n_criticisms)
        self.bandwidth = bandwidth
        self.regularizer = regularizer
        self._bg_X: Optional[np.ndarray] = None
        self._proto_idx: Optional[np.ndarray] = None
        self._crit_idx: Optional[np.ndarray] = None

    def fit(self, X: ArrayLike, y: ArrayLike = None, **kwargs):
        """Select prototypes and criticisms from background data."""
        self._bg_X = ensure_2d(self.preprocessing_fn(X))
        bw = self.bandwidth
        if bw is None:
            n = self._bg_X.shape[0]
            sample = self._bg_X[:min(n, 500)]
            diffs = sample[:, None, :] - sample[None, :, :]
            dists = np.sqrt(np.sum(diffs ** 2, axis=-1))
            bw = float(np.median(dists[dists > 0])) if np.any(dists > 0) else 1.0
        self._bandwidth = bw
        self._proto_idx, self._crit_idx = _mmd_critic_select(
            self._bg_X, self.n_prototypes, self.n_criticisms, bw, self.regularizer
        )
        self.is_fitted = True
        return self

    def explain(self, instances: ArrayLike) -> XAIAdapterResult:
        self._require_fitted()
        raw = ensure_2d(instances)
        x = ensure_2d(self.preprocessing_fn(raw))
        n = x.shape[0]

        proto_X = self._bg_X[self._proto_idx]
        crit_X = self._bg_X[self._crit_idx]

        dists = np.linalg.norm(x[:, None, :] - proto_X[None, :, :], axis=-1)
        nearest_local = np.argmin(dists, axis=1)
        nearest_proto = proto_X[nearest_local]
        nearest_dist = dists[np.arange(n), nearest_local]

        return XAIAdapterResult(
            values=nearest_proto,
            base_values=np.zeros(n, dtype=float),
            method=self.method_name,
            metadata={
                "proto_idx": self._proto_idx,
                "crit_idx": self._crit_idx,
                "proto_X": proto_X,
                "crit_X": crit_X,
                "nearest_proto_idx": self._proto_idx[nearest_local],
                "nearest_proto_dist": nearest_dist,
            },
        )


__all__ = ["PrototypesAdapter"]

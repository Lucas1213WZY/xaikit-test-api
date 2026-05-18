"""Explainable Boosting Machine (EBM) adapter.

InterpretML's EBM is a Generalized Additive Model with pairwise interactions.
It implements the sklearn estimator interface (fit/predict/predict_proba) and
is its own explainer — no separate post-hoc step required.

Local scores are additive feature contributions (structurally equivalent to
SHAP values). Global scores are mean absolute contributions per feature.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from ..base import ArrayLike, XAIAdapterResult, ensure_2d
from ..attribution.base import LocalAttribution, GlobalImportance


def _import_interpret():
    try:
        from interpret.glassbox import ExplainableBoostingClassifier
        return ExplainableBoostingClassifier
    except ImportError as exc:
        raise ImportError(
            "InterpretML is required. Install with: pip install interpret"
        ) from exc


class EBMAdapter(LocalAttribution, GlobalImportance):
    """
    Adapter for InterpretML's ExplainableBoostingClassifier (EBM).

    The EBM is both the predictive model and the explainer. Call fit() to
    train it, then explain() for local per-instance scores or explain_global()
    for overall feature importances.

    Parameters
    ----------
    feature_names : list of str, optional
        Column names passed to the EBM for labelled explanations.
    max_bins : int
        Number of bins for continuous features (default 256).
    interactions : int
        Number of pairwise interaction terms to include (default 10).
    target : int
        Class index whose scores are extracted (default 1).
    ebm_kwargs : dict
        Extra keyword arguments forwarded to ExplainableBoostingClassifier.
    """

    method_name = "ebm"

    def __init__(
        self,
        *,
        feature_names: Optional[List[str]] = None,
        max_bins: int = 256,
        interactions: int = 10,
        target: int = 1,
        **ebm_kwargs,
    ):
        super().__init__(target=target)
        EBC = _import_interpret()
        self._feature_names = feature_names
        self._n_features: Optional[int] = None
        self.ebm = EBC(
            feature_names=feature_names,
            max_bins=max_bins,
            interactions=interactions,
            **ebm_kwargs,
        )

    # ------------------------------------------------------------------
    # sklearn-compatible pass-throughs
    # ------------------------------------------------------------------

    def predict(self, X: ArrayLike) -> np.ndarray:
        return self.ebm.predict(ensure_2d(X))

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        return self.ebm.predict_proba(ensure_2d(X))

    # ------------------------------------------------------------------
    # XAIAdapter interface
    # ------------------------------------------------------------------

    def fit(self, X: ArrayLike, y: ArrayLike, **_) -> "EBMAdapter":
        """Train the EBM on (X, y)."""
        X_arr = ensure_2d(X)
        self._n_features = X_arr.shape[1]
        self.ebm.fit(X_arr, np.asarray(y))
        self.is_fitted = True
        return self

    def explain(self, instances: ArrayLike, y: ArrayLike = None) -> XAIAdapterResult:
        """
        Return local EBM scores for each instance.

        values shape: (n_instances, n_features)  — main-effect terms only.
        Full scores including interaction terms are stored in
        metadata['full_scores'] when interactions are present.
        """
        self._require_fitted()
        rows = ensure_2d(instances)
        n = rows.shape[0]

        # EBM explain_local expects array-like; y is optional (used only for
        # dashboard error colouring) — pass predicted labels if not supplied.
        y_arr = np.asarray(y) if y is not None else self.predict(rows)
        ebm_local = self.ebm.explain_local(rows, y_arr)

        values = np.zeros((n, self._n_features), dtype=float)
        base_values = np.zeros(n, dtype=float)
        full_scores: List[List[float]] = []

        for i in range(n):
            data = ebm_local.data(i)
            scores = list(data.get("scores", []))
            full_scores.append(scores)
            # main-effect scores: first n_features entries
            main = scores[: self._n_features]
            values[i, : len(main)] = main
            intercept = data.get("intercept", 0.0)
            base_values[i] = float(intercept[self.target]
                                   if hasattr(intercept, "__len__")
                                   else intercept)

        has_interactions = any(len(s) > self._n_features for s in full_scores)
        return XAIAdapterResult(
            values=values,
            base_values=base_values,
            method=self.method_name,
            metadata={
                "n_features": self._n_features,
                "has_interactions": has_interactions,
                "full_scores": full_scores if has_interactions else None,
            },
        )

    # ------------------------------------------------------------------
    # GlobalImportance interface
    # ------------------------------------------------------------------

    def explain_global(self) -> Dict[str, Any]:
        """
        Return global EBM feature importances.

        Returns
        -------
        dict with keys:
            'feature_names'  : list of str
            'importances'    : np.ndarray — mean absolute contribution per feature
            'ebm_explanation': raw interpret Explanation object (for dashboard use)
        """
        self._require_fitted()
        ebm_global = self.ebm.explain_global()
        data = ebm_global.data()
        names = list(data.get("names", []))
        scores = np.asarray(data.get("scores", []), dtype=float)
        # main-effect terms only
        return {
            "feature_names": names[: self._n_features],
            "importances": scores[: self._n_features],
            "ebm_explanation": ebm_global,
        }

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def ebm_model(self):
        """Underlying ExplainableBoostingClassifier (for InterpretML dashboard)."""
        return self.ebm

"""Global feature-importance methods."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from .base import GlobalImportance


class SklearnFeatureImportance(GlobalImportance):
    """Global feature importance from sklearn-style fitted estimators."""

    method_name = "sklearn_global_feature_importance"

    def __init__(self, estimator: Any = None, feature_names: Optional[List[str]] = None):
        self.estimator = estimator
        self.feature_names = feature_names
        self.is_fitted = estimator is not None

    def fit(self, estimator: Any, y: Any = None, **kwargs):
        """Store a fitted sklearn-style estimator."""
        self.estimator = estimator
        self.is_fitted = True
        return self

    def explain_global(self) -> Dict[str, Any]:
        """Return global importances as records sorted by absolute importance."""
        if not self.is_fitted:
            raise RuntimeError(f"{self.__class__.__name__} must be fitted before explain_global()")

        if hasattr(self.estimator, "feature_importances_"):
            importances = np.asarray(self.estimator.feature_importances_, dtype=float)
        elif hasattr(self.estimator, "coef_"):
            importances = np.asarray(self.estimator.coef_, dtype=float)
            if importances.ndim > 1:
                importances = importances[0]
        else:
            raise AttributeError("Estimator must expose feature_importances_ or coef_")

        feature_names = self.feature_names or [f"feature_{i}" for i in range(len(importances))]
        records = [
            {"feature": name, "importance": float(value)}
            for name, value in zip(feature_names, importances)
        ]
        records.sort(key=lambda item: abs(item["importance"]), reverse=True)
        return {
            "method": self.method_name,
            "importances": importances,
            "feature_names": feature_names,
            "records": records,
        }


__all__ = [
    "SklearnFeatureImportance",
]

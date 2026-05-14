"""Logistic-regression surrogate method for CoXAM-style explanation tables."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from ..base import ArrayLike, XAIAdapterResult, ensure_2d
from .base import SurrogateMethod


class LogisticRegressionSurrogateMethod(SurrogateMethod):
    """
    Logistic-regression surrogate method for rules-vs-weights comparisons.

    The expected explanation table matches `assets/explanations/coxam/logistic_regression.csv`.
    """

    method_name = "logistic_regression"

    def __init__(
        self,
        *,
        explanation_df: Optional[pd.DataFrame] = None,
        metadata_df: Optional[pd.DataFrame] = None,
        app_id: Optional[str] = None,
        model_name: Optional[str] = None,
        variant: str = "dense",
        top_k: int = 3,
        C: float = 1.0,
        random_state: int = 0,
        max_iter: int = 1000,
        feature_names: Optional[List[str]] = None,
        target: int = 1,
        **logistic_kwargs,
    ):
        super().__init__(target=target)
        self.app_id = app_id
        self.model_name = model_name
        self.variant = variant
        self.top_k = top_k
        self.C = C
        self.random_state = random_state
        self.max_iter = max_iter
        self.feature_names = feature_names
        self.logistic_kwargs = logistic_kwargs
        self.fidelity = None
        self.intercept = 0.0
        self.coefficients = OrderedDict()
        self.explanation_df = None
        self.metadata_df = None
        self.explanation_row = None
        self.metadata_row = None
        if explanation_df is not None and metadata_df is not None:
            self.fit(explanation_df=explanation_df, metadata_df=metadata_df, app_id=app_id, model_name=model_name)

    def fit(
        self,
        X: ArrayLike = None,
        y: ArrayLike = None,
        *,
        explanation_df: Optional[pd.DataFrame] = None,
        metadata_df: Optional[pd.DataFrame] = None,
        app_id: Optional[str] = None,
        model_name: Optional[str] = None,
        variant: Optional[str] = None,
        mode: Optional[str] = None,
        top_k: Optional[int] = None,
        C: Optional[float] = None,
        random_state: Optional[int] = None,
        max_iter: Optional[int] = None,
        feature_names: Optional[List[str]] = None,
        **kwargs,
    ):
        """Fit from a precomputed table or generate a surrogate from X/y."""
        if explanation_df is None and metadata_df is None and y is not None:
            return self._fit_generated(
                X,
                y,
                app_id=app_id,
                model_name=model_name,
                variant=variant,
                top_k=top_k,
                C=C,
                random_state=random_state,
                max_iter=max_iter,
                feature_names=feature_names,
                **kwargs,
            )

        if mode == "generate":
            return self._fit_generated(
                X,
                y,
                app_id=app_id,
                model_name=model_name,
                variant=variant,
                top_k=top_k,
                C=C,
                random_state=random_state,
                max_iter=max_iter,
                feature_names=feature_names,
                **kwargs,
            )

        explanation_df = explanation_df if explanation_df is not None else X
        if explanation_df is None or metadata_df is None:
            raise ValueError("fit requires explanation_df and metadata_df, or X and y to generate a surrogate")

        self.app_id = app_id or self.app_id
        self.model_name = model_name or self.model_name
        self.variant = variant or self.variant
        if self.app_id is None or self.model_name is None:
            raise ValueError("app_id and model_name are required")

        row = explanation_df[explanation_df["appId"] == self.app_id]
        if "model" in row.columns:
            row = row[row["model"] == self.model_name]
        if "variant" in row.columns:
            variant_row = row[row["variant"] == self.variant]
            if not variant_row.empty:
                row = variant_row
        if row.empty:
            raise ValueError(
                f"No logistic-regression explanation found for appId={self.app_id}, "
                f"model_name={self.model_name}, variant={self.variant}"
            )
        self.explanation_df = explanation_df.copy()
        self.explanation_row = row.iloc[0]

        meta_row = metadata_df[metadata_df["appId"] == self.app_id]
        if meta_row.empty:
            raise ValueError(f"No metadata found for appId={self.app_id}")
        self.metadata_df = metadata_df.copy()
        self.metadata_row = meta_row.iloc[0]

        self.fidelity = float(self.explanation_row.get("fidelity", 0.0))
        self._intercept_norm = float(self.explanation_row.get("intercept", 0.0))
        self._parse_coefficients()
        self.is_fitted = True
        return self

    def _fit_generated(
        self,
        X: ArrayLike,
        y: ArrayLike,
        *,
        app_id: Optional[str] = None,
        model_name: Optional[str] = None,
        variant: Optional[str] = None,
        top_k: Optional[int] = None,
        C: Optional[float] = None,
        random_state: Optional[int] = None,
        max_iter: Optional[int] = None,
        feature_names: Optional[List[str]] = None,
        **kwargs,
    ):
        """Generate a logistic-regression surrogate from feature rows and AI predictions."""
        if X is None or y is None:
            raise ValueError("Generated logistic-regression surrogate fitting requires X and y")
        from .generator import _metadata_from_data, generate_logistic_regression_table

        self.app_id = app_id or self.app_id or "custom_dataset"
        self.model_name = model_name or self.model_name or "external_model"
        self.variant = variant or self.variant
        self.top_k = self.top_k if top_k is None else top_k
        self.C = self.C if C is None else C
        self.random_state = self.random_state if random_state is None else random_state
        self.max_iter = self.max_iter if max_iter is None else max_iter
        if feature_names is not None:
            self.feature_names = feature_names
        logistic_kwargs = {**self.logistic_kwargs, **kwargs}
        self.metadata_df = _metadata_from_data(
            np.asarray(X, dtype=float),
            app_id=self.app_id,
            feature_names=self.feature_names,
        )
        self.explanation_df = generate_logistic_regression_table(
            X,
            y,
            app_id=self.app_id,
            model_name=self.model_name,
            variants=(self.variant,),
            top_k=self.top_k,
            C=self.C,
            random_state=self.random_state,
            max_iter=self.max_iter,
            **logistic_kwargs,
        )
        return self.fit(explanation_df=self.explanation_df, metadata_df=self.metadata_df)

    def _parse_coefficients(self) -> None:
        coef_keys = [
            key for key in self.explanation_row.index
            if key.startswith("coef_") and pd.notna(self.explanation_row[key])
        ]

        buckets = {}
        for key in coef_keys:
            value = float(self.explanation_row[key])
            name = key.replace("coef_", "")
            if "=" in name:
                base, cat = name.split("=")
                idx = int(base[1:])
                buckets.setdefault(idx, []).append(("cat", int(cat), value, name))
            else:
                idx = int(name[1:])
                buckets.setdefault(idx, []).append(("cont", None, value, name))

        intercept = float(self._intercept_norm)
        raw_coef_map = {}

        for idx, items in buckets.items():
            conts = [item for item in items if item[0] == "cont"]
            cats = [item for item in items if item[0] == "cat"]

            if len(conts) == 1 and len(cats) == 0:
                c_norm = conts[0][2]
                vmin = self.metadata_row.get(f"v{idx}_min", None)
                vmax = self.metadata_row.get(f"v{idx}_max", None)
                if vmin is None or vmax is None or not np.isfinite(vmin) or not np.isfinite(vmax) or vmax == vmin:
                    raw_coef_map[f"a{idx}"] = c_norm
                else:
                    scale = vmax - vmin
                    raw_coef_map[f"a{idx}"] = c_norm / scale
                    intercept -= vmin * c_norm / scale
            elif len(cats) == 2 and all(cat[1] in (0, 1) for cat in cats):
                c0 = next((value for (_kind, cat, value, _name) in cats if cat == 0), None)
                c1 = next((value for (_kind, cat, value, _name) in cats if cat == 1), None)
                if c0 is not None and c1 is not None and np.isfinite(c0) and np.isfinite(c1):
                    intercept += c0
                    raw_coef_map[f"a{idx}"] = c1 - c0
                else:
                    for (_kind, cat, value, _name) in cats:
                        raw_coef_map[f"a{idx}={cat}"] = value
            else:
                for (_kind, cat, value, _name) in items:
                    raw_coef_map[f"a{idx}" if cat is None else f"a{idx}={cat}"] = value

        def sort_key(key: str):
            if "=" in key:
                base, cat = key.split("=")
                return (int(base[1:]), 1, int(cat), key)
            return (int(key[1:]), 0, -1, key)

        self.intercept = float(intercept)
        self.coefficients = OrderedDict((key, raw_coef_map[key]) for key in sorted(raw_coef_map, key=sort_key))

    def _linear_score(self, raw_input: Union[List, np.ndarray]) -> float:
        raw_array = np.asarray(raw_input)
        score = float(self.intercept)
        for key, coef in self.coefficients.items():
            if "=" in key:
                base, cat_idx = key.split("=")
                col_idx = int(base[1:])
                value = 1.0 if int(raw_array[col_idx]) == int(cat_idx) else 0.0
            else:
                col_idx = int(key[1:])
                value = float(raw_array[col_idx])
            score += float(coef) * value
        return score

    def predict(self, instances: ArrayLike) -> np.ndarray:
        """Return surrogate probabilities for raw feature vectors."""
        self._require_fitted()
        rows = ensure_2d(instances)
        return np.asarray([1.0 / (1.0 + np.exp(-self._linear_score(row))) for row in rows], dtype=float)

    def apply(self, raw_input: Union[List, np.ndarray]) -> float:
        """Return the surrogate probability for a single raw feature vector."""
        return float(self.predict([raw_input])[0])

    def apply_batch(self, instances: List[Union[List, np.ndarray]]) -> List[float]:
        """Apply the surrogate model to multiple instances."""
        return self.predict(instances).tolist()

    def explain(self, instances: ArrayLike) -> XAIAdapterResult:
        """Return per-feature logistic-regression contributions for instances."""
        self._require_fitted()
        rows = ensure_2d(instances)
        n_features = rows.shape[1]
        values = np.zeros((rows.shape[0], n_features), dtype=float)

        for row_idx, raw_input in enumerate(rows):
            for key, coef in self.coefficients.items():
                if "=" in key:
                    base, cat_idx = key.split("=")
                    col_idx = int(base[1:])
                    feature_value = 1.0 if int(raw_input[col_idx]) == int(cat_idx) else 0.0
                else:
                    col_idx = int(key[1:])
                    feature_value = float(raw_input[col_idx])
                if col_idx < n_features:
                    values[row_idx, col_idx] += float(coef) * feature_value

        return XAIAdapterResult(
            values=values,
            base_values=np.full(rows.shape[0], self.intercept, dtype=float),
            method=self.method_name,
            metadata={
                "app_id": self.app_id,
                "model_name": self.model_name,
                "variant": self.variant,
                "fidelity": self.fidelity,
                "predictions": self.predict(rows),
                "coefficients": self.get_coefficients(),
            },
        )

    def get_coefficients(self) -> Dict[str, float]:
        """Return raw-space coefficients."""
        return dict(self.coefficients)

    def get_intercept(self) -> float:
        """Return raw-space intercept."""
        return self.intercept

    def to_explanation_table(self) -> pd.DataFrame:
        """Return the CoXAM-style logistic-regression explanation table."""
        self._require_fitted()
        return self.explanation_df.copy()

    def to_metadata_table(self) -> pd.DataFrame:
        """Return the metadata table used by this surrogate."""
        self._require_fitted()
        return self.metadata_df.copy()

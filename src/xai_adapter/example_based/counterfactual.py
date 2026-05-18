"""Counterfactual explanation adapters (Wachter-style and DiCE)."""

from __future__ import annotations

from typing import Any, Callable, List, Optional

import numpy as np

from ..base import (
    ArrayLike,
    XAIAdapterResult,
    ensure_2d,
)
from .base import ExampleBasedAdapter


class CounterfactualAdapter(ExampleBasedAdapter):
    """Wachter et al. (2017) counterfactual explanations via alibi.

    Finds the nearest point in feature space that the model classifies
    differently (or as a specified target class).

    ``explain()`` returns an ``XAIAdapterResult`` where:
      - ``values``: ``(n_instances, n_features)`` delta from original to CF
        (CF - X).
      - ``metadata["counterfactuals"]``: raw CF feature vectors.
      - ``metadata["success"]``: bool array indicating a valid CF was found.

    Parameters
    ----------
    predict_fn : callable
        Probability function ``f(X_np) -> (n, n_classes)`` array.
    feature_range : tuple
        ``(lower_bounds, upper_bounds)`` per feature for alibi's optimizer.
    target_class : int or 'other'
        Class the counterfactual should be predicted as.
    max_iter : int
        Maximum optimization iterations (default 1000).
    shape : tuple, optional
        Input shape passed to alibi; inferred from data if not given.
    target : int
        Positive class index kept for API consistency (default 1).
    """

    method_name = "counterfactual"

    def __init__(
        self,
        *,
        predict_fn: Callable[[np.ndarray], np.ndarray],
        feature_range: tuple,
        target_class: Any = "other",
        max_iter: int = 1000,
        target: int = 1,
        preprocessing_fn: Optional[Callable] = None,
        postprocessing_fn: Optional[Callable] = None,
        shape: Optional[tuple] = None,
    ):
        super().__init__(
            target=target,
            preprocessing_fn=preprocessing_fn,
            postprocessing_fn=postprocessing_fn,
        )
        try:
            from alibi.explainers import CounterFactual  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "alibi is required for CounterfactualAdapter. "
                "Install with: pip install alibi"
            ) from exc

        self.predict_fn = predict_fn
        self.feature_range = feature_range
        self.target_class = target_class
        self.max_iter = max_iter
        self._shape = shape
        self._explainer = None
        self.is_fitted = True

    def _get_explainer(self, n_features: int):
        if self._explainer is not None:
            return self._explainer
        from alibi.explainers import CounterFactual
        shape = self._shape or (1, n_features)
        self._explainer = CounterFactual(
            predict_fn=lambda x: self.predict_fn(self.preprocessing_fn(x)),
            shape=shape,
            feature_range=self.feature_range,
            target_class=self.target_class,
            max_iter=self.max_iter,
        )
        return self._explainer

    def fit(self, X: ArrayLike = None, y: ArrayLike = None, **kwargs):
        if X is not None:
            x = ensure_2d(self.preprocessing_fn(X))
            self._get_explainer(x.shape[1])
        self.is_fitted = True
        return self

    def explain(self, instances: ArrayLike) -> XAIAdapterResult:
        self._require_fitted()
        raw = ensure_2d(instances)
        x = ensure_2d(self.preprocessing_fn(raw))
        n, n_features = x.shape
        explainer = self._get_explainer(n_features)

        deltas = np.zeros_like(x, dtype=float)
        cfs = np.full_like(x, np.nan, dtype=float)
        success = np.zeros(n, dtype=bool)

        for i in range(n):
            exp = explainer.explain(x[i : i + 1])
            cf = exp.cf
            if cf is not None and cf.get("X") is not None:
                cf_x = np.asarray(cf["X"]).reshape(n_features)
                cfs[i] = cf_x
                deltas[i] = cf_x - x[i]
                success[i] = True

        return XAIAdapterResult(
            values=deltas,
            base_values=np.zeros(n, dtype=float),
            method=self.method_name,
            metadata={"counterfactuals": cfs, "success": success},
        )


class DiCEAdapter(ExampleBasedAdapter):
    """Diverse Counterfactual Explanations (DiCE) via dice-ml.

    Generates a set of *diverse* counterfactual examples for each instance.

    ``explain()`` returns an ``XAIAdapterResult`` where:
      - ``values``: ``(n_instances, n_features)`` delta of the *first* (closest)
        CF per instance (CF[0] - X).
      - ``metadata["all_cfs"]``: list of ``(n_cfs, n_features)`` arrays, one
        per input instance.
      - ``metadata["success"]``: bool array (True if ≥1 valid CF was found).

    Parameters
    ----------
    predict_fn : callable
        Probability function ``f(X_np) -> (n, n_classes)`` array.
    training_data : array-like
        Background dataset used by DiCE for data range inference.
    feature_names : list of str
        Feature column names.
    outcome_name : str
        Target column label (default 'outcome').
    n_cfs : int
        Number of diverse CFs to generate per instance (default 3).
    method : str
        DiCE generation backend: 'random', 'genetic', or 'kdtree'.
    desired_class : int or 'opposite'
        Target class for all generated CFs (default 'opposite').
    target : int
        Column index of the positive class in predict_fn output (default 1).
    """

    method_name = "dice"

    def __init__(
        self,
        *,
        predict_fn: Callable[[np.ndarray], np.ndarray],
        training_data: ArrayLike,
        feature_names: List[str],
        outcome_name: str = "outcome",
        n_cfs: int = 3,
        method: str = "random",
        desired_class: Any = "opposite",
        target: int = 1,
        preprocessing_fn: Optional[Callable] = None,
        postprocessing_fn: Optional[Callable] = None,
    ):
        super().__init__(
            target=target,
            preprocessing_fn=preprocessing_fn,
            postprocessing_fn=postprocessing_fn,
        )
        try:
            import dice_ml  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "dice-ml is required for DiCEAdapter. "
                "Install with: pip install dice-ml"
            ) from exc

        self.predict_fn = predict_fn
        self.feature_names = list(feature_names)
        self.outcome_name = outcome_name
        self.n_cfs = int(n_cfs)
        self.method = method
        self.desired_class = desired_class

        self._dice, self._exp = self._build(training_data)
        self.is_fitted = True

    def _build(self, training_data: ArrayLike):
        import dice_ml
        import pandas as pd

        X_bg = ensure_2d(self.preprocessing_fn(training_data))
        df = pd.DataFrame(X_bg, columns=self.feature_names)
        df[self.outcome_name] = 0

        data_obj = dice_ml.Data(
            dataframe=df,
            continuous_features=self.feature_names,
            outcome_name=self.outcome_name,
        )

        def wrapped_predict(df_input):
            arr = df_input[self.feature_names].values.astype(float)
            return self.predict_fn(arr)[:, self.target].tolist()

        model_obj = dice_ml.Model(model=wrapped_predict, backend="sklearn")
        exp = dice_ml.Dice(data_obj, model_obj, method=self.method)
        return data_obj, exp

    def fit(self, X: ArrayLike = None, y: ArrayLike = None, **kwargs):
        self.is_fitted = True
        return self

    def explain(self, instances: ArrayLike) -> XAIAdapterResult:
        self._require_fitted()
        import pandas as pd

        raw = ensure_2d(instances)
        x = ensure_2d(self.preprocessing_fn(raw))
        n, n_features = x.shape

        deltas = np.zeros((n, n_features), dtype=float)
        all_cfs: List[np.ndarray] = []
        success = np.zeros(n, dtype=bool)

        for i in range(n):
            x_df = pd.DataFrame(x[i : i + 1], columns=self.feature_names)
            try:
                dice_exp = self._exp.generate_counterfactuals(
                    x_df,
                    total_CFs=self.n_cfs,
                    desired_class=self.desired_class,
                )
                cf_df = dice_exp.cf_examples_list[0].final_cfs_df
                if cf_df is not None and len(cf_df) > 0:
                    cf_arr = cf_df[self.feature_names].values.astype(float)
                    all_cfs.append(cf_arr)
                    deltas[i] = cf_arr[0] - x[i]
                    success[i] = True
                else:
                    all_cfs.append(np.empty((0, n_features)))
            except Exception:
                all_cfs.append(np.empty((0, n_features)))

        return XAIAdapterResult(
            values=deltas,
            base_values=np.zeros(n, dtype=float),
            method=self.method_name,
            metadata={"all_cfs": all_cfs, "success": success, "n_cfs": self.n_cfs},
        )


__all__ = ["CounterfactualAdapter", "DiCEAdapter"]

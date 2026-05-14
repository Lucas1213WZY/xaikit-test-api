"""Native SHAP explainer wrappers (TreeExplainer, LinearExplainer, DeepExplainer, GradientExplainer)."""

from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np

from ..base import (
    ArrayLike,
    PostprocessFn,
    PreprocessFn,
    XAIAdapterResult,
    ensure_2d,
)
from .base import LocalAttribution


def _import_shap():
    try:
        import shap
        return shap
    except ImportError as exc:
        raise ImportError("SHAP is required. Install with: pip install shap") from exc


def _extract_shap_values(shap_values, target: int) -> tuple[np.ndarray, np.ndarray]:
    """Extract attribution values and base_values from a shap.Explanation or raw array."""
    if hasattr(shap_values, "values"):
        vals = np.asarray(shap_values.values)
        base = np.asarray(getattr(shap_values, "base_values", np.zeros(vals.shape[0])))
    else:
        vals = np.asarray(shap_values)
        base = np.zeros(vals.shape[0] if vals.ndim >= 2 else 1)

    if vals.ndim == 3:
        vals = vals[:, :, target]
    if base.ndim == 2:
        base = base[:, target]
    return vals, base.reshape(-1)


class ShapTreeExplainer(LocalAttribution):
    """Wrapper around shap.TreeExplainer for tree-based models (XGBoost, LightGBM, sklearn forests)."""

    method_name = "shap_tree"

    def __init__(
        self,
        *,
        model: Any,
        background_data: Optional[ArrayLike] = None,
        feature_perturbation: str = "interventional",
        target: int = 1,
        preprocessing_fn: Optional[PreprocessFn] = None,
        postprocessing_fn: Optional[PostprocessFn] = None,
    ):
        super().__init__(
            target=target,
            preprocessing_fn=preprocessing_fn,
            postprocessing_fn=postprocessing_fn,
        )
        self.model = model
        self.background_data = background_data
        self.feature_perturbation = feature_perturbation
        self.explainer = None
        self.fit()

    def fit(self, X: ArrayLike = None, y: ArrayLike = None, **kwargs):
        shap = _import_shap()
        background = None
        if X is not None:
            background = ensure_2d(self.preprocessing_fn(X))
        elif self.background_data is not None:
            background = ensure_2d(self.preprocessing_fn(self.background_data))

        self.explainer = shap.TreeExplainer(
            self.model,
            data=background,
            feature_perturbation=self.feature_perturbation,
        )
        self.is_fitted = True
        return self

    def explain(self, instances: ArrayLike) -> XAIAdapterResult:
        self._require_fitted()
        raw_instances = ensure_2d(instances)
        x = ensure_2d(self.preprocessing_fn(raw_instances))

        shap_values = self.explainer(x)
        vals, base = _extract_shap_values(shap_values, self.target)

        vals = self._postprocess_values(raw_instances, vals)
        if base.size == 1 and vals.shape[0] > 1:
            base = np.full(vals.shape[0], float(base[0]), dtype=float)
        return XAIAdapterResult(
            values=vals,
            base_values=base.astype(float),
            method=self.method_name,
            metadata={"feature_perturbation": self.feature_perturbation},
        )


class ShapLinearExplainer(LocalAttribution):
    """Wrapper around shap.LinearExplainer for linear/logistic regression models."""

    method_name = "shap_linear"

    def __init__(
        self,
        *,
        model: Any,
        background_data: ArrayLike,
        nsamples: int = 200,
        target: int = 1,
        preprocessing_fn: Optional[PreprocessFn] = None,
        postprocessing_fn: Optional[PostprocessFn] = None,
    ):
        super().__init__(
            target=target,
            preprocessing_fn=preprocessing_fn,
            postprocessing_fn=postprocessing_fn,
        )
        self.model = model
        self.nsamples = int(nsamples)
        self.explainer = None
        self.fit(background_data)

    def fit(self, X: ArrayLike, y: ArrayLike = None, **kwargs):
        shap = _import_shap()
        background = ensure_2d(self.preprocessing_fn(X))
        masker = shap.maskers.Independent(background, max_samples=self.nsamples)
        self.explainer = shap.LinearExplainer(self.model, masker=masker)
        self.is_fitted = True
        return self

    def explain(self, instances: ArrayLike) -> XAIAdapterResult:
        self._require_fitted()
        raw_instances = ensure_2d(instances)
        x = ensure_2d(self.preprocessing_fn(raw_instances))

        shap_values = self.explainer(x)
        vals, base = _extract_shap_values(shap_values, self.target)

        vals = self._postprocess_values(raw_instances, vals)
        if base.size == 1 and vals.shape[0] > 1:
            base = np.full(vals.shape[0], float(base[0]), dtype=float)
        return XAIAdapterResult(
            values=vals,
            base_values=base.astype(float),
            method=self.method_name,
            metadata={"nsamples": self.nsamples},
        )


def _to_numpy(x: Any) -> np.ndarray:
    """Detach and convert any tensor-like or array-like to a numpy array."""
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        x = x.numpy()
    return np.asarray(x)


def _extract_base_values(explainer: Any, target: int, n: int) -> np.ndarray:
    """Pull expected_value out of a shap explainer and broadcast to shape (n,)."""
    base_raw = getattr(explainer, "expected_value", None)
    if base_raw is None:
        return np.zeros(n, dtype=float)
    base_arr = np.asarray(base_raw)
    if base_arr.ndim >= 1 and base_arr.size > target:
        scalar = float(base_arr.flat[target])
    else:
        scalar = float(base_arr.flat[0])
    return np.full(n, scalar, dtype=float)


class ShapDeepExplainer(LocalAttribution):
    """Wrapper around shap.DeepExplainer for PyTorch / TensorFlow deep learning models.

    background_data and the instances passed to explain() must already be in
    model-native format (torch.Tensor for PyTorch, tf.Tensor for TF).
    preprocessing_fn is applied only when explicitly provided and should map
    raw input to the model-native format — it must NOT downcast tensors to numpy.
    """

    method_name = "shap_deep"

    def __init__(
        self,
        *,
        model: Any,
        background_data: ArrayLike,
        target: int = 1,
        preprocessing_fn: Optional[PreprocessFn] = None,
        postprocessing_fn: Optional[PostprocessFn] = None,
    ):
        super().__init__(
            target=target,
            preprocessing_fn=preprocessing_fn,
            postprocessing_fn=postprocessing_fn,
        )
        self._has_custom_preprocessing = preprocessing_fn is not None
        self.model = model
        self.explainer = None
        self.fit(background_data)

    def _prepare(self, X: Any) -> Any:
        """Apply preprocessing only when a custom fn was provided; otherwise pass through."""
        return self.preprocessing_fn(X) if self._has_custom_preprocessing else X

    def fit(self, X: ArrayLike, y: ArrayLike = None, **kwargs):
        shap = _import_shap()
        # Pass data in its native format — shap.DeepExplainer runs the model internally
        # to compute expected_value, so numpy arrays break PyTorch models.
        self.explainer = shap.DeepExplainer(self.model, self._prepare(X))
        self.is_fitted = True
        return self

    def explain(self, instances: ArrayLike) -> XAIAdapterResult:
        self._require_fitted()
        x = self._prepare(instances)
        shap_values = self.explainer.shap_values(x)

        if isinstance(shap_values, list):
            vals = np.asarray(shap_values[self.target])
        else:
            vals = np.asarray(shap_values)

        vals = ensure_2d(vals)
        base = _extract_base_values(self.explainer, self.target, vals.shape[0])

        raw_np = ensure_2d(_to_numpy(instances))
        vals = self._postprocess_values(raw_np, vals)
        return XAIAdapterResult(
            values=vals,
            base_values=base,
            method=self.method_name,
            metadata={},
        )


class ShapGradientExplainer(LocalAttribution):
    """Wrapper around shap.GradientExplainer for PyTorch / TensorFlow models.

    background_data and instances must be in model-native format (torch.Tensor
    for PyTorch).  See ShapDeepExplainer for the preprocessing_fn contract.
    """

    method_name = "shap_gradient"

    def __init__(
        self,
        *,
        model: Any,
        background_data: ArrayLike,
        target: int = 1,
        preprocessing_fn: Optional[PreprocessFn] = None,
        postprocessing_fn: Optional[PostprocessFn] = None,
    ):
        super().__init__(
            target=target,
            preprocessing_fn=preprocessing_fn,
            postprocessing_fn=postprocessing_fn,
        )
        self._has_custom_preprocessing = preprocessing_fn is not None
        self.model = model
        self.explainer = None
        self.fit(background_data)

    def _prepare(self, X: Any) -> Any:
        return self.preprocessing_fn(X) if self._has_custom_preprocessing else X

    def fit(self, X: ArrayLike, y: ArrayLike = None, **kwargs):
        shap = _import_shap()
        self.explainer = shap.GradientExplainer(self.model, self._prepare(X))
        self.is_fitted = True
        return self

    def explain(self, instances: ArrayLike) -> XAIAdapterResult:
        self._require_fitted()
        x = self._prepare(instances)
        shap_values = self.explainer.shap_values(x)

        if isinstance(shap_values, list):
            vals = np.asarray(shap_values[self.target])
        else:
            vals = np.asarray(shap_values)

        vals = ensure_2d(vals)
        base = _extract_base_values(self.explainer, self.target, vals.shape[0])

        raw_np = ensure_2d(_to_numpy(instances))
        vals = self._postprocess_values(raw_np, vals)
        return XAIAdapterResult(
            values=vals,
            base_values=base,
            method=self.method_name,
            metadata={},
        )


__all__ = [
    "ShapTreeExplainer",
    "ShapLinearExplainer",
    "ShapDeepExplainer",
    "ShapGradientExplainer",
]

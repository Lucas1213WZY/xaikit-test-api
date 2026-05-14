"""Captum-backed attribution methods."""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from ..base import (
    ArrayLike,
    PostprocessFn,
    PreprocessFn,
    XAIAdapterResult,
    baseline_from_data,
    ensure_2d,
    select_target,
)
from .base import LocalAttribution


class CaptumAttribution(LocalAttribution):
    """Base class for Captum attribution methods."""

    captum_attr_cls = None
    method_name = "captum"

    def __init__(
        self,
        *,
        model,
        predict_fn: Callable[[np.ndarray], np.ndarray],
        background_data: Optional[ArrayLike] = None,
        baseline: str = "mean",
        device: str = "cpu",
        target: int = 1,
        preprocessing_fn: Optional[PreprocessFn] = None,
        postprocessing_fn: Optional[PostprocessFn] = None,
        **attribute_kwargs,
    ):
        super().__init__(
            target=target,
            preprocessing_fn=preprocessing_fn,
            postprocessing_fn=postprocessing_fn,
        )
        try:
            import torch
        except ImportError as exc:
            raise ImportError("PyTorch and Captum are required. Install with: pip install torch captum") from exc
        if self.captum_attr_cls is None:
            raise TypeError("captum_attr_cls must be defined by subclasses")

        self.torch = torch
        self.model = model
        self.predict_fn = predict_fn
        self.baseline = baseline
        self.device = device
        self.attribute_kwargs = dict(attribute_kwargs)
        self.model.eval()
        self.model.to(device)
        self.attr = self.captum_attr_cls(self.model)
        self.baseline_tensor = None
        self.baseline_value = 0.0
        self.is_fitted = True
        if background_data is not None:
            self.fit(background_data)

    def fit(self, X: ArrayLike = None, y: ArrayLike = None, **kwargs):
        """Fit the Captum baseline from background data."""
        if X is not None:
            baseline_vec = baseline_from_data(self.preprocessing_fn(X), self.baseline)
            self.baseline_tensor = self.torch.tensor(
                baseline_vec,
                dtype=self.torch.float32,
                device=self.device,
            ).reshape(1, -1)
            self.baseline_value = float(select_target(self.predict_fn(baseline_vec.reshape(1, -1)), self.target)[0])
        self.is_fitted = True
        return self

    def explain(self, instances: ArrayLike) -> XAIAdapterResult:
        self._require_fitted()
        raw_instances = ensure_2d(instances)
        x_np = ensure_2d(self.preprocessing_fn(raw_instances))
        x = self.torch.tensor(x_np, dtype=self.torch.float32, device=self.device)

        baselines = (
            self.baseline_tensor.repeat(x.shape[0], 1)
            if self.baseline_tensor is not None
            else self.torch.zeros_like(x)
        )
        attributions = self.attr.attribute(
            x,
            baselines=baselines,
            target=self.target,
            **self.attribute_kwargs,
        )
        if isinstance(attributions, tuple):
            attributions = attributions[0]

        values = self._postprocess_values(raw_instances, attributions.detach().cpu().numpy())
        return XAIAdapterResult(
            values=values,
            base_values=np.full(values.shape[0], self.baseline_value, dtype=float),
            method=self.method_name,
            metadata={"baseline": baselines.detach().cpu().numpy()},
        )


class DeepLift(CaptumAttribution):
    """Captum DeepLift method."""

    method_name = "deeplift"

    def __init__(self, **kwargs):
        try:
            from captum.attr import DeepLift as CaptumDeepLift
        except ImportError as exc:
            raise ImportError("Captum is required for DeepLift. Install with: pip install captum") from exc
        self.captum_attr_cls = CaptumDeepLift
        super().__init__(**kwargs)


class IntegratedGradients(CaptumAttribution):
    """Captum Integrated Gradients method."""

    method_name = "integrated_gradients"

    def __init__(self, *, n_steps: int = 50, **kwargs):
        try:
            from captum.attr import IntegratedGradients as CaptumIntegratedGradients
        except ImportError as exc:
            raise ImportError("Captum is required for IntegratedGradients. Install with: pip install captum") from exc
        self.captum_attr_cls = CaptumIntegratedGradients
        kwargs.setdefault("n_steps", n_steps)
        super().__init__(**kwargs)


class GradientInput(CaptumAttribution):
    """Captum InputXGradient (gradient × input) attribution.

    Unlike DeepLift and IntegratedGradients, this method requires no baseline
    — the attribution is purely ∂f/∂x × x — so ``predict_fn`` and
    ``background_data`` are not used.
    """

    method_name = "gradient_input"

    def __init__(self, *, model, predict_fn: Optional[Callable] = None, **kwargs):
        try:
            from captum.attr import InputXGradient
        except ImportError as exc:
            raise ImportError("Captum is required for GradientInput. Install with: pip install captum") from exc
        self.captum_attr_cls = InputXGradient
        # predict_fn is unused; pass a no-op so the base __init__ is satisfied.
        super().__init__(model=model, predict_fn=predict_fn or (lambda x: x), **kwargs)

    def fit(self, X: ArrayLike = None, y: ArrayLike = None, **kwargs):
        """No-op — GradientInput needs no baseline."""
        self.is_fitted = True
        return self

    def explain(self, instances: ArrayLike) -> XAIAdapterResult:
        """Return gradient × input attributions as an XAIAdapterResult.

        Conversion chain: numpy input → torch tensor → Captum
        InputXGradient → ``.detach().cpu().numpy()`` → XAIAdapterResult.
        """
        self._require_fitted()
        raw_instances = ensure_2d(instances)
        x_np = ensure_2d(self.preprocessing_fn(raw_instances))
        x = self.torch.tensor(x_np, dtype=self.torch.float32, device=self.device)

        attributions = self.attr.attribute(x, target=self.target, **self.attribute_kwargs)
        if isinstance(attributions, tuple):
            attributions = attributions[0]

        values = self._postprocess_values(raw_instances, attributions.detach().cpu().numpy())
        return XAIAdapterResult(
            values=values,
            base_values=np.zeros(values.shape[0], dtype=float),
            method=self.method_name,
        )


__all__ = [
    "CaptumAttribution",
    "DeepLift",
    "GradientInput",
    "IntegratedGradients",
]

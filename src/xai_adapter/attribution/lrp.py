"""Layer-wise Relevance Propagation (LRP) adapter via Captum."""

from __future__ import annotations

from typing import Any, Callable, Optional

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


class LRPAdapter(LocalAttribution):
    """Layer-wise Relevance Propagation via captum.attr.LRP.

    LRP decomposes a model's output score back through the network layers,
    assigning a relevance score to each input feature. Requires a PyTorch
    model with supported layer types (Linear, Conv, BatchNorm, etc.).

    Parameters
    ----------
    model : torch.nn.Module
        PyTorch model to explain.
    predict_fn : callable
        Model prediction function ``f(X_np) -> probabilities``.
    background_data : array-like, optional
        Reference data for computing the zero-relevance baseline.
    baseline : str
        How to aggregate background_data into a baseline vector
        ('mean', 'median', or 'zeros').
    device : str
        PyTorch device string (default 'cpu').
    target : int
        Output class index to explain (default 1).
    rule : str
        LRP propagation rule name. Supported values mirror captum's
        rule names: 'epsilon', 'alpha1beta0', 'alpha2beta1'.
        When None, captum uses its own default per layer type.
    """

    method_name = "lrp"

    def __init__(
        self,
        *,
        model: Any,
        predict_fn: Callable[[np.ndarray], np.ndarray],
        background_data: Optional[ArrayLike] = None,
        baseline: str = "mean",
        device: str = "cpu",
        target: int = 1,
        rule: Optional[str] = None,
        preprocessing_fn: Optional[PreprocessFn] = None,
        postprocessing_fn: Optional[PostprocessFn] = None,
    ):
        super().__init__(
            target=target,
            preprocessing_fn=preprocessing_fn,
            postprocessing_fn=postprocessing_fn,
        )
        try:
            import torch
            from captum.attr import LRP
        except ImportError as exc:
            raise ImportError(
                "PyTorch and Captum are required for LRPAdapter. "
                "Install with: pip install torch captum"
            ) from exc

        self.torch = torch
        self.model = model
        self.predict_fn = predict_fn
        self.baseline = baseline
        self.device = device
        self.rule = rule
        self.model.eval()
        self.model.to(device)
        self.attr = LRP(self.model)
        self.baseline_tensor = None
        self.baseline_value = 0.0
        self.is_fitted = True

        if background_data is not None:
            self.fit(background_data)

    def fit(self, X: ArrayLike, y: ArrayLike = None, **kwargs):
        baseline_vec = baseline_from_data(self.preprocessing_fn(X), self.baseline)
        self.baseline_tensor = self.torch.tensor(
            baseline_vec, dtype=self.torch.float32, device=self.device
        ).reshape(1, -1)
        self.baseline_value = float(
            select_target(self.predict_fn(baseline_vec.reshape(1, -1)), self.target)[0]
        )
        self.is_fitted = True
        return self

    def explain(self, instances: ArrayLike) -> XAIAdapterResult:
        self._require_fitted()
        raw_instances = ensure_2d(instances)
        x_np = ensure_2d(self.preprocessing_fn(raw_instances))
        x = self.torch.tensor(x_np, dtype=self.torch.float32, device=self.device)
        x.requires_grad_(True)

        baselines = (
            self.baseline_tensor.repeat(x.shape[0], 1)
            if self.baseline_tensor is not None
            else self.torch.zeros_like(x)
        )

        attr_kwargs = {"baselines": baselines, "target": self.target}
        if self.rule is not None:
            attr_kwargs["rule_dict"] = self._make_rule_dict(x)

        attributions = self.attr.attribute(x, **attr_kwargs)
        if isinstance(attributions, tuple):
            attributions = attributions[0]

        values = self._postprocess_values(raw_instances, attributions.detach().cpu().numpy())
        return XAIAdapterResult(
            values=values,
            base_values=np.full(values.shape[0], self.baseline_value, dtype=float),
            method=self.method_name,
            metadata={"rule": self.rule, "baseline": self.baseline},
        )

    def _make_rule_dict(self, x: Any) -> dict:
        """Build a Captum rule_dict applying self.rule to all Linear layers."""
        from captum.attr import LRP
        rule_map = {
            "epsilon": "EpsilonRule",
            "alpha1beta0": "Alpha1_Beta0_Rule",
            "alpha2beta1": "Alpha2_Beta1_Rule",
        }
        rule_name = rule_map.get(self.rule, self.rule)
        import captum.attr._utils.lrp_rules as lrp_rules
        rule_cls = getattr(lrp_rules, rule_name, None)
        if rule_cls is None:
            return {}
        import torch.nn as nn
        return {nn.Linear: rule_cls()}


__all__ = ["LRPAdapter"]

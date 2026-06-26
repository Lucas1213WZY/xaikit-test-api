"""Registry for XAI adapters."""

from __future__ import annotations

from typing import Any, Dict, Type


class XAIAdapterRegistry:
    """Small registry for adapter classes."""

    def __init__(self):
        self._registry: Dict[str, Any] = {}

    def register(self, name: str, adapter_class: Type, *aliases: str) -> None:
        """Register an adapter class."""
        self._registry[name.lower()] = adapter_class
        for alias in aliases:
            self._registry[alias.lower()] = adapter_class

    def register_custom(self, name: str, algorithm: Any, *aliases: str) -> None:
        """Register a user-provided function or object as an XAI method."""
        from .attribution import CustomAttribution

        def factory(**kwargs):
            return CustomAttribution(algorithm, method_name=name, **kwargs)

        self.register(name, factory, *aliases)

    def get_class(self, name: str) -> Any:
        """Return the adapter class or factory for a registered name."""
        key = name.lower()
        if key not in self._registry:
            raise ValueError(f"Unknown XAI adapter '{name}'. Available: {self.list_available()}")
        return self._registry[key]

    def create(self, name: str, **kwargs):
        """Instantiate a registered adapter."""
        return self.get_class(name)(**kwargs)

    def list_available(self) -> list[str]:
        """List registered adapter names."""
        return sorted(self._registry.keys())

    def is_registered(self, name: str) -> bool:
        """Check whether a method name or alias is registered."""
        return name.lower() in self._registry


_GLOBAL_REGISTRY = None


def get_adapter_registry() -> XAIAdapterRegistry:
    """Return the global adapter registry."""
    global _GLOBAL_REGISTRY
    if _GLOBAL_REGISTRY is None:
        from .dataset import PrecomputedCSVXAIMethod
        from .attribution import (
            DeepLift,
            GradientInput,
            IntegratedGradients,
            KernelShap,
            LeaveOneFeatureOut,
            Lime,
            LRPAdapter,
            ShapDeepExplainer,
            ShapGradientExplainer,
            ShapLinearExplainer,
            ShapTreeExplainer,
            Sim2RealPropertyAttribution,
            SklearnFeatureImportance,
        )
        from .concept import TCAVAdapter
        from .example_based import CounterfactualAdapter, DiCEAdapter, PrototypesAdapter
        from .surrogate import (
            AnchorsAdapter,
            DecisionTreeSurrogateMethod,
            LogisticRegressionSurrogateMethod,
            RuleListSurrogateMethod,
            RuleSetSurrogateMethod,
        )
        from .interpret import EBMAdapter

        registry = XAIAdapterRegistry()
        registry.register("lofo", LeaveOneFeatureOut, "leave_one_feature_out")
        registry.register("shap_kernel", KernelShap, "shap")
        registry.register("shap_tree", ShapTreeExplainer, "shap_treeexplainer")
        registry.register("shap_linear", ShapLinearExplainer, "shap_linearexplainer")
        registry.register("shap_deep", ShapDeepExplainer, "shap_deepexplainer")
        registry.register("shap_gradient", ShapGradientExplainer, "shap_gradientexplainer")
        registry.register("lime", Lime, "lime_tabular")
        registry.register("gradient_input", GradientInput, "gradient_x_input", "input_gradients")
        registry.register("deeplift", DeepLift, "deep_lift")
        registry.register("integrated_gradients", IntegratedGradients, "ig")
        registry.register("lrp", LRPAdapter, "layer_relevance_propagation")
        registry.register("sklearn_global", SklearnFeatureImportance, "global_feature_importance")
        registry.register("sim2real_property", Sim2RealPropertyAttribution, "property_optimized", "xaisim2real")
        registry.register("precomputed_csv", PrecomputedCSVXAIMethod, "csv", "csv_dataset", "dataset_csv")
        registry.register("decision_tree", DecisionTreeSurrogateMethod, "dt", "rules")
        registry.register("logistic_regression", LogisticRegressionSurrogateMethod, "lr", "weights")
        registry.register("rule_list", RuleListSurrogateMethod, "rulelist")
        registry.register("rule_set", RuleSetSurrogateMethod, "ruleset")
        registry.register("ebm", EBMAdapter, "interpret_ebm", "explainable_boosting")
        registry.register("tcav", TCAVAdapter)
        registry.register("counterfactual", CounterfactualAdapter, "cf", "wachter")
        registry.register("dice", DiCEAdapter, "diverse_counterfactuals")
        registry.register("anchors", AnchorsAdapter, "anchor_tabular")
        registry.register("prototypes", PrototypesAdapter, "mmd_critic", "criticisms")
        _GLOBAL_REGISTRY = registry
    return _GLOBAL_REGISTRY


def create_xai_method(name: str, **kwargs):
    """Create an XAI method adapter from the global registry."""
    # Support requesting adapters for all output classes by passing
    # target=None or target in {"all", "all_classes", "every"}.
    target = kwargs.get("target", 1)
    if target is None or (isinstance(target, str) and target.lower() in {"all", "all_classes", "every"}):
        # Try to infer number of classes from a provided predict_fn or
        # from background/training data. Default to 1 if inference fails.
        predict_fn = kwargs.get("predict_fn")
        sample = None
        # prefer explicit background/training samples if available
        for k in ("background_data", "training_data", "explanation_df"):
            if k in kwargs and kwargs[k] is not None:
                sample = kwargs[k]
                break

        n_classes = 1
        if predict_fn is not None and sample is not None:
            try:
                import numpy as _np
                # pick a small batch
                s = sample[:1] if hasattr(sample, "__len__") else sample
                preds = predict_fn(s)
                arr = _np.asarray(preds)
                if arr.ndim == 2:
                    n_classes = int(arr.shape[1])
                else:
                    n_classes = 1
            except Exception:
                n_classes = 1

        methods = []
        for i in range(n_classes):
            new_kwargs = dict(kwargs)
            new_kwargs["target"] = i
            methods.append(get_adapter_registry().create(name, **new_kwargs))
        return methods

    return get_adapter_registry().create(name, **kwargs)


def register_xai_method(name: str, adapter: Any, *aliases: str) -> None:
    """
    Register a custom XAI method.

    `adapter` can be an adapter class/factory that returns an XAIAdapter-like
    object, or a plain function/object to be wrapped as a CustomAttribution.
    """
    if isinstance(adapter, type):
        get_adapter_registry().register(name, adapter, *aliases)
        return
    get_adapter_registry().register_custom(name, adapter, *aliases)

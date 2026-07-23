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


# CoXAM surrogate methods whose explanation tables can be read from a loader.
_COXAM_METHOD_KEYS = {
    "decision_tree", "dt", "rules",
    "logistic_regression", "lr", "weights",
}


def _train_x(train_data: Any) -> Any:
    return train_data.X if hasattr(train_data, "X") else train_data


def _train_y(train_data: Any) -> Any:
    return getattr(train_data, "y", None)


def _categorical_features(train_data: Any) -> Any:
    return getattr(train_data, "categorical_feature_indices", None)


def _feature_names(train_data: Any) -> Any:
    return getattr(train_data, "feature_names", None)


def _apply_ai_model_kwargs(key: str, ai_model: Any, train_data: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Wire a trained ``ai_model`` + ``train_data`` into adapter kwargs.

    ``ai_model`` is expected to expose ``predict(...)``; gradient/relevance
    methods also use ``ai_model.model`` and prefer
    ``ai_model.forward_logits_or_probs(...)``. Adapters with no standard
    wiring get ``kwargs`` passed through as-is.
    """
    train_x = _train_x(train_data)
    predict_fn = kwargs.pop("predict_fn", ai_model.predict)

    if key in {"lofo", "leave_one_feature_out", "shap", "shap_kernel"}:
        kwargs.update(predict_fn=predict_fn, background_data=train_x)
    elif key in {"lime", "lime_tabular"}:
        kwargs.setdefault("training_data", train_x)
        kwargs.setdefault("training_labels", _train_y(train_data))
        kwargs.setdefault("categorical_features", _categorical_features(train_data))
        kwargs.setdefault("feature_names", _feature_names(train_data))
        kwargs["predict_fn"] = predict_fn
    elif key in {"gradient_input", "gradient_x_input", "input_gradients"}:
        kwargs.setdefault("forward_fn", getattr(ai_model, "forward_logits_or_probs", None))
        kwargs.update(model=ai_model.model, predict_fn=predict_fn, background_data=train_x)
    elif key in {"deeplift", "deep_lift", "integrated_gradients", "ig", "lrp", "layer_relevance_propagation"}:
        kwargs.update(model=ai_model.model, predict_fn=predict_fn, background_data=train_x)
    return kwargs


def _apply_coxam_loader(key: str, loader: Any, kwargs: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
    """Resolve a CoXAM surrogate from a data loader's explanation tables.

    Returns the surrogate factory name (``"rules"``/``"weights"``) and injects
    ``explanation_df`` / ``metadata_df`` into ``kwargs``. ``model_name``, ``depth``
    and ``variant`` from ``kwargs`` are used to filter the loader's table.
    """
    if getattr(getattr(loader, "data_source", None), "source_type", None) != "coxam":
        raise AttributeError("loader= for CoXAM methods requires a CoXAM data loader")

    if key in {"decision_tree", "dt", "rules"}:
        table_name, factory_name = "decision_tree", "rules"
    else:  # logistic_regression / lr / weights
        table_name, factory_name = "logistic_regression", "weights"

    model_name = kwargs.get("model_name")
    explanation_df = loader.get_explanation_table(table_name)
    if "model" in explanation_df.columns and model_name is not None:
        explanation_df = explanation_df[explanation_df["model"] == model_name]
    if table_name == "decision_tree" and "depth" in kwargs and "depth" in explanation_df.columns:
        explanation_df = explanation_df[explanation_df["depth"] == kwargs["depth"]]
    if table_name == "logistic_regression" and "variant" in kwargs and "variant" in explanation_df.columns:
        explanation_df = explanation_df[explanation_df["variant"] == kwargs["variant"]]
    if explanation_df.empty:
        filters = {k: kwargs[k] for k in ("depth", "variant") if k in kwargs}
        raise ValueError(f"No CoXAM rows for method={key}, model_name={model_name}, filters={filters}")

    kwargs["explanation_df"] = explanation_df
    kwargs["metadata_df"] = loader.get_metadata()
    return factory_name, kwargs


def create_xai_method(name: str, *, ai_model: Any = None, train_data: Any = None, loader: Any = None, **kwargs):
    """Create an XAI method adapter from the global registry.

    Args:
        name: Registered adapter name or alias.
        ai_model: Optional trained AI model. When provided, the
            model's ``predict``/``model`` and ``train_data`` are wired into the
            adapter's expected kwargs (LOFO, SHAP kernel, LIME, gradient×input,
            DeepLift, Integrated Gradients, LRP). Ignored for adapters with no
            such wiring.
        train_data: Optional training data used with ``ai_model`` (exposing
            ``X``/``y``/``feature_names``/``categorical_feature_indices``).
        loader: Optional CoXAM data loader. When provided for a rules/weights
            method, the surrogate's explanation and metadata tables are read from
            it instead of being passed explicitly.
        **kwargs: Adapter-specific keyword arguments (e.g. ``predict_fn``,
            ``background_data``, ``explanation_df``, ``target``, ``depth``,
            ``variant``). Pass ``target=None`` or ``target`` in
            {"all", "all_classes", "every"} to build one adapter per output class.
    """
    key = name.lower()

    # CoXAM surrogate identified by its method string; tables read from a loader.
    if loader is not None and key in _COXAM_METHOD_KEYS:
        name, kwargs = _apply_coxam_loader(key, loader, kwargs)
    # Trained AI model: wire the model and training data into the kwargs.
    elif ai_model is not None:
        kwargs = _apply_ai_model_kwargs(key, ai_model, train_data, kwargs)

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

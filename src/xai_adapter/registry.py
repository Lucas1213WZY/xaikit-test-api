"""Registry for XAI adapters."""

from __future__ import annotations

from typing import Any, Dict, Type


class XAIAdapterRegistry:
    """Small registry for adapter classes."""

    def __init__(self):
        self._registry: Dict[str, Any] = {}

    def register(self, name: str, adapter_class: Type, *aliases: str) -> None:
        """Register an adapter class.

        Args:
            name: Name the adapter is registered under; matched case-insensitively.
            adapter_class: The adapter class, or a factory returning one.
            *aliases: Additional names resolving to the same adapter.
        """
        self._registry[name.lower()] = adapter_class
        for alias in aliases:
            self._registry[alias.lower()] = adapter_class

    def register_custom(self, name: str, algorithm: Any, *aliases: str) -> None:
        """Register a user-provided function or object as an XAI method.

        Args:
            name: Name the method is registered under.
            algorithm: The callable or object to wrap in an attribution adapter.
            *aliases: Additional names resolving to the same method.
        """
        from .attribution import CustomAttribution

        def factory(**kwargs):
            return CustomAttribution(algorithm, method_name=name, **kwargs)

        self.register(name, factory, *aliases)

    def get_class(self, name: str) -> Any:
        """Return the adapter class or factory for a registered name.

        Args:
            name: Registered name or alias.

        Raises:
            ValueError: If the name is not registered.
        """
        key = name.lower()
        if key not in self._registry:
            raise ValueError(f"Unknown XAI adapter '{name}'. Available: {self.list_available()}")
        return self._registry[key]

    def create(self, name: str, **kwargs):
        """Instantiate a registered adapter.

        Args:
            name: Registered name or alias.
            **kwargs: Passed to the adapter's constructor.
        """
        return self.get_class(name)(**kwargs)

    def list_available(self) -> list[str]:
        """List registered adapter names."""
        return sorted(self._registry.keys())

    def is_registered(self, name: str) -> bool:
        """Check whether a method name or alias is registered.

        Args:
            name: Name or alias to look up.
        """
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
_PRECOMPUTED_METHOD_KEYS = {
    "precomputed_csv", "csv", "csv_dataset", "dataset_csv",
}
_APP_ID_COLUMNS = ("dataId", "appId", "app_id")
_SIM2REAL_PROPERTIES = {"faithful", "robust", "sparse", "sparse_robust"}


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


def _numbered_columns(table: Any, prefixes: tuple[str, ...], suffix: str = "") -> list[str]:
    """Return numbered columns in numeric order, e.g. v0..v10 or a0_i..a10_i."""
    matches = []
    for column in table.columns:
        if not isinstance(column, str):
            continue
        for prefix_index, prefix in enumerate(prefixes):
            if not column.startswith(prefix) or (suffix and not column.endswith(suffix)):
                continue
            stem = column[len(prefix):-len(suffix) if suffix else None]
            if stem.isdigit():
                matches.append((prefix_index, int(stem), column))
                break
    return [column for _, _, column in sorted(matches)]


def _filter_precomputed_table(
    table: Any,
    *,
    app_id: Any = None,
    model_name: Any = None,
    exp_method: Any = None,
    exp_property: Any = None,
    baseline_property: bool = False,
) -> Any:
    """Filter a loaded precomputed table without mutating the loader."""
    result = table.copy()
    if app_id is not None:
        app_column = next((column for column in _APP_ID_COLUMNS if column in result.columns), None)
        if app_column is not None:
            result = result[result[app_column] == app_id]
    if model_name is not None:
        model_column = next((column for column in ("modelName", "model") if column in result.columns), None)
        if model_column is not None:
            result = result[result[model_column] == model_name]
    if exp_method is not None:
        if "expMethod" not in result.columns:
            raise ValueError("The selected explanation table has no 'expMethod' column")
        result = result[result["expMethod"] == exp_method]
    if exp_property is not None:
        if "expProperty" not in result.columns:
            raise ValueError("The selected explanation table has no 'expProperty' column")
        result = result[result["expProperty"] == exp_property]
    elif baseline_property and "expProperty" in result.columns:
        properties = result["expProperty"].fillna("").astype(str).str.strip()
        result = result[properties == ""]
    return result.copy()


def _merge_precomputed_features(explanation_df: Any, feature_df: Any) -> Any:
    """Join separate explanation and feature assets into one parser-ready table."""
    if _numbered_columns(explanation_df, ("v", "x")):
        return explanation_df.copy()

    feature_columns = _numbered_columns(feature_df, ("v", "x"))
    if not feature_columns:
        raise ValueError("The loader's feature table has no numbered vN/xN columns")
    if "instanceId" not in explanation_df.columns or "instanceId" not in feature_df.columns:
        raise ValueError("Feature and explanation tables must both include 'instanceId'")

    explanation = explanation_df.copy()
    features = feature_df.copy()
    explanation_app = next((col for col in _APP_ID_COLUMNS if col in explanation.columns), None)
    feature_app = next((col for col in _APP_ID_COLUMNS if col in features.columns), None)
    merge_keys = ["instanceId"]
    restore_app_column = explanation_app or feature_app
    if explanation_app is not None and feature_app is not None:
        internal_app_column = "__xai_adapter_app_id"
        explanation = explanation.rename(columns={explanation_app: internal_app_column})
        features = features.rename(columns={feature_app: internal_app_column})
        merge_keys.insert(0, internal_app_column)

    feature_subset = features[[*merge_keys, *feature_columns]]
    if feature_subset.duplicated(merge_keys).any():
        raise ValueError("The loader's feature table has duplicate instance keys")
    merged = explanation.merge(
        feature_subset,
        on=merge_keys,
        how="inner",
        validate="many_to_one",
    )
    if merged.empty and not explanation.empty:
        raise ValueError("No explanation rows matched the loader's feature rows")
    if "__xai_adapter_app_id" in merged.columns and restore_app_column is not None:
        merged = merged.rename(columns={"__xai_adapter_app_id": restore_app_column})
    return merged


def _apply_precomputed_loader(loader: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Build parser-ready precomputed explanations from a unified data loader."""
    if "dataset" in kwargs or "csv_path" in kwargs or "dataframe" in kwargs:
        raise ValueError(
            "Pass either loader= or dataset=/csv_path=/dataframe= for precomputed CSV explanations, not both"
        )

    table_name = kwargs.pop("explanation_type", "attribution")
    exp_method = kwargs.pop("exp_method", None)
    exp_property = kwargs.pop("exp_property", None)
    app_id = kwargs.pop("app_id", None)
    model_name = kwargs.pop("model_name", None)
    source_type = getattr(getattr(loader, "data_source", None), "source_type", None)
    is_sim2real = source_type == "sim2real"

    if exp_property is not None and not is_sim2real:
        raise ValueError("exp_property is supported only for Sim2Real data loaders")

    uses_explanation_variants = is_sim2real and table_name not in {"none", "deltas"}
    if uses_explanation_variants:
        exp_method = "lime" if exp_method is None else str(exp_method).strip().lower()
        if exp_method != "lime":
            raise ValueError("Sim2Real precomputed explanations use exp_method='lime'")
        if exp_property is not None:
            exp_property = str(exp_property).strip().lower()
            if exp_property not in _SIM2REAL_PROPERTIES:
                raise ValueError(
                    "Sim2Real exp_property must be one of: "
                    + ", ".join(sorted(_SIM2REAL_PROPERTIES))
                )
    elif exp_property is not None:
        raise ValueError(
            f"exp_property is not available for Sim2Real explanation_type={table_name!r}"
        )

    try:
        explanation_df = loader.get_explanation_table(table_name)
        feature_df = loader.get_feature_values()
    except AttributeError as exc:
        raise TypeError(
            "loader= for precomputed CSV explanations requires a UnifiedDataLoader-like object"
        ) from exc

    explanation_df = _filter_precomputed_table(
        explanation_df,
        app_id=app_id,
        model_name=model_name,
        exp_method=exp_method,
        exp_property=exp_property,
        baseline_property=uses_explanation_variants and exp_property is None,
    )
    if explanation_df.empty:
        raise ValueError(
            f"No rows found in explanation table {table_name!r} for "
            f"app_id={app_id!r}, model_name={model_name!r}, "
            f"exp_method={exp_method!r}, exp_property={exp_property!r}"
        )

    if "expMethod" in explanation_df.columns:
        available_methods = explanation_df["expMethod"].dropna().unique().tolist()
        if exp_method is None and len(available_methods) > 1:
            raise ValueError(
                f"Explanation table {table_name!r} contains multiple expMethod values "
                f"{sorted(available_methods)}; pass exp_method= to select one"
            )

    explanation_columns = _numbered_columns(explanation_df, ("a",), suffix="_i")
    if not explanation_columns and table_name != "none":
        raise ValueError(
            f"Explanation table {table_name!r} has no numbered aN_i explanation columns"
        )
    if "pred" not in explanation_df.columns:
        raise ValueError(f"Explanation table {table_name!r} has no 'pred' column")

    merged = _merge_precomputed_features(explanation_df, feature_df)
    duplicate_keys = ["instanceId"]
    app_column = next((col for col in _APP_ID_COLUMNS if col in merged.columns), None)
    if app_column is not None:
        duplicate_keys.insert(0, app_column)
    if merged.duplicated(duplicate_keys).any():
        raise ValueError(
            f"Explanation table {table_name!r} has multiple rows per instance after filtering"
        )

    kwargs["dataframe"] = merged
    kwargs.setdefault("feature_columns", _numbered_columns(merged, ("v", "x")))
    kwargs.setdefault("explanation_columns", explanation_columns)
    if table_name == "none":
        kwargs.setdefault("missing_explanation_strategy", "zeros")
    return kwargs


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
        loader: Optional unified data loader. For rules/weights, the CoXAM
            surrogate tables are read from it. For ``precomputed_csv``/``csv``,
            pass ``explanation_type`` and optionally ``exp_method``, ``app_id``,
            and ``model_name`` to join a loaded explanation table with its
            feature rows. Sim2Real loaders default ``exp_method`` to ``"lime"``
            and additionally accept ``exp_property`` as one of ``faithful``,
            ``robust``, ``sparse``, or ``sparse_robust``.
        **kwargs: Adapter-specific keyword arguments (e.g. ``predict_fn``,
            ``background_data``, ``explanation_df``, ``target``, ``depth``,
            ``variant``). Pass ``target=None`` or ``target`` in
            {"all", "all_classes", "every"} to build one adapter per output class.
    """
    key = name.lower()

    # Dataset-backed explanations: join the loader's explanation and feature tables.
    if loader is not None and key in _PRECOMPUTED_METHOD_KEYS:
        kwargs = _apply_precomputed_loader(loader, kwargs)
    # CoXAM surrogate identified by its method string; tables read from a loader.
    elif loader is not None and key in _COXAM_METHOD_KEYS:
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

    Args:
        name: Name the method is registered under.
        adapter: An adapter class, or an algorithm to wrap automatically.
        *aliases: Additional names resolving to the same method.
    """
    if isinstance(adapter, type):
        get_adapter_registry().register(name, adapter, *aliases)
        return
    get_adapter_registry().register_custom(name, adapter, *aliases)

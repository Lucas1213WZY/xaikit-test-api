"""Convenience constructors for XAI adapters."""

from __future__ import annotations

from typing import Any, List, Optional, Sequence

from .base import ArrayLike
from .attribution import make_attribution
from src.data_loaders.xai_dataset import XAIDatasetParser
from .registry import create_xai_method
from .surrogate import GeneratedSurrogateMethods, make_surrogate


def _train_x(train_data: Any) -> ArrayLike:
    if hasattr(train_data, "X"):
        return train_data.X
    return train_data


def _train_y(train_data: Any) -> Optional[ArrayLike]:
    return getattr(train_data, "y", None)


def _categorical_features(train_data: Any) -> Optional[list[int]]:
    return getattr(train_data, "categorical_feature_indices", None)


def _feature_names(train_data: Any) -> Optional[list[str]]:
    return getattr(train_data, "feature_names", None)


def create_xai_method_from_engine(
    name: str,
    *,
    engine: Any,
    train_data: Any,
    preprocessing_fn,
    target: int = 1,
    **kwargs,
):
    """
    Create an XAI method adapter from a CoAX-style `engine` and `train_data`.

    `engine` is expected to expose `predict(...)`; gradient methods also use
    `engine.model` and prefer `engine.forward_logits_or_probs(...)` when it is
    available.
    """
    key = name.lower()
    train_x = _train_x(train_data)
    predict_fn = kwargs.pop("predict_fn", engine.predict)

    if key in {"lofo", "leave_one_feature_out"}:
        return create_xai_method(
            name,
            predict_fn=predict_fn,
            background_data=train_x,
            preprocessing_fn=preprocessing_fn,
            target=target,
            **kwargs,
        )

    if key in {"shap", "shap_kernel"}:
        return create_xai_method(
            name,
            predict_fn=predict_fn,
            background_data=train_x,
            preprocessing_fn=preprocessing_fn,
            target=target,
            **kwargs,
        )

    if key in {"lime", "lime_tabular"}:
        return create_xai_method(
            name,
            predict_fn=predict_fn,
            training_data=train_x,
            training_labels=kwargs.pop("training_labels", _train_y(train_data)),
            categorical_features=kwargs.pop("categorical_features", _categorical_features(train_data)),
            feature_names=kwargs.pop("feature_names", _feature_names(train_data)),
            preprocessing_fn=preprocessing_fn,
            target=target,
            **kwargs,
        )

    if key in {"gradient_input", "gradient_x_input", "input_gradients"}:
        return create_xai_method(
            name,
            model=engine.model,
            forward_fn=kwargs.pop("forward_fn", getattr(engine, "forward_logits_or_probs", None)),
            predict_fn=predict_fn,
            background_data=train_x,
            preprocessing_fn=preprocessing_fn,
            target=target,
            **kwargs,
        )

    if key in {"deeplift", "deep_lift", "integrated_gradients", "ig"}:
        return create_xai_method(
            name,
            model=engine.model,
            predict_fn=predict_fn,
            background_data=train_x,
            preprocessing_fn=preprocessing_fn,
            target=target,
            **kwargs,
        )

    return create_xai_method(name, **kwargs)


def create_custom_xai_method(
    algorithm: Any,
    *,
    method_name: str = "custom",
    **kwargs,
):
    """
    Wrap a user-provided function or object in the common XAI adapter API.

    The algorithm can be a callable, or an object exposing `fit` and `explain`.
    Legacy objects exposing `attribute` are still accepted for compatibility,
    but `explain` is the canonical public method.
    """
    return make_attribution(algorithm, method_name=method_name, **kwargs)


def create_custom_surrogate_method(
    fit_fn: Any,
    explain_fn: Any,
    *,
    method_name: str = "custom",
):
    """
    Wrap user-provided fit/explain callables in the surrogate adapter API.

    The returned method implements the `SurrogateMethod` interface and supports
    sklearn-style `fit(...).explain(...)` chaining.
    """
    return make_surrogate(fit_fn, explain_fn, name=method_name)


def create_coxam_xai_method(
    loader,
    method_type: str,
    app_id: str,
    model_name: str,
    **kwargs,
):
    """
    Create a CoXAM rules-vs-weights method from data-loader explanation tables.

    Args:
        loader: UnifiedDataLoader with CoXAM explanation tables.
        method_type: 'decision_tree'/'rules' or 'logistic_regression'/'weights'.
        app_id: Dataset dataId.
        model_name: Model name, e.g. 'mlp' or 'xgboost'.
        **kwargs: Extra filters, e.g. depth=3 or variant='sparse'.
    """
    if getattr(loader.data_source, "source_type", None) != "coxam":
        raise AttributeError("create_coxam_xai_method requires a CoXAM data loader")

    key = method_type.lower().strip()
    if key in {"decision_tree", "dt", "rules"}:
        table_name = "decision_tree"
        factory_name = "rules"
    elif key in {"logistic_regression", "lr", "weights"}:
        table_name = "logistic_regression"
        factory_name = "weights"
    else:
        raise ValueError("method_type must be 'decision_tree'/'rules' or 'logistic_regression'/'weights'")

    explanation_df = loader.get_explanation_table(table_name)
    if "model" in explanation_df.columns:
        explanation_df = explanation_df[explanation_df["model"] == model_name]
    if table_name == "decision_tree" and "depth" in kwargs and "depth" in explanation_df.columns:
        explanation_df = explanation_df[explanation_df["depth"] == kwargs["depth"]]
    if table_name == "logistic_regression" and "variant" in kwargs and "variant" in explanation_df.columns:
        explanation_df = explanation_df[explanation_df["variant"] == kwargs["variant"]]
    if explanation_df.empty:
        raise ValueError(
            f"No rows found for method={key}, app_id={app_id}, model_name={model_name}, filters={kwargs}"
        )

    return create_xai_method(
        factory_name,
        explanation_df=explanation_df,
        metadata_df=loader.get_metadata(),
        app_id=app_id,
        model_name=model_name,
        **kwargs,
    )


def get_coxam_xai_predictions(
    loader,
    instance_ids: List[int],
    method_type: str,
    app_id: str,
    model_name: str,
    **kwargs,
) -> List[Any]:
    """Apply a CoXAM XAI method to raw features from a data loader."""
    method = create_coxam_xai_method(
        loader,
        method_type=method_type,
        app_id=app_id,
        model_name=model_name,
        **kwargs,
    )
    raw_features = loader.get_features(instance_ids, normalize=False)
    return method.apply_batch(raw_features)


def generate_surrogate_xai_methods(
    *,
    dataset: Optional[XAIDatasetParser] = None,
    csv_path: Optional[str] = None,
    dataframe: Any = None,
    instance_ids: Optional[Sequence[Any]] = None,
    app_id: str = "custom_dataset",
    model_name: str = "external_model",
    methods: Sequence[str] = ("decision_tree", "logistic_regression"),
    depths: Sequence[int] = (3,),
    variants: Sequence[str] = ("dense", "sparse"),
    variant: str = "sparse",
    top_k: int = 3,
    random_state: int = 0,
    **dataset_kwargs,
) -> GeneratedSurrogateMethods:
    """
    Generate rules-vs-weights XAI methods from a dataset CSV.

    Use this path when the user provides new feature rows and AI predictions
    instead of precomputed `assets/explanations/coxam` surrogate tables.
    """
    requested = {method.lower() for method in methods}
    if requested & {"logistic_regression", "lr", "weights"}:
        if variant not in variants:
            raise ValueError(f"variant={variant!r} must be included in variants={list(variants)!r}")

    if dataset is None:
        dataset_kwargs.setdefault("missing_explanation_strategy", "zeros")
        dataset = XAIDatasetParser(csv_path=csv_path, dataframe=dataframe, **dataset_kwargs)

    if instance_ids is None:
        instance_ids = dataset.df[dataset.instance_id_col].tolist()

    X = dataset.get_features(instance_ids)
    y = dataset.get_predictions(instance_ids)
    fitted_methods = {}
    decision_tree_df = None
    logistic_regression_df = None
    metadata_df = None

    if requested & {"decision_tree", "dt", "rules"}:
        depth = depths[0]
        fitted_methods["rules"] = create_xai_method(
            "rules",
            app_id=app_id,
            model_name=model_name,
            depth=depth,
            random_state=random_state,
            feature_names=dataset.feature_columns,
        ).fit(X, y)
        decision_tree_df = fitted_methods["rules"].to_explanation_table()
        metadata_df = fitted_methods["rules"].to_metadata_table()

    if requested & {"logistic_regression", "lr", "weights"}:
        fitted_methods["weights"] = create_xai_method(
            "weights",
            app_id=app_id,
            model_name=model_name,
            variant=variant,
            top_k=top_k,
            random_state=random_state,
            feature_names=dataset.feature_columns,
        ).fit(X, y)
        logistic_regression_df = fitted_methods["weights"].to_explanation_table()
        if metadata_df is None:
            metadata_df = fitted_methods["weights"].to_metadata_table()

    if not fitted_methods:
        raise ValueError("methods must include at least one of: decision_tree/rules or logistic_regression/weights")

    return GeneratedSurrogateMethods(
        decision_tree_df=decision_tree_df,
        logistic_regression_df=logistic_regression_df,
        metadata_df=metadata_df,
        methods=fitted_methods,
    )

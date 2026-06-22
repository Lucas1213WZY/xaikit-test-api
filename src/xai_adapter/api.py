"""Convenience constructors for XAI adapters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Sequence

import numpy as np
import pandas as pd

from .base import ArrayLike
from .attribution import make_attribution
from .registry import create_xai_method
from .surrogate import GeneratedSurrogateMethods, make_surrogate
from src.data_loaders import PreparedDataset, make_train_data_for_xai
from src.data_loaders.xai_dataset import XAIDatasetParser


@dataclass
class ExplanationRunConfig:
    """Inputs shared by explanation-generation workflow steps."""

    data: PreparedDataset
    iv_config: dict[str, dict[str, Any]]
    trained_engine: Any
    model_name: str = "mlp"
    output_dir: Path = Path("generated_explanation")
    target: int = 1
    method_kwargs: Optional[dict[str, dict[str, Any]]] = None

    @property
    def dataset_id(self) -> str:
        return self.data.dataset_id


def init_explanation_run(
    data: PreparedDataset,
    iv_config: dict[str, dict[str, Any]],
    trained_engine: Any,
    *,
    model_name: str = "mlp",
    output_dir: str | Path = "generated_explanation",
    target: int = 1,
    method_kwargs: Optional[dict[str, dict[str, Any]]] = None,
) -> ExplanationRunConfig:
    """Collect all shared XAI-generation settings in one object."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    return ExplanationRunConfig(
        data=data,
        iv_config=iv_config,
        trained_engine=trained_engine,
        model_name=model_name,
        output_dir=output_path,
        target=target,
        method_kwargs=method_kwargs or {
            "shap": {"n_background_samples": 30},
            "shap_kernel": {"n_background_samples": 30},
            "lime": {"num_samples": 1000},
        },
    )


def get_xai_methods_from_design(iv_config: dict[str, dict[str, Any]]) -> list[Any]:
    """Read XAI method levels from `xai_method` or legacy `xai_type` IV names."""
    xai_iv_name = "xai_type" if "xai_type" in iv_config else "xai_method"
    if xai_iv_name not in iv_config:
        raise ValueError("iv_config must include either 'xai_method' or 'xai_type'.")
    return list(iv_config[xai_iv_name]["levels"])


def predict_labels(trained_engine: Any, X: np.ndarray) -> np.ndarray:
    """Convert model predictions/probabilities into integer labels."""
    raw_predictions = trained_engine.predict(X)
    if np.ndim(raw_predictions) > 1:
        return np.argmax(raw_predictions, axis=1)
    return raw_predictions.astype(int)


def generate_xai_explanation_tables(
    config: ExplanationRunConfig,
) -> tuple[list[Path], list[pd.DataFrame]]:
    """Generate one explanation CSV per non-control XAI method."""
    train_data_for_xai = make_train_data_for_xai(config.data.split, config.data.y_train)
    instance_ids = np.asarray(config.data.test_instance_ids)
    predictions = predict_labels(config.trained_engine, config.data.X_test)

    saved_paths: list[Path] = []
    explanation_dfs: list[pd.DataFrame] = []

    for method_name in get_xai_methods_from_design(config.iv_config):
        method_key = str(method_name).lower()

        if method_key in {"none", "no_xai", "control"}:
            print(f"Skipping adapter generation for xai method: {method_name}")
            continue

        print(f"\nGenerating explanations for xai method: {method_name}")
        try:
            explainer = create_xai_method_from_engine(
                method_key,
                engine=config.trained_engine,
                train_data=train_data_for_xai,
                preprocessing_fn=lambda x: np.asarray(x, dtype=np.float32),
                target=config.target,
                **config.method_kwargs.get(method_key, {}),
            )
            result = explainer.explain(config.data.X_test)
            explanation_df = result.to_explanation_df(
                instance_ids=instance_ids,
                predictions=predictions,
                dataset_id=config.dataset_id,
                model_name=config.model_name,
            )
            explanation_df["expMethod"] = method_key

            out_path = config.output_dir / f"{method_key}_{config.model_name}_{config.dataset_id}.csv"
            explanation_df.to_csv(out_path, index=False)
            saved_paths.append(out_path)
            explanation_dfs.append(explanation_df)
            print(f"  Saved: {out_path} shape={explanation_df.shape}")
        except Exception as exc:
            print(f"  Skipped {method_name}: {type(exc).__name__}: {exc}")

    return saved_paths, explanation_dfs


def combine_explanation_tables(
    explanation_dfs: list[pd.DataFrame],
    config: ExplanationRunConfig,
) -> tuple[Optional[Path], Optional[pd.DataFrame]]:
    """Combine generated method-level explanations into one design-engine table."""
    if not explanation_dfs:
        print("\nNo explanation CSVs were generated. Check installed XAI dependencies.")
        return None, None

    combined_df = pd.concat(explanation_dfs, ignore_index=True)
    combined_path = config.output_dir / f"de_{config.model_name}_{config.dataset_id}.csv"
    combined_df.to_csv(combined_path, index=False)
    print(f"\nCombined explanation CSV: {combined_path} shape={combined_df.shape}")
    return combined_path, combined_df


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

    if key in {
        "deeplift",
        "deep_lift",
        "integrated_gradients",
        "ig",
        "lrp",
        "layer_relevance_propagation",
    }:
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

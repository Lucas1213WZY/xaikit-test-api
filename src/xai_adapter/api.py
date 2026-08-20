"""Convenience constructors for XAI adapters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Sequence

import numpy as np
import pandas as pd

from .base import ArrayLike, XAIAdapterResult
from .attribution import make_attribution
from .registry import create_xai_method
from .surrogate import GeneratedSurrogateMethods, make_surrogate
from src.data_loaders import PreparedDataset, make_train_data_for_xai
from src.data_loaders.xai_dataset import XAIDatasetParser
from src.workflow_standard import (
    DATA_ID_COL,
    DEFAULT_EXPLANATION_INSTANCE_LIMIT,
    EXPLANATION_METHOD_COL,
    INSTANCE_ID_COL,
    MODEL_NAME_COL,
    PREDICTION_COL,
    PREDICTION_ONLY_METHOD,
    prediction_labels,
)


@dataclass
class ExplanationRunConfig:
    """Inputs shared by explanation-generation workflow steps."""

    data: PreparedDataset
    iv_config: dict[str, dict[str, Any]]
    trained_ai_model: Any
    model_name: str = "mlp"
    output_dir: Path = Path("generated_explanation")
    target: int = 1
    method_kwargs: Optional[dict[str, dict[str, Any]]] = None
    max_test_instances: int = DEFAULT_EXPLANATION_INSTANCE_LIMIT
    instance_ids: Optional[Sequence[Any]] = None
    instance_ids_by_method: Optional[dict[str, Sequence[Any]]] = None
    predictions_by_instance: Optional[dict[int, Any]] = None

    @property
    def dataset_id(self) -> str:
        return self.data.dataset_id


def init_explanation_run(
    data: PreparedDataset,
    iv_config: dict[str, dict[str, Any]],
    trained_ai_model: Any,
    *,
    model_name: str = "mlp",
    output_dir: str | Path = "generated_explanation",
    target: int = 1,
    method_kwargs: Optional[dict[str, dict[str, Any]]] = None,
    max_test_instances: int = DEFAULT_EXPLANATION_INSTANCE_LIMIT,
    instance_ids: Optional[Sequence[Any]] = None,
    instance_ids_by_method: Optional[dict[str, Sequence[Any]]] = None,
    predictions_by_instance: Optional[dict[int, Any]] = None,
) -> ExplanationRunConfig:
    """Collect all shared XAI-generation settings in one object.

    Args:
        data: The prepared dataset explanations are generated over.
        iv_config: Design IVs supplying the XAI method levels.
        trained_ai_model: The model being explained.
        model_name: Name used in the output filenames.
        output_dir: Directory the explanation tables are written to.
        target: Class index explanations are generated for.
        method_kwargs: Per-method options, keyed by method name.
        max_test_instances: Cap on test instances explained.
        instance_ids: Explain only these instances.
        instance_ids_by_method: Per-method instance restriction, overriding
            ``instance_ids`` for the methods it names.
        predictions_by_instance: Precomputed AI predictions, to avoid
            recomputing them.

    Returns:
        The assembled config, ready for the generate functions.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    return ExplanationRunConfig(
        data=data,
        iv_config=iv_config,
        trained_ai_model=trained_ai_model,
        model_name=model_name,
        output_dir=output_path,
        target=target,
        method_kwargs=method_kwargs or {
            "shap": {"n_background_samples": 30},
            "shap_kernel": {"n_background_samples": 30},
            "lime": {"num_samples": 1000},
        },
        max_test_instances=max_test_instances,
        instance_ids=list(instance_ids) if instance_ids is not None else None,
        instance_ids_by_method={
            str(method).lower(): list(ids)
            for method, ids in (instance_ids_by_method or {}).items()
        } or None,
        predictions_by_instance=predictions_by_instance,
    )


def get_xai_methods_from_design(iv_config: dict[str, dict[str, Any]]) -> list[Any]:
    """Read XAI method levels from `xai_method` or legacy `xai_type` IV names.

    `xai_method` wins when a design names both, matching how instance selection and
    per-trial explanation lookup resolve the method. Preferring `xai_type` here
    made those disagree, and a design declaring both IVs generated an empty pool.

    Args:
        iv_config: The design's independent variables.

    Returns:
        The XAI method levels, or an empty list if neither IV is present.
    """
    xai_iv_name = "xai_method" if "xai_method" in iv_config else "xai_type"
    if xai_iv_name not in iv_config:
        raise ValueError("iv_config must include either 'xai_method' or 'xai_type'.")
    return list(iv_config[xai_iv_name]["levels"])


def predict_labels(trained_ai_model: Any, X: np.ndarray) -> np.ndarray:
    """Convert model predictions/probabilities into integer labels.

    Args:
        trained_ai_model: The model to predict with.
        X: Model-ready feature rows.

    Returns:
        One integer class label per row.
    """
    return prediction_labels(trained_ai_model.predict(X))


def prediction_only_table(
    data: PreparedDataset,
    trained_ai_model: Any,
    *,
    model_name: str = "mlp",
) -> pd.DataFrame:
    """Predict every instance in ``data`` and format it as prediction-only rows.

    A trial instance needs an AI prediction whether or not it also gets a real
    explanation -- e.g. CoAX's ``none`` condition is still scored against it --
    so this covers the whole dataset, not just the instances a real method
    explains. The ``expMethod`` column is always :data:`PREDICTION_ONLY_METHOD`,
    matching what ``generate_ai_prediction_table``/``explanations()`` produce.

    Args:
        data: The prepared dataset to predict over.
        trained_ai_model: The model to predict with.
        model_name: Value written to the ``modelName`` column.

    Returns:
        A DataFrame with ``dataId``/``modelName``/``expMethod``/``instanceId``/``pred`` columns.
    """
    predictions = predict_labels(trained_ai_model, data.split.X_model)
    return pd.DataFrame({
        DATA_ID_COL: data.dataset_id,
        MODEL_NAME_COL: model_name,
        EXPLANATION_METHOD_COL: PREDICTION_ONLY_METHOD,
        INSTANCE_ID_COL: [int(i) for i in data.split.raw_instance_ids],
        PREDICTION_COL: predictions,
    })


def generate_ai_prediction_table(
    config: ExplanationRunConfig,
) -> tuple[Path, pd.DataFrame]:
    """Predict every train/test instance and save a complete prediction table.

    Args:
        config: Settings from ``init_explanation_run``.

    Returns:
        The written path and the prediction table.
    """
    instance_ids = np.asarray(
        list(dict.fromkeys([
            *map(int, config.data.train_instance_ids),
            *map(int, config.data.test_instance_ids),
        ])),
        dtype=int,
    )
    if np.any(instance_ids < 0) or np.any(
        instance_ids >= len(config.data.split.X_model)
    ):
        raise ValueError("Dataset train/test instance IDs contain an out-of-range row.")

    if config.predictions_by_instance is None:
        predictions = predict_labels(
            config.trained_ai_model,
            config.data.split.X_model[instance_ids],
        )
    else:
        missing = [
            int(instance_id)
            for instance_id in instance_ids
            if int(instance_id) not in config.predictions_by_instance
        ]
        if missing:
            raise ValueError(
                "The stored AI prediction mapping is missing dataset rows: "
                f"{missing[:10]}."
            )
        predictions = np.asarray([
            config.predictions_by_instance[int(instance_id)]
            for instance_id in instance_ids
        ])
    prediction_df = pd.DataFrame({
        DATA_ID_COL: config.dataset_id,
        MODEL_NAME_COL: config.model_name,
        EXPLANATION_METHOD_COL: PREDICTION_ONLY_METHOD,
        INSTANCE_ID_COL: instance_ids,
        PREDICTION_COL: predictions.astype(int),
    })
    output_path = (
        config.output_dir
        / f"predictions_{config.model_name}_{config.dataset_id}.csv"
    )
    prediction_df.to_csv(output_path, index=False)
    print(
        "\nSaved complete AI prediction table: "
        f"{output_path} shape={prediction_df.shape}"
    )
    return output_path, prediction_df


#: Methods that model the AI with a *global* surrogate rather than attributing
#: one instance at a time. They take a different constructor shape from the
#: attribution methods -- ``app_id``/``feature_names``/``depth`` instead of
#: ``ai_model``/``preprocessing_fn`` -- and forward unknown kwargs straight to
#: their sklearn estimator, so passing the attribution kwargs raises.
SURROGATE_METHODS: frozenset[str] = frozenset(
    {"decision_tree", "logistic_regression", "rule_list", "rule_set", "decision_list"}
)


def _make_explainer(method_key: str, config: ExplanationRunConfig, train_data_for_xai: Any) -> Any:
    """Construct one explainer with the kwargs its family accepts."""
    method_kwargs = dict(config.method_kwargs.get(method_key, {}))
    if method_key in SURROGATE_METHODS:
        return create_xai_method(
            method_key,
            app_id=config.dataset_id,
            model_name=config.model_name,
            feature_names=list(config.data.split.model_feature_names),
            **method_kwargs,
        )
    return create_xai_method(
        method_key,
        ai_model=config.trained_ai_model,
        train_data=train_data_for_xai,
        preprocessing_fn=lambda x: np.asarray(x, dtype=np.float32),
        target=config.target,
        **method_kwargs,
    )


def _fit_surrogate_if_needed(explainer: Any, config: ExplanationRunConfig) -> None:
    """Fit a surrogate explainer before it is asked to explain anything.

    A surrogate -- decision tree, logistic regression, rule list/set -- is a
    single *global* model of the AI, so it is fitted once over the whole
    dataset against the AI's own predicted labels, and every instance is then
    explained by that same surrogate. The condition (``xai_type``) decides
    which surrogate is *shown*, not what it is fitted on, which is why
    ``hybrid`` needs both families but no extra fitting.

    Attribution methods (LIME, SHAP, gradients) come back already fitted from
    ``create_xai_method``, so the ``is_fitted`` check leaves them untouched
    rather than re-fitting them against labels they do not use.
    """
    if getattr(explainer, "is_fitted", True):
        return

    X = np.asarray(config.data.split.X_model)
    if config.predictions_by_instance is not None:
        y = np.asarray(
            [
                config.predictions_by_instance[int(instance_id)]
                for instance_id in config.data.split.raw_instance_ids
            ]
        )
    else:
        y = predict_labels(config.trained_ai_model, X)
    explainer.fit(X, y)


def generate_xai_explanation_tables(
    config: ExplanationRunConfig,
) -> tuple[list[Path], list[pd.DataFrame]]:
    """Generate one explanation CSV per non-control XAI method.

    Args:
        config: Settings from ``init_explanation_run``.

    Returns:
        The written paths and the tables themselves, in matching order.
    """
    train_data_for_xai = make_train_data_for_xai(config.data.split, config.data.y_train)
    saved_paths: list[Path] = []
    explanation_dfs: list[pd.DataFrame] = []
    attempted: list[str] = []
    failures: dict[str, str] = {}

    for method_name in get_xai_methods_from_design(config.iv_config):
        method_key = str(method_name).lower()

        if method_key in {"none", "no_xai", "control"}:
            print(f"Skipping explanation generation for xai method: {method_name}")
            continue

        instance_ids = _explanation_ids_for_method(config, method_key)
        if len(instance_ids) == 0:
            print(
                f"Skipping explanation generation for {method_name}: "
                "no sampled XAI-visible trial instances."
            )
            continue
        explained_instances = config.data.split.X_model[instance_ids]
        if config.predictions_by_instance is None:
            predictions = predict_labels(
                config.trained_ai_model,
                explained_instances,
            )
        else:
            predictions = np.asarray([
                config.predictions_by_instance[int(instance_id)]
                for instance_id in instance_ids
            ])

        print(f"\nGenerating explanations for xai method: {method_name}")
        print(f"  Sampled instances: {len(instance_ids)}")
        attempted.append(method_key)
        try:
            explainer = _make_explainer(method_key, config, train_data_for_xai)
            _fit_surrogate_if_needed(explainer, config)
            result = explainer.explain(explained_instances)
            result = _aggregate_result_to_raw_features(
                config.data,
                result,
                explained_instances,
            )
            explanation_df = result.to_explanation_df(
                instance_ids=instance_ids,
                predictions=predictions,
                dataset_id=config.dataset_id,
                model_name=config.model_name,
            )
            explanation_df[EXPLANATION_METHOD_COL] = method_key

            out_path = config.output_dir / f"{method_key}_{config.model_name}_{config.dataset_id}.csv"
            explanation_df.to_csv(out_path, index=False)
            saved_paths.append(out_path)
            explanation_dfs.append(explanation_df)
            print(f"  Saved: {out_path} shape={explanation_df.shape}")
        except Exception as exc:
            failures[method_key] = f"{type(exc).__name__}: {exc}"
            print(f"  Skipped {method_name}: {type(exc).__name__}: {exc}")

    # Every method failing leaves a table with nothing but prediction rows,
    # which downstream code cannot distinguish from a design that legitimately
    # shows no explanations. Failing loudly here is the difference between an
    # error and a silently wrong study.
    if attempted and not explanation_dfs:
        detail = "; ".join(f"{name}: {reason}" for name, reason in failures.items())
        raise RuntimeError(
            f"None of the requested XAI methods produced explanations ({detail}). "
            "The combined table would contain only AI-prediction rows."
        )

    return saved_paths, explanation_dfs


def _explanation_ids_for_method(
    config: ExplanationRunConfig,
    method_key: str,
) -> np.ndarray:
    """Resolve the sampled instance IDs that need one method's explanation."""
    if config.instance_ids_by_method is not None:
        values = config.instance_ids_by_method.get(method_key, [])
    elif config.instance_ids is not None:
        values = config.instance_ids
    else:
        test_limit = min(config.max_test_instances, len(config.data.X_test))
        values = np.asarray(config.data.test_instance_ids)[:test_limit]

    instance_ids = np.asarray(
        list(dict.fromkeys(int(value) for value in values)),
        dtype=int,
    )
    if np.any(instance_ids < 0) or np.any(
        instance_ids >= len(config.data.split.X_model)
    ):
        raise ValueError("Explanation instance_ids contain an out-of-range dataset row.")
    if config.predictions_by_instance is not None:
        missing = [
            int(instance_id)
            for instance_id in instance_ids
            if int(instance_id) not in config.predictions_by_instance
        ]
        if missing:
            raise ValueError(
                "Explanation instance IDs are missing stored AI predictions: "
                f"{missing[:10]}."
            )
    return instance_ids


def _aggregate_result_to_raw_features(
    data: PreparedDataset,
    result: XAIAdapterResult,
    explained_instances: np.ndarray,
) -> XAIAdapterResult:
    """Collapse one-hot model attributions back to original feature columns."""
    raw_feature_count = len(data.raw_feature_names)
    values = np.asarray(result.values)

    if values.ndim != 2:
        return result
    if values.shape[1] == raw_feature_count:
        return result
    if values.shape[1] != len(data.model_feature_names):
        return result
    if not data.split.one_hot_encode:
        return result
    if not hasattr(data.dataset, "aggregate_importances"):
        return result

    aggregated_values = data.dataset.aggregate_importances(
        np.asarray(explained_instances),
        values,
    )
    metadata = dict(result.metadata)
    metadata["aggregated_from_model_features"] = True
    metadata["model_feature_names"] = list(data.model_feature_names)
    metadata["raw_feature_names"] = list(data.raw_feature_names)

    return XAIAdapterResult(
        values=np.asarray(aggregated_values),
        base_values=result.base_values,
        method=result.method,
        metadata=metadata,
    )


def generate_sim2real_explanations(
    *,
    model: Any,
    X: ArrayLike,
    properties: Sequence[str] = ("faithful", "sparse", "robust", "sparse_robust"),
    **kwargs,
) -> dict[str, Any]:
    """Generate property-optimized XAIsim2real explanations for one input set.

    Args:
        model: The model being explained.
        X: Feature rows to explain.
        properties: Explanation properties to optimize for, e.g. ``faithful``
            or ``sparse_robust``.
        **kwargs: Passed through to the generator.

    Returns:
        The explanations, keyed by property.
    """
    results = {}
    for property_name in properties:
        explainer = create_xai_method(
            "xaisim2real",
            model=model,
            property_name=property_name,
            **kwargs,
        )
        results[property_name] = explainer.explain(X)
    return results


def combine_explanation_tables(
    explanation_dfs: list[pd.DataFrame],
    config: ExplanationRunConfig,
) -> tuple[Optional[Path], Optional[pd.DataFrame]]:
    """Combine generated method-level explanations into one design-engine table.

    Args:
        explanation_dfs: The per-method tables to merge.
        config: Settings from ``init_explanation_run``.

    Returns:
        The written path and the combined table.
    """
    if not explanation_dfs:
        print("\nNo explanation CSVs were generated. Check installed XAI dependencies.")
        return None, None

    combined_df = pd.concat(explanation_dfs, ignore_index=True)
    combined_path = config.output_dir / f"de_{config.model_name}_{config.dataset_id}.csv"
    combined_df.to_csv(combined_path, index=False)
    print(f"\nCombined explanation CSV: {combined_path} shape={combined_df.shape}")
    return combined_path, combined_df


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

    Args:
        algorithm: The callable or object implementing the method.
        method_name: Name the method is recorded under.
        **kwargs: Passed through to the adapter.
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

    Args:
        fit_fn: Fits the surrogate to features and predictions.
        explain_fn: Produces explanations from the fitted surrogate.
        method_name: Name the method is recorded under.
    """
    return make_surrogate(fit_fn, explain_fn, name=method_name)


def get_coxam_xai_predictions(
    loader,
    instance_ids: List[int],
    method_type: str,
    app_id: str,
    model_name: str,
    **kwargs,
) -> List[Any]:
    """Apply a CoXAM XAI method to raw features from a data loader.

    Args:
        loader: Data loader supplying the raw feature rows.
        instance_ids: Instances to explain.
        method_type: CoXAM method to apply, e.g. ``decision_tree``.
        app_id: Dataset identifier recorded on the rows.
        model_name: Model name recorded on the rows.
        **kwargs: Passed through to the method.

    Returns:
        One explanation per requested instance.
    """
    method = create_xai_method(
        method_type,
        loader=loader,
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
    instead of precomputed `assets/explanations/CoXAM` surrogate tables.

    Args:
        dataset: An already-parsed dataset; alternative to the two below.
        csv_path: Path to a dataset CSV to parse.
        dataframe: An in-memory dataframe to use directly.
        instance_ids: Restrict generation to these instances; all by default.
        app_id: Dataset identifier recorded on the rows.
        model_name: Model name recorded on the rows.
        methods: Surrogate methods to generate.
        depths: Tree depths to fit for the decision-tree surrogate.
        variants: Logistic-regression variants to fit.
        variant: Which fitted variant to return as the method; must be one of
            ``variants``.
        top_k: Features kept in the sparse variant.
        random_state: Seed for surrogate fitting.
        **dataset_kwargs: Passed through to the dataset parser.

    Returns:
        The generated surrogate methods and their tables.

    Raises:
        ValueError: If ``variant`` is not among ``variants``.
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

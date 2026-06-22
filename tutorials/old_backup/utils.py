"""Reusable helpers for `colab_tutorial_full_workflow.ipynb`.

The notebook should read as a workflow. Keep validation, data preparation,
export plumbing, and cognitive-executor glue here so each notebook cell can
focus on one experiment step.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_IV_CONFIG = {
    "xai_method": {
        "type": "between",
        "levels": ["shap", "lime", "none"],
    },
    "tested_w_xai": {
        "type": "within",
        "randomization": "trial",
        "levels": [True, False],
    },
}

DEFAULT_CVS = {
    "age_group": ["young", "adult", "senior"],
    "gender": ["male", "female"],
}

DEFAULT_DVS = {
    "accuracy": ["continuous"],
    "task_time": ["continuous"],
}


def ensure_project_imports() -> None:
    """Make repository modules importable from notebooks run inside tutorials/."""
    project_root = str(PROJECT_ROOT)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)


def validate_iv_config(config: Dict[str, Dict[str, Any]]) -> None:
    """Validate independent-variable type, randomization, and level settings."""
    for name, cfg in config.items():
        iv_type = cfg.get("type")
        if iv_type not in ("within", "between"):
            raise ValueError(f"IV {name!r}: 'type' must be 'within' or 'between'.")

        randomization = cfg.get("randomization", "block")
        if iv_type == "within" and randomization not in ("block", "trial"):
            raise ValueError(f"IV {name!r}: 'randomization' must be 'block' or 'trial'.")
        if iv_type == "between" and "randomization" in cfg:
            raise ValueError(f"IV {name!r}: between-subjects IVs should not set 'randomization'.")

        levels = cfg.get("levels", [])
        if not isinstance(levels, list) or not levels:
            raise ValueError(f"IV {name!r}: 'levels' must be a non-empty list.")
        if len(set(map(str, levels))) != len(levels):
            raise ValueError(f"IV {name!r}: duplicate levels in {levels}.")


def validate_factors(factors: Dict[str, List[Any]]) -> None:
    """Validate control/dependent variable dictionaries."""
    for name, levels in factors.items():
        if not isinstance(levels, list) or not levels:
            raise ValueError(f"Factor {name!r} must map to a non-empty list.")
        if len(set(map(str, levels))) != len(levels):
            raise ValueError(f"Factor {name!r} has duplicate levels: {levels}.")


def set_iv(
    iv_config: Dict[str, Dict[str, Any]],
    name: str,
    iv_type: str,
    levels: List[Any],
    randomization: str = "block",
) -> Dict[str, Dict[str, Any]]:
    """Add or replace one IV after validating the requested configuration."""
    cfg = {"type": iv_type, "levels": levels}
    if iv_type == "within":
        cfg["randomization"] = randomization
    validate_iv_config({name: cfg})
    iv_config[name] = cfg
    return iv_config


def set_factor(factors: Dict[str, List[Any]], name: str, levels: List[Any]) -> Dict[str, List[Any]]:
    """Add or replace one CV/DV factor after validating its levels."""
    validate_factors({name: levels})
    factors[name] = levels
    return factors


def init_experiment_config() -> tuple[Dict[str, Dict[str, Any]], Dict[str, List[Any]], Dict[str, List[Any]]]:
    """Start an empty experiment configuration for iterative IV/CV/DV definition."""
    return {}, {}, {}


def between_iv(levels: List[Any]) -> Dict[str, Any]:
    """Create a between-subjects IV spec."""
    return {"type": "between", "levels": levels}


def within_iv(levels: List[Any], randomization: str = "block") -> Dict[str, Any]:
    """Create a within-subjects IV spec."""
    return {"type": "within", "randomization": randomization, "levels": levels}


def configure_experiment(
    ivs: Optional[Dict[str, Dict[str, Any]]] = None,
    cvs: Optional[Dict[str, List[Any]]] = None,
    dvs: Optional[Dict[str, List[Any]]] = None,
    *,
    available_datasets: Optional[List[str]] = None,
    show: bool = True,
) -> tuple[Dict[str, Dict[str, Any]], Dict[str, List[Any]], Dict[str, List[Any]]]:
    """
    Build, validate, and optionally summarize IV/CV/DV configuration.

    Call with no arguments for the tutorial defaults, or pass custom dictionaries
    built directly or with `between_iv(...)`, `within_iv(...)`, and `set_factor(...)`.
    """
    iv_config = deepcopy(ivs if ivs is not None else DEFAULT_IV_CONFIG)
    control_vars = deepcopy(cvs if cvs is not None else DEFAULT_CVS)
    dependent_vars = deepcopy(dvs if dvs is not None else DEFAULT_DVS)

    validate_iv_config(iv_config)
    validate_factors(control_vars)
    validate_factors(dependent_vars)

    if show:
        print_experiment_config(
            iv_config,
            control_vars,
            dependent_vars,
            available_datasets=available_datasets,
        )

    return iv_config, control_vars, dependent_vars


def validate_experiment_config(
    iv_config: Dict[str, Dict[str, Any]],
    cvs: Dict[str, List[Any]],
    dvs: Dict[str, List[Any]],
    *,
    available_datasets: Optional[List[str]] = None,
    show: bool = True,
) -> tuple[Dict[str, Dict[str, Any]], Dict[str, List[Any]], Dict[str, List[Any]]]:
    """Validate and optionally summarize an iteratively-built experiment config."""
    validate_iv_config(iv_config)
    validate_factors(cvs)
    validate_factors(dvs)

    if show:
        print_experiment_config(
            iv_config,
            cvs,
            dvs,
            available_datasets=available_datasets,
        )

    return iv_config, cvs, dvs


def print_experiment_config(
    iv_config: Dict[str, Dict[str, Any]],
    cvs: Dict[str, List[Any]],
    dvs: Dict[str, List[Any]],
    *,
    available_datasets: Optional[List[str]] = None,
) -> None:
    """Print a compact summary of IV/CV/DV settings."""
    if available_datasets is not None:
        print("Available training datasets:", available_datasets)

    print("\nIV configuration:")
    for name, cfg in iv_config.items():
        randomization = cfg.get("randomization", "-")
        print(
            f"  {name:<20} type={cfg['type']:<8} "
            f"randomization={randomization:<5} levels={cfg['levels']}"
        )
    print(f"\nCVs: {list(cvs.keys())}")
    print(f"DVs: {list(dvs.keys())}")


@dataclass
class DatasetSplit:
    """Container for the dataset artifacts shared across workflow steps."""

    dataset_id: str
    dataset: Any
    df: pd.DataFrame
    X_raw: np.ndarray
    y: np.ndarray
    feature_names: List[str]
    raw_instance_ids: np.ndarray
    X_model: np.ndarray
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    train_instance_ids: np.ndarray
    test_instance_ids: np.ndarray


@dataclass
class PreparedDataset:
    """Notebook-facing wrapper around a dataloader dataset and its split."""

    split: DatasetSplit

    @property
    def dataset_id(self) -> str:
        return self.split.dataset_id

    @property
    def dataset(self) -> Any:
        return self.split.dataset

    @property
    def df(self) -> pd.DataFrame:
        return self.split.df

    @property
    def feature_names(self) -> List[str]:
        return self.split.feature_names

    @property
    def X_train(self) -> np.ndarray:
        return self.split.X_train

    @property
    def X_test(self) -> np.ndarray:
        return self.split.X_test

    @property
    def y_train(self) -> np.ndarray:
        return self.split.y_train

    @property
    def y_test(self) -> np.ndarray:
        return self.split.y_test

    @property
    def train_instance_ids(self) -> np.ndarray:
        return self.split.train_instance_ids

    @property
    def test_instance_ids(self) -> np.ndarray:
        return self.split.test_instance_ids

    @property
    def label_column(self) -> str:
        return self.dataset.target_name or "target"


@dataclass
class ExplanationRunConfig:
    """Inputs shared by explanation-generation notebook steps."""

    data: PreparedDataset
    iv_config: Dict[str, Dict[str, Any]]
    trained_engine: Any
    model_name: str = "mlp"
    output_dir: Path = Path("generated_explanation")
    target: int = 1
    method_kwargs: Optional[Dict[str, Dict[str, Any]]] = None

    @property
    def dataset_id(self) -> str:
        return self.data.dataset_id


@dataclass
class DesignRoles:
    """Result of splitting experiment IVs into design roles."""

    between_ivs: Dict[str, Dict[str, Any]]
    block_within_ivs: Dict[str, Dict[str, Any]]
    trial_within_ivs: Dict[str, Dict[str, Any]]
    within_ivs: Dict[str, Dict[str, Any]]
    all_conditions: List[Dict[str, Any]]


@dataclass
class TrialBuildConfig:
    """Notebook-facing settings for experimental trial generation."""

    data: PreparedDataset
    iv_config: Dict[str, Dict[str, Any]]
    cvs: Dict[str, List[Any]]
    model_name: str = "mlp"
    participants_per_between_condition: int = 25
    trials_per_participant: int = 10
    trial_randomization_strategy: str = "balanced"
    instance_wise_explanation: bool = False
    shuffle_instances: bool = True
    seed: int = 42
    output_dir: Path = Path("experiment_output")
    trials_csv: str = "trials.csv"
    trials_json: str = "trials.json"
    summary_json: str = "design_summary.json"


@dataclass
class TrialGenerationResult:
    """Artifacts produced by a trial-generation run."""

    config: Dict[str, Any]
    trials: List[Dict[str, Any]]
    assignments: List[Dict[str, Any]]
    orders: List[Any]
    strategy: str
    csv_path: str
    json_path: str
    summary_path: str
    design_roles: DesignRoles

    @property
    def experiment_structure(self) -> DesignRoles:
        """Experiment-facing alias for the IV split used to build trials."""
        return self.design_roles


def inspect_design_roles(
    iv_config: Dict[str, Dict[str, Any]],
    *,
    show: bool = True,
) -> DesignRoles:
    """Split IVs by design role and optionally print a compact summary."""
    from src.experiment_design.counterbalance import (
        factorial_conditions,
        split_ivs_by_design_role,
    )

    between_ivs, block_within_ivs, trial_within_ivs = split_ivs_by_design_role(iv_config)
    within_ivs = {**block_within_ivs, **trial_within_ivs}
    all_conditions = factorial_conditions({k: v["levels"] for k, v in iv_config.items()})

    if show:
        print("IV split:")
        print(f"  between-subjects      : {between_ivs}")
        print(f"  block-counterbalanced : {block_within_ivs}")
        print(f"  trial-randomized      : {trial_within_ivs}")
        print(f"\nAll factorial conditions: {len(all_conditions)}")
        for condition in all_conditions:
            print(f"  {condition}")

    return DesignRoles(
        between_ivs=between_ivs,
        block_within_ivs=block_within_ivs,
        trial_within_ivs=trial_within_ivs,
        within_ivs=within_ivs,
        all_conditions=all_conditions,
    )


def inspect_experiment_structure(
    iv_config: Dict[str, Dict[str, Any]],
    *,
    show: bool = True,
) -> DesignRoles:
    """Experiment-facing alias for `inspect_design_roles`."""
    return inspect_design_roles(iv_config, show=show)


def init_trial_build_config(
    data: PreparedDataset,
    iv_config: Dict[str, Dict[str, Any]],
    cvs: Dict[str, List[Any]],
    *,
    model_name: str = "mlp",
    participants_per_between_condition: int = 25,
    trials_per_participant: int = 10,
    trial_randomization_strategy: str = "balanced",
    instance_wise_explanation: bool = False,
    shuffle_instances: bool = True,
    seed: int = 42,
    output_dir: str | Path = "experiment_output",
    trials_csv: str = "trials.csv",
    trials_json: str = "trials.json",
    summary_json: str = "design_summary.json",
) -> TrialBuildConfig:
    """Collect trial-generation settings in one notebook-friendly config object."""
    return TrialBuildConfig(
        data=data,
        iv_config=iv_config,
        cvs=cvs,
        model_name=model_name,
        participants_per_between_condition=participants_per_between_condition,
        trials_per_participant=trials_per_participant,
        trial_randomization_strategy=trial_randomization_strategy,
        instance_wise_explanation=instance_wise_explanation,
        shuffle_instances=shuffle_instances,
        seed=seed,
        output_dir=Path(output_dir),
        trials_csv=trials_csv,
        trials_json=trials_json,
        summary_json=summary_json,
    )


def generate_experimental_trials(
    config: TrialBuildConfig,
    *,
    show: bool = True,
    preview_rows: int = 10,
) -> TrialGenerationResult:
    """Build and export experimental trials from a notebook trial config."""
    from src.experiment_design.counterbalance import (
        assign_participants,
        build_trial_sequence,
        choose_counterbalancing,
        export_design_summary,
        export_trials_csv,
        export_trials_json,
        factorial_conditions,
        make_within_condition_order_labels,
    )

    experiment_structure = inspect_experiment_structure(config.iv_config, show=False)
    within_labels = make_within_condition_order_labels(experiment_structure.block_within_ivs)
    orders, strategy = choose_counterbalancing(within_labels)

    between_groups = (
        factorial_conditions(experiment_structure.between_ivs)
        if experiment_structure.between_ivs
        else [{}]
    )
    n_participants = config.participants_per_between_condition * len(between_groups)
    assignments = assign_participants(
        n_participants,
        orders,
        experiment_structure.between_ivs or None,
    )

    instance_pool = [
        {"dataId": config.data.dataset_id, "instanceId": str(instance_id)}
        for instance_id in config.data.test_instance_ids
    ]
    controlled_vars = build_controlled_vars(config.model_name, config.cvs)

    trials = build_trial_sequence(
        assignments=assignments,
        instance_pool=instance_pool,
        trials_per_participant=config.trials_per_participant,
        controlled_vars=controlled_vars,
        id_map={"dataId": "dataId", "instanceId": "instanceId"},
        trial_randomized_ivs=experiment_structure.trial_within_ivs or None,
        trial_randomization_strategy=config.trial_randomization_strategy,
        instance_wise_explanation=config.instance_wise_explanation,
        shuffle_instances=config.shuffle_instances,
        seed=config.seed,
    )

    output_dir = config.output_dir
    output_dir.mkdir(exist_ok=True)
    csv_path = export_trials_csv(trials, output_dir / config.trials_csv)
    json_path = export_trials_json(trials, output_dir / config.trials_json)
    summary_path = export_design_summary(
        iv_config={k: v["levels"] for k, v in config.iv_config.items()},
        between_ivs=experiment_structure.between_ivs,
        within_ivs=experiment_structure.within_ivs,
        block_within_ivs=experiment_structure.block_within_ivs,
        trial_within_ivs=experiment_structure.trial_within_ivs,
        trial_randomization_strategy=config.trial_randomization_strategy,
        trials_per_participant=config.trials_per_participant,
        strategy=strategy,
        orders=orders,
        assignments=assignments,
        path=output_dir / config.summary_json,
    )

    if show:
        print(f"Counterbalancing strategy: {strategy}")
        print(f"Participant assignments: {len(assignments)} total")
        print(f"Instance pool rows: {len(instance_pool)}")
        print(f"Trial rows: {len(trials)}")
        print("Exported trial artifacts:")
        print(f"  CSV     : {csv_path}")
        print(f"  JSON    : {json_path}")
        print(f"  Summary : {summary_path}")
        preview_trial_rows(trials, experiment_structure, n=preview_rows)

    return TrialGenerationResult(
        config={
            "ivs": config.iv_config,
            "cvs": config.cvs,
            "dataset": {
                "dataset_id": config.data.dataset_id,
                "model_type": config.model_name,
                "id_map": {"dataId": "dataId", "instanceId": "instanceId"},
            },
            "sampling": {
                "participants_per_between_condition": config.participants_per_between_condition,
                "trials_per_participant": config.trials_per_participant,
                "trial_randomization_strategy": config.trial_randomization_strategy,
                "instance_wise_explanation": config.instance_wise_explanation,
                "shuffle_instances": config.shuffle_instances,
            },
            "output": {
                "out_dir": str(config.output_dir),
                "trials_csv": config.trials_csv,
                "trials_json": config.trials_json,
                "summary_json": config.summary_json,
            },
            "seed": config.seed,
        },
        trials=trials,
        assignments=assignments,
        orders=orders,
        strategy=strategy,
        csv_path=str(csv_path),
        json_path=str(json_path),
        summary_path=str(summary_path),
        design_roles=experiment_structure,
    )


def preview_trial_rows(
    trials: List[Dict[str, Any]],
    experiment_structure: DesignRoles,
    *,
    n: int = 10,
) -> None:
    """Print a compact preview of trial assignment, condition, and instance ids."""
    key_cols = [
        "participantId",
        "trialId",
        "block",
        "trialWithinBlock",
        "withinCondition",
        *experiment_structure.between_ivs.keys(),
        *experiment_structure.block_within_ivs.keys(),
        *experiment_structure.trial_within_ivs.keys(),
        "dataId",
        "instanceId",
    ]

    print(f"\nPreviewing first {min(n, len(trials))} trial rows:")
    for trial in trials[:n]:
        print({k: trial[k] for k in key_cols if k in trial})


def generate_trials_from_ui_config(
    config: Dict[str, Any],
    *,
    show: bool = True,
) -> TrialGenerationResult:
    """Generate trial CSV/JSON/summary artifacts from a UI-exported config."""
    from src.experiment_design.counterbalance import (
        assign_participants,
        build_trial_sequence,
        choose_counterbalancing,
        export_design_summary,
        export_trials_csv,
        export_trials_json,
        factorial_conditions,
        make_within_condition_order_labels,
    )

    iv_config = config["ivs"]
    cvs = config["cvs"]
    dataset_cfg = config["dataset"]
    sampling_cfg = config["sampling"]
    output_cfg = config["output"]

    design_roles = inspect_design_roles(iv_config, show=False)
    within_labels = make_within_condition_order_labels(design_roles.block_within_ivs)
    orders, strategy = choose_counterbalancing(within_labels)

    between_groups = (
        factorial_conditions(design_roles.between_ivs)
        if design_roles.between_ivs
        else [{}]
    )
    n_participants = sampling_cfg["participants_per_between_condition"] * len(between_groups)
    assignments = assign_participants(
        n_participants,
        orders,
        design_roles.between_ivs or None,
    )

    instance_pool = load_csv_records(dataset_cfg["explanation_csv"])
    controlled_vars = build_controlled_vars(dataset_cfg["model_type"], cvs)

    trials = build_trial_sequence(
        assignments=assignments,
        instance_pool=instance_pool,
        trials_per_participant=sampling_cfg["trials_per_participant"],
        controlled_vars=controlled_vars,
        id_map=dataset_cfg["id_map"],
        trial_randomized_ivs=design_roles.trial_within_ivs or None,
        trial_randomization_strategy=sampling_cfg["trial_randomization_strategy"],
        instance_wise_explanation=sampling_cfg.get("instance_wise_explanation", False),
        shuffle_instances=sampling_cfg["shuffle_instances"],
        seed=config["seed"],
    )

    out_dir = Path(output_cfg["out_dir"])
    out_dir.mkdir(exist_ok=True)
    csv_path = export_trials_csv(trials, out_dir / output_cfg["trials_csv"])
    json_path = export_trials_json(trials, out_dir / output_cfg["trials_json"])
    summary_path = export_design_summary(
        iv_config={k: v["levels"] for k, v in iv_config.items()},
        between_ivs=design_roles.between_ivs,
        within_ivs=design_roles.within_ivs,
        block_within_ivs=design_roles.block_within_ivs,
        trial_within_ivs=design_roles.trial_within_ivs,
        trial_randomization_strategy=sampling_cfg["trial_randomization_strategy"],
        trials_per_participant=sampling_cfg["trials_per_participant"],
        strategy=strategy,
        orders=orders,
        assignments=assignments,
        path=out_dir / output_cfg["summary_json"],
    )

    if show:
        print(f"Converted UI config into {len(trials)} trial rows.")
        print(f"  CSV     : {csv_path}")
        print(f"  JSON    : {json_path}")
        print(f"  Summary : {summary_path}")

    return TrialGenerationResult(
        config=config,
        trials=trials,
        assignments=assignments,
        orders=orders,
        strategy=strategy,
        csv_path=csv_path,
        json_path=json_path,
        summary_path=summary_path,
        design_roles=design_roles,
    )


def generate_trials_from_ui_json(
    config_path: str | Path,
    *,
    show: bool = True,
) -> Optional[TrialGenerationResult]:
    """Load a UI-exported JSON file and generate trial artifacts from it."""
    config_path = Path(config_path)
    if not config_path.exists():
        print(f"UI config not found: {config_path}")
        return None

    config = load_json_config(config_path)
    return generate_trials_from_ui_config(config, show=show)


def prepare_dataset(
    dataset_id: str,
    *,
    feature_cols: Optional[Sequence[str]] = None,
    num_features: Optional[int] = None,
    rank_features_by_target: bool = True,
    test_size: float = 0.2,
    random_state: int = 42,
    show_available: bool = True,
    show_summary: bool = True,
) -> PreparedDataset:
    """
    Load a dataset through `src.data_loaders`, split it, and return one wrapper.

    This keeps the notebook from manually unpacking train/test arrays,
    instance ids, feature names, and label-column metadata.

    `feature_cols` optionally selects feature columns by dataset feature name.
    `rank_features_by_target` orders candidate features by absolute correlation
    with the target before model training and explanation generation.
    `num_features` optionally keeps only the first N ranked/selected features.
    """
    ensure_project_imports()
    from src.data_loaders import list_original_datasets, load_original_dataset

    if show_available:
        print("Available training datasets:", list_original_datasets())

    dataset = load_original_dataset(dataset_id)
    selected_features = _resolve_feature_selection(
        dataset,
        feature_cols=feature_cols,
        num_features=num_features,
        rank_features_by_target=rank_features_by_target,
    )
    if selected_features is not None:
        dataset = dataset.use_specific_features(selected_features)

    split = split_loaded_dataset(
        dataset_id,
        dataset,
        test_size=test_size,
        random_state=random_state,
    )

    if show_summary:
        print_dataset_split_summary(split)

    return PreparedDataset(split=split)


def _resolve_feature_selection(
    dataset: Any,
    *,
    feature_cols: Optional[Sequence[str]] = None,
    num_features: Optional[int] = None,
    rank_features_by_target: bool = True,
) -> Optional[List[str]]:
    """Validate, rank, and combine optional feature-list/count filters."""
    if num_features is not None and num_features <= 0:
        raise ValueError("num_features must be a positive integer.")

    available_features = list(dataset.feature_names)
    selected_features = list(feature_cols) if feature_cols is not None else list(available_features)

    missing_features = [feature for feature in selected_features if feature not in available_features]
    if missing_features:
        raise ValueError(
            f"Feature(s) not found: {missing_features}. "
            f"Available features: {list(available_features)}"
        )

    if rank_features_by_target:
        selected_features = _rank_features_by_target_correlation(dataset, selected_features)

    if num_features is not None:
        selected_features = selected_features[:num_features]

    if feature_cols is None and num_features is None and not rank_features_by_target:
        return None
    return selected_features


def _rank_features_by_target_correlation(dataset: Any, feature_names: Sequence[str]) -> List[str]:
    """Rank features by absolute Pearson correlation with the target."""
    y = pd.Series(np.asarray(dataset.y, dtype=float))
    ranked = []

    for original_position, feature_name in enumerate(feature_names):
        feature_idx = dataset.feature_names.index(feature_name)
        values = pd.to_numeric(pd.Series(dataset.X[:, feature_idx]), errors="coerce")
        valid = values.notna() & y.notna()
        if valid.sum() < 2 or values[valid].nunique() <= 1 or y[valid].nunique() <= 1:
            corr = 0.0
        else:
            corr = float(values[valid].corr(y[valid]))
            if np.isnan(corr):
                corr = 0.0
        ranked.append((feature_name, abs(corr), corr, original_position))

    ranked.sort(key=lambda item: (-item[1], item[3]))
    return [feature_name for feature_name, _abs_corr, _corr, _position in ranked]


def init_explanation_run(
    data: PreparedDataset,
    iv_config: Dict[str, Dict[str, Any]],
    trained_engine: Any,
    *,
    model_name: str = "mlp",
    output_dir: str | Path = "generated_explanation",
    target: int = 1,
    method_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
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


def get_xai_methods_from_design(iv_config: Dict[str, Dict[str, Any]]) -> List[Any]:
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
) -> tuple[List[Path], List[pd.DataFrame]]:
    """Generate one explanation CSV per non-control XAI method."""
    from src.xai_adapter import create_xai_method_from_engine

    train_data_for_xai = make_train_data_for_xai(config.data.split, config.data.y_train)
    instance_ids = np.asarray(config.data.test_instance_ids)
    predictions = predict_labels(config.trained_engine, config.data.X_test)

    saved_paths: List[Path] = []
    explanation_dfs: List[pd.DataFrame] = []

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
    explanation_dfs: List[pd.DataFrame],
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


def load_dataset_and_split(
    dataset_id: str,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
) -> DatasetSplit:
    """
    Load a tabular training dataset through `src.data_loaders` and split it.

    `X_raw` keeps the original feature values for trial/cognitive inspection.
    `X_model` uses the dataset object's model-preparation path, so training and
    XAI generation follow the repository's existing preprocessing logic.
    """
    ensure_project_imports()
    from src.data_loaders import load_original_dataset

    dataset = load_original_dataset(dataset_id)
    return split_loaded_dataset(
        dataset_id,
        dataset,
        test_size=test_size,
        random_state=random_state,
    )


def split_loaded_dataset(
    dataset_id: str,
    dataset: Any,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
) -> DatasetSplit:
    """
    Split an already-loaded `src.data_loaders` dataset object.

    Use this when the notebook should visibly call the repository dataloader
    before delegating split mechanics to the helper layer.
    """
    X_model, y = dataset.prepare_data_for_model(one_hot_encode=True)
    X_raw = np.asarray(dataset.X, dtype=np.float32)
    y = np.asarray(y)
    raw_instance_ids = np.arange(len(y))

    target_name = dataset.target_name or "target"
    df = pd.DataFrame(X_raw, columns=dataset.feature_names)
    df[target_name] = y

    stratify = y if len(np.unique(y)) > 1 else None
    X_train, X_test, y_train, y_test, train_instance_ids, test_instance_ids = train_test_split(
        np.asarray(X_model, dtype=np.float32),
        y,
        raw_instance_ids,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    return DatasetSplit(
        dataset_id=dataset_id,
        dataset=dataset,
        df=df,
        X_raw=X_raw,
        y=y,
        feature_names=list(dataset.feature_names),
        raw_instance_ids=raw_instance_ids,
        X_model=np.asarray(X_model, dtype=np.float32),
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        train_instance_ids=train_instance_ids,
        test_instance_ids=test_instance_ids,
    )


def print_dataset_split_summary(split: DatasetSplit) -> None:
    """Print a compact check of the selected dataset and split."""
    print(f"Dataset   : {split.dataset_id}  ({split.X_model.shape[0]} rows, {split.X_model.shape[1]} model features)")
    print(f"Features  : {split.feature_names}")
    print(f"Train set : {split.X_train.shape[0]} samples  ({split.X_train.shape[0] / len(split.y) * 100:.0f}%)")
    print(f"Test set  : {split.X_test.shape[0]} samples  ({split.X_test.shape[0] / len(split.y) * 100:.0f}%)")
    for label in np.unique(split.y_train):
        print(f"Class balance (train) -> class {label}: {(split.y_train == label).sum()}")
    print(f"First test instanceIds: {split.test_instance_ids[:10].tolist()}")


def make_train_data_for_xai(split: DatasetSplit, y_train: np.ndarray) -> SimpleNamespace:
    """Package model-training data in the shape expected by XAI adapters."""
    return SimpleNamespace(
        X=split.X_train,
        y=y_train,
        feature_names=split.feature_names,
    )


def load_csv_records(path: str | Path) -> list[dict[str, Any]]:
    """Load a CSV as a list of dictionaries."""
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_json_config(path: str | Path) -> dict[str, Any]:
    """Load a JSON config exported by the UI."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_controlled_vars(model_type: str, cvs: Dict[str, List[Any]]) -> Dict[str, str]:
    """Encode control-variable levels as trial-table metadata columns."""
    return {
        "model_type": model_type,
        **{f"CV_{k}_levels": "|".join(str(v) for v in vals) for k, vals in cvs.items()},
    }


def load_explanation_pool(
    explanation_csv: str | Path = "generated_explanation/de_mlp_wine_quality.csv",
) -> pd.DataFrame:
    """Load generated explanations used by the trial generator/executor."""
    return pd.read_csv(explanation_csv)


def get_trial_instance_attributes(
    trial_info: Dict[str, Any],
    raw_dataset: pd.DataFrame,
    *,
    label_column: str,
) -> Dict[str, float]:
    """Extract raw feature values for the trial's original dataset row id."""
    instance_id = int(trial_info["instanceId"])
    row = raw_dataset.iloc[instance_id]
    feature_row = row.drop(labels=[label_column], errors="ignore")
    return {k: float(v) for k, v in feature_row.items()}


def get_trial_ai_prediction(
    trial_info: Dict[str, Any],
    explanation_pool: pd.DataFrame,
) -> Optional[int]:
    """Return the trained AI model prediction stored in the explanation CSV."""
    if "pred" not in explanation_pool.columns:
        return None

    instance_id = int(trial_info["instanceId"])
    matches = explanation_pool[explanation_pool["instanceId"].astype(int) == instance_id]
    if matches.empty:
        return None
    return int(matches.iloc[0]["pred"])


def get_trial_instance_explanation(
    trial_info: Dict[str, Any],
    explanation_pool: pd.DataFrame,
) -> Dict[str, float]:
    """Select explanation values matching the trial's XAI method and instance."""
    xai_method = str(trial_info.get("xai_method", trial_info.get("xai_type", "none"))).lower()
    if xai_method in {"none", "no_xai", "control"}:
        return {}

    instance_id = int(trial_info["instanceId"])
    matches = explanation_pool[
        (explanation_pool["instanceId"].astype(int) == instance_id)
        & (explanation_pool["expMethod"].astype(str).str.lower() == xai_method)
    ]
    if matches.empty:
        return {}

    row = matches.iloc[0]
    explanation_cols = [c for c in explanation_pool.columns if c.startswith("a") and c.endswith("_i")]
    explanation = {c: float(row[c]) for c in explanation_cols}

    for optional_col in ["pred", "i_max", "intercept"]:
        if optional_col in row:
            explanation[optional_col] = float(row[optional_col])

    return explanation


def build_single_trial_cognitive_input(
    trial_info: Dict[str, Any],
    raw_dataset: pd.DataFrame,
    explanation_pool: pd.DataFrame,
    *,
    label_column: str,
) -> Dict[str, Any]:
    """Combine trial metadata, raw instance attributes, AI pred, and XAI values."""
    return {
        "trial_info": dict(trial_info),
        "instance_attributes": get_trial_instance_attributes(
            trial_info,
            raw_dataset,
            label_column=label_column,
        ),
        "instance_explanation": get_trial_instance_explanation(
            trial_info,
            explanation_pool,
        ),
        "ai_prediction": get_trial_ai_prediction(trial_info, explanation_pool),
    }


def dummy_cognitive_model(
    cognitive_params: Dict[str, float],
    dvs: Dict[str, List[Any]],
    trial_data: Dict[str, Any],
) -> Dict[str, Any]:
    """Placeholder cognitive model using current CoXAM-style parameter names."""
    trial_info = trial_data.get("trial_info", {})
    explanation = trial_data.get("instance_explanation", {}) or {}

    has_xai = bool(explanation) and str(trial_info.get("xai_method", "none")).lower() != "none"
    attr_values = [abs(v) for k, v in explanation.items() if k.startswith("a") and k.endswith("_i")]
    explanation_strength = float(np.mean(attr_values)) if attr_values else 0.0

    retrieval_threshold = float(cognitive_params.get("cog_retrieval_threshold", -0.3))
    chi_value = float(cognitive_params.get("cog_chi", 0.001))
    ddm_a = float(cognitive_params.get("cog_ddm_a", 0.8))
    ddm_s = float(cognitive_params.get("cog_ddm_s", 1.0))
    lapse = float(cognitive_params.get("lapse", 0.005))
    latency_factor = float(cognitive_params.get("cog_latency_factor", 0.2))
    T_enc = float(cognitive_params.get("cog_T_enc", 1.5))
    T_op = float(cognitive_params.get("cog_T_op", 0.5))

    # Higher retrieval thresholds and stronger explanations increase the
    # placeholder probability; lapse reserves a small error floor.
    accuracy_probability = 0.5 + (0.08 * ddm_a) + (0.05 * ddm_s) - (0.03 * abs(retrieval_threshold))
    if has_xai:
        accuracy_probability += chi_value + (0.05 * explanation_strength)
    accuracy_probability = float(np.clip(accuracy_probability, lapse, 1.0 - lapse))

    pred_time = T_enc + T_op + latency_factor * (1.0 + abs(retrieval_threshold))
    if has_xai:
        pred_time += ddm_a * max(ddm_s, 0.0) + explanation_strength
    pred_time = float(max(pred_time, 0.0))

    outputs: Dict[str, Any] = {}
    for dv_name in dvs:
        dv_key = dv_name.lower()
        if "accuracy" in dv_key or "prob" in dv_key or "correct" in dv_key:
            outputs[dv_name] = accuracy_probability
        elif "time" in dv_key or "rt" in dv_key or "duration" in dv_key:
            outputs[dv_name] = pred_time
        else:
            outputs[dv_name] = accuracy_probability

    outputs["prob_correct"] = accuracy_probability
    outputs["pred_time"] = pred_time
    outputs["agent_prediction"] = bool(accuracy_probability > 0.5)
    outputs["ai_prediction"] = trial_data.get("ai_prediction")
    return outputs


def default_cognitive_params() -> Dict[str, float]:
    """Return placeholder parameters named like current `src/cognitive_models` artifacts."""
    return {
        "cog_retrieval_threshold": -0.3,
        "cog_latency_factor": 0.2,
        "cog_T_enc": 1.5,
        "cog_T_op": 0.5,
        "cog_ddm_a": 0.8,
        "cog_ddm_s": 1.0,
        "cog_chi": 0.001,
        "lapse": 0.005,
    }


def select_trial_rows(
    trials_df: pd.DataFrame,
    mode: str,
    *,
    participant_id: Optional[int] = None,
    condition_filter: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """Select trial rows for one trial, participant, condition, or experiment."""
    mode = mode.lower()
    selected = trials_df.copy()

    if mode in {"trial", "trial_by_trial"}:
        return selected.iloc[:1].copy()

    if mode in {"participant", "participant_by_participant"}:
        if participant_id is None:
            participant_id = int(selected["participantId"].iloc[0])
        return selected[selected["participantId"].astype(int) == int(participant_id)].copy()

    if mode in {"condition", "whole_condition"}:
        if not condition_filter:
            raise ValueError("condition_filter is required for mode='whole_condition'.")
        for col, value in condition_filter.items():
            selected = selected[selected[col].astype(str) == str(value)]
        return selected.copy()

    if mode in {"experiment", "whole_experiment"}:
        return selected.copy()

    raise ValueError(
        "mode must be one of: trial_by_trial, participant_by_participant, "
        "whole_condition, whole_experiment"
    )


def run_experiment_executor(
    trials: List[Dict[str, Any]] | pd.DataFrame,
    cognitive_params: Dict[str, float],
    dvs: Dict[str, List[Any]],
    raw_dataset: pd.DataFrame,
    explanation_pool: pd.DataFrame,
    *,
    mode: str = "whole_experiment",
    participant_id: Optional[int] = None,
    condition_filter: Optional[Dict[str, Any]] = None,
    cognitive_model=dummy_cognitive_model,
    label_column: str,
    explanation_prefix: str = "exp_",
    cognitive_param_prefix: str = "cog_param_",
) -> pd.DataFrame:
    """Run cognitive agents over selected trials and append results."""
    trials_df = pd.DataFrame(trials).copy()
    selected = select_trial_rows(
        trials_df,
        mode,
        participant_id=participant_id,
        condition_filter=condition_filter,
    )

    executed_rows = []
    for _, trial_row in selected.iterrows():
        trial_info = trial_row.to_dict()
        trial_context = build_single_trial_cognitive_input(
            trial_info,
            raw_dataset,
            explanation_pool,
            label_column=label_column,
        )
        model_outputs = cognitive_model(cognitive_params, dvs, trial_context)

        explanation_cols = {
            f"{explanation_prefix}{k}": v
            for k, v in trial_context["instance_explanation"].items()
        }
        cognitive_param_cols = {
            f"{cognitive_param_prefix}{k}": v
            for k, v in cognitive_params.items()
        }

        ai_prediction = model_outputs.get("ai_prediction")
        agent_prediction = model_outputs.get("agent_prediction")
        cognitive_correct_vs_ai = (
            None if ai_prediction is None else bool(int(agent_prediction) == int(ai_prediction))
        )

        executed_rows.append({
            **trial_info,
            **explanation_cols,
            **cognitive_param_cols,
            **model_outputs,
            "cognitive_correct_vs_ai": cognitive_correct_vs_ai,
        })

    return pd.DataFrame(executed_rows)


def save_simulated_results(
    simulated_results: pd.DataFrame,
    *,
    out_dir: str | Path = "experiment_output",
) -> tuple[str, str]:
    """Save simulated experiment results as CSV and JSON."""
    os.makedirs(out_dir, exist_ok=True)
    csv_path = str(Path(out_dir) / "simulated_results.csv")
    json_path = str(Path(out_dir) / "simulated_results.json")

    simulated_results.to_csv(csv_path, index=False)
    simulated_results.to_json(json_path, orient="records", indent=2)
    return csv_path, json_path

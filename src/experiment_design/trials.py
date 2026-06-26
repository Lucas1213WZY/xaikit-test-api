"""High-level trial generation helpers for experiment designs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from src.data_loaders import PreparedDataset, load_csv_records, load_json_config
from src.workflow_standard import DEFAULT_EXPLANATION_INSTANCE_LIMIT

from .config import (
    DesignRoles,
    build_controlled_vars,
    inspect_design_roles,
    inspect_experiment_structure,
)
from .counterbalance import (
    assign_participants,
    build_trial_sequence,
    choose_counterbalancing,
    export_design_summary,
    export_trials_csv,
    export_trials_json,
    factorial_conditions,
    make_within_condition_order_labels,
)
from .support import validate_ui_config_support


@dataclass
class TrialBuildConfig:
    """Notebook-facing settings for experimental trial generation."""

    data: PreparedDataset
    iv_config: dict[str, dict[str, Any]]
    cvs: dict[str, list[Any]]
    model_name: Optional[str] = None
    participants_per_between_condition: int = 25
    trials_per_participant: int = 10
    trial_randomization_strategy: str = "balanced"
    instance_wise_explanation: bool = False
    shuffle_instances: bool = True
    max_trial_instances: Optional[int] = DEFAULT_EXPLANATION_INSTANCE_LIMIT
    seed: int = 42
    output_dir: Path = Path("experiment_output")
    trials_csv: str = "trials.csv"
    trials_json: str = "trials.json"
    summary_json: str = "design_summary.json"


@dataclass
class TrialGenerationResult:
    """Artifacts produced by a trial-generation run."""

    config: dict[str, Any]
    trials: list[dict[str, Any]]
    assignments: list[dict[str, Any]]
    orders: list[Any]
    strategy: str
    csv_path: str
    json_path: str
    summary_path: str
    design_roles: DesignRoles

    @property
    def experiment_structure(self) -> DesignRoles:
        """Experiment-facing alias for the IV split used to build trials."""
        return self.design_roles


def init_trial_build_config(
    data: PreparedDataset,
    iv_config: dict[str, dict[str, Any]],
    cvs: dict[str, list[Any]],
    *,
    model_name: Optional[str] = None,
    participants_per_between_condition: int = 25,
    trials_per_participant: int = 10,
    trial_randomization_strategy: str = "balanced",
    instance_wise_explanation: bool = False,
    shuffle_instances: bool = True,
    max_trial_instances: Optional[int] = DEFAULT_EXPLANATION_INSTANCE_LIMIT,
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
        max_trial_instances=max_trial_instances,
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

    trial_instance_ids = config.data.test_instance_ids
    if config.max_trial_instances is not None:
        trial_instance_ids = trial_instance_ids[:config.max_trial_instances]

    instance_pool = [
        {"dataId": config.data.dataset_id, "instanceId": str(instance_id)}
        for instance_id in trial_instance_ids
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
    output_dir.mkdir(parents=True, exist_ok=True)
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

    dataset_config = {
        "dataset_id": config.data.dataset_id,
        "id_map": {"dataId": "dataId", "instanceId": "instanceId"},
    }
    if config.model_name is not None:
        dataset_config["model_type"] = config.model_name

    return TrialGenerationResult(
        config={
            "ivs": config.iv_config,
            "cvs": config.cvs,
            "dataset": dataset_config,
            "sampling": {
                "participants_per_between_condition": config.participants_per_between_condition,
                "trials_per_participant": config.trials_per_participant,
                "trial_randomization_strategy": config.trial_randomization_strategy,
                "instance_wise_explanation": config.instance_wise_explanation,
                "shuffle_instances": config.shuffle_instances,
                "max_trial_instances": config.max_trial_instances,
                "instance_pool_rows": len(instance_pool),
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
    trials: list[dict[str, Any]],
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
    config: dict[str, Any],
    *,
    validate_support: bool = True,
    strict: bool = False,
    show: bool = True,
) -> TrialGenerationResult:
    """Generate trial CSV/JSON/summary artifacts from a UI-exported config."""
    if validate_support:
        validate_ui_config_support(
            config,
            stage="trial_generation",
            strict=strict,
            show=show,
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
    controlled_vars = build_controlled_vars(dataset_cfg.get("model_type"), cvs)

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
    out_dir.mkdir(parents=True, exist_ok=True)
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
        csv_path=str(csv_path),
        json_path=str(json_path),
        summary_path=str(summary_path),
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

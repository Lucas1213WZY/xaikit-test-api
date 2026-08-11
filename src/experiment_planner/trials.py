"""High-level trial generation helpers for experiment designs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Any, Optional, Sequence

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

#: The fixed name a multi-dataset study's between-subjects IV must use.
#: `data_by_dataset` being set is the only signal multi-dataset mode needs;
#: this stays a hardcoded convention rather than a second configurable field,
#: the same way `cognitive_model_id` is a fixed name elsewhere rather than a
#: caller-chosen one.
DATASET_IV_NAME = "dataset"


@dataclass
class TrialBuildConfig:
    """Notebook-facing settings for experimental trial generation."""

    data: Optional[PreparedDataset]
    iv_config: dict[str, dict[str, Any]]
    cvs: dict[str, list[Any]]
    model_name: Optional[str] = None
    participants_per_between_condition: int = 24
    num_training: int = 0
    num_testing: int = 12
    ai_predictions_by_instance: Optional[dict[int, Any]] = None
    counterbalancing_strategy: str = "auto"
    trial_randomization_strategy: str = "balanced"
    instance_wise_explanation: bool = False
    shuffle_instances: bool = True
    max_trial_instances: Optional[int] = DEFAULT_EXPLANATION_INSTANCE_LIMIT
    allowed_instance_ids: Optional[Sequence[int]] = None
    seed: int = 42
    output_dir: Path = Path("experiment_output")
    trials_csv: str = "trials.csv"
    trials_json: str = "trials.json"
    summary_json: str = "design_summary.json"
    #: Multi-dataset alternative to `data`: one PreparedDataset per level of the
    #: `DATASET_IV_NAME` between-subjects IV, keyed by dataset id. Setting this
    #: is itself the signal that this is a multi-dataset study -- each
    #: participant's trials then sample only from their own assigned dataset's
    #: instances instead of one dataset shared by everyone.
    data_by_dataset: Optional[dict[str, PreparedDataset]] = None


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
    data: Optional[PreparedDataset],
    iv_config: dict[str, dict[str, Any]],
    cvs: dict[str, list[Any]],
    *,
    model_name: Optional[str] = None,
    participants_per_between_condition: int = 24,
    num_training: int = 0,
    num_testing: int = 12,
    ai_predictions_by_instance: Optional[dict[int, Any]] = None,
    counterbalancing_strategy: str = "auto",
    trial_randomization_strategy: str = "balanced",
    instance_wise_explanation: bool = False,
    shuffle_instances: bool = True,
    max_trial_instances: Optional[int] = DEFAULT_EXPLANATION_INSTANCE_LIMIT,
    allowed_instance_ids: Optional[Sequence[int]] = None,
    seed: int = 42,
    output_dir: str | Path = "experiment_output",
    trials_csv: str = "trials.csv",
    trials_json: str = "trials.json",
    summary_json: str = "design_summary.json",
    data_by_dataset: Optional[dict[str, PreparedDataset]] = None,
) -> TrialBuildConfig:
    """Collect trial-generation settings in one notebook-friendly config object.

    Args:
        data: The prepared dataset trials are sampled from. Pass ``None`` (and
            use ``data_by_dataset`` instead) for a multi-dataset study.
        iv_config: Independent variables defining the conditions. For a
            multi-dataset study this must declare a between-subjects IV named
            ``DATASET_IV_NAME`` ("dataset") whose levels match
            ``data_by_dataset``'s keys.
        cvs: Control variables recorded on each trial.
        model_name: Name recorded on the trials.
        participants_per_between_condition: Participants per between-subjects cell.
        num_training: Instances shown in the training phase.
        num_testing: Held-out instances shown in the testing phase.
        ai_predictions_by_instance: Predicted label per instance, required for
            prediction-balanced sampling. Not yet supported together with
            ``data_by_dataset``.
        counterbalancing_strategy: How conditions are ordered across
            participants; ``auto`` picks from the design.
        trial_randomization_strategy: How trials are ordered within a participant.
        instance_wise_explanation: Give each instance its own explanation.
        shuffle_instances: Shuffle the instance pool before sampling.
        max_trial_instances: Cap on distinct instances used.
        allowed_instance_ids: Restrict sampling to these instances.
        seed: Seed for sampling and ordering.
        output_dir: Directory the artifacts are written to.
        trials_csv: Filename for the trial table.
        trials_json: Filename for the trial JSON.
        summary_json: Filename for the design summary.
        data_by_dataset: Multi-dataset alternative to ``data`` -- one prepared
            dataset per level of the ``DATASET_IV_NAME`` between-subjects IV,
            keyed by dataset id.

    Returns:
        The assembled config, ready for ``generate_experimental_trials``.
    """
    if num_training < 0:
        raise ValueError("num_training cannot be negative.")
    if num_testing < 1:
        raise ValueError("num_testing must be at least 1.")
    if (data is None) == (data_by_dataset is None):
        raise ValueError("Pass exactly one of data or data_by_dataset.")
    if data_by_dataset is not None and ai_predictions_by_instance is not None:
        raise ValueError(
            "AI-prediction-balanced sampling is not yet supported for "
            "multi-dataset trial generation."
        )
    return TrialBuildConfig(
        data=data,
        iv_config=iv_config,
        cvs=cvs,
        model_name=model_name,
        participants_per_between_condition=participants_per_between_condition,
        num_training=num_training,
        num_testing=num_testing,
        ai_predictions_by_instance=ai_predictions_by_instance,
        counterbalancing_strategy=counterbalancing_strategy,
        trial_randomization_strategy=trial_randomization_strategy,
        instance_wise_explanation=instance_wise_explanation,
        shuffle_instances=shuffle_instances,
        max_trial_instances=max_trial_instances,
        data_by_dataset=data_by_dataset,
        allowed_instance_ids=list(allowed_instance_ids) if allowed_instance_ids is not None else None,
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
    """Build and export experimental trials from a notebook trial config.

    Args:
        config: Settings from ``init_trial_build_config``.
        show: Print a preview of the generated trials.
        preview_rows: Rows to print when ``show`` is True.

    Returns:
        The generated trials and the paths they were written to.
    """
    experiment_structure = inspect_experiment_structure(config.iv_config, show=False)
    within_labels = make_within_condition_order_labels(experiment_structure.block_within_ivs)
    orders, strategy = choose_counterbalancing(
        within_labels,
        strategy=config.counterbalancing_strategy,
    )

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

    multi_dataset = config.data_by_dataset is not None
    if multi_dataset and DATASET_IV_NAME not in experiment_structure.between_ivs:
        raise ValueError(
            f"data_by_dataset was given, but iv_config has no between-subjects "
            f"IV named {DATASET_IV_NAME!r} for it to key off of."
        )

    def _filtered_ids(dataset: PreparedDataset) -> tuple[list[Any], list[Any]]:
        """One dataset's (train_instance_ids, trial_instance_ids), with
        allowed_instance_ids/max_trial_instances applied -- the same per-dataset
        filtering the single-dataset path always did, just callable per level."""
        train_ids = dataset.train_instance_ids
        trial_ids = dataset.test_instance_ids
        if config.allowed_instance_ids is not None:
            # Constrained generation: keep only instances an external asset set can
            # serve, so trials stay runnable against a fixed study corpus.
            allowed = {int(value) for value in config.allowed_instance_ids}
            train_ids = [i for i in train_ids if int(i) in allowed]
            trial_ids = [i for i in trial_ids if int(i) in allowed]
            if not trial_ids:
                raise ValueError(
                    f"No testing instances remain for dataset {dataset.dataset_id!r} "
                    "after applying allowed_instance_ids. The dataset split and the "
                    "allowed set do not overlap."
                )
            if config.num_training > 0 and not train_ids:
                raise ValueError(
                    f"No training instances remain for dataset {dataset.dataset_id!r} "
                    "after applying allowed_instance_ids."
                )
        if config.max_trial_instances is not None:
            trial_ids = trial_ids[:config.max_trial_instances]
        return train_ids, trial_ids

    controlled_vars = build_controlled_vars(config.model_name, config.cvs)

    if multi_dataset:
        train_instance_ids_by_level: dict[str, list[Any]] = {}
        instance_pool_by_level: dict[str, list[dict[str, Any]]] = {}
        for level, dataset in config.data_by_dataset.items():
            train_ids, trial_ids = _filtered_ids(dataset)
            train_instance_ids_by_level[level] = train_ids
            instance_pool_by_level[level] = [
                {"dataId": level, "instanceId": str(instance_id)}
                for instance_id in trial_ids
            ]
        instance_pool = [row for pool in instance_pool_by_level.values() for row in pool]

        trials = build_trial_sequence(
            assignments=assignments,
            instance_pool_by_level=instance_pool_by_level,
            pool_selector_key=DATASET_IV_NAME,
            trials_per_participant=config.num_testing,
            controlled_vars=controlled_vars,
            id_map={"dataId": "dataId", "instanceId": "instanceId"},
            trial_randomized_ivs=experiment_structure.trial_within_ivs or None,
            trial_randomization_strategy=config.trial_randomization_strategy,
            instance_wise_explanation=config.instance_wise_explanation,
            shuffle_instances=config.shuffle_instances,
            seed=config.seed,
        )
        trials = _add_training_and_testing_phases(
            trials,
            train_instance_ids=None,
            train_instance_ids_by_level=train_instance_ids_by_level,
            dataset_id=None,
            num_training=config.num_training,
            condition_columns=[
                *experiment_structure.between_ivs,
                *experiment_structure.block_within_ivs,
            ],
            test_only_columns=list(experiment_structure.trial_within_ivs),
            seed=config.seed,
        )
        # config.ai_predictions_by_instance is guaranteed None here --
        # init_trial_build_config already rejects that combination.
    else:
        train_instance_ids, trial_instance_ids = _filtered_ids(config.data)
        instance_pool = [
            {"dataId": config.data.dataset_id, "instanceId": str(instance_id)}
            for instance_id in trial_instance_ids
        ]

        trials = build_trial_sequence(
            assignments=assignments,
            instance_pool=instance_pool,
            trials_per_participant=config.num_testing,
            controlled_vars=controlled_vars,
            id_map={"dataId": "dataId", "instanceId": "instanceId"},
            trial_randomized_ivs=experiment_structure.trial_within_ivs or None,
            trial_randomization_strategy=config.trial_randomization_strategy,
            instance_wise_explanation=config.instance_wise_explanation,
            shuffle_instances=config.shuffle_instances,
            seed=config.seed,
        )
        trials = _add_training_and_testing_phases(
            trials,
            train_instance_ids=train_instance_ids,
            dataset_id=config.data.dataset_id,
            num_training=config.num_training,
            condition_columns=[
                *experiment_structure.between_ivs,
                *experiment_structure.block_within_ivs,
            ],
            test_only_columns=list(experiment_structure.trial_within_ivs),
            seed=config.seed,
        )
        if config.ai_predictions_by_instance is not None:
            trials = _balance_phase_instances_by_ai_prediction(
                trials,
                train_instance_ids=train_instance_ids,
                test_instance_ids=trial_instance_ids,
                predictions_by_instance=config.ai_predictions_by_instance,
                seed=config.seed,
            )
    trials = _assign_shown_xai_type(trials, seed=config.seed)

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
        counterbalancing_strategy=config.counterbalancing_strategy,
        trial_randomization_strategy=config.trial_randomization_strategy,
        trials_per_participant=config.num_testing,
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
        "dataset_id": (
            config.data.dataset_id
            if config.data is not None
            else list(config.data_by_dataset)
        ),
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
                "num_training": config.num_training,
                "num_testing": config.num_testing,
                "balanced_by_ai_prediction": (
                    config.ai_predictions_by_instance is not None
                ),
                "counterbalancing_strategy": config.counterbalancing_strategy,
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


def _add_training_and_testing_phases(
    trials: list[dict[str, Any]],
    *,
    train_instance_ids: Optional[Any] = None,
    dataset_id: Optional[str] = None,
    train_instance_ids_by_level: Optional[dict[str, Any]] = None,
    num_training: int,
    seed: int,
    condition_columns: Optional[list[str]] = None,
    test_only_columns: Optional[list[str]] = None,
) -> list[dict[str, Any]]:
    """Place all training rows before all randomized testing rows per participant.

    Pass either ``train_instance_ids``/``dataset_id`` (single-dataset: every
    participant draws training instances from the same pool) or
    ``train_instance_ids_by_level`` (multi-dataset: each participant draws from
    their own assigned dataset's pool, found via the ``dataId`` their trial rows
    already carry from the per-level instance pool built upstream).
    """
    if num_training < 0:
        raise ValueError("num_training cannot be negative.")
    if (train_instance_ids is None) == (train_instance_ids_by_level is None):
        raise ValueError(
            "Pass exactly one of train_instance_ids or train_instance_ids_by_level."
        )

    participant_ids = list(dict.fromkeys(trial["participantId"] for trial in trials))
    condition_columns = [
        column for column in (condition_columns or [])
        if any(column in trial for trial in trials)
    ]
    test_only_columns = list(test_only_columns or [])
    rng = random.Random(seed)
    phased_trials: list[dict[str, Any]] = []
    for participant_id in participant_ids:
        participant_trials = [
            dict(trial) for trial in trials if trial["participantId"] == participant_id
        ]
        if not participant_trials:
            continue

        if train_instance_ids_by_level is not None:
            participant_data_id = participant_trials[0].get("dataId", dataset_id)
            available_ids = list(train_instance_ids_by_level[participant_data_id])
        else:
            participant_data_id = dataset_id
            available_ids = list(train_instance_ids)
        if num_training > len(available_ids):
            raise ValueError(
                f"Requested {num_training} cognitive training instances, "
                f"but participant {participant_id}'s training split "
                f"(dataset {participant_data_id!r}) contains only "
                f"{len(available_ids)}."
            )

        condition_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
        for trial in participant_trials:
            condition_key = tuple(trial.get(column) for column in condition_columns)
            condition_groups.setdefault(condition_key, []).append(trial)

        if num_training and num_training < len(condition_groups):
            raise ValueError(
                "num_training must provide at least one training instance for "
                f"each of the participant's {len(condition_groups)} conditions."
            )
        base_training, extra_training = divmod(
            num_training,
            max(1, len(condition_groups)),
        )
        participant_training_trials: list[dict[str, Any]] = []
        for condition_position, condition_trials in enumerate(condition_groups.values()):
            condition_training_count = base_training + (
                1 if condition_position < extra_training else 0
            )
            condition_training_ids = available_ids.copy()
            rng.shuffle(condition_training_ids)
            template = condition_trials[0]
            for phase_position, instance_id in enumerate(
                condition_training_ids[:condition_training_count],
                start=1,
            ):
                training_trial = dict(template)
                for column in test_only_columns:
                    training_trial.pop(column, None)
                training_trial.update({
                    "trialWithinBlock": phase_position,
                    "dataId": participant_data_id,
                    "instanceId": str(instance_id),
                    "phase": "training",
                    "phaseTrialId": len(participant_training_trials) + 1,
                })
                participant_training_trials.append(training_trial)

        participant_testing_trials: list[dict[str, Any]] = []
        for phase_position, trial in enumerate(participant_trials, start=1):
            trial.update({
                "phase": "testing",
                "phaseTrialId": phase_position,
            })
            participant_testing_trials.append(trial)

        participant_sequence = [
            *participant_training_trials,
            *participant_testing_trials,
        ]
        for trial_id, trial in enumerate(participant_sequence, start=1):
            trial["trialId"] = trial_id
            phased_trials.append(trial)

    return phased_trials


#: Condition levels that show more than one explanation family, mapped to the
#: families a trial in that condition alternates between. A condition naming a
#: single family (``decision_tree``, ``logistic_regression``) needs no entry --
#: every one of its trials shows that family.
MULTI_FAMILY_XAI_TYPES: dict[str, tuple[str, ...]] = {
    "hybrid": ("decision_tree", "logistic_regression"),
}


def _assign_shown_xai_type(
    trials: list[dict[str, Any]],
    *,
    seed: int,
) -> list[dict[str, Any]]:
    """Record which explanation family each trial actually shows.

    For a single-family condition (``xai_type='decision_tree'``) this just
    echoes the condition. For a multi-family condition (``xai_type='hybrid'``,
    which shows both) the family alternates randomly per trial, balanced as
    evenly as the trial count allows and shuffled with the study's own seed, so
    the assignment is reproducible and auditable from the exported trial table
    rather than being re-drawn at simulation time.

    Alternation is assigned independently per participant and phase, so a
    participant's training trials and testing trials are each balanced in their
    own right.
    """
    if not any("xai_type" in trial for trial in trials):
        return trials

    rng = random.Random(seed)
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for trial in trials:
        xai_type = str(trial.get("xai_type", "")).strip().lower()
        families = MULTI_FAMILY_XAI_TYPES.get(xai_type)
        if families is None:
            # Single-family (or non-CoXAM) condition: the shown family is the
            # condition itself, with no alternation to schedule.
            if "xai_type" in trial:
                trial["shown_xai_type"] = trial["xai_type"]
            continue
        key = (trial.get("participantId"), trial.get("phase"), xai_type)
        groups.setdefault(key, []).append(trial)

    for (_participant_id, _phase, xai_type), group_trials in groups.items():
        families = MULTI_FAMILY_XAI_TYPES[xai_type]
        repeats, remainder = divmod(len(group_trials), len(families))
        schedule = list(families) * repeats + list(families[:remainder])
        rng.shuffle(schedule)
        for trial, family in zip(group_trials, schedule):
            trial["shown_xai_type"] = family

    return trials


def _balance_phase_instances_by_ai_prediction(
    trials: list[dict[str, Any]],
    *,
    train_instance_ids: Any,
    test_instance_ids: Any,
    predictions_by_instance: dict[int, Any],
    seed: int,
) -> list[dict[str, Any]]:
    """Assign a randomized half-and-half prediction mix within each phase."""
    balanced_trials = [dict(trial) for trial in trials]
    rng = random.Random(seed)
    participant_ids = list(dict.fromkeys(
        trial["participantId"] for trial in balanced_trials
    ))

    phase_pools = {
        "training": [int(value) for value in train_instance_ids],
        "testing": [int(value) for value in test_instance_ids],
    }
    all_labels = list(dict.fromkeys(
        predictions_by_instance[int(instance_id)]
        for instance_ids in phase_pools.values()
        for instance_id in instance_ids
        if int(instance_id) in predictions_by_instance
    ))
    if len(all_labels) != 2:
        raise ValueError(
            "AI-prediction-balanced sampling requires exactly two predicted "
            f"classes; found {all_labels}."
        )

    pools_by_phase: dict[str, dict[Any, list[int]]] = {}
    for phase, instance_ids in phase_pools.items():
        by_label = {label: [] for label in all_labels}
        missing = []
        for instance_id in instance_ids:
            if instance_id not in predictions_by_instance:
                missing.append(instance_id)
                continue
            by_label[predictions_by_instance[instance_id]].append(instance_id)
        if missing:
            raise ValueError(
                f"Missing AI predictions for {phase} instance IDs: {missing[:10]}."
            )
        empty_labels = [label for label, ids in by_label.items() if not ids]
        if empty_labels:
            raise ValueError(
                f"The {phase} pool has no instances for predicted class(es) "
                f"{empty_labels}."
            )
        pools_by_phase[phase] = by_label

    for participant_position, participant_id in enumerate(participant_ids):
        for phase in ("training", "testing"):
            row_positions = [
                index
                for index, trial in enumerate(balanced_trials)
                if trial["participantId"] == participant_id
                and str(trial.get("phase", "testing")).lower() == phase
            ]
            if not row_positions:
                continue

            half = len(row_positions) // 2
            class_counts = [half, half]
            if len(row_positions) % 2:
                class_counts[participant_position % 2] += 1
            desired_labels = [
                label
                for label, count in zip(all_labels, class_counts)
                for _ in range(count)
            ]
            rng.shuffle(desired_labels)

            required_by_label = {
                label: desired_labels.count(label) for label in all_labels
            }
            selected_by_label: dict[Any, list[int]] = {}
            for label in all_labels:
                available = pools_by_phase[phase][label]
                required = required_by_label[label]
                if required > len(available):
                    raise ValueError(
                        f"Not enough {phase} instances predicted as {label!r}: "
                        f"need {required}, found {len(available)}."
                    )
                selected_by_label[label] = rng.sample(available, required)

            label_offsets = {label: 0 for label in all_labels}
            for row_position, label in zip(row_positions, desired_labels):
                offset = label_offsets[label]
                instance_id = selected_by_label[label][offset]
                label_offsets[label] += 1
                balanced_trials[row_position]["instanceId"] = str(instance_id)
                balanced_trials[row_position]["ai_prediction"] = label

    return balanced_trials


def preview_trial_rows(
    trials: list[dict[str, Any]],
    experiment_structure: DesignRoles,
    *,
    n: int = 10,
) -> None:
    """Print a compact preview of trial assignment, condition, and instance ids.

    Args:
        trials: The generated trial rows.
        experiment_structure: IVs split by design role, used for the headings.
        n: How many rows to print.
    """
    key_cols = [
        "participantId",
        "trialId",
        "phase",
        "phaseTrialId",
        "block",
        "trialWithinBlock",
        "withinCondition",
        *experiment_structure.between_ivs.keys(),
        *experiment_structure.block_within_ivs.keys(),
        *experiment_structure.trial_within_ivs.keys(),
        "dataId",
        "instanceId",
        "ai_prediction",
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
    """Generate trial CSV/JSON/summary artifacts from a UI-exported config.

    Args:
        config: The UI-exported configuration dict.
        validate_support: Check the config against the support standard first.
        strict: Treat support warnings as failures.
        show: Print the validation report and trial preview.

    Returns:
        The generated trials and the paths they were written to.
    """
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
    counterbalancing_strategy = sampling_cfg.get("counterbalancing_strategy", "auto")
    orders, strategy = choose_counterbalancing(
        within_labels,
        strategy=counterbalancing_strategy,
    )

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
        counterbalancing_strategy=counterbalancing_strategy,
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
    """Load a UI-exported JSON file and generate trial artifacts from it.

    Args:
        config_path: Path to the exported JSON.
        show: Print the validation report and trial preview.

    Returns:
        The generated trials, or None if the config could not be used.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        print(f"UI config not found: {config_path}")
        return None

    config = load_json_config(config_path)
    return generate_trials_from_ui_config(config, show=show)

"""
Experimental Design: Counterbalancing, Factorial Arrangements, and Trial Sequencing
------------------------------------------------------------------------------------
Algorithm sources:
  - Bradley (1958) JASA — Balanced Latin Square construction (even/odd n)
  - valentin-schwind/balanced-latinsquare-generator — Bradley reference impl
  - clicumu/pyDOE2 fullfact() — full factorial Cartesian product logic
  - Python itertools.permutations — complete counterbalancing for n <= 4
  - NYUCCL/counterbalancing — round-robin participant-to-order assignment
  - PsychoPy data.TrialHandler — flat trial-row dataframe structure
"""

from __future__ import annotations

import csv
import json
import random
from collections import Counter
from itertools import permutations, product
from pathlib import Path
from typing import Any

import numpy as np


# ── 1. FACTORIAL CONDITIONS ────────────────────────────────────────────────────

def factorial_conditions(iv_config: dict[str, list]) -> list[dict]:
    """
    Generate ALL factorial combinations from independent variable levels.

    Source: pyDOE2 fullfact() — Cartesian product of all IV levels.

    Args:
        iv_config: {'IV_name': [level1, level2, ...], ...}

    Returns:
        List of dicts, one per condition combination.

    Example:
        factorial_conditions({'xai_method': ['shap','lime'], 'display': ['bar','heatmap']})
        → [{'xai_method':'shap','display':'bar'}, {'xai_method':'shap','display':'heatmap'}, ...]
    """
    names = list(iv_config.keys())
    levels = [iv_config[n] for n in names]
    return [dict(zip(names, combo)) for combo in product(*levels)]


def split_ivs_by_design_role(iv_config: dict[str, dict]) -> tuple[dict, dict, dict]:
    """
    Split IV config into between-subjects, block-counterbalanced within-subjects,
    and trial-randomized within-subjects IV dictionaries.

    Within-subjects IVs default to block-level counterbalancing for backwards
    compatibility. Add {'randomization': 'trial'} to randomize a within IV at the
    trial level instead.

    Args:
        iv_config: {
            'xai_method': {'type': 'within', 'randomization': 'block', 'levels': [...]},
            'tested_w_xai': {'type': 'within', 'randomization': 'trial', 'levels': [...]},
            'display': {'type': 'between', 'levels': [...]},
        }

    Returns:
        (between_ivs, block_within_ivs, trial_within_ivs), each as
        {'IV_name': [levels...]}.
    """
    between_ivs = {}
    block_within_ivs = {}
    trial_within_ivs = {}

    for name, spec in iv_config.items():
        iv_type = spec["type"]
        levels = spec["levels"]

        if iv_type == "between":
            between_ivs[name] = levels
        elif iv_type == "within":
            randomization = spec.get("randomization", "block")
            if randomization == "block":
                block_within_ivs[name] = levels
            elif randomization == "trial":
                trial_within_ivs[name] = levels
            else:
                raise ValueError(
                    f"Unsupported randomization for IV '{name}': {randomization!r}. "
                    "Use 'block' or 'trial'."
                )
        else:
            raise ValueError(
                f"Unsupported IV type for '{name}': {iv_type!r}. "
                "Use 'within' or 'between'."
            )

    return between_ivs, block_within_ivs, trial_within_ivs


def make_within_condition_order_labels(within_ivs: dict[str, list]) -> list:
    """
    Build the condition list sent into counterbalancing.

    Returns dict conditions instead of flattened strings so build_trial_sequence()
    can write named IV columns such as xai_method='shap'.
    """
    if not within_ivs:
        return [{"withinCondition": "single_condition"}]
    return factorial_conditions(within_ivs)


# ── 2. COUNTERBALANCING STRATEGIES ────────────────────────────────────────────

def complete_counterbalancing(conditions: list) -> list[list]:
    """
    Return ALL n! orderings of conditions.

    Source: Python itertools.permutations.
    Use only when len(conditions) <= 4 (4! = 24 orders).

    Args:
        conditions: list of condition labels or dicts.

    Returns:
        List of all permutation orderings.
    """
    if len(conditions) > 4:
        raise ValueError(
            f"Complete counterbalancing requires n <= 4 (got {len(conditions)}). "
            "Use balanced_latin_square() instead."
        )
    return [list(p) for p in permutations(conditions)]


def balanced_latin_square(conditions: list) -> list[list]:
    """
    Generate a Balanced Latin Square for n conditions.

    Source: Bradley (1958) JASA; valentin-schwind/balanced-latinsquare-generator.
    Each condition appears in each position exactly once.
    Each condition immediately precedes every other condition exactly once
    (twice if n is odd), eliminating immediate carryover effects.

    Args:
        conditions: list of condition labels (any type).

    Returns:
        List of rows, each row is one participant's ordering of conditions.
        Number of rows = n (even n) or 2n (odd n).
    """
    n = len(conditions)
    square_indices = []

    for start in range(n):
        row = [None] * n
        left = 0
        right = 0
        for k in range(n):
            if k % 2 == 0:
                row[k] = (start + left) % n
                left += 1
            else:
                row[k] = (start - right - 1) % n
                right += 1
        square_indices.append(row)

    # For odd n, append reversed rows to balance carryover in both directions
    if n % 2 == 1:
        square_indices += [row[::-1] for row in square_indices]

    return [[conditions[i] for i in row] for row in square_indices]


def choose_counterbalancing(
    within_conditions: list,
    strategy: str = "auto",
) -> tuple[list[list], str]:
    """
    Select a counterbalancing strategy for within-subjects conditions.

    Strategies:
        auto                  n <= 4 uses complete counterbalancing; otherwise
                              Bradley balanced Latin square.
        complete              all n! orderings; requires n <= 4.
        balanced_latin_square Bradley (1958) balanced Latin square.

    Args:
        within_conditions: list of condition labels for within-subjects IV.
        strategy: one of 'auto', 'complete', or 'balanced_latin_square'.

    Returns:
        (orders, strategy_name)
    """
    n = len(within_conditions)
    strategy = strategy.lower()
    aliases = {
        "auto": "auto",
        "complete": "complete",
        "complete_counterbalancing": "complete",
        "balanced_latin_square": "balanced_latin_square",
        "balanced_latin_square_bradley_1958": "balanced_latin_square",
        "bradley": "balanced_latin_square",
        "bradley_1958": "balanced_latin_square",
    }
    if strategy not in aliases:
        raise ValueError(
            "counterbalancing_strategy must be one of 'auto', 'complete', "
            f"or 'balanced_latin_square' (got {strategy!r})."
        )

    resolved_strategy = aliases[strategy]
    if resolved_strategy == "auto":
        resolved_strategy = "complete" if n <= 4 else "balanced_latin_square"

    if resolved_strategy == "complete":
        return complete_counterbalancing(within_conditions), "complete_counterbalancing"
    return balanced_latin_square(within_conditions), "balanced_latin_square_bradley_1958"


def _condition_key(condition: Any) -> str:
    """Stable condition key for diagnostics and JSON summaries."""
    return json.dumps(condition, sort_keys=True, default=str)


def counterbalancing_diagnostics(orders: list[list]) -> dict[str, Any]:
    """Return position and immediate-pair balance diagnostics for condition orders."""
    if not orders:
        return {
            "n_conditions": 0,
            "n_orders": 0,
            "position_counts": {},
            "immediate_pair_counts": {},
            "position_balanced": True,
            "immediate_pair_balanced": True,
        }

    n_conditions = len(orders[0])
    condition_keys = sorted({_condition_key(condition) for row in orders for condition in row})
    position_counts = {
        str(position + 1): dict(
            Counter(_condition_key(row[position]) for row in orders)
        )
        for position in range(n_conditions)
    }
    pair_counts = Counter(
        f"{_condition_key(row[position])} -> {_condition_key(row[position + 1])}"
        for row in orders
        for position in range(max(0, n_conditions - 1))
    )

    expected_position_count = (
        len(orders) / n_conditions
        if n_conditions
        else 0
    )
    n_ordered_pairs = n_conditions * (n_conditions - 1)
    expected_immediate_pair_count = (
        sum(pair_counts.values()) / n_ordered_pairs
        if n_ordered_pairs
        else 0
    )

    position_balanced = all(
        counts.get(condition, 0) == expected_position_count
        for counts in position_counts.values()
        for condition in condition_keys
    )
    immediate_pair_balanced = all(
        pair_counts.get(f"{left} -> {right}", 0) == expected_immediate_pair_count
        for left in condition_keys
        for right in condition_keys
        if left != right
    )

    return {
        "n_conditions": n_conditions,
        "n_orders": len(orders),
        "expected_position_count": expected_position_count,
        "expected_immediate_pair_count": expected_immediate_pair_count,
        "position_balanced": position_balanced,
        "immediate_pair_balanced": immediate_pair_balanced,
        "position_counts": position_counts,
        "immediate_pair_counts": dict(sorted(pair_counts.items())),
    }


def to_psychopy_trial_list(order: list) -> list[dict[str, Any]]:
    """
    Convert one generated order into a PsychoPy-compatible trialList.

    The generated list can be passed to psychopy.data.TrialHandler with
    method='sequential'. PsychoPy stays optional; the Bradley order generation
    remains local and testable.
    """
    trial_list = []
    for condition in order:
        if isinstance(condition, dict):
            trial_list.append(condition.copy())
        else:
            trial_list.append({"withinCondition": condition})
    return trial_list


def make_psychopy_trial_handler(order: list, *, n_reps: int = 1, seed: int | None = None):
    """Create a PsychoPy TrialHandler for an already-counterbalanced order."""
    try:
        from psychopy import data
    except ImportError as exc:
        raise ImportError(
            "PsychoPy is not installed. Install it to use "
            "make_psychopy_trial_handler(), or use to_psychopy_trial_list() "
            "without PsychoPy."
        ) from exc

    return data.TrialHandler(
        trialList=to_psychopy_trial_list(order),
        nReps=n_reps,
        method="sequential",
        seed=seed,
    )


# ── 3. PARTICIPANT ASSIGNMENT ──────────────────────────────────────────────────

def assign_participants(
    n_participants: int,
    counterbalance_orders: list[list],
    between_iv_groups: dict[str, list] | None = None,
) -> list[dict]:
    """
    Assign each participant to a counterbalancing order and between-subjects group.

    Source: NYUCCL/counterbalancing — round-robin assignment.

    Args:
        n_participants:          Total number of participants.
        counterbalance_orders:   List of within-subjects condition orderings.
        between_iv_groups:       {'IV_name': [level1, level2, ...]} for between IVs.
                                 Participants are round-robin assigned across all
                                 between-subjects factorial combinations.

    Returns:
        List of participant assignment dicts.
    """
    n_orders = len(counterbalance_orders)

    # Generate all between-subjects factorial combinations
    if between_iv_groups:
        b_names = list(between_iv_groups.keys())
        b_levels = [between_iv_groups[k] for k in b_names]
        between_combos = [dict(zip(b_names, combo)) for combo in product(*b_levels)]
    else:
        between_combos = [{}]

    n_between = len(between_combos)
    assignments = []

    for p_idx in range(n_participants):
        participant_id = p_idx + 1
        order = counterbalance_orders[p_idx % n_orders]
        between_group = between_combos[p_idx % n_between]

        assignments.append({
            "participantId": participant_id,
            "within_order": order,
            **between_group,
        })

    return assignments


# ── 4. TRIAL SEQUENCING ────────────────────────────────────────────────────────

def build_trial_sequence(
    assignments: list[dict],
    instance_pool: list[dict],
    trials_per_condition: int | None = None,
    trials_per_participant: int | None = None,
    controlled_vars: dict[str, Any] | None = None,
    id_map: dict[str, str] | None = None,
    trial_randomized_ivs: dict[str, list] | None = None,
    trial_randomization_strategy: str = "balanced",
    instance_wise_explanation: bool = False,
    include_instance_fields: bool | None = None,
    shuffle_instances: bool = True,
    seed: int | None = None,
) -> list[dict]:
    """
    Build a flat trial-level sequence for all participants.

    Source: PsychoPy data.TrialHandler flat-row structure.

    Each trial row contains:
        participantId, trialId (within participant), trialWithinBlock,
        block (condition position index), withinCondition,
        between-subjects IV cols, instanceId, dataId,
        all controlled_vars (CV1, CV2, ...).

    Block-level within conditions can be strings (legacy behavior) or dicts.
    Dict conditions are expanded to named IV columns in every row.

    Trial-level randomized within IVs are supplied via trial_randomized_ivs.
    Prefer trials_per_participant when trial-level randomization is used. With
    trial_randomization_strategy='balanced', the participant-level total is split
    evenly across all block x trial-level condition cells. With 'random', the
    participant-level total is split as evenly as possible across blocks and each
    trial samples one trial-level combo randomly.

    Args:
        assignments:          Output of assign_participants().
        instance_pool:        List of dicts from your explanation CSV.
        trials_per_condition: Legacy mode. Number of data instances shown per
                              block x trial-level condition cell.
        trials_per_participant:
                              Preferred mode. Total number of trials generated
                              for each participant.
        controlled_vars:      Fixed metadata cols added to every trial
                              e.g. {'modelName': 'mlp', 'dataset': 'wine_quality'}.
        id_map:               Optional dict that maps trial ID fields to column
                              names in instance_pool dicts, e.g.:
                                {'dataId': 'dataId', 'instanceId': 'instanceId'}
                              If a key is missing from the map or not found in the
                              instance dict, IDs are auto-generated sequentially.
                              Pass None to use auto-generated IDs for both.
        trial_randomized_ivs:  {'IV_name': [levels...]} for within-subjects IVs
                              randomized at trial level rather than block level.
        trial_randomization_strategy:
                              'balanced' (default) or 'random'.
        instance_wise_explanation:
                              If True, copy remaining instance_pool explanation
                              columns such as pred, expMethod, i_max, a0_i, ...
                              into every trial row. Default False keeps trial rows
                              lightweight with only IDs and design metadata.
        include_instance_fields:
                              Deprecated alias for instance_wise_explanation.
        shuffle_instances:    Randomly shuffle instance pool per participant-block.
        seed:                 Random seed for reproducibility.

    Returns:
        List of trial dicts (one per row in the export CSV/JSON).
    """
    rng = random.Random(seed)
    all_trials = []
    auto_data_id = 1
    auto_instance_id = 1

    # Resolve ID column names from map, or fall back to defaults then auto
    data_id_col     = (id_map or {}).get("dataId",     "dataId")
    instance_id_col = (id_map or {}).get("instanceId", "instanceId")
    trial_condition_combos = (
        factorial_conditions(trial_randomized_ivs)
        if trial_randomized_ivs
        else [{}]
    )

    if trial_randomization_strategy not in {"balanced", "random"}:
        raise ValueError(
            "trial_randomization_strategy must be 'balanced' or 'random' "
            f"(got {trial_randomization_strategy!r})."
        )
    if include_instance_fields is not None:
        instance_wise_explanation = include_instance_fields
    if trials_per_condition is None and trials_per_participant is None:
        raise ValueError(
            "Pass either trials_per_participant or trials_per_condition."
        )

    for assignment in assignments:
        p_id = assignment["participantId"]
        participant_trial_id = 1
        within_order = assignment["within_order"]
        between_cols = {k: v for k, v in assignment.items()
                        if k not in ("participantId", "within_order")}
        n_blocks = len(within_order)

        if trials_per_participant is not None:
            if trial_randomization_strategy == "balanced":
                n_cells = n_blocks * len(trial_condition_combos)
                if trials_per_participant % n_cells != 0:
                    raise ValueError(
                        "trials_per_participant must divide evenly across "
                        "block x trial-level condition cells for balanced "
                        f"randomization. Got {trials_per_participant} trials "
                        f"for {n_cells} cells."
                    )
                repeats_per_cell = trials_per_participant // n_cells
                block_trial_counts = [
                    repeats_per_cell * len(trial_condition_combos)
                    for _ in within_order
                ]
            else:
                base_trials = trials_per_participant // n_blocks
                remainder = trials_per_participant % n_blocks
                block_trial_counts = [
                    base_trials + (1 if i < remainder else 0)
                    for i in range(n_blocks)
                ]
                rng.shuffle(block_trial_counts)
        else:
            repeats_per_cell = trials_per_condition
            block_trial_counts = [
                len(trial_condition_combos) * trials_per_condition
                if trial_randomization_strategy == "balanced"
                else trials_per_condition
                for _ in within_order
            ]

        for block_idx, condition in enumerate(within_order):
            if isinstance(condition, dict):
                block_condition_cols = {
                    k: v for k, v in condition.items()
                    if k != "withinCondition"
                }
                within_condition_label = "_".join(
                    str(v) for k, v in condition.items()
                    if k != "withinCondition"
                ) or str(condition.get("withinCondition", "single_condition"))
            else:
                block_condition_cols = {}
                within_condition_label = str(condition)

            if trial_randomization_strategy == "balanced":
                if trials_per_participant is not None:
                    repeats = repeats_per_cell
                else:
                    repeats = trials_per_condition
                trial_conditions = [
                    combo.copy()
                    for combo in trial_condition_combos
                    for _ in range(repeats)
                ]
                rng.shuffle(trial_conditions)
            else:
                trial_conditions = [
                    rng.choice(trial_condition_combos).copy()
                    for _ in range(block_trial_counts[block_idx])
                ]

            # Sample instances for this block
            available = instance_pool.copy()
            if shuffle_instances:
                rng.shuffle(available)
            if len(available) < len(trial_conditions):
                raise ValueError(
                    f"Not enough instances for participant {p_id}, block {block_idx + 1}: "
                    f"need {len(trial_conditions)}, got {len(available)}."
                )
            block_instances = available[:len(trial_conditions)]

            for trial_within_block, (instance, trial_condition_cols) in enumerate(
                zip(block_instances, trial_conditions),
                start=1,
            ):
                # dataId: from id_map column → else auto-increment
                if data_id_col in instance:
                    data_id = instance[data_id_col]
                else:
                    data_id = auto_data_id
                    auto_data_id += 1

                # instanceId: from id_map column → else auto-increment
                if instance_id_col in instance:
                    instance_id = instance[instance_id_col]
                else:
                    instance_id = auto_instance_id
                    auto_instance_id += 1

                trial_row = {
                    "participantId":      p_id,
                    "trialId":            participant_trial_id,
                    "block":              block_idx + 1,
                    "trialWithinBlock":   trial_within_block,
                    "withinCondition":    within_condition_label,
                    **block_condition_cols,
                    **trial_condition_cols,
                    **between_cols,
                    "dataId":             data_id,
                    "instanceId":         instance_id,
                }

                if instance_wise_explanation:
                    # Append remaining instance-level fields (e.g. pred, expMethod).
                    # Exclude both the mapped source columns and the canonical output
                    # keys ("dataId", "instanceId") so auto-generated values are not
                    # overwritten by a same-named field from the instance dict.
                    skip = {data_id_col, instance_id_col, "dataId", "instanceId"}
                    for k, v in instance.items():
                        if k not in skip:
                            trial_row[k] = v

                # Append controlled variables (CV1, CV2, ...)
                if controlled_vars:
                    trial_row.update(controlled_vars)

                all_trials.append(trial_row)
                participant_trial_id += 1

    return all_trials


# ── 5. EXPORT ──────────────────────────────────────────────────────────────────

def export_trials_csv(trials: list[dict], path: str | Path) -> Path:
    """Export trial list to CSV. Returns the written path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not trials:
        raise ValueError("Trial list is empty — nothing to export.")
    fieldnames = list(dict.fromkeys(
        field
        for trial in trials
        for field in trial
    ))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(trials)
    return path


def export_trials_json(trials: list[dict], path: str | Path) -> Path:
    """Export trial list to JSON. Returns the written path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            trials,
            f,
            indent=2,
            ensure_ascii=False,
            default=_json_default,
        )
    return path


def export_design_summary(
    iv_config: dict,
    between_ivs: dict,
    within_ivs: dict,
    strategy: str,
    orders: list[list],
    assignments: list[dict],
    path: str | Path,
    block_within_ivs: dict | None = None,
    trial_within_ivs: dict | None = None,
    counterbalancing_strategy: str | None = None,
    trial_randomization_strategy: str | None = None,
    trials_per_condition: int | None = None,
    trials_per_participant: int | None = None,
) -> Path:
    """Export a human-readable design summary JSON alongside the trial CSV."""
    path = Path(path)
    diagnostics = counterbalancing_diagnostics(orders)
    summary = {
        "iv_config": iv_config,
        "between_subjects_ivs": between_ivs,
        "within_subjects_ivs": within_ivs,
        "block_counterbalanced_within_ivs": (
            block_within_ivs if block_within_ivs is not None else within_ivs
        ),
        "trial_randomized_within_ivs": trial_within_ivs or {},
        "requested_counterbalancing_strategy": counterbalancing_strategy,
        "trial_randomization_strategy": trial_randomization_strategy,
        "trials_per_condition": trials_per_condition,
        "trials_per_participant": trials_per_participant,
        "counterbalancing_strategy": strategy,
        "n_conditions": diagnostics["n_conditions"],
        "n_orders": diagnostics["n_orders"],
        "counterbalancing_diagnostics": diagnostics,
        "orders": [
            {"order_id": i + 1, "sequence": o} for i, o in enumerate(orders)
        ],
        "n_participants": len(assignments),
        "participant_assignments": [
            {
                "participantId": a["participantId"],
                "within_order": a["within_order"],
                **{k: v for k, v in a.items()
                   if k not in ("participantId", "within_order")},
            }
            for a in assignments
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=_json_default)
    return path


def _json_default(value: Any) -> Any:
    """Convert common scientific-Python values for artifact JSON export."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(
        f"Object of type {value.__class__.__name__} is not JSON serializable"
    )


# ── 6. HIGH-LEVEL PIPELINE ────────────────────────────────────────────────────

def build_experiment_plan(
    iv_config: dict[str, dict],
    instance_pool: list[dict],
    n_participants: int,
    trials_per_condition: int | None = None,
    trials_per_participant: int | None = None,
    controlled_vars: dict[str, Any] | None = None,
    id_map: dict[str, str] | None = None,
    output_dir: str | Path = "experiment_output",
    counterbalancing_strategy: str = "auto",
    trial_randomization_strategy: str = "balanced",
    instance_wise_explanation: bool = False,
    shuffle_instances: bool = True,
    seed: int | None = 42,
) -> dict:
    """
    Full pipeline: classify IVs → factorial conditions → counterbalance →
    assign participants → build trials → export CSV + JSON + summary.

    Args:
        iv_config: {
            'xai_method': {'type': 'within', 'levels': ['shap','lime','attention']},
            'display':    {'type': 'between', 'levels': ['bar','heatmap']},
        }
        instance_pool:         List of dicts from your explanation CSV.
        n_participants:        Total participants.
        trials_per_condition:  Legacy mode. Data instances shown per block x
                               trial-level condition cell.
        trials_per_participant:
                               Preferred mode. Total trials generated for each
                               participant.
        controlled_vars:       Fixed cols added to every trial row
                               e.g. {'CV1': 'mlp', 'CV2': 'wine_quality'}.
        id_map:                Maps 'dataId'/'instanceId' to column names in
                               instance_pool dicts. Auto-generates sequential IDs
                               for any key not found in the instance dict. Examples:
                                 {'dataId': 'dataId', 'instanceId': 'instanceId'}
                                 None → both IDs are auto-generated (1, 2, 3 ...).
        output_dir:            Folder for output files.
        counterbalancing_strategy:
                               'auto', 'complete', or 'balanced_latin_square'
                               for block-level within-subject conditions.
        trial_randomization_strategy:
                               For within IVs with {'randomization': 'trial'}:
                               'balanced' splits trials_per_participant evenly
                               across condition cells; 'random' samples one
                               trial-level combo for each trial.
        instance_wise_explanation:
                               If True, copy extra instance/explanation columns
                               from instance_pool into trial rows. Default False
                               exports only IDs and design metadata.
        shuffle_instances:     Shuffle instance pool per participant/block.
        seed:                  Random seed.

    Returns:
        dict with keys: 'trials', 'assignments', 'orders', 'strategy',
                        'csv_path', 'json_path', 'summary_path'.
    """
    output_dir = Path(output_dir)

    # Classify IVs
    between_ivs, block_within_ivs, trial_within_ivs = split_ivs_by_design_role(iv_config)
    within_ivs = {**block_within_ivs, **trial_within_ivs}

    # All factorial conditions (for documentation)
    all_conditions = factorial_conditions({k: v["levels"] for k, v in iv_config.items()})

    # Only block-level within IVs are sent into counterbalancing.
    within_labels = make_within_condition_order_labels(block_within_ivs)

    orders, strategy = choose_counterbalancing(
        within_labels,
        strategy=counterbalancing_strategy,
    )

    # Assign participants
    assignments = assign_participants(n_participants, orders, between_ivs or None)

    # Build trials
    trials = build_trial_sequence(
        assignments=assignments,
        instance_pool=instance_pool,
        trials_per_condition=trials_per_condition,
        trials_per_participant=trials_per_participant,
        controlled_vars=controlled_vars,
        id_map=id_map,
        trial_randomized_ivs=trial_within_ivs or None,
        trial_randomization_strategy=trial_randomization_strategy,
        instance_wise_explanation=instance_wise_explanation,
        shuffle_instances=shuffle_instances,
        seed=seed,
    )

    # Export
    csv_path     = export_trials_csv(trials,    output_dir / "trials.csv")
    json_path    = export_trials_json(trials,   output_dir / "trials.json")
    summary_path = export_design_summary(
        iv_config={k: v["levels"] for k, v in iv_config.items()},
        between_ivs=between_ivs,
        within_ivs=within_ivs,
        strategy=strategy,
        orders=orders,
        assignments=assignments,
        path=output_dir / "design_summary.json",
        block_within_ivs=block_within_ivs,
        trial_within_ivs=trial_within_ivs,
        counterbalancing_strategy=counterbalancing_strategy,
        trial_randomization_strategy=trial_randomization_strategy,
        trials_per_condition=trials_per_condition,
        trials_per_participant=trials_per_participant,
    )

    print(
        "IV classification:  "
        f"between={list(between_ivs)}, "
        f"within_block={list(block_within_ivs)}, "
        f"within_trial={list(trial_within_ivs)}"
    )
    print(f"Factorial conditions: {len(all_conditions)} total")
    print(f"Counterbalancing: {strategy} → {len(orders)} orders")
    trial_condition_count = (
        len(factorial_conditions(trial_within_ivs))
        if trial_within_ivs
        else 1
    )
    if trials_per_participant is not None:
        trials_each = trials_per_participant
    elif trial_randomization_strategy == "balanced":
        trials_each = len(within_labels) * trial_condition_count * trials_per_condition
    else:
        trials_each = len(within_labels) * trials_per_condition
    print(f"Participants: {n_participants}, Trials each: {trials_each}")
    print(f"Total trial rows: {len(trials)}")
    print(f"Exported → {csv_path}, {json_path}, {summary_path}")

    return {
        "trials":       trials,
        "assignments":  assignments,
        "orders":       orders,
        "strategy":     strategy,
        "all_conditions": all_conditions,
        "csv_path":     csv_path,
        "json_path":    json_path,
        "summary_path": summary_path,
    }

"""Experiment-design configuration helpers."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    import pandas as pd

from .counterbalance import factorial_conditions, split_ivs_by_design_role


DEFAULT_IV_CONFIG = {
    "xai_method": {
        "type": "within",
        "randomization": "block",
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
    "forward_accuracy": ["continuous"],
}


@dataclass
class DesignRoles:
    """Result of splitting experiment IVs into design roles."""

    between_ivs: dict[str, list[Any]]
    block_within_ivs: dict[str, list[Any]]
    trial_within_ivs: dict[str, list[Any]]
    within_ivs: dict[str, list[Any]]
    all_conditions: list[dict[str, Any]]


def validate_iv_config(config: dict[str, dict[str, Any]]) -> None:
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


def validate_factors(factors: dict[str, list[Any]]) -> None:
    """Validate control/dependent variable dictionaries."""
    for name, levels in factors.items():
        if not isinstance(levels, list) or not levels:
            raise ValueError(f"Factor {name!r} must map to a non-empty list.")
        if len(set(map(str, levels))) != len(levels):
            raise ValueError(f"Factor {name!r} has duplicate levels: {levels}.")


def set_iv(
    iv_config: dict[str, dict[str, Any]],
    name: str,
    iv_type: str,
    levels: list[Any],
    randomization: str = "block",
) -> dict[str, dict[str, Any]]:
    """Add or replace one IV after validating the requested configuration."""
    cfg = {"type": iv_type, "levels": levels}
    if iv_type == "within":
        cfg["randomization"] = randomization
    validate_iv_config({name: cfg})
    iv_config[name] = cfg
    return iv_config


def set_factor(factors: dict[str, list[Any]], name: str, levels: list[Any]) -> dict[str, list[Any]]:
    """Add or replace one CV/DV factor after validating its levels."""
    validate_factors({name: levels})
    factors[name] = levels
    return factors


def init_experiment_config() -> tuple[dict[str, dict[str, Any]], dict[str, list[Any]], dict[str, list[Any]]]:
    """Start an empty experiment configuration for iterative IV/CV/DV definition."""
    return {}, {}, {}


def between_iv(levels: list[Any]) -> dict[str, Any]:
    """Create a between-subjects IV spec."""
    return {"type": "between", "levels": levels}


def within_iv(levels: list[Any], randomization: str = "block") -> dict[str, Any]:
    """Create a within-subjects IV spec."""
    return {"type": "within", "randomization": randomization, "levels": levels}


def configure_experiment(
    ivs: Optional[dict[str, dict[str, Any]]] = None,
    cvs: Optional[dict[str, list[Any]]] = None,
    dvs: Optional[dict[str, list[Any]]] = None,
    *,
    available_datasets: Optional[list[str]] = None,
    support_check: bool = True,
    strict: bool = False,
    show: bool = True,
) -> tuple[dict[str, dict[str, Any]], dict[str, list[Any]], dict[str, list[Any]]]:
    """Build, validate, and optionally summarize IV/CV/DV configuration."""
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

    if support_check:
        from .support import validate_design_support

        validate_design_support(
            iv_config,
            control_vars,
            dependent_vars,
            strict=strict,
            show=show,
        )

    return iv_config, control_vars, dependent_vars


def validate_experiment_config(
    iv_config: dict[str, dict[str, Any]],
    cvs: dict[str, list[Any]],
    dvs: dict[str, list[Any]],
    *,
    available_datasets: Optional[list[str]] = None,
    support_check: bool = True,
    strict: bool = False,
    show: bool = True,
) -> tuple[dict[str, dict[str, Any]], dict[str, list[Any]], dict[str, list[Any]]]:
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

    if support_check:
        from .support import validate_design_support

        validate_design_support(
            iv_config,
            cvs,
            dvs,
            strict=strict,
            show=show,
        )

    return iv_config, cvs, dvs


def print_experiment_config(
    iv_config: dict[str, dict[str, Any]],
    cvs: dict[str, list[Any]],
    dvs: dict[str, list[Any]],
    *,
    available_datasets: Optional[list[str]] = None,
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


def inspect_design_roles(
    iv_config: dict[str, dict[str, Any]],
    *,
    show: bool = True,
) -> DesignRoles:
    """Split IVs by design role and optionally print a compact summary."""
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
    iv_config: dict[str, dict[str, Any]],
    *,
    show: bool = True,
) -> DesignRoles:
    """Experiment-facing alias for `inspect_design_roles`."""
    return inspect_design_roles(iv_config, show=show)


def build_controlled_vars(model_type: Optional[str], cvs: dict[str, list[Any]]) -> dict[str, str]:
    """Encode control-variable levels as trial-table metadata columns."""
    controlled_vars = {
        **{f"CV_{k}_levels": "|".join(str(v) for v in vals) for k, vals in cvs.items()},
    }
    if model_type is not None:
        controlled_vars["model_type"] = model_type
    return controlled_vars


def select_trial_rows(
    trials_df: pd.DataFrame,
    mode: str,
    *,
    participant_id: Optional[int] = None,
    condition_filter: Optional[dict[str, Any]] = None,
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

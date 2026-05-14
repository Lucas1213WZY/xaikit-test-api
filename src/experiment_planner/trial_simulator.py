"""
Trial Simulator for CoXAM Forward Simulation

High-level interface for running single-participant or batch CoXAM episodes
using the trained meta-model and headless strategy policies.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from src.cognitive_models.cr_agent.forward_meta_router import (
    load_forward_strategies,
    run_meta_on_batch,
)

from .cognitive_params import CoXAMCogParams, make_memory_factory
from .experimental_design import ExperimentDesign


@dataclass
class EpisodeResult:
    """Result of a single CoXAM forward episode."""

    nll: float
    total_reward: float
    logs: Dict[str, List[Any]]
    meta: Dict[str, Any]


def simulate_episode(
    *,
    meta_model: PPO,
    strategies: Dict[str, Any],
    X_raw: np.ndarray,
    y_raw: np.ndarray,
    design: ExperimentDesign,
    training_cog_params: Dict[str, Any],
    chi_value: float = 0.01,
    rng: Optional[np.random.Generator] = None,
    deterministic: bool = False,
) -> EpisodeResult:
    """
    Run one episode for a single participant configuration.

    Args:
        meta_model: Trained PPO for strategy selection
        strategies: Dict returned by load_forward_strategies
        X_raw: (N, F) raw features
        y_raw: (N,) ground-truth labels
        design: ExperimentDesign with schedules and episode_cogs
        training_cog_params: Parameter ranges from training (for normalization)
        chi_value: Time-cost coefficient
        rng: Random generator
        deterministic: Use deterministic policy selection
    """
    if rng is None:
        rng = np.random.default_rng()

    result = run_meta_on_batch(
        meta_model=meta_model,
        strategies=strategies,
        X_raw=X_raw,
        y_raw=y_raw,
        with_xai_schedule=design.with_xai_schedule,
        trial_type_schedule=design.trial_type_schedule,
        condition=design.condition,
        dataset_id=design.dataset_id,
        episode_cogs=design.episode_cogs,
        training_cog_params=training_cog_params,
        chi_value=chi_value,
        rng=rng,
        deterministic=deterministic,
    )

    probs_correct = result["logs"]["prob_correct"]
    losses = [-math.log(max(p, 1e-12)) for p in probs_correct if p > 0]
    nll = float(np.mean(losses)) if losses else float("inf")

    return EpisodeResult(
        nll=nll,
        total_reward=result["total_reward"],
        logs=result["logs"],
        meta=result["meta"],
    )


def simulate_participants(
    *,
    meta_model_path: str,
    dt_model_path: str,
    lr_calc_model_path: str,
    lr_heur_model_path: str,
    dt_exps: Dict[int, Any],
    lr_exps: Dict[int, Any],
    X_raw: np.ndarray,
    y_raw: np.ndarray,
    param_rows: List[Dict[str, Any]],
    design: ExperimentDesign,
    training_cog_params: Dict[str, Any],
    chi_value: float = 0.01,
    ddm_a_bins: int = 3,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Simulate multiple participants (one per param_row) and return a flat DataFrame.

    Each param_row is a dict (or pandas Series) of CoXAM cognitive parameters.
    Strategies are rebuilt per participant so each gets an independent memory state.
    """
    meta_model = PPO.load(meta_model_path)
    rows: List[Dict[str, Any]] = []

    for pidx, param_row in enumerate(param_rows):
        if isinstance(param_row, dict):
            cog = CoXAMCogParams.from_dict(param_row)
            decay = float(param_row.get("decay_param", 0.5))
        else:
            cog = CoXAMCogParams.from_csv_row(param_row)
            import pandas as _pd
            decay = float(param_row.get("decay_param", 0.5)) if not _pd.isna(param_row.get("decay_param", 0.5)) else 0.5

        mem_factory = make_memory_factory(decay_param=decay)
        strategies = load_forward_strategies(
            dt_model_path=dt_model_path,
            lr_calc_model_path=lr_calc_model_path,
            lr_heur_model_path=lr_heur_model_path,
            dt_exps=dt_exps,
            lr_exps=lr_exps,
            memory_factory=mem_factory,
            training_cog_params=training_cog_params,
            ddm_a_bins=ddm_a_bins,
        )

        ep_design = ExperimentDesign(
            N=design.N,
            condition=design.condition,
            with_xai_schedule=design.with_xai_schedule,
            trial_type_schedule=design.trial_type_schedule,
            dataset_id=design.dataset_id,
            episode_cogs=cog.to_dict(),
        )

        result = simulate_episode(
            meta_model=meta_model,
            strategies=strategies,
            X_raw=X_raw,
            y_raw=y_raw,
            design=ep_design,
            training_cog_params=training_cog_params,
            chi_value=chi_value,
            rng=np.random.default_rng(seed + pidx),
        )

        logs = result.logs
        for t in range(len(logs["strategy_name"])):
            rows.append({
                "participant_idx": pidx,
                "trial_idx": t,
                "strategy": logs["strategy_name"][t],
                "with_xai": logs["with_xai_used"][t],
                "trial_type": logs["trial_type"][t],
                "condition": design.condition,
                "dataset_id": design.dataset_id,
                "prob_correct": logs["prob_correct"][t],
                "pred_time": logs["pred_time"][t],
                "reward": logs["reward"][t],
                "nll": result.nll,
                **cog.to_dict(),
            })

    return pd.DataFrame(rows)

"""
Experimental Design Utilities for CoXAM Simulations

Builds per-episode schedules and trial structures.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

COND_DT = "DT"
COND_LR = "LR"
COND_DTLR = "DT+LR"


@dataclass
class ExperimentDesign:
    """Complete specification for one CoXAM forward episode."""

    N: int
    condition: str
    with_xai_schedule: np.ndarray   # shape (N,) bool
    trial_type_schedule: np.ndarray  # shape (N,) object, values "DT"|"LR"
    dataset_id: int
    episode_cogs: Dict[str, Any]


def build_design(
    *,
    N: int,
    condition: str,
    dataset_id: int,
    episode_cogs: Dict[str, Any],
    with_xai_ratio: float = 0.5,
    rng: Optional[np.random.Generator] = None,
) -> ExperimentDesign:
    """
    Build a randomized experimental design for one episode.

    Args:
        N: Number of trials
        condition: "DT", "LR", or "DT+LR"
        dataset_id: Dataset identifier passed to headless policies
        episode_cogs: Cognitive parameters dict (from CoXAMCogParams.to_dict)
        with_xai_ratio: Fraction of trials showing XAI
        rng: Random generator
    """
    from src.cognitive_models.cr_agent.forward_meta_router import (
        build_with_xai_schedule,
        build_trial_type_schedule,
    )

    if rng is None:
        rng = np.random.default_rng()

    return ExperimentDesign(
        N=N,
        condition=condition,
        with_xai_schedule=build_with_xai_schedule(N, with_xai_ratio, rng),
        trial_type_schedule=build_trial_type_schedule(N, condition, rng),
        dataset_id=dataset_id,
        episode_cogs=dict(episode_cogs),
    )


def grid_designs(
    *,
    N: int,
    conditions: Sequence[str] = (COND_DT, COND_LR, COND_DTLR),
    dataset_ids: Sequence[int] = (0,),
    with_xai_ratios: Sequence[float] = (0.5,),
    episode_cogs: Dict[str, Any],
    rng: Optional[np.random.Generator] = None,
) -> List[ExperimentDesign]:
    """Build a grid of designs across all condition × dataset × xai_ratio combinations."""
    if rng is None:
        rng = np.random.default_rng()

    designs = []
    for condition in conditions:
        for dataset_id in dataset_ids:
            for ratio in with_xai_ratios:
                designs.append(build_design(
                    N=N,
                    condition=condition,
                    dataset_id=dataset_id,
                    episode_cogs=episode_cogs,
                    with_xai_ratio=ratio,
                    rng=rng,
                ))
    return designs

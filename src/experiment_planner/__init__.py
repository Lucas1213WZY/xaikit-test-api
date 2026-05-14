"""CoXAM experiment planning and forward simulation utilities."""

from .cognitive_params import CoXAMCogParams, make_memory_factory, DEFAULT_TRAINING_COG_PARAMS
from .experimental_design import ExperimentDesign, build_design, grid_designs, COND_DT, COND_LR, COND_DTLR
from .trial_simulator import EpisodeResult, simulate_episode, simulate_participants

__all__ = [
    "CoXAMCogParams",
    "make_memory_factory",
    "DEFAULT_TRAINING_COG_PARAMS",
    "ExperimentDesign",
    "build_design",
    "grid_designs",
    "COND_DT",
    "COND_LR",
    "COND_DTLR",
    "EpisodeResult",
    "simulate_episode",
    "simulate_participants",
]

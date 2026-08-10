"""Sim2Real virtual-participant simulation.

Mirrors the CoAX and CoXAM packages: ``sim2real_trial_executor`` bridges a
study's trials onto the fitted attribution-sum model, and
``sim2real_study_runner`` is the entry point ``server/pipeline.py`` and
``xaikitTest.run_experiment`` call.
"""

from .sim2real_trial_executor import (  # noqa: F401
    DEFAULT_SIM2REAL_PARAMS,
    SIM2REAL_EXP_PROPERTIES,
    build_sim2real_model,
    build_sim2real_projector,
    run_sim2real_experiment_executor,
    sim2real_available_instance_ids,
)
from .sim2real_study_runner import run_sim2real_study  # noqa: F401

__all__ = [
    "DEFAULT_SIM2REAL_PARAMS",
    "SIM2REAL_EXP_PROPERTIES",
    "build_sim2real_model",
    "build_sim2real_projector",
    "run_sim2real_experiment_executor",
    "run_sim2real_study",
    "sim2real_available_instance_ids",
]

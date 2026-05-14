"""Virtual experiment execution API."""

from .coax_executor import (
    VirtualExperimentConfig,
    simulate_virtual_experiment,
    save_virtual_experiment_results,
    build_parser,
    main,
)

__all__ = [
    # CoAX
    "VirtualExperimentConfig",
    "simulate_virtual_experiment",
    "save_virtual_experiment_results",
    "build_parser",
    "main",
]

# CoXAM requires stable_baselines3 — import only when available
try:
    from .coxam_executor import (
        CoXAMExperimentConfig,
        simulate_virtual_experiment as simulate_coxam_experiment,
        save_virtual_experiment_results as save_coxam_experiment_results,
    )
    __all__ += [
        "CoXAMExperimentConfig",
        "simulate_coxam_experiment",
        "save_coxam_experiment_results",
    ]
except ImportError:
    pass

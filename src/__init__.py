"""XAIKit — public API.

Two equivalent ways to use the toolkit:

1. Guided orchestrator (a thin facade that delegates to the stage modules)::

       from src import xaikitTest
       exp = xaikitTest("my_study")

2. Stage modules directly, without the facade (each is independently usable)::

       from src import prepare_dataset, create_xai_method, build_experiment_plan
       data = prepare_dataset("wine_quality")
       method = create_xai_method("shap", ai_model=engine, train_data=...)

Symbols are resolved lazily (PEP 562), so ``import src`` stays cheap and only the
subsystems you actually touch (torch, xgboost, ...) get imported.
"""

from __future__ import annotations

import importlib
from typing import Any

# public name -> module that provides it
_EXPORTS: dict[str, str] = {
    # Orchestrator facade
    "xaikitTest": "src.api",
    # Data loading / preparation
    "prepare_dataset": "src.data_loaders",
    "PreparedDataset": "src.data_loaders",
    "UnifiedDataLoader": "src.data_loaders",
    "XAIDatasetParser": "src.data_loaders",
    # AI models + evaluation
    "ModelManager": "src.ai_models",
    "evaluate_model": "src.ai_models",
    "metrics_table": "src.ai_models",
    # XAI generation
    "create_xai_method": "src.xai_adapter",
    "create_custom_xai_method": "src.xai_adapter",
    "register_xai_method": "src.xai_adapter",
    "XAIAdapterResult": "src.xai_adapter",
    "generate_xai_explanation_tables": "src.xai_adapter",
    # Experiment planning
    "build_experiment_plan": "src.experiment_planner",
    "configure_experiment": "src.experiment_planner",
    "generate_experimental_trials": "src.experiment_planner",
    "edit_study_protocol": "src.experiment_planner",
    "preview_experiment_walkthrough": "src.experiment_planner",
    "preview_participant_trials": "src.experiment_planner",
    "init_experiment_config": "src.experiment_planner",
    # Virtual execution
    "run_experiment_executor": "src.virtual_experiment_executor",
    "save_simulated_results": "src.virtual_experiment_executor",
    "run_virtual_experiment": "src.virtual_experiment_executor",
    # Cognitive models
    "KNNBaseline": "src.cognitive_models",
    "default_cognitive_params": "src.cognitive_models",
    "dummy_cognitive_model": "src.cognitive_models",
    # Statistical analysis
    "analyze_iv_dv": "src.statistical_analyst",
    "pairwise_condition_tests": "src.statistical_analyst",
    # Result visualization
    "plot_dv_by_two_ivs": "src.result_visualizer",
    "plot_iv_dv_grid": "src.result_visualizer",
    "InteractionPlot": "src.result_visualizer",
    "ResultGrid": "src.result_visualizer",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    module_path = _EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module 'src' has no attribute {name!r}")
    return getattr(importlib.import_module(module_path), name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))

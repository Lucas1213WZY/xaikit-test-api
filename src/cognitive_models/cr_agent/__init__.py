"""
CR Agent - Cognitive Reasoning Agent RL System

Unified RL agent layer for strategy selection with integrated environments
and cognitive policies (both forward and counterfactual reasoning).

Consolidated Architecture (API-driven):
- headless_policies.py: Forward policies (DT, LR Calc, LR Heur)
- forward_meta_router.py: Forward episode orchestration with PPO meta model
- counterfactual_meta_router.py: Counterfactual Gym environment
- interface: High-level API for running episodes
- registry: Metadata and registration
- weights: Pre-trained model weights

All strategies loaded from cognitive_models API layer (single source of truth).

Usage:
    from src.cognitive_models.cr_agent import CRAgentRunner
    runner = CRAgentRunner(...)
    results = runner.run_episode(X_raw, y_raw, condition='DT', ...)
"""

from .interface import CRAgentRunner, MetaRunner
from .forward_meta_router import run_meta_on_batch, load_forward_strategies
from .counterfactual_meta_router import CounterfactualMetaRouter, load_counterfactual_strategies
from .registry import AgentRegistry, EnvironmentRegistry

__all__ = [
    "CRAgentRunner",
    "MetaRunner",
    "run_meta_on_batch",
    "load_forward_strategies",
    "CounterfactualMetaRouter",
    "load_counterfactual_strategies",
    "AgentRegistry",
    "EnvironmentRegistry",
]

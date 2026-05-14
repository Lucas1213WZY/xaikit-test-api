"""
Reasoning Strategies Module

This module provides a plugin-based architecture for reasoning strategies used in
cognitive agent simulations. Strategies are registered dynamically and can be
instantiated via the StrategyRegistry.

Structure:
- interface.py: Abstract contracts and enums (ReasoningStrategy, CounterfactualStrategy)
- registry.py: Plugin management system (StrategyRegistry)
- forward/: Forward reasoning strategies (inference-only)
  - coax_forward_rs.py: CoAX exemplar-based strategies
  - coxam_forward_rs.py: CoXAM memory-based strategies
- counterfactual/: Counterfactual explanation strategies
  - coxam_counterfactual_rs.py: CoXAM counterfactual strategies

Public API:
- StrategyRegistry: Main registry for strategy management
- ReasoningStrategy: Base class for all forward strategies
- CounterfactualStrategy: Base class for counterfactual strategies
- StrategyType, ReasoningMode: Enums for strategy classification
"""

from .interface import (
    StrategyType,
    ReasoningMode,
    StrategyConfig,
    StrategyMetadata,
    ReasoningStrategy,
    CounterfactualStrategy,
)

from .registry import StrategyRegistry

# Import forward strategies
from .forward.coax_forward_rs import (
    SensitiveFeatures,
    SalientFeatures,
    ImportanceCategorization,
    AttributionSum,
)

from .forward.coxam_forward_rs import (
    LRCalculation,
    LRHeuristic,
    DTTraversal,
)

# Import counterfactual strategies
from .counterfactual.coxam_counterfactual_rs import (
    ZeroOutLRHeuristic,
    ZeroOutLRDisplayed,
    ChangeDTPath,
    RecallChanges,
    MemoryBasedCF,
)

__all__ = [
    # Interface and enums
    'StrategyType',
    'ReasoningMode',
    'StrategyConfig',
    'StrategyMetadata',
    'ReasoningStrategy',
    'CounterfactualStrategy',
    # Registry
    'StrategyRegistry',
    # CoAX forward strategies
    'SensitiveFeatures',
    'SalientFeatures',
    'ImportanceCategorization',
    'AttributionSum',
    # CoXAM forward strategies
    'LRCalculation',
    'LRHeuristic',
    'DTTraversal',
    # CoXAM counterfactual strategies
    'ZeroOutLRHeuristic',
    'ZeroOutLRDisplayed',
    'ChangeDTPath',
    'RecallChanges',
    'MemoryBasedCF',
]


def initialize_strategies():
    """
    Initialize strategy registry by registering all available strategies.
    
    Call this once at application startup before using strategies.
    """
    def _cfg(name: str, strategy_type: StrategyType) -> StrategyConfig:
        return StrategyConfig(strategy_name=name, strategy_type=strategy_type)

    # Register CoAX forward strategies
    StrategyRegistry.register(
        'sensitive_features',
        SensitiveFeatures,
        SensitiveFeatures(_cfg('sensitive_features', StrategyType.COAX_FORWARD)).metadata
    )
    
    StrategyRegistry.register(
        'salient_features',
        SalientFeatures,
        SalientFeatures(_cfg('salient_features', StrategyType.COAX_FORWARD)).metadata
    )
    
    StrategyRegistry.register(
        'importance_categorization',
        ImportanceCategorization,
        ImportanceCategorization(_cfg('importance_categorization', StrategyType.COAX_FORWARD)).metadata
    )
    
    StrategyRegistry.register(
        'attribution_sum',
        AttributionSum,
        AttributionSum(_cfg('attribution_sum', StrategyType.COAX_FORWARD)).metadata
    )
    
    # Register CoXAM counterfactual strategies
    StrategyRegistry.register(
        'zeroout_lr_heuristic',
        ZeroOutLRHeuristic,
        ZeroOutLRHeuristic(_cfg('zeroout_lr_heuristic', StrategyType.COXAM_COUNTERFACTUAL)).metadata
    )
    
    StrategyRegistry.register(
        'zeroout_lr_displayed',
        ZeroOutLRDisplayed,
        ZeroOutLRDisplayed(_cfg('zeroout_lr_displayed', StrategyType.COXAM_COUNTERFACTUAL)).metadata
    )
    
    StrategyRegistry.register(
        'change_dt_path',
        ChangeDTPath,
        ChangeDTPath(_cfg('change_dt_path', StrategyType.COXAM_COUNTERFACTUAL)).metadata
    )
    
    StrategyRegistry.register(
        'recall_changes',
        RecallChanges,
        RecallChanges(_cfg('recall_changes', StrategyType.COXAM_COUNTERFACTUAL)).metadata
    )
    
    StrategyRegistry.register(
        'memory_based_cf',
        MemoryBasedCF,
        MemoryBasedCF(_cfg('memory_based_cf', StrategyType.COXAM_COUNTERFACTUAL)).metadata
    )

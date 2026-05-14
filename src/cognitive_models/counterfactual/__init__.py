"""
Counterfactual reasoning strategies initialization and exports.

Exports:
- ZeroOutLRHeuristic: Suggest features to zero-out to flip LR decision
- ZeroOutLRDisplayed: Use displayed LR for zero-out suggestions
- ChangeDTPath: Suggest features to follow different DT path
- RecallChanges: Retrieve stored counterfactual changes from memory
- MemoryBasedCF: Use memory to suggest alternative decisions
"""

from .coxam_counterfactual_rs import (
    ZeroOutLRHeuristic,
    ZeroOutLRDisplayed,
    ChangeDTPath,
    RecallChanges,
    MemoryBasedCF
)

__all__ = [
    'ZeroOutLRHeuristic',
    'ZeroOutLRDisplayed',
    'ChangeDTPath',
    'RecallChanges',
    'MemoryBasedCF',
]

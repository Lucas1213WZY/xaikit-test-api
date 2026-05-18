"""Example-based explanation adapters (counterfactuals and prototypes)."""

from .counterfactual import CounterfactualAdapter, DiCEAdapter
from .prototypes import PrototypesAdapter

__all__ = [
    "CounterfactualAdapter",
    "DiCEAdapter",
    "PrototypesAdapter",
]

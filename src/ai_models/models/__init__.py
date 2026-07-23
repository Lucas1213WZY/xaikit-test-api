"""Concrete AI model implementations.

Each module holds the PyTorch and TensorFlow variants of one model family,
plus the synthetic ground-truth functions. These are the low-level classes that
``model_api`` wraps in its UnifiedModel/ModelManager interface.
"""

from .mlp import MLPEngine, MLPModel, TFMLPEngine
from .xgboost import TFXGBoostEngine, XGBoostEngine
from .synthetic import (
    BaseSim2RealFunction,
    SparseFunction,
    TrendWiggleFunction,
    WiggleFunction,
    create_sim2real_function,
)

__all__ = [
    "MLPEngine",
    "MLPModel",
    "TFMLPEngine",
    "XGBoostEngine",
    "TFXGBoostEngine",
    "BaseSim2RealFunction",
    "SparseFunction",
    "TrendWiggleFunction",
    "WiggleFunction",
    "create_sim2real_function",
]

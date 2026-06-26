"""
XAIK Unified AI Models Layer
==============================
Clean, unified interface for loading, training, and using models.

Supported frameworks: PyTorch (mlp, xgboost) and TensorFlow (mlp_tf, xgboost_tf).
Use cognitive_agent='coax' or 'coxam' to select the agent-specific variant.

Quick Start:
    from src.ai_models import ModelManager

    manager = ModelManager()
    model = manager.load_model('wine_quality', 'mlp', source='coxam')
    predictions = manager.predict(X_test)

    # TensorFlow MLP
    tf_model = manager.load_model('adult', 'mlp_tf', source='coax')
"""

from .registry import ModelRegistry
from .models import (
    UnifiedModel,
    MLPUnifiedModel,
    XGBoostUnifiedModel,
    TFMLPUnifiedModel,
    TFXGBoostUnifiedModel,
    Sim2RealUnifiedModel,
    ModelManager,
    MODEL_REQUIRES_ONE_HOT_ENCODING,
    classification_metrics,
    load_pretrained_model,
    requires_one_hot_encoding,
)
from .sim2real import (
    BaseSim2RealFunction,
    SparseFunction,
    TrendWiggleFunction,
    WiggleFunction,
    create_sim2real_function,
)

__all__ = [
    'ModelManager',
    'ModelRegistry',
    'UnifiedModel',
    'MLPUnifiedModel',
    'XGBoostUnifiedModel',
    'TFMLPUnifiedModel',
    'TFXGBoostUnifiedModel',
    'Sim2RealUnifiedModel',
    'MODEL_REQUIRES_ONE_HOT_ENCODING',
    'BaseSim2RealFunction',
    'SparseFunction',
    'TrendWiggleFunction',
    'WiggleFunction',
    'create_sim2real_function',
    'classification_metrics',
    'load_pretrained_model',
    'requires_one_hot_encoding',
]

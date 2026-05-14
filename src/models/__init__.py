"""
XAIK Unified Models Layer
==========================
Clean, unified interface for loading, training, and using models.

Quick Start:
    from src.models import ModelManager
    
    manager = ModelManager()
    model = manager.load_model('wine_quality', 'mlp')
    predictions = manager.predict(X_test)
"""

from .registry import ModelRegistry
from .models import (
    UnifiedModel,
    MLPUnifiedModel,
    XGBoostUnifiedModel,
    ModelManager,
    load_pretrained_model,
)

__all__ = [
    'ModelManager',
    'ModelRegistry',
    'UnifiedModel',
    'MLPUnifiedModel',
    'XGBoostUnifiedModel',
    'load_pretrained_model',
]

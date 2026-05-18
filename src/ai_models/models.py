"""
Models Core
===========
Unified interface for MLP and XGBoost models (PyTorch and TensorFlow).

Supported model_type values:
  'mlp'          — PyTorch MLP  (src.ai_models.mlp.MLPEngine)
  'xgboost'      — XGBoost      (src.ai_models.xgboost.XGBoostEngine)
  'mlp_tf'       — TF/Keras MLP (src.ai_models.mlp_tf.TFMLPEngine)
  'xgboost_tf'   — TF-compatible XGBoost (src.ai_models.xgboost_tf.TFXGBoostEngine)

Pass source='coax' or source='coxam' to select the cognitive-agent variant.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Dict, Optional

import numpy as np

from .mlp import MLPEngine
from .xgboost import XGBoostEngine
from .mlp_tf import TFMLPEngine
from .xgboost_tf import TFXGBoostEngine


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------

class UnifiedModel(ABC):
    def __init__(self, dataset_name: str, model_type: str):
        self.dataset_name = dataset_name
        self.model_type = model_type
        self.is_trained = False
        self.metadata = {
            'dataset': dataset_name,
            'model_type': model_type,
            'framework': self._get_framework(),
        }

    @abstractmethod
    def _get_framework(self) -> str: ...

    @abstractmethod
    def load(self, weight_path: str) -> None: ...

    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray,
              X_dev: Optional[np.ndarray] = None,
              y_dev: Optional[np.ndarray] = None, **kwargs) -> Dict: ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float: ...

    @abstractmethod
    def save(self, weight_path: str) -> None: ...

    def get_info(self) -> Dict:
        return {**self.metadata, 'is_trained': self.is_trained}


# ---------------------------------------------------------------------------
# PyTorch MLP
# ---------------------------------------------------------------------------

class MLPUnifiedModel(UnifiedModel):
    """PyTorch MLP.  cognitive_agent controls coax/coxam variant."""

    def __init__(self, dataset_name: str, input_dim: int, num_classes: int,
                 cognitive_agent: str = 'coxam', **kwargs):
        super().__init__(dataset_name, 'mlp')
        self.engine = MLPEngine(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dimension=kwargs.get('hidden_dimension', 50),
            dropout_rate=kwargs.get('dropout_rate', 0),
            device_id=kwargs.get('device_id', -1),
            cognitive_agent=cognitive_agent,
        )
        self.metadata.update({'input_dim': input_dim, 'num_classes': num_classes,
                              'cognitive_agent': cognitive_agent})

    def _get_framework(self) -> str: return 'pytorch'

    def load(self, weight_path: str) -> None:
        self.engine.load(os.path.basename(weight_path))
        self.is_trained = True

    def train(self, X, y, X_dev=None, y_dev=None, **kwargs) -> Dict:
        epochs = kwargs.get('epochs', 300)
        batch_size = kwargs.get('batch_size', 1000)
        self.engine.train(X, y, X_dev=X_dev, y_dev=y_dev,
                          epochs=epochs, batch_size=batch_size)
        self.is_trained = True
        return {'epochs': epochs, 'batch_size': batch_size, 'framework': 'pytorch'}

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.engine.predict(X)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        return self.engine.evaluate(X, y)

    def save(self, weight_path: str) -> None:
        self.engine.save(os.path.basename(weight_path))


# ---------------------------------------------------------------------------
# XGBoost
# ---------------------------------------------------------------------------

class XGBoostUnifiedModel(UnifiedModel):
    """XGBoost.  cognitive_agent controls coax/coxam variant."""

    def __init__(self, dataset_name: str, cognitive_agent: str = 'coxam', **kwargs):
        super().__init__(dataset_name, 'xgboost')
        self.engine = XGBoostEngine(
            cognitive_agent=cognitive_agent,
            learning_rate=kwargs.get('learning_rate', 0.05),
            num_boost_round=kwargs.get('num_boost_round', None),
        )
        self.metadata.update({'cognitive_agent': cognitive_agent,
                              'hyperparams': {'learning_rate': kwargs.get('learning_rate', 0.05)}})

    def _get_framework(self) -> str: return 'xgboost'

    def load(self, weight_path: str) -> None:
        self.engine.load(os.path.basename(weight_path))
        self.is_trained = True

    def train(self, X, y, X_dev=None, y_dev=None, **kwargs) -> Dict:
        self.engine.train(X, y, X_dev=X_dev, y_dev=y_dev, **kwargs)
        self.is_trained = True
        return {'num_boost_round': self.engine.num_boost_round, 'framework': 'xgboost'}

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.engine.predict(X)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        return self.engine.evaluate(X, y)

    def save(self, weight_path: str) -> None:
        self.engine.save(os.path.basename(weight_path))


# ---------------------------------------------------------------------------
# TensorFlow MLP
# ---------------------------------------------------------------------------

class TFMLPUnifiedModel(UnifiedModel):
    """TensorFlow/Keras MLP.  cognitive_agent controls coax/coxam variant."""

    def __init__(self, dataset_name: str, input_dim: int, num_classes: int,
                 cognitive_agent: str = 'coxam', **kwargs):
        super().__init__(dataset_name, 'mlp_tf')
        self.engine = TFMLPEngine(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dimension=kwargs.get('hidden_dimension', 50),
            dropout_rate=kwargs.get('dropout_rate', 0.0),
            cognitive_agent=cognitive_agent,
        )
        self.metadata.update({'input_dim': input_dim, 'num_classes': num_classes,
                              'cognitive_agent': cognitive_agent})

    def _get_framework(self) -> str: return 'tensorflow'

    def load(self, weight_path: str) -> None:
        self.engine.load(os.path.basename(weight_path))
        self.is_trained = True

    def train(self, X, y, X_dev=None, y_dev=None, **kwargs) -> Dict:
        epochs = kwargs.get('epochs', 300)
        batch_size = kwargs.get('batch_size', 1000)
        self.engine.train(X, y, X_dev=X_dev, y_dev=y_dev,
                          epochs=epochs, batch_size=batch_size)
        self.is_trained = True
        return {'epochs': epochs, 'batch_size': batch_size, 'framework': 'tensorflow'}

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.engine.predict(X)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        return self.engine.evaluate(X, y)

    def save(self, weight_path: str) -> None:
        self.engine.save(os.path.basename(weight_path))


# ---------------------------------------------------------------------------
# TensorFlow-compatible XGBoost
# ---------------------------------------------------------------------------

class TFXGBoostUnifiedModel(UnifiedModel):
    """TF-compatible XGBoost (sklearn API).  cognitive_agent controls variant."""

    def __init__(self, dataset_name: str, cognitive_agent: str = 'coxam', **kwargs):
        super().__init__(dataset_name, 'xgboost_tf')
        self.engine = TFXGBoostEngine(
            cognitive_agent=cognitive_agent,
            learning_rate=kwargs.get('learning_rate', 0.05),
            num_boost_round=kwargs.get('num_boost_round', None),
        )
        self.metadata.update({'cognitive_agent': cognitive_agent})

    def _get_framework(self) -> str: return 'tensorflow'

    def load(self, weight_path: str) -> None:
        self.engine.load(os.path.basename(weight_path))
        self.is_trained = True

    def train(self, X, y, X_dev=None, y_dev=None, **kwargs) -> Dict:
        self.engine.train(X, y, X_dev=X_dev, y_dev=y_dev)
        self.is_trained = True
        return {'framework': 'tensorflow', 'backend': 'xgboost-sklearn'}

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.engine.predict(X)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        return self.engine.evaluate(X, y)

    def save(self, weight_path: str) -> None:
        self.engine.save(os.path.basename(weight_path))


# ---------------------------------------------------------------------------
# Model Manager
# ---------------------------------------------------------------------------

_DATASET_DEFAULTS = {
    'wine_quality':         {'input_dim': 11,  'num_classes': 2},
    'forest_cover':         {'input_dim': 54,  'num_classes': 7},
    'adult':                {'input_dim': 14,  'num_classes': 2},
    'german_credit':        {'input_dim': 24,  'num_classes': 2},
    'mushrooms':            {'input_dim': 22,  'num_classes': 2},
    'heart_disease':        {'input_dim': 13,  'num_classes': 2},
    'king_county_housing':  {'input_dim': 16,  'num_classes': 2},
    'prima_diabetes':       {'input_dim':  8,  'num_classes': 2},
    'breast_cancer':        {'input_dim': 30,  'num_classes': 2},
    'cardiotocography':     {'input_dim': 21,  'num_classes': 3},
}

_MODEL_CLASSES = {
    'mlp':          MLPUnifiedModel,
    'xgboost':      XGBoostUnifiedModel,
    'mlp_tf':       TFMLPUnifiedModel,
    'xgboost_tf':   TFXGBoostUnifiedModel,
}


class ModelManager:
    """
    Central manager for loading and using trained models.

    Usage
    -----
    manager = ModelManager()
    model = manager.load_model('wine_quality', 'mlp', source='coxam')
    predictions = manager.predict(X_test)
    """

    def __init__(self, registry=None):
        from .registry import ModelRegistry
        self.registry = registry or ModelRegistry()
        self.loaded_models: Dict[str, UnifiedModel] = {}
        self.active_model: Optional[UnifiedModel] = None

    def load_model(self, dataset: str, model_type: str, source: str = 'coxam',
                   auto_activate: bool = True) -> UnifiedModel:
        key = f'{dataset}_{model_type}_{source}'
        if key in self.loaded_models:
            if auto_activate:
                self.active_model = self.loaded_models[key]
            return self.loaded_models[key]

        model_info = self.registry.get_model_info(dataset, model_type, source)
        if not model_info:
            raise ValueError(
                f"Model not found: {dataset} ({model_type}) from {source}\n"
                f"Available: {self.registry.list_available_models()}"
            )

        meta = _DATASET_DEFAULTS.get(dataset, {'input_dim': 13, 'num_classes': 2})
        cls = _MODEL_CLASSES.get(model_type)
        if cls is None:
            raise ValueError(f"Unknown model_type '{model_type}'. "
                             f"Choose from {list(_MODEL_CLASSES)}")

        if model_type in ('mlp', 'mlp_tf'):
            model = cls(dataset_name=dataset, cognitive_agent=source,
                        input_dim=meta['input_dim'], num_classes=meta['num_classes'])
        else:
            model = cls(dataset_name=dataset, cognitive_agent=source)

        model.load(model_info['weight_path'])
        self.loaded_models[key] = model
        if auto_activate:
            self.active_model = model

        print(f"✓ Loaded {model_type} ({source}) for '{dataset}'")
        return model

    def create_model(self, dataset: str, model_type: str, input_dim: int,
                     num_classes: int, source: str = 'coxam', **kwargs) -> UnifiedModel:
        cls = _MODEL_CLASSES.get(model_type)
        if cls is None:
            raise ValueError(f"Unknown model_type '{model_type}'.")

        if model_type in ('mlp', 'mlp_tf'):
            model = cls(dataset_name=dataset, input_dim=input_dim,
                        num_classes=num_classes, cognitive_agent=source, **kwargs)
        else:
            model = cls(dataset_name=dataset, cognitive_agent=source, **kwargs)

        key = f'{dataset}_{model_type}_{source}_custom'
        self.loaded_models[key] = model
        self.active_model = model
        print(f"✓ Created new {model_type} ({source}) for '{dataset}'")
        return model

    def predict(self, X: np.ndarray, model: Optional[UnifiedModel] = None) -> np.ndarray:
        return (model or self._require_active()).predict(X)

    def train(self, X: np.ndarray, y: np.ndarray,
              model: Optional[UnifiedModel] = None,
              X_dev: Optional[np.ndarray] = None,
              y_dev: Optional[np.ndarray] = None, **kwargs) -> Dict:
        return (model or self._require_active()).train(X, y, X_dev=X_dev, y_dev=y_dev, **kwargs)

    def evaluate(self, X: np.ndarray, y: np.ndarray,
                 model: Optional[UnifiedModel] = None) -> float:
        return (model or self._require_active()).evaluate(X, y)

    def save_model(self, weight_path: str, model: Optional[UnifiedModel] = None) -> None:
        (model or self._require_active()).save(weight_path)
        print(f"✓ Saved to {weight_path}")

    def get_active_model(self) -> Optional[UnifiedModel]:
        return self.active_model

    def set_active_model(self, model_key: str) -> UnifiedModel:
        if model_key not in self.loaded_models:
            raise ValueError(f"'{model_key}' not loaded")
        self.active_model = self.loaded_models[model_key]
        return self.active_model

    def list_loaded_models(self) -> Dict:
        return {k: m.get_info() for k, m in self.loaded_models.items()}

    def list_available_pretrained(self) -> Dict:
        return self.registry.to_dict()

    def _require_active(self) -> UnifiedModel:
        if not self.active_model:
            raise ValueError("No active model. Call load_model() or create_model() first.")
        return self.active_model


def load_pretrained_model(dataset: str, model_type: str = 'mlp',
                          source: str = 'coxam') -> UnifiedModel:
    """Quick helper to load a pre-trained model."""
    return ModelManager().load_model(dataset, model_type, source)

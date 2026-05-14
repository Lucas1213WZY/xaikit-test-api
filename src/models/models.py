"""
Models Core
===========
Unified interface for MLP and XGBoost models.
Consolidates all model classes and the main ModelManager API.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple
import numpy as np
import sys
import os


class UnifiedModel(ABC):
    """Abstract interface for unified model access."""
    
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
    def _get_framework(self) -> str:
        pass
    
    @abstractmethod
    def load(self, weight_path: str) -> None:
        pass
    
    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray, X_dev: Optional[np.ndarray] = None, 
              y_dev: Optional[np.ndarray] = None, **kwargs) -> Dict:
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass
    
    @abstractmethod
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        pass
    
    @abstractmethod
    def save(self, weight_path: str) -> None:
        pass
    
    def get_info(self) -> Dict:
        return {
            **self.metadata,
            'is_trained': self.is_trained,
        }


class MLPUnifiedModel(UnifiedModel):
    """Unified interface for MLP (PyTorch) models."""
    
    def __init__(self, dataset_name: str, input_dim: int, num_classes: int, **kwargs):
        super().__init__(dataset_name, 'mlp')
        
        sys.path.insert(0, '/Users/wangzhuoyulucas/Documents/GitHub/xaik-tool-cognitive-agent/src/models/coax/mlp')
        sys.path.insert(0, '/Users/wangzhuoyulucas/Documents/GitHub/xaik-tool-cognitive-agent/src/models/coax')
        from model import MLPEngine
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.kwargs = {
            'input_dim': input_dim,
            'num_classes': num_classes,
            'hidden_dimension': kwargs.get('hidden_dimension', 50),
            'dropout_rate': kwargs.get('dropout_rate', 0),
            'device_id': kwargs.get('device_id', -1),
        }
        self.engine = MLPEngine(**self.kwargs)
        self.metadata['input_dim'] = input_dim
        self.metadata['num_classes'] = num_classes
    
    def _get_framework(self) -> str:
        return 'pytorch'
    
    def load(self, weight_path: str) -> None:
        self.engine.load(os.path.basename(weight_path))
        self.is_trained = True
    
    def train(self, X: np.ndarray, y: np.ndarray, X_dev: Optional[np.ndarray] = None,
              y_dev: Optional[np.ndarray] = None, **kwargs) -> Dict:
        epochs = kwargs.get('epochs', 300)
        batch_size = kwargs.get('batch_size', 1000)
        self.engine.train(X, y, X_dev=X_dev, y_dev=y_dev, epochs=epochs, batch_size=batch_size)
        self.is_trained = True
        return {'epochs': epochs, 'batch_size': batch_size, 'framework': 'pytorch'}
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.engine.predict(X)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        return self.engine.evaluate(X, y)
    
    def save(self, weight_path: str) -> None:
        self.engine.save(os.path.basename(weight_path))


class XGBoostUnifiedModel(UnifiedModel):
    """Unified interface for XGBoost models."""
    
    def __init__(self, dataset_name: str, **kwargs):
        super().__init__(dataset_name, 'xgboost')
        
        sys.path.insert(0, '/Users/wangzhuoyulucas/Documents/GitHub/xaik-tool-cognitive-agent/src/models/coax/xgboost')
        sys.path.insert(0, '/Users/wangzhuoyulucas/Documents/GitHub/xaik-tool-cognitive-agent/src/models/coax')
        from model import XGBoostEngine
        
        self.xgb_kwargs = {
            'learning_rate': kwargs.get('learning_rate', 0.05),
            'num_boost_round': kwargs.get('num_boost_round', 50),
        }
        self.engine = XGBoostEngine(**self.xgb_kwargs)
        self.metadata['hyperparams'] = self.xgb_kwargs
    
    def _get_framework(self) -> str:
        return 'xgboost'
    
    def load(self, weight_path: str) -> None:
        self.engine.load(os.path.basename(weight_path))
        self.is_trained = True
    
    def train(self, X: np.ndarray, y: np.ndarray, X_dev: Optional[np.ndarray] = None,
              y_dev: Optional[np.ndarray] = None, **kwargs) -> Dict:
        self.engine.train(X, y, X_dev=X_dev, y_dev=y_dev, **kwargs)
        self.is_trained = True
        return {'num_boost_round': self.xgb_kwargs['num_boost_round'], 'framework': 'xgboost'}
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.engine.predict(X)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        return self.engine.evaluate(X, y)
    
    def save(self, weight_path: str) -> None:
        self.engine.save(os.path.basename(weight_path))


class ModelManager:
    """
    Central manager for loading and using trained models.
    Supports loading, training, prediction, and model management.
    """
    
    def __init__(self, registry=None):
        from .registry import ModelRegistry
        self.registry = registry or ModelRegistry()
        self.loaded_models: Dict[str, UnifiedModel] = {}
        self.active_model: Optional[UnifiedModel] = None
    
    def load_model(self, dataset: str, model_type: str, source: str = 'coxam',
                   auto_activate: bool = True) -> UnifiedModel:
        """Load a pre-trained model."""
        model_key = f'{dataset}_{model_type}_{source}'
        
        if model_key in self.loaded_models:
            self.active_model = self.loaded_models[model_key]
            return self.active_model
        
        model_info = self.registry.get_model_info(dataset, model_type, source)
        if not model_info:
            raise ValueError(
                f"Model not found: {dataset} ({model_type}) from {source}\n"
                f"Available: {self.registry.list_available_models()}"
            )
        
        metadata = self._load_metadata(dataset, source)
        
        if model_type == 'mlp':
            model = MLPUnifiedModel(
                dataset_name=dataset,
                input_dim=metadata.get('input_dim', 13),
                num_classes=metadata.get('num_classes', 2),
            )
        elif model_type == 'xgboost':
            model = XGBoostUnifiedModel(dataset_name=dataset)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model.load(model_info['weight_path'])
        self.loaded_models[model_key] = model
        if auto_activate:
            self.active_model = model
        
        print(f"✓ Loaded {model_type} model for {dataset} from {source}")
        return model
    
    def create_model(self, dataset: str, model_type: str, input_dim: int, 
                    num_classes: int, **kwargs) -> UnifiedModel:
        """Create a new model for training."""
        if model_type == 'mlp':
            model = MLPUnifiedModel(dataset_name=dataset, input_dim=input_dim,
                                   num_classes=num_classes, **kwargs)
        elif model_type == 'xgboost':
            model = XGBoostUnifiedModel(dataset_name=dataset, **kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model_key = f'{dataset}_{model_type}_custom'
        self.loaded_models[model_key] = model
        self.active_model = model
        
        print(f"✓ Created new {model_type} model for {dataset}")
        return model
    
    def predict(self, X: np.ndarray, model: Optional[UnifiedModel] = None) -> np.ndarray:
        """Make predictions."""
        model = model or self.active_model
        if not model:
            raise ValueError("No model loaded. Call load_model() first.")
        return model.predict(X)
    
    def train(self, X: np.ndarray, y: np.ndarray, model: Optional[UnifiedModel] = None,
              X_dev: Optional[np.ndarray] = None, y_dev: Optional[np.ndarray] = None,
              **kwargs) -> Dict:
        """Train a model."""
        model = model or self.active_model
        if not model:
            raise ValueError("No model loaded. Call create_model() first.")
        return model.train(X, y, X_dev=X_dev, y_dev=y_dev, **kwargs)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray, 
                model: Optional[UnifiedModel] = None) -> float:
        """Evaluate model accuracy."""
        model = model or self.active_model
        if not model:
            raise ValueError("No model loaded. Call load_model() first.")
        return model.evaluate(X, y)
    
    def save_model(self, weight_path: str, model: Optional[UnifiedModel] = None) -> None:
        """Save model weights."""
        model = model or self.active_model
        if not model:
            raise ValueError("No model loaded.")
        model.save(weight_path)
        print(f"✓ Model saved to {weight_path}")
    
    def get_active_model(self) -> Optional[UnifiedModel]:
        """Get currently active model."""
        return self.active_model
    
    def set_active_model(self, model_key: str) -> UnifiedModel:
        """Set active model by key."""
        if model_key not in self.loaded_models:
            raise ValueError(f"Model '{model_key}' not loaded")
        self.active_model = self.loaded_models[model_key]
        return self.active_model
    
    def list_loaded_models(self) -> Dict[str, Dict]:
        """List all loaded models."""
        return {key: model.get_info() for key, model in self.loaded_models.items()}
    
    def list_available_pretrained(self) -> Dict:
        """Get all available pre-trained models from registry."""
        return self.registry.to_dict()
    
    def _load_metadata(self, dataset: str, source: str) -> Dict:
        """Load dataset metadata."""
        for base_path in [
            f'/Users/wangzhuoyulucas/Documents/GitHub/xaik-tool-cognitive-agent/src/coxam/datasets/{dataset}',
            f'/Users/wangzhuoyulucas/Documents/GitHub/xaik-tool-cognitive-agent/src/coax/datasets/{dataset}',
        ]:
            metadata_path = os.path.join(base_path, 'metadata.csv')
            if os.path.exists(metadata_path):
                try:
                    import pandas as pd
                    df = pd.read_csv(metadata_path).iloc[0]
                    return {
                        'input_dim': len(eval(df['feature_names'])),
                        'num_classes': len(eval(df['target_options'])),
                    }
                except:
                    pass
        
        defaults = {
            'wine_quality': {'input_dim': 11, 'num_classes': 2},
            'forest_cover': {'input_dim': 54, 'num_classes': 7},
            'mushrooms': {'input_dim': 22, 'num_classes': 2},
            'heart_disease': {'input_dim': 13, 'num_classes': 2},
            'king_county_housing': {'input_dim': 16, 'num_classes': 2},
            'prima_diabetes': {'input_dim': 8, 'num_classes': 2},
            'breast_cancer': {'input_dim': 30, 'num_classes': 2},
            'cardiotocography': {'input_dim': 21, 'num_classes': 3},
            'adult': {'input_dim': 14, 'num_classes': 2},
            'german_credit': {'input_dim': 24, 'num_classes': 2},
        }
        return defaults.get(dataset, {'input_dim': 13, 'num_classes': 2})


def load_pretrained_model(dataset: str, model_type: str = 'mlp', 
                         source: str = 'coxam') -> UnifiedModel:
    """Quick function to load a pre-trained model."""
    manager = ModelManager()
    return manager.load_model(dataset, model_type, source)

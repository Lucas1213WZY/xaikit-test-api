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
from typing import Any, Dict, Optional

import numpy as np

from .mlp import MLPEngine
from .xgboost import XGBoostEngine
from .mlp_tf import TFMLPEngine
from .xgboost_tf import TFXGBoostEngine


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def _labels_and_scores_from_predictions(
    predictions: np.ndarray,
    *,
    positive_label: int = 1,
    threshold: float = 0.5,
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Convert model outputs into class labels and optional score/proba values."""
    preds = np.asarray(predictions)

    if preds.ndim == 2:
        return np.argmax(preds, axis=1), preds

    flat = preds.reshape(-1)
    is_float = np.issubdtype(flat.dtype, np.floating)
    looks_like_binary_score = is_float and np.nanmin(flat) >= 0 and np.nanmax(flat) <= 1
    if looks_like_binary_score:
        return (flat >= threshold).astype(int), flat
    return flat.astype(int), None


def classification_metrics(
    y_true: np.ndarray,
    predictions: np.ndarray,
    *,
    positive_label: int = 1,
    threshold: float = 0.5,
    include_report: bool = False,
) -> Dict[str, Any]:
    """Compute common classification metrics from model predictions."""
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        balanced_accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y_true_arr = np.asarray(y_true).reshape(-1)
    y_pred, y_score = _labels_and_scores_from_predictions(
        predictions,
        positive_label=positive_label,
        threshold=threshold,
    )
    labels = np.unique(np.concatenate([y_true_arr, y_pred]))
    is_binary = len(labels) <= 2

    metrics: Dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true_arr, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_arr, y_pred)),
        "precision_macro": float(precision_score(y_true_arr, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true_arr, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true_arr, y_pred, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true_arr, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true_arr, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true_arr, y_pred, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true_arr, y_pred, labels=labels).tolist(),
        "labels": labels.tolist(),
    }

    if is_binary:
        metrics.update({
            "precision": float(precision_score(y_true_arr, y_pred, pos_label=positive_label, zero_division=0)),
            "recall": float(recall_score(y_true_arr, y_pred, pos_label=positive_label, zero_division=0)),
            "f1": float(f1_score(y_true_arr, y_pred, pos_label=positive_label, zero_division=0)),
        })

    try:
        if y_score is not None:
            if is_binary:
                if np.asarray(y_score).ndim == 2:
                    positive_index = list(labels).index(positive_label) if positive_label in labels else -1
                    positive_scores = y_score[:, positive_index]
                else:
                    positive_scores = y_score
                metrics["roc_auc"] = float(roc_auc_score(y_true_arr, positive_scores))
                metrics["average_precision"] = float(average_precision_score(y_true_arr, positive_scores))
            else:
                metrics["roc_auc_ovr"] = float(roc_auc_score(y_true_arr, y_score, multi_class="ovr"))
                metrics["roc_auc_ovo"] = float(roc_auc_score(y_true_arr, y_score, multi_class="ovo"))
    except ValueError:
        metrics["roc_auc"] = None

    if include_report:
        metrics["classification_report"] = classification_report(
            y_true_arr,
            y_pred,
            zero_division=0,
            output_dict=True,
        )

    return metrics


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

    def evaluate_metrics(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        positive_label: int = 1,
        threshold: float = 0.5,
        include_report: bool = False,
    ) -> Dict[str, Any]:
        """Return accuracy, F1, precision/recall, AUC, and confusion matrix."""
        return classification_metrics(
            y,
            self.predict(X),
            positive_label=positive_label,
            threshold=threshold,
            include_report=include_report,
        )


# ---------------------------------------------------------------------------
# PyTorch MLP
# ---------------------------------------------------------------------------

class MLPUnifiedModel(UnifiedModel):
    """PyTorch MLP. cognitive_agent controls optional coax/coxam variant hooks."""

    def __init__(self, dataset_name: str, input_dim: int, num_classes: int,
                 cognitive_agent: str = 'custom', **kwargs):
        super().__init__(dataset_name, 'mlp')
        self.engine = MLPEngine(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dimension=kwargs.get('hidden_dimension', 50),
            dropout_rate=kwargs.get('dropout_rate', 0),
            device_id=kwargs.get('device_id', -1),
            cognitive_agent=cognitive_agent,
        )
        self.metadata.update({'input_dim': input_dim, 'num_classes': num_classes})
        if cognitive_agent in {'coax', 'coxam'}:
            self.metadata['cognitive_agent'] = cognitive_agent

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
    """XGBoost. cognitive_agent controls optional coax/coxam variant behavior."""

    def __init__(self, dataset_name: str, cognitive_agent: str = 'custom', **kwargs):
        super().__init__(dataset_name, 'xgboost')
        self.engine = XGBoostEngine(
            cognitive_agent=cognitive_agent,
            learning_rate=kwargs.get('learning_rate', 0.05),
            num_boost_round=kwargs.get('num_boost_round', None),
        )
        self.metadata.update({'hyperparams': {'learning_rate': kwargs.get('learning_rate', 0.05)}})
        if cognitive_agent in {'coax', 'coxam'}:
            self.metadata['cognitive_agent'] = cognitive_agent

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
    """TensorFlow/Keras MLP. cognitive_agent controls optional coax/coxam variant hooks."""

    def __init__(self, dataset_name: str, input_dim: int, num_classes: int,
                 cognitive_agent: str = 'custom', **kwargs):
        super().__init__(dataset_name, 'mlp_tf')
        self.engine = TFMLPEngine(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dimension=kwargs.get('hidden_dimension', 50),
            dropout_rate=kwargs.get('dropout_rate', 0.0),
            cognitive_agent=cognitive_agent,
        )
        self.metadata.update({'input_dim': input_dim, 'num_classes': num_classes})
        if cognitive_agent in {'coax', 'coxam'}:
            self.metadata['cognitive_agent'] = cognitive_agent

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
    """TF-compatible XGBoost (sklearn API). cognitive_agent controls optional variant."""

    def __init__(self, dataset_name: str, cognitive_agent: str = 'custom', **kwargs):
        super().__init__(dataset_name, 'xgboost_tf')
        self.engine = TFXGBoostEngine(
            cognitive_agent=cognitive_agent,
            learning_rate=kwargs.get('learning_rate', 0.05),
            num_boost_round=kwargs.get('num_boost_round', None),
        )
        if cognitive_agent in {'coax', 'coxam'}:
            self.metadata['cognitive_agent'] = cognitive_agent

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
                     num_classes: int, source: Optional[str] = None, **kwargs) -> UnifiedModel:
        """Create a new untrained model.

        Omitting `source` creates a generic/custom model. Pass source='coax' or
        source='coxam' only when the new model should use those variant hooks.
        """
        cls = _MODEL_CLASSES.get(model_type)
        if cls is None:
            raise ValueError(f"Unknown model_type '{model_type}'.")

        cognitive_agent = kwargs.pop('cognitive_agent', source or 'custom')
        if model_type in ('mlp', 'mlp_tf'):
            model = cls(dataset_name=dataset, input_dim=input_dim,
                        num_classes=num_classes, cognitive_agent=cognitive_agent, **kwargs)
        else:
            model = cls(dataset_name=dataset, cognitive_agent=cognitive_agent, **kwargs)

        key = f'{dataset}_{model_type}_{cognitive_agent}_custom'
        self.loaded_models[key] = model
        self.active_model = model
        if source:
            print(f"✓ Created new {model_type} ({source}) for '{dataset}'")
        else:
            print(f"✓ Created new {model_type} for '{dataset}'")
        return model

    def predict(self, X: np.ndarray, model: Optional[UnifiedModel] = None) -> np.ndarray:
        return (model or self._require_active()).predict(X)

    def train(self, X: np.ndarray, y: np.ndarray,
              model: Optional[UnifiedModel] = None,
              X_dev: Optional[np.ndarray] = None,
              y_dev: Optional[np.ndarray] = None, **kwargs) -> Dict:
        return (model or self._require_active()).train(X, y, X_dev=X_dev, y_dev=y_dev, **kwargs)

    def train_until_accuracy(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        target_accuracy: float,
        max_epochs: int = 300,
        check_every_epochs: int = 10,
        batch_size: int = 1000,
        model: Optional[UnifiedModel] = None,
        eval_X: Optional[np.ndarray] = None,
        eval_y: Optional[np.ndarray] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Train in chunks and stop once accuracy reaches `target_accuracy`.

        By default, the stop condition uses training accuracy on `X`/`y`.
        Pass `eval_X` and `eval_y` to stop on another split.
        """
        if not 0 < target_accuracy <= 1:
            raise ValueError("target_accuracy must be between 0 and 1.")
        if max_epochs <= 0:
            raise ValueError("max_epochs must be positive.")
        if check_every_epochs <= 0:
            raise ValueError("check_every_epochs must be positive.")

        active_model = model or self._require_active()
        eval_X = X if eval_X is None else eval_X
        eval_y = y if eval_y is None else eval_y

        total_epochs = 0
        history = []

        while total_epochs < max_epochs:
            epochs_this_round = min(check_every_epochs, max_epochs - total_epochs)
            active_model.train(
                X,
                y,
                epochs=epochs_this_round,
                batch_size=batch_size,
                **kwargs,
            )
            active_model.is_trained = True
            total_epochs += epochs_this_round

            accuracy = active_model.evaluate(eval_X, eval_y)
            history.append({"epochs": total_epochs, "accuracy": accuracy})
            print(f"After {total_epochs} epochs: accuracy={accuracy:.4f}")

            if accuracy >= target_accuracy:
                print(f"Reached target accuracy {target_accuracy:.4f}; stopping training.")
                break

        final_accuracy = history[-1]["accuracy"]
        return {
            "target_accuracy": target_accuracy,
            "final_accuracy": final_accuracy,
            "epochs": total_epochs,
            "batch_size": batch_size,
            "reached_target": final_accuracy >= target_accuracy,
            "history": history,
        }

    def evaluate(self, X: np.ndarray, y: np.ndarray,
                 model: Optional[UnifiedModel] = None) -> float:
        return (model or self._require_active()).evaluate(X, y)

    def evaluate_metrics(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model: Optional[UnifiedModel] = None,
        *,
        positive_label: int = 1,
        threshold: float = 0.5,
        include_report: bool = False,
    ) -> Dict[str, Any]:
        """Compute common classification metrics for the active or supplied model."""
        return (model or self._require_active()).evaluate_metrics(
            X,
            y,
            positive_label=positive_label,
            threshold=threshold,
            include_report=include_report,
        )

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

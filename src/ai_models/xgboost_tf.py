"""TensorFlow-compatible XGBoost engine — unified for CoAX and CoXAM.

Uses XGBClassifier (sklearn API) so it integrates with:
  - SHAP TreeExplainer  (via .sklearn_model property)
  - LIME               (via predict_proba callable)
  - TF-based XAI pipelines (via tf_predict, which wraps predict in tf.py_function)

cognitive_agent='coxam' enables predict_proba; default num_boost_round=50.
cognitive_agent='coax'  omits predict_proba; default num_boost_round=10.

Weight files (.json) are read from / written to:
    src/ai_models/<cognitive_agent>/xgboost/<file_name>
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score

_ROOT = Path(__file__).parent
_BOOST_DEFAULTS = {'coax': 10, 'coxam': 50}


class _BaseEngine(ABC):
    gradient_based: bool = False

    @abstractmethod
    def predict(self, X_dense) -> np.ndarray: ...
    @abstractmethod
    def train(self, X, y, **kw): ...
    @abstractmethod
    def save(self, file_name: str | None = None): ...
    @abstractmethod
    def load(self, file_name: str): ...


class TFXGBoostEngine(_BaseEngine):
    """
    Parameters
    ----------
    cognitive_agent : 'coax' | 'coxam'
        'coxam' — enables predict_proba; default num_boost_round=50;
                  weights from coxam/xgboost/.
        'coax'  — omits predict_proba; default num_boost_round=10;
                  weights from coax/xgboost/.
    learning_rate : float
    num_boost_round : int or None
        Overrides the cognitive_agent default when provided.
    """

    def __init__(self, cognitive_agent: str = 'coxam', learning_rate: float = 0.05,
                 num_boost_round: int | None = None, **kwargs):
        from xgboost import XGBClassifier

        self.gradient_based = False
        self._agent = cognitive_agent
        self._weight_dir = _ROOT / cognitive_agent / 'xgboost'
        self._n_rounds = num_boost_round or _BOOST_DEFAULTS.get(cognitive_agent, 50)

        self.model = XGBClassifier(
            n_estimators=self._n_rounds,
            learning_rate=learning_rate,
            eval_metric='logloss',
            **{k: v for k, v in kwargs.items()},
        )

    def train(self, X, y, X_dev=None, y_dev=None, **_):
        eval_set = [(X, y)]
        if X_dev is not None and y_dev is not None:
            eval_set.append((X_dev, y_dev))
        self.model.fit(X, y, eval_set=eval_set, verbose=False)

    def predict(self, X_dense) -> np.ndarray:
        return self.model.predict_proba(np.atleast_2d(X_dense))

    # coxam-only: explicit sklearn-style interface used by LIME / surrogate adapters
    def predict_proba(self, X_dense) -> np.ndarray:
        if self._agent != 'coxam':
            raise AttributeError(
                "predict_proba is only available for cognitive_agent='coxam'"
            )
        return self.predict(X_dense)

    def tf_predict(self, X_dense):
        """Return class probabilities as tf.Tensor for TF-based XAI pipelines."""
        import tensorflow as tf
        X_np = X_dense.numpy() if hasattr(X_dense, 'numpy') else np.atleast_2d(X_dense)
        return tf.constant(self.predict(X_np), dtype=tf.float32)

    def evaluate(self, X, y) -> float:
        return accuracy_score(y, self.model.predict(X))

    def save(self, file_name: str | None = None):
        path = self._weight_dir / (file_name or 'tf_xgb_model.json')
        self.model.save_model(str(path))
        print(f"[tf-xgboost/{self._agent}] saved → {path}")

    def load(self, file_name: str):
        path = self._weight_dir / file_name
        self.model.load_model(str(path))
        print(f"[tf-xgboost/{self._agent}] loaded ← {path}")

    @property
    def sklearn_model(self):
        """Underlying XGBClassifier for direct use with SHAP TreeExplainer."""
        return self.model

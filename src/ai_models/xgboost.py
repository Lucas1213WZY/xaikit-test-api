"""XGBoost engine — unified for CoAX and CoXAM.

cognitive_agent='coxam' enables predict_proba and uses num_boost_round=50.
cognitive_agent='coax'  omits predict_proba and uses num_boost_round=10.

Weight files are read from / written to:
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


class XGBoostEngine(_BaseEngine):
    """
    Parameters
    ----------
    cognitive_agent : 'coax' | 'coxam'
        'coax'  — no predict_proba; default num_boost_round=10;
                  weights from coax/xgboost/.
        'coxam' — adds predict_proba; default num_boost_round=50;
                  weights from coxam/xgboost/.
    learning_rate : float
    num_boost_round : int or None
        Overrides the cognitive_agent default when provided.
    """

    def __init__(self, cognitive_agent='coxam', learning_rate=0.05,
                 num_boost_round: int | None = None, **kwargs):
        self.gradient_based = False
        self._agent = cognitive_agent
        self._weight_dir = _ROOT / cognitive_agent / 'xgboost'
        self.num_boost_round = num_boost_round or _BOOST_DEFAULTS.get(cognitive_agent, 50)

        self.params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'learning_rate': learning_rate,
            **kwargs,
        }
        self.model = None

    def train(self, X, y, X_dev=None, y_dev=None, **_):
        import xgboost as xgb
        dtrain = xgb.DMatrix(X, label=y)
        evals = [(dtrain, 'train')]
        if X_dev is not None and y_dev is not None:
            evals.append((xgb.DMatrix(X_dev, label=y_dev), 'dev'))
        self.model = xgb.train(self.params, dtrain, self.num_boost_round, evals=evals)

    def predict(self, X_dense) -> np.ndarray:
        import xgboost as xgb
        p1 = self.model.predict(xgb.DMatrix(np.atleast_2d(X_dense)))
        if self.params['objective'] == 'binary:logistic':
            return np.vstack((1 - p1, p1)).T
        return np.array(p1)

    # coxam-only: sklearn-style probability accessor used by LIME / surrogate adapters
    def predict_proba(self, X_dense) -> np.ndarray:
        if self._agent != 'coxam':
            raise AttributeError(
                "predict_proba is only available for cognitive_agent='coxam'"
            )
        return self.predict(X_dense)

    def evaluate(self, X, y) -> float:
        import xgboost as xgb
        preds = (self.model.predict(xgb.DMatrix(X)) > 0.5).astype(int)
        return accuracy_score(y, preds)

    def save(self, file_name: str | None = None):
        path = self._weight_dir / (file_name or 'model_weights.json')
        self.model.save_model(str(path))
        print(f"[xgboost/{self._agent}] saved → {path}")

    def load(self, file_name: str):
        import xgboost as xgb
        path = self._weight_dir / file_name
        self.model = xgb.Booster()
        self.model.load_model(str(path))
        print(f"[xgboost/{self._agent}] loaded ← {path}")

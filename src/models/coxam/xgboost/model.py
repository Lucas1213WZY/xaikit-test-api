import xgboost as xgb
import numpy as np
from sklearn.metrics import accuracy_score
import os
from models.base_engine import BaseEngine


class XGBoostEngine(BaseEngine):
    def __init__(self, *args, **kwargs):
        super(XGBoostEngine, self).__init__(*args, **kwargs)

        # Default parameters for XGBoost, can be overridden via kwargs
        default_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'learning_rate': 0.05,
        }
        num_boost_round = kwargs.pop('num_boost_round', 50)
        self.params = {**default_params, **kwargs}
        self.num_boost_round = num_boost_round
        self.model = None

        self.gradient_based = False

    def train(self, X, y, X_dev=None, y_dev=None, batch_size=None, **kwargs):
        dtrain = xgb.DMatrix(X, label=y)
        evals = [(dtrain, 'train')]
        if X_dev is not None and y_dev is not None:
            ddev = xgb.DMatrix(X_dev, label=y_dev)
            evals.append((ddev, 'dev'))
        self.model = xgb.train(self.params, dtrain, self.num_boost_round, evals=evals)

    def predict(self, X_dense):
        X_dense = np.atleast_2d(X_dense)
        ddata = xgb.DMatrix(X_dense)
        if self.params['objective'] == 'binary:logistic':
            prob_class1 = self.model.predict(ddata)
            prob_class0 = 1 - prob_class1
            probabilities = np.vstack((prob_class0, prob_class1)).T
            return probabilities
        return np.array(self.model.predict(ddata))
    
    def predict_proba(self, X_dense):
        X_dense = np.atleast_2d(X_dense)
        ddata = xgb.DMatrix(X_dense)
        if self.params['objective'] == 'binary:logistic':
            prob_class1 = self.model.predict(ddata)
            prob_class0 = 1 - prob_class1
            probabilities = np.vstack((prob_class0, prob_class1)).T
            return probabilities
        return self.model.predict(ddata)

    def evaluate(self, X, y):
        X = xgb.DMatrix(X)
        predictions = np.array(self.model.predict(X))
        predictions = (predictions > 0.5).astype(int)
        return accuracy_score(y, predictions)

    def save(self, file_name=None):
        if file_name is None:
            file_name = "model_weights.json"
        file_path = os.path.join(os.path.dirname(__file__), file_name)
        self.model.save_model(file_path)
        print(f"Model weights saved to {file_path}")

    def load(self, file_name):
        file_path = os.path.join(os.path.dirname(__file__), file_name)
        self.model = xgb.Booster()
        self.model.load_model(file_path)
        print(f"Model weights loaded from {file_path}")

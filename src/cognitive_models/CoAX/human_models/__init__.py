from .base_model import BaseModel


import numpy as np
from .GCM import GeneralizedContextModel
from .decision_tree import DecisionTreeModel
from .naive_bayes import NaiveBayesModel
from .dummy_model import DummyModel
from .logistic_regression import LogisticRegressionModel
from .MLP import MLPModel
from .KNN import KNNModel

from .CoAX.gcm_multiple_strategies import *
# Need to change the above later

class BaselineModelHandler:
    """
    Handles initialization, training, and inference for various baseline models.
    Adds logic to handle untrained models and proxy methods for seamless access to model methods.
    """

    def __init__(self, model_type='DecisionTree', **kwargs):
        """
        Initializes the baseline model.

        Args:
            model_type (str): The type of model (e.g., 'DecisionTree', 'NaiveBayes').
            kwargs: Additional hyperparameters to pass to the model constructor.
        """
        # Dynamically load the appropriate model class
        if model_type == 'DecisionTree':
            self._model = DecisionTreeModel(**kwargs)
        elif model_type == "KNN":
            self._model = KNNModel(**kwargs)
        elif model_type == 'NaiveBayes':
            self._model = NaiveBayesModel(**kwargs)
        elif model_type == 'Dummy':
            self._model = DummyModel(**kwargs)
        elif model_type == 'GCM':
            self._model = GeneralizedContextModel(**kwargs)
        elif model_type == 'LogisticRegression':
            self._model = LogisticRegressionModel(**kwargs)
        elif model_type == 'MLP':
            self._model = MLPModel(**kwargs)
            self.train_calls = 0  # Track number of calls to train for MLP
        elif model_type == 'Random':
            self._model = RandomModel(**kwargs)
        else:
            raise ValueError(f"Unsupported model type: '{model_type}'.")

        self.is_trained = False  # Track whether the model has been trained

    def add_exemplar(self, features, label):
        """
        Adds a training example to the model.

        Args:
            features (list or array): The feature vector.
            label (int): The corresponding label (0 or 1).
        """
        self._model.add_exemplar(features, label)

    def train(self):
        """
        Trains the model if it has a `train` method and updates the `is_trained` flag.
        """
        if hasattr(self._model, 'train'):
            if isinstance(self._model, MLPModel):
                if not self.is_trained:
                    self._model.train()
                    self.is_trained = True
                else:
                    self.train_calls += 1
                    if self.train_calls % 3 == 0:
                        self._model.train()
            else:
                self._model.train()
                self.is_trained = True
        else:
            # If the underlying model doesn't have a train method, assume it's always ready
            self.is_trained = True

    def infer(self, features, actual_ai_prediction=None):
        """
        Predicts probabilities for the given features.
        Handles untrained models by returning uniform probabilities.

        Args:
            features (list or array): The feature vector to predict.
            actual_ai_prediction (optional): Additional input for certain model types (e.g., DummyModel).

        Returns:
            dict: A dictionary of predicted probabilities for each class.
        """
        if not self.is_trained:
            # Default guess if the model hasn't been trained
            return {0: 0.5, 1: 0.5}

        # Delegate inference to the underlying model
        if isinstance(self._model, DummyModel):
            return self._model.infer(features, actual_ai_prediction)
        return self._model.infer(features)

    def __getattr__(self, name):
        """
        Proxy method to delegate attribute access to the underlying model.
        
        Args:
            name (str): The name of the attribute or method.

        Returns:
            The attribute or method from the underlying model, if it exists.
        """
        if hasattr(self._model, name):
            return getattr(self._model, name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

class ExplanationAwareHumanModel:
    """
    A wrapper that manages two baseline models (same model type & hyperparameters):
    1) One for inference without explanation
    2) One for inference with explanation

    During the train (feedback) phase, it updates whichever models were used 
    (no explanation and/or with explanation).
    """

    def __init__(self, model_type='DecisionTree', **kwargs):
        """
        Creates two internal models of the same type and hyperparameters.

        Args:
            model_type (str): The model type (e.g., 'DecisionTree', 'NaiveBayes', etc.).
            model_kwargs (dict): A dictionary of hyperparameters to pass to the model constructor.
        """

        self.model_type = model_type
        self.model_kwargs = kwargs or {}

        # Instantiate two identical baseline models:
        #    self.model_no_exp for no explanation
        #    self.model_with_exp for with explanation
        self.model_no_exp = BaselineModelHandler(model_type=self.model_type, **self.model_kwargs)
        self.model_with_exp = BaselineModelHandler(model_type=self.model_type, **self.model_kwargs)

        # Keep track of the last inference scenarios for training during feedback
        self.last_inference_no_exp_features = None
        self.last_inference_with_exp_features = None

    def new_instance(self):
        """
        Resets internal references for each new trial.
        """
        self.last_inference_no_exp_features = None
        self.last_inference_with_exp_features = None

    def infer(self, ui):
        """
        Simulates the 'inference' step:
         1) Checks if an explanation is displayed on the UI.
         2) If yes, we combine features + explanation and use model_with_exp.
            Otherwise, we use model_no_exp.
         3) Stores which models were used so that feedback() trains the correct ones.
        """
        # Retrieve feature values from the UI
        feature_values = ui.get_value('features') or []
        
        # Check whether an explanation is displayed
        explanation = ui.get_value('explanation')
        has_explanation = (explanation is not None)

        # Concatenate if explanation is present
        if has_explanation:
            combined_input = list(feature_values) + list(explanation)
            prediction_dict = self.model_with_exp.infer(combined_input, actual_ai_prediction=ui.get_value('ai_prediction'))
            self.last_inference_with_exp_features = combined_input
        else:
            prediction_dict = self.model_no_exp.infer(feature_values, actual_ai_prediction=ui.get_value('ai_prediction'))
            self.last_inference_no_exp_features = feature_values

        # Example time cost
        time_used = 0.5
        return prediction_dict, time_used

    def feedback(self, ui):
        """
        Simulates the 'feedback' step:
         1) Retrieves the correct label (or any supervised signal).
         2) Trains both models if both were used during inference.
        """
        # Suppose the correct label is stored under key 'correct_label' in the UI
        label = ui.get_value('ai_prediction')
        if label is None:
            # If there's no label, we can't train
            return 0.1

        # Train the "no explanation" model if it was used
        if self.last_inference_no_exp_features is not None:
            self.model_no_exp.add_exemplar(self.last_inference_no_exp_features, label)
            self.model_no_exp.train()

        # Train the "with explanation" model if it was used
        if self.last_inference_with_exp_features is not None:
            self.model_with_exp.add_exemplar(self.last_inference_with_exp_features, label)
            self.model_with_exp.train()

        # Example time cost for feedback
        return 0.2


class RandomModel:
    """
    A baseline model that returns 0 and 1 predictions with equal probability,
    ignoring any features or explanations.
    """
    def __init__(self, **kwargs):
        # You can store additional arguments here if needed
        pass

    def add_exemplar(self, features, label):
        """
        A no-op for random model (it does not learn from exemplars).
        """
        pass

    def train(self):
        """
        A no-op for random model (no training step required).
        """
        pass

    def infer(self, features):
        """
        Returns a dictionary with probabilities for each class.
        Always 50/50 for random baseline (or any desired distribution).
        """
        return {0: 0.5, 1: 0.5}

from sklearn.linear_model import LogisticRegression
import numpy as np

class LogisticRegressionModel:
    """A class to handle a logistic regression model for classification tasks with adaptive smoothing and sparsity constraint."""

    def __init__(self, smoothing_factor=0.01, k=None):
        """
        Initialize the Logistic Regression model.

        Parameters:
        - smoothing_factor (float): The base smoothing factor to apply to class probabilities.
        - k (int or None): Maximum number of non-zero coefficients. If None, no constraint is applied.
        """
        self.smoothing_factor = smoothing_factor
        self.k = k
        self.features = []
        self.labels = []
        self.known_classes = set()  # Track the known classes in the dataset
        self.model = self._create_model()

    def _create_model(self):
        """
        Create a logistic regression model with L1 regularization to enforce sparsity.
        """
        if self.k is None:
            return LogisticRegression(penalty='l2', solver='liblinear')  # Default without sparsity
        else:
            # Use L1 regularization to enforce sparsity
            return LogisticRegression(penalty='l1', solver='liblinear', C=1.0)

    def add_exemplar(self, features, label):
        """
        Add a training example to the model.

        Parameters:
        - features (list or array): The feature vector.
        - label (int): The corresponding label (0 or 1).
        """
        self.features.append(features)
        self.labels.append(label)
        self.known_classes.add(label)

    def train(self):
        """Train the logistic regression model using the accumulated training data."""
        if len(self.features) > 0 and len(self.labels) > 0:
            X = np.array(self.features)
            y = np.array(self.labels)
            self.model.fit(X, y)

            if self.k is not None:
                self._enforce_sparsity()
        else:
            raise ValueError("No training data available. Add exemplars before training.")

    def _enforce_sparsity(self):
        """
        Adjust the regularization strength to ensure the number of non-zero coefficients is less than or equal to k.
        """
        non_zero_count = np.sum(self.model.coef_ != 0)
        C = 1.0  # Start with default regularization strength

        while non_zero_count > self.k:
            C *= 0.9  # Increase regularization strength (decrease C)
            self.model = LogisticRegression(penalty='l1', solver='liblinear', C=C)
            X = np.array(self.features)
            y = np.array(self.labels)
            self.model.fit(X, y)
            non_zero_count = np.sum(self.model.coef_ != 0)

    def infer(self, features):
        """
        Predict the probabilities for the given feature vector with adaptive smoothing applied.

        Parameters:
        - features (list or array): The feature vector to predict.

        Returns:
        - dict: A dictionary with smoothed predicted probabilities for each class.
        """
        if len(self.known_classes) == 0:
            # If no training has occurred, return uniform probabilities
            return {0: 0.5, 1: 0.5}

        # Get predicted probabilities for the input features
        probabilities = self.model.predict_proba([features])[0]

        # Ensure all known classes are accounted for in probabilities
        all_classes = sorted(self.known_classes)
        class_probabilities = {cls: probabilities[i] if i < len(probabilities) else 0 for i, cls in enumerate(all_classes)}

        # Apply adaptive smoothing by adding the smoothing factor to each class probability
        smoothed_probabilities = {cls: prob + self.smoothing_factor for cls, prob in class_probabilities.items()}

        # Normalize the smoothed probabilities to ensure they sum to 1
        total_prob = sum(smoothed_probabilities.values())
        normalized_probabilities = {cls: prob / total_prob for cls, prob in smoothed_probabilities.items()}

        # Ensure all known classes are present in the output
        for cls in self.known_classes:
            if cls not in normalized_probabilities:
                normalized_probabilities[cls] = 0.0

        return normalized_probabilities

    def print_weights(self, feature_names=None):
        """
        Print the logistic regression weights, ranked by their absolute value.

        Parameters:
        - feature_names (list or None): The names of the features. If None, indices are used instead.
        """
        if not hasattr(self.model, 'coef_'):
            raise ValueError("The model has not been trained yet. Train the model before inspecting weights.")

        # Get the weights (coefficients) of the logistic regression model
        weights = self.model.coef_[0]
        if feature_names is None:
            feature_names = [f"Feature {i}" for i in range(len(weights))]

        # Create a list of feature names and their corresponding weights
        feature_weights = list(zip(feature_names, weights))

        # Sort the features by the absolute value of their weights
        sorted_features = sorted(feature_weights, key=lambda x: abs(x[1]), reverse=True)

        # Print the sorted weights
        print("Logistic Regression Coefficients (ranked by absolute weight):")
        for feature, weight in sorted_features:
            print(f"{feature}: {weight:.4f}")

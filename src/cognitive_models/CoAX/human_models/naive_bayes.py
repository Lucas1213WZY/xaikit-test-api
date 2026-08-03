from sklearn.naive_bayes import GaussianNB
import numpy as np

class NaiveBayesModel:
    """A class to handle a Naive Bayes model for classification tasks."""

    def __init__(self, var_smoothing=1e-9, temperature=1.0, **kwargs):
        """
        Initialize the Naive Bayes classifier.

        Parameters:
        - var_smoothing (float): Portion of the largest variance of all features
          used to initialize the variances. This is added to variances for
          stability.
        - temperature (float): Controls the sharpness of the output probabilities.
          A higher temperature makes the predictions more uniform, while a lower
          temperature increases confidence (sharper predictions).
        - **kwargs: Additional keyword arguments for GaussianNB.
        """
        self.model = GaussianNB(var_smoothing=var_smoothing, **kwargs)
        self.features = []
        self.labels = []
        self.known_classes = set()  # Track known classes in the dataset
        self.temperature = temperature  # Sharpness control

    def add_exemplar(self, features, label):
        """
        Add a training example to the model.

        Parameters:
        - features (list or array): The feature vector.
        - label (int or str): The corresponding label.
        """
        self.features.append(features)
        self.labels.append(label)
        self.known_classes.add(label)

    def train(self):
        """Train the Naive Bayes model using the accumulated training data."""
        if len(self.features) == 1:
            # Duplicate the single exemplar with small noise to prevent zero variance error
            noise = np.random.normal(0, 1e-5, size=len(self.features[0]))
            duplicate_features = np.array(self.features[0]) + noise
            self.features.append(duplicate_features)
            self.labels.append(self.labels[0])  # Duplicate the label as well
        
        if len(self.features) > 0 and len(self.labels) > 0:
            X = np.array(self.features)
            y = np.array(self.labels)
            self.model.fit(X, y)
            self.known_classes = set(self.model.classes_)  # Update known classes based on the trained model
        else:
            raise ValueError("No training data available. Add exemplars before training.")

    def infer(self, features):
        """
        Predict the probabilities for the given feature vector.

        Parameters:
        - features (list or array): The feature vector to predict.

        Returns:
        - dict: A dictionary with predicted probabilities for each class.
        """
        if not self.known_classes:
            # If no training has occurred, return uniform probabilities
            return {cls: 1.0 / len(self.known_classes) for cls in self.known_classes}

        # Predict probabilities
        probabilities = self.model.predict_proba([features])[0]

        # Apply temperature scaling for sharpness control
        scaled_probabilities = np.power(probabilities, 1 / self.temperature)
        
        sum_scaled = np.sum(scaled_probabilities)
        
        if sum_scaled == 0:
            # Handle zero sum by returning uniform probabilities
            scaled_probabilities = np.ones_like(scaled_probabilities) / len(scaled_probabilities)
        else:
            # Normalize the scaled probabilities
            scaled_probabilities /= sum_scaled

        # Map probabilities to the known classes
        class_probabilities = {cls: prob for cls, prob in zip(self.model.classes_, scaled_probabilities)}

        # Ensure all known classes are represented in the output
        for cls in self.known_classes:
            if cls not in class_probabilities:
                class_probabilities[cls] = 0.0

        return class_probabilities

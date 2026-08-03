import numpy as np


class GeneralizedContextModel:
    def __init__(self, sensitivity=1.0, recency_factor=0.0):
        """
        Initialize the Generalized Context Model with sensitivity and optional recency factor.

        Parameters:
        - sensitivity (float): The sensitivity parameter 'c' controlling the steepness of the similarity function.
        - recency_factor (float): Controls how much more weight is given to later exemplars.
                                  0 means no extra emphasis on later exemplars.
        """
        self.feature_dim = None  # Dynamically determined from the first exemplar
        self.attention_weights = None
        self.sensitivity = sensitivity  # Sensitivity parameter
        self.exemplars = []  # List to store exemplars (features and labels)
        self.recency_factor = recency_factor

    def _initialize_attention_weights(self, feature_dim):
        """Initialize attention weights when the first exemplar is added."""
        self.feature_dim = feature_dim
        self.attention_weights = np.ones(feature_dim) / feature_dim  # Equal attention weights

    def add_exemplar(self, features, label):
        """
        Add a new exemplar to the model.

        Parameters:
        - features (list or np.array): The feature vector of the exemplar.
        - label (str or int): The category label of the exemplar.
        """
        features = np.array(features)

        # Initialize attention weights dynamically based on the first exemplar
        if self.feature_dim is None:
            self._initialize_attention_weights(len(features))

        if len(features) != self.feature_dim:
            raise ValueError(f"Feature vector length ({len(features)}) does not match expected dimension ({self.feature_dim}).")

        self.exemplars.append({'features': features, 'label': label})

    def _calculate_similarity(self, exemplar_features, test_features):
        """
        Calculate the similarity between an exemplar and the test features using the sensitivity parameter.

        Parameters:
        - exemplar_features (np.array): Feature vector of the exemplar.
        - test_features (np.array): Feature vector of the test instance.

        Returns:
        - similarity (float): The computed similarity.
        """
        # Manhattan distance weighted by attention
        distance = np.sum(self.attention_weights * np.abs(exemplar_features - test_features))
        # Similarity is an exponential decay of the distance scaled by sensitivity
        similarity = np.exp(-self.sensitivity * distance)
        return similarity

    def infer(self, test_features):
        """
        Infer the probability distribution over categories for a new test exemplar.

        Parameters:
        - test_features (list or np.array): The feature vector of the test instance.

        Returns:
        - probabilities (dict): A probability distribution over categories.
        """
        if not self.exemplars:
            raise ValueError("No exemplars in the model. Add exemplars before inference.")

        test_features = np.array(test_features)

        if len(test_features) != self.feature_dim:
            raise ValueError(f"Test feature vector length ({len(test_features)}) does not match expected dimension ({self.feature_dim}).")

        # Number of exemplars
        n = len(self.exemplars)

        # Compute weights based on recency
        if n > 1 and self.recency_factor != 0:
            indices = np.arange(n)
            weights = (indices / (n - 1)) * self.recency_factor + 1.0
        else:
            weights = np.ones(n)  # No difference in weight

        # Calculate similarity to each exemplar and apply weight
        similarities = []
        labels = []
        for i, exemplar in enumerate(self.exemplars):
            sim = self._calculate_similarity(exemplar['features'], test_features)
            sim *= weights[i]  # Apply recency-based weight
            similarities.append(sim)
            labels.append(exemplar['label'])

        # Sum similarities for each category
        category_similarities = {}
        for sim, label in zip(similarities, labels):
            if label in category_similarities:
                category_similarities[label] += sim
            else:
                category_similarities[label] = sim

        # Convert similarities to probabilities
        total_similarity = sum(category_similarities.values())
        probabilities = {label: sim / total_similarity for label, sim in category_similarities.items()}

        return probabilities


if __name__ == "__main__":
    # Example usage
    gcm = GeneralizedContextModel(sensitivity=10, recency_factor=0.5)

    # Add exemplars
    gcm.add_exemplar([0.1, 0.2, 0.3], 'A')
    gcm.add_exemplar([0.4, 0.5, 0.6], 'B')
    gcm.add_exemplar([0.3, 0.25, 0.35], 'A')  # A later exemplar

    # Infer probabilities for a new test instance
    test_features = [0.4, 0.5, 0.35]
    probabilities = gcm.infer(test_features)
    print(probabilities)

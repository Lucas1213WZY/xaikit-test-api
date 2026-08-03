from sklearn.neighbors import KNeighborsClassifier
import numpy as np

class KNNModel:
    """A class to handle a K-Nearest Neighbors model for classification tasks with adaptive smoothing."""

    def __init__(self, n_neighbors=5, smoothing_factor=0.01, **kwargs):
        """
        Initialize the KNN model.

        Parameters:
        - n_neighbors (int): The number of neighbors to consider.
        - smoothing_factor (float): The smoothing factor to apply to class counts.
        """
        self.n_neighbors = n_neighbors
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors)
        self.features = []
        self.labels = []
        self.smoothing_factor = smoothing_factor
        self.known_classes = set()

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
        """Train the KNN model using the accumulated training data."""
        if len(self.features) > 0 and len(self.labels) > 0:
            X = np.array(self.features)
            y = np.array(self.labels)
            self.model.fit(X, y)
        else:
            raise ValueError("No training data available. Add exemplars before training.")

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
        
        # Ensure that n_neighbors is not greater than the number of training samples
        n_neighbors = min(self.n_neighbors, len(self.features))
        self.model.n_neighbors = n_neighbors
        
        # Get the class probabilities from KNN
        neighbors = self.model.kneighbors([features], return_distance=False)[0]
        neighbor_labels = [self.labels[i] for i in neighbors]
        
        # Count occurrences of each class in the neighbors
        class_counts = {cls: neighbor_labels.count(cls) for cls in self.known_classes}
        
        # Apply smoothing to avoid zero probabilities
        smoothed_counts = {cls: count + self.smoothing_factor for cls, count in class_counts.items()}
        
        # Normalize to get probabilities
        total_counts = sum(smoothed_counts.values())
        probabilities = {cls: count / total_counts for cls, count in smoothed_counts.items()}
        
        # Ensure all known classes are present in the output
        for cls in self.known_classes:
            if cls not in probabilities:
                probabilities[cls] = 0.0
        
        return probabilities

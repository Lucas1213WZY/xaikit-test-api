from sklearn.tree import DecisionTreeClassifier, plot_tree, export_graphviz
import numpy as np
import matplotlib.pyplot as plt
from graphviz import Source



class DecisionTreeModel:
    """A class to handle a decision tree model for classification tasks with adaptive smoothing and regularization."""

    def __init__(self, max_depth=None, smoothing_factor=0.01):
        """
        Initialize the Decision Tree model.

        Parameters:
        - max_depth (int or None): The maximum depth of the tree.
        - smoothing_factor (float): The base smoothing factor to apply to class counts in the leaf nodes.
        """
        self.model = DecisionTreeClassifier(max_depth=max_depth)
        self.features = []
        self.labels = []
        self.smoothing_factor = smoothing_factor
        self.known_classes = set()  # Track the known classes in the dataset

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
        """Train the decision tree using the accumulated training data."""
        if len(self.features) > 0 and len(self.labels) > 0:
            X = np.array(self.features)
            y = np.array(self.labels)
            self.model.fit(X, y)
        else:
            raise ValueError("No training data available. Add exemplars before training.")

    def infer(self, features):
        """
        Predict the probabilities for the given feature vector with adaptive smoothing applied to class counts.

        Parameters:
        - features (list or array): The feature vector to predict.

        Returns:
        - dict: A dictionary with smoothed predicted probabilities for each class.
        """
        if len(self.known_classes) == 0:
            # If no training has occurred, return uniform probabilities
            return {0: 0.5, 1: 0.5}

        # Find the leaf node for the input features
        leaf_index = self.model.apply([features])[0]

        # Get the class counts in that leaf node
        node_counts = self.model.tree_.value[leaf_index][0]

        # Ensure all known classes are accounted for in probabilities
        all_classes = sorted(self.known_classes)
        class_counts = {cls: node_counts[i] if i < len(node_counts) else 0 for i, cls in enumerate(all_classes)}

        # Compute the depth of the leaf node
        node_depth = self._get_node_depth(leaf_index)

        # Apply adaptive smoothing based on tree depth
        adaptive_smoothing_factor = self.smoothing_factor * (1 + node_depth)

        # Apply adaptive smoothing by adding the smoothing factor to each class count
        smoothed_counts = {cls: count + adaptive_smoothing_factor for cls, count in class_counts.items()}

        # Calculate the smoothed probabilities
        total_counts = sum(smoothed_counts.values())
        probabilities = {cls: count / total_counts for cls, count in smoothed_counts.items()}

        # Ensure all known classes are present in the output
        for cls in self.known_classes:
            if cls not in probabilities:
                probabilities[cls] = 0.0

        return probabilities

    def _get_node_depth(self, leaf_index):
        """Get the depth of a given node in the decision tree."""
        children_left = self.model.tree_.children_left
        children_right = self.model.tree_.children_right

        node_depth = 0
        current_node = 0

        # Traverse the tree to find the depth of the leaf node
        stack = [(0, 0)]  # (node_id, depth)
        while stack:
            node_id, depth = stack.pop()
            if node_id == leaf_index:
                node_depth = depth
                break
            if children_left[node_id] != children_right[node_id]:  # If not a leaf node
                stack.append((children_left[node_id], depth + 1))
                stack.append((children_right[node_id], depth + 1))

        return node_depth

    def plot_tree(self, feature_names=None, class_names=True, title="Decision Tree"):
        """Plot the trained decision tree."""
        plt.figure(figsize=(12, 8))
        plot_tree(self.model, feature_names=feature_names, class_names=class_names, filled=True, impurity=False)
        plt.title(title, fontsize=16)
        plt.show()

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class MLPModel:
    """
    A class to handle a feedforward neural network for classification tasks with adaptive smoothing.
    """
    def __init__(self, hidden_dim=32, smoothing_factor=0.01, epochs=30, learning_rate=0.001):
        """
        Initialize the MLP model.

        Parameters:
        - hidden_dim (int): The number of hidden units in the hidden layer (default: 32).
        - smoothing_factor (float): The base smoothing factor to apply to class probabilities.
        """
        self.smoothing_factor = smoothing_factor
        self.hidden_dim = hidden_dim
        self.known_classes = set()  # Track the known classes in the dataset
        self.input_dim = None       # Input dimension will be dynamically determined
        self.epochs = epochs

        self.model = None
        self.criterion = None
        self.optimizer = None

        self.features = []
        self.labels = []
        self.lr = learning_rate

    def _initialize_model(self):
        """
        Dynamically initialize the MLP model once the input dimension and number of classes are known.
        """
        if self.input_dim is None:
            raise ValueError("Input dimension is not set. Ensure data is added before initializing the model.")

        sorted_classes = sorted(self.known_classes)
        self.class_mapping = {cls: i for i, cls in enumerate(sorted_classes)}  # Map classes to indices
        num_classes = len(sorted_classes)

        self.model = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, num_classes),
        )
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def add_exemplar(self, features, label):
        """
        Add a training example to the model.

        Parameters:
        - features (list or array): The feature vector.
        - label (int): The corresponding label.
        """
        if self.input_dim is None:
            self.input_dim = len(features)

        self.features.append(features)
        self.labels.append(label)
        # If a *new* class is seen, add it and (optionally) re-initialize or re-train
        if label not in self.known_classes:
            self.known_classes.add(label)
            # Optional: re-initialize model if new classes appear
            if self.model is not None:
                self._initialize_model()  # Then re-train if desired

    def train(self):
        """
        Train the MLP model using the accumulated training data.
        """
        if len(self.features) == 0 or len(self.labels) == 0:
            raise ValueError("No training data available. Add exemplars before training.")

        # Make sure we have a model matching the current known classes
        if self.model is None:
            self._initialize_model()
        else:
            # Optional: If known_classes changed, re-init the model
            # and do a fresh training run
            current_num_outputs = len(self.model[-1].weight)
            if current_num_outputs != len(self.known_classes):
                self._initialize_model()

        # Convert labels to zero-based indices
        X = torch.tensor(self.features, dtype=torch.float32)
        y = torch.tensor([self.class_mapping[label] for label in self.labels], dtype=torch.long)

        self.model.train()
        for epoch in range(self.epochs):
            self.optimizer.zero_grad()
            outputs = self.model(X)
            loss = self.criterion(outputs, y)
            loss.backward()
            self.optimizer.step()

    def infer(self, features):
        """
        Predict the probabilities for the given feature vector with adaptive smoothing applied.
        """
        # 1) If no classes are known, return empty
        if len(self.known_classes) == 0:
            return {}

        # 2) If exactly one class is known, that class is always predicted
        if len(self.known_classes) == 1:
            single_class = next(iter(self.known_classes))
            return {single_class: 1.0}

        # 3) Otherwise, multi-class scenario
        if self.model is None:
            raise ValueError("The model has not been initialized. Train the model first.")

        # Ensure the model is consistent with current known classes
        sorted_classes = sorted(self.known_classes)
        class_mapping = {cls: i for i, cls in enumerate(sorted_classes)}

        self.model.eval()
        with torch.no_grad():
            input_tensor = torch.tensor([features], dtype=torch.float32)
            logits = self.model(input_tensor)
            probabilities = torch.softmax(logits, dim=1).numpy()[0]

        # 4) Apply smoothing & normalization
        smoothed_probabilities = {}
        for cls in sorted_classes:
            cls_idx = class_mapping[cls]
            smoothed_probabilities[cls] = probabilities[cls_idx] + self.smoothing_factor

        total_prob = sum(smoothed_probabilities.values())
        normalized_probabilities = {
            cls: prob / total_prob for cls, prob in smoothed_probabilities.items()
        }

        return normalized_probabilities

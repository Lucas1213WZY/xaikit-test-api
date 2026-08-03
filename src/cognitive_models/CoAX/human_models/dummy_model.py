class DummyModel:
    """A dummy model that always outputs the actual AI prediction."""
    
    def __init__(self):
        pass

    def add_exemplar(self, features, label):
        """Dummy method to match the interface of other models."""
        pass

    def infer(self, test_features, actual_ai_prediction):
        """
        Return a probability distribution where the actual AI prediction has probability 1.0.
        
        Parameters:
        - test_features (list or np.array): The feature vector (not used in this model).
        - actual_ai_prediction (str or int): The actual AI prediction.

        Returns:
        - probabilities (dict): A probability distribution with 1.0 for the actual AI prediction.
        """
        return {actual_ai_prediction: 1.0}

from abc import ABC, abstractmethod

class BaseEngine(ABC):
    def __init__(self, **kwargs):
        self.gradient_based = True

    @abstractmethod
    def predict(self, index=None):
        pass

    @abstractmethod
    def train(self, X, y, **args):
        pass

    @abstractmethod
    def save(self, file_name=None):
        pass


    @abstractmethod
    def load(self, file_load=None):
        pass
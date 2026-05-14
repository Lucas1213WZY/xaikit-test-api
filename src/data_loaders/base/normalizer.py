"""Abstract base class for feature normalizers."""

from abc import ABC, abstractmethod
from typing import List, Union
import numpy as np


class BaseNormalizer(ABC):
    """Abstract base class for feature normalization strategies."""

    def __init__(self, name: str = None):
        """
        Initialize the normalizer.
        
        Args:
            name: Display name for the normalizer
        """
        self.name = name or self.__class__.__name__

    @abstractmethod
    def normalize(self, value: Union[float, int], min_val: float, max_val: float) -> float:
        """
        Normalize a single feature value.
        
        Args:
            value: Raw feature value to normalize
            min_val: Minimum value boundary (strategy-dependent)
            max_val: Maximum value boundary (strategy-dependent)
            
        Returns:
            Normalized value
        """
        pass

    @abstractmethod
    def normalize_array(self, values: List, min_vals: List, max_vals: List) -> List:
        """
        Normalize an array of values.
        
        Args:
            values: List of raw feature values
            min_vals: List of minimum boundaries
            max_vals: List of maximum boundaries
            
        Returns:
            List of normalized values
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"

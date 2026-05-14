"""Abstract base class for data sources."""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd


class BaseDataSource(ABC):
    """Abstract base class for loading data from different sources (CoAX, CoXAM, custom)."""

    def __init__(self, source_type: str):
        """
        Initialize the data source.
        
        Args:
            source_type: Type of source (e.g., 'coax', 'coxam', 'custom')
        """
        self.source_type = source_type
        self.feature_values_df = None
        self.metadata_df = None
        self.ai_predictions_df = None
        self.explanation_data = None
        self._filters = {}

    @abstractmethod
    def load(self, **kwargs) -> None:
        """
        Load data from the source.
        
        Args:
            **kwargs: Source-specific loading parameters
        """
        pass

    @abstractmethod
    def get_features(self, instance_ids: List[int], normalize: bool = True) -> List[List[float]]:
        """
        Get normalized feature vectors for instances.
        
        Args:
            instance_ids: List of instance IDs to fetch
            normalize: Whether to normalize features
            
        Returns:
            List of feature vectors
        """
        pass

    @abstractmethod
    def get_predictions(self, instance_ids: List[int]) -> List[Any]:
        """
        Get AI model predictions for instances.
        
        Args:
            instance_ids: List of instance IDs
            
        Returns:
            List of predictions
        """
        pass

    @abstractmethod
    def get_explanations(self, instance_ids: List[int]) -> List[Dict[str, Any]]:
        """
        Get explanation data for instances.
        
        Args:
            instance_ids: List of instance IDs
            
        Returns:
            List of explanation dicts
        """
        pass

    @abstractmethod
    def add_filter(self, name: str, condition) -> 'BaseDataSource':
        """
        Add a filter condition.
        
        Args:
            name: Filter name
            condition: Filter function or condition
            
        Returns:
            Self for chaining
        """
        pass

    @abstractmethod
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the loaded data."""
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(source_type='{self.source_type}')"

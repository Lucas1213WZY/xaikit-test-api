"""Z-score (standardization) feature normalizer."""

from typing import List, Union
import numpy as np
from ..base.normalizer import BaseNormalizer


class ZScoreNormalizer(BaseNormalizer):
    """
    Z-score normalization (standardization).
    Normalizes values using mean and standard deviation.
    
    Formula: (value - mean) / std
    """

    def __init__(self, name: str = "ZScore"):
        super().__init__(name)

    def normalize(self, value: Union[float, int], mean: float, std: float) -> float:
        """
        Normalize a single value using z-score.
        
        Args:
            value: Raw feature value
            mean: Mean of the feature
            std: Standard deviation of the feature
            
        Returns:
            Z-score normalized value
        """
        if np.isnan(mean) or np.isnan(std) or std == 0:
            return float(value)
        
        return float((value - mean) / std)

    def normalize_array(self, values: List, means: List, stds: List) -> List:
        """
        Normalize multiple values using z-score.
        
        Args:
            values: List of raw values
            means: List of means per feature
            stds: List of standard deviations per feature
            
        Returns:
            List of z-score normalized values
        """
        return [
            self.normalize(v, means[i], stds[i])
            for i, v in enumerate(values)
        ]

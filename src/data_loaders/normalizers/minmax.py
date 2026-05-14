"""Min-Max (0-1) feature normalizer used in CoAX and CoXAM."""

from typing import List, Union
import numpy as np
from ..base.normalizer import BaseNormalizer


class MinMaxNormalizer(BaseNormalizer):
    """
    Min-Max normalization (0-1 scale).
    Used across CoAX and CoXAM frameworks.
    
    Normalizes values to [0, 1] range based on min/max boundaries:
    - Values < min_val -> 0
    - Values > max_val -> 1
    - Otherwise: (value - min_val) / (max_val - min_val)
    """

    def __init__(self, name: str = "MinMax"):
        super().__init__(name)

    def normalize(self, value: Union[float, int], min_val: float, max_val: float) -> float:
        """
        Normalize a single value to [0, 1].
        
        Args:
            value: Raw feature value
            min_val: Minimum boundary
            max_val: Maximum boundary
            
        Returns:
            Normalized value in [0, 1]
        """
        if np.isnan(min_val) or np.isnan(max_val):
            return float(value)
        
        if value < min_val:
            return 0.0
        elif value > max_val:
            return 1.0
        else:
            return float((value - min_val) / (max_val - min_val))

    def normalize_array(self, values: List, min_vals: List, max_vals: List) -> List:
        """
        Normalize multiple values element-wise.
        
        Args:
            values: List of raw values
            min_vals: List of min boundaries
            max_vals: List of max boundaries
            
        Returns:
            List of normalized values
        """
        return [
            self.normalize(v, min_vals[i], max_vals[i])
            for i, v in enumerate(values)
        ]

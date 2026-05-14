"""
CoAX-style exemplar-based memory backend.

Implements simple temporal decay activation model used by CoAX system.
All exemplars are stored with equal importance, activated by similarity and recency.
"""

from typing import List, Optional, Tuple, Any, Dict, Union
import numpy as np
from datetime import datetime
from .interface import MemoryInterface, Exemplar, ReasoningContext, MemoryBackend
from .utils import (
    euclidean_distance, cosine_similarity, temporal_decay, 
    get_timestamp_diff, normalize_probabilities
)


def _masked_norm(features: np.ndarray) -> float:
    """Compute vector norm while ignoring masked None/NaN dimensions."""
    arr = np.asarray(features, dtype=object)
    valid = []
    for value in arr:
        if value is None:
            valid.append(False)
            continue
        try:
            valid.append(not bool(np.isnan(value)))
        except (TypeError, ValueError):
            valid.append(True)

    if not any(valid):
        return 0.0
    return float(np.linalg.norm(arr[np.asarray(valid)].astype(float)))


class ExemplarMemory(MemoryInterface):
    """
    CoAX-style exemplar-based memory.
    
    Storage: Dictionary of exemplars with temporal decay
    Retrieval: Similarity-based + recency (time decay)
    Activation: Temporal decay * similarity weighting
    
    Key characteristics:
    - All exemplars equally weighted (single pool)
    - Simple temporal decay (no BLL, no associative strength)
    - Similarity computed via feature distance
    - External time management
    """
    
    def __init__(self, context: ReasoningContext):
        super().__init__(context)
        self.exemplars: Dict[str, Exemplar] = {}
        self.access_history: Dict[str, List[datetime]] = {}
        
        # Validate backend
        if self.context.backend != MemoryBackend.EXEMPLAR:
            raise ValueError(f"ExemplarMemory requires backend=EXEMPLAR, got {self.backend}")
    
    def store(self, key: str, value: Union[Exemplar, None]) -> None:
        """
        Store an exemplar.
        
        Args:
            key: Unique exemplar identifier
            value: Exemplar object
        """
        if not isinstance(value, Exemplar):
            raise TypeError(f"ExemplarMemory.store() expects Exemplar, got {type(value)}")
        
        self.exemplars[key] = value
        self.access_history[key] = [datetime.now()]
    
    def retrieve(self, query: Any, k: int = 1,
                 similarity_threshold: Optional[float] = None) -> List[Tuple[str, float, Exemplar]]:
        """
        Retrieve top-k exemplars most similar to query.
        
        Activation = temporal_decay(time) * similarity(query_features, exemplar_features)
        
        Args:
            query: Query exemplar or feature vector for matching
            k: Number of exemplars to retrieve
            similarity_threshold: Minimum activation to include
            
        Returns:
            List of (key, activation_score, exemplar) tuples, sorted by activation desc
        """
        if not self.exemplars:
            return []
        
        if isinstance(query, Exemplar):
            query_features = query.features
        elif isinstance(query, np.ndarray):
            query_features = query
        elif isinstance(query, dict) and 'features' in query:
            query_features = query['features']
        else:
            raise TypeError(f"Query must be Exemplar, ndarray, or dict with 'features', got {type(query)}")
        
        activations = []
        current_time = self.context.current_time or datetime.now()
        
        # Compute activation for each exemplar
        for key, exemplar in self.exemplars.items():
            # Time-based decay
            time_since = get_timestamp_diff(exemplar.timestamp, current_time)
            decay_weight = temporal_decay(time_since, self.context.decay_param)
            
            # Feature similarity
            if self.context.feature_similarity == "cosine":
                # Cosine: normalize features first
                q_norm = query_features / (np.linalg.norm(query_features) + 1e-10)
                e_norm = exemplar.features / (np.linalg.norm(exemplar.features) + 1e-10)
                sim = 1.0 - cosine_similarity(q_norm, e_norm)
            else:
                # Default: euclidean distance -> similarity
                dist = euclidean_distance(query_features, exemplar.features)
                # Normalize distance to [0, 1]: higher distance = lower similarity
                max_dist = _masked_norm(query_features) + _masked_norm(exemplar.features)
                sim = max(0.0, 1.0 - dist / (max_dist + 1e-10)) if max_dist > 0 else 1.0
            
            # Combined activation
            activation = decay_weight * sim
            
            # Apply threshold if specified
            if similarity_threshold is not None and activation < similarity_threshold:
                continue
            
            activations.append((key, activation, exemplar))
        
        # Sort by activation descending
        activations.sort(key=lambda x: x[1], reverse=True)
        
        # Return top-k
        return activations[:k]
    
    def retrieve_with_latency(self, query: Any, k: int = 1) -> Tuple[List[Tuple[str, Exemplar]], float]:
        """
        Retrieve exemplars with simulated latency.
        
        Note: CoAX doesn't have direct latency model; this provides compatibility.
        Latency is constant in exemplar retrieval (fast lookup).
        
        Returns:
            Tuple of ([(key, exemplar), ...], latency_ms)
        """
        retrieved = self.retrieve(query, k)
        # Simple exemplar lookup: constant latency ~1ms
        latency = 1.0
        result = [(key, exemplar) for key, _, exemplar in retrieved]
        return result, latency
    
    def get(self, key: str) -> Optional[Exemplar]:
        """Get a specific exemplar by key."""
        return self.exemplars.get(key)
    
    def update_activation(self, key: str, increase: float) -> None:
        """
        Update exemplar activation on retrieval (reinforcement).
        
        In exemplar-based system, this typically increases temporal weight
        by recording a new access timestamp (resetting decay).
        """
        if key in self.exemplars:
            # Record access
            if key not in self.access_history:
                self.access_history[key] = []
            self.access_history[key].append(datetime.now())
            
            # Optional: update exemplar's explicit activation field
            exemplar = self.exemplars[key]
            exemplar.activation = min(1.0, exemplar.activation + increase)
    
    def clear(self) -> None:
        """Clear all exemplars."""
        self.exemplars.clear()
        self.access_history.clear()
    
    def get_size(self) -> int:
        """Get number of stored exemplars."""
        return len(self.exemplars)
    
    def export_state(self) -> Dict[str, Any]:
        """Export memory state for inspection/debugging."""
        return {
            "backend": self.backend.value,
            "exemplars_count": len(self.exemplars),
            "exemplars": {
                key: {
                    "label": str(exemplar.label),
                    "features_shape": exemplar.features.shape,
                    "activation": float(exemplar.activation),
                    "temporal_decay": float(exemplar.temporal_decay),
                    "timestamp": exemplar.timestamp.isoformat() if exemplar.timestamp else None,
                    "accesses": len(self.access_history.get(key, []))
                }
                for key, exemplar in self.exemplars.items()
            },
            "context": {
                "decay_param": self.context.decay_param,
                "feature_similarity": self.context.feature_similarity
            }
        }
    
    def import_state(self, state: Dict[str, Any]) -> None:
        """Import memory state (restoration)."""
        # Note: Importing exemplar features requires access to original data
        # This is a simplified version that doesn't restore actual numpy arrays
        # Full restoration requires application-specific handling
        exemplars_data = state.get("exemplars", {})
        for key in exemplars_data:
            # Would need original features to fully restore
            pass
    
    def get_exemplars(self) -> Dict[str, Exemplar]:
        """Get all exemplars for inspection."""
        return dict(self.exemplars)
    
    def get_access_count(self, key: str) -> int:
        """Get number of times an exemplar has been accessed."""
        return len(self.access_history.get(key, []))
    
    def get_most_recent_access(self, key: str) -> Optional[datetime]:
        """Get timestamp of most recent access."""
        accesses = self.access_history.get(key, [])
        return accesses[-1] if accesses else None

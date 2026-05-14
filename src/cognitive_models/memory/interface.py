"""
Memory abstraction interface supporting both CoAX and CoXAM backends.

This module defines the abstract contracts that both exemplar-based (CoAX) 
and ACT-R-based (CoXAM) memory backends must implement, enabling unified
access to heterogeneous memory systems through a common interface.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from enum import Enum
import numpy as np
from datetime import datetime


class MemoryBackend(Enum):
    """Supported memory backend types."""
    EXEMPLAR = "exemplar"  # CoAX-style simple temporal decay
    ACTR = "actr"          # CoXAM-style probabilistic ACT-R


@dataclass
class Exemplar:
    """Represents a CoAX-style exemplar in memory."""
    label: Union[int, str]
    features: np.ndarray
    label_probs: Dict[Union[int, str], float]
    explanation_vector: np.ndarray
    temporal_decay: float = 1.0
    activation: float = 1.0
    timestamp: Optional[datetime] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class Chunk:
    """Represents a CoXAM-style chunk in declarative memory."""
    chunk_id: str
    chunk_type: str  # e.g., 'feature-weight', 'decision-rule'
    slots: Dict[str, Any]
    creation_time: float
    reference_count: int = 0
    activations: List[float] = field(default_factory=list)
    prob_refreshes: List[Tuple[float, float]] = field(default_factory=list)

    def __post_init__(self):
        if not self.activations:
            self.activations = [1.0]

    def add_prob_refresh(self, time: float, p: float) -> None:
        """Record a probabilistic activation refresh at `time` with probability `p`."""
        if p <= 0.0:
            return
        self.prob_refreshes.append((float(time), max(0.0, min(1.0, float(p)))))


@dataclass
class ReasoningContext:
    """Configuration context passed to memory operations."""
    backend: MemoryBackend
    current_time: Optional[float] = None
    decay_param: float = 0.5
    retrieval_threshold: float = -np.inf
    latency_factor: float = 0.0
    latency_exponent: float = 1.0
    activation_noise: float = 0.0
    max_assoc_strength: float = 1.8
    mismatch_penalty: float = 1.5
    wm_capacity: int = 4
    feature_similarity: Optional[str] = None  # "euclidean", "cosine", None
    
    def get_backend_params(self) -> Dict[str, float]:
        """Extract backend-specific parameters."""
        if self.backend == MemoryBackend.EXEMPLAR:
            return {
                "decay_param": self.decay_param,
                "feature_similarity": self.feature_similarity
            }
        else:  # ACTR
            return {
                "decay_param": self.decay_param,
                "retrieval_threshold": self.retrieval_threshold,
                "latency_factor": self.latency_factor,
                "latency_exponent": self.latency_exponent,
                "activation_noise": self.activation_noise,
                "max_assoc_strength": self.max_assoc_strength,
                "mismatch_penalty": self.mismatch_penalty,
                "wm_capacity": self.wm_capacity,
            }


class MemoryInterface(ABC):
    """
    Abstract base class defining the unified memory contract.
    
    Any memory backend (exemplar or ACT-R) must implement this interface
    to be compatible with the unified memory system.
    """
    
    def __init__(self, context: ReasoningContext):
        """Initialize memory with reasoning context."""
        self.context = context
        self.backend = context.backend
        
    @abstractmethod
    def store(self, key: str, value: Union[Exemplar, Chunk]) -> None:
        """
        Store a memory item (exemplar or chunk).
        
        Args:
            key: Unique identifier for the memory item
            value: The exemplar or chunk to store
        """
        pass
    
    @abstractmethod
    def retrieve(self, query: Any, k: int = 1, 
                 similarity_threshold: Optional[float] = None) -> List[Tuple[str, float, Any]]:
        """
        Retrieve top-k memory items matching the query.
        
        Returns:
            List of (key, activation_score, item) tuples
        """
        pass
    
    @abstractmethod
    def retrieve_with_latency(self, query: Any, k: int = 1) -> Tuple[List[Tuple[str, Any]], float]:
        """
        Retrieve items and compute access latency.
        
        Returns:
            Tuple of ([(key, item), ...], latency_ms)
        """
        pass
    
    @abstractmethod
    def get(self, key: str) -> Optional[Union[Exemplar, Chunk]]:
        """Get a specific memory item by key."""
        pass
    
    @abstractmethod
    def update_activation(self, key: str, increase: float) -> None:
        """Update activation for a memory item (e.g., on successful retrieval)."""
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all memory items."""
        pass
    
    @abstractmethod
    def get_size(self) -> int:
        """Get number of items in memory."""
        pass
    
    @abstractmethod
    def export_state(self) -> Dict[str, Any]:
        """Export complete memory state for inspection/debugging."""
        pass
    
    @abstractmethod
    def import_state(self, state: Dict[str, Any]) -> None:
        """Import memory state (useful for replay/restoration)."""
        pass


class ActivationFunction(ABC):
    """Base class for activation computation strategies."""
    
    @abstractmethod
    def compute(self, memory_item: Union[Exemplar, Chunk], 
                query: Any, context: ReasoningContext) -> float:
        """Compute activation score for query/item pair."""
        pass


class SimilarityFunction(ABC):
    """Base class for similarity computation."""
    
    @abstractmethod
    def compute(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute similarity between two vectors."""
        pass

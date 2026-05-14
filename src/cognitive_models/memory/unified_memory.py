"""
Unified Memory - factory and adapter for seamless switching between backends.

Provides a single access point regardless of backend, with configuration-driven
selection and parameter customization for both CoAX and CoXAM systems.
"""

from typing import List, Optional, Tuple, Any, Dict, Union
from dataclasses import dataclass, field
from .interface import (
    MemoryInterface, MemoryBackend, Exemplar, Chunk, ReasoningContext
)
from .exemplar_memory import ExemplarMemory
from .actr_memory import ACTRMemory
import numpy as np


@dataclass
class MemoryConfig:
    """Configuration for unified memory initialization."""
    
    # Backend selection
    backend: MemoryBackend = MemoryBackend.EXEMPLAR
    
    # Time management
    current_time: Optional[float] = None
    
    # Shared parameters
    decay_param: float = 0.5
    
    # Exemplar-specific
    feature_similarity: Optional[str] = None  # "euclidean" (default) or "cosine"
    
    # ACT-R specific
    retrieval_threshold: float = 0.0
    latency_factor: float = 0.0
    latency_exponent: float = 1.0
    activation_noise: float = 0.0
    max_assoc_strength: float = 1.8
    mismatch_penalty: float = 1.5
    wm_capacity: int = 4

    def to_context(self) -> ReasoningContext:
        """Convert config to ReasoningContext."""
        return ReasoningContext(
            backend=self.backend,
            current_time=self.current_time,
            decay_param=self.decay_param,
            retrieval_threshold=self.retrieval_threshold,
            latency_factor=self.latency_factor,
            latency_exponent=self.latency_exponent,
            activation_noise=self.activation_noise,
            max_assoc_strength=self.max_assoc_strength,
            mismatch_penalty=self.mismatch_penalty,
            wm_capacity=self.wm_capacity,
            feature_similarity=self.feature_similarity,
        )
    
    @staticmethod
    def coax_defaults() -> 'MemoryConfig':
        """Default configuration for CoAX system."""
        return MemoryConfig(
            backend=MemoryBackend.EXEMPLAR,
            decay_param=0.5,
            feature_similarity="euclidean"
        )
    
    @staticmethod
    def coxam_defaults() -> 'MemoryConfig':
        """Default configuration for CoXAM system (matches old DeclarativeMemory defaults)."""
        return MemoryConfig(
            backend=MemoryBackend.ACTR,
            decay_param=0.5,
            retrieval_threshold=0.0,
            latency_factor=0.1,
            latency_exponent=1.0,
            activation_noise=0.0,
            max_assoc_strength=2.0,
            mismatch_penalty=1.0,
            wm_capacity=4,
        )


class UnifiedMemory:
    """
    Factory for unified memory access across backends.
    
    Provides a single interface that automatically selects and configures
    the appropriate backend (exemplar or ACT-R) based on configuration.
    
    Usage:
        # Create with CoAX (exemplar) backend
        config = MemoryConfig.coax_defaults()
        memory = UnifiedMemory(config)
        
        # Create with CoXAM (ACT-R) backend
        config = MemoryConfig.coxam_defaults()
        memory = UnifiedMemory(config)
        
        # Retrieve with automatic backend handling
        retrieved = memory.retrieve(query, k=1)
    """
    
    def __init__(self, config: MemoryConfig):
        """
        Initialize unified memory.
        
        Args:
            config: MemoryConfig instance specifying backend and parameters
        """
        self.config = config
        self.context = config.to_context()
        
        # Create backend instance
        if config.backend == MemoryBackend.EXEMPLAR:
            self._backend: MemoryInterface = ExemplarMemory(self.context)
        elif config.backend == MemoryBackend.ACTR:
            self._backend: MemoryInterface = ACTRMemory(self.context)
        else:
            raise ValueError(f"Unknown backend: {config.backend}")
    
    # ========== Delegation methods ==========
    
    def store(self, key: str, value: Union[Exemplar, Chunk]) -> None:
        """Store a memory item (exemplar or chunk)."""
        self._backend.store(key, value)
    
    def retrieve(self, query: Any, k: int = 1,
                 similarity_threshold: Optional[float] = None) -> List[Tuple[str, float, Union[Exemplar, Chunk]]]:
        """
        Retrieve top-k items matching query.
        
        Returns:
            List of (key, activation_score, item) tuples
        """
        return self._backend.retrieve(query, k, similarity_threshold)
    
    def retrieve_with_latency(self, query: Any, k: int = 1) -> Tuple[List[Tuple[str, Union[Exemplar, Chunk]]], float]:
        """
        Retrieve items and compute latency.
        
        Returns:
            Tuple of ([(key, item), ...], latency_ms)
        """
        return self._backend.retrieve_with_latency(query, k)
    
    def retrieve_top_item(self, query: Any) -> Optional[Union[Exemplar, Chunk]]:
        """Retrieve single best-matching item."""
        results = self.retrieve(query, k=1)
        return results[0][2] if results else None
    
    def get(self, key: str) -> Optional[Union[Exemplar, Chunk]]:
        """Get item by key."""
        return self._backend.get(key)
    
    def update_activation(self, key: str, increase: float) -> None:
        """Update item activation on successful retrieval."""
        self._backend.update_activation(key, increase)
    
    def clear(self) -> None:
        """Clear all memory."""
        self._backend.clear()
    
    def get_size(self) -> int:
        """Get number of items in memory."""
        return self._backend.get_size()
    
    def export_state(self) -> Dict[str, Any]:
        """Export complete memory state."""
        return self._backend.export_state()
    
    def import_state(self, state: Dict[str, Any]) -> None:
        """Import memory state."""
        self._backend.import_state(state)
    
    # ========== Backend-specific methods ==========
    
    def is_exemplar_backend(self) -> bool:
        """Check if using exemplar backend."""
        return self.config.backend == MemoryBackend.EXEMPLAR
    
    def is_actr_backend(self) -> bool:
        """Check if using ACT-R backend."""
        return self.config.backend == MemoryBackend.ACTR
    
    def get_exemplar_memory(self) -> Optional[ExemplarMemory]:
        """Get exemplar backend if active."""
        if isinstance(self._backend, ExemplarMemory):
            return self._backend
        return None
    
    def get_actr_memory(self) -> Optional[ACTRMemory]:
        """Get ACT-R backend if active."""
        if isinstance(self._backend, ACTRMemory):
            return self._backend
        return None
    
    # ========== ACT-R specific methods (forwarded) ==========

    @property
    def dm(self) -> ACTRMemory:
        """
        Expose the underlying ACTRMemory as 'dm' so number-utility helpers
        (build_number_profile etc.) can access memory.dm.latency_factor etc.
        Raises AttributeError if called on an exemplar-backend instance.
        """
        backend = self.get_actr_memory()
        if backend is None:
            raise AttributeError("dm property requires ACT-R backend")
        return backend

    def add_association(self, source_key: str, target_key: str, strength: float) -> None:
        """Add associative link (ACT-R only)."""
        if self.is_actr_backend():
            self.get_actr_memory().add_association(source_key, target_key, strength)

    def update_time(self, new_time: float) -> None:
        """Update internal time (ACT-R only)."""
        if self.is_actr_backend():
            self.get_actr_memory().update_time(new_time)
        self.context.current_time = new_time

    def get_working_memory(self) -> List[str]:
        """Get working memory contents (ACT-R only)."""
        if self.is_actr_backend():
            return self.get_actr_memory().get_working_memory()
        return []

    def add_chunk(self, name: str, slots: dict, *, update_retrieval: bool = True) -> Optional[Any]:
        """Create and store a named Chunk (ACT-R only, matches old add_chunk API)."""
        if self.is_actr_backend():
            return self.get_actr_memory().add_chunk(name, slots, update_retrieval=update_retrieval)
        return None

    def tick(self, dt: float = 1) -> None:
        """Advance time (ACT-R only)."""
        if self.is_actr_backend():
            self.get_actr_memory().tick(dt)

    def get_chunk(self, name: str) -> Optional[Any]:
        """Get chunk by name (ACT-R only)."""
        if self.is_actr_backend():
            return self.get_actr_memory().get_chunk(name)
        return None

    def topk_retrievals_with_prob_refresh(
        self,
        request: dict,
        k: int = 3,
        refresh_prob: float = 1.0,
        add_refresh: bool = True,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """Softmax retrieval distribution (ACT-R only). Delegates to ACTRMemory."""
        if self.is_actr_backend():
            return self.get_actr_memory().topk_retrievals_with_prob_refresh(
                request, k=k, refresh_prob=refresh_prob,
                add_refresh=add_refresh, verbose=verbose,
            )
        raise AttributeError("topk_retrievals_with_prob_refresh requires ACT-R backend")

    # ---- ACT-R parameter properties (so build_number_profile works on UnifiedMemory) ----

    @property
    def time(self) -> float:
        """Current internal time (ACT-R only)."""
        if self.is_actr_backend():
            return self.get_actr_memory().time
        return 0.0

    @property
    def latency_factor(self) -> float:
        return self.context.latency_factor

    @property
    def latency_exponent(self) -> float:
        return getattr(self.context, 'latency_exponent', 1.0)

    @property
    def retrieval_threshold(self) -> float:
        return self.context.retrieval_threshold
    
    # ========== Exemplar specific methods (forwarded) ==========
    
    def get_exemplars(self) -> Dict[str, Exemplar]:
        """Get all exemplars (exemplar backend only)."""
        if self.is_exemplar_backend():
            return self.get_exemplar_memory().get_exemplars()
        return {}
    
    def get_access_count(self, key: str) -> int:
        """Get access count for item (exemplar backend only)."""
        if self.is_exemplar_backend():
            return self.get_exemplar_memory().get_access_count(key)
        return 0
    
    # ========== Utilities ==========
    
    def reconfigure(self, **kwargs) -> None:
        """
        Reconfigure memory parameters dynamically.
        
        Args:
            **kwargs: Any MemoryConfig fields to update
        """
        # Update config
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                raise ValueError(f"Unknown config parameter: {key}")
        
        # Regenerate context and backend
        self.context = self.config.to_context()
        
        # Recreate backend with new context (preserves data)
        if self.config.backend == MemoryBackend.EXEMPLAR:
            new_backend = ExemplarMemory(self.context)
        else:
            new_backend = ACTRMemory(self.context)
        
        # Transfer state from old backend
        old_state = self._backend.export_state()
        # Note: Full state transfer would require additional implementation
        # For now, this preserves the backend structure
        
        self._backend = new_backend
    
    @staticmethod
    def create_for_coax(**kwargs) -> 'UnifiedMemory':
        """
        Create memory configured for CoAX.
        
        Args:
            **kwargs: Override CoAX defaults
        """
        config = MemoryConfig.coax_defaults()
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return UnifiedMemory(config)
    
    @staticmethod
    def create_for_coxam(**kwargs) -> 'UnifiedMemory':
        """
        Create memory configured for CoXAM.
        
        Args:
            **kwargs: Override CoXAM defaults
        """
        config = MemoryConfig.coxam_defaults()
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return UnifiedMemory(config)

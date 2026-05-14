"""
Memory package public API.

Import memory types from this package root:

    from src.cognitive_models.memory import UnifiedMemory, MemoryConfig, Exemplar
"""

from .interface import (
    MemoryInterface,
    MemoryBackend,
    Exemplar,
    Chunk,
    ReasoningContext,
    ActivationFunction,
    SimilarityFunction,
)
from .unified_memory import UnifiedMemory, MemoryConfig
from .exemplar_memory import ExemplarMemory
from .actr_memory import ACTRMemory
from .utils import (
    euclidean_distance,
    cosine_similarity,
    temporal_decay,
    base_level_learning,
    compute_retrieval_latency,
    normalize_probabilities,
    compute_chunk_similarity,
)

__all__ = [
    "UnifiedMemory",
    "MemoryConfig",
    "ExemplarMemory",
    "ACTRMemory",
    "MemoryInterface",
    "ActivationFunction",
    "SimilarityFunction",
    "Exemplar",
    "Chunk",
    "ReasoningContext",
    "MemoryBackend",
    "euclidean_distance",
    "cosine_similarity",
    "temporal_decay",
    "base_level_learning",
    "compute_retrieval_latency",
    "normalize_probabilities",
    "compute_chunk_similarity",
]

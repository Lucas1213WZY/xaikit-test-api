"""Base classes and abstract interfaces for the unified data loader system."""

from .normalizer import BaseNormalizer
from .data_source import BaseDataSource

__all__ = [
    "BaseNormalizer",
    "BaseDataSource",
]

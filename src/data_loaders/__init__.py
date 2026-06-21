"""
Unified data loader system for CoAX, CoXAM, and custom data sources.

A comprehensive, extensible API layer that unifies data loading, normalization, 
and filtering across different frameworks.

Key Components:
- UnifiedDataLoader: Main API for loading and accessing data
- Data Sources: Adapters for CoAX, CoXAM, and custom sources
- XAI Dataset Parser: External CSVs with instances, AI predictions, and explanations
- Normalizers: Min-Max, Z-Score, and custom feature normalization
- Filters: Composable filter builder for data queries

Example Usage:
    from src.data_loaders import UnifiedDataLoader
    
    # Load CoAX data
    loader = UnifiedDataLoader.from_coax(
        feature_file="assets/ai_datasets/coax/values.csv",
        metadata_file="assets/ai_datasets/coax/metadata.csv"
    )
    
    # Apply filters
    filter_builder = loader.filter().by_app("wine_quality")
    loader.apply_filter(filter_builder)
    
    # Get data
    features, predictions = loader.get_instances([1, 2, 3])
    
    # Use XAI methods from src.xai_adapter
    from src.xai_adapter import get_adapter_registry
    registry = get_adapter_registry()
"""

__version__ = "0.1.0"

# Core API
from .unified_loader import UnifiedDataLoader
from .xai_dataset import CognitiveExplanationRecord, XAIDatasetLoader, XAIDatasetParser
from .original_dataset import list_original_datasets, load_original_dataset

# Data sources
from .sources import CoAXDataSource, CoXAMDataSource

# Normalizers
from .normalizers import MinMaxNormalizer, ZScoreNormalizer

# Filters
from .filters import FilterBuilder

# Base classes
from .base import BaseDataSource, BaseNormalizer

__all__ = [
    # Core
    "UnifiedDataLoader",
    "XAIDatasetParser",
    "XAIDatasetLoader",
    "CognitiveExplanationRecord",
    "load_original_dataset",
    "list_original_datasets",
    
    # Data sources
    "CoAXDataSource",
    "CoXAMDataSource",
    
    # Normalizers
    "MinMaxNormalizer",
    "ZScoreNormalizer",
    
    # Filters
    "FilterBuilder",
    
    # Base classes
    "BaseDataSource",
    "BaseNormalizer",
]

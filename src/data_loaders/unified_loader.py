"""Unified data loader API - main entry point for CoAX, CoXAM, and custom sources."""

import os
from typing import Dict, List, Any, Optional, Union
import pandas as pd
import numpy as np

from .sources import CoAXDataSource, CoXAMDataSource
from .normalizers import MinMaxNormalizer, ZScoreNormalizer
from .filters import FilterBuilder
from .base import BaseDataSource, BaseNormalizer


class UnifiedDataLoader:
    """
    Unified data loader providing a single API for CoAX, CoXAM, and custom data sources.
    
    Features:
    - Plugin-based architecture for extensibility
    - Composable filters (by_app, by_participant, by_condition, etc.)
    - Built-in normalization (MinMax, ZScore, custom)
    - Loaded explanation-table access; XAI methods live in src.xai_adapter
    - Consistent interface across different data sources
    
    Example:
        # Load CoAX data
        loader = UnifiedDataLoader.from_coax(
            feature_file="assets/ai_datasets/coax/values.csv",
            metadata_file="assets/ai_datasets/coax/metadata.csv",
            prediction_file="assets/ai_datasets/coax/none.csv"
        )
        
        # Apply filters
        loader.filter().by_app("wine_quality").by_condition("LR")
        
        # Get data
        features, predictions = loader.get_instances([1, 2, 3])
        
        # Use XAI methods
        from src.xai_adapter import get_adapter_registry
        registry = get_adapter_registry()
        lr_exp = registry.create('logistic_regression', 
            explanation_df=lr_df, metadata_df=metadata_df, 
            app_id="wine_quality", model_name="mlp")
    """

    def __init__(self, data_source: BaseDataSource, 
                 normalizer: BaseNormalizer = None):
        """
        Initialize unified loader.
        
        Args:
            data_source: Instance of BaseDataSource (CoAXDataSource, CoXAMDataSource, etc.)
            normalizer: Feature normalizer (defaults to MinMaxNormalizer)
        """
        self.data_source = data_source
        self.normalizer = normalizer or MinMaxNormalizer()
        self.filter_builder = FilterBuilder()
        self.explanation_tables: Dict[str, pd.DataFrame] = {}

    # ===================== Factory Methods =====================

    @staticmethod
    def from_coax(feature_file: str, metadata_file: str,
                  prediction_file: str = None,
                  explanation_columns: List[str] = None,
                  normalizer: BaseNormalizer = None) -> 'UnifiedDataLoader':
        """
        Create loader from CoAX synthetic data files.
        
        Args:
            feature_file: Path to values.csv
            metadata_file: Path to metadata.csv
            prediction_file: Path to predictions.csv (optional)
            explanation_columns: Columns to use for explanations
            normalizer: Custom normalizer (optional)
            
        Returns:
            UnifiedDataLoader instance
        """
        # Load CSVs
        feature_df = pd.read_csv(feature_file)
        metadata_df = pd.read_csv(metadata_file)
        prediction_df = pd.read_csv(prediction_file) if prediction_file else None

        # Create data source
        data_source = CoAXDataSource(normalizer=normalizer)
        data_source.load(
            feature_values_df=feature_df,
            metadata_df=metadata_df,
            explanation_values_df=prediction_df,
            explanation_columns=explanation_columns or []
        )

        return UnifiedDataLoader(data_source, normalizer=normalizer)

    @staticmethod
    def from_coxam(feature_file: str, metadata_file: str,
                   participant_file: str = None,
                   prediction_file: str = None,
                   explanation_columns: List[str] = None,
                   normalizer: BaseNormalizer = None) -> 'UnifiedDataLoader':
        """
        Create loader from CoXAM experiment data files.
        
        Args:
            feature_file: Path to values.csv
            metadata_file: Path to metadata.csv
            participant_file: Path to participant trials CSV (optional)
            prediction_file: Path to predictions.csv (optional)
            explanation_columns: Columns to use for explanations
            normalizer: Custom normalizer (optional)
            
        Returns:
            UnifiedDataLoader instance
        """
        # Load CSVs
        feature_df = pd.read_csv(feature_file)
        metadata_df = pd.read_csv(metadata_file)
        participant_df = pd.read_csv(participant_file) if participant_file else None
        prediction_df = pd.read_csv(prediction_file) if prediction_file else None

        # Create data source
        data_source = CoXAMDataSource(normalizer=normalizer)
        data_source.load(
            feature_values_df=feature_df,
            metadata_df=metadata_df,
            ai_predictions_df=prediction_df,
            participant_data_df=participant_df,
            explanation_columns=explanation_columns or []
        )

        return UnifiedDataLoader(data_source, normalizer=normalizer)

    @staticmethod
    def from_custom(data_source: BaseDataSource,
                    normalizer: BaseNormalizer = None) -> 'UnifiedDataLoader':
        """
        Create loader from custom data source.
        
        Args:
            data_source: Instance implementing BaseDataSource
            normalizer: Custom normalizer (optional)
            
        Returns:
            UnifiedDataLoader instance
        """
        if not isinstance(data_source, BaseDataSource):
            raise TypeError("data_source must inherit from BaseDataSource")
        return UnifiedDataLoader(data_source, normalizer=normalizer)

    @staticmethod
    def from_assets(source: str,
                    assets_root: str = "assets",
                    app_id: str = None,
                    coax_explanation_type: str = "importance",
                    normalizer: BaseNormalizer = None) -> 'UnifiedDataLoader':
        """
        Create loader directly from standardized assets directory.

        Expected layout:
            assets/
              data/{coax|coxam}/{values.csv,metadata.csv,none.csv}
              explanations/coax/{attribution.csv,importance.csv}
              explanations/coxam/{decision_tree.csv,logistic_regression.csv}

        Args:
            source: Data source type ('coax' or 'coxam')
            assets_root: Root assets directory
            app_id: Optional dataId filter to keep only one dataset
            coax_explanation_type: For CoAX, one of {'attribution', 'importance'}
            normalizer: Optional custom normalizer

        Returns:
            UnifiedDataLoader instance with optional explanation tables attached
        """
        source = source.lower().strip()
        if source not in {"coax", "coxam"}:
            raise ValueError("source must be 'coax' or 'coxam'")

        data_dir = os.path.join(assets_root, "data", source)
        exp_dir = os.path.join(assets_root, "explanations", source)

        feature_file = os.path.join(data_dir, "values.csv")
        metadata_file = os.path.join(data_dir, "metadata.csv")
        prediction_file = os.path.join(data_dir, "none.csv")

        if source == "coax":
            explanation_file = os.path.join(exp_dir, f"{coax_explanation_type}.csv")
            if os.path.exists(explanation_file):
                prediction_file = explanation_file

            loader = UnifiedDataLoader.from_coax(
                feature_file=feature_file,
                metadata_file=metadata_file,
                prediction_file=prediction_file,
                explanation_columns=[]
            )

            if os.path.exists(explanation_file):
                explanation_df = loader.get_ai_predictions()
                explanation_columns = [
                    c for c in explanation_df.columns
                    if (c.startswith("a") and c.endswith("_i")) or c == "intercept"
                ]
                loader.data_source.explanation_columns = explanation_columns
                loader.explanation_tables[coax_explanation_type] = explanation_df.copy()

            other_type = "attribution" if coax_explanation_type == "importance" else "importance"
            other_file = os.path.join(exp_dir, f"{other_type}.csv")
            if os.path.exists(other_file):
                loader.explanation_tables[other_type] = pd.read_csv(other_file)

        else:
            loader = UnifiedDataLoader.from_coxam(
                feature_file=feature_file,
                metadata_file=metadata_file,
                prediction_file=prediction_file
            )

            dt_file = os.path.join(exp_dir, "decision_tree.csv")
            lr_file = os.path.join(exp_dir, "logistic_regression.csv")
            if os.path.exists(dt_file):
                loader.explanation_tables["decision_tree"] = pd.read_csv(dt_file)
            if os.path.exists(lr_file):
                loader.explanation_tables["logistic_regression"] = pd.read_csv(lr_file)

        if app_id:
            source_obj = loader.data_source
            source_obj.feature_values_df = source_obj.feature_values_df[
                source_obj.feature_values_df["dataId"] == app_id
            ].copy()
            source_obj.metadata_df = source_obj.metadata_df[
                source_obj.metadata_df["dataId"] == app_id
            ].copy()

            if source_obj.ai_predictions_df is not None and "dataId" in source_obj.ai_predictions_df.columns:
                source_obj.ai_predictions_df = source_obj.ai_predictions_df[
                    source_obj.ai_predictions_df["dataId"] == app_id
                ].copy()

            for key, table in list(loader.explanation_tables.items()):
                if "dataId" in table.columns:
                    loader.explanation_tables[key] = table[table["dataId"] == app_id].copy()

        if normalizer is not None:
            loader.normalizer = normalizer
            loader.data_source.normalizer = normalizer

        return loader

    # ===================== Core API =====================

    def filter(self) -> FilterBuilder:
        """
        Get filter builder for composable filtering.
        
        Returns:
            FilterBuilder instance for chaining
            
        Example:
            loader.filter().by_app("wine").by_condition("LR").apply(data)
        """
        return FilterBuilder()

    def apply_filter(self, filter_builder: FilterBuilder) -> None:
        """
        Apply a filter to the data source.
        
        Args:
            filter_builder: FilterBuilder with accumulated conditions
        """
        filtered_df = filter_builder.apply(self.data_source.feature_values_df)
        # Create a condition that selects these row indices
        indices = set(filtered_df.index)
        condition = lambda df: df.index.isin(indices)
        self.data_source.add_filter("composed", condition)

    def get_instances(self, instance_ids: List[int], 
                     normalize: bool = True) -> tuple:
        """
        Get features and predictions for instances.
        
        Args:
            instance_ids: List of instance IDs to retrieve
            normalize: Whether to apply normalization
            
        Returns:
            Tuple of (features, predictions)
        """
        features = self.data_source.get_features(instance_ids, normalize=normalize)
        predictions = self.data_source.get_predictions(instance_ids)
        return features, predictions

    def load_instances(self, instance_ids: List[int], normalize: bool = True) -> tuple:
        """
        Backward-compatible alias for legacy simulation code.

        Returns:
            Tuple of (features, predictions)
        """
        return self.get_instances(instance_ids, normalize=normalize)

    def get_features(self, instance_ids: List[int], normalize: bool = True) -> List[List[float]]:
        """Get feature vectors."""
        return self.data_source.get_features(instance_ids, normalize=normalize)

    def get_predictions(self, instance_ids: List[int]) -> List[Any]:
        """Get AI predictions."""
        return self.data_source.get_predictions(instance_ids)

    def get_explanations(self, instance_ids: List[int]) -> List[Dict[str, Any]]:
        """Get explanation data."""
        return self.data_source.get_explanations(instance_ids)

    # ===================== CoXAM-Specific Methods =====================

    def get_participant_trials(self, participant_id: int, phase: str = None) -> pd.DataFrame:
        """
        Get participant trial data (CoXAM only).
        
        Args:
            participant_id: Participant identifier
            phase: Optional phase filter
            
        Returns:
            DataFrame of trials
        """
        if not isinstance(self.data_source, CoXAMDataSource):
            raise AttributeError("Participant trials only available for CoXAM data source")
        return self.data_source.get_participant_trials(participant_id, phase)

    def get_participant_ids(self) -> List[int]:
        """Get all participant IDs (CoXAM only)."""
        if not isinstance(self.data_source, CoXAMDataSource):
            raise AttributeError("Participants only available for CoXAM data source")
        return self.data_source.get_participant_ids()

    # ===================== Data Summary & Inspection =====================

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the loaded and filtered data."""
        return self.data_source.get_summary()

    def get_metadata(self) -> pd.DataFrame:
        """Get the metadata DataFrame."""
        return self.data_source.metadata_df

    def get_feature_values(self) -> pd.DataFrame:
        """Get the feature values DataFrame."""
        return self.data_source.feature_values_df

    def get_ai_predictions(self) -> pd.DataFrame:
        """Get the AI predictions DataFrame."""
        return self.data_source.ai_predictions_df

    def list_apps(self) -> List[str]:
        """List all available app/dataset IDs."""
        if self.data_source.metadata_df is None:
            return []
        return self.data_source.metadata_df['dataId'].unique().tolist()

    def list_explanation_tables(self) -> List[str]:
        """List loaded explanation table names (assets mode)."""
        return sorted(self.explanation_tables.keys())

    def get_explanation_table(self, name: str) -> pd.DataFrame:
        """
        Get a raw explanation table loaded from assets.

        Args:
            name: Table key, e.g. 'importance', 'attribution', 'decision_tree', 'logistic_regression'

        Returns:
            Explanation DataFrame
        """
        if name not in self.explanation_tables:
            available = self.list_explanation_tables()
            raise ValueError(f"Explanation table '{name}' not available. Available: {available}")
        return self.explanation_tables[name]

    # ===================== Utility Methods =====================

    def __repr__(self) -> str:
        summary = self.get_summary()
        return (f"UnifiedDataLoader(source={summary.get('source_type')}, "
                f"instances={summary.get('n_instances')}, "
                f"apps={summary.get('n_apps')})")

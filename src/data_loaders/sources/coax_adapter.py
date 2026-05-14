"""CoAX data source adapter for synthetic data loading."""

import random
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from ..base.data_source import BaseDataSource
from ..normalizers.minmax import MinMaxNormalizer


class CoAXDataSource(BaseDataSource):
    """
    Data source adapter for CoAX synthetic data.
    Loads feature values, metadata, predictions, and explanations.
    
    From: src/coax/data_loader.py AIDatasetLoader
    """

    def __init__(self, normalizer=None):
        """
        Initialize CoAX data source.
        
        Args:
            normalizer: Optional normalizer (defaults to MinMaxNormalizer)
        """
        super().__init__(source_type='coax')
        self.normalizer = normalizer or MinMaxNormalizer()
        self.explanation_columns = []

    def load(self, feature_values_df: pd.DataFrame, metadata_df: pd.DataFrame,
             explanation_values_df: pd.DataFrame = None,
             explanation_columns: List[str] = None) -> None:
        """
        Load CoAX data.
        
        Args:
            feature_values_df: Features (v0, v1, v2, ...)
            metadata_df: Metadata with min-max bounds (v{i}_min, v{i}_max)
            explanation_values_df: Optional explanation values
            explanation_columns: Columns to extract from explanations
        """
        self.feature_values_df = feature_values_df
        self.metadata_df = metadata_df
        self.ai_predictions_df = explanation_values_df
        self.explanation_columns = explanation_columns or []
        self._validate_loaded_data()

    def _validate_loaded_data(self) -> None:
        """Validate that loaded data has required structure."""
        if self.feature_values_df is None:
            raise ValueError("feature_values_df must be loaded")
        if self.metadata_df is None:
            raise ValueError("metadata_df must be loaded")
        if 'instanceId' not in self.feature_values_df.columns:
            raise ValueError("feature_values_df must have 'instanceId' column")
        if 'appId' not in self.feature_values_df.columns:
            raise ValueError("feature_values_df must have 'appId' column")

    def get_features(self, instance_ids: List[int], normalize: bool = True) -> List[List[float]]:
        """
        Get feature vectors for instances.
        
        Args:
            instance_ids: List of instance IDs
            normalize: If True, apply min-max normalization
            
        Returns:
            List of feature vectors
        """
        scaled_features = []

        for instance_id in instance_ids:
            feature_row = self.feature_values_df[
                self.feature_values_df['instanceId'] == instance_id
            ]
            if feature_row.empty:
                raise ValueError(f"Instance {instance_id} not found")
            
            feature_row = feature_row.iloc[0]
            app_id = feature_row['appId']

            # Get metadata for this app
            app_metadata = self.metadata_df[self.metadata_df['appId'] == app_id]
            if app_metadata.empty:
                raise ValueError(f"No metadata for appId: {app_id}")
            app_metadata = app_metadata.iloc[0]

            # Extract features (v0, v1, v2, ...)
            scaled_row = []
            i = 0
            while True:
                val_col = f'v{i}'
                if val_col not in feature_row.index or pd.isna(feature_row[val_col]):
                    break

                value = feature_row[val_col]
                if normalize:
                    min_col = f'v{i}_min'
                    max_col = f'v{i}_max'
                    min_val = app_metadata.get(min_col, None) if min_col in app_metadata.index else None
                    max_val = app_metadata.get(max_col, None) if max_col in app_metadata.index else None
                    
                    if pd.isna(min_val) or pd.isna(max_val):
                        scaled_row.append(float(value))
                    else:
                        scaled_row.append(self.normalizer.normalize(value, min_val, max_val))
                else:
                    scaled_row.append(float(value))

                i += 1

            scaled_features.append(scaled_row)

        return scaled_features

    def get_predictions(self, instance_ids: List[int]) -> List[Any]:
        """Get AI predictions for instances."""
        predictions = []
        for instance_id in instance_ids:
            if self.ai_predictions_df is None or self.ai_predictions_df.empty:
                predictions.append(None)
            else:
                pred_row = self.ai_predictions_df[
                    self.ai_predictions_df['instanceId'] == instance_id
                ]
                if not pred_row.empty:
                    predictions.append(pred_row.iloc[0].get('pred', None))
                else:
                    predictions.append(None)
        return predictions

    def get_explanations(self, instance_ids: List[int]) -> List[Dict[str, Any]]:
        """Get explanation data for instances."""
        explanations = []
        for instance_id in instance_ids:
            if self.ai_predictions_df is None or self.ai_predictions_df.empty:
                explanations.append({})
            else:
                exp_row = self.ai_predictions_df[
                    self.ai_predictions_df['instanceId'] == instance_id
                ]
                if not exp_row.empty:
                    exp_row = exp_row.iloc[0]
                    # Normalize by i_max if present
                    i_max = exp_row.get('i_max', 1.0) if 'i_max' in exp_row.index else 1.0
                    explanation_dict = {}
                    for col in self.explanation_columns:
                        if col in exp_row.index:
                            val = exp_row[col]
                            explanation_dict[col] = val / i_max if pd.notna(val) and i_max != 0 else None
                    explanations.append(explanation_dict)
                else:
                    explanations.append({})
        return explanations

    def load_random(self, n_samples: int) -> tuple:
        """
        Load random instances.
        
        Args:
            n_samples: Number of random instances
            
        Returns:
            Tuple of (features, predictions, explanations)
        """
        available_ids = self.feature_values_df['instanceId'].unique().tolist()
        if len(available_ids) < n_samples:
            raise ValueError(f"Only {len(available_ids)} instances available, need {n_samples}")
        
        selected_ids = random.sample(available_ids, n_samples)
        features = self.get_features(selected_ids, normalize=True)
        predictions = self.get_predictions(selected_ids)
        explanations = self.get_explanations(selected_ids)
        return features, predictions, explanations

    def add_filter(self, name: str, condition) -> 'CoAXDataSource':
        """Add filter condition."""
        if not callable(condition):
            raise TypeError("Condition must be callable")
        
        # Apply filter to all DataFrames
        mask = condition(self.feature_values_df)
        self.feature_values_df = self.feature_values_df[mask]
        
        # Filter metadata based on remaining app IDs
        remaining_app_ids = self.feature_values_df['appId'].unique()
        self.metadata_df = self.metadata_df[
            self.metadata_df['appId'].isin(remaining_app_ids)
        ]
        
        # Filter predictions if present
        if self.ai_predictions_df is not None:
            self.ai_predictions_df = self.ai_predictions_df[
                self.ai_predictions_df['instanceId'].isin(
                    self.feature_values_df['instanceId']
                )
            ]
        
        self._filters[name] = condition
        return self

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of loaded data."""
        return {
            'source_type': self.source_type,
            'n_instances': len(self.feature_values_df),
            'n_features': self._count_features(),
            'n_apps': len(self.metadata_df),
            'app_ids': self.metadata_df['appId'].unique().tolist(),
            'has_predictions': self.ai_predictions_df is not None,
            'n_explanation_columns': len(self.explanation_columns),
            'filters_applied': len(self._filters)
        }

    def _count_features(self) -> int:
        """Count number of features in a row."""
        if len(self.feature_values_df) == 0:
            return 0
        sample_row = self.feature_values_df.iloc[0]
        i = 0
        while f'v{i}' in sample_row.index and pd.notna(sample_row[f'v{i}']):
            i += 1
        return i

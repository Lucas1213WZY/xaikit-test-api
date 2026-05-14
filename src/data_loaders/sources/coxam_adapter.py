"""CoXAM data source adapter for experiment data loading."""

from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from ..base.data_source import BaseDataSource
from ..normalizers.minmax import MinMaxNormalizer


class CoXAMDataSource(BaseDataSource):
    """
    Data source adapter for CoXAM experimental data.
    Loads features, metadata, participant trials, and explainer data.
    
    From: src/coxam/match_performance_to_participant_v0.3.ipynb + utils.py
    """

    def __init__(self, normalizer=None):
        """
        Initialize CoXAM data source.
        
        Args:
            normalizer: Optional normalizer (defaults to MinMaxNormalizer)
        """
        super().__init__(source_type='coxam')
        self.normalizer = normalizer or MinMaxNormalizer()
        self.explanation_columns = []
        self.participant_data_df = None
        self.explanation_data = {}

    def load(self, feature_values_df: pd.DataFrame, metadata_df: pd.DataFrame,
             ai_predictions_df: pd.DataFrame = None,
             participant_data_df: pd.DataFrame = None,
             explanation_columns: List[str] = None) -> None:
        """
        Load CoXAM data (experiments + features).
        
        Args:
            feature_values_df: Features DataFrame
            metadata_df: Metadata DataFrame
            ai_predictions_df: Optional AI predictions
            participant_data_df: Optional participant trial DataFrame
            explanation_columns: Columns to use for explanations
        """
        self.feature_values_df = feature_values_df
        self.metadata_df = metadata_df
        self.ai_predictions_df = ai_predictions_df
        self.participant_data_df = participant_data_df
        self.explanation_columns = explanation_columns or []
        self._validate_loaded_data()

    def _validate_loaded_data(self) -> None:
        """Validate the loaded data structure."""
        if self.feature_values_df is None:
            raise ValueError("feature_values_df must be loaded")
        if self.metadata_df is None:
            raise ValueError("metadata_df must be loaded")

    def get_features(self, instance_ids: List[int], normalize: bool = True) -> List[List[float]]:
        """Get normalized features for instances."""
        scaled_features = []

        for instance_id in instance_ids:
            feature_row = self.feature_values_df[
                self.feature_values_df['instanceId'] == instance_id
            ]
            if feature_row.empty:
                raise ValueError(f"Instance {instance_id} not found")
            
            feature_row = feature_row.iloc[0]
            app_id = feature_row.get('appId', feature_row.get('app_id'))

            # Get metadata
            app_metadata = self.metadata_df[
                self.metadata_df['appId'] == app_id
            ]
            if app_metadata.empty:
                raise ValueError(f"No metadata for appId: {app_id}")
            app_metadata = app_metadata.iloc[0]

            # Extract v0, v1, v2, ...
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
                    min_val = app_metadata.get(min_col) if min_col in app_metadata.index else None
                    max_val = app_metadata.get(max_col) if max_col in app_metadata.index else None
                    
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
        """Get AI predictions."""
        predictions = []
        for instance_id in instance_ids:
            if self.ai_predictions_df is None or self.ai_predictions_df.empty:
                predictions.append(None)
            else:
                pred_row = self.ai_predictions_df[
                    self.ai_predictions_df['instanceId'] == instance_id
                ]
                if not pred_row.empty:
                    predictions.append(pred_row.iloc[0].get('pred'))
                else:
                    predictions.append(None)
        return predictions

    def get_explanations(self, instance_ids: List[int]) -> List[Dict[str, Any]]:
        """Get explanation data."""
        explanations = []
        for instance_id in instance_ids:
            explanation_dict = {}
            if self.ai_predictions_df is not None:
                exp_row = self.ai_predictions_df[
                    self.ai_predictions_df['instanceId'] == instance_id
                ]
                if not exp_row.empty:
                    exp_row = exp_row.iloc[0]
                    for col in self.explanation_columns:
                        if col in exp_row.index:
                            explanation_dict[col] = exp_row[col]
            explanations.append(explanation_dict)
        return explanations

    def get_participant_trials(self, participant_id: int, 
                               phase: str = None) -> pd.DataFrame:
        """
        Get trials for a specific participant.
        
        Args:
            participant_id: Participant identifier
            phase: Optional phase filter ('forward', 'counterfactual', etc.)
            
        Returns:
            DataFrame of participant trials
        """
        if self.participant_data_df is None:
            raise ValueError("Participant data not loaded")

        # Find participant ID column (handle naming variations)
        id_cols = ['Participant Id', 'Participant ID', 'participant_id']
        id_col = None
        for col in id_cols:
            if col in self.participant_data_df.columns:
                id_col = col
                break
        
        if id_col is None:
            raise ValueError("No participant ID column found")

        trials = self.participant_data_df[self.participant_data_df[id_col] == participant_id]
        
        if phase is not None:
            phase_cols = ['Phase', 'phase']
            phase_col = None
            for col in phase_cols:
                if col in trials.columns:
                    phase_col = col
                    break
            
            if phase_col:
                trials = trials[trials[phase_col] == phase]

        return trials.sort_values('Trial Index') if 'Trial Index' in trials.columns else trials

    def get_participant_ids(self) -> List[int]:
        """Get list of all participant IDs."""
        if self.participant_data_df is None:
            raise ValueError("Participant data not loaded")
        
        id_cols = ['Participant Id', 'Participant ID', 'participant_id']
        for col in id_cols:
            if col in self.participant_data_df.columns:
                return self.participant_data_df[col].unique().tolist()
        
        raise ValueError("No participant ID column found")

    def add_filter(self, name: str, condition) -> 'CoXAMDataSource':
        """Add filter condition."""
        if not callable(condition):
            raise TypeError("Condition must be callable")
        
        # Apply to features
        mask = condition(self.feature_values_df)
        self.feature_values_df = self.feature_values_df[mask]
        
        # Filter metadata
        if self.metadata_df is not None:
            remaining_app_ids = self.feature_values_df['appId'].unique()
            self.metadata_df = self.metadata_df[
                self.metadata_df['appId'].isin(remaining_app_ids)
            ]
        
        # Filter predictions
        if self.ai_predictions_df is not None:
            self.ai_predictions_df = self.ai_predictions_df[
                self.ai_predictions_df['instanceId'].isin(
                    self.feature_values_df['instanceId']
                )
            ]
        
        # Filter participant trials if applicable
        if self.participant_data_df is not None:
            self.participant_data_df = self.participant_data_df[
                self.participant_data_df['Instance Id'].isin(
                    self.feature_values_df['instanceId']
                )
            ]
        
        self._filters[name] = condition
        return self

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of loaded data."""
        summary = {
            'source_type': self.source_type,
            'n_instances': len(self.feature_values_df),
            'n_features': self._count_features(),
            'n_apps': len(self.metadata_df),
            'app_ids': self.metadata_df['appId'].unique().tolist(),
            'has_predictions': self.ai_predictions_df is not None,
        }
        
        if self.participant_data_df is not None:
            summary['n_participants'] = len(self.get_participant_ids())
            summary['n_trials'] = len(self.participant_data_df)
        
        summary['filters_applied'] = len(self._filters)
        return summary

    def _count_features(self) -> int:
        """Count number of features per instance."""
        if len(self.feature_values_df) == 0:
            return 0
        sample_row = self.feature_values_df.iloc[0]
        i = 0
        while f'v{i}' in sample_row.index and pd.notna(sample_row[f'v{i}']):
            i += 1
        return i

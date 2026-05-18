"""Composable filter builder for data constraints."""

from typing import Any, Callable, Dict, List, Optional
import pandas as pd


class FilterBuilder:
    """
    Builder pattern for composable data filters.
    Enables chaining multiple filter conditions: loader.filter().by_app("wine").by_condition("LR")
    """

    def __init__(self, data_df: Optional[pd.DataFrame] = None):
        """
        Initialize filter builder.
        
        Args:
            data_df: Optional initial DataFrame to filter
        """
        self.data_df = data_df
        self.conditions: List[Callable] = []

    def by_app(self, app_id: str) -> 'FilterBuilder':
        """Filter by app/dataset ID."""
        self.conditions.append(lambda df: df[df.get('dataId', df.get('app_id')) == app_id].index)
        return self

    def by_participant(self, participant_id: int) -> 'FilterBuilder':
        """Filter by participant ID."""
        cols_to_check = ['Participant Id', 'Participant ID', 'participant_id']
        
        def condition(df):
            for col in cols_to_check:
                if col in df.columns:
                    return df[df[col] == participant_id].index
            return pd.Index([])
        
        self.conditions.append(condition)
        return self

    def by_condition(self, condition_name: str) -> 'FilterBuilder':
        """Filter by experimental condition (e.g., 'DT', 'LR')."""
        cols_to_check = ['Condition', 'condition', 'model_type']
        
        def condition(df):
            for col in cols_to_check:
                if col in df.columns:
                    return df[df[col] == condition_name].index
            return pd.Index([])
        
        self.conditions.append(condition)
        return self

    def by_model(self, model_name: str) -> 'FilterBuilder':
        """Filter by model name."""
        cols_to_check = ['Model', 'model', 'modelName', 'model_name']
        
        def condition(df):
            for col in cols_to_check:
                if col in df.columns:
                    return df[df[col] == model_name].index
            return pd.Index([])
        
        self.conditions.append(condition)
        return self

    def by_xai_type(self, xai_type: str) -> 'FilterBuilder':
        """Filter by XAI explanation type."""
        cols_to_check = ['XAIType', 'xai_type', 'XaiType']
        
        def condition(df):
            for col in cols_to_check:
                if col in df.columns:
                    return df[df[col] == xai_type].index
            return pd.Index([])
        
        self.conditions.append(condition)
        return self

    def by_phase(self, phase: str) -> 'FilterBuilder':
        """Filter by trial phase (e.g., 'forward', 'counterfactual')."""
        cols_to_check = ['Phase', 'phase']
        
        def condition(df):
            for col in cols_to_check:
                if col in df.columns:
                    return df[df[col] == phase].index
            return pd.Index([])
        
        self.conditions.append(condition)
        return self

    def by_custom(self, condition_func: Callable[[pd.DataFrame], pd.Index]) -> 'FilterBuilder':
        """
        Apply a custom filter function.
        
        Args:
            condition_func: Function that takes DataFrame and returns filtered indices
        """
        self.conditions.append(condition_func)
        return self

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all accumulated conditions to a DataFrame.
        
        Args:
            df: DataFrame to filter
            
        Returns:
            Filtered DataFrame
        """
        result_indices = set(range(len(df)))
        
        for condition in self.conditions:
            try:
                indices = condition(df)
                if isinstance(indices, pd.Index):
                    result_indices &= set(indices)
                else:
                    result_indices &= set(indices)
            except Exception:
                # Skip conditions that can't be applied to this df
                pass
        
        return df.iloc[sorted(list(result_indices))]

    def get_indices(self, df: pd.DataFrame) -> List[int]:
        """Get filtered row indices without returning the data."""
        result_indices = set(range(len(df)))
        
        for condition in self.conditions:
            try:
                indices = condition(df)
                if isinstance(indices, pd.Index):
                    result_indices &= set(indices)
                else:
                    result_indices &= set(indices)
            except Exception:
                pass
        
        return sorted(list(result_indices))

    def __repr__(self) -> str:
        return f"FilterBuilder(conditions={len(self.conditions)})"

"""Thin XAI wrappers around parsed dataset-backed explanations."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from src.data_loaders.xai_dataset import XAIDatasetParser

from .base import ArrayLike, XAIAdapter, XAIAdapterResult, ensure_2d


class PrecomputedCSVXAIMethod(XAIAdapter):
    """
    XAI method wrapper for precomputed CSV explanation vectors.

    Data parsing belongs to `src.data_loaders.XAIDatasetParser`; this class only
    adapts parsed explanation vectors to the common XAIAdapter `.explain(...)`
    result shape.
    """

    method_name = "precomputed_csv"

    def __init__(
        self,
        *,
        dataset: Optional[XAIDatasetParser] = None,
        csv_path: Optional[str] = None,
        dataframe: Any = None,
        target: int = 1,
        **dataset_kwargs,
    ):
        super().__init__(target=target)
        self.dataset = dataset or XAIDatasetParser(csv_path=csv_path, dataframe=dataframe, **dataset_kwargs)
        self.is_fitted = True

    @property
    def df(self):
        """Expose the parsed dataframe for compatibility with previous callers."""
        return self.dataset.df

    @property
    def instance_id_col(self):
        return self.dataset.instance_id_col

    @property
    def prediction_col(self):
        return self.dataset.prediction_col

    @property
    def feature_columns(self):
        return self.dataset.feature_columns

    @property
    def explanation_columns(self):
        return self.dataset.explanation_columns

    def get_features(self, instance_ids):
        """Return feature vectors for instance ids."""
        return self.dataset.get_features(instance_ids)

    def get_predictions(self, instance_ids):
        """Return stored AI predictions for instance ids."""
        return self.dataset.get_predictions(instance_ids)

    def get_explanations(self, instance_ids):
        """Return precomputed explanation vectors for instance ids."""
        return self.dataset.get_explanations(instance_ids)

    def get_records(self, instance_ids):
        """Return parsed records in the form cognitive models need."""
        return self.dataset.get_records(instance_ids)

    def explain(self, instances: ArrayLike) -> XAIAdapterResult:
        """
        Return explanation vectors for instance ids.

        Unlike model-backed adapters, `instances` here means CSV instance ids.
        Use `get_records(...)` when the cognitive model also needs the feature
        vector and stored AI prediction.
        """
        instance_ids = self.dataset.coerce_instance_ids(instances)
        values = ensure_2d(self.dataset.get_explanations(instance_ids))
        base_values = np.asarray(self.dataset.get_intercepts(instance_ids), dtype=float)
        return XAIAdapterResult(
            values=values,
            base_values=base_values,
            method=self.method_name,
            metadata={
                "instance_ids": instance_ids,
                "predictions": self.dataset.get_predictions(instance_ids),
                "feature_columns": self.dataset.feature_columns,
                "explanation_columns": self.dataset.explanation_columns,
            },
        )


# Backward-compatible name for the previous xai_adapter class.
CSVDatasetAdapter = PrecomputedCSVXAIMethod


__all__ = [
    "PrecomputedCSVXAIMethod",
    "CSVDatasetAdapter",
]

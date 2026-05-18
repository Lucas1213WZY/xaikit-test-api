"""CSV-backed XAI dataset parsing utilities."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np


def _is_missing(value: Any) -> bool:
    try:
        return bool(np.isnan(value))
    except (TypeError, ValueError):
        return value is None


def _load_dataframe(csv_path: Optional[str] = None, dataframe: Any = None):
    if dataframe is not None:
        return dataframe.copy()
    if csv_path is None:
        raise ValueError("Either csv_path or dataframe must be provided")
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("pandas is required for XAIDatasetParser. Install with: pip install pandas") from exc
    return pd.read_csv(csv_path)


def _sort_feature_like_columns(columns: Iterable[str], prefixes: Sequence[str]) -> List[str]:
    patterns = [re.compile(rf"^{re.escape(prefix)}(\d+)(?:_i)?$") for prefix in prefixes]

    def sort_key(column: str):
        for group_idx, pattern in enumerate(patterns):
            match = pattern.match(column)
            if match:
                return (group_idx, int(match.group(1)), column)
        return (len(patterns), 0, column)

    return sorted(columns, key=sort_key)


@dataclass
class CognitiveExplanationRecord:
    """One parsed row for cognitive-model consumption."""

    instance_id: Any
    features: np.ndarray
    ai_prediction: Any
    explanation: np.ndarray
    intercept: Optional[float] = None
    metadata: Dict[str, Any] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "features": self.features,
            "ai_prediction": self.ai_prediction,
            "explanation": self.explanation,
            "intercept": self.intercept,
            "metadata": self.metadata or {},
        }


class XAIDatasetParser:
    """
    Parser for CSV-provided instances, AI predictions, and explanations.

    This belongs in the data-loading layer: it knows how to read external
    dataset rows and expose features, AI predictions, and optional precomputed
    explanation vectors in a consistent shape.
    """

    def __init__(
        self,
        *,
        csv_path: Optional[str] = None,
        dataframe: Any = None,
        instance_id_col: str = "instanceId",
        prediction_col: str = "pred",
        feature_columns: Optional[Sequence[str]] = None,
        explanation_columns: Optional[Sequence[str]] = None,
        intercept_col: str = "intercept",
        normalize_explanations_by: Optional[str] = None,
        missing_explanation_strategy: str = "error",
        metadata_columns: Optional[Sequence[str]] = None,
    ):
        self.df = _load_dataframe(csv_path=csv_path, dataframe=dataframe)
        self.csv_path = csv_path
        self.instance_id_col = instance_id_col
        self.prediction_col = prediction_col
        self.intercept_col = intercept_col
        self.normalize_explanations_by = normalize_explanations_by
        self.missing_explanation_strategy = missing_explanation_strategy
        self.feature_columns = list(feature_columns) if feature_columns is not None else self._infer_feature_columns()
        self.explanation_columns = (
            list(explanation_columns)
            if explanation_columns is not None
            else self._infer_explanation_columns()
        )
        self.metadata_columns = list(metadata_columns) if metadata_columns is not None else self._infer_metadata_columns()
        self._validate_columns()

    @classmethod
    def from_csv(cls, csv_path: str, **kwargs) -> "XAIDatasetParser":
        """Create a parser from a CSV path."""
        return cls(csv_path=csv_path, **kwargs)

    @classmethod
    def from_dataframe(cls, dataframe: Any, **kwargs) -> "XAIDatasetParser":
        """Create a parser from an in-memory dataframe."""
        return cls(dataframe=dataframe, **kwargs)

    def _infer_feature_columns(self) -> List[str]:
        return _sort_feature_like_columns(
            [col for col in self.df.columns if re.match(r"^v\d+$", col)],
            prefixes=("v",),
        )

    def _infer_explanation_columns(self) -> List[str]:
        candidates = [
            col for col in self.df.columns
            if re.match(r"^a\d+_i$", col)
            or re.match(r"^a\d+$", col)
            or re.match(r"^attr\d+$", col)
            or re.match(r"^importance\d+$", col)
        ]
        return _sort_feature_like_columns(candidates, prefixes=("a", "attr", "importance"))

    def _infer_metadata_columns(self) -> List[str]:
        reserved = {
            self.instance_id_col,
            self.prediction_col,
            self.intercept_col,
            self.normalize_explanations_by,
            *self.feature_columns,
            *self.explanation_columns,
        }
        return [col for col in self.df.columns if col not in reserved]

    def _validate_columns(self) -> None:
        if self.instance_id_col not in self.df.columns:
            raise ValueError(f"CSV must include instance id column '{self.instance_id_col}'")
        if self.prediction_col not in self.df.columns:
            raise ValueError(f"CSV must include prediction column '{self.prediction_col}'")
        if not self.feature_columns:
            raise ValueError("No feature columns found. Pass feature_columns or include x0, x1, ... columns")
        if not self.explanation_columns and self.missing_explanation_strategy == "error":
            raise ValueError(
                "No explanation columns found. Pass explanation_columns or include a0_i, a1_i, ... columns"
            )
        if self.missing_explanation_strategy not in {"error", "zeros", "features"}:
            raise ValueError("missing_explanation_strategy must be one of: 'error', 'zeros', 'features'")
        missing = [
            col for col in [
                *self.feature_columns,
                *self.explanation_columns,
                self.normalize_explanations_by,
            ]
            if col is not None and col not in self.df.columns
        ]
        if missing:
            raise ValueError(f"CSV is missing required columns: {missing}")

    def _rows_for_instance_ids(self, instance_ids: Sequence[Any]):
        rows = []
        for instance_id in instance_ids:
            match = self.df[self.df[self.instance_id_col] == instance_id]
            if match.empty:
                raise ValueError(f"Instance {instance_id!r} not found in XAI dataset")
            rows.append(match.iloc[0])
        return rows

    def coerce_instance_ids(self, instances: Any) -> List[Any]:
        """Coerce a scalar or sequence of ids to a list."""
        if np.isscalar(instances):
            return [instances]
        if isinstance(instances, np.ndarray):
            if instances.ndim == 0:
                return [instances.item()]
            if instances.ndim == 1:
                return instances.tolist()
        if isinstance(instances, (str, bytes)):
            return [instances]
        return list(instances)

    def _explanation_for_row(self, row) -> np.ndarray:
        if not self.explanation_columns:
            if self.missing_explanation_strategy == "zeros":
                return np.zeros(len(self.feature_columns), dtype=float)
            if self.missing_explanation_strategy == "features":
                return np.asarray([row[col] for col in self.feature_columns], dtype=float)
        values = np.asarray([row[col] for col in self.explanation_columns], dtype=float)
        if self.normalize_explanations_by:
            denom = row[self.normalize_explanations_by]
            if not _is_missing(denom) and float(denom) != 0.0:
                values = values / float(denom)
        return values

    def _intercept_for_row(self, row) -> float:
        if self.intercept_col in row.index and not _is_missing(row[self.intercept_col]):
            return float(row[self.intercept_col])
        return 0.0

    def _metadata_for_row(self, row) -> Dict[str, Any]:
        return {col: row[col] for col in self.metadata_columns if col in row.index}

    def get_features(self, instance_ids: Sequence[Any]) -> np.ndarray:
        """Return feature vectors for instance ids."""
        rows = self._rows_for_instance_ids(instance_ids)
        return np.asarray([[row[col] for col in self.feature_columns] for row in rows], dtype=float)

    def get_predictions(self, instance_ids: Sequence[Any]) -> List[Any]:
        """Return stored AI predictions for instance ids."""
        return [row[self.prediction_col] for row in self._rows_for_instance_ids(instance_ids)]

    def get_explanations(self, instance_ids: Sequence[Any]) -> np.ndarray:
        """Return explanation vectors for instance ids."""
        return np.asarray([self._explanation_for_row(row) for row in self._rows_for_instance_ids(instance_ids)])

    def get_intercepts(self, instance_ids: Sequence[Any]) -> np.ndarray:
        """Return optional intercept/base values for instance ids."""
        return np.asarray([self._intercept_for_row(row) for row in self._rows_for_instance_ids(instance_ids)])

    def get_records(self, instance_ids: Sequence[Any]) -> List[CognitiveExplanationRecord]:
        """Return parsed records in the form cognitive models need."""
        records = []
        for row in self._rows_for_instance_ids(instance_ids):
            records.append(
                CognitiveExplanationRecord(
                    instance_id=row[self.instance_id_col],
                    features=np.asarray([row[col] for col in self.feature_columns], dtype=float),
                    ai_prediction=row[self.prediction_col],
                    explanation=self._explanation_for_row(row),
                    intercept=self._intercept_for_row(row),
                    metadata=self._metadata_for_row(row),
                )
            )
        return records


# Backward-readable alias for call sites that think in "loader" terms.
XAIDatasetLoader = XAIDatasetParser

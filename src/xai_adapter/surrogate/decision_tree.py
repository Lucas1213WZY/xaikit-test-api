"""Decision-tree surrogate method for CoXAM-style explanation tables."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from ..base import ArrayLike, XAIAdapterResult, ensure_2d
from .base import SurrogateMethod


class DecisionTreeSurrogateMethod(SurrogateMethod):
    """
    Decision-tree surrogate method for rules-vs-weights comparisons.

    The expected explanation table matches `assets/explanations/coxam/decision_tree.csv`.
    """

    method_name = "decision_tree"

    def __init__(
        self,
        *,
        explanation_df: Optional[pd.DataFrame] = None,
        metadata_df: Optional[pd.DataFrame] = None,
        app_id: Optional[str] = None,
        model_name: Optional[str] = None,
        depth: int = 3,
        random_state: int = 0,
        class_labels: Optional[List[Any]] = None,
        feature_names: Optional[List[str]] = None,
        target: int = 1,
        **tree_kwargs,
    ):
        super().__init__(target=target)
        self.app_id = app_id
        self.model_name = model_name
        self.depth = depth
        self.random_state = random_state
        self.class_labels = class_labels
        self.feature_names = feature_names
        self.tree_kwargs = tree_kwargs
        self.fidelity = None
        self.tree_structure = []
        self.nodes_by_id = {}
        self.explanation_df = None
        self.metadata_df = None
        self.explanation_row = None
        self.metadata_row = None
        if explanation_df is not None and metadata_df is not None:
            self.fit(explanation_df=explanation_df, metadata_df=metadata_df, app_id=app_id, model_name=model_name)

    def fit(
        self,
        X: ArrayLike = None,
        y: ArrayLike = None,
        *,
        explanation_df: Optional[pd.DataFrame] = None,
        metadata_df: Optional[pd.DataFrame] = None,
        app_id: Optional[str] = None,
        model_name: Optional[str] = None,
        depth: Optional[int] = None,
        mode: Optional[str] = None,
        random_state: Optional[int] = None,
        class_labels: Optional[List[Any]] = None,
        feature_names: Optional[List[str]] = None,
        **kwargs,
    ):
        """Fit from a precomputed table or generate a surrogate from X/y."""
        if explanation_df is None and metadata_df is None and y is not None:
            return self._fit_generated(
                X,
                y,
                app_id=app_id,
                model_name=model_name,
                depth=depth,
                random_state=random_state,
                class_labels=class_labels,
                feature_names=feature_names,
                **kwargs,
            )

        if mode == "generate":
            return self._fit_generated(
                X,
                y,
                app_id=app_id,
                model_name=model_name,
                depth=depth,
                random_state=random_state,
                class_labels=class_labels,
                feature_names=feature_names,
                **kwargs,
            )

        explanation_df = explanation_df if explanation_df is not None else X
        if explanation_df is None or metadata_df is None:
            raise ValueError("fit requires explanation_df and metadata_df, or X and y to generate a surrogate")

        self.app_id = app_id or self.app_id
        self.model_name = model_name or self.model_name
        self.depth = depth or self.depth
        if self.app_id is None or self.model_name is None:
            raise ValueError("app_id and model_name are required")

        row = explanation_df[explanation_df["dataId"] == self.app_id]
        if "model" in row.columns:
            row = row[row["model"] == self.model_name]
        if "depth" in row.columns:
            depth_row = row[row["depth"] == self.depth]
            if not depth_row.empty:
                row = depth_row
        if row.empty:
            raise ValueError(
                f"No decision-tree explanation found for dataId={self.app_id}, "
                f"model_name={self.model_name}, depth={self.depth}"
            )
        self.explanation_df = explanation_df.copy()
        self.explanation_row = row.iloc[0]

        meta_row = metadata_df[metadata_df["dataId"] == self.app_id]
        if meta_row.empty:
            raise ValueError(f"No metadata found for dataId={self.app_id}")
        self.metadata_df = metadata_df.copy()
        self.metadata_row = meta_row.iloc[0]

        self.fidelity = float(self.explanation_row.get("fidelity", 0.0))
        self.tree_structure = json.loads(self.explanation_row["tree_structure"])
        self.nodes_by_id = {node["node"]: node for node in self.tree_structure}
        self.class_labels = self._parse_class_labels()
        self.is_fitted = True
        return self

    def _fit_generated(
        self,
        X: ArrayLike,
        y: ArrayLike,
        *,
        app_id: Optional[str] = None,
        model_name: Optional[str] = None,
        depth: Optional[int] = None,
        random_state: Optional[int] = None,
        class_labels: Optional[List[Any]] = None,
        feature_names: Optional[List[str]] = None,
        **kwargs,
    ):
        """Generate a decision-tree surrogate from feature rows and AI predictions."""
        if X is None or y is None:
            raise ValueError("Generated decision-tree surrogate fitting requires X and y")
        from .generator import _metadata_from_data, generate_decision_tree_table

        self.app_id = app_id or self.app_id or "custom_dataset"
        self.model_name = model_name or self.model_name or "external_model"
        self.depth = depth or self.depth
        self.random_state = self.random_state if random_state is None else random_state
        if class_labels is not None:
            self.class_labels = class_labels
        if feature_names is not None:
            self.feature_names = feature_names
        tree_kwargs = {**self.tree_kwargs, **kwargs}
        self.metadata_df = _metadata_from_data(
            np.asarray(X, dtype=float),
            app_id=self.app_id,
            feature_names=self.feature_names,
        )
        self.explanation_df = generate_decision_tree_table(
            X,
            y,
            app_id=self.app_id,
            model_name=self.model_name,
            depths=(self.depth,),
            random_state=self.random_state,
            class_labels=self.class_labels,
            **tree_kwargs,
        )
        return self.fit(explanation_df=self.explanation_df, metadata_df=self.metadata_df)

    def _parse_class_labels(self):
        if "class_labels" not in self.explanation_row:
            return None
        try:
            return json.loads(self.explanation_row["class_labels"])
        except Exception:
            return None

    def _feature_index(self, feature_key: str) -> int:
        base = feature_key.split("=")[0]
        return int(base[1:])

    def _max_feature_index(self) -> int:
        indices = []
        for node in self.tree_structure:
            feature = node.get("feature")
            if feature:
                indices.append(self._feature_index(feature))
        return max(indices) if indices else -1

    def _traverse(self, raw_input: Union[List, np.ndarray]):
        raw_array = np.asarray(raw_input)
        node = self.nodes_by_id[0]
        path = []

        while not node["is_leaf"]:
            feature_key = node["feature"]
            col_idx = self._feature_index(feature_key)
            if "=" in feature_key:
                _base, cat_idx = feature_key.split("=")
                value = 1.0 if int(raw_array[col_idx]) == int(cat_idx) else 0.0
            else:
                value = float(raw_array[col_idx])

            went_left = value <= node["threshold"]
            path.append(
                {
                    "node": node["node"],
                    "feature": feature_key,
                    "feature_index": col_idx,
                    "threshold": node["threshold"],
                    "value": value,
                    "direction": "left" if went_left else "right",
                }
            )
            node = self.nodes_by_id[node["left"] if went_left else node["right"]]

        return node, path

    def predict(self, instances: ArrayLike) -> np.ndarray:
        """Return surrogate class predictions for raw feature vectors."""
        self._require_fitted()
        return np.asarray([self.apply(row)["class_index"] for row in ensure_2d(instances)], dtype=int)

    def apply(self, raw_input: Union[List, np.ndarray]) -> Dict[str, Any]:
        """Apply the tree surrogate to a single raw feature vector."""
        self._require_fitted()
        node, path = self._traverse(raw_input)
        class_index = int(np.argmax(node["value"]))
        return {
            "probs": node["value"],
            "class_index": class_index,
            "class_label": self.class_labels[class_index] if self.class_labels else None,
            "path": path,
        }

    def apply_batch(self, instances: List[Union[List, np.ndarray]]) -> List[Dict[str, Any]]:
        """Apply the tree surrogate to multiple instances."""
        return [self.apply(instance) for instance in instances]

    def explain(self, instances: ArrayLike) -> XAIAdapterResult:
        """
        Return a path-indicator explanation matrix for instances.

        Each feature used on the decision path receives 1.0. Other features are
        0.0. Detailed branch decisions are included in result metadata.
        """
        self._require_fitted()
        rows = ensure_2d(instances)
        n_features = max(rows.shape[1], self._max_feature_index() + 1)
        values = np.zeros((rows.shape[0], n_features), dtype=float)
        predictions = []
        paths = []

        for row_idx, raw_input in enumerate(rows):
            applied = self.apply(raw_input)
            predictions.append(applied)
            paths.append(applied["path"])
            for step in applied["path"]:
                values[row_idx, step["feature_index"]] = 1.0

        return XAIAdapterResult(
            values=values,
            base_values=np.zeros(rows.shape[0], dtype=float),
            method=self.method_name,
            metadata={
                "app_id": self.app_id,
                "model_name": self.model_name,
                "depth": self.depth,
                "fidelity": self.fidelity,
                "predictions": predictions,
                "paths": paths,
            },
        )

    def get_tree_structure(self) -> List[Dict[str, Any]]:
        """Return the underlying tree structure."""
        self._require_fitted()
        return self.tree_structure

    def to_explanation_table(self) -> pd.DataFrame:
        """Return the CoXAM-style decision-tree explanation table."""
        self._require_fitted()
        return self.explanation_df.copy()

    def to_metadata_table(self) -> pd.DataFrame:
        """Return the metadata table used by this surrogate."""
        self._require_fitted()
        return self.metadata_df.copy()

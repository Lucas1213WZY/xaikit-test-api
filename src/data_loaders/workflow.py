"""Notebook-friendly dataset preparation helpers."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .original_dataset import list_original_datasets, load_original_dataset


DEFAULT_DATASET_FEATURES: dict[str, list[str]] = {
    "adult": ["Age", "Capital Gain", "Marital Status", "Years of Education", "Sex"],
    "wine_quality": ["Alcohol", "pH", "Sulphates", "SO2", "Vinegar Taint"],
    "heart_disease": ["Stress Test", "Colored Vessels", "Chest Pain", "ST Slope", "Sex"],
    "king_county_housing": ["Living Room Sqft", "Above Sqft", "Latitude", "Grade", "Longitude"],
    "prima_diabetes": ["Glucose", "BMI", "DPF", "Age", "Pregnancies"],
    "mushrooms": ["Ring", "Height", "Width", "Bruises", "Cap Diameter"],
    "forest_cover": [
        "Elevation",
        "Horizontal_Distance_To_Roadways",
        "Soil_Type_28",
        "Hillshade_Noon",
        "Wilderness_Area_2",
    ],
}


def get_default_feature_cols(dataset_id: str) -> Optional[list[str]]:
    """Return the default tutorial feature subset for a dataset, if configured."""
    features = DEFAULT_DATASET_FEATURES.get(dataset_id.lower().strip())
    return list(features) if features is not None else None


@dataclass
class DatasetSplit:
    """Container for the dataset artifacts shared across workflow steps."""

    dataset_id: str
    dataset: Any
    df: pd.DataFrame
    X_raw: np.ndarray
    y: np.ndarray
    feature_names: list[str]
    raw_feature_names: list[str]
    model_feature_names: list[str]
    raw_instance_ids: np.ndarray
    X_model: np.ndarray
    X_raw_train: np.ndarray
    X_raw_test: np.ndarray
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    train_instance_ids: np.ndarray
    test_instance_ids: np.ndarray
    one_hot_encode: bool = True


@dataclass
class PreparedDataset:
    """Notebook-facing wrapper around a dataloader dataset and its split."""

    split: DatasetSplit

    @property
    def dataset_id(self) -> str:
        return self.split.dataset_id

    @property
    def dataset(self) -> Any:
        return self.split.dataset

    @property
    def df(self) -> pd.DataFrame:
        return self.split.df

    @property
    def feature_names(self) -> list[str]:
        return self.split.feature_names

    @property
    def raw_feature_names(self) -> list[str]:
        return self.split.raw_feature_names

    @property
    def model_feature_names(self) -> list[str]:
        return self.split.model_feature_names

    @property
    def X_train_raw(self) -> np.ndarray:
        return self.split.X_raw_train

    @property
    def X_test_raw(self) -> np.ndarray:
        return self.split.X_raw_test

    @property
    def X_train(self) -> np.ndarray:
        return self.split.X_train

    @property
    def X_test(self) -> np.ndarray:
        return self.split.X_test

    @property
    def y_train(self) -> np.ndarray:
        return self.split.y_train

    @property
    def y_test(self) -> np.ndarray:
        return self.split.y_test

    @property
    def train_instance_ids(self) -> np.ndarray:
        return self.split.train_instance_ids

    @property
    def test_instance_ids(self) -> np.ndarray:
        return self.split.test_instance_ids

    @property
    def label_column(self) -> str:
        return self.dataset.target_name or "target"


def prepare_dataset(
    dataset_id: str,
    *,
    model_type: str = "mlp",
    feature_cols: Optional[Sequence[str]] = None,
    num_features: Optional[int] = None,
    rank_features_by_target: bool = True,
    use_default_features: bool = True,
    requires_one_hot_encoding: Optional[bool] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    show_available: bool = True,
    show_summary: bool = True,
) -> PreparedDataset:
    """Load a dataset through `src.data_loaders`, split it, and return one wrapper."""
    if show_available:
        print("Available training datasets:", list_original_datasets())

    if requires_one_hot_encoding is None:
        from src.ai_models import requires_one_hot_encoding as model_requires_one_hot_encoding

        requires_one_hot_encoding = model_requires_one_hot_encoding(model_type)

    dataset = load_original_dataset(dataset_id)
    resolved_feature_cols = feature_cols
    if resolved_feature_cols is None and use_default_features:
        resolved_feature_cols = get_default_feature_cols(dataset_id)
    selected_features = _resolve_feature_selection(
        dataset,
        feature_cols=resolved_feature_cols,
        num_features=num_features,
        rank_features_by_target=rank_features_by_target,
    )
    if selected_features is not None:
        dataset = dataset.use_specific_features(selected_features)

    split = split_loaded_dataset(
        dataset_id,
        dataset,
        one_hot_encode=requires_one_hot_encoding,
        test_size=test_size,
        random_state=random_state,
    )

    if show_summary:
        print_dataset_split_summary(split)

    return PreparedDataset(split=split)


def _resolve_feature_selection(
    dataset: Any,
    *,
    feature_cols: Optional[Sequence[str]] = None,
    num_features: Optional[int] = None,
    rank_features_by_target: bool = True,
) -> Optional[list[str]]:
    """Validate, rank, and combine optional feature-list/count filters."""
    if num_features is not None and num_features <= 0:
        raise ValueError("num_features must be a positive integer.")

    available_features = list(dataset.feature_names)
    selected_features = list(feature_cols) if feature_cols is not None else list(available_features)

    missing_features = [feature for feature in selected_features if feature not in available_features]
    if missing_features:
        raise ValueError(
            f"Feature(s) not found: {missing_features}. "
            f"Available features: {list(available_features)}"
        )

    if rank_features_by_target:
        selected_features = _rank_features_by_target_correlation(dataset, selected_features)

    if num_features is not None:
        selected_features = selected_features[:num_features]

    if feature_cols is None and num_features is None and not rank_features_by_target:
        return None
    return selected_features


def _rank_features_by_target_correlation(dataset: Any, feature_names: Sequence[str]) -> list[str]:
    """Rank features by absolute Pearson correlation with the target."""
    y = pd.Series(np.asarray(dataset.y, dtype=float))
    ranked = []

    for original_position, feature_name in enumerate(feature_names):
        feature_idx = dataset.feature_names.index(feature_name)
        values = pd.to_numeric(pd.Series(dataset.X[:, feature_idx]), errors="coerce")
        valid = values.notna() & y.notna()
        if valid.sum() < 2 or values[valid].nunique() <= 1 or y[valid].nunique() <= 1:
            corr = 0.0
        else:
            corr = float(values[valid].corr(y[valid]))
            if np.isnan(corr):
                corr = 0.0
        ranked.append((feature_name, abs(corr), corr, original_position))

    ranked.sort(key=lambda item: (-item[1], item[3]))
    return [feature_name for feature_name, _abs_corr, _corr, _position in ranked]


def load_dataset_and_split(
    dataset_id: str,
    *,
    one_hot_encode: bool = True,
    test_size: float = 0.2,
    random_state: int = 42,
) -> DatasetSplit:
    """Load a tabular training dataset through `src.data_loaders` and split it."""
    dataset = load_original_dataset(dataset_id)
    return split_loaded_dataset(
        dataset_id,
        dataset,
        one_hot_encode=one_hot_encode,
        test_size=test_size,
        random_state=random_state,
    )


def split_loaded_dataset(
    dataset_id: str,
    dataset: Any,
    *,
    one_hot_encode: bool = True,
    test_size: float = 0.2,
    random_state: int = 42,
) -> DatasetSplit:
    """Split an already-loaded `src.data_loaders` dataset object."""
    X_model, y = dataset.prepare_data_for_model(one_hot_encode=one_hot_encode)
    X_raw = np.asarray(dataset.X, dtype=np.float32)
    X_model = np.asarray(X_model, dtype=np.float32)
    y = np.asarray(y)
    raw_instance_ids = np.arange(len(y))

    target_name = dataset.target_name or "target"
    df = pd.DataFrame(X_raw, columns=dataset.feature_names)
    df[target_name] = y

    stratify = y if len(np.unique(y)) > 1 else None
    train_instance_ids, test_instance_ids = train_test_split(
        raw_instance_ids,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )
    train_instance_ids = np.asarray(train_instance_ids)
    test_instance_ids = np.asarray(test_instance_ids)

    return DatasetSplit(
        dataset_id=dataset_id,
        dataset=dataset,
        df=df,
        X_raw=X_raw,
        y=y,
        feature_names=list(dataset.feature_names),
        raw_feature_names=list(dataset.feature_names),
        model_feature_names=_model_feature_names(dataset, one_hot_encode),
        raw_instance_ids=raw_instance_ids,
        X_model=X_model,
        X_raw_train=X_raw[train_instance_ids],
        X_raw_test=X_raw[test_instance_ids],
        X_train=X_model[train_instance_ids],
        X_test=X_model[test_instance_ids],
        y_train=y[train_instance_ids],
        y_test=y[test_instance_ids],
        train_instance_ids=train_instance_ids,
        test_instance_ids=test_instance_ids,
        one_hot_encode=one_hot_encode,
    )


def reencode_prepared_dataset(
    prepared: PreparedDataset,
    *,
    model_type: str = "mlp",
    requires_one_hot_encoding: Optional[bool] = None,
    show_summary: bool = True,
) -> PreparedDataset:
    """
    Rebuild only the model-facing arrays while preserving the original split.

    This lets a workflow choose the dataset before choosing the AI model. Raw
    rows, labels, and instance IDs remain fixed; only X_train/X_test change to
    match the encoding required by the selected model.
    """
    if requires_one_hot_encoding is None:
        from src.ai_models import requires_one_hot_encoding as model_requires_one_hot_encoding

        requires_one_hot_encoding = model_requires_one_hot_encoding(model_type)

    split = prepared.split
    X_model, y = split.dataset.prepare_data_for_model(one_hot_encode=requires_one_hot_encoding)
    X_model = np.asarray(X_model, dtype=np.float32)
    y = np.asarray(y)

    if len(y) != len(split.y):
        raise ValueError("Cannot re-encode dataset because the row count changed.")

    new_split = replace(
        split,
        y=y,
        X_model=X_model,
        X_train=X_model[split.train_instance_ids],
        X_test=X_model[split.test_instance_ids],
        y_train=y[split.train_instance_ids],
        y_test=y[split.test_instance_ids],
        model_feature_names=_model_feature_names(split.dataset, requires_one_hot_encoding),
        one_hot_encode=requires_one_hot_encoding,
    )

    if show_summary:
        print_dataset_split_summary(new_split)

    return PreparedDataset(split=new_split)


def _model_feature_names(dataset: Any, one_hot_encode: bool) -> list[str]:
    """Return feature names in the same order as prepare_instances_for_model."""
    raw_names = list(dataset.feature_names)
    categorical_options = getattr(dataset, "categorical_feature_options", {}) or {}
    if not one_hot_encode or not categorical_options:
        return raw_names

    names: list[str] = []
    for idx, feature_name in enumerate(raw_names):
        if idx in categorical_options:
            names.extend(f"{feature_name}={option}" for option in categorical_options[idx])
        else:
            names.append(feature_name)
    return names


def print_dataset_split_summary(split: DatasetSplit) -> None:
    """Print a compact check of the selected dataset and split."""
    print(f"Dataset   : {split.dataset_id}  ({split.X_model.shape[0]} rows, {split.X_model.shape[1]} model features)")
    print(f"Features  : {split.feature_names}")
    print(f"Encoding  : {'one-hot' if split.one_hot_encode else 'native'}")
    print(f"Train set : {split.X_train.shape[0]} samples  ({split.X_train.shape[0] / len(split.y) * 100:.0f}%)")
    print(f"Test set  : {split.X_test.shape[0]} samples  ({split.X_test.shape[0] / len(split.y) * 100:.0f}%)")
    for label in np.unique(split.y_train):
        print(f"Class balance (train) -> class {label}: {(split.y_train == label).sum()}")
    print(f"First test instanceIds: {split.test_instance_ids[:10].tolist()}")


def make_train_data_for_xai(split: DatasetSplit, y_train: np.ndarray) -> SimpleNamespace:
    """Package model-training data in the shape expected by XAI adapters."""
    return SimpleNamespace(
        X=split.X_train,
        y=y_train,
        feature_names=split.model_feature_names,
        raw_feature_names=split.raw_feature_names,
    )


def load_csv_records(path: str | Path) -> list[dict[str, Any]]:
    """Load a CSV as a list of dictionaries."""
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_json_config(path: str | Path) -> dict[str, Any]:
    """Load a JSON config exported by the UI."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

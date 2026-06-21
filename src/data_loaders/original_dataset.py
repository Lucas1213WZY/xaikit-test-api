"""Adapter for the original tabular datasets used by tutorial workflows."""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
from typing import Any


_DATASET_MODULES = {
    "adult": "adult",
    "wine_quality": "wine_quality",
    "heart_disease": "heart_disease",
    "king_county_housing": "king_county_housing",
    "prima_diabetes": "prima_diabetes",
    "mushrooms": "mushrooms",
    "forest_cover": "forest_cover",
    "breast_cancer": "breast_cancer",
    "cardiotocography": "cardiotocography",
}


def _install_local_datasets_alias() -> None:
    """Expose assets/original_datasets as the legacy `datasets` package."""
    repo_root = Path(__file__).resolve().parents[2]
    original_datasets_root = repo_root / "assets" / "original_datasets"

    datasets_module = types.ModuleType("datasets")
    datasets_module.__path__ = [str(original_datasets_root)]
    sys.modules["datasets"] = datasets_module


def load_original_dataset(dataset_id: str, **kwargs: Any) -> Any:
    """
    Load one of the repository's original tabular datasets.

    The dataset modules under `assets/original_datasets` predate the current
    package layout and import `datasets.tabular_dataset`. This adapter keeps
    that legacy import working while exposing the loaders through `src`.
    """
    dataset_key = dataset_id.lower().strip()
    if dataset_key not in _DATASET_MODULES:
        available = ", ".join(sorted(_DATASET_MODULES))
        raise ValueError(f"Unknown dataset_id {dataset_id!r}. Available: {available}")

    _install_local_datasets_alias()
    module = importlib.import_module(f"datasets.{_DATASET_MODULES[dataset_key]}.load")
    return module.load_data(**kwargs)


def list_original_datasets() -> list[str]:
    """Return the dataset ids supported by `load_original_dataset`."""
    return sorted(_DATASET_MODULES)

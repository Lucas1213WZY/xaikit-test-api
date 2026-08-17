"""Build a ``TabularDataset`` from an arbitrary user CSV or dataframe.

Every dataset ``prepare_dataset`` can load -- ``adult``, ``wine_quality``,
``mushrooms``, ... -- is a module under ``assets/original_datasets`` that
returns a ``TabularDataset`` (see e.g. ``assets/original_datasets/mushrooms
/load.py``). This module builds that same object generically from a caller's
own tabular data, so it flows through ``prepare_dataset``'s existing
feature-selection, one-hot-encoding and train/test-split pipeline exactly like
a bundled dataset -- no separate code path, and no bespoke handling needed
downstream in ``train_AI_model``/``explanations``/``generate_trials``.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn import preprocessing

from .original_dataset import _install_local_datasets_alias


def load_custom_dataset(
    *,
    csv_path: Optional[str] = None,
    dataframe: Optional[pd.DataFrame] = None,
    target_col: str,
    feature_cols: Optional[Sequence[str]] = None,
    categorical_cols: Optional[Sequence[str]] = None,
    positive_class: Optional[Any] = None,
    threshold: Optional[float] = None,
    dataset_name: str = "custom",
) -> Any:
    """Load a CSV/dataframe of raw feature rows into a ``TabularDataset``.

    Args:
        csv_path: Path to a CSV to read. Pass this or ``dataframe``, not both.
        dataframe: An in-memory dataframe to use directly.
        target_col: Column holding the label to predict. Dropped from the
            feature columns automatically.
        feature_cols: Use exactly these columns as features; every other
            non-target column by default.
        categorical_cols: Columns to treat as categorical. Columns of
            non-numeric dtype are detected automatically and do not need to
            be listed; pass this to also treat a numeric column (e.g. a
            small-integer code) as categorical.
        positive_class: Which target value maps to class 1; every other value
            maps to 0, one-vs-rest, however many distinct values the target
            has -- e.g. Iris' 3-class ``Species`` with
            ``positive_class="Iris-setosa"``. For an already-two-class target
            this defaults to whichever value is *not* the first row's value,
            matching how the bundled loaders pick a positive class (e.g.
            mushrooms' ``'p' -> 0``); pass it explicitly for a reproducible
            choice rather than relying on row order. Ignored when
            ``threshold`` is given.
        threshold: Binarize a continuous/ordinal ``target_col`` as
            ``target_col >= threshold`` -> class 1, the same way
            ``wine_quality`` turns its 0-10 ``quality`` score into a target
            with ``quality >= 7``. Use this when the raw target is not
            already two-valued (most real target columns aren't); leave unset
            for a target that already has exactly two distinct values.
        dataset_name: Recorded on the returned dataset for display.

    Returns:
        A ``TabularDataset``, the same object every bundled loader under
        ``assets/original_datasets`` returns.

    Raises:
        ValueError: If both or neither of ``csv_path``/``dataframe`` are
            given, if ``target_col`` is missing, or -- when ``threshold`` is
            not given -- if the target does not already have exactly two
            distinct values.
    """
    if (csv_path is None) == (dataframe is None):
        raise ValueError("Pass exactly one of csv_path or dataframe.")
    df = pd.read_csv(csv_path) if csv_path is not None else dataframe.copy()

    if target_col not in df.columns:
        raise ValueError(f"target_col {target_col!r} not found in columns: {list(df.columns)}")

    resolved_feature_cols = (
        list(feature_cols) if feature_cols is not None
        else [column for column in df.columns if column != target_col]
    )
    missing = [column for column in resolved_feature_cols if column not in df.columns]
    if missing:
        raise ValueError(f"feature_cols not found in columns: {missing}")

    X = df[resolved_feature_cols].copy()
    y_raw = df[target_col]

    categorical_cols = set(categorical_cols or ())
    categorical_indices = [
        index
        for index, column in enumerate(resolved_feature_cols)
        if column in categorical_cols or not pd.api.types.is_numeric_dtype(X[column])
    ]

    X_numpy = X.to_numpy(dtype=object)
    categorical_feature_options: dict[int, list[Any]] = {}
    for index in categorical_indices:
        encoder = preprocessing.LabelEncoder()
        column_values = pd.Series(X_numpy[:, index]).astype(str)
        encoder.fit(column_values)
        X_numpy[:, index] = encoder.transform(column_values)
        categorical_feature_options[index] = list(encoder.classes_)
    X_numpy = X_numpy.astype(float)

    if threshold is not None:
        y_numerical = (pd.to_numeric(y_raw) >= threshold).astype(int).to_numpy()
    elif positive_class is not None:
        # One-vs-rest: positive_class == this value -> 1, everything else
        # (however many other distinct values there are) -> 0. Iris'
        # Species has 3 classes, e.g. positive_class="Iris-setosa" -- there
        # is no "vs the other two" to name explicitly, so this treats
        # "not positive_class" as the negative class rather than requiring
        # the caller to enumerate it.
        y_numerical = np.where(y_raw.to_numpy() == positive_class, 1, 0)
    else:
        target_values = pd.unique(y_raw)
        if len(target_values) != 2:
            raise ValueError(
                f"target_col {target_col!r} has {len(target_values)} distinct value(s); "
                "pass positive_class=... for one-vs-rest (e.g. Iris's "
                "'Iris-setosa' vs the rest), or threshold=... to binarize a "
                "continuous/ordinal target (e.g. wine quality's quality >= 7)."
            )
        positive_class = next(value for value in target_values if value != target_values[0])
        y_numerical = np.where(y_raw.to_numpy() == positive_class, 1, 0)

    _install_local_datasets_alias()
    from datasets.tabular_dataset import TabularDataset

    return TabularDataset(
        X_numpy,
        y_numerical,
        feature_names=list(resolved_feature_cols),
        target_name=str(target_col),
        target_options=["Type 1", "Type 2"],
        categorical_feature_options=categorical_feature_options,
        dataset_name=dataset_name,
    )


__all__ = ["load_custom_dataset"]

"""Descriptive statistics helpers (aggregation, central tendency, dispersion)."""

from __future__ import annotations

from typing import List, Optional, Sequence, Union

import pandas as pd

ColumnSpec = Union[str, Sequence[str]]


def _as_list(cols: Optional[ColumnSpec]) -> Optional[List[str]]:
    if cols is None:
        return None
    if isinstance(cols, str):
        return [cols]
    return list(cols)


def describe(
    data: pd.DataFrame,
    value_cols: Optional[ColumnSpec] = None,
    group_cols: Optional[ColumnSpec] = None,
) -> pd.DataFrame:
    """Summary statistics (count, mean, std, min, quartiles, max).

    Equivalent to ``pandas.DataFrame.describe()``, optionally computed per
    group.

    Args:
        data: Source DataFrame.
        value_cols: Column(s) to summarize. Defaults to all numeric columns.
        group_cols: Optional column(s) to group by before summarizing.

    Returns:
        DataFrame of summary statistics.
    """
    value_cols = _as_list(value_cols)
    group_cols = _as_list(group_cols)
    frame = data if value_cols is None else data[list(group_cols or []) + value_cols]
    if group_cols:
        return frame.groupby(group_cols).describe()
    return frame.describe()


def aggregate(
    data: pd.DataFrame,
    group_cols: ColumnSpec,
    value_cols: ColumnSpec,
    funcs: Union[str, Sequence[str]] = ("mean", "median", "std", "sem", "count"),
) -> pd.DataFrame:
    """Group-wise aggregation over one or more value columns.

    Args:
        data: Source DataFrame.
        group_cols: Column(s) to group by.
        value_cols: Column(s) to aggregate.
        funcs: Aggregation function name(s), e.g. "mean", "median", "std",
            "sem", "var", "min", "max", "count". Anything accepted by
            ``DataFrame.groupby(...).agg(...)``.

    Returns:
        DataFrame indexed by ``group_cols`` with one column per
        (value_col, func) combination.
    """
    group_cols = _as_list(group_cols)
    value_cols = _as_list(value_cols)
    funcs = [funcs] if isinstance(funcs, str) else list(funcs)
    return data.groupby(group_cols)[value_cols].agg(funcs)


def mean(data: pd.DataFrame, value_cols: ColumnSpec, group_cols: Optional[ColumnSpec] = None):
    """Mean of ``value_cols``, optionally grouped by ``group_cols``."""
    return _reduce(data, value_cols, group_cols, "mean")


def median(data: pd.DataFrame, value_cols: ColumnSpec, group_cols: Optional[ColumnSpec] = None):
    """Median of ``value_cols``, optionally grouped by ``group_cols``."""
    return _reduce(data, value_cols, group_cols, "median")


def std(data: pd.DataFrame, value_cols: ColumnSpec, group_cols: Optional[ColumnSpec] = None, ddof: int = 1):
    """Sample standard deviation of ``value_cols``, optionally grouped."""
    return _reduce(data, value_cols, group_cols, "std", ddof=ddof)


def variance(data: pd.DataFrame, value_cols: ColumnSpec, group_cols: Optional[ColumnSpec] = None, ddof: int = 1):
    """Sample variance of ``value_cols``, optionally grouped."""
    return _reduce(data, value_cols, group_cols, "var", ddof=ddof)


def sem(data: pd.DataFrame, value_cols: ColumnSpec, group_cols: Optional[ColumnSpec] = None):
    """Standard error of the mean of ``value_cols``, optionally grouped."""
    return _reduce(data, value_cols, group_cols, "sem")


def _reduce(data: pd.DataFrame, value_cols: ColumnSpec, group_cols: Optional[ColumnSpec], func_name: str, **kwargs):
    value_cols = _as_list(value_cols)
    group_cols = _as_list(group_cols)
    if group_cols:
        grouped = data.groupby(group_cols)[value_cols]
        return getattr(grouped, func_name)(**kwargs)
    series_or_df = data[value_cols]
    return getattr(series_or_df, func_name)(**kwargs)

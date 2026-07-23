"""Post-hoc pairwise comparisons: pairwise t-tests, Bonferroni, Tukey HSD."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any, Optional, Sequence

import pandas as pd
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests


@dataclass
class PostHocResult:
    """Table of pairwise comparisons plus the correction method applied."""

    table: pd.DataFrame
    method: str


def pairwise_condition_tests(
    data: pd.DataFrame,
    *,
    value_col: str,
    condition_cols: Sequence[str],
    participant_col: str = "participantId",
    correction: Optional[str] = "holm",
) -> PostHocResult:
    """Compare every condition cell, pairing observations when participants overlap."""
    condition_cols = list(condition_cols)
    required = [participant_col, value_col, *condition_cols]
    missing = [column for column in required if column not in data]
    if missing:
        raise ValueError(f"Data is missing columns: {missing}.")

    clean = data[required].copy()
    clean[value_col] = pd.to_numeric(clean[value_col], errors="coerce")
    clean = clean.dropna(subset=required)
    participant_data = (
        clean.groupby(
            [participant_col, *condition_cols],
            as_index=False,
            dropna=False,
        )[value_col]
        .mean()
    )
    conditions = list(
        participant_data[condition_cols]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )

    rows: list[dict[str, Any]] = []
    for condition_a, condition_b in combinations(conditions, 2):
        sample_a = _condition_sample(
            participant_data,
            condition_cols,
            condition_a,
            participant_col,
            value_col,
        )
        sample_b = _condition_sample(
            participant_data,
            condition_cols,
            condition_b,
            participant_col,
            value_col,
        )
        paired_values = sample_a.merge(
            sample_b,
            on=participant_col,
            suffixes=("_a", "_b"),
        )
        if len(paired_values) >= 2:
            values_a = paired_values[f"{value_col}_a"]
            values_b = paired_values[f"{value_col}_b"]
            statistic, p_value = stats.ttest_rel(values_a, values_b)
            test = "paired_t"
        else:
            values_a = sample_a[value_col]
            values_b = sample_b[value_col]
            statistic, p_value = stats.ttest_ind(
                values_a,
                values_b,
                equal_var=False,
            )
            test = "welch_t"

        rows.append({
            "condition_a": _condition_label(condition_cols, condition_a),
            "condition_b": _condition_label(condition_cols, condition_b),
            "test": test,
            "n_a": len(values_a),
            "n_b": len(values_b),
            "mean_a": values_a.mean(),
            "mean_b": values_b.mean(),
            "mean_difference": values_a.mean() - values_b.mean(),
            "statistic": statistic,
            "p_value": p_value,
        })

    table = pd.DataFrame(rows)
    method = "none"
    if correction is not None and not table.empty:
        valid = table["p_value"].notna()
        table["p_value_corrected"] = float("nan")
        table["reject_null"] = False
        if valid.any():
            reject, adjusted, _, _ = multipletests(
                table.loc[valid, "p_value"],
                method=correction,
            )
            table.loc[valid, "p_value_corrected"] = adjusted
            table.loc[valid, "reject_null"] = reject
        method = correction
    return PostHocResult(table=table, method=method)


def _condition_sample(
    data: pd.DataFrame,
    columns: Sequence[str],
    levels: Sequence[Any],
    participant_col: str,
    value_col: str,
) -> pd.DataFrame:
    mask = pd.Series(True, index=data.index)
    for column, level in zip(columns, levels):
        mask &= data[column] == level
    return data.loc[mask, [participant_col, value_col]]


def _condition_label(columns: Sequence[str], levels: Sequence[Any]) -> str:
    return " | ".join(
        f"{column}={level}" for column, level in zip(columns, levels)
    )


def pairwise_ttests(
    data: pd.DataFrame,
    value_col: str,
    group_col: str,
    paired: bool = False,
    correction: Optional[str] = "bonferroni",
    equal_var: bool = True,
) -> PostHocResult:
    """Pairwise t-tests across all levels of ``group_col``.

    Args:
        data: Source DataFrame in long format.
        value_col: Numeric column being compared.
        group_col: Categorical column defining the groups to compare.
        paired: If True, run paired (related-samples) t-tests — requires
            each group to have the same number of rows in the same order.
        correction: Multiple-comparison correction to apply, passed to
            ``statsmodels.stats.multitest.multipletests`` (e.g.
            "bonferroni", "holm", "fdr_bh"). Pass ``None`` to skip
            correction and report raw p-values only.
        equal_var: Passed to ``scipy.stats.ttest_ind`` (ignored when
            ``paired=True``).

    Returns:
        PostHocResult with one row per group pair, raw and (optionally)
        corrected p-values.
    """
    groups = sorted(data[group_col].dropna().unique())
    rows = []
    for group_a, group_b in combinations(groups, 2):
        sample_a = data.loc[data[group_col] == group_a, value_col]
        sample_b = data.loc[data[group_col] == group_b, value_col]
        if paired:
            statistic, p_value = stats.ttest_rel(sample_a, sample_b)
        else:
            statistic, p_value = stats.ttest_ind(sample_a, sample_b, equal_var=equal_var)
        rows.append(
            {
                "group_a": group_a,
                "group_b": group_b,
                "n_a": len(sample_a),
                "n_b": len(sample_b),
                "mean_a": sample_a.mean(),
                "mean_b": sample_b.mean(),
                "statistic": statistic,
                "p_value": p_value,
            }
        )
    table = pd.DataFrame(rows)

    method = "none"
    if correction is not None and not table.empty:
        reject, p_adj, _, _ = multipletests(table["p_value"], method=correction)
        table["p_value_corrected"] = p_adj
        table["reject_null"] = reject
        method = correction

    return PostHocResult(table=table, method=method)


def bonferroni_correction(p_values: Sequence[float], alpha: float = 0.05) -> pd.DataFrame:
    """Apply Bonferroni correction to a list/array of p-values.

    Thin, explicit wrapper around
    ``statsmodels.stats.multitest.multipletests(method="bonferroni")`` for
    callers who already have p-values (e.g. from a custom set of tests)
    and just need the correction step.

    Args:
        p_values: Iterable of raw p-values.
        alpha: Family-wise significance level.

    Returns:
        DataFrame with columns: p_value, p_value_corrected, reject_null.
    """
    reject, p_adj, _, _ = multipletests(p_values, alpha=alpha, method="bonferroni")
    return pd.DataFrame({"p_value": p_values, "p_value_corrected": p_adj, "reject_null": reject})


def tukey_hsd(data: pd.DataFrame, value_col: str, group_col: str, alpha: float = 0.05) -> PostHocResult:
    """Tukey's HSD post-hoc test.

    An alternative to Bonferroni-corrected pairwise t-tests that accounts
    for all pairwise comparisons jointly, typically with more power than
    Bonferroni when comparing many groups.

    Args:
        data: Source DataFrame in long format.
        value_col: Numeric column being compared.
        group_col: Categorical column defining the groups to compare.
        alpha: Family-wise significance level.

    Returns:
        PostHocResult with the Tukey HSD summary table.
    """
    result = pairwise_tukeyhsd(endog=data[value_col], groups=data[group_col], alpha=alpha)
    summary_data = result.summary().data
    table = pd.DataFrame(summary_data[1:], columns=summary_data[0])
    return PostHocResult(table=table, method="tukey_hsd")

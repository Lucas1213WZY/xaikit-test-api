"""Small-multiple bar plots for every dependent-variable/IV pair."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd


@dataclass
class ResultGrid:
    """Figure, axes, and aggregated values used by an IV/DV plot grid."""

    figure: Any
    axes: np.ndarray
    summary: pd.DataFrame


@dataclass
class InteractionPlot:
    """Figure, axis, and aggregated values for one DV against two IVs."""

    figure: Any
    axis: Any
    summary: pd.DataFrame


def plot_iv_dv_grid(
    responses: pd.DataFrame,
    *,
    ivs: Sequence[str],
    dvs: Sequence[str],
    participant_column: str = "participantId",
    phase: Optional[str] = "testing",
    errorbar: Optional[str] = "sem",
    iv_levels: Optional[Mapping[str, Sequence[Any]]] = None,
    title: Optional[str] = "Experiment results",
    value_labels: bool = True,
) -> ResultGrid:
    """Plot participant-level mean DV bars for every DV × IV combination."""
    import matplotlib.pyplot as plt

    ivs = list(dict.fromkeys(ivs))
    dvs = list(dict.fromkeys(dvs))
    if not ivs:
        raise ValueError("Pass at least one independent variable in ivs.")
    if not dvs:
        raise ValueError("Pass at least one dependent variable in dvs.")
    required = [participant_column, *ivs, *dvs]
    missing = [column for column in required if column not in responses]
    if missing:
        raise ValueError(f"Response data is missing columns: {missing}.")
    if errorbar not in {None, "sem", "std"}:
        raise ValueError("errorbar must be one of: 'sem', 'std', or None.")

    data = responses.copy()
    if phase is not None and "phase" in data:
        data = data[data["phase"].astype(str).str.lower() == phase.lower()]
    if data.empty:
        raise ValueError(f"No {phase or ''} response rows are available to plot.")

    figure, axes = plt.subplots(
        len(dvs),
        len(ivs),
        figsize=(4.6 * len(ivs), 3.7 * len(dvs)),
        squeeze=False,
        constrained_layout=True,
    )
    summaries: list[pd.DataFrame] = []
    configured_levels = iv_levels or {}

    for row_index, dv in enumerate(dvs):
        numeric_dv = pd.to_numeric(data[dv], errors="coerce")
        for column_index, iv in enumerate(ivs):
            axis = axes[row_index, column_index]
            pair_data = data[[participant_column, iv]].copy()
            pair_data[dv] = numeric_dv
            pair_data = pair_data.dropna(subset=[participant_column, iv, dv])

            participant_data = (
                pair_data.groupby(
                    [participant_column, iv],
                    as_index=False,
                    dropna=False,
                )[dv]
                .mean()
            )
            summary = (
                participant_data.groupby(iv, as_index=False, dropna=False)[dv]
                .agg(["count", "mean", "std", "sem"])
                .reset_index()
            )
            summary.insert(0, "dv", dv)
            summary.insert(0, "iv", iv)
            summary = summary.rename(columns={iv: "level"})
            summaries.append(summary)

            if summary.empty:
                axis.text(
                    0.5,
                    0.5,
                    "No testing data",
                    ha="center",
                    va="center",
                    transform=axis.transAxes,
                )
                axis.set_axis_off()
                continue

            levels = _ordered_levels(
                summary["level"].tolist(),
                configured_levels.get(iv),
            )
            plotted = (
                summary.set_index("level")
                .reindex(levels)
                .reset_index()
            )
            errors = (
                None
                if errorbar is None
                else plotted[errorbar].fillna(0.0).to_numpy(dtype=float)
            )
            positions = np.arange(len(plotted))
            bars = axis.bar(
                positions,
                plotted["mean"].to_numpy(dtype=float),
                yerr=errors,
                capsize=4 if errors is not None else 0,
                color="C0",
                alpha=0.82,
            )
            axis.set_xticks(positions, [str(level) for level in plotted["level"]])
            axis.tick_params(axis="x", rotation=25)
            axis.set_xlabel(iv)
            axis.set_ylabel(dv)
            axis.set_title(f"{dv} by {iv}")
            axis.grid(axis="y", alpha=0.25)

            values = plotted["mean"].to_numpy(dtype=float)
            if len(values) and np.nanmin(values) >= 0 and np.nanmax(values) <= 1:
                axis.set_ylim(0, 1.05)
            if value_labels:
                axis.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)

    if title:
        figure.suptitle(title)
    summary_table = (
        pd.concat(summaries, ignore_index=True)
        if summaries
        else pd.DataFrame(
            columns=["iv", "dv", "level", "count", "mean", "std", "sem"]
        )
    )
    return ResultGrid(figure=figure, axes=axes, summary=summary_table)


def plot_dv_by_two_ivs(
    responses: pd.DataFrame,
    *,
    x_iv: str,
    hue_iv: str,
    dv: str,
    participant_column: str = "participantId",
    phase: Optional[str] = "testing",
    errorbar: Optional[str] = "sem",
    x_levels: Optional[Sequence[Any]] = None,
    hue_levels: Optional[Sequence[Any]] = None,
    x_labels: Optional[Mapping[Any, str]] = None,
    hue_labels: Optional[Mapping[Any, str]] = None,
    title: Optional[str] = None,
    value_labels: bool = True,
) -> InteractionPlot:
    """Plot participant-level mean DV bars using one IV for x and one for color."""
    import matplotlib.pyplot as plt

    required = [participant_column, x_iv, hue_iv, dv]
    missing = [column for column in required if column not in responses]
    if missing:
        raise ValueError(f"Response data is missing columns: {missing}.")
    if errorbar not in {None, "sem", "std"}:
        raise ValueError("errorbar must be one of: 'sem', 'std', or None.")

    data = responses.copy()
    if phase is not None and "phase" in data:
        data = data[data["phase"].astype(str).str.lower() == phase.lower()]
    data[dv] = pd.to_numeric(data[dv], errors="coerce")
    data = data.dropna(subset=[participant_column, x_iv, hue_iv, dv])
    if data.empty:
        raise ValueError(f"No {phase or ''} response rows are available to plot.")

    participant_data = (
        data.groupby(
            [participant_column, x_iv, hue_iv],
            as_index=False,
            dropna=False,
        )[dv]
        .mean()
    )
    summary = (
        participant_data.groupby([x_iv, hue_iv], as_index=False, dropna=False)[dv]
        .agg(["count", "mean", "std", "sem"])
        .reset_index()
        .rename(columns={x_iv: "x_level", hue_iv: "hue_level"})
    )
    summary.insert(0, "dv", dv)
    summary.insert(0, "hue_iv", hue_iv)
    summary.insert(0, "x_iv", x_iv)

    ordered_x = _ordered_levels(summary["x_level"].tolist(), x_levels)
    ordered_hue = _ordered_levels(summary["hue_level"].tolist(), hue_levels)
    figure, axis = plt.subplots(figsize=(max(6.4, 1.6 * len(ordered_x)), 4.2))

    group_positions = np.arange(len(ordered_x), dtype=float)
    width = 0.8 / max(1, len(ordered_hue))
    for hue_index, hue_level in enumerate(ordered_hue):
        means: list[float] = []
        errors: list[float] = []
        for x_level in ordered_x:
            row = summary[
                summary["x_level"].map(lambda value: _same_level(value, x_level))
                & summary["hue_level"].map(
                    lambda value: _same_level(value, hue_level)
                )
            ]
            means.append(float(row["mean"].iloc[0]) if not row.empty else np.nan)
            if errorbar is not None:
                error = float(row[errorbar].iloc[0]) if not row.empty else 0.0
                errors.append(0.0 if np.isnan(error) else error)

        positions = (
            group_positions
            - 0.4
            + width / 2
            + hue_index * width
        )
        bars = axis.bar(
            positions,
            means,
            width=width,
            yerr=None if errorbar is None else errors,
            capsize=4 if errorbar is not None else 0,
            label=_display_label(hue_level, hue_labels),
            alpha=0.82,
        )
        if value_labels:
            axis.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)

    axis.set_xticks(
        group_positions,
        [_display_label(level, x_labels) for level in ordered_x],
    )
    axis.set_xlabel(x_iv)
    axis.set_ylabel(dv)
    axis.set_title(title or f"{dv} by {x_iv} and {hue_iv}")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(title=hue_iv)

    values = summary["mean"].to_numpy(dtype=float)
    if len(values) and np.nanmin(values) >= 0 and np.nanmax(values) <= 1:
        axis.set_ylim(0, 1.05)
    figure.tight_layout()
    return InteractionPlot(figure=figure, axis=axis, summary=summary)


def _ordered_levels(
    observed: Sequence[Any],
    configured: Optional[Sequence[Any]],
) -> list[Any]:
    """Keep configured level order, followed by any unexpected observed levels."""
    observed = list(observed)
    if configured is None:
        return observed
    ordered = [
        level
        for level in configured
        if any(_same_level(level, value) for value in observed)
    ]
    ordered.extend(
        value
        for value in observed
        if not any(_same_level(value, level) for level in ordered)
    )
    return ordered


def _same_level(left: Any, right: Any) -> bool:
    try:
        return bool(left == right)
    except (TypeError, ValueError):
        return str(left) == str(right)


def _display_label(
    value: Any,
    labels: Optional[Mapping[Any, str]],
) -> str:
    if labels is not None:
        for level, label in labels.items():
            if _same_level(value, level):
                return str(label)
    return str(value)

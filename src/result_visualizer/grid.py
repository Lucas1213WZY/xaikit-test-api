"""Small-multiple bar plots for every dependent-variable/IV pair."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from .intervals import ci95_multiplier
from .labels import pretty
from .palette import categorical_color, level_color

#: Fraction of the space between adjacent x positions a bar actually fills.
#: Matplotlib's own default (1.0) leaves no gap between neighbouring bars,
#: which is what "too wide" meant here.
_BAR_WIDTH = 0.48


def _label_bars(
    axis: Any,
    bars: Any,
    values: np.ndarray,
    errors: Optional[np.ndarray] = None,
) -> None:
    """Draw a value label above (or below, if negative) each bar.

    A hand-rolled replacement for matplotlib's own ``Axes.bar_label``: that
    method crashes (``IndexError`` in matplotlib 3.5.3) whenever a bar's
    height is NaN alongside a non-None ``yerr`` -- an empty error-bar segment
    gets drawn for the missing bar, and ``bar_label`` indexes into it as if
    it had two points. NaN heights are routine here (an x/hue combination
    with no observed data), so this is a real, not merely theoretical, path.
    """
    for index, (bar, value) in enumerate(zip(bars, values)):
        if not np.isfinite(value):
            continue
        error = 0.0
        if errors is not None and np.isfinite(errors[index]):
            error = errors[index]
        height = bar.get_height()
        endpoint = height + error if height >= 0 else height - error
        axis.annotate(
            f"{value:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, endpoint),
            xytext=(0, 3 if height >= 0 else -3),
            textcoords="offset points",
            ha="center",
            va="bottom" if height >= 0 else "top",
            fontsize=8,
        )


def _despine(axis: Any) -> None:
    """Keep the left and bottom frame only.

    The top and right spines carry no information and box the data in; dropping
    them is the usual convention for a bar chart.
    """
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)


def _proportion_ylim(
    values: np.ndarray,
    errors: Optional[np.ndarray] = None,
    *,
    labelled: bool = True,
) -> Optional[tuple[float, float]]:
    """A 0-1 axis with room above the tallest bar, or None if not a proportion.

    A bar at exactly 1.000 -- a saturated condition, which happens -- reaches
    the top of a 0..1.05 axis, and its value label then sits outside the axes
    where it collides with the title. The headroom is measured from the top of
    the error bar so the label always has somewhere to go.
    """
    finite = values[np.isfinite(values)] if values.size else values
    if not finite.size or finite.min() < 0 or finite.max() > 1:
        return None

    top = float(finite.max())
    if errors is not None and errors.size:
        with np.errstate(invalid="ignore"):
            capped = values + np.where(np.isfinite(errors), errors, 0.0)
        capped = capped[np.isfinite(capped)]
        if capped.size:
            top = max(top, float(capped.max()))
    # Enough for the label's own line height above the tallest error bar.
    return (0.0, max(1.05, top + (0.10 if labelled else 0.04)))


def _finish_axis(
    axis: Any,
    values: np.ndarray,
    errors: Optional[np.ndarray] = None,
    *,
    labelled: bool = True,
) -> None:
    """Apply the shared bar-chart axis conventions."""
    _despine(axis)
    limits = _proportion_ylim(values, errors, labelled=labelled)
    if limits is not None:
        axis.set_ylim(*limits)


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
    errorbar: Optional[str] = "ci95",
    iv_levels: Optional[Mapping[str, Sequence[Any]]] = None,
    level_labels: Optional[Mapping[str, Mapping[Any, str]]] = None,
    title: Optional[str] = "Experiment results",
    value_labels: bool = True,
) -> ResultGrid:
    """Plot participant-level mean DV bars for every DV × IV combination.

    Args:
        responses: Responses to plot.
        ivs: Independent variables, one column of panels each.
        dvs: Dependent variables, one row of panels each.
        participant_column: Column identifying participants.
        phase: Restrict to one trial phase, e.g. ``testing``.
        errorbar: Error bar statistic -- ``ci95`` (Student-t 95% CI half-width,
            the default), ``sem``, ``std``, or None to hide them.
        iv_levels: Level order per IV.
        level_labels: Display names per IV level, e.g.
            ``{"tested_w_xai": {True: "w/ XAI", False: "w/o XAI"}}``. Without
            it a level is drawn as ``pretty(level)``, which renders a boolean
            factor as ``True``/``False`` -- the storage value, not the
            condition it stands for. The same role ``x_labels``/``hue_labels``
            play in ``plot_dv_by_two_ivs``.
        title: Figure title.
        value_labels: Print the value above each bar.
    """
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
    if errorbar not in {None, "sem", "std", "ci95"}:
        raise ValueError("errorbar must be one of: 'ci95', 'sem', 'std', or None.")

    data = responses.copy()
    if phase is not None and "phase" in data:
        data = data[data["phase"].astype(str).str.lower() == phase.lower()]
    if data.empty:
        raise ValueError(f"No {phase or ''} response rows are available to plot.")

    figure, axes = plt.subplots(
        len(dvs),
        len(ivs),
        figsize=(3.4 * len(ivs), 2.8 * len(dvs)),
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
            summary["ci95"] = summary["sem"] * summary["count"].map(ci95_multiplier)
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
            # One color per condition/level, not one flat color for the whole
            # panel -- and the same index gives the same color in every other
            # panel and in plot_dv_by_two_ivs, so a level reads as one color
            # across the whole figure.
            #
            # level_color keeps that positional scheme for every ordinary
            # factor, and overrides it for the few whose levels carry meaning:
            # tested_w_xai is blue when the explanation was shown and red when
            # it was not, whichever way the levels happen to sort.
            colors = [
                level_color(iv, level, i)
                for i, level in enumerate(plotted["level"].tolist())
            ]
            bars = axis.bar(
                positions,
                plotted["mean"].to_numpy(dtype=float),
                yerr=errors,
                capsize=4 if errors is not None else 0,
                width=_BAR_WIDTH,
                color=colors,
                alpha=0.9,
                edgecolor="white",
                linewidth=0.6,
            )
            axis.set_xticks(
                positions,
                [
                    _display_label(level, (level_labels or {}).get(iv))
                    for level in plotted["level"]
                ],
            )
            axis.tick_params(axis="x", rotation=25, labelsize=8)
            axis.tick_params(axis="y", labelsize=8)
            axis.set_xlabel(pretty(iv), fontsize=9)
            axis.set_ylabel(pretty(dv), fontsize=9)
            axis.set_title(f"{pretty(dv)} by {pretty(iv)}", fontsize=9)

            values = plotted["mean"].to_numpy(dtype=float)
            _finish_axis(axis, values, errors, labelled=value_labels)
            if value_labels:
                _label_bars(axis, bars, values, errors)

    if title:
        figure.suptitle(title, fontsize=10.5)
    summary_table = (
        pd.concat(summaries, ignore_index=True)
        if summaries
        else pd.DataFrame(
            columns=["iv", "dv", "level", "count", "mean", "std", "sem", "ci95"]
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
    errorbar: Optional[str] = "ci95",
    x_levels: Optional[Sequence[Any]] = None,
    hue_levels: Optional[Sequence[Any]] = None,
    x_labels: Optional[Mapping[Any, str]] = None,
    hue_labels: Optional[Mapping[Any, str]] = None,
    title: Optional[str] = None,
    value_labels: bool = True,
) -> InteractionPlot:
    """Plot participant-level mean DV bars using one IV for x and one for color.

    Args:
        responses: Responses to plot.
        x_iv: IV placed on the x axis.
        hue_iv: IV distinguished by colour within each x group.
        dv: Dependent variable to plot.
        participant_column: Column identifying participants.
        phase: Restrict to one trial phase, e.g. ``testing``.
        errorbar: Error bar statistic -- ``ci95`` (Student-t 95% CI half-width,
            the default), ``sem``, ``std``, or None to hide them.
        x_levels: Order of x-axis levels.
        hue_levels: Order of colour levels.
        x_labels: Display names for x-axis levels.
        hue_labels: Display names for colour levels.
        title: Figure title.
        value_labels: Print the value above each bar.
    """
    import matplotlib.pyplot as plt

    required = [participant_column, x_iv, hue_iv, dv]
    missing = [column for column in required if column not in responses]
    if missing:
        raise ValueError(f"Response data is missing columns: {missing}.")
    if errorbar not in {None, "sem", "std", "ci95"}:
        raise ValueError("errorbar must be one of: 'ci95', 'sem', 'std', or None.")

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
    summary["ci95"] = summary["sem"] * summary["count"].map(ci95_multiplier)
    summary.insert(0, "dv", dv)
    summary.insert(0, "hue_iv", hue_iv)
    summary.insert(0, "x_iv", x_iv)

    ordered_x = _ordered_levels(summary["x_level"].tolist(), x_levels)
    ordered_hue = _ordered_levels(summary["hue_level"].tolist(), hue_levels)
    figure, axis = plt.subplots(figsize=(max(5.2, 1.15 * len(ordered_x)), 3.4))

    group_positions = np.arange(len(ordered_x), dtype=float)
    # 0.72 rather than a full 1.0 leaves a visible gap between x-groups; each
    # hue's own share is then narrowed again so adjacent conditions within a
    # group don't touch either.
    group_span = 0.72
    width = (group_span / max(1, len(ordered_hue))) * _BAR_WIDTH
    slot = group_span / max(1, len(ordered_hue))
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
            - group_span / 2
            + slot / 2
            + hue_index * slot
        )
        bars = axis.bar(
            positions,
            means,
            width=width,
            yerr=None if errorbar is None else errors,
            capsize=4 if errorbar is not None else 0,
            label=_display_label(hue_level, hue_labels),
            color=level_color(hue_iv, hue_level, hue_index),
            alpha=0.9,
            edgecolor="white",
            linewidth=0.6,
        )
        if value_labels:
            _label_bars(
                axis,
                bars,
                np.asarray(means, dtype=float),
                np.asarray(errors, dtype=float) if errorbar is not None else None,
            )

    axis.set_xticks(
        group_positions,
        [_display_label(level, x_labels) for level in ordered_x],
    )
    axis.tick_params(axis="x", labelsize=8)
    axis.tick_params(axis="y", labelsize=8)
    axis.set_xlabel(pretty(x_iv), fontsize=9)
    axis.set_ylabel(pretty(dv), fontsize=9)
    axis.set_title(
        title or f"{pretty(dv)} by {pretty(x_iv)} and {pretty(hue_iv)}",
        fontsize=9.5,
    )
    axis.legend(title=pretty(hue_iv), fontsize=8, title_fontsize=8)

    values = summary["mean"].to_numpy(dtype=float)
    summary_errors = (
        summary[errorbar].to_numpy(dtype=float)
        if errorbar is not None and errorbar in summary
        else None
    )
    _finish_axis(axis, values, summary_errors)
    figure.tight_layout()
    return InteractionPlot(figure=figure, axis=axis, summary=summary)


def _ordered_levels(
    observed: Sequence[Any],
    configured: Optional[Sequence[Any]],
) -> list[Any]:
    """Keep configured level order, followed by any unexpected observed levels.

    ``observed`` is deduplicated (order preserved) regardless of path: a
    caller summarizing two IVs at once passes a column with one repeat per
    level of the *other* IV -- e.g. four x-axis levels each repeated once per
    hue level -- and without this, ``plot_dv_by_two_ivs`` drew one bar group
    per repeat instead of per level, silently multiplying the bars on any plot
    with more than one hue level.
    """
    deduped: list[Any] = []
    for value in observed:
        if not any(_same_level(value, existing) for existing in deduped):
            deduped.append(value)
    observed = deduped
    if configured is None:
        return _hybrid_last(observed)
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
    return _hybrid_last(ordered)


def _hybrid_last(levels: list[Any]) -> list[Any]:
    """CoXAM's ``hybrid`` condition plots after ``decision_tree``/
    ``logistic_regression``, not wherever it happened to sort or appear in a
    configured order -- it names two families rather than one, so it reads
    better as the summary bar after the two pure conditions than interleaved
    with them.
    """
    is_hybrid = lambda level: str(level).strip().lower() == "hybrid"
    return [level for level in levels if not is_hybrid(level)] + [
        level for level in levels if is_hybrid(level)
    ]


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
    return pretty(value)

"""Render a study's human-vs-model comparison panels as one lettered PNG.

Consumes the exact panel shape ``assets/build_human_vs_model_plot_data.py``
already writes to ``assets/human_vs_model_plot_data.json`` --
``{title, dv, note, categories, series: [{name, values, error, n}]}`` -- so
this has no study-specific logic of its own. Reuses the bar/whisker/label
conventions from ``grid.py`` and the shared palette so a figure drawn here
matches every other chart in this package.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

from .grid import _BAR_WIDTH, _despine, _label_bars
from .palette import categorical_color

__all__ = ["render_panel_grid_png", "render_comparison_figure"]

#: "Human" always takes this palette step (orange) so it reads the same way
#: across every study; the first non-Human series takes step 0 (blue), the
#: same colors the reference figures use.
_HUMAN_COLOR_INDEX = 1
_MODEL_COLOR_INDEX = 0

_LETTERS = "abcdefghijklmnopqrstuvwxyz"


def _series_color(name: str, other_index: list[int]) -> str:
    if name == "Human":
        return categorical_color(_HUMAN_COLOR_INDEX)
    index = other_index[0]
    other_index[0] += 1
    # Skip the index reserved for Human so a second/third model series never
    # collides with it.
    if index >= _HUMAN_COLOR_INDEX:
        index += 1
    return categorical_color(index)


def _draw_panel(axis: Any, panel: dict[str, Any], *, letter: str) -> None:
    categories: list[str] = panel["categories"]
    series: list[dict[str, Any]] = panel["series"]
    positions = np.arange(len(categories))

    group_span = 0.72
    width = (group_span / max(1, len(series))) * _BAR_WIDTH
    slot = group_span / max(1, len(series))
    other_index = [_MODEL_COLOR_INDEX]

    for series_index, entry in enumerate(series):
        values = np.array(
            [np.nan if v is None else float(v) for v in entry["values"]], dtype=float
        )
        errors = np.array(
            [0.0 if e is None else float(e) for e in entry.get("error", [])], dtype=float
        ) if entry.get("error") is not None else None
        bar_positions = positions - group_span / 2 + slot / 2 + series_index * slot
        bars = axis.bar(
            bar_positions,
            values,
            width=width,
            yerr=errors,
            capsize=4 if errors is not None else 0,
            label=entry["name"],
            color=_series_color(entry["name"], other_index),
            alpha=0.9,
            edgecolor="white",
            linewidth=0.6,
        )
        _label_bars(axis, bars, values, errors)

    axis.set_xticks(positions, categories)
    axis.tick_params(axis="x", labelsize=8, rotation=0)
    axis.tick_params(axis="y", labelsize=8)
    axis.set_ylabel(panel.get("dv", ""), fontsize=8.5)
    axis.set_title(f"{letter}) {panel['title']}", fontsize=9, loc="left")
    axis.set_ylim(0, 1.12)
    _despine(axis)
    axis.legend(fontsize=7, loc="best", framealpha=0.85)


def render_panel_grid_png(
    panels: Sequence[dict[str, Any]],
    *,
    title: str,
    ncols: int = 2,
) -> Any:
    """Draw every panel as a lettered subplot (a, b, c, ...) in one figure.

    Args:
        panels: Panel dicts in the shape ``_panel()`` builds:
            ``{title, dv, categories, series: [{name, values, error, n}]}``.
        title: Figure suptitle.
        ncols: Subplots per row.

    Raises:
        ValueError: If ``panels`` is empty.
    """
    import matplotlib.pyplot as plt

    if not panels:
        raise ValueError("Pass at least one panel to render.")

    nrows = math.ceil(len(panels) / ncols)
    figure, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(6.4 * ncols, 3.2 * nrows),
        squeeze=False,
        constrained_layout=True,
    )
    flat_axes = axes.flatten()
    for index, panel in enumerate(panels):
        _draw_panel(flat_axes[index], panel, letter=_LETTERS[index % len(_LETTERS)])
    for index in range(len(panels), len(flat_axes)):
        flat_axes[index].set_axis_off()

    if title:
        figure.suptitle(title, fontsize=11)
    return figure


# ---------------------------------------------------------------------------
# The published-figure layout
# ---------------------------------------------------------------------------


def _grouped_rows(panels: Sequence[dict[str, Any]]) -> list[tuple[str, list[dict[str, Any]]]]:
    """Panels grouped into rows by their ``section``, order preserved.

    A panel with no ``section`` forms its own row, so a study that never sets
    one still renders as it did before.
    """
    rows: list[tuple[str, list[dict[str, Any]]]] = []
    for panel in panels:
        # A pooled overview is useful in the data the UI reads, but it is not
        # one of the study's condition rows, so it stays out of the figure.
        if panel.get("role") == "summary":
            continue
        section = str(panel.get("section") or "")
        if rows and section and rows[-1][0] == section:
            rows[-1][1].append(panel)
        else:
            rows.append((section, [panel]))
    return rows


def _shared_limits(panels: Sequence[dict[str, Any]]) -> tuple[float, float]:
    """One y range for every panel, so bar heights compare across the figure.

    Zoomed to the data rather than pinned to 0-1: these accuracies sit in a
    narrow band, and a full 0-100% axis flattens every difference in it. Ends
    are rounded outward to a 10-point boundary so the ticks stay round.
    """
    lows: list[float] = []
    highs: list[float] = []
    for panel in panels:
        for entry in panel.get("series", []):
            values = [v for v in entry.get("values", []) if v is not None]
            errors = entry.get("error") or []
            for index, value in enumerate(values):
                error = 0.0
                if index < len(errors) and errors[index] is not None:
                    error = float(errors[index])
                lows.append(float(value) - error)
                highs.append(float(value) + error)
    if not lows:
        return (0.0, 1.0)
    low = max(0.0, math.floor((min(lows) - 0.04) * 10) / 10)
    high = min(1.0, math.ceil((max(highs) + 0.04) * 10) / 10)
    if high - low < 0.2:  # keep a readable band for a very flat study
        high = min(1.0, low + 0.2)
    return (low, high)


def _draw_reference_panel(
    axis: Any,
    panel: dict[str, Any],
    *,
    letter: str,
    show_ylabel: bool,
    limits: tuple[float, float],
) -> list[Any]:
    """One panel in the published style: no legend, no per-bar numbers."""
    from matplotlib.ticker import PercentFormatter

    categories: list[str] = panel["categories"]
    series: list[dict[str, Any]] = panel["series"]
    positions = np.arange(len(categories))

    group_span = 0.72
    width = (group_span / max(1, len(series))) * 0.9
    slot = group_span / max(1, len(series))
    other_index = [_MODEL_COLOR_INDEX]
    handles: list[Any] = []

    for series_index, entry in enumerate(series):
        values = np.array(
            [np.nan if v is None else float(v) for v in entry["values"]], dtype=float
        )
        errors = (
            np.array(
                [0.0 if e is None else float(e) for e in entry.get("error", [])],
                dtype=float,
            )
            if entry.get("error") is not None
            else None
        )
        bars = axis.bar(
            positions - group_span / 2 + slot / 2 + series_index * slot,
            values,
            width=width,
            yerr=errors,
            capsize=3,
            label=entry["name"],
            color=_series_color(entry["name"], other_index),
            error_kw={"elinewidth": 1.1, "capthick": 1.1},
        )
        handles.append(bars)

    axis.set_xticks(positions, categories)
    axis.tick_params(axis="x", labelsize=8.5, length=0)
    axis.tick_params(axis="y", labelsize=8.5)
    axis.set_ylim(*limits)
    axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    # The letter sits with the panel's own short label, as in the figures.
    axis.set_title(
        f"{letter}) {panel.get('subtitle') or panel.get('title', '')}",
        fontsize=9.5,
        loc="left",
    )
    if show_ylabel:
        axis.set_ylabel(panel.get("dv", ""), fontsize=9)
    else:
        axis.tick_params(axis="y", labelleft=False)
    _despine(axis)
    return handles


def render_comparison_figure(
    panels: Sequence[dict[str, Any]],
    *,
    title: str = "",
) -> Any:
    """Draw the panels the way the published comparison figures are laid out.

    One row per ``section``, a spanning heading above each row, a single shared
    y scale in percent, and one legend for the whole figure -- rather than
    ``render_panel_grid_png``'s uniform grid of self-contained subplots, where
    every panel repeats the legend and axis label and each is scaled 0-100%.

    Args:
        panels: Panel dicts as ``assets/human_vs_model_plot_data.json`` carries
            them, optionally with ``section``, ``subtitle`` and ``role``.
        title: Kept for call compatibility; the section headings carry the
            labelling, so no figure-level heading is drawn over them.

    Raises:
        ValueError: If ``panels`` is empty.
    """
    import matplotlib.pyplot as plt

    if not panels:
        raise ValueError("Pass at least one panel to render.")

    rows = _grouped_rows(panels)
    limits = _shared_limits(panels)
    widest = max(len(row_panels) for _, row_panels in rows)

    figure = plt.figure(figsize=(3.15 * widest + 0.9, 2.9 * len(rows) + 1.1))
    # A thin final strip holds the shared legend: a figure-level legend gets
    # squeezed out from under a stack of subfigures, and matplotlib 3.5 has no
    # "outside lower center" location to place it with.
    heights = [1.0] * len(rows) + [0.16]
    panes = figure.subfigures(len(rows) + 1, 1, hspace=0.06, height_ratios=heights)
    subfigures, legend_pane = list(panes[:-1]), panes[-1]

    letters = iter(_LETTERS)
    legend_handles: list[Any] = []
    legend_labels: list[str] = []

    for subfigure, (section, row_panels) in zip(subfigures, rows):
        axes = subfigure.subplots(1, len(row_panels), squeeze=False)[0]
        for index, (axis, panel) in enumerate(zip(axes, row_panels)):
            handles = _draw_reference_panel(
                axis,
                panel,
                letter=next(letters, "?"),
                show_ylabel=index == 0,
                limits=limits,
            )
            if not legend_handles:
                legend_handles = handles
                legend_labels = [entry["name"] for entry in panel["series"]]
        if section:
            subfigure.suptitle(section, fontsize=10.5, y=1.0)

    if legend_handles:
        legend_pane.legend(
            legend_handles,
            legend_labels,
            loc="center",
            ncol=len(legend_labels),
            frameon=False,
            fontsize=10,
        )
    return figure

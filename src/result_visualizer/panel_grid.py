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

from .grid import _BAR_WIDTH, _label_bars
from .palette import categorical_color

__all__ = ["render_panel_grid_png"]

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

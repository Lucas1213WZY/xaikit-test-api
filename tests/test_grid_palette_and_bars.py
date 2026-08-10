"""Shared palette, bar width, and a real duplicate-bar bug in the grid plots."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mc
import pandas as pd
import pytest

from src.result_visualizer import plot_dv_by_two_ivs, plot_iv_dv_grid
from src.result_visualizer.grid import _ordered_levels
from src.result_visualizer.palette import CATEGORICAL_LIGHT, categorical_color


def _responses(n_x: int = 4, n_hue: int = 2) -> pd.DataFrame:
    x_levels = [f"x{i}" for i in range(n_x)]
    hue_levels = [f"h{i}" for i in range(n_hue)]
    rows = [
        {"participantId": p, "x": x, "hue": h, "phase": "testing", "dv": 0.5}
        for p in range(6)
        for x in x_levels
        for h in hue_levels
    ]
    return pd.DataFrame(rows)


# -- the duplicate-bar bug -------------------------------------------------


def test_ordered_levels_dedupes_a_repeated_column():
    """The exact input plot_dv_by_two_ivs's two-key summary produces: each
    x-level appears once per hue level, not once."""
    repeated = ["a", "b", "a", "b", "a", "b"]
    assert _ordered_levels(repeated, None) == ["a", "b"]


def test_ordered_levels_dedupes_with_a_configured_order_too():
    repeated = ["b", "a", "b", "a"]
    assert _ordered_levels(repeated, ["a", "b"]) == ["a", "b"]


def test_interaction_plot_draws_exactly_one_bar_group_per_level_pair():
    """Regression: before deduping, an x-level repeated once per hue level was
    treated as that many distinct levels, multiplying the bars drawn --
    4 x-levels x 2 hues x 2 (bug) = 16 patches instead of 8."""
    df = _responses(n_x=4, n_hue=2)
    plot = plot_dv_by_two_ivs(df, x_iv="x", hue_iv="hue", dv="dv", errorbar=None)
    assert len(plot.axis.patches) == 4 * 2


def test_interaction_plot_scales_with_more_hue_levels():
    df = _responses(n_x=3, n_hue=5)
    plot = plot_dv_by_two_ivs(df, x_iv="x", hue_iv="hue", dv="dv", errorbar=None)
    assert len(plot.axis.patches) == 3 * 5


# -- the shared palette -----------------------------------------------------


def test_grid_panel_assigns_one_palette_color_per_condition():
    df = _responses(n_x=4, n_hue=1).rename(columns={"x": "condition"})
    grid = plot_iv_dv_grid(df, ivs=["condition"], dvs=["dv"], phase="testing")
    bars = list(grid.axes.flat[0].patches)
    colors = [mc.to_hex(bar.get_facecolor()) for bar in bars]
    assert colors == list(CATEGORICAL_LIGHT[: len(bars)])


def test_interaction_plot_colors_come_from_the_same_shared_palette():
    df = _responses(n_x=3, n_hue=2)
    plot = plot_dv_by_two_ivs(df, x_iv="x", hue_iv="hue", dv="dv", errorbar=None)
    colors = {mc.to_hex(p.get_facecolor()) for p in plot.axis.patches}
    assert colors == {categorical_color(0), categorical_color(1)}


def test_categorical_color_cycles_past_eight_rather_than_raising():
    assert categorical_color(8) == categorical_color(0)
    assert categorical_color(9) == categorical_color(1)


def test_dark_and_light_palettes_are_the_same_length_and_differ():
    from src.result_visualizer.palette import CATEGORICAL_DARK

    assert len(CATEGORICAL_DARK) == len(CATEGORICAL_LIGHT)
    assert CATEGORICAL_DARK != CATEGORICAL_LIGHT


# -- bar width ---------------------------------------------------------------


def test_grid_bars_leave_a_visible_gap():
    """The literal bug report: bars filled the full space with no gap."""
    df = _responses(n_x=4, n_hue=1).rename(columns={"x": "condition"})
    grid = plot_iv_dv_grid(df, ivs=["condition"], dvs=["dv"], phase="testing")
    widths = {round(bar.get_width(), 3) for bar in grid.axes.flat[0].patches}
    assert widths == {0.48}
    assert all(w < 1.0 for w in widths)


def test_interaction_plot_bars_do_not_touch_within_a_group():
    df = _responses(n_x=2, n_hue=3)
    plot = plot_dv_by_two_ivs(df, x_iv="x", hue_iv="hue", dv="dv", errorbar=None)
    lefts = sorted(bar.get_x() for bar in plot.axis.patches if bar.get_x() < 0.5)
    widths = [bar.get_width() for bar in plot.axis.patches]
    # Consecutive bars within the first group must not overlap.
    for (left, width), next_left in zip(zip(lefts, widths), lefts[1:]):
        assert left + width <= next_left + 1e-9


# -- hybrid always plots last ------------------------------------------------


def test_hybrid_sorts_last_with_no_configured_order():
    from src.result_visualizer.grid import _ordered_levels

    assert _ordered_levels(["logistic_regression", "hybrid", "decision_tree"], None) == [
        "logistic_regression", "decision_tree", "hybrid",
    ]


def test_hybrid_sorts_last_even_when_configured_order_puts_it_first():
    from src.result_visualizer.grid import _ordered_levels

    assert _ordered_levels(
        ["hybrid", "decision_tree", "logistic_regression"],
        ["hybrid", "decision_tree", "logistic_regression"],
    ) == ["decision_tree", "logistic_regression", "hybrid"]


def test_no_hybrid_present_is_unaffected():
    from src.result_visualizer.grid import _ordered_levels

    assert _ordered_levels(["b", "a"], None) == ["b", "a"]


# -- prettified labels --------------------------------------------------


def test_pretty_keeps_xai_as_an_acronym():
    from src.result_visualizer.labels import pretty

    assert pretty("xai_type") == "XAI Type"
    assert pretty("counterfactual_accuracy") == "Counterfactual Accuracy"


def test_bar_label_survives_a_missing_x_hue_combination():
    """Regression: matplotlib's own Axes.bar_label raises IndexError when a
    bar's height is NaN alongside a non-None yerr (an empty error-bar segment
    is drawn for the missing bar). A missing x/hue combination is routine
    here, so plot_dv_by_two_ivs must not crash on it."""
    df = pd.DataFrame({
        "participantId": [0, 1, 2, 3],
        "x": ["a", "a", "b", "b"],
        "hue": ["h0", "h1", "h0", "h1"],
        "phase": "testing",
        "dv": [0.5, 0.5, 0.5, 0.5],
    })
    # Drop one x/hue combination entirely so its mean is NaN in the summary.
    df = df[~((df["x"] == "b") & (df["hue"] == "h1"))]
    plot = plot_dv_by_two_ivs(df, x_iv="x", hue_iv="hue", dv="dv", errorbar="sem")
    assert plot.axis is not None


def test_grid_panel_draws_hybrid_bar_last():
    df = pd.DataFrame({
        "participantId": list(range(6)),
        "condition": ["hybrid", "decision_tree", "logistic_regression"] * 2,
        "dv": [0.5] * 6,
    })
    grid = plot_iv_dv_grid(df, ivs=["condition"], dvs=["dv"], phase=None)
    labels = [t.get_text() for t in grid.axes.flat[0].get_xticklabels()]
    assert labels[-1] == "Hybrid"


# -- 95% CI is now the default error bar, not SEM ----------------------------


def _varying_responses(n_x: int = 4, n_hue: int = 2) -> pd.DataFrame:
    """Like _responses, but with real participant-to-participant spread --
    a constant dv gives sem=0, which can't tell ci95 apart from sem."""
    x_levels = [f"x{i}" for i in range(n_x)]
    hue_levels = [f"h{i}" for i in range(n_hue)]
    rows = [
        {
            "participantId": p,
            "x": x,
            "hue": h,
            "phase": "testing",
            "dv": 0.1 * p,
        }
        for p in range(6)
        for x in x_levels
        for h in hue_levels
    ]
    return pd.DataFrame(rows)


def test_ci95_is_the_default_errorbar_and_wider_than_sem():
    """ci95 = sem * a Student-t multiplier > 1, so it must be strictly wider
    than sem for the same data -- and it's what a plot draws by default now."""
    df = _varying_responses(n_x=1, n_hue=1).rename(columns={"x": "condition"})
    grid_default = plot_iv_dv_grid(df, ivs=["condition"], dvs=["dv"], phase="testing")
    grid_sem = plot_iv_dv_grid(
        df, ivs=["condition"], dvs=["dv"], phase="testing", errorbar="sem"
    )
    ci95_row = grid_default.summary.iloc[0]
    sem_row = grid_sem.summary.iloc[0]
    # Same underlying data either way -- only which column drives the error bar differs.
    assert ci95_row["sem"] == pytest.approx(sem_row["sem"])
    assert ci95_row["ci95"] > sem_row["sem"] > 0


def test_interaction_plot_also_defaults_to_ci95():
    df = _varying_responses(n_x=2, n_hue=2)
    plot_default = plot_dv_by_two_ivs(df, x_iv="x", hue_iv="hue", dv="dv")
    assert "ci95" in plot_default.summary.columns
    row = plot_default.summary.iloc[0]
    assert row["ci95"] > row["sem"] > 0


def test_ci95_multiplier_matches_the_shared_helper():
    from src.result_visualizer.intervals import ci95_multiplier

    df = _responses(n_x=1, n_hue=1).rename(columns={"x": "condition"})
    grid = plot_iv_dv_grid(df, ivs=["condition"], dvs=["dv"], phase="testing")
    row = grid.summary.iloc[0]
    assert row["ci95"] == pytest.approx(row["sem"] * ci95_multiplier(int(row["count"])))

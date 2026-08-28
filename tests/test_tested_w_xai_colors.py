"""tested_w_xai is colored by meaning, not by where its level happens to sort.

Every other factor keeps the positional scheme: the nth level takes the nth
palette step, which is right when the levels are arbitrary. tested_w_xai is not
arbitrary -- False sorts first and so took the blue that belongs to the
explanation condition, while True, the condition the study exists to test, took
the orange. Filtering to one level or adding a third repainted them again.

Blue (#2a78d6) now means the explanation was shown and red (#e34948) means it
was not, whatever else is in the figure. The pair was validated on its own
against the light surface: protan delta-E 21.6, tritan 34.5, normal 32.3.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import pytest

from src.result_visualizer.palette import (
    CATEGORICAL_LIGHT,
    categorical_color,
    level_color,
    semantic_color,
)

WITH_XAI = CATEGORICAL_LIGHT[0]     # blue
WITHOUT_XAI = CATEGORICAL_LIGHT[7]  # red


def _results():
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        [
            {
                "participantId": p,
                "xai_type": xai,
                "tested_w_xai": tested,
                "phase": "testing",
                "counterfactual_accuracy": rng.uniform(0.2, 0.7),
            }
            for xai in ("decision_tree", "logistic_regression")
            for tested in (False, True)
            for p in range(6)
        ]
    )


# -- the rule itself --------------------------------------------------------


def test_with_xai_is_blue_and_without_is_red():
    assert semantic_color("tested_w_xai", True) == WITH_XAI
    assert semantic_color("tested_w_xai", False) == WITHOUT_XAI


@pytest.mark.parametrize(
    "spelling, expected",
    [
        (True, WITH_XAI), ("True", WITH_XAI), ("w/ XAI", WITH_XAI), (1, WITH_XAI),
        (False, WITHOUT_XAI), ("False", WITHOUT_XAI), ("w/o XAI", WITHOUT_XAI), (0, WITHOUT_XAI),
    ],
)
def test_every_spelling_of_the_condition_gets_the_same_color(spelling, expected):
    """It arrives as a bool, as a string after a CSV round trip, or as the
    corpus's own 'w/ XAI' -- one condition, so one color."""
    assert semantic_color("tested_w_xai", spelling) == expected


def test_the_color_does_not_depend_on_where_the_level_sorts():
    """The bug: position decided the color, so False took index 0's blue."""
    assert level_color("tested_w_xai", True, 0) == WITH_XAI
    assert level_color("tested_w_xai", True, 5) == WITH_XAI
    assert level_color("tested_w_xai", False, 0) == WITHOUT_XAI


def test_every_other_factor_keeps_the_positional_scheme():
    """Only tested_w_xai gets a rule; arbitrary factors are fine as they were."""
    for index in range(4):
        assert level_color("xai_type", "decision_tree", index) == categorical_color(index)
        assert level_color("xai_property", "faithful", index) == categorical_color(index)
    assert semantic_color("xai_type", "decision_tree") is None


def test_an_unreadable_level_falls_back_rather_than_guessing():
    assert semantic_color("tested_w_xai", "maybe") is None
    assert level_color("tested_w_xai", "maybe", 2) == categorical_color(2)


# -- what the plots actually draw -------------------------------------------


def test_the_grouped_two_iv_plot_uses_the_rule():
    from src.result_visualizer import plot_dv_by_two_ivs

    plot = plot_dv_by_two_ivs(
        _results(), x_iv="xai_type", hue_iv="tested_w_xai",
        dv="counterfactual_accuracy",
    )
    axis = getattr(plot, "figure", plot).get_axes()[0]
    drawn = {
        container.get_label(): mcolors.to_hex(container.patches[0].get_facecolor())
        for container in axis.containers
        if container.get_label() and not container.get_label().startswith("_")
    }
    assert drawn["True"] == WITH_XAI
    assert drawn["False"] == WITHOUT_XAI


def test_the_single_iv_grid_uses_the_rule():
    from src.result_visualizer import plot_iv_dv_grid

    grid = plot_iv_dv_grid(
        _results(), ivs=["tested_w_xai"], dvs=["counterfactual_accuracy"]
    )
    axis = getattr(grid, "figure", grid).get_axes()[0]
    order = [label.get_text() for label in axis.get_xticklabels()]
    drawn = [mcolors.to_hex(patch.get_facecolor()) for patch in axis.patches[: len(order)]]
    by_level = dict(zip(order, drawn))
    assert by_level["True"] == WITH_XAI
    assert by_level["False"] == WITHOUT_XAI


def test_an_ordinary_factor_still_plots_in_palette_order():
    from src.result_visualizer import plot_iv_dv_grid

    grid = plot_iv_dv_grid(
        _results(), ivs=["xai_type"], dvs=["counterfactual_accuracy"]
    )
    axis = getattr(grid, "figure", grid).get_axes()[0]
    drawn = [mcolors.to_hex(patch.get_facecolor()) for patch in axis.patches[:2]]
    assert drawn == [categorical_color(0), categorical_color(1)]

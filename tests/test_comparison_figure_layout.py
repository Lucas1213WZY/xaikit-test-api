"""The human-vs-model figures use the published layout, not a uniform grid.

``render_panel_grid_png`` gives every panel its own legend, its own y label and
a 0-100% scale. The published comparison figures instead put one row per
section under a spanning heading, share a single zoomed percent axis across the
whole figure, and carry one legend at the bottom -- which is what
``render_comparison_figure`` draws.
"""

import matplotlib

matplotlib.use("Agg")

import pytest

from src.result_visualizer.panel_grid import (
    _grouped_rows,
    _shared_limits,
    render_comparison_figure,
)


def _panel(section, subtitle, values, error=0.03, role=""):
    return {
        "title": f"{section} — {subtitle}",
        "section": section,
        "subtitle": subtitle,
        "role": role,
        "dv": "Forward accuracy",
        "categories": ["Rules", "Weights", "Hybrid"],
        "series": [
            {"name": "Human", "values": values, "error": [error] * len(values)},
            {"name": "CoXAM", "values": values, "error": [error] * len(values)},
        ],
    }


# ---------------------------------------------------------------------------
# Rows
# ---------------------------------------------------------------------------


def test_panels_sharing_a_section_become_one_row():
    panels = [
        _panel("Wine Quality", "w/o XAI", [0.64, 0.71, 0.72]),
        _panel("Wine Quality", "w/ XAI", [0.74, 0.72, 0.70]),
        _panel("Mushrooms", "w/o XAI", [0.69, 0.62, 0.60]),
        _panel("Mushrooms", "w/ XAI", [0.78, 0.67, 0.69]),
    ]
    rows = _grouped_rows(panels)
    assert [(section, len(items)) for section, items in rows] == [
        ("Wine Quality", 2),
        ("Mushrooms", 2),
    ]


def test_a_summary_panel_is_kept_in_the_data_but_left_out_of_the_figure():
    # The pooled overview is real data the UI can show, but it is not one of
    # the study's condition rows and would add a fourth row to a 3-row figure.
    panels = [
        _panel("Overall", "every dataset", [0.75, 0.67, 0.69], role="summary"),
        _panel("Adult income", "w/o XAI", [0.70, 0.75, 0.77]),
        _panel("Adult income", "w/ XAI", [0.66, 0.79, 0.86]),
    ]
    rows = _grouped_rows(panels)
    assert [section for section, _ in rows] == ["Adult income"]


def test_a_panel_without_a_section_stands_alone():
    panels = [_panel("", "first", [0.7, 0.7, 0.7]), _panel("", "second", [0.8, 0.8, 0.8])]
    assert len(_grouped_rows(panels)) == 2


# ---------------------------------------------------------------------------
# Shared scale
# ---------------------------------------------------------------------------


def test_the_axis_is_zoomed_to_the_data_not_pinned_to_zero():
    # A full 0-100% axis flattens accuracies that all sit in a narrow band.
    low, high = _shared_limits([_panel("A", "x", [0.64, 0.71, 0.72])])
    assert low >= 0.5
    assert high <= 1.0


def test_the_range_covers_the_error_bars():
    low, high = _shared_limits([_panel("A", "x", [0.95, 0.95, 0.95], error=0.04)])
    assert high >= 0.99


def test_a_very_flat_study_still_gets_a_readable_band():
    low, high = _shared_limits([_panel("A", "x", [0.70, 0.70, 0.70], error=0.0)])
    assert high - low >= 0.2


def test_limits_never_leave_the_proportion_range():
    low, high = _shared_limits([_panel("A", "x", [0.02, 0.99, 0.5], error=0.05)])
    assert low >= 0.0
    assert high <= 1.0


# ---------------------------------------------------------------------------
# The figure itself
# ---------------------------------------------------------------------------


def test_the_figure_has_one_axes_per_panel_plus_a_legend_strip():
    panels = [
        _panel("Wine Quality", "w/o XAI", [0.64, 0.71, 0.72]),
        _panel("Wine Quality", "w/ XAI", [0.74, 0.72, 0.70]),
        _panel("Mushrooms", "w/o XAI", [0.69, 0.62, 0.60]),
    ]
    figure = render_comparison_figure(panels)
    drawn = [axis for subfigure in figure.subfigs for axis in subfigure.axes]
    assert len(drawn) == 3
    # One legend for the whole figure, on its own strip -- not one per panel.
    assert not any(axis.get_legend() for axis in drawn)
    assert any(subfigure.legends for subfigure in figure.subfigs)


def test_every_panel_shares_one_y_range():
    panels = [
        _panel("A", "x", [0.64, 0.71, 0.72]),
        _panel("B", "y", [0.90, 0.92, 0.95]),
    ]
    figure = render_comparison_figure(panels)
    drawn = [axis for subfigure in figure.subfigs for axis in subfigure.axes]
    limits = {axis.get_ylim() for axis in drawn}
    assert len(limits) == 1


def test_only_the_leftmost_panel_of_a_row_is_labelled():
    panels = [
        _panel("Wine Quality", "w/o XAI", [0.64, 0.71, 0.72]),
        _panel("Wine Quality", "w/ XAI", [0.74, 0.72, 0.70]),
    ]
    figure = render_comparison_figure(panels)
    axes = figure.subfigs[0].axes
    assert axes[0].get_ylabel()
    assert not axes[1].get_ylabel()


def test_the_top_and_right_frame_are_dropped():
    figure = render_comparison_figure([_panel("A", "x", [0.7, 0.7, 0.7])])
    axis = figure.subfigs[0].axes[0]
    assert not axis.spines["top"].get_visible()
    assert not axis.spines["right"].get_visible()
    assert axis.spines["left"].get_visible()
    assert axis.spines["bottom"].get_visible()


def test_an_empty_panel_list_is_refused():
    with pytest.raises(ValueError):
        render_comparison_figure([])

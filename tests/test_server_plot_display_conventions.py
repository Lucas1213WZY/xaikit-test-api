"""The server's plot payloads carry the same level order and vocabulary the
figures do.

The UI draws its own chart from the payload's ``summary`` rows rather than the
optional PNG, so an order or a label that lives only in the matplotlib figure
never reaches the screen. These pin both halves: what the figure is told, and
what the rows carry.
"""

import matplotlib

matplotlib.use("Agg")
import pandas as pd
import pytest

from server.pipeline import grid_plot_payload, interaction_plot_payload


class _Design:
    ivs = [
        {"name": "xai_type", "source_label": "XAI Type"},
        {"name": "tested_w_xai", "source_label": "Tested with XAI"},
    ]
    cvs: list = []
    rvs: list = []
    dvs = [{"name": "forward_accuracy", "source_label": "Task Accuracy"}]
    simulatable_dvs = ["forward_accuracy"]


class _Study:
    def __init__(self, results: pd.DataFrame) -> None:
        self.simulated_results = results
        self.design_export = _Design()


@pytest.fixture
def study() -> _Study:
    # xai_type deliberately arrives in the design's own order (none,
    # attribution, importance) so a payload that simply echoes the data would
    # fail the ordering assertions below.
    rows = [
        {
            "participantId": participant,
            "phase": "testing",
            "xai_type": xai_type,
            "tested_w_xai": tested,
            "forward_accuracy": 0.5 + 0.1 * index,
        }
        for index, xai_type in enumerate(["none", "attribution", "importance"])
        for tested in (True, False)
        for participant in range(1, 5)
    ]
    return _Study(pd.DataFrame(rows))


def test_interaction_rows_run_in_display_order(study):
    payload = interaction_plot_payload(
        study, x_iv="tested_w_xai", hue_iv="xai_type", dv="forward_accuracy"
    )
    rows = payload["summary"]
    assert [row["x_level"] for row in rows][:3] == [False, False, False]
    assert [row["hue_level"] for row in rows][:3] == ["none", "importance", "attribution"]


def test_interaction_rows_carry_the_condition_names_not_the_booleans(study):
    payload = interaction_plot_payload(
        study, x_iv="tested_w_xai", hue_iv="xai_type", dv="forward_accuracy"
    )
    labels = {row["x_level_label"] for row in payload["summary"]}
    assert labels == {"w/o XAI", "w/ XAI"}
    assert payload["summary"][0]["hue_level_label"] == "None"


def test_grid_rows_run_in_display_order_within_each_panel(study):
    payload = grid_plot_payload(study, ivs=["xai_type", "tested_w_xai"], dvs=["forward_accuracy"])
    rows = payload["summary"]
    xai_rows = [row for row in rows if row["iv"] == "xai_type"]
    presence_rows = [row for row in rows if row["iv"] == "tested_w_xai"]
    assert [row["level"] for row in xai_rows] == ["none", "importance", "attribution"]
    assert [row["level"] for row in presence_rows] == [False, True]
    # Panels stay whole rather than interleaving.
    assert [row["iv"] for row in rows] == ["xai_type"] * 3 + ["tested_w_xai"] * 2


def test_grid_rows_carry_a_level_label(study):
    payload = grid_plot_payload(study, ivs=["tested_w_xai"], dvs=["forward_accuracy"])
    assert [row["level_label"] for row in payload["summary"]] == ["w/o XAI", "w/ XAI"]


def test_a_factor_with_no_convention_is_left_exactly_as_it_was():
    """The conventions are a registry, not a global re-sort: an unregistered
    factor must keep the order and labels the plot helpers give it."""
    results = pd.DataFrame(
        {
            "participantId": [1, 2, 3, 4] * 2,
            "phase": ["testing"] * 8,
            "layout": ["zebra", "alpha"] * 4,
            "forward_accuracy": [0.5, 0.6] * 4,
        }
    )

    class _LayoutDesign:
        ivs = [{"name": "layout", "source_label": "Layout"}]
        cvs: list = []
        rvs: list = []
        dvs = [{"name": "forward_accuracy", "source_label": "Task Accuracy"}]
        simulatable_dvs = ["forward_accuracy"]

    study = _Study(results)
    study.design_export = _LayoutDesign()
    payload = grid_plot_payload(study, ivs=["layout"], dvs=["forward_accuracy"])
    levels = [row["level"] for row in payload["summary"]]
    assert levels == sorted(levels)          # the helper's own groupby order
    assert [row["level_label"] for row in payload["summary"]] == ["Alpha", "Zebra"]

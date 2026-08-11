"""Plot/analysis requests may name variables the way the UI shows them.

Regression for "400: Response data is missing columns: ['XAI Type', 'Tested
with XAI']" from the interaction-plot panel. The design export keeps two
spellings of every variable -- ``name`` is the slug the planner and the results
table use (``xai_type``), ``source_label`` is the free text the UI displays and
posts back (``"XAI Type"``) -- and the plot helpers only ever accepted the
former.
"""

import pandas as pd
import pytest

from server.pipeline import (
    _resolve_variable_names,
    grid_plot_payload,
    interaction_plot_payload,
    resolve_variable_name,
)


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
    return _Study(
        pd.DataFrame(
            {
                "participantId": [1, 1, 2, 2, 3, 3, 4, 4],
                "phase": ["testing"] * 8,
                "xai_type": ["none", "none", "importance", "importance"] * 2,
                "tested_w_xai": [True, False] * 4,
                "forward_accuracy": [1, 0, 1, 1, 0, 1, 1, 0],
            }
        )
    )


def test_a_display_label_resolves_to_the_results_column(study):
    assert resolve_variable_name(study, "XAI Type") == "xai_type"
    assert resolve_variable_name(study, "Tested with XAI") == "tested_w_xai"
    assert resolve_variable_name(study, "Task Accuracy") == "forward_accuracy"


def test_an_actual_column_name_is_left_alone(study):
    assert resolve_variable_name(study, "xai_type") == "xai_type"


def test_label_matching_ignores_case_and_padding(study):
    assert resolve_variable_name(study, "  xai type  ") == "xai_type"


def test_an_unknown_name_is_returned_unchanged_so_it_still_errors(study):
    # Silently rewriting a genuinely wrong name would hide the mistake; the
    # plot helper should still raise against the real column list.
    assert resolve_variable_name(study, "not_a_variable") == "not_a_variable"


def test_none_and_empty_are_passed_through(study):
    assert _resolve_variable_names(study, None) is None
    assert resolve_variable_name(study, "") == ""


def test_interaction_plot_accepts_the_ui_labels_that_used_to_400(study):
    payload = interaction_plot_payload(
        study,
        x_iv="XAI Type",
        hue_iv="Tested with XAI",
        dv="Task Accuracy",
    )
    assert payload["spec"]["x_iv"] == "xai_type"
    assert payload["spec"]["hue_iv"] == "tested_w_xai"
    assert payload["spec"]["dv"] == "forward_accuracy"
    assert payload["summary"]


def test_grid_plot_accepts_the_ui_labels_too(study):
    payload = grid_plot_payload(study, ivs=["XAI Type"], dvs=["Task Accuracy"])
    assert payload["spec"]["ivs"] == ["xai_type"]
    assert payload["spec"]["dvs"] == ["forward_accuracy"]


# ---------------------------------------------------------------------------
# Runner bookkeeping columns are not plot dimensions
# ---------------------------------------------------------------------------


def test_plot_variables_lists_the_designs_factors_not_every_column(study):
    from server.pipeline import plot_variables

    study.simulated_results["explanation_type"] = "dt"
    assert plot_variables(study) == ["xai_type", "tested_w_xai"]


def test_splitting_on_explanation_type_is_refused_with_a_pointer(study):
    # CoXAM writes the surrogate it showed per trial here. Splitting on it
    # drops Hybrid (shown as dt/lr per trial) and invents a "none" bar from
    # the without-XAI trials -- a chart that looks like a result and is not.
    study.simulated_results["explanation_type"] = "dt"
    with pytest.raises(ValueError) as error:
        interaction_plot_payload(
            study, x_iv="dataset", hue_iv="explanation_type", dv="forward_accuracy"
        )
    assert "xai_type" in str(error.value)


def test_the_grid_plot_refuses_it_too(study):
    from server.pipeline import grid_plot_payload as grid

    study.simulated_results["explanation_type"] = "dt"
    with pytest.raises(ValueError):
        grid(study, ivs=["explanation_type"], dvs=["forward_accuracy"])

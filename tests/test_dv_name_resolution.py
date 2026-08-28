"""A DV named the way the UI names it must reach the runner that produces it.

The design planner calls the measure ``counterfactual_simulation``; the
executor's column, and the name CoXAM's task router looks for, is
``counterfactual_accuracy``. ``set_design_export`` resolved that alias, but
``set_design`` -- the direct API call every tutorial notebook makes -- stored the
name verbatim. Nothing raised: the DV was simply unrecognised, so CoXAM fell
back to its *forward* agent, and the run finished with no DV column at all. The
failure surfaced only much later, at ``analyze_iv_dv``, reporting a missing
column that no stage had ever been asked to produce.
"""

from __future__ import annotations

import pytest

from src.api import _resolve_dv_names


def test_the_planners_name_for_the_measure_reaches_the_runners_name():
    assert _resolve_dv_names({"counterfactual_simulation": ["continuous"]}) == {
        "counterfactual_accuracy": ["continuous"]
    }
    assert _resolve_dv_names({"forward_simulation": ["continuous"]}) == {
        "forward_accuracy": ["continuous"]
    }


def test_the_ui_label_resolves_too():
    """A researcher copying the label out of the planner should not be punished."""
    assert _resolve_dv_names({"Counterfactual Simulation": ["continuous"]}) == {
        "counterfactual_accuracy": ["continuous"]
    }


def test_the_short_forms_resolve():
    assert set(_resolve_dv_names({"counterfactual_sim": [1], "forward_sim": [1]})) == {
        "counterfactual_accuracy",
        "forward_accuracy",
    }


def test_an_already_canonical_name_is_untouched():
    dvs = {"counterfactual_accuracy": ["continuous"], "forward_accuracy": ["continuous"]}
    assert _resolve_dv_names(dvs) == dvs


def test_a_measure_the_simulator_does_not_produce_is_left_alone():
    """A study may legitimately record measures no agent simulates."""
    assert _resolve_dv_names({"trust_rating": ["continuous"]}) == {
        "trust_rating": ["continuous"]
    }


def test_set_design_and_add_dv_both_resolve(tmp_path):
    from src.api import xaikitTest

    study = xaikitTest(output_dir=tmp_path)
    study.auto_validate_design = False

    study.set_design(dvs={"counterfactual_simulation": ["continuous"]}, show=False)
    assert study.DVs == {"counterfactual_accuracy": ["continuous"]}

    study.add_dv("forward_simulation", ["continuous"])
    assert "forward_accuracy" in study.DVs


def test_the_resolved_name_is_what_routes_the_coxam_task():
    """The whole point: an unresolved name routes to the wrong agent."""
    from src.virtual_experiment_executor.experiment_simualtion.CoXAM.coxam_study_runner import (
        _tasks_implied_by_dvs,
    )

    assert _tasks_implied_by_dvs({"counterfactual_simulation": ["continuous"]}) == set()
    assert _tasks_implied_by_dvs(
        _resolve_dv_names({"counterfactual_simulation": ["continuous"]})
    ) == {"counterfactual"}

"""A counterfactual run must fill the columns its consumers read.

The two CoXAM tasks ask for different things -- forward asks the participant to
name a class, counterfactual asks them to name an *edit* -- so the counterfactual
runner emitted no ``agent_prediction`` at all. Every consumer keyed on that
column (the results table, the UI panel) therefore showed nothing for a run that
had produced 1080 perfectly good rows, with the DV filled and the success flag
present under a name nothing was looking for.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.virtual_experiment_executor.experiment_simualtion.CoXAM.coxam_counterfactual_runner import (
    _result_row,
)

DVS = {"counterfactual_accuracy": ["continuous"]}


def _info(*, original: int, counterfactual: int, success: bool):
    return {
        "condition": "DT",
        "shown_xai_type": "DT",
        "strategy": "change_path_dt",
        "depth": 2,
        "feature_changed": "a4",
        "delta": 3.06,
        "ai_prediction_original": original,
        "ai_prediction_counterfactual": counterfactual,
        "success": success,
        "invalid_under_condition": False,
        "time": 12.5,
    }


def test_a_counterfactual_row_reports_the_changed_ai_prediction():
    row = _result_row(
        {"phase": "testing"}, 0, _info(original=0, counterfactual=1, success=True), DVS
    )
    assert row["agent_prediction"] == 1
    assert row["ai_prediction"] == 0
    assert row["counterfactual_success"] is True


def test_an_unsuccessful_change_reports_the_unchanged_prediction():
    row = _result_row(
        {"phase": "testing"}, 0, _info(original=0, counterfactual=0, success=False), DVS
    )
    assert row["agent_prediction"] == 0
    assert row["agent_prediction"] == row["ai_prediction"]
    assert row["counterfactual_success"] is False


def test_agent_prediction_agrees_with_the_success_flag():
    """success is exactly 'the AI's prediction changed', verified on real runs."""
    for original, counterfactual, success in ((0, 1, True), (1, 0, True), (0, 0, False), (1, 1, False)):
        row = _result_row(
            {"phase": "testing"}, 0,
            _info(original=original, counterfactual=counterfactual, success=success),
            DVS,
        )
        changed = row["ai_prediction"] != row["agent_prediction"]
        assert changed == row["counterfactual_success"]


def test_the_dv_and_the_participant_response_are_both_present():
    """What a consumer needs to show a counterfactual trial."""
    row = _result_row(
        {"phase": "testing"}, 0, _info(original=0, counterfactual=1, success=True), DVS
    )
    for column in (
        "counterfactual_accuracy",   # the DV
        "agent_prediction",          # the shared column consumers read
        "counterfactual_success",    # did the edit flip the prediction
        "feature_changed", "delta",  # what the participant actually proposed
        "ai_prediction", "ai_prediction_counterfactual",
    ):
        assert column in row, column
    assert row["counterfactual_accuracy"] == 1

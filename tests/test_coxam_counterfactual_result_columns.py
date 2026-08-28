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


class _Data:
    def __init__(self, names):
        self.raw_feature_names = list(names)


class _Study:
    def __init__(self, names):
        self.data = _Data(names)


MUSHROOM_FEATURES = ("Bruises", "Height", "Width", "Shape", "Cap Diameter", "Gill")


def test_the_changed_feature_is_reported_by_name_not_by_position():
    """``a0`` is the corpus's column key, not something a reader can act on.

    A participant's proposed edit is only legible as "Cap Diameter -0.60"; the
    positional id leaves the results table and the UI showing an index whose
    mapping lived in one dict in the trial executor.
    """
    row = _result_row(
        {"phase": "testing"},
        0,
        _info(original=0, counterfactual=1, success=True),
        DVS,
        MUSHROOM_FEATURES,
    )
    assert row["feature_changed"] == "a4"
    assert row["feature_changed_name"] == "Cap Diameter"


def test_a_trial_that_changed_nothing_names_no_feature():
    info = {**_info(original=0, counterfactual=0, success=False), "feature_changed": None}
    row = _result_row({"phase": "testing"}, 0, info, DVS, MUSHROOM_FEATURES)
    assert row["feature_changed"] is None
    assert row["feature_changed_name"] is None


def test_an_unmappable_index_falls_back_to_the_raw_id():
    """Better an unresolved label than a confidently wrong feature name."""
    info = {**_info(original=0, counterfactual=1, success=True), "feature_changed": "a9"}
    assert _result_row({"phase": "testing"}, 0, info, DVS, MUSHROOM_FEATURES)[
        "feature_changed_name"
    ] == "a9"
    assert _result_row({"phase": "testing"}, 0, _info(original=0, counterfactual=1, success=True), DVS)[
        "feature_changed_name"
    ] == "a4"


def test_the_names_come_from_the_study_dataset_in_corpus_order():
    from src.virtual_experiment_executor.experiment_simualtion.CoXAM.coxam_counterfactual_runner import (
        _feature_display_names,
    )

    class _Bundle:
        app_id = "mushrooms"

    assert _feature_display_names(_Study(MUSHROOM_FEATURES), _Bundle()) == MUSHROOM_FEATURES
    # No prepared dataset -> the published corpus table is the fallback.
    assert _feature_display_names(object(), _Bundle()) == MUSHROOM_FEATURES


def test_the_row_reports_the_change_the_ai_actually_saw():
    """`delta` is the proposal; the env overshoots it and clamps it to bounds.

    Rendering `delta` as "the change" states a number the model was never shown
    -- and at a bound the two can differ by the whole overshoot margin.
    """
    info = {
        **_info(original=0, counterfactual=1, success=True),
        "delta": -0.6,
        "feature_value_before": 3.2,
        "feature_value_after": 2.45,
    }
    row = _result_row({"phase": "testing"}, 0, info, DVS, MUSHROOM_FEATURES)
    assert row["feature_value_before"] == 3.2
    assert row["feature_value_after"] == 2.45
    assert row["applied_delta"] == pytest.approx(-0.75)
    assert row["delta"] == -0.6


def test_a_trial_that_changed_nothing_reports_no_values():
    info = {
        **_info(original=0, counterfactual=0, success=False),
        "feature_changed": None,
        "feature_value_before": None,
        "feature_value_after": None,
    }
    row = _result_row({"phase": "testing"}, 0, info, DVS, MUSHROOM_FEATURES)
    assert row["feature_value_before"] is None
    assert row["applied_delta"] is None

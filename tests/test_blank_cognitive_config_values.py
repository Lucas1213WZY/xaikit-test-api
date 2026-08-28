"""A blank field in the design planner means "use the model default".

The planner exports every parameter slot it renders, so a slider the researcher
never touched arrives as ``cognitiveConfig: {"Retrieval Threshold": ""}`` rather
than being omitted. ``_apply_cognitive_model`` treated that empty string as a
supplied value and wrote it over the agent's own default, so
``study.cognitive_params`` carried ``memory_recall_threshold: ''``. Every CoXAM
counterfactual run built from such a design then died in
``_complete_cognitive_params`` on ``float('')`` -- including the API path the
bug-fix tutorials use, for both datasets and every simulation mode.
"""

from __future__ import annotations

import pytest

from src.experiment_planner.design_export import (
    _is_unset_cognitive_value,
    normalize_cognitive_params,
)


def test_a_blank_value_is_not_a_supplied_value():
    assert _is_unset_cognitive_value("")
    assert _is_unset_cognitive_value("   ")
    assert _is_unset_cognitive_value(None)


def test_zero_and_false_are_real_choices_and_are_kept():
    """A researcher who sets a parameter to 0 has configured it."""
    assert not _is_unset_cognitive_value(0)
    assert not _is_unset_cognitive_value(0.0)
    assert not _is_unset_cognitive_value(False)
    assert not _is_unset_cognitive_value("0")


def test_normalization_drops_blanks_and_keeps_the_rest():
    resolved = normalize_cognitive_params(
        "coxam", {"Retrieval Threshold": "", "Counterfactual Margin": "0.25"}
    )
    assert "memory_recall_threshold" not in resolved
    assert resolved["counterfactual_overshoot_fraction"] == pytest.approx(0.25)


def test_an_untouched_slider_leaves_the_agents_default_in_place():
    from src.api import xaikitTest
    from src.experiment_planner.design_export import _apply_cognitive_model

    class _Report:
        def add_warning(self, *args, **kwargs):
            pass

    class _Design:
        resolved_framework = "coxam"
        user_tasks = ("counterfactual",)
        cognitive_config = {"Retrieval Threshold": ""}
        report = _Report()

    class _Study:
        def set_cognitive_model(self, *, cognitive_model_id, cognitive_params):
            self.cognitive_params = cognitive_params

    study = _Study()
    _apply_cognitive_model(study, _Design())

    threshold = study.cognitive_params["memory_recall_threshold"]
    assert isinstance(threshold, float)
    # The counterfactual task's own default, not the forward one.
    assert threshold == pytest.approx(-0.75)


def test_a_filled_field_still_wins_over_the_default():
    class _Report:
        def add_warning(self, *args, **kwargs):
            pass

    class _Design:
        resolved_framework = "coxam"
        user_tasks = ("counterfactual",)
        cognitive_config = {"Retrieval Threshold": "-1.5"}
        report = _Report()

    class _Study:
        def set_cognitive_model(self, *, cognitive_model_id, cognitive_params):
            self.cognitive_params = cognitive_params

    from src.experiment_planner.design_export import _apply_cognitive_model

    study = _Study()
    _apply_cognitive_model(study, _Design())
    assert study.cognitive_params["memory_recall_threshold"] == pytest.approx(-1.5)


def test_every_resolved_parameter_survives_the_float_coercion_the_runner_applies():
    """The crash was one float() call downstream; pin the contract here."""
    from src.virtual_experiment_executor.experiment_simualtion.CoXAM.coxam_counterfactual_runner import (
        _complete_cognitive_params,
    )

    completed = _complete_cognitive_params(
        normalize_cognitive_params(
            "coxam", {"Retrieval Threshold": "", "Counterfactual Margin": "0.25"}
        )
    )
    assert all(isinstance(value, float) for value in completed.values())

    # Nothing configured at all resolves to the env's own defaults, which is
    # what None means here -- not an empty parameter dict.
    assert (
        _complete_cognitive_params(
            normalize_cognitive_params("coxam", {"Retrieval Threshold": ""})
        )
        is None
    )

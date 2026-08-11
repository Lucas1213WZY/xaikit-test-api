"""A DV can be globally valid and still be wrong for the chosen model.

Regression for a CoAX design declaring ``counterfactual_sim``. CoAX simulates
forward only, but ``counterfactual_accuracy`` is in ``SUPPORTED_DVS``, so the
framework check never ran and the DV passed through untouched. The runners then
fill *any* DV whose name contains "accuracy" with agent-vs-AI agreement --
forward-simulation correctness -- so the results carried forward numbers under
a "Counterfactual Simulation Accuracy" label, with nothing warning about it.

The design must still run (coerced, not rejected), but say so loudly.
"""

import warnings



from src.experiment_planner.design_export import parse_design_export


def _design(user_model: str, measure: str) -> dict:
    return {
        "researchQuestions": "na",
        "studyDesign": {
            "dependentVariables": [{"measure": measure, "name": "", "formula": ""}],
            "independentVariables": [
                {
                    "factor": "XAI Type",
                    "levelsOrRange": "None | Attribution | Importance",
                    "allocation": "Between-subjects",
                }
            ],
            "dataset": "Adult Income",
            "participantsPerCondition": 2,
            "trialsPerParticipant": 6,
        },
        "apparatus": [],
        "userModel": user_model,
    }


def _parse(raw):
    design = parse_design_export(raw)
    return design, design.report


def _dv_names(design) -> list[str]:
    return [dv["name"] for dv in design.dvs]


def test_coax_coerces_a_counterfactual_dv_to_forward():
    design, report = _parse(_design("CoAX", "counterfactual_sim"))
    assert _dv_names(design) == ["forward_accuracy"]
    assert design.dvs[0]["coerced"] is True
    # The original request is preserved so the UI can still show what was asked.
    assert design.dvs[0]["source_label"] == "counterfactual_sim"


def test_the_coercion_is_warned_about_not_silent():
    _, report = _parse(_design("CoAX", "counterfactual_sim"))
    messages = " ".join(issue.message for issue in report.warnings)
    assert "coax" in messages
    assert "does not simulate it" in messages
    # Naming a supported measure "unsupported" would read as a contradiction.
    assert "doesn't support" not in messages


def test_the_design_still_runs_rather_than_being_rejected():
    design, report = _parse(_design("CoAX", "counterfactual_sim"))
    assert design.simulatable_dvs == ["forward_accuracy"]
    assert not report.errors


def test_coax_leaves_a_forward_dv_alone():
    design, report = _parse(_design("CoAX", "forward_sim"))
    assert _dv_names(design) == ["forward_accuracy"]
    assert design.dvs[0]["coerced"] is False
    assert not report.warnings


def test_coxam_supports_both_so_counterfactual_passes_through():
    design, report = _parse(_design("CoXAM", "counterfactual_sim"))
    assert _dv_names(design) == ["counterfactual_accuracy"]
    assert design.dvs[0]["coerced"] is False


def test_an_entirely_unknown_measure_still_coerces_with_the_old_message():
    _, report = _parse(_design("CoAX", "trust_rating"))
    messages = " ".join(issue.message for issue in report.warnings)
    assert "doesn't support" in messages


def test_coax_runner_warns_when_asked_to_fill_a_non_forward_dv():
    # Guards the path a design export no longer reaches but a notebook can:
    # setting study.DVs directly bypasses parse-time coercion entirely.
    from src.virtual_experiment_executor.experiment_simualtion.CoAX import (
        coax_trial_executor,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        # No trials, so nothing is simulated -- the DV check runs first and is
        # what this asserts.
        coax_trial_executor.run_coax_experiment_executor(
            [], cognitive_model=None, dvs={"counterfactual_accuracy": ["continuous"]}
        )
    assert any(
        "forward-simulation values carrying another task's name" in str(item.message)
        for item in caught
    )

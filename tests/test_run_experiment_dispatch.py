"""``study.run_experiment()`` must run the agent that was selected.

``set_cognitive_model(cognitive_model_id="coxam")`` leaves ``cognitive_model``
as the placeholder, because coax and coxam are simulated by dedicated runners
rather than by the generic executor. Before this dispatch existed, selecting
either agent and calling ``run_experiment()`` silently produced placeholder
responses with no error -- a wrong answer that looked like a successful run.
"""

from types import SimpleNamespace

import pandas as pd
import pytest

from src.api import xaikitTest


@pytest.fixture
def study():
    """A study with just enough state to get past the ``_require_*`` guards."""
    test = xaikitTest()
    test.data = object()
    test.trials = [{"participantId": 1, "instanceId": 0}]
    return test


@pytest.fixture
def spy(monkeypatch):
    """Replace both agent runners with recorders; returns the call log."""
    calls = {}

    def record(name):
        def runner(study_arg, **kwargs):
            calls[name] = {"study": study_arg, **kwargs}
            return pd.DataFrame([{"ran": name}])

        return runner

    monkeypatch.setattr(
        "src.virtual_experiment_executor.experiment_simualtion.CoXAM."
        "coxam_study_runner.run_coxam_study",
        record("coxam"),
    )
    monkeypatch.setattr(
        "src.virtual_experiment_executor.experiment_simualtion.CoAX."
        "coax_study_runner.run_coax_study",
        record("coax"),
    )
    return calls


@pytest.mark.parametrize("agent", ["coax", "coxam"])
def test_selecting_an_agent_runs_that_agents_runner(study, spy, agent):
    study.set_cognitive_model(cognitive_model_id=agent)

    results = study.run_experiment(mode="whole_experiment")

    assert list(spy) == [agent], f"expected only {agent} to run, got {list(spy)}"
    assert results.iloc[0]["ran"] == agent
    assert spy[agent]["study"] is study
    assert spy[agent]["mode"] == "whole_experiment"
    assert spy[agent]["store"] is True


@pytest.mark.parametrize("agent", ["coax", "coxam"])
def test_results_are_stored_on_the_study(study, spy, agent):
    """save_results/analyze_iv_dv read simulated_results, so it must be set."""
    study.set_cognitive_model(cognitive_model_id=agent)
    study.run_experiment()

    assert study.simulated_results is not None
    assert study.simulated_results.iloc[0]["ran"] == agent


@pytest.mark.parametrize("agent", ["coax", "coxam"])
def test_selection_arguments_reach_the_runner(study, spy, agent):
    study.set_cognitive_model(cognitive_model_id=agent)
    study.run_experiment(
        mode="whole_condition",
        participant_id=7,
        condition_filter={"xai_type": "decision_tree"},
    )

    assert spy[agent]["participant_id"] == 7
    assert spy[agent]["condition_filter"] == {"xai_type": "decision_tree"}


def test_coxam_counterfactual_task_is_passed_through(study, spy):
    """The whole point of the passthrough: reaching the other coxam agent."""
    study.set_cognitive_model(cognitive_model_id="coxam")
    study.run_experiment(user_task="counterfactual_simulation")

    assert spy["coxam"]["user_task"] == "counterfactual_simulation"


def test_coxam_is_not_handed_an_explanation_pool(study, spy):
    """It fits its own surrogates; the combined table is the wrong schema."""
    study.set_cognitive_model(cognitive_model_id="coxam")
    study.run_experiment(explanation_pool=pd.DataFrame({"expMethod": ["shap"]}))

    assert "explanation_pool" not in spy["coxam"]


def test_coax_does_receive_the_explanation_pool(study, spy):
    pool = pd.DataFrame({"expMethod": ["shap"]})
    study.set_cognitive_model(cognitive_model_id="coax")
    study.run_experiment(explanation_pool=pool)

    assert spy["coax"]["explanation_pool"] is pool


def test_the_placeholder_path_is_untouched(study, spy, monkeypatch):
    """Baselines and explicit models must still use the generic executor."""
    called = {}

    def generic(**kwargs):
        called["generic"] = kwargs
        return pd.DataFrame([{"ran": "generic"}])

    monkeypatch.setattr("src.virtual_experiment_executor.run_experiment_executor", generic)
    monkeypatch.setattr(
        "src.api.ensure_prediction_coverage", lambda pool, **kwargs: pool
    )
    study.combined_explanations = pd.DataFrame({"expMethod": ["shap"]})
    study.trained_ai_model = object()
    study.data = SimpleNamespace(df=pd.DataFrame({"a": [1]}), label_column="label")

    results = study.run_experiment()

    assert not spy, "no agent runner should have been called"
    assert results.iloc[0]["ran"] == "generic"


def test_runner_options_without_an_agent_raise_rather_than_being_ignored(study, spy):
    """Silently dropping user_task would be the same class of bug as before."""
    with pytest.raises(TypeError, match="user_task"):
        study.run_experiment(user_task="counterfactual_simulation")


def test_agent_runners_list_matches_what_the_server_pipeline_routes():
    """The notebook API and the HTTP API must agree on which agents are real."""
    from server import pipeline

    assert set(xaikitTest._AGENT_RUNNERS) == (
        set(pipeline.COAX_FRAMEWORKS)
        | set(pipeline.COXAM_FRAMEWORKS)
        | set(pipeline.SIM2REAL_FRAMEWORKS)
    )


# -- UI cognitive parameters reach the simulation -------------------------


def test_ui_labels_resolve_to_each_agents_parameter_names():
    from src.experiment_planner.design_export import normalize_cognitive_params

    assert normalize_cognitive_params(
        "sim2real",
        {
            "Max Features Attended": "4",
            "Aggregation Strategy": "value_weighted",
            "Confidence Responsiveness": "-1.5",
        },
    ) == {
        "max_features_attended": 4,
        "aggregation": "value_weighted",
        "confidence_intercept": -1.5,
    }
    assert normalize_cognitive_params(
        "coxam", {"Retrieval Threshold": "-0.5", "Diffusion Noise": "0.5"}
    ) == {"memory_recall_threshold": -0.5, "decision_noise": 0.5}


def test_already_resolved_names_pass_through_unchanged():
    """API callers send internal names; normalising must not disturb them."""
    from src.experiment_planner.design_export import normalize_cognitive_params

    params = {"max_features_attended": 4, "aggregation": "value_weighted"}
    assert normalize_cognitive_params("sim2real", params) == params
    assert normalize_cognitive_params("sim2real", None) is None
    assert normalize_cognitive_params("sim2real", {}) == {}


def test_string_values_are_coerced_to_numbers_and_bools():
    from src.experiment_planner.design_export import _coerce_cognitive_value

    assert _coerce_cognitive_value("-1.5") == -1.5
    assert _coerce_cognitive_value("4") == 4 and isinstance(_coerce_cognitive_value("4"), int)
    assert _coerce_cognitive_value("true") is True
    assert _coerce_cognitive_value("value_weighted") == "value_weighted"
    assert _coerce_cognitive_value(0.3) == 0.3


def test_stored_cognitive_params_are_routed_to_the_runner():
    """`set_cognitive_model(cognitive_params=...)` used to be parsed and dropped:
    the run succeeded on defaults, so a UI-set parameter silently did nothing."""
    from src.api import xaikitTest

    assert xaikitTest._COGNITIVE_PARAM_KEYWORDS["coxam"] == "eval_params"
    assert xaikitTest._COGNITIVE_PARAM_KEYWORDS["sim2real"] == "cognitive_params"
    # CoAX is deliberately absent: its parameters are nested per condition.
    assert "coax" not in xaikitTest._COGNITIVE_PARAM_KEYWORDS


def test_the_three_ui_controls_reach_the_sim2real_model():
    from src.cognitive_models.cognitive_models.sim2real.gcm_strategies import (
        Sim2RealAttributionProjector,
    )
    from src.experiment_planner.design_export import normalize_cognitive_params
    from src.virtual_experiment_executor.experiment_simualtion.Sim2Real.sim2real_trial_executor import (
        build_sim2real_model,
    )

    params = normalize_cognitive_params(
        "sim2real",
        {
            "Max Features Attended": "4",
            "Aggregation Strategy": "value_weighted",
            "Confidence Responsiveness": "-1.5",
        },
    )
    model = build_sim2real_model(
        params,
        exp_property="faithful",
        projector=Sim2RealAttributionProjector.from_assets(),
    )
    assert model.max_features_attended == 4
    assert model.aggregation == "value_weighted"
    assert model.confidence_intercept == -1.5

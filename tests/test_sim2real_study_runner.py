"""The Sim2Real runner must drive the fitted attribution-sum model from trials.

Sim2Real is counterfactual-only and reads a fixed published corpus rather than
a study's own prepared dataset, so these tests exercise a real projector -- it
is 39 instances and cheap to load.
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.virtual_experiment_executor.experiment_simualtion.Sim2Real.sim2real_trial_executor import (
    FITTED_SIM2REAL_PARAMS,
    NEUTRAL_SIM2REAL_PARAMS,
    SIM2REAL_EXP_PROPERTIES,
    _canonical_exp_property,
    _trial_exp_property,
    build_sim2real_projector,
    fitted_sim2real_params,
    run_sim2real_experiment_executor,
    sim2real_available_instance_ids,
)
from src.virtual_experiment_executor.experiment_simualtion.Sim2Real.sim2real_study_runner import (
    run_sim2real_study,
)


@pytest.fixture(scope="module")
def projector():
    return build_sim2real_projector()


@pytest.fixture(scope="module")
def instance_ids(projector):
    return sim2real_available_instance_ids(projector)


def _trials(instance_ids, properties=SIM2REAL_EXP_PROPERTIES, per_property=3):
    return pd.DataFrame(
        [
            {
                "participantId": 1 + index % 2,
                "instanceId": instance_id,
                "xai_property": prop,
                "phase": "testing",
            }
            for index, (prop, instance_id) in enumerate(
                (prop, instance_id)
                for prop in properties
                for instance_id in instance_ids[:per_property]
            )
        ]
    )


# -- condition vocabulary --------------------------------------------------


@pytest.mark.parametrize("value", SIM2REAL_EXP_PROPERTIES)
def test_declared_properties_pass_through(value):
    assert _canonical_exp_property(value) == value


@pytest.mark.parametrize("value", ["Sparse Robust", "sparse-robust", "SPARSE_ROBUST"])
def test_property_spelling_is_normalized(value):
    assert _canonical_exp_property(value) == "sparse_robust"


@pytest.mark.parametrize("value", [None, "", "none", "baseline", "LIME"])
def test_baseline_condition_maps_to_none(value):
    """project(exp_property=None) is the unoptimized LIME baseline."""
    assert _canonical_exp_property(value) is None


def test_unknown_property_is_rejected():
    with pytest.raises(ValueError, match="Unknown Sim2Real explanation property"):
        _canonical_exp_property("attribution")


def test_condition_is_read_from_any_of_its_three_spellings():
    for column in ("xai_property", "explanation_property", "exp_property"):
        assert _trial_exp_property({column: "robust", "instanceId": 0}) == "robust"


def test_properties_match_the_support_matrix():
    from src.experiment_planner.support import load_support_matrix

    declared = load_support_matrix()["groups"]["xai_properties"]["sim2real"]
    assert list(SIM2REAL_EXP_PROPERTIES) == declared


# -- fitted parameters -----------------------------------------------------


def test_every_condition_has_fitted_parameters():
    assert set(FITTED_SIM2REAL_PARAMS) == set(SIM2REAL_EXP_PROPERTIES)


def test_fitted_params_override_the_neutral_defaults():
    """The neutral comparison_scale of 1.0 makes the model answer by tie-break."""
    for prop in SIM2REAL_EXP_PROPERTIES:
        params = fitted_sim2real_params(prop)
        assert params["comparison_scale"] > 1.0, prop
        # Everything the fitted table omits still comes from the defaults.
        assert params["lapse_rate"] == NEUTRAL_SIM2REAL_PARAMS["lapse_rate"]


def test_baseline_condition_falls_back_to_neutral_params():
    """No participant was fitted on the unoptimized baseline."""
    assert fitted_sim2real_params(None) == NEUTRAL_SIM2REAL_PARAMS


def test_explicit_overrides_beat_the_fitted_population():
    params = fitted_sim2real_params("robust", comparison_scale=3.5)
    assert params["comparison_scale"] == 3.5


# -- running ---------------------------------------------------------------


def test_each_trial_produces_one_scored_row(instance_ids):
    trials = _trials(instance_ids)
    results = run_sim2real_experiment_executor(
        trials, dvs={"counterfactual_accuracy": [0, 1]}
    )

    assert len(results) == len(trials)
    assert results["counterfactual_accuracy"].notna().all()
    assert set(results["agent_response_increases"].unique()) <= {0, 1}
    assert results["ground_truth_increases"].notna().all()


def test_conditions_are_simulated_with_their_own_parameters(instance_ids):
    """One shared model would give all four conditions identical behaviour."""
    trials = _trials(instance_ids, per_property=8)
    results = run_sim2real_experiment_executor(trials, dvs={"counterfactual_accuracy": [0, 1]})

    by_condition = results.groupby("exp_property")["probability_income_increases"].mean()
    assert len(by_condition) == len(SIM2REAL_EXP_PROPERTIES)
    assert by_condition.nunique() > 1, "conditions must not collapse onto one response"


def test_instances_outside_the_corpus_are_rejected_with_the_fix(instance_ids):
    trials = pd.DataFrame(
        [{"participantId": 1, "instanceId": 999_999, "xai_property": "faithful"}]
    )
    with pytest.raises(ValueError, match="outside the Sim2Real corpus"):
        run_sim2real_experiment_executor(trials)


def test_study_runner_stores_results(instance_ids):
    trials = _trials(instance_ids, per_property=2)
    study = SimpleNamespace(
        trials=trials.to_dict("records"),
        DVs={"counterfactual_accuracy": [0, 1]},
        simulated_results=None,
    )

    results = run_sim2real_study(study, mode="whole_experiment")

    assert len(results) == len(trials)
    assert study.simulated_results is not None
    assert "counterfactual_accuracy" in results


def test_study_runner_requires_generated_trials():
    study = SimpleNamespace(trials=[], DVs={}, simulated_results=None)
    with pytest.raises(RuntimeError, match="No generated trials"):
        run_sim2real_study(study)


def test_sim2real_is_declared_counterfactual_only():
    """Its model compares before/after confidence; there is no forward mode."""
    from src.experiment_planner.support import load_support_matrix

    spec = load_support_matrix()["cognitive_models"]["sim2real"]
    assert spec["tasks"] == ["counterfactual_simulation"]
    assert spec["dvs"] == ["counterfactual_accuracy"]


def test_pipeline_routes_sim2real_designs():
    from server import pipeline

    study = SimpleNamespace(design_export=SimpleNamespace(model_framework="sim2real"))
    assert pipeline.participant_runner(study) == "sim2real"


# -- strategy selection at simulation time ---------------------------------


@pytest.mark.parametrize(
    "strategy", ["attribution_sum", "sensitive_features", "salient_features"]
)
def test_every_strategy_simulates(instance_ids, strategy):
    from src.virtual_experiment_executor.experiment_simualtion.Sim2Real.sim2real_trial_executor import (
        run_sim2real_experiment_executor,
    )

    results = run_sim2real_experiment_executor(
        _trials(instance_ids, per_property=2),
        dvs={"counterfactual_accuracy": [0, 1]},
        strategy=strategy,
    )
    assert results["counterfactual_accuracy"].notna().all()


@pytest.mark.parametrize("strategy", ["sensitive_features", "salient_features"])
def test_an_exemplar_strategy_is_fitted_before_it_simulates(instance_ids, strategy):
    """The bug this guards: an unfitted exemplar model has no memory at all, so
    it scores every instance at 0.5, produces a zero delta, and answers a
    constant 0.5 -- silently, with no error and a plausible-looking accuracy."""
    from src.virtual_experiment_executor.experiment_simualtion.Sim2Real.sim2real_trial_executor import (
        run_sim2real_experiment_executor,
    )

    results = run_sim2real_experiment_executor(
        _trials(instance_ids, per_property=4),
        dvs={"counterfactual_accuracy": [0, 1]},
        strategy=strategy,
    )
    probabilities = results["probability_income_increases"]
    assert probabilities.nunique() > 1, f"{strategy} returned a constant response"
    assert not np.allclose(probabilities, 0.5), f"{strategy} never left the 0.5 prior"


def test_building_an_exemplar_strategy_without_a_projector_is_refused():
    """Rather than returning a model that answers 0.5 for the whole study."""
    from src.virtual_experiment_executor.experiment_simualtion.Sim2Real.sim2real_trial_executor import (
        build_sim2real_model,
    )

    with pytest.raises(ValueError, match="must be fitted on the training instances"):
        build_sim2real_model({}, exp_property="sparse", strategy="sensitive_features")


def test_an_unknown_simulation_strategy_is_rejected(projector):
    from src.virtual_experiment_executor.experiment_simualtion.Sim2Real.sim2real_trial_executor import (
        build_sim2real_model,
    )

    with pytest.raises(ValueError, match="Unknown sim2real strategy"):
        build_sim2real_model({}, strategy="telepathy", projector=projector)


def test_the_server_request_carries_the_sim2real_controls():
    """A UI cannot select a strategy the API does not expose."""
    from server.schemas import SimulationRequest

    request = SimulationRequest(sim2real_strategy="sensitive_features")
    assert request.sim2real_strategy == "sensitive_features"
    assert request.sim2real_params is None
    assert request.sim2real_normalize_by_i_max is True

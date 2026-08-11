"""Sim2Real picks its cognitive model per condition, not once per study.

``attribution_sum`` reads the explanation's attributions, so it needs the
changed feature to have a visible one. Measured on the corpus that holds for
robust (100% of trials), faithful (76%) and sparse_robust (55%), but for
``sparse`` it holds on 6.9% -- and there the model degenerates: confidence
delta of exactly 0.0, response probability pinned at 0.4695, the same answer on
every trial, 0.310 against a real human 0.757.

Against real human accuracy (faithful 0.708, sparse 0.757, robust 0.912,
sparse_robust 0.920, from assets/human_data/Sim2Real/sim2real_human_trials.csv),
routing sparse and sparse_robust to exemplar similarity takes mean absolute
error from 0.2015 to 0.0373.
"""

import pandas as pd
import pytest

from src.virtual_experiment_executor.experiment_simualtion.Sim2Real.sim2real_trial_executor import (
    AUTO_STRATEGY,
    FITTED_SIM2REAL_GCM_PARAMS,
    SIM2REAL_STRATEGY_BY_PROPERTY,
    build_sim2real_model,
    sim2real_params_for,
    sim2real_strategy_for,
)


# ---------------------------------------------------------------------------
# Strategy resolution
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "exp_property, expected",
    [
        ("sparse", "sensitive_features"),
        ("sparse_robust", "sensitive_features"),
        ("faithful", "attribution_sum"),
        ("robust", "attribution_sum"),
    ],
)
def test_auto_routes_each_condition_to_its_measured_strategy(exp_property, expected):
    assert sim2real_strategy_for(exp_property, AUTO_STRATEGY) == expected


def test_an_unknown_condition_falls_back_to_attribution_sum():
    assert sim2real_strategy_for("baseline", AUTO_STRATEGY) == "attribution_sum"
    assert sim2real_strategy_for(None, AUTO_STRATEGY) == "attribution_sum"


def test_an_explicit_strategy_overrides_every_condition():
    # Forcing one model across the study stays possible -- reproducing an older
    # run, or measuring one strategy everywhere.
    for exp_property in ("sparse", "robust", "faithful", "sparse_robust"):
        assert sim2real_strategy_for(exp_property, "attribution_sum") == "attribution_sum"
        assert sim2real_strategy_for(exp_property, "salient_features") == "salient_features"


# ---------------------------------------------------------------------------
# Parameter resolution
# ---------------------------------------------------------------------------


def test_an_exemplar_condition_starts_from_its_own_grid_searched_params():
    params = sim2real_params_for("sparse", "sensitive_features")
    assert params["sensitivity"] == 30.0
    assert params["k"] == 3


def test_the_two_exemplar_conditions_are_tuned_separately():
    # Same strategy, different sensitivity: the exemplar model ignores the
    # explanation, so one shared value would give both conditions the same
    # accuracy and only one of them can match its human target.
    assert (
        FITTED_SIM2REAL_GCM_PARAMS["sparse"]["sensitivity"]
        != FITTED_SIM2REAL_GCM_PARAMS["sparse_robust"]["sensitivity"]
    )


def test_attribution_conditions_do_not_pick_up_exemplar_params():
    # The attribution fit and the exemplar grid search are different parameter
    # spaces; mixing them would silently hand sensitivity/k to a model that has
    # no such parameters.
    assert sim2real_params_for("sparse", "attribution_sum") == {}
    assert sim2real_params_for("robust", "attribution_sum") == {}


def test_caller_overrides_win_over_the_condition_defaults():
    params = sim2real_params_for("sparse", "sensitive_features", {"sensitivity": 99.0})
    assert params["sensitivity"] == 99.0
    assert params["k"] == 3  # untouched keys survive


def test_the_resolved_params_actually_reach_the_model():
    model = build_sim2real_model(
        sim2real_params_for("sparse", "attribution_sum", {"max_features_attended": 7}),
        exp_property="sparse",
        strategy="attribution_sum",
    )
    assert model.max_features_attended == 7


# ---------------------------------------------------------------------------
# The routing survives a whole simulated study
# ---------------------------------------------------------------------------


def _condition_accuracy(strategy_kwargs) -> pd.Series:
    import json
    import tempfile
    from pathlib import Path

    from server.pipeline import (
        build_study,
        run_dataset_stage,
        run_explanations_stage,
        run_simulation_stage,
        run_trials_stage,
    )
    from server.schemas import (
        DatasetStageRequest,
        ExplanationStageRequest,
        SimulationRequest,
        TrialsStageRequest,
    )

    export = (
        Path(__file__).resolve().parents[1]
        / "tutorials"
        / "experiment_output"
        / "experiment-design_sim2real_minimal.json"
    )
    raw = json.loads(export.read_text())
    study = build_study(
        raw, project_name="sim2real-routing", output_dir=Path(tempfile.mkdtemp())
    )
    run_dataset_stage(study, DatasetStageRequest())
    run_trials_stage(study, TrialsStageRequest())
    run_explanations_stage(study, ExplanationStageRequest())
    run_simulation_stage(
        study,
        SimulationRequest(mode="whole_experiment", **strategy_kwargs),
        output_subdir="sim",
    )
    results = study.simulated_results
    testing = results[results["phase"].astype(str) == "testing"]
    return testing.groupby("xai_property")["counterfactual_accuracy"].mean()


HUMAN_ACCURACY = {
    "faithful": 0.708,
    "sparse": 0.757,
    "robust": 0.912,
    "sparse_robust": 0.920,
}


@pytest.mark.slow
def test_auto_beats_a_single_forced_strategy_against_real_human_accuracy():
    auto = _condition_accuracy({})
    forced = _condition_accuracy({"sim2real_strategy": "attribution_sum"})

    human = pd.Series(HUMAN_ACCURACY)
    auto_error = (auto - human).abs().mean()
    forced_error = (forced - human).abs().mean()

    assert auto_error < forced_error / 2
    # sparse is the condition the routing exists for: a constant predictor
    # before, a real fit after.
    assert abs(auto["sparse"] - human["sparse"]) < 0.05
    assert abs(forced["sparse"] - human["sparse"]) > 0.3
    # The conditions that already read attributions must be untouched.
    assert auto["faithful"] == forced["faithful"]
    assert auto["robust"] == forced["robust"]


# ---------------------------------------------------------------------------
# Response-level lapse: robust
# ---------------------------------------------------------------------------


def test_only_robust_carries_a_response_lapse():
    from src.virtual_experiment_executor.experiment_simualtion.Sim2Real.sim2real_trial_executor import (
        sim2real_lapse_for,
    )

    # robust is the one condition whose evidence is unambiguous on every trial,
    # so it is the one the model is otherwise perfect on.
    assert sim2real_lapse_for("robust") > 0.0
    for exp_property in ("faithful", "sparse", "sparse_robust", "baseline", None):
        assert sim2real_lapse_for(exp_property) == 0.0


def test_the_lapse_is_sized_to_the_human_error_rate():
    from src.virtual_experiment_executor.experiment_simualtion.Sim2Real.sim2real_trial_executor import (
        sim2real_lapse_for,
    )

    # A lapse answers at random, so it is wrong half the time: a perfect model
    # lands at 1 - lapse/2. Human robust accuracy is 0.912.
    expected = 1.0 - sim2real_lapse_for("robust") / 2.0
    assert abs(expected - HUMAN_ACCURACY["robust"]) < 0.01


@pytest.mark.slow
def test_robust_stops_simulating_a_perfect_participant():
    accuracy = _condition_accuracy({})
    # Was exactly 1.000 before the lapse; the human value is 0.912.
    assert accuracy["robust"] < 1.0
    assert abs(accuracy["robust"] - HUMAN_ACCURACY["robust"]) < 0.05


@pytest.mark.slow
def test_the_lapse_is_seeded_so_a_study_reproduces():
    first = _condition_accuracy({})
    second = _condition_accuracy({})
    pd.testing.assert_series_equal(first, second)


@pytest.mark.slow
def test_the_lapse_restores_between_participant_variance():
    import json
    import tempfile
    from pathlib import Path

    from server.pipeline import (
        build_study,
        run_dataset_stage,
        run_explanations_stage,
        run_simulation_stage,
        run_trials_stage,
    )
    from server.schemas import (
        DatasetStageRequest,
        ExplanationStageRequest,
        SimulationRequest,
        TrialsStageRequest,
    )

    export = (
        Path(__file__).resolve().parents[1]
        / "tutorials"
        / "experiment_output"
        / "experiment-design_sim2real_minimal.json"
    )
    study = build_study(
        json.loads(export.read_text()),
        project_name="sim2real-lapse",
        output_dir=Path(tempfile.mkdtemp()),
    )
    run_dataset_stage(study, DatasetStageRequest())
    run_trials_stage(study, TrialsStageRequest())
    run_explanations_stage(study, ExplanationStageRequest())
    run_simulation_stage(
        study, SimulationRequest(mode="whole_experiment"), output_subdir="sim"
    )
    results = study.simulated_results
    robust = results[
        (results["phase"].astype(str) == "testing")
        & (results["xai_property"] == "robust")
    ]
    # Without a lapse the model is deterministic, so every participant returns
    # identical answers and 30 participants carry no more information than one.
    per_participant = robust.groupby("participantId")["counterfactual_accuracy"].mean()
    assert per_participant.nunique() > 1

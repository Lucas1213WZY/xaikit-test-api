"""``/simulate`` with ``mode="diverse_participant"``.

Mocks the study the way ``test_server_pipeline_routing`` does: what is under
test is the pipeline's own routing -- which mode each run gets, which sampling
options are forwarded, and what the payload reports -- not the simulation.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
import pytest

from server.pipeline import run_simulation_stage
from server.schemas import SimulationRequest
from src.experiment_planner.design_export import DesignExport


def _design(framework="Sim2Real", baselines=None):
    # baseline_model_ids is derived from ml_proxy_baselines, so declare the
    # labels the UI would send rather than the resolved ids.
    return DesignExport(
        raw={},
        study_title="",
        research_questions=[],
        consent_text="",
        procedure_steps=[],
        ivs=[],
        model_framework=framework,
        ml_proxy_baselines=list(baselines or []),
    )


def _study(design, participant_parameters=None):
    study = MagicMock()
    study.design_export = design
    study.trained_ai_model = object()
    study.save_results.return_value = ("x.csv", "x.json")
    study.run_experiment.return_value = pd.DataFrame(
        {"phase": ["testing"], "step": [0], "participantId": [1], "forward_accuracy": [1]}
    )
    study.participant_parameters = participant_parameters
    study.participant_parameters_path = (
        None if participant_parameters is None else "parameters.csv"
    )
    return study


def test_default_mode_is_unchanged_and_sends_no_sampling_options():
    """Every existing caller -- the UI never sends mode at all -- is untouched."""
    study = _study(_design())
    run_simulation_stage(study, SimulationRequest(), output_subdir="x")

    kwargs = study.run_experiment.call_args.kwargs
    assert kwargs["mode"] == "whole_experiment"
    assert "sampling_seed" not in kwargs
    assert "sampling_replace" not in kwargs


def test_diverse_mode_forwards_the_sampling_options():
    study = _study(_design())
    request = SimulationRequest(mode="diverse_participant", sampling_seed=7)
    run_simulation_stage(study, request, output_subdir="x")

    kwargs = study.run_experiment.call_args.kwargs
    assert kwargs["mode"] == "diverse_participant"
    assert kwargs["sampling_seed"] == 7
    assert kwargs["sampling_replace"] is None


@pytest.mark.parametrize("framework", ["CoAX", "CoXAM", "Sim2Real"])
def test_every_agent_route_forwards_the_mode(framework):
    study = _study(_design(framework))
    study.trials = [{"participantId": 1, "xai_type": "attribution"}]
    run_simulation_stage(
        study, SimulationRequest(mode="diverse_participant", sampling_seed=2), output_subdir="x"
    )
    assert study.run_experiment.call_args.kwargs["sampling_seed"] == 2


def test_proxy_baselines_run_shared_instead_of_crashing():
    """A design naming mlProxyBaselines alongside an agent must still run.

    run_experiment refuses diverse_participant for a baseline -- it has no
    fitted human population -- so inheriting request.mode here would 500 the
    whole call for those designs.
    """
    study = _study(_design(baselines=["knn"]))
    result = run_simulation_stage(
        study, SimulationRequest(mode="diverse_participant"), output_subdir="x"
    )

    modes = [call.kwargs["mode"] for call in study.run_experiment.call_args_list]
    assert modes[0] == "diverse_participant"
    assert modes[1:] == ["whole_experiment"]
    assert result["mode"] == "diverse_participant"
    assert result["baseline_mode"] == "whole_experiment"


def test_the_primary_draw_survives_the_baseline_runs():
    """Each baseline run clears participant_parameters; restore the real one."""
    parameters = pd.DataFrame({"participantId": [1], "fitted_participant_id": [42]})
    study = _study(_design(baselines=["knn"]), participant_parameters=parameters)

    def _clear(*_args, **kwargs):
        if kwargs.get("mode") != "diverse_participant":
            study.participant_parameters = None
        return pd.DataFrame(
            {"phase": ["testing"], "step": [0], "participantId": [1], "forward_accuracy": [1]}
        )

    study.run_experiment.side_effect = _clear
    result = run_simulation_stage(
        study, SimulationRequest(mode="diverse_participant"), output_subdir="x"
    )
    assert study.participant_parameters is parameters
    assert result["participant_parameters"] == [
        {"participantId": 1, "fitted_participant_id": 42}
    ]


def test_payload_reports_the_saved_assignment_file():
    parameters = pd.DataFrame({"participantId": [1], "fitted_participant_id": [42]})
    study = _study(_design(), participant_parameters=parameters)
    result = run_simulation_stage(
        study, SimulationRequest(mode="diverse_participant"), output_subdir="x"
    )
    assert result["files"]["participant_parameters"] == "parameters.csv"
    assert result["files"]["csv"] == "x.csv"
    assert result["files"]["json"] == "x.json"


def test_shared_run_reports_no_assignment():
    study = _study(_design())
    result = run_simulation_stage(study, SimulationRequest(), output_subdir="x")
    assert result["participant_parameters"] == []
    assert result["files"]["participant_parameters"] is None
    assert result["baseline_mode"] is None

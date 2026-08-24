"""``study.run_experiment(mode="diverse_participant")`` end to end.

Drives the public API the notebooks call, rather than the runners underneath:
the mode has to survive dispatch, reach the runner, come back with provenance
attached, and be saved.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import src.api as api_module
from src.api import xaikitTest
from src.virtual_experiment_executor.experiment_simualtion.Sim2Real.sim2real_trial_executor import (
    sim2real_available_instance_ids,
)

PARTICIPANTS_PER_CONDITION = 4
PROPERTIES = ("faithful", "robust")


@pytest.fixture(scope="module")
def study_trials():
    instance_ids = sim2real_available_instance_ids(split="test")[:12]
    rows = []
    participant = 0
    for exp_property in PROPERTIES:
        for _ in range(PARTICIPANTS_PER_CONDITION):
            participant += 1
            for position, instance_id in enumerate(instance_ids):
                rows.append(
                    {
                        "participantId": participant,
                        "trialId": position,
                        "instanceId": int(instance_id),
                        "xai_property": exp_property,
                        "phase": "testing",
                    }
                )
    return rows


def _study(trials, tmp_path):
    study = xaikitTest(output_dir=str(tmp_path))
    study.set_design(
        iv_config={"xai_property": {"type": "between", "levels": list(PROPERTIES)}},
        dvs={"counterfactual_accuracy": ["continuous"]},
        show=False,
    )
    # _run_agent_experiment requires a prepared dataset even for Sim2Real,
    # whose runner then serves its stimuli from the published corpus.
    study.prepare_dataset(dataset_id="sim2real")
    study.trials = list(trials)
    study.set_cognitive_model(cognitive_model_id="sim2real")
    return study


def test_diverse_mode_runs_and_records_its_assignment(study_trials, tmp_path):
    study = _study(study_trials, tmp_path)
    results = study.run_experiment(mode="diverse_participant")

    assert not results.empty
    assert {"fitted_participant_id", "parameter_source", "parameter_pool"} <= set(
        results.columns
    )
    parameters = study.participant_parameters
    assert parameters is not None
    assert len(parameters) == PARTICIPANTS_PER_CONDITION * len(PROPERTIES)
    assert parameters["participantId"].is_unique
    assert parameters["fitted_participant_id"].nunique() > 1


def test_diverse_mode_beats_whole_experiment_on_variance(study_trials, tmp_path):
    study = _study(study_trials, tmp_path)

    def spread(mode):
        results = study.run_experiment(mode=mode)
        per_participant = results.groupby(["xai_property", "participantId"])[
            "counterfactual_accuracy"
        ].mean()
        return per_participant.groupby("xai_property").std()

    shared = spread("whole_experiment")
    diverse = spread("diverse_participant")
    assert shared["faithful"] == 0.0
    assert diverse["faithful"] > 0.0


def test_a_later_shared_run_does_not_keep_the_old_assignment(study_trials, tmp_path):
    study = _study(study_trials, tmp_path)
    study.run_experiment(mode="diverse_participant")
    assert study.participant_parameters is not None

    study.run_experiment(mode="whole_experiment")
    assert study.participant_parameters is None


def test_save_results_writes_the_assignment_but_still_returns_two_paths(
    study_trials, tmp_path
):
    """server/pipeline.py unpacks exactly two paths from this call."""
    study = _study(study_trials, tmp_path)
    study.run_experiment(mode="diverse_participant")

    csv_path, json_path = study.save_results(out_dir="simulated")
    assert Path(csv_path).is_file() and Path(json_path).is_file()

    parameters_path = Path(study.participant_parameters_path)
    assert parameters_path.is_file()
    assert parameters_path.name == "participant_parameters.csv"
    saved = pd.read_csv(parameters_path)
    assert len(saved) == PARTICIPANTS_PER_CONDITION * len(PROPERTIES)


def test_a_baseline_model_is_refused_with_a_usable_message(study_trials, tmp_path):
    """A model with no fitted humans behind it must not silently run shared."""
    study = _study(study_trials, tmp_path)
    study.set_cognitive_model(cognitive_model_id="knn")
    with pytest.raises(ValueError, match="no such population|research agent"):
        study.run_experiment(mode="diverse_participant")


def test_sampling_seed_reaches_the_runner(study_trials, tmp_path):
    study = _study(study_trials, tmp_path)
    study.run_experiment(mode="diverse_participant", sampling_seed=1)
    first = study.participant_parameters["fitted_participant_id"].tolist()
    study.run_experiment(mode="diverse_participant", sampling_seed=2)
    second = study.participant_parameters["fitted_participant_id"].tolist()
    assert first != second

    study.run_experiment(mode="diverse_participant", sampling_seed=1)
    assert study.participant_parameters["fitted_participant_id"].tolist() == first

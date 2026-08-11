"""A Sim2Real design export can now leave the dataset and apparatus instance
ids unspecified.

Sim2Real always runs against the same single published corpus (appId
``adult_sim2real``, built from the ``adult`` dataset), so a leaner export that
omits ``studyDesign.dataset`` and/or the apparatus's ``instanceIds``/
``trainingInstanceIds`` falls back to the exact values every prior export
declared explicitly (dataset ``adult``, testing instances 10-30, training
instances 0-9) rather than failing with "no dataset given". The fallback is
scoped to a design that declares the ``xai_property`` IV -- the same
discriminator ``DesignExport.resolved_framework`` already uses to detect a
Sim2Real design regardless of what ``userModel`` says -- so a non-Sim2Real
design with a genuinely blank dataset still raises as before.
"""

import json
from pathlib import Path

import pytest

from server.pipeline import (
    build_study,
    run_dataset_stage,
    run_simulation_stage,
    run_trials_stage,
)
from server.schemas import DatasetStageRequest, SimulationRequest, TrialsStageRequest
from src.experiment_planner.design_export import (
    SIM2REAL_DEFAULT_DATASET_ID,
    SIM2REAL_DEFAULT_TEST_INSTANCE_IDS,
    SIM2REAL_DEFAULT_TRAIN_INSTANCE_IDS,
    parse_design_export,
)

EXPORT_DIR = Path(__file__).resolve().parents[1] / "tutorials" / "experiment_output"
NEW_SHAPE_EXPORT = EXPORT_DIR / "experiment-design_sim2real_minimal.json"


def _sim2real_raw(*, dataset="", apparatus_params=None):
    return {
        "studyDesign": {
            "dataset": dataset,
            "independentVariables": [
                {
                    "factor": "XAI Property",
                    "levelsOrRange": "faithful | sparse | robust | sparse_robust",
                    "allocation": "Between-subjects",
                }
            ],
        },
        "apparatus": (
            [{"id": "a1", "params": apparatus_params}] if apparatus_params is not None else []
        ),
        "userModel": "CoAX",
    }


def test_blank_dataset_and_apparatus_default_to_the_sim2real_corpus():
    design = parse_design_export(_sim2real_raw())

    assert design.resolved_framework == "sim2real"
    assert design.dataset_id == SIM2REAL_DEFAULT_DATASET_ID
    assert design.apparatus_instance_ids == list(SIM2REAL_DEFAULT_TEST_INSTANCE_IDS)
    assert design.apparatus_training_instance_ids == list(SIM2REAL_DEFAULT_TRAIN_INSTANCE_IDS)


def test_explicit_values_still_win_over_the_default():
    design = parse_design_export(_sim2real_raw(
        dataset="Adult Income",
        apparatus_params={"appId": "adult_sim2real", "instanceIds": "0-4", "trainingInstanceIds": "5-6"},
    ))

    assert design.apparatus_instance_ids == [0, 1, 2, 3, 4]
    assert design.apparatus_training_instance_ids == [5, 6]


def test_a_non_sim2real_design_with_a_blank_dataset_is_unaffected():
    raw = {
        "studyDesign": {
            "dataset": "",
            "independentVariables": [
                {"factor": "XAI Type", "levelsOrRange": "decision_tree | logistic_regression", "allocation": "Within-subjects"}
            ],
        },
        "apparatus": [],
        "userModel": "CoXAM",
    }
    design = parse_design_export(raw)

    assert design.resolved_framework != "sim2real"
    assert design.dataset_id == ""
    assert design.apparatus_instance_ids == []


@pytest.mark.skipif(not NEW_SHAPE_EXPORT.is_file(), reason="fixture export not present")
def test_the_new_minimal_sim2real_export_runs_the_full_pipeline(tmp_path):
    """End-to-end regression against the real export that motivated this."""
    raw = json.loads(NEW_SHAPE_EXPORT.read_text())
    study = build_study(raw, project_name="sim2real-minimal", output_dir=tmp_path)

    dataset_result = run_dataset_stage(study, DatasetStageRequest())
    assert dataset_result["dataset"]["dataset_id"] == "adult"
    assert dataset_result["model"] is None

    trials_result = run_trials_stage(study, TrialsStageRequest())
    assert trials_result["counts"]["trials"] > 0
    assert trials_result["counts"]["training"] > 0
    assert trials_result["counts"]["testing"] > 0

    sim_result = run_simulation_stage(study, SimulationRequest(mode="whole_experiment"), output_subdir="sim")
    assert sim_result["counts"]["steps"] == trials_result["counts"]["trials"]

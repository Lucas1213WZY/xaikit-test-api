"""A design can name mlProxyBaselines alongside a real participant runner.

A design whose primary framework is Sim2Real (or CoXAM/CoAX) can also declare
``mlProxyBaselines`` (KNN/Decision Tree/MLP) for comparison. Those baselines
run through the generic executor, which needs a real ``trained_ai_model`` and
explanation table -- both of which the primary framework alone would
otherwise let the dataset/explanation stages skip (Sim2Real always; CoXAM/CoAX
for a corpus-covered dataset). ``run_simulation_stage`` then runs the primary
framework once and each declared baseline once more against the same trials,
tagging every run's rows with ``cognitive_model_id`` and concatenating them
into one ``study.simulated_results`` table instead of the last run silently
overwriting the others.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from server.pipeline import (
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
from src.experiment_planner.design_export import DesignExport

EXPORT_DIR = Path(__file__).resolve().parents[1] / "tutorials" / "experiment_output"
BASELINE_EXPORT = EXPORT_DIR / "experiment-design_sim2real_with_baselines.json"


def _sim2real_design(*, baseline_labels=()):
    """A minimal Sim2Real-resolving design. ``baseline_model_ids`` is a
    computed @property (from ``ml_proxy_baselines``), so the raw UI labels are
    set here rather than the canonical ids directly."""
    return DesignExport(
        raw={}, study_title="", research_questions=[], consent_text="",
        procedure_steps=[], ivs=[{"name": "xai_property", "iv_type": "between",
                                   "levels": ["faithful"], "randomization": None,
                                   "counterbalancing": "", "source_label": "XAI Property"}],
        model_framework="CoAX", dataset_id="adult",
        ml_proxy_baselines=list(baseline_labels),
    )


# ---------------------------------------------------------------------------
# run_dataset_stage / run_explanations_stage: training/explanations are not
# skipped for Sim2Real when baselines are declared
# ---------------------------------------------------------------------------


def test_dataset_stage_trains_for_sim2real_when_baselines_are_declared():
    study = MagicMock()
    design = _sim2real_design(baseline_labels=["KNN"])
    study.design_export = design
    study.prepare_dataset.return_value = MagicMock(
        dataset_id="adult", feature_names=["a"], model_feature_names=["a"],
        y_train=[0, 1], y_test=[0],
    )
    study.model_name = "mlp"
    study.test_accuracy.return_value = 0.9
    for attr in ("training_summary_table", "training_history_table"):
        getattr(study, attr).return_value.to_dict.return_value = []
    for attr in ("metrics_table", "confusion_matrix_table"):
        getattr(study, attr).return_value.reset_index.return_value.to_dict.return_value = []

    result = run_dataset_stage(study, DatasetStageRequest())

    study.train_AI_model.assert_called_once()
    assert result["model"] is not None
    assert result.get("model_skipped_reason") is None


def test_dataset_stage_still_skips_sim2real_training_without_baselines():
    """Backward compatibility: no mlProxyBaselines -> the original skip."""
    study = MagicMock()
    study.design_export = _sim2real_design()
    study.prepare_dataset.return_value = MagicMock(
        dataset_id="adult", feature_names=["a"], model_feature_names=["a"],
        y_train=[0, 1], y_test=[0],
    )

    result = run_dataset_stage(study, DatasetStageRequest())

    study.train_AI_model.assert_not_called()
    assert result["model"] is None


def test_explanations_stage_runs_for_sim2real_when_baselines_are_declared():
    study = MagicMock()
    study.design_export = _sim2real_design(baseline_labels=["KNN"])
    study.trained_ai_model = object()
    pool = pd.DataFrame({"expMethod": ["lime"]})
    study.explanations.return_value = ("path.csv", pool)
    study.explanation_paths = ["path.csv"]

    result = run_explanations_stage(study, ExplanationStageRequest())

    study.explanations.assert_called_once()
    assert result["skipped_reason"] is None


def test_balance_by_ai_prediction_default_excludes_sim2real_even_with_a_trained_model():
    """The apparatus's fixed, curated instance split must not be balanced by
    a baseline model's predictions -- a real regression this session, caused
    by forcing Sim2Real's dataset stage to train for its baselines."""
    study = MagicMock()
    study.design_export = DesignExport(
        raw={}, study_title="", research_questions=[], consent_text="",
        procedure_steps=[], ivs=[{"name": "xai_property", "iv_type": "between",
                                   "levels": ["faithful"], "randomization": None,
                                   "counterbalancing": "", "source_label": "XAI Property"}],
        model_framework="CoAX", participants_per_condition=1, trials_per_participant=10,
    )
    study.trained_ai_model = object()  # trained, for the baselines
    study.generate_trials.return_value = MagicMock(trials=pd.DataFrame({"participantId": [1]}))

    try:
        run_trials_stage(study, TrialsStageRequest())
    except Exception:
        pass  # only the balance_by_ai_prediction kwarg is under test here

    assert study.generate_trials.call_args.kwargs["balance_by_ai_prediction"] is False


# ---------------------------------------------------------------------------
# run_simulation_stage: primary run + one run per baseline, tagged & concatenated
# ---------------------------------------------------------------------------


def test_simulation_stage_runs_the_primary_and_every_baseline():
    """Sim2Real's own run is now also dispatched through study.run_experiment
    (like coax/coxam), so all three -- primary plus both baselines -- are
    three sequential study.run_experiment(...) calls here."""
    study = MagicMock()
    study.design_export = _sim2real_design(baseline_labels=["KNN", "Decision Tree"])
    study.cognitive_model_id = "sim2real"
    study.cognitive_model = object()
    study.cognitive_params = {}
    study.save_results.return_value = ("x.csv", "x.json")
    study.run_experiment.side_effect = [
        pd.DataFrame({"participantId": [1, 2], "phase": ["testing", "testing"]}),  # primary (sim2real)
        pd.DataFrame({"participantId": [3], "phase": ["testing"]}),  # knn
        pd.DataFrame({"participantId": [4], "phase": ["testing"]}),  # decision_tree
    ]

    result = run_simulation_stage(study, SimulationRequest(mode="whole_experiment"), output_subdir="x")

    assert result["baseline_model_ids"] == ["knn", "decision_tree"]
    assert result["cognitive_model_id"] == "sim2real"  # restored after the baseline loop
    # primary (2 rows) + 2 baselines (1 row each) = 4 rows total.
    assert result["counts"]["steps"] == 4
    assert sorted(study.simulated_results["cognitive_model_id"].unique()) == [
        "decision_tree", "knn", "sim2real",
    ]
    set_calls = [c.kwargs.get("cognitive_model_id") for c in study.set_cognitive_model.call_args_list]
    assert set_calls == ["knn", "decision_tree"]


def test_simulation_stage_skips_the_baseline_loop_with_no_declared_baselines():
    """Backward compatibility: no mlProxyBaselines -> exactly one run, as before."""
    study = MagicMock()
    study.design_export = _sim2real_design()
    study.cognitive_model_id = "sim2real"
    study.save_results.return_value = ("x.csv", "x.json")
    study.run_experiment.return_value = pd.DataFrame({"participantId": [1], "phase": ["testing"]})

    result = run_simulation_stage(study, SimulationRequest(mode="whole_experiment"), output_subdir="x")

    study.set_cognitive_model.assert_not_called()
    assert study.run_experiment.call_count == 1
    assert result["counts"]["steps"] == 1
    assert result["baseline_model_ids"] == []


# ---------------------------------------------------------------------------
# End-to-end regression against the real export that motivated this
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not BASELINE_EXPORT.is_file(), reason="fixture export not present")
def test_sim2real_design_with_baselines_runs_the_full_pipeline(tmp_path):
    from server.pipeline import build_study

    raw = json.loads(BASELINE_EXPORT.read_text())
    study = build_study(raw, project_name="sim2real-baselines", output_dir=tmp_path)
    assert study.design_export.baseline_model_ids == ["knn", "decision_tree", "mlp_baseline"]

    dataset_result = run_dataset_stage(
        study, DatasetStageRequest(max_epochs=15, target_score=0.5, check_every_epochs=5)
    )
    assert dataset_result["model"] is not None

    trials_result = run_trials_stage(study, TrialsStageRequest())
    assert trials_result["counts"]["trials"] > 0

    explanations_result = run_explanations_stage(study, ExplanationStageRequest())
    assert explanations_result["skipped_reason"] is None

    sim_result = run_simulation_stage(study, SimulationRequest(mode="whole_experiment"), output_subdir="sim")
    assert sim_result["runner"] == "sim2real"
    assert sim_result["baseline_model_ids"] == ["knn", "decision_tree", "mlp_baseline"]
    assert sorted(study.simulated_results["cognitive_model_id"].unique()) == [
        "decision_tree", "knn", "mlp_baseline", "sim2real",
    ]

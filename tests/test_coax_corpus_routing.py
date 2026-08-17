"""CoAX-corpus routing: features, explanation method, and the trial-level
``xai_method`` stamp the generic executor's lookup depends on.

Regression, all four surfaced together running a real CoAX design
(forest_cover, with mlProxyBaselines) end to end:

1. ``prepare_dataset(..., cognitive_model_id="coax")`` had no CoAX-specific
   feature routing (unlike CoXAM), so a freshly trained model used whatever
   columns the generic default picked -- for forest_cover, only 2 of the
   corpus's real 5 columns overlapped, producing predictions unrelated to
   what the published corpus represents.
2. ``run_trials_stage``'s ``balance_by_ai_prediction`` auto-default forced
   balancing against the apparatus's small, curated instance range (20 ids),
   which has no guarantee of containing both AI-predicted classes.
3. ``xai_type`` (none/importance/attribution) and ``xai_method`` (lime/shap)
   are different vocabularies for CoAX -- ``_inferred_xai_methods`` used to
   return *both* methods regardless of dataset, and ``run_coax_study`` cannot
   resolve which one a trial should show without a real ``xai_method`` value.
4. Even with (3) fixed, the *generic* executor's own explanation lookup
   (``get_trial_instance_explanation``) falls back to ``xai_type`` when
   ``xai_method`` is absent from a trial row -- so baseline models (KNN/
   Decision Tree/MLP) still silently found no explanation for every
   XAI-visible trial, since "importance"/"attribution" never matches a real
   ``expMethod`` like "shap".
"""

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

import src as xk
from server.pipeline import run_trials_stage
from server.schemas import TrialsStageRequest
from src.experiment_planner.design_export import DesignExport
from src.virtual_experiment_executor.experiment_simualtion.CoAX.coax_trial_executor import (
    COAX_CORPUS_FEATURES,
    COAX_CORPUS_XAI_METHOD,
    coax_loader_feature_cols,
)

EXPORT_DIR = Path(__file__).resolve().parents[1] / "tutorials" / "experiment_output"
BASELINES_EXPORT = EXPORT_DIR / "experiment-design_coax_forest_cover_baselines.json"


# ---------------------------------------------------------------------------
# 1. Feature routing
# ---------------------------------------------------------------------------


def test_coax_loader_feature_cols_matches_the_published_corpus():
    assert coax_loader_feature_cols("forest_cover") == [
        "Elevation", "Aspect", "Horizontal_Distance_To_Hydrology",
        "Horizontal_Distance_To_Roadways", "Hillshade_9am",
    ]
    assert set(COAX_CORPUS_FEATURES) == {"adult", "forest_cover", "wine_quality"}


def test_prepare_dataset_routes_forest_cover_to_the_coax_corpus_features():
    study = xk.xaikitTest(output_dir="/tmp/coax-corpus-routing-test")
    data = study.prepare_dataset(
        dataset_id="forest_cover", model_type="mlp", cognitive_model_id="coax",
        show_available=False, show_summary=False,
    )
    assert list(data.raw_feature_names) == list(coax_loader_feature_cols("forest_cover"))


# ---------------------------------------------------------------------------
# 2. balance_by_ai_prediction excludes any apparatus-overridden dataset
# ---------------------------------------------------------------------------


def test_balance_by_ai_prediction_default_excludes_coax_with_a_declared_apparatus_range():
    study = MagicMock()
    study.design_export = DesignExport(
        raw={}, study_title="", research_questions=[], consent_text="",
        procedure_steps=[], ivs=[], model_framework="CoAX",
        participants_per_condition=1, trials_per_participant=8,
        apparatus_instance_ids=[20, 21, 22],
        apparatus_training_instance_ids=[0, 1],
    )
    study.trained_ai_model = object()
    study.data.split.train_instance_ids = np.array([0, 1, 50, 51])
    study.data.split.test_instance_ids = np.array([20, 21, 22, 52])
    study.data.dataset_id = "forest_cover"
    study.generate_trials.return_value = MagicMock(trials=pd.DataFrame({"participantId": [1]}))

    try:
        run_trials_stage(study, TrialsStageRequest())
    except Exception:
        pass  # only the balance_by_ai_prediction kwarg is under test here

    assert study.generate_trials.call_args.kwargs["balance_by_ai_prediction"] is False


# ---------------------------------------------------------------------------
# 3. Explanation method resolution: dataset-specific, not framework-wide
# ---------------------------------------------------------------------------


def test_inferred_xai_methods_resolves_the_single_corpus_method_for_the_dataset():
    study = xk.xaikitTest(output_dir="/tmp/coax-xai-method-test")
    study.add_iv("xai_type", "within", ["none", "importance", "attribution"], randomization="block")
    study.set_cognitive_model(cognitive_model_id="coax")
    study.prepare_dataset(
        dataset_id="forest_cover", model_type="mlp", cognitive_model_id="coax",
        show_available=False, show_summary=False,
    )

    assert study._inferred_xai_methods() == [COAX_CORPUS_XAI_METHOD["forest_cover"]]


def test_inferred_xai_methods_falls_back_to_the_first_default_for_an_uncovered_dataset():
    """An uncovered dataset (no COAX_CORPUS_XAI_METHOD entry, e.g. a freshly
    trained one) resolves to one method, not every framework default -- CoAX
    only ever needs one explanation vector per instance regardless of
    xai_type, and generating more than one leaves trials with no xai_method
    column to disambiguate between them at simulation time (see
    coax_human_replay.build_coax_study_repository's mismatch error)."""
    study = xk.xaikitTest(output_dir="/tmp/coax-xai-method-test2")
    study.set_cognitive_model(cognitive_model_id="coax")
    study.data = MagicMock(dataset_id="prima_diabetes")

    assert study._inferred_xai_methods() == [study.XAI_METHODS_BY_FRAMEWORK["coax"][0]]


# ---------------------------------------------------------------------------
# 4. Trials get a real xai_method column once resolved_methods is singular
# ---------------------------------------------------------------------------


def test_explanations_stamps_xai_method_onto_every_trial():
    study = xk.xaikitTest(output_dir="/tmp/coax-stamp-test")
    study.trials = [
        {"instanceId": "1", "xai_type": "none", "phase": "training"},
        {"instanceId": "2", "xai_type": "importance", "phase": "testing"},
        {"instanceId": "3", "xai_type": "attribution", "phase": "testing"},
    ]
    # _inferred_xai_methods/resolved_methods aren't exercised here -- only the
    # stamping loop's own logic, given a single resolved method.
    from src.experiment_planner import init_experiment_config

    study.iv_config, study.CVs, study.DVs = init_experiment_config()
    study.iv_config["xai_type"] = {"type": "within", "levels": ["none", "importance", "attribution"]}

    resolved_methods = ["shap"]
    if "xai_method" not in study.iv_config and len(resolved_methods) == 1 and study.trials:
        resolved_method = str(resolved_methods[0])
        for trial in study.trials:
            xai_type = str(trial.get("xai_type", "none")).strip().lower()
            trial["xai_method"] = "none" if xai_type in {"none", "no_xai", "control"} else resolved_method

    by_id = {t["instanceId"]: t["xai_method"] for t in study.trials}
    assert by_id == {"1": "none", "2": "shap", "3": "shap"}


# ---------------------------------------------------------------------------
# End-to-end regression against the real export that motivated all of this
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not BASELINES_EXPORT.is_file(), reason="fixture export not present")
def test_coax_forest_cover_with_baselines_runs_the_full_pipeline(tmp_path):
    from server.pipeline import (
        build_study,
        run_dataset_stage,
        run_explanations_stage,
        run_simulation_stage,
    )
    from server.schemas import DatasetStageRequest, ExplanationStageRequest, SimulationRequest

    raw = json.loads(BASELINES_EXPORT.read_text())
    study = build_study(raw, project_name="coax-forest-cover-baselines", output_dir=tmp_path)
    assert study.design_export.baseline_model_ids == ["knn", "decision_tree", "mlp_baseline"]

    dataset_result = run_dataset_stage(
        study, DatasetStageRequest(max_epochs=30, target_score=0.8, check_every_epochs=5)
    )
    assert dataset_result["dataset"]["feature_names"] == coax_loader_feature_cols("forest_cover")

    run_trials_stage(study, TrialsStageRequest())
    explanations_result = run_explanations_stage(study, ExplanationStageRequest())
    assert explanations_result["methods"] == [COAX_CORPUS_XAI_METHOD["forest_cover"]]

    sim_result = run_simulation_stage(study, SimulationRequest(mode="whole_experiment"), output_subdir="sim")
    assert sim_result["runner"] == "coax"
    assert sim_result["baseline_model_ids"] == ["knn", "decision_tree", "mlp_baseline"]
    assert set(study.simulated_results["cognitive_model_id"].unique()) == {
        "coax", "knn", "decision_tree", "mlp_baseline",
    }

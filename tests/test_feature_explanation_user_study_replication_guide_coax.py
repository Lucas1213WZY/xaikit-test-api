from pathlib import Path
import json

from src.virtual_experiment_executor.experiment_simualtion.CoAX.coax_trial_executor import (
    COAX_DATA_DIR,
    COAX_EXPLANATIONS_DIR,
    make_coax_model,
    run_coax_experiment_executor,
)


def test_coax_notebook_uses_new_names():
    notebook_path = Path(__file__).resolve().parents[1] / "tutorials" / "feature_explanation_user_study_replication_guide_coax.ipynb"
    payload = json.loads(notebook_path.read_text(encoding="utf-8"))
    notebook_text = json.dumps(payload)

    assert "ai_predictions" in notebook_text
    assert "explanations" in notebook_text
    assert "prediction_pool" not in notebook_text
    assert "explanation_pool" not in notebook_text


def test_coax_runtime_helpers_are_available():
    model = make_coax_model(
        "AttributionSum",
        decay_param=0.5,
        retrieval_threshold=-0.3,
        sensitivity=15.0,
        scaling_factor=1.0,
        k=2,
        explanation_type="importance",
    )

    assert model.__class__.__name__ in {
        "AttributionSum",
        "SensitiveFeatures",
        "SalientFeatures",
        "ImportanceCategorization",
    }
    assert COAX_DATA_DIR.exists()
    assert COAX_EXPLANATIONS_DIR.exists()
    assert callable(run_coax_experiment_executor)

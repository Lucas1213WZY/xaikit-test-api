import pandas as pd
import pytest
from types import SimpleNamespace

from src.api import XAIKitTest
from src.experiment_planner.preview import _trial_preview_html, build_walkthrough_pages
from src.experiment_planner.protocol import normalize_study_protocol, validate_study_protocol


def _protocol():
    return {
        "study_title": "Explanation study",
        "research_questions": ["Does display type change prediction accuracy?"],
        "study_summary": "A short participant session.",
        "consent_text": "Participation is voluntary. Select continue to consent.",
        "start_survey_questions": ["How familiar are you with AI?"],
        "end_survey_questions": ["How difficult was the task?"],
        "procedure_steps": [
            {"title": "Consent", "kind": "consent", "description": "Read and choose."},
            {"title": "Start survey", "kind": "survey", "description": "Background."},
            {"title": "Practice", "kind": "practice", "description": "Try examples."},
            {"title": "Questions", "kind": "trials", "description": "Answer questions."},
            {"title": "End survey", "kind": "survey", "description": "Ratings."},
        ],
    }


def test_protocol_normalizes_multiline_fields():
    protocol = _protocol()
    protocol["research_questions"] = "RQ1\n\nRQ2"
    normalized = normalize_study_protocol(protocol)
    assert normalized["research_questions"] == ["RQ1", "RQ2"]
    assert validate_study_protocol(normalized) == []


def test_protocol_requires_consent_and_trial_step():
    protocol = _protocol()
    protocol["consent_text"] = ""
    protocol["procedure_steps"] = protocol["procedure_steps"][:3]
    problems = validate_study_protocol(protocol)
    assert "Add the consent information participants will see." in problems
    assert "Mark one procedure step with kind='trials'." in problems


def test_walkthrough_expands_trial_step_in_order():
    trials = pd.DataFrame([
        {"participantId": 1, "trialId": 1, "instanceId": 10},
        {"participantId": 1, "trialId": 2, "instanceId": 11},
    ])
    pages = build_walkthrough_pages(_protocol(), trials)
    page_types = [page["page_type"] for page in pages]
    assert page_types == [
        "researcher_review", "consent", "survey", "practice", "trial", "trial", "survey"
    ]
    assert pages[2]["questions"] == ["How familiar are you with AI?"]
    assert pages[-1]["questions"] == ["How difficult was the task?"]


def test_execution_gate_requires_preview_and_confirmation():
    study = XAIKitTest("gate_test")
    study.set_study_protocol(**_protocol())
    with pytest.raises(RuntimeError, match="Experiment execution is locked"):
        study.run_experiment(require_walkthrough_approval=True)
    with pytest.raises(RuntimeError, match="Preview the experiment walkthrough"):
        study.approve_walkthrough(confirmed=True)
    study.walkthrough_previewed = True
    study.approve_walkthrough(confirmed=True)
    assert study.walkthrough_approved is True


def test_training_preview_falls_back_when_explanations_only_cover_test_split():
    data = SimpleNamespace(
        df=pd.DataFrame({"feature": [1.0, 2.0], "target": [0, 1]}),
        label_column="target",
        raw_feature_names=["feature"],
    )
    explanation_pool = pd.DataFrame({
        "instanceId": [1],
        "expMethod": ["shap"],
        "pred": [1],
        "a0_i": [0.5],
    })

    rendered = _trial_preview_html(
        {
            "participantId": 1,
            "trialId": 1,
            "phase": "training",
            "instanceId": 0,
            "xai_method": "shap",
            "tested_w_xai": True,
        },
        explanation_pool,
        data,
        slide_number=1,
        slide_count=1,
        visualization="importance",
        top_n=1,
        class_labels=None,
    )

    assert "Could not render this explanation" not in rendered
    assert "outside the generated test-explanation pool" in rendered
    assert "feature" in rendered


def test_testing_preview_hides_ai_prediction_from_summary_and_plot():
    data = SimpleNamespace(
        df=pd.DataFrame({"feature": [1.0], "target": [0]}),
        label_column="target",
        raw_feature_names=["feature"],
    )
    explanation_pool = pd.DataFrame({
        "instanceId": [0],
        "expMethod": ["shap"],
        "pred": [1],
        "a0_i": [0.5],
    })

    rendered = _trial_preview_html(
        {
            "participantId": 1,
            "trialId": 1,
            "phase": "testing",
            "instanceId": 0,
            "xai_method": "shap",
            "tested_w_xai": True,
        },
        explanation_pool,
        data,
        slide_number=1,
        slide_count=1,
        visualization="importance",
        top_n=1,
        class_labels=["Type 1", "Type 2"],
    )

    assert "ai_prediction" not in rendered
    assert "AI prediction" not in rendered

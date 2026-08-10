"""server.pipeline.posthoc_for -- pairwise condition means and p-values.

Complements the existing analysis_for (one omnibus test per IV x DV): this
wires src/statistical_analyst's pairwise_condition_tests into the server so
the UI can annotate a bar plot with significance between specific condition
cells, e.g. Rules-with-XAI vs Weights-without-XAI.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
import pytest

from server.pipeline import posthoc_for
from server.serialization import analysis_payload
from src.experiment_planner.design_export import DesignExport
from src.statistical_analyst import analyze_iv_dv


def _study_with_results() -> MagicMock:
    study = MagicMock()
    study.design_export = DesignExport(
        raw={},
        study_title="",
        research_questions=[],
        consent_text="",
        procedure_steps=[],
        ivs=[{"name": "condition", "type": "between", "levels": []}],
        model_framework="coxam",
    )
    study.simulated_results = pd.DataFrame({
        "participantId": [1, 1, 2, 2, 3, 3],
        "phase": "testing",
        "condition": ["decision_tree", "hybrid"] * 3,
        "accuracy": [0.9, 0.5, 0.8, 0.6, 0.95, 0.55],
    })
    from src.statistical_analyst import pairwise_condition_tests

    study.pairwise_condition_tests.side_effect = (
        lambda *, dv, condition_cols, correction, phase: pairwise_condition_tests(
            study.simulated_results[study.simulated_results["phase"] == phase],
            value_col=dv,
            condition_cols=condition_cols,
            correction=correction,
        )
    )
    return study


def test_posthoc_defaults_condition_cols_to_the_designs_own_ivs():
    study = _study_with_results()
    payload = posthoc_for(study, dv="accuracy")
    assert payload["condition_cols"] == ["condition"]
    study.pairwise_condition_tests.assert_called_once()
    _, kwargs = study.pairwise_condition_tests.call_args
    assert kwargs["condition_cols"] == ["condition"]


def test_posthoc_reports_a_mean_and_p_value_per_pair():
    study = _study_with_results()
    payload = posthoc_for(study, dv="accuracy")
    assert payload["method"] == "holm"
    assert len(payload["comparisons"]) == 1
    row = payload["comparisons"][0]
    assert {"mean_a", "mean_b", "p_value", "p_value_corrected"} <= row.keys()
    assert row["mean_a"] != row["mean_b"]


def test_posthoc_requires_results_first():
    study = MagicMock()
    study.simulated_results = None
    with pytest.raises(ValueError, match="No simulation"):
        posthoc_for(study, dv="accuracy")


def test_posthoc_labels_conditions_with_the_coxam_vocabulary():
    study = _study_with_results()
    payload = posthoc_for(study, dv="accuracy")
    row = payload["comparisons"][0]
    assert row["condition_a_label"] == "Condition=Rules"
    assert row["condition_b_label"] == "Condition=Hybrid"
    assert payload["dv_label"] == "Accuracy"
    assert payload["condition_cols_label"] == ["Condition"]


def test_analysis_payload_labels_the_condition_and_iv_dv_names():
    responses = pd.DataFrame({
        "participantId": [1, 1, 2, 2, 3, 3],
        "phase": "testing",
        "xai_type": ["decision_tree", "hybrid"] * 3,
        "counterfactual_accuracy": [0.9, 0.5, 0.8, 0.6, 0.95, 0.55],
    })
    result = analyze_iv_dv(responses, iv="xai_type", dv="counterfactual_accuracy")
    payload = analysis_payload(result)
    assert payload["iv_label"] == "XAI Type"
    assert payload["dv_label"] == "Counterfactual Accuracy"
    labels = {row["xai_type"]: row["level_label"] for row in payload["descriptives"]}
    assert labels == {"decision_tree": "Rules", "hybrid": "Hybrid"}


def test_posthoc_raises_clearly_with_no_ivs_and_no_explicit_columns():
    study = _study_with_results()
    study.design_export = DesignExport(
        raw={},
        study_title="",
        research_questions=[],
        consent_text="",
        procedure_steps=[],
        ivs=[],
        model_framework="coxam",
    )
    with pytest.raises(ValueError, match="No condition columns"):
        posthoc_for(study, dv="accuracy")

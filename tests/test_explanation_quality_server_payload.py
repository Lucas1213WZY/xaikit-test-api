"""``/explanations`` reports explanation quality only when asked.

The payload shape is a contract with a UI this repo cannot see (the
design-planner is a separate project), so the default response must stay
exactly what it was: the ``quality`` key is *omitted*, not emptied.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest

from server.pipeline import _with_explanation_quality, run_explanations_stage
from server.schemas import ExplanationStageRequest
from src.experiment_planner.design_export import DesignExport

BASE_KEYS = {"combined_table", "rows", "by_method", "methods", "files", "skipped_reason"}


def _pool():
    return pd.DataFrame(
        {
            "expMethod": ["lime", "lime", "shap", "shap"],
            "faithfulness_aopc": [0.1, 0.3, 0.5, 0.7],
            "faithfulness_corr": [0.2, 0.2, 0.8, 0.8],
            "faithfulness_loss": [1.0, 1.0, 0.0, 0.0],
            "sparsity_nonzero": [2.0, 2.0, 4.0, 4.0],
            "sparsity_gini": [0.5, 0.5, 0.1, 0.1],
            "complexity_entropy": [0.3, 0.3, 1.2, 1.2],
            "robustness_lipschitz": [1.0, 1.0, 2.0, 2.0],
            "quality_note": ["", "", "", ""],
        }
    )


def test_a_default_request_gets_the_payload_it_always_got():
    payload = {key: None for key in BASE_KEYS}
    result = _with_explanation_quality(dict(payload), _pool(), ExplanationStageRequest())

    assert set(result) == BASE_KEYS
    assert "quality" not in result
    assert "quality_columns" not in result


def test_enabling_the_metrics_adds_per_method_means():
    result = _with_explanation_quality(
        {key: None for key in BASE_KEYS},
        _pool(),
        ExplanationStageRequest(quality_metrics=True),
    )

    assert set(result) == BASE_KEYS | {"quality", "quality_columns"}
    methods = {row["expMethod"] for row in result["quality"]}
    assert methods == {"lime", "shap"}
    lime = next(row for row in result["quality"] if row["expMethod"] == "lime")
    assert lime["faithfulness_aopc"] == pytest.approx(0.2)
    assert lime["n_rows"] == 2


def test_a_pool_without_quality_columns_reports_an_empty_list():
    """A corpus-loaded pool has no scores; that is not an error."""
    result = _with_explanation_quality(
        {key: None for key in BASE_KEYS},
        pd.DataFrame({"expMethod": ["lime"], "a0_i": [0.5]}),
        ExplanationStageRequest(quality_metrics=True),
    )
    assert result["quality"] == []


def test_the_request_reaches_the_study_unchanged():
    """Both new options must be forwarded, not silently dropped."""
    study = MagicMock()
    study.design_export = DesignExport(
        raw={}, study_title="", research_questions=[], consent_text="",
        procedure_steps=[], ivs=[], model_framework="CoAX",
    )
    study.trained_ai_model = object()
    study.data_by_dataset = {}
    study.explanations.return_value = ("combined.csv", _pool())
    study.explanation_paths = []

    run_explanations_stage(
        study,
        ExplanationStageRequest(quality_metrics=True, quality_metric_kwargs={"radius": 0.2}),
    )

    kwargs = study.explanations.call_args.kwargs
    assert kwargs["quality_metrics"] is True
    assert kwargs["quality_metric_kwargs"] == {"radius": 0.2}


def test_a_default_request_forwards_the_metrics_switched_off():
    study = MagicMock()
    study.design_export = DesignExport(
        raw={}, study_title="", research_questions=[], consent_text="",
        procedure_steps=[], ivs=[], model_framework="CoAX",
    )
    study.trained_ai_model = object()
    study.data_by_dataset = {}
    study.explanations.return_value = ("combined.csv", _pool())
    study.explanation_paths = []

    result = run_explanations_stage(study, ExplanationStageRequest())

    assert study.explanations.call_args.kwargs["quality_metrics"] is False
    assert "quality" not in result

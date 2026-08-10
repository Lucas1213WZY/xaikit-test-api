"""Turn API-layer return values into JSON the UI can render.

Every study stage returns pandas/numpy objects. Nothing here re-computes: it
only reshapes what the API layer already produced.
"""

from __future__ import annotations

import base64
import io
import math
from typing import Any, Optional

import numpy as np
import pandas as pd


def jsonable(value: Any) -> Any:
    """Recursively convert numpy/pandas values into JSON-safe Python ones."""
    if value is None:
        return None
    if isinstance(value, (str, bool, int)) and not isinstance(value, np.generic):
        return value
    if isinstance(value, float):
        return None if math.isnan(value) or math.isinf(value) else value
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, np.ndarray):
        return [jsonable(item) for item in value.tolist()]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if value is pd.NaT or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, pd.Series):
        return [jsonable(item) for item in value.tolist()]
    if isinstance(value, pd.DataFrame):
        return frame_records(value)
    if isinstance(value, dict):
        # Probability dicts are keyed by class label, which JSON needs as a string.
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(item) for item in value]
    return str(value)


def frame_records(
    frame: pd.DataFrame,
    *,
    limit: Optional[int] = None,
    offset: int = 0,
) -> list[dict[str, Any]]:
    """Rows of a DataFrame as JSON-safe dicts."""
    if frame is None or frame.empty:
        return []
    window = frame.iloc[offset : None if limit is None else offset + limit]
    return [
        {str(column): jsonable(value) for column, value in record.items()}
        for record in window.to_dict(orient="records")
    ]


def frame_payload(
    frame: pd.DataFrame,
    *,
    limit: Optional[int] = None,
    offset: int = 0,
) -> dict[str, Any]:
    """A DataFrame as ``{columns, rows, total, offset}`` for table rendering."""
    if frame is None:
        return {"columns": [], "rows": [], "total": 0, "offset": offset}
    return {
        "columns": [str(column) for column in frame.columns],
        "rows": frame_records(frame, limit=limit, offset=offset),
        "total": int(len(frame)),
        "offset": int(offset),
    }


def analysis_payload(result: Any) -> dict[str, Any]:
    """One ``IVDVAnalysisResult`` as descriptives plus the inferential test."""
    from src.result_visualizer import pretty

    inferential = result.inferential
    descriptives = frame_records(result.descriptives)
    for row in descriptives:
        if result.iv in row:
            row["level_label"] = pretty(row[result.iv])
    payload: dict[str, Any] = {
        "iv": result.iv,
        "iv_label": pretty(result.iv),
        "dv": result.dv,
        "dv_label": pretty(result.dv),
        "method": result.method,
        "descriptives": descriptives,
        "participant_level_data": frame_records(result.participant_level_data),
        "inferential": {"formula": getattr(inferential, "formula", None)},
    }
    # anova() carries a table; mixed_effects_model() carries a text summary.
    table = getattr(inferential, "table", None)
    if isinstance(table, pd.DataFrame):
        payload["inferential"]["table"] = frame_records(table.reset_index())
    summary = getattr(inferential, "summary", None)
    if isinstance(summary, str):
        payload["inferential"]["summary"] = summary
    groups = getattr(inferential, "groups", None)
    if groups is not None:
        payload["inferential"]["groups"] = str(groups)
    return payload


def posthoc_payload(result: Any) -> dict[str, Any]:
    """One ``PostHocResult`` as a table of pairwise condition comparisons."""
    from src.result_visualizer import prettify_condition_label

    comparisons = frame_records(result.table)
    for row in comparisons:
        if "condition_a" in row:
            row["condition_a_label"] = prettify_condition_label(row["condition_a"])
        if "condition_b" in row:
            row["condition_b_label"] = prettify_condition_label(row["condition_b"])
    return {
        "method": result.method,
        "comparisons": comparisons,
    }


def report_payload(report: Any) -> dict[str, Any]:
    """A ``ValidationReport`` as the UI's error/warning/reminder lists."""
    if report is None:
        return {}
    return {
        "stage": report.stage,
        "ok": bool(report.ok),
        "errors": [issue.to_dict() for issue in report.errors],
        "warnings": [issue.to_dict() for issue in report.warnings],
        "reminders": [issue.to_dict() for issue in report.reminders],
    }


def figure_png(figure: Any) -> str:
    """A matplotlib figure as a base64 data URI, closed after encoding."""
    import matplotlib.pyplot as plt

    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=144, bbox_inches="tight")
    plt.close(figure)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def design_payload(design: Any) -> dict[str, Any]:
    """The normalized design export as the UI's own summary view."""
    if design is None:
        return {}
    return {
        "study_title": design.study_title,
        "dataset_id": design.dataset_id,
        "model_framework": design.model_framework,
        "design_type": design.design_type,
        "participants_per_condition": design.participants_per_condition,
        "total_participants": design.total_participants,
        "trials_per_participant": design.trials_per_participant,
        "total_trials": design.total_trials,
        "ivs": jsonable(design.ivs),
        "between_ivs": jsonable(design.between_ivs),
        "within_ivs": jsonable(design.within_ivs),
        "cvs": jsonable(design.cvs),
        "dvs": jsonable(design.dvs),
        "simulatable_dvs": list(design.simulatable_dvs),
        "procedure_steps": jsonable(design.procedure_steps),
        "cognitive_config": jsonable(design.cognitive_config),
        "summary_rows": jsonable(design.summary_rows()),
        "parse_report": report_payload(getattr(design, "report", None)),
    }

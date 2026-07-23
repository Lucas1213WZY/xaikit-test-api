"""One-dependent-variable versus one-independent-variable analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from .anova import anova
from .mixed_effects import mixed_effects_model


@dataclass
class IVDVAnalysisResult:
    """Descriptive and inferential results for one DV against one IV."""

    iv: str
    dv: str
    method: str
    descriptives: pd.DataFrame
    inferential: Any
    participant_level_data: pd.DataFrame


def analyze_iv_dv(
    responses: pd.DataFrame,
    *,
    iv: str,
    dv: str,
    participant_column: str = "participantId",
) -> IVDVAnalysisResult:
    """Analyze one DV by one IV, accounting for repeated participants."""
    required = [iv, dv, participant_column]
    missing = [column for column in required if column not in responses]
    if missing:
        raise ValueError(f"Response data is missing columns: {missing}.")

    data = responses.copy()
    if "phase" in data:
        data = data[data["phase"].astype(str).str.lower() == "testing"]
    data[dv] = pd.to_numeric(data[dv], errors="coerce")
    data = data.dropna(subset=required)
    if data.empty:
        raise ValueError("No testing responses remain for the requested IV and DV.")
    if data[iv].nunique() < 2:
        raise ValueError(f"IV {iv!r} must contain at least two levels.")

    participant_data = (
        data.groupby([participant_column, iv], as_index=False, dropna=False)[dv]
        .mean()
    )
    descriptives = (
        participant_data.groupby(iv, as_index=False)[dv]
        .agg(["count", "mean", "std", "sem"])
        .reset_index()
    )

    formula = f"{_quote(dv)} ~ C({_quote(iv)})"
    repeated = participant_data.groupby(participant_column)[iv].nunique().max() > 1
    if repeated:
        inferential = mixed_effects_model(
            participant_data,
            formula=formula,
            groups=participant_column,
        )
        method = "mixed_effects"
    else:
        inferential = anova(participant_data, formula=formula)
        method = "one_way_anova"

    return IVDVAnalysisResult(
        iv=iv,
        dv=dv,
        method=method,
        descriptives=descriptives,
        inferential=inferential,
        participant_level_data=participant_data,
    )


def _quote(column: str) -> str:
    """Quote a DataFrame column for a Patsy formula."""
    return f"Q({column!r})"

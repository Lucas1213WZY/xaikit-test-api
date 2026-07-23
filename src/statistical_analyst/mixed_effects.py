"""Linear mixed-effects models via statsmodels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd
import statsmodels.formula.api as smf


@dataclass
class MixedEffectsResult:
    """Fitted linear mixed-effects model."""

    model: object
    formula: str
    groups: str
    summary: str


def mixed_effects_model(
    data: pd.DataFrame,
    formula: str,
    groups: str,
    re_formula: Optional[str] = None,
    vc_formula: Optional[dict] = None,
) -> MixedEffectsResult:
    """Fit a linear mixed-effects model (random intercepts and/or slopes).

    Args:
        data: Source DataFrame in long format.
        formula: Fixed-effects formula, e.g. ``"score ~ condition * time"``.
        groups: Column name identifying the random-effects grouping unit
            (e.g. subject/participant ID).
        re_formula: Random-effects formula for random slopes, e.g.
            ``"~time"`` for a random intercept + random slope on ``time``
            per group. Omit for a random-intercept-only model.
        vc_formula: Optional variance-components formula dict for crossed
            or nested random effects beyond a single grouping factor.

    Returns:
        MixedEffectsResult with the fitted model and its summary text.
    """
    model = smf.mixedlm(
        formula,
        data=data,
        groups=data[groups],
        re_formula=re_formula,
        vc_formula=vc_formula,
    ).fit()
    return MixedEffectsResult(model=model, formula=formula, groups=groups, summary=str(model.summary()))

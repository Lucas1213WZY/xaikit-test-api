"""ANOVA and fixed-effects regression via statsmodels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.regression.linear_model import RegressionResultsWrapper
from statsmodels.stats.anova import anova_lm


@dataclass
class AnovaResult:
    """ANOVA table plus the underlying fitted OLS model."""

    table: pd.DataFrame
    model: RegressionResultsWrapper
    formula: str


def anova(data: pd.DataFrame, formula: str, typ: int = 2) -> AnovaResult:
    """Fit an OLS model and run an ANOVA on it.

    Works for one-way, two-way/factorial, and ANCOVA designs — the design
    is entirely specified by the R-style ``formula``, e.g.:

    - One-way:            ``"score ~ C(group)"``
    - Two-way/factorial:  ``"score ~ C(factor_a) * C(factor_b)"``
    - ANCOVA (covariate): ``"score ~ C(group) + covariate"``

    Args:
        data: Source DataFrame.
        formula: Patsy/R-style formula.
        typ: Sum-of-squares type (1, 2, or 3). Type 2 is a reasonable
            default for balanced or mildly unbalanced designs without
            significant interactions; use type 3 for unbalanced designs
            that include interaction terms.

    Returns:
        AnovaResult with the ANOVA table and the underlying fitted model.
    """
    model = smf.ols(formula, data=data).fit()
    table = anova_lm(model, typ=typ)
    return AnovaResult(table=table, model=model, formula=formula)


@dataclass
class FixedEffectsResult:
    """Fitted fixed-effects (OLS) regression model."""

    model: RegressionResultsWrapper
    formula: str
    summary: str


def fixed_effects_model(
    data: pd.DataFrame,
    formula: str,
    cov_type: str = "nonrobust",
    cov_kwds: Optional[dict] = None,
) -> FixedEffectsResult:
    """Fit a fixed-effects (OLS) regression model.

    Fixed effects for a categorical grouping variable are added to the
    formula directly, e.g. ``"y ~ x + C(subject)"`` includes one dummy
    variable (fixed effect) per subject.

    Args:
        data: Source DataFrame.
        formula: Patsy/R-style formula, e.g. ``"y ~ x1 + x2 + C(group)"``.
        cov_type: Covariance estimator, e.g. "nonrobust", "HC1", "HC3",
            "cluster". See the ``statsmodels`` ``OLS.fit()`` docs.
        cov_kwds: Extra keyword arguments for the covariance estimator,
            e.g. ``{"groups": data["cluster_id"]}`` for ``cov_type="cluster"``.

    Returns:
        FixedEffectsResult with the fitted model and its summary text.
    """
    model = smf.ols(formula, data=data).fit(cov_type=cov_type, cov_kwds=cov_kwds)
    return FixedEffectsResult(model=model, formula=formula, summary=str(model.summary()))

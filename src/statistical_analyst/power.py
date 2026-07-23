"""Power analysis via ``statsmodels.stats.power``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from statsmodels.stats.power import FTestAnovaPower, TTestIndPower, TTestPower

_ANALYZERS = {
    "t_test": TTestPower,
    "t_test_ind": TTestIndPower,
    "anova": FTestAnovaPower,
}


@dataclass
class PowerAnalysisResult:
    """Result of solving a power analysis for its missing parameter."""

    test: str
    effect_size: Optional[float]
    nobs: Optional[float]
    alpha: float
    power: Optional[float]
    solved_for: str


def power_analysis(
    test: str = "t_test_ind",
    effect_size: Optional[float] = None,
    nobs: Optional[float] = None,
    alpha: float = 0.05,
    power: Optional[float] = None,
    k_groups: Optional[int] = None,
    **kwargs,
) -> PowerAnalysisResult:
    """Solve for the missing parameter in a power analysis.

    Exactly one of ``effect_size``, ``nobs``, ``power`` must be left as
    ``None`` — it is solved for from the other two plus ``alpha``. This
    covers the three common use cases: required sample size (solve for
    ``nobs``), achieved power (solve for ``power``), or minimum detectable
    effect (solve for ``effect_size``).

    Args:
        test: One of "t_test" (one-sample/paired), "t_test_ind"
            (independent two-sample), or "anova" (one-way, F-test).
        effect_size: Cohen's d (t-tests) or Cohen's f (anova).
        nobs: Sample size per group.
        alpha: Significance level.
        power: Target statistical power (1 - beta).
        k_groups: Number of groups, required for ``test="anova"``.
        **kwargs: Extra keyword arguments forwarded to the statsmodels
            solver (e.g. ``ratio`` for ``TTestIndPower``, ``alternative``).

    Returns:
        PowerAnalysisResult with all quantities filled in.
    """
    if test not in _ANALYZERS:
        raise ValueError(f"Unknown test '{test}'. Choose from {sorted(_ANALYZERS)}.")

    missing = [name for name, val in (("effect_size", effect_size), ("nobs", nobs), ("power", power)) if val is None]
    if len(missing) != 1:
        raise ValueError("Exactly one of effect_size, nobs, power must be None (the value to solve for).")
    solve_for = missing[0]

    analyzer = _ANALYZERS[test]()
    nobs_kw = "nobs1" if test == "t_test_ind" else "nobs"
    params = {"effect_size": effect_size, nobs_kw: nobs, "alpha": alpha, "power": power}
    if test == "anova":
        if k_groups is None:
            raise ValueError("k_groups is required for test='anova'.")
        params["k_groups"] = k_groups
    params.update(kwargs)
    params = {key: value for key, value in params.items() if value is not None}

    solved_value = analyzer.solve_power(**params)

    result = {"effect_size": effect_size, "nobs": nobs, "power": power}
    result[solve_for] = solved_value

    return PowerAnalysisResult(
        test=test,
        effect_size=result["effect_size"],
        nobs=result["nobs"],
        alpha=alpha,
        power=result["power"],
        solved_for=solve_for,
    )

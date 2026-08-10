"""Shared interval-estimation helpers for plots and comparison reports."""

from __future__ import annotations


def ci95_multiplier(n: int) -> float:
    """Student-t multiplier for a 95% CI on a mean of ``n`` observations.

    The normal 1.96 is wrong at the sample sizes typical here -- t(0.975,
    df=10) is 2.23, 14% wider -- so the exact value is used and only falls
    back to 1.96 if SciPy is unavailable.
    """
    if n < 2:
        return 0.0
    try:
        from scipy import stats
    except ImportError:  # pragma: no cover - SciPy ships with the environment
        return 1.959963984540054
    return float(stats.t.ppf(0.975, n - 1))

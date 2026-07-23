"""
Statistical analysis API layer.

Wraps statsmodels (and scipy for pairwise t-tests) behind a consistent set
of functions for experimental data analysis: descriptive statistics,
power analysis, ANOVA, post-hoc pairwise comparisons with multiple-
comparison correction, and fixed/mixed-effects regression models.
"""

from .anova import AnovaResult, FixedEffectsResult, anova, fixed_effects_model
from .descriptive import aggregate, describe, mean, median, sem, std, variance
from .iv_dv import IVDVAnalysisResult, analyze_iv_dv
from .mixed_effects import MixedEffectsResult, mixed_effects_model
from .posthoc import (
    PostHocResult,
    bonferroni_correction,
    pairwise_condition_tests,
    pairwise_ttests,
    tukey_hsd,
)
from .power import PowerAnalysisResult, power_analysis

__all__ = [
    "aggregate",
    "describe",
    "mean",
    "median",
    "sem",
    "std",
    "variance",
    "IVDVAnalysisResult",
    "analyze_iv_dv",
    "PowerAnalysisResult",
    "power_analysis",
    "AnovaResult",
    "FixedEffectsResult",
    "anova",
    "fixed_effects_model",
    "PostHocResult",
    "bonferroni_correction",
    "pairwise_condition_tests",
    "pairwise_ttests",
    "tukey_hsd",
    "MixedEffectsResult",
    "mixed_effects_model",
]

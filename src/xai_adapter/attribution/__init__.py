"""Feature attribution methods and plugin adapters."""

from .base import Attribution, CustomAttribution, GlobalImportance, LocalAttribution, make_attribution
from .captum import CaptumAttribution, DeepLift, GradientInput, IntegratedGradients
from .global_importance import SklearnFeatureImportance
from .lime import Lime
from .perturbation import KernelShap, LeaveOneFeatureOut
from .shap import ShapDeepExplainer, ShapGradientExplainer, ShapLinearExplainer, ShapTreeExplainer

__all__ = [
    "Attribution",
    "LocalAttribution",
    "GlobalImportance",
    "CustomAttribution",
    "make_attribution",
    "CaptumAttribution",
    "DeepLift",
    "IntegratedGradients",
    "GradientInput",
    "KernelShap",
    "LeaveOneFeatureOut",
    "Lime",
    "SklearnFeatureImportance",
    "ShapTreeExplainer",
    "ShapLinearExplainer",
    "ShapDeepExplainer",
    "ShapGradientExplainer",
]

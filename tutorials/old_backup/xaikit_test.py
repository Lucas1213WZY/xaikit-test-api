"""Compatibility import for the high-level XAIKit workflow API."""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.api import (
    ExplanationRunConfig,
    XAIKitTest,
    combine_explanation_tables,
    generate_xai_explanation_tables,
    get_xai_methods_from_design,
    init_explanation_run,
    predict_labels,
    xaikitTest,
)

__all__ = [
    "ExplanationRunConfig",
    "XAIKitTest",
    "xaikitTest",
    "combine_explanation_tables",
    "generate_xai_explanation_tables",
    "get_xai_methods_from_design",
    "init_explanation_run",
    "predict_labels",
]

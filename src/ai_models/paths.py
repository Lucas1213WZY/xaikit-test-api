"""Filesystem locations for AI model weights.

Model weights live under the repo's ``assets`` directory (not inside the
package source), organized by model type then cognitive agent:

    assets/model_weights/<model_type>/<cognitive_agent>/<file_name>

e.g. ``assets/model_weights/mlp/coxam/wine_quality_model_weights.pth``.
"""

from __future__ import annotations

from pathlib import Path

# paths.py -> src/ai_models/ -> src/ -> <repo root>
_REPO_ROOT = Path(__file__).resolve().parents[2]

MODEL_WEIGHTS_ROOT = _REPO_ROOT / "assets" / "model_weights"


def model_weights_dir(model_type: str, cognitive_agent: str) -> Path:
    """Directory holding weights for a (model_type, cognitive_agent) pair.

    Args:
        model_type: e.g. ``"mlp"`` or ``"xgboost"``.
        cognitive_agent: e.g. ``"coax"`` or ``"coxam"``.

    Returns:
        ``assets/model_weights/<model_type>/<cognitive_agent>`` as a Path.
        The directory is not created; callers that write should ``mkdir`` first.
    """
    return MODEL_WEIGHTS_ROOT / model_type / cognitive_agent

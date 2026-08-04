"""Simple machine-proxy cognitive baselines."""

from .decision_tree import DecisionTreeBaseline
from .knn import KNNBaseline
from .logistic_regression import LogisticRegressionBaseline
from .mlp import MLPBaseline


_BASELINE_ALIASES = {
    "knn": "knn",
    "knn_baseline": "knn",
    "decision_tree": "decision_tree",
    "decision_tree_baseline": "decision_tree",
    "dt": "decision_tree",
    "logistic_regression": "logistic_regression",
    "logistic_regression_baseline": "logistic_regression",
    "logistic": "logistic_regression",
    "lr": "logistic_regression",
    "mlp": "mlp_baseline",
    "mlp_baseline": "mlp_baseline",
}

#: The canonical baseline ids, for callers that need to list or report them.
BASELINE_MODEL_IDS = tuple(sorted(set(_BASELINE_ALIASES.values())))


def normalize_baseline_model_id(model_id: str) -> str:
    """Return the canonical baseline id, or the normalized input if unknown.

    Args:
        model_id: Id or alias to canonicalize.
    """
    normalized = model_id.lower().strip().replace("-", "_")
    return _BASELINE_ALIASES.get(normalized, normalized)


def is_baseline_model_id(model_id: str) -> bool:
    """Whether an id or alias refers to a registered machine proxy.

    Args:
        model_id: Id or alias to check.
    """
    normalized = model_id.lower().strip().replace("-", "_")
    return normalized in _BASELINE_ALIASES


def create_baseline_model(model_id: str, **kwargs):
    """Create a registered baseline cognitive model by API id.

    Args:
        model_id: Baseline to build -- ``knn``, ``decision_tree``,
            ``logistic_regression`` or ``mlp``. Aliases are accepted.
        **kwargs: Constructor arguments for the chosen baseline.
    """
    normalized = normalize_baseline_model_id(model_id)
    if normalized == "knn":
        return KNNBaseline(**kwargs)
    if normalized == "decision_tree":
        return DecisionTreeBaseline(**kwargs)
    if normalized == "logistic_regression":
        return LogisticRegressionBaseline(**kwargs)
    if normalized == "mlp_baseline":
        return MLPBaseline(**kwargs)
    raise ValueError(
        f"Unknown baseline cognitive model {model_id!r}. Choose from "
        "'knn', 'decision_tree', 'logistic_regression', or 'mlp'."
    )


__all__ = [
    "BASELINE_MODEL_IDS",
    "DecisionTreeBaseline",
    "KNNBaseline",
    "LogisticRegressionBaseline",
    "MLPBaseline",
    "create_baseline_model",
    "is_baseline_model_id",
    "normalize_baseline_model_id",
]

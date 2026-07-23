"""Evaluation and reporting for trained AI models.

Standalone functions (no orchestrator required): each takes an explicit
``ModelManager`` and/or ``PreparedDataset`` plus the training metadata it needs.
Presentation (tables/plots) lives here so the model layer owns its own reporting
and the high-level facade can stay a thin delegator.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from src.plotting import _patch_matplotlib_inline_rcparams

_PREFERRED_METRIC_COLUMNS = [
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "average_precision",
]


def positive_class_scores(predictions: Any, y: Sequence[Any], *, positive_label: int = 1) -> np.ndarray:
    """Reduce raw predictions/probabilities to positive-class scores."""
    preds = np.asarray(predictions)
    labels = np.unique(np.asarray(y))

    if preds.ndim == 2:
        positive_index = list(labels).index(positive_label) if positive_label in labels else -1
        return preds[:, positive_index]

    flat = preds.reshape(-1)
    if np.issubdtype(flat.dtype, np.floating):
        return flat
    return (flat == positive_label).astype(float)


def evaluate_model(
    manager: Any,
    data: Any,
    *,
    split: str = "both",
    positive_label: int = 1,
    threshold: float = 0.5,
    include_report: bool = False,
) -> dict[str, dict[str, Any]]:
    """Evaluate a trained model with classic classification metrics per split."""
    split = split.lower()
    results: dict[str, dict[str, Any]] = {}
    if split in {"train", "both"}:
        results["train"] = manager.evaluate_metrics(
            data.X_train, data.y_train,
            positive_label=positive_label, threshold=threshold, include_report=include_report,
        )
    if split in {"test", "both"}:
        results["test"] = manager.evaluate_metrics(
            data.X_test, data.y_test,
            positive_label=positive_label, threshold=threshold, include_report=include_report,
        )
    if not results:
        raise ValueError("split must be one of: 'train', 'test', or 'both'.")
    return results


def training_summary_table(training_info: dict[str, Any], model_name: Any, dataset_id: Any) -> pd.DataFrame:
    """Return a one-row summary of the latest training run."""
    if training_info is None:
        raise RuntimeError("Train a model before requesting the training summary.")
    summary = {key: value for key, value in training_info.items() if key != "history"}
    summary["model_type"] = model_name
    summary["dataset"] = dataset_id
    return pd.DataFrame([summary])


def training_history_table(training_info: dict[str, Any], model_name: Any = None, dataset_id: Any = None) -> pd.DataFrame:
    """Return the accuracy checkpoint history from the latest training run."""
    if training_info is None:
        raise RuntimeError("Train a model before requesting training history.")
    history = training_info.get("history", [])
    if not history:
        return training_summary_table(training_info, model_name, dataset_id)

    history_df = pd.DataFrame(history)
    target_metric = training_info.get("target_metric", "accuracy")
    target_score = training_info.get("target_score", training_info.get("target_accuracy"))
    if target_score is not None:
        history_df["target_metric"] = target_metric
        history_df["target_score"] = target_score
        history_df["reached_target"] = history_df[target_metric] >= target_score
        if target_metric == "accuracy":
            history_df["target_accuracy"] = target_score
    return history_df


def plot_training_history(training_info: dict[str, Any], *, ax: Any = None) -> Any:
    """Plot metric checkpoints from training with a target score."""
    history_df = training_history_table(training_info)
    target_metric = training_info.get("target_metric", "accuracy") if training_info is not None else "accuracy"
    if "epochs" not in history_df or target_metric not in history_df:
        raise RuntimeError(f"Training history does not contain epoch-level {target_metric} checkpoints.")

    import matplotlib
    _patch_matplotlib_inline_rcparams(matplotlib)
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))

    ax.plot(history_df["epochs"], history_df[target_metric], marker="o", label=f"Training {target_metric}")
    if "target_score" in history_df:
        target_score = float(history_df["target_score"].iloc[0])
        ax.axhline(target_score, color="black", linestyle="--", linewidth=1, label="Target")
    ax.set_title(f"Training {target_metric} Checkpoints")
    ax.set_xlabel("Epochs")
    ax.set_ylabel(target_metric)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.25)
    ax.legend()
    return ax


def metrics_table(metrics: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """Return scalar evaluation metrics as a compact split-by-metric table."""
    if not metrics:
        raise RuntimeError("Evaluate the model before requesting the metrics table.")
    rows = {
        split: {key: value for key, value in split_metrics.items() if np.isscalar(value) or value is None}
        for split, split_metrics in metrics.items()
    }
    metrics_df = pd.DataFrame.from_dict(rows, orient="index")
    ordered = [c for c in _PREFERRED_METRIC_COLUMNS if c in metrics_df.columns]
    ordered.extend(c for c in metrics_df.columns if c not in ordered)
    return metrics_df.loc[:, ordered]


def confusion_matrix_table(
    manager: Any,
    data: Any,
    *,
    split: str = "test",
    positive_label: int = 1,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Return a labeled confusion matrix for the requested split."""
    split = split.lower()
    if split == "train":
        X, y = data.X_train, data.y_train
    elif split == "test":
        X, y = data.X_test, data.y_test
    else:
        raise ValueError("split must be one of: 'train' or 'test'.")

    metrics = manager.evaluate_metrics(X, y, positive_label=positive_label, threshold=threshold)
    labels = metrics["labels"]
    return pd.DataFrame(
        metrics["confusion_matrix"],
        index=[f"actual_{label}" for label in labels],
        columns=[f"predicted_{label}" for label in labels],
    )


def plot_confusion_matrix(
    manager: Any,
    data: Any,
    *,
    split: str = "test",
    positive_label: int = 1,
    threshold: float = 0.5,
    ax: Any = None,
) -> Any:
    """Plot a labeled confusion matrix for the requested split."""
    import matplotlib
    _patch_matplotlib_inline_rcparams(matplotlib)
    import matplotlib.pyplot as plt

    matrix_df = confusion_matrix_table(
        manager, data, split=split, positive_label=positive_label, threshold=threshold,
    )
    if ax is None:
        _, ax = plt.subplots(figsize=(4.5, 4))

    image = ax.imshow(matrix_df.to_numpy(), cmap="Blues")
    ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(matrix_df.columns)), matrix_df.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(matrix_df.index)), matrix_df.index)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")
    ax.set_title(f"{split.title()} Confusion Matrix")

    for row_index, row in enumerate(matrix_df.to_numpy()):
        for col_index, value in enumerate(row):
            ax.text(col_index, row_index, int(value), ha="center", va="center")
    return ax


def plot_auc_curves(
    manager: Any,
    data: Any,
    *,
    split: str = "both",
    positive_label: int = 1,
    ax: Any = None,
) -> Any:
    """Plot ROC curves and AUC values for train/test predictions."""
    from sklearn.metrics import auc, roc_curve
    import matplotlib
    _patch_matplotlib_inline_rcparams(matplotlib)
    import matplotlib.pyplot as plt

    split = split.lower()
    split_data = []
    if split in {"train", "both"}:
        split_data.append(("train", data.X_train, data.y_train))
    if split in {"test", "both"}:
        split_data.append(("test", data.X_test, data.y_test))
    if not split_data:
        raise ValueError("split must be one of: 'train', 'test', or 'both'.")

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    for split_name, X, y in split_data:
        scores = positive_class_scores(manager.predict(X), y, positive_label=positive_label)
        if len(np.unique(y)) < 2:
            continue
        fpr, tpr, _thresholds = roc_curve(y, scores, pos_label=positive_label)
        ax.plot(fpr, tpr, linewidth=2, label=f"{split_name} AUC={auc(fpr, tpr):.3f}")

    ax.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1, label="chance")
    ax.set_title("ROC-AUC Curves")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.25)
    ax.legend()
    return ax

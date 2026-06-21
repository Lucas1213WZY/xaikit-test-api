"""Visualization helpers for project-standard explanation tables."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def explanation_value_columns(explanation_df: pd.DataFrame) -> List[str]:
    """Return attribution/importance columns in a0_i, a1_i, ... order."""
    value_cols = [
        col for col in explanation_df.columns
        if col.startswith("a") and col.endswith("_i") and col[1:-2].isdigit()
    ]
    return sorted(value_cols, key=lambda col: int(col[1:-2]))


def _feature_labels_for_explanation(
    value_cols: List[str],
    feature_names: Optional[List[str]] = None,
) -> List[str]:
    labels = []
    feature_names = feature_names or []
    for col in value_cols:
        feature_idx = int(col[1:-2])
        labels.append(feature_names[feature_idx] if feature_idx < len(feature_names) else col)
    return labels


def _select_explanation_row(
    explanation_df: pd.DataFrame,
    *,
    instance_id: Optional[Any] = None,
    method: Optional[str] = None,
) -> pd.Series:
    rows = explanation_df.copy()
    if method is not None and "expMethod" in rows:
        rows = rows[rows["expMethod"].astype(str) == str(method)]
    if instance_id is not None and "instanceId" in rows:
        rows = rows[rows["instanceId"].astype(str) == str(instance_id)]
    if rows.empty:
        raise ValueError("No explanation row matched the requested method/instance_id.")
    return rows.iloc[0]


def _raw_feature_values_for_instance(
    data: Optional[Any],
    instance_id: Optional[Any],
    feature_names: List[str],
) -> Dict[str, Any]:
    if data is None or instance_id is None or not hasattr(data, "df"):
        return {}
    try:
        row_idx = int(instance_id)
    except (TypeError, ValueError):
        row_idx = instance_id

    if row_idx in data.df.index:
        row = data.df.loc[row_idx]
    elif isinstance(row_idx, int) and 0 <= row_idx < len(data.df):
        row = data.df.iloc[row_idx]
    else:
        return {}
    return {feature: row.get(feature) for feature in feature_names}


def _format_feature_value(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if abs(number) >= 100:
        return f"{number:.0f}"
    if abs(number) >= 10:
        return f"{number:.1f}".rstrip("0").rstrip(".")
    return f"{number:.2f}".rstrip("0").rstrip(".")


def _format_feature_label(label: Any) -> str:
    """Format feature names for display while preserving common scientific labels."""
    text = str(label)
    special_labels = {
        "ph": "pH",
        "so2": "SO2",
        "co2": "CO2",
        "no2": "NO2",
    }
    key = text.replace(" ", "").lower()
    if key in special_labels:
        return special_labels[key]
    return text.replace("_", " ").title()


def _importance_for_prediction(attributions: Any, prediction: Any) -> np.ndarray:
    """
    Convert signed attributions into class-supporting importance values.

    If prediction is 0, negative attributions support that prediction and are
    shown as positive magnitudes. If prediction is 1, positive attributions
    support that prediction. Non-supporting attributions are set to 0.
    """
    values = np.asarray(attributions, dtype=float)
    try:
        pred = int(prediction)
    except (TypeError, ValueError):
        pred = 1

    if pred == 0:
        return np.where(values < 0, np.abs(values), 0.0)
    if pred == 1:
        return np.where(values > 0, values, 0.0)
    return np.maximum(values, 0.0)


def plot_explanation_visual(
    explanation_df: pd.DataFrame,
    data: Optional[Any] = None,
    *,
    visualization: str = "influence",
    instance_id: Optional[Any] = None,
    method: Optional[str] = None,
    feature_names: Optional[List[str]] = None,
    top_n: int = 5,
    class_labels: Optional[List[str]] = None,
    title: Optional[str] = None,
):
    """
    Plot a local explanation in the Attribute/Value/Influence or Importance format.

    `visualization="influence"` shows signed attribution direction by class.
    `visualization="importance"` shows only attributions that support the
    predicted class: negative attributions for prediction 0, positive
    attributions for prediction 1. Other attributions are set to 0.
    Feature rows preserve the predefined order from `feature_names` / aN_i.
    """
    import matplotlib.pyplot as plt

    visualization = visualization.lower()
    if visualization not in {"influence", "importance"}:
        raise ValueError("visualization must be 'influence' or 'importance'.")

    row = _select_explanation_row(
        explanation_df,
        instance_id=instance_id,
        method=method,
    )
    value_cols = explanation_value_columns(explanation_df)
    labels = _feature_labels_for_explanation(value_cols, feature_names)
    raw_values = _raw_feature_values_for_instance(data, row.get("instanceId"), labels)

    plot_df = pd.DataFrame({
        "feature": labels,
        "attribution": row[value_cols].astype(float).to_numpy(),
    })
    if visualization == "importance":
        plot_df["attribution"] = _importance_for_prediction(
            plot_df["attribution"],
            row.get("pred"),
        )
    plot_df["abs_attribution"] = plot_df["attribution"].abs()
    plot_df["raw_value"] = [raw_values.get(feature) for feature in plot_df["feature"]]

    shown = plot_df.head(top_n).copy()
    if len(plot_df) > top_n:
        remainder = plot_df.iloc[top_n:]
        others_value = remainder["attribution"].sum()
        shown = pd.concat([
            shown,
            pd.DataFrame([{
                "feature": "Others",
                "attribution": others_value,
                "abs_attribution": abs(others_value),
                "raw_value": None,
            }]),
        ], ignore_index=True)

    row_count = len(shown)
    y_positions = np.arange(row_count)
    fig_height = max(4, 0.55 * row_count + 1.6)
    fig, axes = plt.subplots(
        1,
        5,
        figsize=(10.5, fig_height),
        gridspec_kw={"width_ratios": [2.0, 1.0, 1.4, 2.7, 2.0]},
    )
    attr_ax, value_ax, raw_ax, panel_ax, pred_ax = axes

    for ax in (attr_ax, value_ax, raw_ax, pred_ax):
        ax.set_ylim(-0.8, row_count - 0.2)
        ax.invert_yaxis()
        ax.axis("off")

    attr_ax.set_title("Attribute", fontweight="bold", pad=10)
    value_ax.set_title("Value", fontweight="bold", pad=10)
    panel_ax.set_title(visualization.capitalize(), fontweight="bold", pad=10)

    for idx, item in shown.reset_index(drop=True).iterrows():
        is_others = item["feature"] == "Others"
        text_color = "#9a9a9a" if is_others else "#111111"
        attr_ax.text(1.0, idx, _format_feature_label(item["feature"]), ha="right", va="center", color=text_color, fontsize=11)
        value_ax.text(0.5, idx, _format_feature_value(item["raw_value"]), ha="center", va="center", fontsize=11)

    raw_ax.set_xlim(0, 1)
    for idx, item in shown.reset_index(drop=True).iterrows():
        raw_ax.barh(idx, 1.0, height=0.32, color="#dddddd", edgecolor="#eeeeee")
        raw_value = item["raw_value"]
        feature = item["feature"]
        fraction = 0.0
        if data is not None and hasattr(data, "df") and feature in data.df.columns and raw_value is not None and not pd.isna(raw_value):
            feature_min = float(pd.to_numeric(data.df[feature], errors="coerce").min())
            feature_max = float(pd.to_numeric(data.df[feature], errors="coerce").max())
            if feature_max > feature_min:
                fraction = (float(raw_value) - feature_min) / (feature_max - feature_min)
                fraction = float(np.clip(fraction, 0, 1))
        raw_ax.barh(idx, fraction, height=0.32, color="#000000")

    values = shown["attribution"].astype(float).to_numpy()
    if visualization == "importance":
        panel_values = values
        max_value = max(float(panel_values.max()), 1e-9)
        panel_ax.set_xlim(0, max_value * 1.1)
        panel_ax.barh(y_positions, panel_values, height=0.55, color="#208b27")
    else:
        max_value = max(float(np.abs(values).max()), 1e-9)
        panel_ax.set_xlim(-max_value * 1.15, max_value * 1.15)
        colors = ["#ef2d32" if value < 0 else "#3f8ee8" for value in values]
        panel_ax.barh(y_positions, values, height=0.55, color=colors)
        panel_ax.axvline(0, color="#222222", linewidth=0.9)
        labels_for_classes = class_labels or ["Type 1", "Type 2"]
        panel_ax.text(-max_value, row_count - 0.05, labels_for_classes[0], color="#ef2d32", ha="left", va="top")
        panel_ax.text(max_value, row_count - 0.05, labels_for_classes[-1], color="#3f8ee8", ha="right", va="top")

    panel_ax.set_ylim(-0.8, row_count - 0.2)
    panel_ax.invert_yaxis()
    panel_ax.set_yticks([])
    panel_ax.set_xticks([])
    for spine in panel_ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.3)

    pred_ax.set_xlim(0, 1)
    pred_ax.text(0.5, 0.30, "AI prediction", ha="center", va="center", fontweight="bold", fontsize=11)
    pred = row.get("pred", "")
    class_labels = class_labels or ["Type 1", "Type 2"]
    try:
        pred_label = class_labels[int(pred)]
    except (TypeError, ValueError, IndexError):
        pred_label = str(pred)
    pred_ax.add_patch(plt.Rectangle((0.12, 0.55), 0.76, 0.55, fill=False, linewidth=1.3, color="#111111"))
    pred_ax.text(0.5, 0.82, pred_label, ha="center", va="center", color="#3f8ee8", fontsize=12)

    if title:
        fig.suptitle(title, y=1.02, fontweight="bold")
    fig.tight_layout()
    return fig, axes


def plot_instance_explanation(
    explanation_df: pd.DataFrame,
    data: Optional[Any] = None,
    *,
    instance_id: Optional[Any] = None,
    method: Optional[str] = None,
    feature_names: Optional[List[str]] = None,
    top_n: int = 5,
    title: Optional[str] = None,
):
    """Plot the default local influence visualization for one explanation row."""
    return plot_explanation_visual(
        explanation_df,
        data,
        visualization="influence",
        instance_id=instance_id,
        method=method,
        feature_names=feature_names,
        top_n=top_n,
        title=title,
    )


def plot_global_explanation_importance(
    explanation_df: pd.DataFrame,
    *,
    method: Optional[str] = None,
    feature_names: Optional[List[str]] = None,
    top_n: Optional[int] = None,
    title: Optional[str] = None,
):
    """Plot mean class-supporting importance by feature across explanation rows."""
    import matplotlib.pyplot as plt

    rows = explanation_df.copy()
    if method is not None and "expMethod" in rows:
        rows = rows[rows["expMethod"].astype(str) == str(method)]
    if rows.empty:
        raise ValueError("No explanation rows matched the requested method.")

    value_cols = explanation_value_columns(rows)
    labels = _feature_labels_for_explanation(value_cols, feature_names)
    attribution_matrix = rows[value_cols].astype(float).to_numpy()
    if "pred" in rows:
        importance_matrix = np.vstack([
            _importance_for_prediction(attribution_matrix[row_idx], pred)
            for row_idx, pred in enumerate(rows["pred"])
        ])
    else:
        importance_matrix = np.maximum(attribution_matrix, 0.0)
    importances = importance_matrix.mean(axis=0)

    plot_df = pd.DataFrame({
        "feature": [_format_feature_label(label) for label in labels],
        "importance": importances.to_numpy(),
    })
    plot_df = plot_df.sort_values("importance", ascending=False)
    if top_n is not None:
        plot_df = plot_df.head(top_n)
    plot_df = plot_df.sort_values("importance")

    fig_height = max(3, 0.35 * len(plot_df) + 1.5)
    fig, ax = plt.subplots(figsize=(8, fig_height))
    ax.barh(plot_df["feature"], plot_df["importance"], color="#4c78a8")
    ax.set_xlabel("Mean class-supporting importance")
    ax.set_ylabel("Feature")
    ax.set_title(title or f"{method or 'All methods'} global explanation importance")
    fig.tight_layout()
    return fig, ax

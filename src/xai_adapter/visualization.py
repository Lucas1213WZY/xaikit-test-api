"""Visualization helpers for project-standard explanation tables."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Shared leaf helper; re-exported here for backward compatibility with callers
# that import it from this module.
from src.plotting import _patch_matplotlib_inline_rcparams


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
        rows = rows[
            rows["expMethod"].astype(str).str.lower() == str(method).lower()
        ]
    if instance_id is not None and "instanceId" in rows:
        rows = rows[rows["instanceId"].astype(str) == str(instance_id)]
    if rows.empty:
        method_rows = explanation_df
        if method is not None and "expMethod" in method_rows:
            method_rows = method_rows[
                method_rows["expMethod"].astype(str).str.lower()
                == str(method).lower()
            ]
        available_ids = (
            method_rows["instanceId"].dropna().astype(str).unique().tolist()[:10]
            if "instanceId" in method_rows
            else []
        )
        raise ValueError(
            "No explanation row matched "
            f"method={method!r}, instance_id={instance_id!r}. "
            f"Available instance IDs for this method include: {available_ids}."
        )
    return rows.iloc[0]


def _raw_feature_values_for_instance(
    data: Optional[Any],
    instance_id: Optional[Any],
    feature_names: List[str],
) -> Dict[str, Any]:
    row = _raw_row_for_instance(data, instance_id)
    if row is None:
        return {}
    values = {}
    for feature in feature_names:
        raw_feature = _raw_feature_name_for_display(feature, row.index)
        values[feature] = _decode_raw_feature_value(data, raw_feature, row.get(raw_feature))
    return values


def _raw_row_for_instance(data: Optional[Any], instance_id: Optional[Any]) -> Optional[pd.Series]:
    if data is None or instance_id is None or not hasattr(data, "df"):
        return None
    try:
        row_idx = int(instance_id)
    except (TypeError, ValueError):
        row_idx = instance_id

    if row_idx in data.df.index:
        return data.df.loc[row_idx]
    if isinstance(row_idx, int) and 0 <= row_idx < len(data.df):
        return data.df.iloc[row_idx]
    return None


def _raw_feature_name_for_display(feature: str, raw_columns: Any) -> str:
    """Map encoded labels like `Sex=Female` back to the raw dataframe column."""
    if feature in raw_columns:
        return feature
    if "=" in feature:
        base = feature.split("=", 1)[0]
        if base in raw_columns:
            return base
    return feature


def _decode_raw_feature_value(data: Any, raw_feature: str, value: Any) -> Any:
    """Return dataset category labels for raw categorical values when possible."""
    dataset = getattr(data, "dataset", None)
    raw_names = getattr(data, "raw_feature_names", getattr(data, "feature_names", []))
    categorical_options = getattr(dataset, "categorical_feature_options", {}) or {}
    if raw_feature not in raw_names:
        return value

    feature_idx = raw_names.index(raw_feature)
    if feature_idx not in categorical_options:
        return value

    try:
        option_idx = int(value)
    except (TypeError, ValueError):
        return value

    options = categorical_options[feature_idx]
    if 0 <= option_idx < len(options):
        return options[option_idx]
    return value


def _categorical_state_for_feature(
    data: Optional[Any],
    instance_id: Optional[Any],
    feature: str,
) -> Optional[List[bool]]:
    """Return per-circle selected state for categorical or one-hot features."""
    row = _raw_row_for_instance(data, instance_id)
    if row is None:
        return None

    raw_feature = _raw_feature_name_for_display(feature, row.index)
    dataset = getattr(data, "dataset", None)
    raw_names = getattr(data, "raw_feature_names", getattr(data, "feature_names", []))
    categorical_options = getattr(dataset, "categorical_feature_options", {}) or {}
    if raw_feature not in raw_names:
        return None

    feature_idx = raw_names.index(raw_feature)
    if feature_idx not in categorical_options:
        return None

    try:
        category_index = int(row.get(raw_feature))
    except (TypeError, ValueError):
        return None

    num_options = len(categorical_options[feature_idx])
    if num_options <= 0:
        return None
    if num_options == 1:
        return [category_index == 0, category_index == 1]
    return [option_idx == category_index for option_idx in range(num_options)]


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


def _feature_label_from_key(
    feature_key: str,
    feature_names: Optional[List[str]] = None,
) -> str:
    base = feature_key.split("=", 1)[0]
    try:
        idx = int(base[1:])
    except (TypeError, ValueError):
        return feature_key
    if feature_names and idx < len(feature_names):
        label = feature_names[idx]
    else:
        label = base
    if "=" in feature_key:
        return f"{label}={feature_key.split('=', 1)[1]}"
    return label


def _feature_index_from_key(feature_key: str) -> int:
    base = feature_key.split("=", 1)[0]
    return int(base[1:])


def _format_factor_value(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if abs(number) >= 100:
        return f"{number:.0f}"
    if abs(number) >= 10:
        return f"{number:.1f}".rstrip("0").rstrip(".")
    if abs(number) >= 1:
        return f"{number:.2f}".rstrip("0").rstrip(".")
    return f"{number:.3f}".rstrip("0").rstrip(".")


def _instance_array(instance: Any) -> np.ndarray:
    array = np.asarray(instance)
    if array.ndim == 2:
        if array.shape[0] != 1:
            raise ValueError("Instance-view plots expect one instance, not a batch.")
        return array[0]
    return array


def _value_for_feature_key(instance: np.ndarray, feature_key: str) -> float:
    idx = _feature_index_from_key(feature_key)
    if "=" in feature_key:
        _base, cat_idx = feature_key.split("=", 1)
        return 1.0 if int(instance[idx]) == int(cat_idx) else 0.0
    return float(instance[idx])


def _raw_display_value_for_index(
    data: Optional[Any],
    instance_id: Optional[Any],
    instance: np.ndarray,
    feature_idx: int,
    feature_names: List[str],
) -> Any:
    if feature_idx < len(feature_names):
        raw_values = _raw_feature_values_for_instance(data, instance_id, [feature_names[feature_idx]])
        if feature_names[feature_idx] in raw_values:
            return raw_values[feature_names[feature_idx]]
    if feature_idx < len(instance):
        return instance[feature_idx]
    return None


def _draw_raw_value_meter(
    ax: Any,
    y_pos: float,
    *,
    data: Optional[Any],
    instance_id: Optional[Any],
    instance: np.ndarray,
    feature_idx: int,
    feature_label: str,
) -> None:
    category_state = _categorical_state_for_feature(data, instance_id, feature_label)
    if category_state is not None:
        x_positions = np.linspace(0.16, 0.84, len(category_state))
        for selected, x_pos in zip(category_state, x_positions):
            ax.scatter(
                [x_pos],
                [y_pos],
                s=72,
                facecolors="#000000" if selected else "white",
                edgecolors="#000000" if selected else "#999999",
                linewidths=1.1,
                zorder=3,
            )
        return

    ax.barh(y_pos, 1.0, height=0.30, color="#dddddd", edgecolor="#eeeeee")
    fraction = 0.0
    if data is not None and hasattr(data, "df") and feature_label in getattr(data, "df").columns:
        raw_value = _raw_display_value_for_index(data, instance_id, instance, feature_idx, [feature_label])
        feature_min = float(pd.to_numeric(data.df[feature_label], errors="coerce").min())
        feature_max = float(pd.to_numeric(data.df[feature_label], errors="coerce").max())
        raw_number = pd.to_numeric(pd.Series([raw_value]), errors="coerce").iloc[0]
        if feature_max > feature_min and not pd.isna(raw_number):
            fraction = float(np.clip((float(raw_number) - feature_min) / (feature_max - feature_min), 0, 1))
    elif feature_idx < len(instance):
        value = float(instance[feature_idx])
        finite = np.asarray(instance, dtype=float)
        if finite.size and np.nanmax(finite) > np.nanmin(finite):
            fraction = float(np.clip((value - np.nanmin(finite)) / (np.nanmax(finite) - np.nanmin(finite)), 0, 1))
    ax.barh(y_pos, fraction, height=0.30, color="#000000")


def _surrogate_feature_names(surrogate: Any, feature_names: Optional[List[str]]) -> List[str]:
    if feature_names is not None:
        return list(feature_names)
    names = getattr(surrogate, "feature_names", None)
    if names is not None:
        return list(names)
    return []


_CLASS_EDGE_COLORS = ["#ef2d32", "#3f8ee8"]
_CLASS_FACE_COLORS = ["#fff1f2", "#eef4ff"]
_CLASS_EDGE_MUTED  = ["#ffaaaa", "#aaccff"]


def _class_label(class_labels: Optional[List[str]], class_index: Any) -> str:
    labels = class_labels or ["Type 1", "Type 2"]
    try:
        return str(labels[int(class_index)])
    except (TypeError, ValueError, IndexError):
        return str(class_index)


def _class_text_color(class_index: int) -> str:
    idx = max(0, min(int(class_index), len(_CLASS_EDGE_COLORS) - 1))
    return _CLASS_EDGE_COLORS[idx]


def _leaf_node_colors(class_index: int, *, active: bool) -> Tuple[str, str]:
    """Return (edge, face) colors for a decision-tree leaf node."""
    idx = max(0, min(int(class_index), len(_CLASS_EDGE_COLORS) - 1))
    if active:
        return _CLASS_EDGE_COLORS[idx], _CLASS_FACE_COLORS[idx]
    return _CLASS_EDGE_MUTED[idx], "white"


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
    show_ai_prediction: bool = True,
):
    """
    Plot a local explanation in the Attribute/Value/Influence or Importance format.

    `visualization="influence"` shows signed attribution direction by class.
    `visualization="importance"` shows only attributions that support the
    predicted class: negative attributions for prediction 0, positive
    attributions for prediction 1. Other attributions are set to 0.
    Feature rows preserve the predefined order from `feature_names` / aN_i.
    """
    import matplotlib

    _patch_matplotlib_inline_rcparams(matplotlib)
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
    width_ratios = [2.0, 1.0, 1.4, 2.7]
    if show_ai_prediction:
        width_ratios.append(2.0)
    fig, axes = plt.subplots(
        1,
        len(width_ratios),
        figsize=(10.5 if show_ai_prediction else 8.5, fig_height),
        gridspec_kw={"width_ratios": width_ratios},
    )
    attr_ax, value_ax, raw_ax, panel_ax = axes[:4]
    pred_ax = axes[4] if show_ai_prediction else None

    non_panel_axes = [attr_ax, value_ax, raw_ax]
    if pred_ax is not None:
        non_panel_axes.append(pred_ax)
    for ax in non_panel_axes:
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
        raw_value = item["raw_value"]
        feature = item["feature"]
        fraction = 0.0
        category_state = _categorical_state_for_feature(data, row.get("instanceId"), feature)
        if category_state is not None:
            num_options = len(category_state)
            x_positions = np.linspace(0.16, 0.84, num_options)
            for selected, x_pos in zip(category_state, x_positions):
                raw_ax.scatter(
                    [x_pos],
                    [idx],
                    s=92,
                    facecolors="#000000" if selected else "white",
                    edgecolors="#000000" if selected else "#999999",
                    linewidths=1.4,
                    zorder=3,
                )
            continue

        raw_ax.barh(idx, 1.0, height=0.32, color="#dddddd", edgecolor="#eeeeee")
        raw_feature = (
            _raw_feature_name_for_display(feature, data.df.columns)
            if data is not None and hasattr(data, "df")
            else feature
        )
        if data is not None and hasattr(data, "df") and raw_feature in data.df.columns and raw_value is not None and not pd.isna(raw_value):
            feature_min = float(pd.to_numeric(data.df[raw_feature], errors="coerce").min())
            feature_max = float(pd.to_numeric(data.df[raw_feature], errors="coerce").max())
            raw_number = pd.to_numeric(pd.Series([raw_value]), errors="coerce").iloc[0]
            if feature_max > feature_min and not pd.isna(raw_number):
                fraction = (float(raw_number) - feature_min) / (feature_max - feature_min)
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

    if pred_ax is not None:
        pred_ax.set_xlim(0, 1)
        pred_ax.text(0.5, 0.30, "AI prediction", ha="center", va="center", fontweight="bold", fontsize=11)
        pred = row.get("pred", "")
        class_labels = class_labels or ["Type 1", "Type 2"]
        try:
            pred_label = class_labels[int(pred)]
        except (TypeError, ValueError, IndexError):
            pred_label = str(pred)
        pred_ax.add_patch(plt.Rectangle((0.12, 0.55), 0.76, 0.55, fill=False, linewidth=1.3, color="#111111"))
        pred_ax.text(0.5, 0.82, pred_label, ha="center", va="center", color=_class_text_color(int(pred)), fontsize=12)

    if title:
        fig.suptitle(title, y=1.02, fontweight="bold")
    fig.tight_layout()
    return fig, axes


def plot_logistic_regression_instance_view(
    surrogate: Any,
    instance: Any,
    data: Optional[Any] = None,
    *,
    instance_id: Optional[Any] = None,
    feature_names: Optional[List[str]] = None,
    class_labels: Optional[List[str]] = None,
    top_n: Optional[int] = None,
    title: Optional[str] = None,
):
    """
    Plot the CoXAM-style factor/partial-sum instance view for a fitted logistic surrogate.

    The displayed rows come from the surrogate's learned coefficients, so sparse
    variants naturally show fewer factors while dense variants show all factors.
    """
    import matplotlib

    _patch_matplotlib_inline_rcparams(matplotlib)
    import matplotlib.pyplot as plt

    if not getattr(surrogate, "is_fitted", False):
        raise RuntimeError("Logistic surrogate must be fitted before plotting.")
    if not hasattr(surrogate, "get_coefficients") or not hasattr(surrogate, "get_intercept"):
        raise TypeError("surrogate must expose get_coefficients() and get_intercept().")

    instance_array = _instance_array(instance)
    names = _surrogate_feature_names(surrogate, feature_names)
    coefficients = list(surrogate.get_coefficients().items())
    if top_n is not None:
        coefficients = sorted(
            coefficients,
            key=lambda item: abs(float(item[1]) * _value_for_feature_key(instance_array, item[0])),
            reverse=True,
        )[:top_n]

    rows = []
    for feature_key, coefficient in coefficients:
        feature_idx = _feature_index_from_key(feature_key)
        label = _feature_label_from_key(feature_key, names)
        feature_value = _value_for_feature_key(instance_array, feature_key)
        display_value = _raw_display_value_for_index(data, instance_id, instance_array, feature_idx, names)
        rows.append({
            "feature_key": feature_key,
            "feature_idx": feature_idx,
            "label": label,
            "value": display_value,
            "factor": float(coefficient),
            "partial": float(coefficient) * feature_value,
        })

    intercept = float(surrogate.get_intercept())
    rows.append({
        "feature_key": "intercept",
        "feature_idx": -1,
        "label": "Adjustment",
        "value": None,
        "factor": intercept,
        "partial": intercept,
    })

    row_count = len(rows)
    fig_height = max(4.2, 0.58 * row_count + 1.5)
    fig, axes = plt.subplots(
        1,
        7,
        figsize=(13.6, fig_height),
        gridspec_kw={"width_ratios": [1.8, 0.9, 1.35, 0.32, 1.15, 1.5, 2.2]},
    )
    attr_ax, value_ax, meter_ax, symbol_ax, factor_ax, partial_ax, pred_ax = axes
    y_positions = np.arange(row_count)

    for ax in (attr_ax, value_ax, meter_ax, symbol_ax, factor_ax, partial_ax, pred_ax):
        ax.set_ylim(-0.8, row_count - 0.2)
        ax.invert_yaxis()
        ax.axis("off")

    attr_ax.set_title("Attribute", fontweight="bold", pad=10)
    value_ax.set_title("Value", fontweight="bold", pad=10)
    factor_ax.set_title("Factor", fontweight="bold", pad=10)
    partial_ax.set_title("Partial Sum", fontweight="bold", pad=10)
    symbol_ax.set_title("x", fontweight="bold", pad=10)

    meter_ax.set_xlim(0, 1)
    factor_ax.set_xlim(0, 1)
    partial_ax.set_xlim(0, 1)

    factor_ax.add_patch(plt.Rectangle((0.02, -0.48), 0.96, row_count - 0.02, facecolor="#afd1f5", edgecolor="#111111", linewidth=1.6))
    partial_ax.add_patch(plt.Rectangle((0.02, -0.48), 0.96, row_count - 0.02, facecolor="#afd1f5", edgecolor="none", linewidth=0))

    for idx, item in enumerate(rows):
        is_adjustment = item["label"] == "Adjustment"
        text_color = "#aaaaaa" if is_adjustment else "#111111"
        attr_ax.text(1.0, idx, _format_feature_label(item["label"]), ha="right", va="center", fontsize=11, color=text_color)
        value_ax.text(0.5, idx, _format_feature_value(item["value"]), ha="center", va="center", fontsize=11, color=text_color)
        symbol_ax.text(0.5, idx, "+" if is_adjustment else "x", ha="center", va="center", fontsize=12)
        factor_ax.text(0.86, idx, _format_factor_value(item["factor"]), ha="right", va="center", fontsize=11)
        partial_ax.text(0.88, idx, _format_factor_value(item["partial"]), ha="right", va="center", fontsize=11)
        if not is_adjustment:
            draw_label = names[item["feature_idx"]] if item["feature_idx"] < len(names) else item["label"]
            _draw_raw_value_meter(
                meter_ax,
                idx,
                data=data,
                instance_id=instance_id,
                instance=instance_array,
                feature_idx=item["feature_idx"],
                feature_label=draw_label,
            )

    score = sum(item["partial"] for item in rows)
    probability = float(surrogate.apply(instance_array)) if hasattr(surrogate, "apply") else 1.0 / (1.0 + np.exp(-score))
    pred_index = int(probability >= 0.5)
    pred_label = _class_label(class_labels, pred_index)

    pred_color = _class_text_color(pred_index)
    pred_ax.set_xlim(0, 1)
    pred_ax.text(0.18, 0.35, "AI Explainer", ha="center", va="center", fontweight="bold", fontsize=11)
    pred_ax.text(0.73, 0.35, "AI prediction", ha="center", va="center", fontweight="bold", fontsize=11)
    pred_ax.add_patch(plt.Rectangle((0.03, 0.60), 0.31, 0.55, fill=False, linewidth=1.3, color="#111111"))
    pred_ax.add_patch(plt.Rectangle((0.58, 0.60), 0.33, 0.55, fill=False, linewidth=1.3, color="#111111"))
    pred_ax.text(0.185, 0.80, _format_factor_value(score), ha="center", va="center", color=pred_color, fontsize=10)
    pred_ax.text(0.185, 0.97, f"({pred_label})", ha="center", va="center", color=pred_color, fontsize=9)
    pred_ax.text(0.745, 0.875, pred_label, ha="center", va="center", color=pred_color, fontsize=11)

    if title:
        fig.suptitle(title, y=1.02, fontweight="bold")
    fig.tight_layout()
    return fig, axes


def _tree_depth(nodes_by_id: Dict[int, Dict[str, Any]], node_id: int = 0) -> int:
    node = nodes_by_id[node_id]
    if node.get("is_leaf"):
        return 0
    return 1 + max(_tree_depth(nodes_by_id, int(node["left"])), _tree_depth(nodes_by_id, int(node["right"])))


def _assign_tree_positions(
    nodes_by_id: Dict[int, Dict[str, Any]],
    node_id: int,
    depth: int,
    leaf_cursor: List[int],
    positions: Dict[int, Tuple[float, float]],
) -> float:
    node = nodes_by_id[node_id]
    if node.get("is_leaf"):
        x_pos = float(leaf_cursor[0])
        leaf_cursor[0] += 1
    else:
        left_x = _assign_tree_positions(nodes_by_id, int(node["left"]), depth + 1, leaf_cursor, positions)
        right_x = _assign_tree_positions(nodes_by_id, int(node["right"]), depth + 1, leaf_cursor, positions)
        x_pos = (left_x + right_x) / 2.0
    positions[node_id] = (x_pos, -float(depth))
    return x_pos


def _draw_tree_node(
    ax: Any,
    *,
    node: Dict[str, Any],
    x_pos: float,
    y_pos: float,
    feature_names: List[str],
    class_labels: Optional[List[str]],
    is_path_node: bool,
    predicted_leaf_id: Optional[int],
) -> None:
    import matplotlib.pyplot as plt

    if node.get("is_leaf"):
        values = np.asarray(node.get("value", []), dtype=float)
        class_index = int(np.argmax(values)) if values.size else 0
        label = _class_label(class_labels, class_index)
        is_active = node.get("node") == predicted_leaf_id
        edge, face = _leaf_node_colors(class_index, active=is_active)
        ax.text(
            x_pos,
            y_pos,
            label,
            ha="center",
            va="center",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.28", facecolor=face, edgecolor=edge, linewidth=1.1),
            zorder=4,
        )
        return

    label = _format_feature_label(_feature_label_from_key(str(node.get("feature")), feature_names))
    edge = "#111111" if is_path_node else "#888888"
    ax.text(
        x_pos,
        y_pos,
        label,
        ha="center",
        va="center",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.34", facecolor="white", edgecolor=edge, linewidth=1.1),
        zorder=4,
    )


def plot_decision_tree_instance_view(
    surrogate: Any,
    instance: Any,
    data: Optional[Any] = None,
    *,
    instance_id: Optional[Any] = None,
    feature_names: Optional[List[str]] = None,
    class_labels: Optional[List[str]] = None,
    title: Optional[str] = None,
):
    """
    Plot the CoXAM-style attribute/value instance view beside a fitted decision tree.

    The tree layout is built from the loaded surrogate tree structure, so both the
    displayed depth and split labels adapt to the fitted decision-tree settings.
    """
    import matplotlib

    _patch_matplotlib_inline_rcparams(matplotlib)
    import matplotlib.pyplot as plt

    if not getattr(surrogate, "is_fitted", False):
        raise RuntimeError("Decision-tree surrogate must be fitted before plotting.")
    if not hasattr(surrogate, "get_tree_structure") or not hasattr(surrogate, "apply"):
        raise TypeError("surrogate must expose get_tree_structure() and apply().")

    instance_array = _instance_array(instance)
    names = _surrogate_feature_names(surrogate, feature_names)
    tree_structure = surrogate.get_tree_structure()
    nodes_by_id = {int(node["node"]): node for node in tree_structure}
    applied = surrogate.apply(instance_array)
    path = applied.get("path", [])
    path_node_ids = {int(step["node"]) for step in path}
    leaf_id = None
    if path:
        last_step = path[-1]
        last_node = nodes_by_id[int(last_step["node"])]
        leaf_id = int(last_node["left"] if last_step["direction"] == "left" else last_node["right"])

    used_indices = []
    for node in tree_structure:
        feature_key = node.get("feature")
        if feature_key:
            idx = _feature_index_from_key(str(feature_key))
            if idx not in used_indices:
                used_indices.append(idx)
    if not used_indices:
        used_indices = list(range(min(len(names), len(instance_array))))

    row_count = len(used_indices)
    max_depth = _tree_depth(nodes_by_id)
    fig_height = max(4.2, 0.66 * row_count + 1.4, 1.0 * (max_depth + 2))
    fig, axes = plt.subplots(
        1,
        4,
        figsize=(13.0, fig_height),
        gridspec_kw={"width_ratios": [1.7, 0.85, 1.2, 5.7]},
    )
    attr_ax, value_ax, meter_ax, tree_ax = axes

    for ax in (attr_ax, value_ax, meter_ax):
        ax.set_ylim(-0.8, row_count - 0.2)
        ax.invert_yaxis()
        ax.axis("off")

    attr_ax.set_title("Attribute", fontweight="bold", pad=10)
    value_ax.set_title("Value", fontweight="bold", pad=10)
    meter_ax.set_xlim(0, 1)

    for row_idx, feature_idx in enumerate(used_indices):
        label = names[feature_idx] if feature_idx < len(names) else f"a{feature_idx}"
        value = _raw_display_value_for_index(data, instance_id, instance_array, feature_idx, names)
        attr_ax.text(1.0, row_idx, _format_feature_label(label), ha="right", va="center", fontsize=11)
        value_ax.text(0.5, row_idx, _format_feature_value(value), ha="center", va="center", fontsize=11)
        _draw_raw_value_meter(
            meter_ax,
            row_idx,
            data=data,
            instance_id=instance_id,
            instance=instance_array,
            feature_idx=feature_idx,
            feature_label=label,
        )

    positions: Dict[int, Tuple[float, float]] = {}
    _assign_tree_positions(nodes_by_id, 0, 0, [0], positions)
    xs = [pos[0] for pos in positions.values()]
    ys = [pos[1] for pos in positions.values()]
    tree_ax.set_xlim(min(xs) - 0.9, max(xs) + 0.9)
    tree_ax.set_ylim(min(ys) - 0.65, 0.65)
    tree_ax.axis("off")

    for node_id, node in nodes_by_id.items():
        if node.get("is_leaf"):
            continue
        x0, y0 = positions[node_id]
        threshold = float(node.get("threshold", 0.0))
        for child_key, sign, ha in (("left", "<=", "right"), ("right", ">", "left")):
            child_id = int(node[child_key])
            x1, y1 = positions[child_id]
            edge_color = "#111111" if node_id in path_node_ids else "#8c8c8c"
            tree_ax.plot([x0, x1], [y0 - 0.08, y1 + 0.12], color=edge_color, linewidth=1.0, zorder=1)
            label_x = (x0 + x1) / 2.0
            label_y = (y0 + y1) / 2.0 + 0.04
            tree_ax.text(label_x, label_y, f"{sign} {_format_factor_value(threshold)}", fontsize=8, ha=ha, va="center")

    for node_id, node in nodes_by_id.items():
        x_pos, y_pos = positions[node_id]
        _draw_tree_node(
            tree_ax,
            node=node,
            x_pos=x_pos,
            y_pos=y_pos,
            feature_names=names,
            class_labels=class_labels or getattr(surrogate, "class_labels", None),
            is_path_node=node_id in path_node_ids,
            predicted_leaf_id=leaf_id,
        )

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
    import matplotlib

    _patch_matplotlib_inline_rcparams(matplotlib)
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

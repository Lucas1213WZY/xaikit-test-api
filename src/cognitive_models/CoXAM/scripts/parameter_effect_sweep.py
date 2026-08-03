"""
Run parameter sweeps for the simplified reasoning strategies.

The script writes:
  - forward_results.csv
  - counterfactual_results.csv
  - one PNG per swept parameter with accuracy/success and time curves

Default output directory: outputs/parameter_effect_sweeps
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dt_memory import add_dt_to_memory, cf_change_path_dt, dt_traverse, recall_change_dt
from src.heuristic_lr_model import add_lr_heuristic_to_memory, cf_lr_heuristic, lr_heuristic
from src.lr_memory import add_lr_calculation_to_memory, cf_lr_calculation, lr_calculation, recall_change_lr
from src.memory import CombinedMemory, DeclarativeMemory
from src.utils import AIDatasetLoader, DecisionTreeInterpreter, LogisticRegressionInterpreter, filter_by_app_and_model


FIXED_DISPLAYED_SIGNIFICANT_FIGURES = 2
FIXED_READ_SECONDS_PER_ITEM = 1.0
FIXED_MENTAL_CALCULATION_SECONDS = 0.0
FIXED_MEMORY_RECALL_NOISE = 0.5
FIXED_RETRIEVAL_CANDIDATE_COUNT = 3
FIXED_SIMULATION_SAMPLE_COUNT = 16
FIXED_DEPTH_CHOICE_TEMPERATURE = 1.0


DEFAULT_FORWARD_SWEEPS = {
    "memory_recall_threshold": [-5.0, -3.0, -1.0, 0.0, 1.0, 2.0],
    "decision_boundary": [0.6, 1.0, 1.5, 1.7],
    "decision_noise": [0.3, 0.5, 0.7],
    "selected_feature_count": [2, 4, None],
}

DEFAULT_COUNTERFACTUAL_SWEEPS = {
    "memory_recall_threshold": [-5.0, -3.0, -1.0, 0.0, 1.0, 2.0],
    "max_memory_refresh_probability": [0.25, 0.75, 1.0],
    "counterfactual_tree_depth": [0, 1, 2],
    "counterfactual_overshoot_fraction": [0.0, 0.1, 0.3, 0.5],
    "selected_feature_count": [2, 4, None],
}

DEFAULT_APPS = ["mushrooms", "wine_quality"]
GENERATED_OUTPUT_PATTERNS = [
    "forward_results.csv",
    "counterfactual_results.csv",
    "forward_*.png",
    "counterfactual_*.png",
]


@dataclass(frozen=True)
class EvaluationBundle:
    app_id: str
    model_name: str
    loader: AIDatasetLoader
    lr_exp: LogisticRegressionInterpreter
    dt_exp: DecisionTreeInterpreter
    bounds: dict[str, tuple[float, float]]
    instance_ids: list[int]


def load_bundles(data_dir: Path, apps: list[str] | None) -> list[EvaluationBundle]:
    values_df = pd.read_csv(data_dir / "values.csv")
    metadata_df = pd.read_csv(data_dir / "metadata.csv")
    prediction_df = pd.read_csv(data_dir / "none.csv")
    lr_df = pd.read_csv(data_dir / "logistic_regression.csv")
    dt_df = pd.read_csv(data_dir / "decision_tree.csv")

    base_loader = AIDatasetLoader(values_df, metadata_df, prediction_df)
    app_models = prediction_df[["appId", "modelName"]].drop_duplicates()
    if apps:
        app_models = app_models[app_models["appId"].isin(apps)]

    bundles: list[EvaluationBundle] = []
    for row in app_models.itertuples(index=False):
        app_id = str(row.appId)
        model_name = str(row.modelName)
        try:
            loader = filter_by_app_and_model(base_loader, app_id, model_name)
            lr_exp = LogisticRegressionInterpreter(lr_df, metadata_df, app_id, model_name)
            dt_exp = DecisionTreeInterpreter(dt_df, metadata_df, app_id, model_name, depth=2)
            bounds = loader.get_bounds_for_app(app_id, normalized=False)
        except Exception:
            continue

        ids = loader.feature_values_df["instanceId"].dropna().astype(int).tolist()
        if ids:
            bundles.append(EvaluationBundle(app_id, model_name, loader, lr_exp, dt_exp, bounds, ids))
    return bundles


def make_memory(
    memory_recall_threshold: float,
    memory_recall_noise: float,
    *,
    working_memory_capacity: int = 7,
) -> CombinedMemory:
    dm = DeclarativeMemory(
        memory_recall_threshold=memory_recall_threshold,
        cue_association_strength=2.0,
        memory_mismatch_penalty=-2.0,
        memory_recall_noise=memory_recall_noise,
    )
    return CombinedMemory(dm, working_memory_capacity=working_memory_capacity)


def prime_lr_calculation_memory(bundle: EvaluationBundle, params: dict[str, Any]) -> CombinedMemory:
    memory = make_memory(params["memory_recall_threshold"], params["memory_recall_noise"])
    add_lr_calculation_to_memory(
        bundle.lr_exp,
        memory,
        intercept_significant_figures=FIXED_DISPLAYED_SIGNIFICANT_FIGURES,
        coefficient_significant_figures=FIXED_DISPLAYED_SIGNIFICANT_FIGURES,
    )
    memory.tick(1.0)
    return memory


def prime_lr_heuristic_memory(bundle: EvaluationBundle, params: dict[str, Any]) -> CombinedMemory:
    memory = make_memory(params["memory_recall_threshold"], params["memory_recall_noise"])
    add_lr_heuristic_to_memory(bundle.lr_exp, memory)
    memory.tick(1.0)
    return memory


def prime_dt_memory(bundle: EvaluationBundle, params: dict[str, Any]) -> CombinedMemory:
    memory = make_memory(params["memory_recall_threshold"], params["memory_recall_noise"])
    add_dt_to_memory(memory, bundle.dt_exp, threshold_significant_figures=FIXED_DISPLAYED_SIGNIFICANT_FIGURES)
    memory.tick(1.0)
    return memory


def prime_lr_counterfactual_recall_memory(
    bundle: EvaluationBundle,
    params: dict[str, Any],
    raw_x: np.ndarray,
    selected_feature_indices: list[int] | None,
) -> CombinedMemory:
    memory = prime_lr_calculation_memory(bundle, params)
    cf_lr_calculation(
        raw_x,
        bundle.lr_exp,
        bundle.bounds,
        displayed_significant_figures=FIXED_DISPLAYED_SIGNIFICANT_FIGURES,
        read_seconds_per_item=FIXED_READ_SECONDS_PER_ITEM,
        mental_calculation_seconds=FIXED_MENTAL_CALCULATION_SECONDS,
        memory=memory,
        selected_feature_indices=selected_feature_indices,
    )
    return memory


def prime_dt_counterfactual_recall_memory(
    bundle: EvaluationBundle,
    params: dict[str, Any],
    raw_x: np.ndarray,
) -> CombinedMemory:
    memory = prime_dt_memory(bundle, params)
    cf_change_path_dt(
        raw_x,
        bundle.dt_exp,
        bundle.bounds,
        explanation_access_mode="retrieve",
        memory=memory,
        counterfactual_tree_depth=params["counterfactual_tree_depth"],
        displayed_significant_figures=FIXED_DISPLAYED_SIGNIFICANT_FIGURES,
        read_seconds_per_item=FIXED_READ_SECONDS_PER_ITEM,
        mental_calculation_seconds=FIXED_MENTAL_CALCULATION_SECONDS,
        simulation_sample_count=params["simulation_sample_count"],
        retrieval_candidate_count=params["retrieval_candidate_count"],
        max_memory_refresh_probability=params["max_memory_refresh_probability"],
        depth_choice_temperature=params["depth_choice_temperature"],
    )
    return memory


def selected_indices(feature_vector: np.ndarray, selected_feature_count: int | None) -> list[int] | None:
    if selected_feature_count is None:
        return None
    return list(range(min(int(selected_feature_count), len(feature_vector))))


def sample_instance_ids(bundle: EvaluationBundle, sample_size: int, rng: np.random.Generator) -> list[int]:
    ids = np.asarray(bundle.instance_ids, dtype=int)
    if len(ids) <= sample_size:
        return ids.tolist()
    return rng.choice(ids, size=sample_size, replace=False).astype(int).tolist()


def normalize_probabilities(probs: np.ndarray) -> np.ndarray:
    normalized = np.asarray(probs, dtype=float).copy()
    normalized[~np.isfinite(normalized)] = 0.0
    normalized[normalized < 0.0] = 0.0
    total = float(normalized.sum())
    if total <= 0.0:
        return np.full(len(normalized), 1.0 / max(1, len(normalized)), dtype=float)
    return normalized / total


def sample_prediction(probs: np.ndarray, rng: np.random.Generator) -> int:
    normalized = normalize_probabilities(probs)
    if len(normalized) == 0:
        return 0
    return int(rng.choice(np.arange(len(normalized)), p=normalized))


def forward_params_with(base: dict[str, Any], parameter_name: str, parameter_value: Any) -> dict[str, Any]:
    params = dict(base)
    params[parameter_name] = parameter_value
    return params


def run_forward_sweeps(
    bundles: list[EvaluationBundle],
    *,
    sample_size: int,
    repeats: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = {
        "memory_recall_threshold": -1.0,
        "memory_recall_noise": FIXED_MEMORY_RECALL_NOISE,
        "decision_boundary": 1.0,
        "decision_noise": 0.4,
        "retrieval_candidate_count": FIXED_RETRIEVAL_CANDIDATE_COUNT,
        "simulation_sample_count": FIXED_SIMULATION_SAMPLE_COUNT,
        "selected_feature_count": None,
    }
    rows: list[dict[str, Any]] = []
    for parameter_name, values in DEFAULT_FORWARD_SWEEPS.items():
        for parameter_value in values:
            params = forward_params_with(base, parameter_name, parameter_value)
            print(f"[forward] parameter={parameter_name} value={parameter_value}")
            for bundle in bundles:
                print(
                    f"  dataset={bundle.app_id} model={bundle.model_name} "
                    f"sample_size={sample_size} repeats={repeats}"
                )
                print(
                    "    strategies=lr_calculation_read, lr_calculation_retrieve, "
                    "lr_heuristic, dt_read, dt_retrieve"
                )
                for repeat in range(repeats):
                    sampled_ids = sample_instance_ids(bundle, sample_size, rng)
                    print(f"    repeat={repeat + 1}/{repeats} sampled_instances={len(sampled_ids)}")
                    raw_instances, labels = bundle.loader.load_instances(sampled_ids, normalize=False)
                    norm_instances, _ = bundle.loader.load_instances(sampled_ids, normalize=True)
                    for instance_id, raw_x, norm_x, label in zip(sampled_ids, raw_instances, norm_instances, labels):
                        y = int(label)
                        raw_x_arr = np.asarray(raw_x, dtype=float)
                        norm_x_arr = np.asarray(norm_x, dtype=float)
                        active_raw = selected_indices(raw_x_arr, params["selected_feature_count"])
                        active_norm = selected_indices(norm_x_arr, params["selected_feature_count"])

                        evaluations = [
                            (
                                "lr_calculation_read",
                                lambda: lr_calculation(
                                    raw_x_arr,
                                    prime_lr_calculation_memory(bundle, params),
                                    bundle.lr_exp,
                                    explanation_access_mode="read",
                                    displayed_significant_figures=FIXED_DISPLAYED_SIGNIFICANT_FIGURES,
                                    read_seconds_per_item=FIXED_READ_SECONDS_PER_ITEM,
                                    mental_calculation_seconds=FIXED_MENTAL_CALCULATION_SECONDS,
                                    decision_boundary=params["decision_boundary"],
                                    decision_noise=params["decision_noise"],
                                    selected_feature_indices=active_raw,
                                    simulation_sample_count=params["simulation_sample_count"],
                                    retrieval_candidate_count=params["retrieval_candidate_count"],
                                ),
                            ),
                            (
                                "lr_calculation_retrieve",
                                lambda: lr_calculation(
                                    raw_x_arr,
                                    prime_lr_calculation_memory(bundle, params),
                                    bundle.lr_exp,
                                    explanation_access_mode="retrieve",
                                    displayed_significant_figures=FIXED_DISPLAYED_SIGNIFICANT_FIGURES,
                                    read_seconds_per_item=FIXED_READ_SECONDS_PER_ITEM,
                                    mental_calculation_seconds=FIXED_MENTAL_CALCULATION_SECONDS,
                                    decision_boundary=params["decision_boundary"],
                                    decision_noise=params["decision_noise"],
                                    selected_feature_indices=active_raw,
                                    simulation_sample_count=params["simulation_sample_count"],
                                    retrieval_candidate_count=params["retrieval_candidate_count"],
                                ),
                            ),
                            (
                                "lr_heuristic",
                                lambda: lr_heuristic(
                                    norm_x_arr,
                                    prime_lr_heuristic_memory(bundle, params),
                                    bundle.lr_exp,
                                    simulation_sample_count=params["simulation_sample_count"],
                                    retrieval_candidate_count=params["retrieval_candidate_count"],
                                    read_seconds_per_item=FIXED_READ_SECONDS_PER_ITEM,
                                    mental_calculation_seconds=FIXED_MENTAL_CALCULATION_SECONDS,
                                    decision_boundary=params["decision_boundary"],
                                    decision_noise=params["decision_noise"],
                                    selected_feature_indices=active_norm,
                                ),
                            ),
                            (
                                "dt_read",
                                lambda: dt_traverse(
                                    raw_x_arr,
                                    prime_dt_memory(bundle, params),
                                    bundle.dt_exp,
                                    explanation_access_mode="read",
                                    displayed_significant_figures=FIXED_DISPLAYED_SIGNIFICANT_FIGURES,
                                    read_seconds_per_item=FIXED_READ_SECONDS_PER_ITEM,
                                    decision_boundary=params["decision_boundary"],
                                    decision_noise=params["decision_noise"],
                                    simulation_sample_count=params["simulation_sample_count"],
                                    retrieval_candidate_count=params["retrieval_candidate_count"],
                                ),
                            ),
                            (
                                "dt_retrieve",
                                lambda: dt_traverse(
                                    raw_x_arr,
                                    prime_dt_memory(bundle, params),
                                    bundle.dt_exp,
                                    explanation_access_mode="retrieve",
                                    displayed_significant_figures=FIXED_DISPLAYED_SIGNIFICANT_FIGURES,
                                    read_seconds_per_item=FIXED_READ_SECONDS_PER_ITEM,
                                    decision_boundary=params["decision_boundary"],
                                    decision_noise=params["decision_noise"],
                                    simulation_sample_count=params["simulation_sample_count"],
                                    retrieval_candidate_count=params["retrieval_candidate_count"],
                                ),
                            ),
                        ]

                        for strategy, evaluate in evaluations:
                            probs, prediction_time, _info = evaluate()
                            normalized_probs = normalize_probabilities(probs)
                            predicted = sample_prediction(normalized_probs, rng)
                            rows.append(
                                {
                                    "task": "forward",
                                    "app_id": bundle.app_id,
                                    "model_name": bundle.model_name,
                                    "repeat": repeat,
                                    "instance_id": instance_id,
                                    "strategy": strategy,
                                    "parameter": parameter_name,
                                    "parameter_value": "all" if parameter_value is None else parameter_value,
                                    "accuracy": float(predicted == y),
                                    "probability_assigned_to_target": float(normalized_probs[y]),
                                    "time_seconds": float(prediction_time),
                                }
                            )
    return pd.DataFrame(rows)


def apply_counterfactual_change(
    feature_vector: np.ndarray,
    feature_key: str,
    bounds: dict[str, tuple[float, float]],
    delta: float,
    overshoot_fraction: float,
) -> np.ndarray:
    x_new = np.asarray(feature_vector, dtype=float).copy()
    base_key = feature_key.split("=")[0]
    idx = int(base_key[1:])
    if base_key in bounds:
        lo, hi = bounds[base_key]
        overshoot = (hi - lo) * overshoot_fraction
        overshoot = -overshoot if delta < 0 else overshoot
        x_new[idx] = min(max(x_new[idx] + delta + overshoot, lo), hi)
    else:
        x_new[idx] = x_new[idx] + delta
    return x_new


def lr_label(lr_exp: LogisticRegressionInterpreter, feature_vector: np.ndarray) -> int:
    return int(lr_exp.apply_to_instance(feature_vector) > 0)


def dt_label(dt_exp: DecisionTreeInterpreter, feature_vector: np.ndarray) -> int:
    return int(dt_exp.apply_to_instance(feature_vector)["class_index"])


def expected_counterfactual_success(
    out: dict[str, Any],
    raw_x: np.ndarray,
    bounds: dict[str, tuple[float, float]],
    baseline_label: int,
    label_fn,
    overshoot_fraction: float,
) -> tuple[float, float]:
    success = 0.0
    expected_time = 0.0
    for feature_key, vals in out.items():
        if feature_key == "expected_time":
            continue
        p = float(vals.get("p_selected", 0.0))
        changed = apply_counterfactual_change(raw_x, feature_key, bounds, vals["mean_delta"], overshoot_fraction)
        success += p * float(label_fn(changed) != baseline_label)
        expected_time += p * float(vals["mean_time"])
    return success, expected_time


def run_counterfactual_sweeps(
    bundles: list[EvaluationBundle],
    *,
    sample_size: int,
    repeats: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = {
        "memory_recall_threshold": -1.0,
        "memory_recall_noise": FIXED_MEMORY_RECALL_NOISE,
        "retrieval_candidate_count": FIXED_RETRIEVAL_CANDIDATE_COUNT,
        "simulation_sample_count": FIXED_SIMULATION_SAMPLE_COUNT,
        "max_memory_refresh_probability": 1.0,
        "counterfactual_tree_depth": 1,
        "depth_choice_temperature": FIXED_DEPTH_CHOICE_TEMPERATURE,
        "counterfactual_overshoot_fraction": 0.1,
        "selected_feature_count": None,
    }
    rows: list[dict[str, Any]] = []
    for parameter_name, values in DEFAULT_COUNTERFACTUAL_SWEEPS.items():
        for parameter_value in values:
            params = forward_params_with(base, parameter_name, parameter_value)
            print(f"[counterfactual] parameter={parameter_name} value={parameter_value}")
            for bundle in bundles:
                print(
                    f"  dataset={bundle.app_id} model={bundle.model_name} "
                    f"sample_size={sample_size} repeats={repeats}"
                )
                print(
                    "    strategies=cf_lr_calculation, cf_lr_heuristic, cf_dt_read, "
                    "cf_dt_retrieve, recall_change_dt, recall_change_lr"
                )
                for repeat in range(repeats):
                    sampled_ids = sample_instance_ids(bundle, sample_size, rng)
                    print(f"    repeat={repeat + 1}/{repeats} sampled_instances={len(sampled_ids)}")
                    raw_instances, labels = bundle.loader.load_instances(sampled_ids, normalize=False)
                    norm_instances, _ = bundle.loader.load_instances(sampled_ids, normalize=True)
                    for instance_id, raw_x, norm_x, label in zip(sampled_ids, raw_instances, norm_instances, labels):
                        raw_x_arr = np.asarray(raw_x, dtype=float)
                        norm_x_arr = np.asarray(norm_x, dtype=float)
                        active_raw = selected_indices(raw_x_arr, params["selected_feature_count"])
                        active_norm = selected_indices(norm_x_arr, params["selected_feature_count"])
                        ai_label = int(label)
                        lr_current = lr_label(bundle.lr_exp, raw_x_arr)
                        dt_current = dt_label(bundle.dt_exp, raw_x_arr)
                        lr_direction = "increase" if ai_label == 0 else "decrease"

                        evaluations = [
                            (
                                "cf_lr_calculation",
                                lambda: (
                                    cf_lr_calculation(
                                        raw_x_arr,
                                        bundle.lr_exp,
                                        bundle.bounds,
                                        displayed_significant_figures=FIXED_DISPLAYED_SIGNIFICANT_FIGURES,
                                        read_seconds_per_item=FIXED_READ_SECONDS_PER_ITEM,
                                        mental_calculation_seconds=FIXED_MENTAL_CALCULATION_SECONDS,
                                        memory=prime_lr_calculation_memory(bundle, params),
                                        selected_feature_indices=active_raw,
                                    ),
                                    lr_current,
                                    lambda changed: lr_label(bundle.lr_exp, changed),
                                ),
                            ),
                            (
                                "cf_lr_heuristic",
                                lambda: (
                                    cf_lr_heuristic(
                                        norm_x_arr,
                                        prime_lr_heuristic_memory(bundle, params),
                                        bundle.lr_exp,
                                        bundle.bounds,
                                        retrieval_candidate_count=params["retrieval_candidate_count"],
                                        read_seconds_per_item=FIXED_READ_SECONDS_PER_ITEM,
                                        mental_calculation_seconds=FIXED_MENTAL_CALCULATION_SECONDS,
                                        selected_feature_indices=active_norm,
                                        actual_label=ai_label,
                                    ),
                                    lr_current,
                                    lambda changed: lr_label(bundle.lr_exp, changed),
                                ),
                            ),
                            (
                                "cf_dt_read",
                                lambda: (
                                    cf_change_path_dt(
                                        raw_x_arr,
                                        bundle.dt_exp,
                                        bundle.bounds,
                                        explanation_access_mode="read",
                                        counterfactual_tree_depth=params["counterfactual_tree_depth"],
                                        displayed_significant_figures=FIXED_DISPLAYED_SIGNIFICANT_FIGURES,
                                        read_seconds_per_item=FIXED_READ_SECONDS_PER_ITEM,
                                        mental_calculation_seconds=FIXED_MENTAL_CALCULATION_SECONDS,
                                        depth_choice_temperature=params["depth_choice_temperature"],
                                    ),
                                    dt_current,
                                    lambda changed: dt_label(bundle.dt_exp, changed),
                                ),
                            ),
                            (
                                "cf_dt_retrieve",
                                lambda: (
                                    cf_change_path_dt(
                                        raw_x_arr,
                                        bundle.dt_exp,
                                        bundle.bounds,
                                        explanation_access_mode="retrieve",
                                        memory=prime_dt_memory(bundle, params),
                                        counterfactual_tree_depth=params["counterfactual_tree_depth"],
                                        displayed_significant_figures=FIXED_DISPLAYED_SIGNIFICANT_FIGURES,
                                        read_seconds_per_item=FIXED_READ_SECONDS_PER_ITEM,
                                        mental_calculation_seconds=FIXED_MENTAL_CALCULATION_SECONDS,
                                        simulation_sample_count=params["simulation_sample_count"],
                                        retrieval_candidate_count=params["retrieval_candidate_count"],
                                        max_memory_refresh_probability=params["max_memory_refresh_probability"],
                                        depth_choice_temperature=params["depth_choice_temperature"],
                                    ),
                                    dt_current,
                                    lambda changed: dt_label(bundle.dt_exp, changed),
                                ),
                            ),
                            (
                                "recall_change_dt",
                                lambda: (
                                    recall_change_dt(
                                        raw_x_arr,
                                        prime_dt_counterfactual_recall_memory(bundle, params, raw_x_arr),
                                        bundle.bounds,
                                        displayed_significant_figures=FIXED_DISPLAYED_SIGNIFICANT_FIGURES,
                                        retrieved_combo_count=params["retrieval_candidate_count"],
                                    ),
                                    dt_current,
                                    lambda changed: dt_label(bundle.dt_exp, changed),
                                ),
                            ),
                            (
                                "recall_change_lr",
                                lambda: (
                                    recall_change_lr(
                                        prime_lr_counterfactual_recall_memory(bundle, params, raw_x_arr, active_raw),
                                        retrieved_combo_count=params["retrieval_candidate_count"],
                                        preferred_change_direction=lr_direction,
                                    ),
                                    lr_current,
                                    lambda changed: lr_label(bundle.lr_exp, changed),
                                ),
                            ),
                        ]

                        for strategy, evaluate in evaluations:
                            out, current_label, label_fn = evaluate()
                            success, cf_time = expected_counterfactual_success(
                                out,
                                raw_x_arr,
                                bundle.bounds,
                                current_label,
                                label_fn,
                                params["counterfactual_overshoot_fraction"],
                            )
                            rows.append(
                                {
                                    "task": "counterfactual",
                                    "app_id": bundle.app_id,
                                    "model_name": bundle.model_name,
                                    "repeat": repeat,
                                    "instance_id": instance_id,
                                    "strategy": strategy,
                                    "parameter": parameter_name,
                                    "parameter_value": "all" if parameter_value is None else parameter_value,
                                    "success": float(success),
                                    "time_seconds": float(cf_time),
                                }
                            )
    return pd.DataFrame(rows)


def plot_metric_grid(df: pd.DataFrame, task: str, metric: str, out_dir: Path) -> None:
    if df.empty:
        return
    summary = (
        df.groupby(["parameter", "parameter_value", "app_id", "model_name", "strategy"], as_index=False)[metric]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary["ci95"] = 1.96 * summary["std"].fillna(0.0) / np.sqrt(summary["count"].clip(lower=1))

    for parameter_name, param_df in summary.groupby("parameter", sort=False):
        dataset_keys = (
            param_df[["app_id", "model_name"]]
            .drop_duplicates()
            .sort_values(["app_id", "model_name"])
            .itertuples(index=False, name=None)
        )
        dataset_keys = list(dataset_keys)
        fig, axes = plt.subplots(
            1,
            len(dataset_keys),
            figsize=(8 * len(dataset_keys), 5),
            sharey=True,
            squeeze=False,
        )

        value_order = (
            df[df["parameter"] == parameter_name]["parameter_value"]
            .drop_duplicates()
            .tolist()
        )
        x_labels = [str(value) for value in value_order]
        x_positions = np.arange(len(value_order))

        for ax, (app_id, model_name) in zip(axes[0], dataset_keys):
            dataset_df = param_df[(param_df["app_id"] == app_id) & (param_df["model_name"] == model_name)]
            for strategy, strategy_df in dataset_df.groupby("strategy", sort=False):
                strategy_df = strategy_df.set_index("parameter_value").reindex(value_order).reset_index()
                ax.errorbar(
                    x_positions,
                    strategy_df["mean"].to_numpy(dtype=float),
                    yerr=strategy_df["ci95"].to_numpy(dtype=float),
                    marker="o",
                    capsize=4,
                    linewidth=1.5,
                    label=strategy,
                )
            ax.set_title(f"{app_id} / {model_name}")
            ax.set_xlabel(parameter_name)
            ax.set_xticks(x_positions)
            ax.set_xticklabels(x_labels, rotation=25, ha="right")
            ax.grid(True, alpha=0.3)
        axes[0][0].set_ylabel(f"{metric} mean with 95% CI")
        fig.suptitle(f"{task}: {metric} vs {parameter_name}", y=1.02)
        axes[0][-1].legend(fontsize=8, loc="best")
        fig.tight_layout()
        plot_path = out_dir / f"{task}_{metric}_{parameter_name}.png"
        fig.savefig(plot_path, dpi=160)
        print(f"Saved {plot_path}")
        plt.close(fig)


def prepare_output_dir(out_dir: Path, *, overwrite: bool) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if not overwrite:
        return
    removed = 0
    for pattern in GENERATED_OUTPUT_PATTERNS:
        for path in out_dir.glob(pattern):
            if path.is_file():
                path.unlink()
                removed += 1
    if removed:
        print(f"Removed {removed} previous generated output file(s) from {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "datasets")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "parameter_effect_sweeps")
    parser.add_argument("--max-instances", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--apps",
        nargs="*",
        default=DEFAULT_APPS,
        help="Datasets to sweep. Defaults to mushrooms and wine_quality.",
    )
    parser.add_argument("--task", choices=["forward", "counterfactual", "both"], default="both")
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Keep existing generated CSV/PNG files in the output directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepare_output_dir(args.output_dir, overwrite=not args.no_overwrite)
    print(f"Selected datasets: {', '.join(args.apps)}")
    bundles = load_bundles(args.data_dir, apps=args.apps)
    print(f"Loaded {len(bundles)} dataset/model bundle(s).")
    if not bundles:
        raise RuntimeError("No matching dataset/model bundles were found.")
    if args.task in ("forward", "both"):
        forward_df = run_forward_sweeps(
            bundles,
            sample_size=args.max_instances,
            repeats=args.repeats,
            seed=args.seed,
        )
        forward_csv = args.output_dir / "forward_results.csv"
        forward_df.to_csv(forward_csv, index=False)
        print(f"Saved {forward_csv}")
        plot_metric_grid(forward_df, "forward", "accuracy", args.output_dir)
        plot_metric_grid(forward_df, "forward", "time_seconds", args.output_dir)
    if args.task in ("counterfactual", "both"):
        counterfactual_df = run_counterfactual_sweeps(
            bundles,
            sample_size=args.max_instances,
            repeats=args.repeats,
            seed=args.seed + 1,
        )
        counterfactual_csv = args.output_dir / "counterfactual_results.csv"
        counterfactual_df.to_csv(counterfactual_csv, index=False)
        print(f"Saved {counterfactual_csv}")
        plot_metric_grid(counterfactual_df, "counterfactual", "success", args.output_dir)
        plot_metric_grid(counterfactual_df, "counterfactual", "time_seconds", args.output_dir)


if __name__ == "__main__":
    main()

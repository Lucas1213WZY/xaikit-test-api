import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .cognitive import (
    MENTAL_CALCULATION_TIME,
    READ_TIME,
    drift_diffusion_decision,
    lr_evidence,
    slider_step,
    snap_to_step,
)


def add_lr_heuristic_to_memory(lr_exp, memory, initial_belief_variance: float = 1.0):
    memory.add_chunk(
        "LR_intercept_prob",
        {"type": "intercept_prob", "mu": 0.0, "var": float(initial_belief_variance)},
    )

    for feature_key, coefficient in lr_exp.coefficients.items():
        coefficient_mean = np.sign(coefficient) if coefficient != 0 else 0.0
        memory.add_chunk(
            f"LR_coef_prob_{feature_key.replace('=', '_')}",
            {
                "type": "coef_prob",
                "feature_key": feature_key,
                "feature_name": lr_exp._format_feature(feature_key),
                "mu": float(coefficient_mean),
                "var": float(initial_belief_variance),
            },
        )


def _base_feature_index(feature_key: str) -> int:
    return int(feature_key.split("=")[0][1:])


def _feature_value_from_key(feature_key: str, feature_vector: np.ndarray) -> Tuple[float, bool]:
    if "=" in feature_key:
        base_key, category = feature_key.split("=")
        column = int(base_key[1:])
        return float(int(feature_vector[column]) == int(category)), False
    column = int(feature_key[1:])
    return float(feature_vector[column]), True


def _sample_retrieved_chunk(retrieval_result: Dict[str, Any], rng: np.random.Generator):
    top_chunks = retrieval_result.get("top_k", [])
    choices = [chunk for chunk, _probability in top_chunks] + [None]
    probabilities = [float(probability) for _chunk, probability in top_chunks]
    probabilities.append(float(retrieval_result.get("p_none", 0.0)))

    probabilities = np.asarray(probabilities, dtype=float)
    probabilities[~np.isfinite(probabilities)] = 0.0
    probabilities[probabilities < 0.0] = 0.0

    total = probabilities.sum()
    if total > 0.0:
        probabilities /= total
    else:
        probabilities = np.array([1.0] + [0.0] * (len(choices) - 1), dtype=float)

    return choices[rng.choice(len(choices), p=probabilities)]


def _chunk_mean_variance(chunk, *, missing_variance: float) -> tuple[float, float]:
    if chunk is None:
        return 0.0, missing_variance
    variance = chunk.slots.get("var", chunk.slots.get("sigma", 0.0) ** 2)
    return float(chunk.slots.get("mu", 0.0)), float(variance)


def _selected_lr_coefficients(lr_exp, selected_feature_indices: Optional[List[int]]):
    selected = set(selected_feature_indices) if selected_feature_indices is not None else None
    for feature_key in lr_exp.coefficients.keys():
        if selected is None or _base_feature_index(feature_key) in selected:
            yield feature_key


def _retrieve_lr_beliefs(memory, lr_exp, feature_vector, retrieval_candidate_count, selected_feature_indices):
    intercept_retrieval = memory.topk_retrievals_with_prob_refresh(
        {"type": "intercept_prob"},
        retrieval_candidate_count=retrieval_candidate_count,
        memory_refresh_probability=1.0,
        add_refresh=True,
    )
    retrieval_time = float(intercept_retrieval.get("expected_rt", 0.0))

    feature_beliefs = []
    for feature_key in _selected_lr_coefficients(lr_exp, selected_feature_indices):
        coefficient_retrieval = memory.topk_retrievals_with_prob_refresh(
            {"type": "coef_prob", "feature_key": feature_key},
            retrieval_candidate_count=retrieval_candidate_count,
            memory_refresh_probability=1.0,
            add_refresh=True,
        )
        retrieval_time += float(coefficient_retrieval.get("expected_rt", 0.0))
        value, is_numeric = _feature_value_from_key(feature_key, feature_vector)
        feature_beliefs.append((feature_key, value, is_numeric, coefficient_retrieval))

    return intercept_retrieval, feature_beliefs, retrieval_time


def lr_heuristic(
    feature_vector,
    memory,
    lr_exp,
    *,
    simulation_sample_count: int = 40,
    retrieval_candidate_count: int = 3,
    read_seconds_per_item: float = READ_TIME,
    mental_calculation_seconds: float = MENTAL_CALCULATION_TIME,
    decision_boundary: float = 1.5,
    decision_noise: float = 1.0,
    evidence_scaling: str = "l2",
    selected_feature_indices: Optional[List[int]] = None,
    rng: Optional[np.random.Generator] = None,
):
    rng = rng or np.random.default_rng()
    x = np.asarray(feature_vector, dtype=float)
    intercept_retrieval, feature_beliefs, retrieval_time = _retrieve_lr_beliefs(
        memory, lr_exp, x, retrieval_candidate_count, selected_feature_indices
    )

    p1_samples = []
    decision_times = []
    drift_samples = []
    for _ in range(simulation_sample_count):
        intercept_chunk = _sample_retrieved_chunk(intercept_retrieval, rng)
        intercept_mean, intercept_variance = _chunk_mean_variance(intercept_chunk, missing_variance=0.01)
        terms = [rng.normal(intercept_mean, math.sqrt(max(intercept_variance, 1e-12)))]

        for _feature_key, value, _is_numeric, coefficient_retrieval in feature_beliefs:
            coefficient_chunk = _sample_retrieved_chunk(coefficient_retrieval, rng)
            coefficient_mean, coefficient_variance = _chunk_mean_variance(
                coefficient_chunk, missing_variance=0.0
            )
            coefficient = (
                rng.normal(coefficient_mean, math.sqrt(max(coefficient_variance, 0.0)))
                if coefficient_variance > 0.0
                else coefficient_mean
            )
            terms.append(coefficient * value)

        evidence = lr_evidence(terms, evidence_scaling=evidence_scaling)
        p_upper, decision_time, drift = drift_diffusion_decision(
            evidence,
            decision_boundary=decision_boundary,
            decision_noise=decision_noise,
            gain=1.0,
        )
        p1_samples.append(p_upper)
        decision_times.append(decision_time)
        drift_samples.append(drift)

    p1 = float(np.mean(p1_samples)) if p1_samples else 0.5
    read_time = read_seconds_per_item * len(feature_beliefs)
    calculation_time = mental_calculation_seconds * len(feature_beliefs)
    mean_decision_time = float(np.mean(decision_times)) if decision_times else 0.0
    total_time = retrieval_time + read_time + calculation_time + mean_decision_time
    memory.tick(total_time)

    info = {
        "decision": {
            "p1": p1,
            "v_ratio_mean": float(np.mean(drift_samples)) if drift_samples else 0.0,
        },
        "timing": {
            "retrieval_rt_sum": retrieval_time,
            "read_time_sum": read_time,
            "intuitive_ops": len(feature_beliefs),
            "intuitive_op_cost": mental_calculation_seconds,
            "ddm_rt_mean": mean_decision_time,
            "total_time": total_time,
        },
        "chunks": {
            "intercept": {
                "chosen_name": (
                    intercept_retrieval.get("top_k", [])[0][0].name
                    if intercept_retrieval.get("top_k")
                    else None
                )
            },
            "features": [
                {
                    "key": feature_key,
                    "value": float(value),
                    "chosen_name": (
                        coefficient_retrieval.get("top_k", [])[0][0].name
                        if coefficient_retrieval.get("top_k")
                        else None
                    ),
                    "is_numeric": bool(is_numeric),
                }
                for feature_key, value, is_numeric, coefficient_retrieval in feature_beliefs
            ],
        },
    }

    return np.array([1.0 - p1, p1], dtype=float), total_time, info


def refresh_lr_heuristic_in_memory(
    memory,
    lr_exp,
    info,
    actual: int,
    *,
    selected_feature_indices: list[int] = None,
    min_learning_curvature: float = 1e-4,
):
    actual = int(actual)
    predicted_probability = float(info["decision"]["p1"])
    selected = set(selected_feature_indices) if selected_feature_indices is not None else None

    def update_belief(mean, variance, feature_value):
        variance = max(float(variance), 1e-12)
        curvature = max(predicted_probability * (1.0 - predicted_probability), min_learning_curvature)
        precision = (1.0 / variance) + curvature * feature_value * feature_value
        new_variance = 1.0 / precision
        new_mean = float(mean) + new_variance * feature_value * (actual - predicted_probability)
        return new_mean, new_variance

    intercept_chunk_name = info["chunks"]["intercept"]["chosen_name"]
    if intercept_chunk_name:
        intercept_chunk = memory.get_chunk(intercept_chunk_name)
        if intercept_chunk:
            mean, variance = update_belief(
                intercept_chunk.slots.get("mu", 0.0),
                intercept_chunk.slots.get("var", 1.0),
                1.0,
            )
            intercept_chunk.slots["mu"] = mean
            intercept_chunk.slots["var"] = variance

    for feature_info in info["chunks"]["features"]:
        chunk_name = feature_info.get("chosen_name")
        if not chunk_name:
            continue
        feature_key = feature_info.get("key", "")
        if selected is not None and _base_feature_index(feature_key) not in selected:
            continue
        coefficient_chunk = memory.get_chunk(chunk_name)
        if not coefficient_chunk:
            continue
        mean, variance = update_belief(
            coefficient_chunk.slots.get("mu", 0.0),
            coefficient_chunk.slots.get("var", 1.0),
            float(feature_info["value"]),
        )
        coefficient_chunk.slots["mu"] = mean
        coefficient_chunk.slots["var"] = variance

    memory.tick(20)


def _denormalize(normalized_value: float, low: float, high: float) -> float:
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return float(normalized_value)
    return float(low + normalized_value * (high - low))


def _numeric_boundary_delta(
    value_norm: float,
    coefficient: float,
    score: float,
    bounds: Tuple[float, float],
    *,
    flip_direction: bool,
    feasibility_leeway: float,
):
    if abs(coefficient) < 1e-12:
        return False, 0.0

    delta_norm = -score / coefficient
    if flip_direction:
        delta_norm = -delta_norm
    target_norm = value_norm + delta_norm

    low, high = bounds
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return False, 0.0
    if np.isfinite(target_norm) and 0.0 <= target_norm <= 1.0:
        target_orig = _denormalize(target_norm, low, high)
    elif np.isfinite(target_norm) and -feasibility_leeway <= target_norm <= 1.0 + feasibility_leeway:
        target_orig = _denormalize(1.0 if delta_norm > 0.0 else 0.0, low, high)
    else:
        return False, 0.0

    step = slider_step((low, high))
    current_orig = _denormalize(value_norm, low, high)
    landed_orig = snap_to_step(target_orig, (low, high), step)
    return low <= landed_orig <= high, float(landed_orig - current_orig)


def _save_lr_change_combo(memory, output):
    memory_time = getattr(getattr(memory, "dm", memory), "time", 0.0)
    by_direction = {"increase": {}, "decrease": {}}
    for feature_key, values in output.items():
        probability = float(values.get("p_selected", 0.0))
        delta = float(values.get("mean_delta", 0.0))
        if probability <= 0.0:
            continue
        if "=" in feature_key:
            by_direction["increase"][feature_key] = values
            by_direction["decrease"][feature_key] = values
        elif delta > 0:
            by_direction["increase"][feature_key] = values
        elif delta < 0:
            by_direction["decrease"][feature_key] = values

    for direction, features in by_direction.items():
        if not features:
            continue
        mass = float(sum(values["p_selected"] for values in features.values()))
        probabilities = {
            key: float(values["p_selected"] / mass) for key, values in features.items()
        }
        slots = {
            "type": "lr_change_combo",
            "direction": direction,
            "features": list(features.keys()),
            "p_select": probabilities,
            "delta": {key: float(values["mean_delta"]) for key, values in features.items()},
            "time": {key: float(values["mean_time"]) for key, values in features.items()},
            "mass": mass,
            "n_updates": 1,
            "expected_time": float(
                sum(probabilities[key] * features[key]["mean_time"] for key in features)
            ),
        }
        chunk_name = f"lr_change_combo:{direction}"
        chunk = memory.get_chunk(chunk_name) if hasattr(memory, "get_chunk") else None
        if chunk is None:
            chunk = memory.add_chunk(chunk_name, slots, update_retrieval=False)
        else:
            chunk.slots.update(slots)
        if hasattr(chunk, "add_prob_refresh"):
            chunk.add_prob_refresh(memory_time, max(0.0, min(1.0, mass)))


def cf_lr_heuristic(
    feature_vector,
    memory,
    lr_exp,
    bounds: Dict[str, Tuple[float, float]],
    *,
    retrieval_candidate_count: int = 3,
    read_seconds_per_item: float = READ_TIME,
    mental_calculation_seconds: float = MENTAL_CALCULATION_TIME,
    selected_feature_indices: Optional[List[int]] = None,
    feasibility_leeway: float = 2.0,
    actual_label: Optional[int] = None,
) -> Dict[str, Dict[str, float]]:
    rng = np.random.default_rng()
    x_norm = np.asarray(feature_vector, dtype=float)
    intercept_retrieval, feature_beliefs, retrieval_time = _retrieve_lr_beliefs(
        memory, lr_exp, x_norm, retrieval_candidate_count, selected_feature_indices
    )

    intercept_chunk = _sample_retrieved_chunk(intercept_retrieval, rng)
    intercept_mean, intercept_variance = _chunk_mean_variance(intercept_chunk, missing_variance=0.01)
    score = rng.normal(intercept_mean, math.sqrt(max(intercept_variance, 1e-12)))

    coefficient_stats = {}
    for feature_key, value_norm, _is_numeric, coefficient_retrieval in feature_beliefs:
        coefficient_chunk = _sample_retrieved_chunk(coefficient_retrieval, rng)
        coefficient_mean, coefficient_variance = _chunk_mean_variance(
            coefficient_chunk, missing_variance=1.0
        )
        coefficient_stats[feature_key] = (coefficient_mean, coefficient_variance)
        score += coefficient_mean * value_norm

    score_sign = 1 if score >= 0.0 else -1
    if actual_label is None:
        flip_direction = False
    else:
        actual_sign = 1 if int(actual_label) == 1 else -1
        flip_direction = score_sign != actual_sign

    base_time = (
        retrieval_time
        + read_seconds_per_item * len(feature_beliefs)
        + mental_calculation_seconds * len(feature_beliefs)
    )
    output = {}
    weights = {}
    for feature_key, value_norm, is_numeric, _coefficient_retrieval in feature_beliefs:
        coefficient_mean, coefficient_variance = coefficient_stats.get(feature_key, (0.0, 1.0))
        delta = 0.0
        feasible = True
        if is_numeric:
            feature_bounds = bounds.get(feature_key, (-np.inf, np.inf))
            feasible, delta = _numeric_boundary_delta(
                value_norm,
                coefficient_mean,
                score,
                feature_bounds,
                flip_direction=flip_direction,
                feasibility_leeway=feasibility_leeway,
            )
        else:
            if flip_direction:
                delta = -value_norm if value_norm >= 0.5 else 0.0
            else:
                delta = 1.0 - value_norm if value_norm < 0.5 else 0.0

        signal_to_noise = abs(coefficient_mean) / math.sqrt(coefficient_variance + 1e-8)
        weights[feature_key] = signal_to_noise if feasible and abs(delta) > 0.0 else 0.0
        output[feature_key] = {
            "p_selected": 0.0,
            "mean_delta": float(delta),
            "mean_time": float(base_time + (mental_calculation_seconds if delta != 0.0 else 0.0)),
        }

    total_weight = sum(weights.values())
    if total_weight <= 0.0 and output:
        total_weight = float(len(output))
        weights = {key: 1.0 for key in output}
    for feature_key in output:
        output[feature_key]["p_selected"] = float(weights.get(feature_key, 0.0) / total_weight)

    if memory is not None and output:
        _save_lr_change_combo(memory, output)

    output["expected_time"] = float(
        sum(values["p_selected"] * values["mean_time"] for values in output.values())
    )
    return output

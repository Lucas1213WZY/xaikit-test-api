from collections import Counter
from typing import Any, Dict, Optional, Tuple

import math
import random

import numpy as np

from .cognitive import (
    DDM_NON_DECISION_TIME,
    MENTAL_CALCULATION_TIME,
    READ_TIME,
    drift_diffusion_decision,
    round_to_sf,
    slider_step,
    snap_to_step,
)
from .memory import CombinedMemory, build_number_profile, digits_to_value, remember_number_to_sf

DT_BRANCH_EVIDENCE = 0.2

# ==============================
# 1) Store DT in memory (simple)
# ==============================
def add_dt_to_memory(memory, dt_exp, *, threshold_significant_figures: int = 2):
    """
    Store a decision tree in memory.

    For each internal node nid with feature feat_key:
      - FEATURE:     name="Node_{nid}_feature"
                     slots={"type":"feature","node":nid,"feat_key":feat_key,"depth":d}
      - THRESH PTR:  name="Node_{nid}_thr_ptr"  (numeric only)
                     slots={"type":"thr_ptr","node":nid,"feat_key":feat_key,"thr_key": f"thr:{nid}:{feat_key}","depth":d}
      - THRESH NUM:  remember_number_to_sf(memory, key=f"thr:{nid}:{feat_key}", value=threshold, max_sf=threshold_significant_figures)

    Also writes:
      - NODE TYPE:   "Node_{nid}_type"  {"type":"node_type","node":nid,"is_leaf":bool,"depth":d}
      - CHILD PTRS:  "Node_{nid}_left"/"Node_{nid}_right" -> child nid
      - LEAF CLASS:  "Node_{nid}_class" {"type":"class_label","node":nid,"value":label,"depth":d}
    """
    nodes = {n["node"]: n for n in dt_exp.tree_structure}

    def _walk(nid: int, depth: int = 0):
        n = nodes[nid]
        memory.add_chunk(f"Node_{nid}_type",
                         {"type":"node_type","node":nid,"is_leaf":bool(n["is_leaf"]),"depth":depth})

        if n["is_leaf"]:
            maj = int(np.argmax(n["value"]))
            label = (dt_exp.class_labels[maj]
                     if getattr(dt_exp, "class_labels", None) and maj < len(dt_exp.class_labels)
                     else f"class {maj}")
            memory.add_chunk(f"Node_{nid}_class",
                             {"type":"class_label","node":nid,"value":label,"depth":depth})
            return

        feat_key = n["feature"]
        memory.add_chunk(f"Node_{nid}_feature",
                         {"type":"feature","node":nid,"feat_key":feat_key,"depth":depth})

        # Two-step threshold: pointer + stored number (numeric only)
        if "=" not in feat_key:
            thr_key = f"thr:{nid}:{feat_key}"
            memory.add_chunk(f"Node_{nid}_thr_ptr",
                             {"type":"thr_ptr","node":nid,"feat_key":feat_key,"thr_key":thr_key,"depth":depth})
            remember_number_to_sf(memory, key=thr_key, value=float(n["threshold"]), max_sf=threshold_significant_figures)

        # Child pointers
        memory.add_chunk(f"Node_{nid}_left",
                         {"type":"child_ptr","node":nid,"which":"left","value":n["left"],"depth":depth})
        memory.add_chunk(f"Node_{nid}_right",
                         {"type":"child_ptr","node":nid,"which":"right","value":n["right"],"depth":depth})

        _walk(n["left"], depth+1)
        _walk(n["right"], depth+1)

    _walk(0)

def _is_cat(k: str) -> bool:
    return ("=" in k)

def _feat_base(k: str) -> str:
    return k.split("=")[0] if "=" in k else k

def _base_index_from_key(key: str) -> int:
    """Return the integer column index from keys like 'a3' or 'a3=1'."""
    return int(_feat_base(key)[1:])

def _categorical_go_left(raw_value, cat_idx, threshold) -> bool:
    """Decision-tree categorical splits are stored as one-hot equality tests."""
    encoded_value = 1.0 if int(raw_value) == int(cat_idx) else 0.0
    return bool(encoded_value <= float(threshold))

def _node_lookup(dt_exp: Any) -> dict:
    return {n["node"]: n for n in dt_exp.tree_structure}

def _class_count(nodes: dict) -> int:
    leaf = next(n for n in nodes.values() if n["is_leaf"])
    return len(leaf["value"])

def _majority_class(node: dict) -> int:
    return int(np.argmax(node["value"]))

def _round_for_display(value: float, significant_figures: int) -> float:
    return float(round_to_sf(float(value), significant_figures))

def _memory_time(memory: Any) -> float:
    return getattr(getattr(memory, "dm", None), "time", getattr(memory, "time", 0.0))

def _weighted_pick(options, rng):
    """
    Pick one option using its 'prob' field.

    This is intentionally tiny and allocation-light because DT traversal calls it
    many times inside Monte Carlo loops.
    """
    total = 0.0
    for option in options:
        total += float(option["prob"])
    if total <= 0.0:
        return None

    draw = rng.random() * total
    cumulative = 0.0
    for option in options:
        cumulative += float(option["prob"])
        if draw <= cumulative:
            return option
    return options[-1]

def _draw_topk_filtered(retrieval_result: dict, rng, used_chunks: set):
    """
    Sample from a precomputed retrieval distribution.

    Chunks already used on the same traversal path are ignored so that a simulated
    person does not reuse the same memory chunk twice in one path.
    """
    candidates = []
    for chunk, probability in retrieval_result.get("top_k", []):
        if chunk is None or chunk not in used_chunks:
            candidates.append({"chunk": chunk, "prob": float(probability)})
    candidates.append({"chunk": None, "prob": float(retrieval_result.get("p_none", 0.0))})

    picked = _weighted_pick(candidates, rng)
    return None if picked is None else picked["chunk"]

def _retrieval_plan(memory, dt_exp, displayed_significant_figures: int, retrieval_candidate_count: int):
    """
    Precompute all retrieval menus once, before Monte Carlo traversal starts.

    The expensive part is scoring every chunk in memory. Doing it here means each
    rollout only samples from small top-k menus instead of rescoring memory.
    """
    feature_topk = {}
    threshold_pointer_topk = {}
    number_profiles = {}

    for node in dt_exp.tree_structure:
        if node["is_leaf"]:
            continue

        node_id = node["node"]
        feature_key = node["feature"]
        feature_topk[node_id] = memory.topk_retrievals_with_prob_refresh(
            {"type": "feature", "node": node_id},
            retrieval_candidate_count=retrieval_candidate_count,
            memory_refresh_probability=0.0,
            add_refresh=False,
        )

        if not _is_cat(feature_key):
            threshold_pointer_topk[node_id] = memory.topk_retrievals_with_prob_refresh(
                {"type": "thr_ptr", "node": node_id, "feat_key": feature_key},
                retrieval_candidate_count=retrieval_candidate_count,
                memory_refresh_probability=0.0,
                add_refresh=False,
            )
            for chunk, _probability in threshold_pointer_topk[node_id].get("top_k", []):
                if chunk is None:
                    continue
                threshold_key = chunk.slots.get("thr_key")
                if threshold_key and threshold_key not in number_profiles:
                    number_profiles[threshold_key] = build_number_profile(
                        memory,
                        key=threshold_key,
                        requested_significant_figures=displayed_significant_figures,
                        retrieval_candidate_count=retrieval_candidate_count,
                        memory_refresh_probability=0.0,
                    )

    return feature_topk, threshold_pointer_topk, number_profiles

def _retrieved_feature_is_valid(node_feature: str, retrieved_feature: str) -> bool:
    """A retrieved feature must match the current DT node's split type and base feature."""
    if _is_cat(node_feature):
        return retrieved_feature == node_feature
    return (not _is_cat(retrieved_feature)) and (_feat_base(retrieved_feature) == node_feature)

def _sample_threshold_from_profile(
    profile: dict,
    *,
    displayed_significant_figures: int,
    rng,
    used_meta_chunks: set,
    used_digit_chunks: set,
) -> Optional[float]:
    """
    Reconstruct a threshold from the remembered sign/scale and digit chunks.

    Missing metadata means the number cannot be reconstructed. Missing later
    digits shortens the remembered precision, which mirrors retrieve_number_to_sf.
    """
    meta_options = [
        option
        for option in profile["meta_with_chunks"]
        if option["chunk_name"] is None or option["chunk_name"] not in used_meta_chunks
    ]
    chosen_meta = _weighted_pick(meta_options, rng)
    if chosen_meta is None or chosen_meta["value"] is None:
        return None
    if chosen_meta["chunk_name"] is not None:
        used_meta_chunks.add(chosen_meta["chunk_name"])

    sign, p10 = chosen_meta["value"]
    digits = []
    for position in range(displayed_significant_figures):
        digit_options = [
            option
            for option in profile["digits_with_chunks"][position]
            if option["chunk_name"] is None or option["chunk_name"] not in used_digit_chunks
        ]
        chosen_digit = _weighted_pick(digit_options, rng)
        if chosen_digit is None or chosen_digit["value"] is None:
            break
        if chosen_digit["chunk_name"] is not None:
            used_digit_chunks.add(chosen_digit["chunk_name"])
        digits.append(int(chosen_digit["value"]))

    return digits_to_value(sign, p10, digits, len(digits)) if digits else 0.0

def _safe_uniform(num_classes: int) -> np.ndarray:
    return np.full(num_classes, 1.0 / num_classes, dtype=float)

def dt_traverse(
    feature_vector,
    memory,
    dt_exp,
    *,
    explanation_access_mode: str = "retrieve",
    displayed_significant_figures: int = 2,
    read_seconds_per_item: float = READ_TIME,
    decision_boundary: float = 1.5,
    decision_noise: float = 1.0,
    simulation_sample_count: int = 64,
    retrieval_candidate_count: int = 3,
    max_memory_refresh_probability: float = 1.0,
    rng: Optional[random.Random] = None,
):
    """
    Traverse a decision tree with either visible explanation text or memory recall.

    Returns:
      (class_probabilities, expected_time, info)

    Read mode uses the literal tree feature/threshold values. Retrieve mode first
    precomputes memory retrieval distributions, then each Monte Carlo rollout only
    samples from those small distributions.
    """
    if explanation_access_mode not in {"read", "retrieve"}:
        raise ValueError('explanation_access_mode must be "read" or "retrieve"')
    if explanation_access_mode == "retrieve" and memory is None:
        raise ValueError('explanation_access_mode="retrieve" requires a memory object.')

    rng = rng or random.Random()
    x = np.asarray(feature_vector, dtype=float)
    nodes = _node_lookup(dt_exp)
    num_classes = _class_count(nodes)
    uniform = _safe_uniform(num_classes)
    sample_count = max(1, int(simulation_sample_count))

    # Scoring memory scans every chunk. Pre-plan once so the inner Monte Carlo
    # loop only draws from already-computed top-k menus.
    feature_topk = {}
    threshold_pointer_topk = {}
    number_profiles = {}
    if explanation_access_mode == "retrieve":
        feature_topk, threshold_pointer_topk, number_profiles = _retrieval_plan(
            memory,
            dt_exp,
            displayed_significant_figures,
            retrieval_candidate_count,
        )

    probs_acc = np.zeros(num_classes, dtype=float)
    time_acc = 0.0
    read_time = 0.0
    retrieve_time = 0.0
    decision_time = 0.0

    feature_sel_counts = Counter()
    threshold_key_counts = Counter()
    leaf_counts = Counter()
    categorical_decisions = 0
    numeric_decisions = 0

    for _ in range(sample_count):
        node_id = 0
        run_time = 0.0

        # No-reuse state is per rollout/path. It prevents sampling the same
        # remembered chunk twice during one simulated traversal.
        used_feature_chunks = set()
        used_pointer_chunks = set()
        used_threshold_keys = set()
        used_number_meta_chunks = set()
        used_number_digit_chunks = set()

        while True:
            node = nodes[node_id]

            if node["is_leaf"]:
                probs_acc[_majority_class(node)] += 1.0
                leaf_counts[node_id] += 1
                break

            node_feature = node["feature"]
            node_is_categorical = _is_cat(node_feature)

            # 1) Feature label: read it from the explanation, or recall it.
            if explanation_access_mode == "read":
                run_time += read_seconds_per_item
                read_time += read_seconds_per_item
                feature_key = node_feature
            else:
                feature_result = feature_topk[node_id]
                feature_rt = float(feature_result["expected_rt"])
                run_time += feature_rt
                retrieve_time += feature_rt

                feature_chunk = _draw_topk_filtered(feature_result, rng, used_feature_chunks)
                if feature_chunk is None:
                    probs_acc += uniform
                    break

                used_feature_chunks.add(feature_chunk)
                feature_sel_counts[feature_chunk] += 1
                feature_key = feature_chunk.slots.get("feat_key", node_feature)

                # If memory supplies the wrong feature for this node, the model
                # treats the traversal as lost and assigns uniform class mass.
                if not _retrieved_feature_is_valid(node_feature, feature_key):
                    probs_acc += uniform
                    break

            # 2) Numeric threshold: categorical tests carry their category in the
            # feature key, so only numeric tests need threshold retrieval/reading.
            threshold_value = None
            if not node_is_categorical:
                if explanation_access_mode == "read":
                    run_time += read_seconds_per_item
                    read_time += read_seconds_per_item
                    threshold_value = _round_for_display(node["threshold"], displayed_significant_figures)
                else:
                    pointer_result = threshold_pointer_topk.get(node_id)
                    if pointer_result is None:
                        probs_acc += uniform
                        break

                    pointer_rt = float(pointer_result["expected_rt"])
                    run_time += pointer_rt
                    retrieve_time += pointer_rt

                    pointer_chunk = _draw_topk_filtered(pointer_result, rng, used_pointer_chunks)
                    if pointer_chunk is None:
                        probs_acc += uniform
                        break

                    used_pointer_chunks.add(pointer_chunk)
                    threshold_key = pointer_chunk.slots.get("thr_key")
                    if (
                        not threshold_key
                        or threshold_key not in number_profiles
                        or threshold_key in used_threshold_keys
                    ):
                        probs_acc += uniform
                        break

                    used_threshold_keys.add(threshold_key)
                    threshold_key_counts[threshold_key] += 1

                    number_profile = number_profiles[threshold_key]
                    threshold_value = _sample_threshold_from_profile(
                        number_profile,
                        displayed_significant_figures=displayed_significant_figures,
                        rng=rng,
                        used_meta_chunks=used_number_meta_chunks,
                        used_digit_chunks=used_number_digit_chunks,
                    )
                    if threshold_value is None:
                        probs_acc += uniform
                        break

                    threshold_value = _round_for_display(threshold_value, displayed_significant_figures)
                    number_rt = float(number_profile["expected_rt"])
                    run_time += number_rt
                    retrieve_time += number_rt

            # 3) Read the current instance value used by the split.
            run_time += read_seconds_per_item
            read_time += read_seconds_per_item
            if node_is_categorical:
                base, category_index = node_feature.split("=")
                feature_index = _base_index_from_key(base)
                go_left_literal = _categorical_go_left(
                    x[feature_index],
                    category_index,
                    node["threshold"],
                )
            else:
                feature_index = _base_index_from_key(feature_key)
                stimulus_value = float(x[feature_index])

            # 4) Choose the branch. Both numeric comparisons and categorical
            # equality tests use weak signed evidence so branch decisions stay
            # stochastic under the DDM.
            if node_is_categorical:
                evidence = DT_BRANCH_EVIDENCE if go_left_literal else -DT_BRANCH_EVIDENCE
                p_left, rt_decision, _drift = drift_diffusion_decision(
                    evidence,
                    decision_boundary=decision_boundary,
                    decision_noise=decision_noise,
                    gain=1.0,
                )
                go_left = rng.random() < p_left
                categorical_decisions += 1
            else:
                if stimulus_value == threshold_value:
                    evidence = 0
                elif stimulus_value < threshold_value:
                    evidence = DT_BRANCH_EVIDENCE
                else:
                    evidence = -DT_BRANCH_EVIDENCE

                p_left, rt_decision, _drift = drift_diffusion_decision(
                    evidence,
                    decision_boundary=decision_boundary,
                    decision_noise=decision_noise,
                    gain=1.0,
                )
                go_left = rng.random() < p_left
                numeric_decisions += 1

            decision_time += rt_decision
            run_time += rt_decision
            node_id = node["left"] if go_left else node["right"]

        time_acc += run_time

    probs = probs_acc / float(sample_count)
    expected_time = float(time_acc) / float(sample_count)
    read_time /= float(sample_count)
    retrieve_time /= float(sample_count)
    decision_time /= float(sample_count)

    # Refresh memory after estimating the rollout distribution. This preserves the
    # "many possible paths, one expected time tick" interpretation.
    if explanation_access_mode == "retrieve" and feature_sel_counts:
        now = _memory_time(memory)
        for chunk, count in feature_sel_counts.items():
            refresh_probability = min(max_memory_refresh_probability, float(count) / float(sample_count))
            if refresh_probability > 0.0:
                chunk.add_prob_refresh(now, refresh_probability)

    if explanation_access_mode == "retrieve" and threshold_key_counts:
        for threshold_key, count in threshold_key_counts.items():
            refresh_probability = min(max_memory_refresh_probability, float(count) / float(sample_count))
            if refresh_probability > 0.0:
                build_number_profile(
                    memory,
                    key=threshold_key,
                    requested_significant_figures=displayed_significant_figures,
                    retrieval_candidate_count=retrieval_candidate_count,
                    memory_refresh_probability=refresh_probability,
                )

    if hasattr(memory, "tick"):
        memory.tick(expected_time)

    info = {
        "explanation_access_mode": explanation_access_mode,
        "simulation_sample_count": int(sample_count),
        "displayed_significant_figures": int(displayed_significant_figures),
        "ddm": {"a": decision_boundary, "s": decision_noise, "Tnd": DDM_NON_DECISION_TIME},
        "read_seconds_per_item": read_seconds_per_item,
        "refresh_counts": {
            "feature": {getattr(chunk, "name", ""): int(count) for chunk, count in feature_sel_counts.items()},
            "thr_key": dict(threshold_key_counts),
        },
        "timing": {
            "read_time": float(read_time),
            "retrieve_time": float(retrieve_time),
            "decision_time": float(decision_time),
            "expected_time": float(expected_time),
        },
        "path": {
            "leaf_counts": {int(node_id): int(count) for node_id, count in leaf_counts.items()},
            "categorical_decisions": int(categorical_decisions),
            "numeric_decisions": int(numeric_decisions),
        },
    }
    return probs, expected_time, info

def refresh_dt_path_in_memory(
    memory, dt_exp, feature_vector, *, threshold_significant_figures: int = 2
):
    """
    Deterministically traverse the DT using dt_exp thresholds and the instance values,
    and refresh the chunks that *would* be used along that path (no probabilities).
    Uses chunk.update_retrieval(now) for a direct, non-probabilistic refresh.
    """
    x = np.asarray(feature_vector, float)
    nodes = {n["node"]: n for n in dt_exp.tree_structure}
    now = memory.dm.time if hasattr(memory, "dm") else memory.time

    nid = 0
    while True:
        n = nodes[nid]

        # Leaf: refresh its class chunk and stop
        if n["is_leaf"]:
            leaf_nm = f"Node_{nid}_class"
            ch = memory.get_chunk(leaf_nm)
            if ch: ch.update_retrieval(now)
            break

        # Feature chunk (used at this node)
        feat_key = n["feature"]
        feat_nm = f"Node_{nid}_feature"
        chf = memory.get_chunk(feat_nm)
        if chf: chf.update_retrieval(now)

        # Decide branch using literal values from dt_exp / instance
        if "=" in feat_key:
            base, cat_idx = feat_key.split("=")
            att = int(base[1:])
            go_left = _categorical_go_left(x[att], cat_idx, n["threshold"])
            # Child pointer chunk actually used
            child_nm = f"Node_{nid}_{'left' if go_left else 'right'}"
            ch_child = memory.get_chunk(child_nm)
            if ch_child: ch_child.update_retrieval(now)
            # Move to child
            nid = n["left"] if go_left else n["right"]

        else:
            # Numeric threshold path
            # Refresh the threshold pointer chunk
            thr_ptr_nm = f"Node_{nid}_thr_ptr"
            chp = memory.get_chunk(thr_ptr_nm)
            if chp: chp.update_retrieval(now)

            # Also refresh the number memory that was stored by remember_number_to_sf
            thr_key = f"thr:{nid}:{feat_key}"
            meta_nm = f"num:{thr_key}:meta"
            chm = memory.get_chunk(meta_nm)
            if chm: chm.update_retrieval(now)

            # Refresh first `threshold_significant_figures` digit chunks (if present)
            for pos in range(1, threshold_significant_figures + 1):
                d_nm = f"num:{thr_key}:d{pos}"
                cd = memory.get_chunk(d_nm)
                if cd: cd.update_retrieval(now)

            # Compare and follow the used child; refresh that pointer
            att = int(feat_key[1:])
            go_left = (float(x[att]) <= float(n["threshold"]))
            child_nm = f"Node_{nid}_{'left' if go_left else 'right'}"
            ch_child = memory.get_chunk(child_nm)
            if ch_child: ch_child.update_retrieval(now)
            nid = n["left"] if go_left else n["right"]

def cf_change_path_dt(
    feature_vector: np.ndarray,
    dt_exp: Any,
    bounds: Dict[str, Tuple[float, float]],
    *,
    explanation_access_mode: str,                              # "read" (with XAI) or "retrieve" (without XAI)
    counterfactual_tree_depth: Optional[int] = None,     # Laplace center over decision depths
    displayed_significant_figures: int = 2,
    # timing (aligned with dt_traverse)
    read_seconds_per_item: float = READ_TIME,
    mental_calculation_seconds: float = MENTAL_CALCULATION_TIME,
    decision_boundary: float = 1.5,
    decision_noise: float = 1.0,
    # retrieve-explanation_access_mode configs
    memory: Any = None,
    simulation_sample_count: int = 64,
    retrieval_candidate_count: int = 3,
    max_memory_refresh_probability: float = 1.0,          # used only in retrieve-explanation_access_mode
    # depth selection smoothing
    depth_choice_temperature: float = 1.0,                       # smaller => sharper around center
    # rng
    rng: Optional[np.random.Generator] = None,
):
    """
    With XAI ("read"):   Expected selection over depths via Laplace weights (no depth sampling).
    Without XAI ("retrieve"): Monte Carlo; each run samples depth via Laplace, traverses stochastically,
                              then attempts a minimal flip at the chosen depth.

    Returns:
      out: dict { "a0": {"p_selected", "mean_delta", "mean_time"}, ... }
    """
    rng = rng or np.random.default_rng()
    rnd = random.Random(rng.integers(0, 2**31 - 1))

    x = np.asarray(feature_vector, dtype=float)
    nodes = {n["node"]: n for n in dt_exp.tree_structure}
    k = len(x)

    # ---------- small helpers ----------
    def _is_cat_key(key: str) -> bool:
        return "=" in key

    def _feat_key_for_node(n) -> str:
        return n["feature"]

    def _thr_for_node(n) -> float:
        return float(n["threshold"])

    def _left(n) -> int:
        return int(n["left"])

    def _right(n) -> int:
        return int(n["right"])

    def _base_index_from_key(key: str) -> int:
        base = key.split('=')[0]  # "aN" from "aN" or "aN=K"
        return int(base[1:])

    def _bounds_for_feat_key(fk: str) -> Tuple[float, float]:
        if _is_cat_key(fk):
            return (-np.inf, np.inf)
        return bounds.get(fk, bounds.get(f"a{_base_index_from_key(fk)}", (-np.inf, np.inf)))

    def _round_sf(v: float) -> float:
        return float(round_to_sf(float(v), displayed_significant_figures))

    def _literal_path_for_x(x_arr: np.ndarray) -> list:
        """Deterministic literal route (uses true node thresholds & x<=thr for branch)."""
        path = []
        nid = 0
        while True:
            n = nodes[nid]
            path.append(n)
            if n["is_leaf"]:
                break
            fk = _feat_key_for_node(n)
            if _is_cat_key(fk):
                base, cat_idx = fk.split("=")
                col = _base_index_from_key(base)
                go_left = _categorical_go_left(x_arr[col], cat_idx, _thr_for_node(n))
            else:
                col = _base_index_from_key(fk)
                thr = _thr_for_node(n)
                go_left = float(x_arr[col]) <= float(thr)
            nid = _left(n) if go_left else _right(n)
        return path

    def _depth_weights(m: int, center: int) -> np.ndarray:
        idx = np.arange(m)
        depth_probability_floor = 1e-9
        w = np.exp(-np.abs(idx - center) / max(depth_choice_temperature, 1e-6)) + depth_probability_floor
        return (w / w.sum()).astype(float)

    def _flip_at_node(x_arr: np.ndarray, n) -> Tuple[Optional[str], float, float]:
        """
        Minimal flip across the test at node n.
        Returns (feat_key_used, delta_applied, extra_time).
        """
        fk = _feat_key_for_node(n)
        if _is_cat_key(fk):
            base, cat_idx = fk.split("=")
            col = _base_index_from_key(base)
            cur = int(x_arr[col])
            x_new = int(cat_idx) if cur != int(cat_idx) else (int(cat_idx) ^ 1)
            delta = float(x_new - cur)
            x_arr[col] = x_new
            return fk, delta, read_seconds_per_item + mental_calculation_seconds
        # numeric
        col = _base_index_from_key(fk)
        thr = _thr_for_node(n)
        lo, hi = _bounds_for_feat_key(fk)
        step = slider_step((lo, hi))
        cur = float(x_arr[col])
        target = thr + step if cur <= thr else thr - step
        target = snap_to_step(target, (lo, hi), step)
        # target = _round_sf(target)
        if not (lo <= target <= hi) or np.isclose(target, cur):
            return None, 0.0, read_seconds_per_item
        delta = float(target - cur)
        x_arr[col] = float(target)
        return fk, delta, read_seconds_per_item + mental_calculation_seconds

    # ---------- READ MODE (with XAI): expected distribution over depths ----------
    if explanation_access_mode == "read":
        # literal path defines decision nodes / indices
        path = _literal_path_for_x(x.copy())
        dec_nodes = [n for n in path if not n["is_leaf"]]
        if not dec_nodes:
            out = {f"a{i}": {"p_selected": 0.0, "mean_delta": 0.0, "mean_time": 0.0} for i in range(k)}
            return out

        m = len(dec_nodes)
        center = max(0, min(counterfactual_tree_depth if counterfactual_tree_depth is not None else m - 1, m - 1))
        w = _depth_weights(m, center)  # Laplace over indices

        # Precompute cumulative basic time to reach each depth index.
        # Counterfactual strategies do not include DDM time.
        cum_time = np.zeros(m, float)
        running = 0.0
        for i, n in enumerate(dec_nodes):
            fk = _feat_key_for_node(n)
            if _is_cat_key(fk):
                running += read_seconds_per_item + read_seconds_per_item + mental_calculation_seconds
            else:
                running += read_seconds_per_item + read_seconds_per_item + read_seconds_per_item + mental_calculation_seconds
            cum_time[i] = running

        # Aggregate expected selection outcomes over depths
        out = {f"a{i}": {"p_selected": 0.0, "mean_delta": 0.0, "mean_time": 0.0} for i in range(k)}
        for i, n in enumerate(dec_nodes):
            x_tmp = x.copy()
            fk, dlt, t_flip = _flip_at_node(x_tmp, n)
            if fk is None:
                continue
            col = _base_index_from_key(fk.split("=")[0] if "=" in fk else fk)
            out_key = f"a{col}"
            out[out_key]["p_selected"] += float(w[i])
            out[out_key]["mean_delta"] += float(w[i]) * float(dlt)
            out[out_key]["mean_time"]  += float(w[i]) * float(cum_time[i] + t_flip)



        # normalize_feature_probabilities mean_time/delta by probability mass per feature (so they are conditional means)
        for key, vals in out.items():
            p = vals["p_selected"]
            if p > 0:
                vals["mean_delta"] = vals["mean_delta"] / p
                vals["mean_time"]  = vals["mean_time"]  / p

        expected_time = sum(v["p_selected"] * v["mean_time"] for v in out.values())
        out["expected_time"] = float(expected_time)
        return out

    # ---------- RETRIEVE MODE (without XAI): Monte-Carlo with Laplace depth each run ----------
    if explanation_access_mode != "retrieve":
        raise ValueError('explanation_access_mode must be "read" or "retrieve"')

    if memory is None:
        raise ValueError("explanation_access_mode='retrieve' requires a `memory` object.")

    # literal path only to define max possible decision-depth cardinality for Laplace
    lit_path = _literal_path_for_x(x.copy())
    lit_dec_nodes = [n for n in lit_path if not n["is_leaf"]]
    if not lit_dec_nodes:
        out = {f"a{i}": {"p_selected": 0.0, "mean_delta": 0.0, "mean_time": 0.0} for i in range(k)}
        out["expected_time"] = 0.0
        return out

    m = len(lit_dec_nodes)
    center = max(0, min(counterfactual_tree_depth if counterfactual_tree_depth is not None else m - 1, m - 1))
    base_w = _depth_weights(m, center)

    # ===== Pre-plan retrieval menus (no refresh, no time advance) =====
    def _topk_no_refresh(req: dict):
        return memory.topk_retrievals_with_prob_refresh(
            req, retrieval_candidate_count=retrieval_candidate_count, memory_refresh_probability=0.0, add_refresh=False
        )

    feature_topk = {}
    thrptr_topk  = {}
    num_profiles = {}

    for n in dt_exp.tree_structure:
        if n["is_leaf"]:
            continue
        nid = n["node"]
        feature_topk[nid] = _topk_no_refresh({"type": "feature", "node": nid})

        if not _is_cat_key(n["feature"]):
            thrptr_topk[nid] = _topk_no_refresh({"type": "thr_ptr", "node": nid, "feat_key": n["feature"]})
            # pre-build number profiles for any threshold-pointer choices
            for ch, _p in thrptr_topk[nid].get("top_k", []):
                if ch is None:
                    continue
                thr_key = ch.slots.get("thr_key")
                if thr_key and thr_key not in num_profiles:
                    num_profiles[thr_key] = build_number_profile(
                        memory, key=thr_key, requested_significant_figures=displayed_significant_figures, retrieval_candidate_count=retrieval_candidate_count, memory_refresh_probability=0.0
                    )

    # ===== Helpers for MC traversal using pre-planned menus =====
    def _draw_topk_filtered(retrieval_result, used_set):
        """Sample from pre-planned top-k but avoid reusing chunks; include None via p_none."""
        choices, probs = [], []
        for ch, p in retrieval_result.get("top_k", []):
            if ch is None or ch not in used_set:
                choices.append(ch)
                probs.append(float(p))
        choices.append(None)
        probs.append(float(retrieval_result.get("p_none", 0.0)))
        tot = sum(probs)
        if tot <= 0:
            return None
        return rnd.choices(choices, weights=probs, k=1)[0]

    def _pick_one(options):
        if not options:
            return None, None
        w = [float(o["prob"]) for o in options]
        tot = sum(w)
        if tot <= 0:
            return None, None
        idx = rnd.choices(range(len(options)), weights=w, k=1)[0]
        return idx, options[idx]

    def _traverse_once_to_depth_retrieve_planned(x_arr: np.ndarray, target_idx: int):
        """
        Uses pre-planned feature_topk, thrptr_topk, and num_profiles.
        Returns:
          path_nodes: list of nodes visited
          run_time: float
          usage: dict (for refresh accounting)
          thr_seq: list of threshold values per decision node in order (None for categorical)
        """
        path = []
        node_id = 0
        run_time = 0.0
        d = 0

        used_feature_chunks = set()
        used_thrptr_chunks  = set()
        used_thr_keys       = set()
        used_num_meta_chunks  = set()
        used_num_digit_chunks = set()

        feat_chunk_counts = {}
        thr_key_local     = {}
        thr_seq           = []  # aligned with decision nodes encountered on this path

        while True:
            n = nodes[node_id]
            path.append(n)
            if n["is_leaf"]:
                break
            if d == target_idx:
                break

            node_feat = n["feature"]
            is_cat = _is_cat_key(node_feat)

            # --- feature retrieval ---
            retF = feature_topk[n["node"]]
            run_time += float(retF.get("expected_rt", 0.0))
            chF = _draw_topk_filtered(retF, used_feature_chunks)
            if chF is None:
                break
            used_feature_chunks.add(chF)
            feat_chunk_counts[chF] = feat_chunk_counts.get(chF, 0) + 1
            feat_key = chF.slots.get("feat_key", node_feat)

            # validate against node
            if is_cat:
                if feat_key != node_feat:
                    break
            else:
                if _is_cat_key(feat_key) or (feat_key.split('=')[0] != node_feat):
                    break

            # --- threshold retrieval (numeric only) ---
            thr_val = None
            if not is_cat:
                retP = thrptr_topk.get(n["node"])
                if retP is None:
                    break
                run_time += float(retP.get("expected_rt", 0.0))
                chP = _draw_topk_filtered(retP, used_thrptr_chunks)
                if chP is None:
                    break
                used_thrptr_chunks.add(chP)

                thr_key = chP.slots.get("thr_key")
                if not thr_key or (thr_key not in num_profiles) or (thr_key in used_thr_keys):
                    break
                used_thr_keys.add(thr_key)
                thr_key_local[thr_key] = thr_key_local.get(thr_key, 0) + 1

                prof = num_profiles[thr_key]

                # meta
                meta_opts = [o for o in prof["meta_with_chunks"]
                             if (o["chunk_name"] is None) or (o["chunk_name"] not in used_num_meta_chunks)]
                if not meta_opts:
                    break
                _, chosen_meta = _pick_one(meta_opts)
                if chosen_meta is None or chosen_meta["value"] is None:
                    break
                if chosen_meta["chunk_name"] is not None:
                    used_num_meta_chunks.add(chosen_meta["chunk_name"])
                sign, p10 = chosen_meta["value"]

                # digits
                digits = []
                for pos in range(1, displayed_significant_figures + 1):
                    opts = [o for o in prof["digits_with_chunks"][pos - 1]
                            if (o["chunk_name"] is None) or (o["chunk_name"] not in used_num_digit_chunks)]
                    if not opts:
                        break
                    _, pick = _pick_one(opts)
                    if pick is None or pick["value"] is None:
                        break
                    if pick["chunk_name"] is not None:
                        used_num_digit_chunks.add(pick["chunk_name"])
                    digits.append(int(pick["value"]))
                thr_val = digits_to_value(sign, p10, digits, len(digits)) if digits else 0.0
                thr_val = _round_sf(thr_val)
                run_time += float(prof["expected_rt"])

            # read stimulus & DDM decision
            if is_cat:
                base, cat_idx = node_feat.split("=")
                att = int(base[1:])
                run_time += read_seconds_per_item
                go_left_literal = _categorical_go_left(x_arr[att], cat_idx, n["threshold"])
                e = DT_BRANCH_EVIDENCE if go_left_literal else -DT_BRANCH_EVIDENCE
                p_up, rt_decision, _ = drift_diffusion_decision(
                    e,
                    decision_boundary=decision_boundary,
                    decision_noise=decision_noise,
                    gain=1.0,
                )
                go_left = (rnd.random() < p_up)
                run_time += rt_decision + mental_calculation_seconds
                thr_seq.append(None)  # categorical
            else:
                att = int(node_feat[1:])
                run_time += read_seconds_per_item
                val = float(x_arr[att])
                if val == thr_val:
                    e = 0
                elif val < thr_val:
                    e = DT_BRANCH_EVIDENCE
                else:
                    e = -DT_BRANCH_EVIDENCE
                p_up, rt_decision, _ = drift_diffusion_decision(e, decision_boundary=decision_boundary, decision_noise=decision_noise, gain=1.0)
                go_left = (rnd.random() < p_up)
                run_time += rt_decision + mental_calculation_seconds
                thr_seq.append(float(thr_val))

            node_id = _left(n) if go_left else _right(n)
            d += 1

        usage = {"feature_chunks": feat_chunk_counts, "thr_keys": thr_key_local}
        return path, run_time, usage, thr_seq


    # ====== Monte-Carlo using preplanned menus ======
    select_mass = {f"a{i}": 0.0 for i in range(k)}
    delta_sum   = {f"a{i}": 0.0 for i in range(k)}
    time_sum    = {f"a{i}": 0.0 for i in range(k)}

    total_time_runs = 0.0

    # refresh ledgers
    feature_sel_counts = {}
    thr_key_counts     = {}

    # threshold accumulation for features we actually flipped at (for saving)
    thr_acc_sum  = {}  # feature -> sum(threshold used at picked node)
    thr_acc_cnt  = {}  # feature -> count

    N_runs = max(1, int(simulation_sample_count))
    N = 0
    for _ in range(N_runs):
        depth_idx = int(rng.choice(np.arange(m), p=base_w))
        path_stoch, run_time, usage, thr_seq = _traverse_once_to_depth_retrieve_planned(x.copy(), depth_idx)
        total_time_runs += float(run_time)

        # aggregate for refresh
        for ch, c in usage["feature_chunks"].items():
            feature_sel_counts[ch] = feature_sel_counts.get(ch, 0) + int(c)
        for tk, c in usage["thr_keys"].items():
            thr_key_counts[tk] = thr_key_counts.get(tk, 0) + int(c)

        dec_nodes = [n for n in path_stoch if not n["is_leaf"]]
        if not dec_nodes:
            continue
        picked_idx = min(depth_idx, len(dec_nodes) - 1)

        # attempt flip at chosen depth
        x_tmp = x.copy()
        fk, dlt, t_flip = _flip_at_node(x_tmp, dec_nodes[picked_idx])
        run_time += float(t_flip)

        if fk is None:
            continue

        col = _base_index_from_key(fk.split("=")[0] if "=" in fk else fk)
        fkey = f"a{col}"
        select_mass[fkey] += 1.0
        delta_sum[fkey]   += float(dlt)
        time_sum[fkey]    += float(run_time)

        # record threshold used at this decision depth if numeric
        thr_here = thr_seq[picked_idx] if (picked_idx < len(thr_seq)) else None
        if thr_here is not None:
            thr_acc_sum[fkey] = thr_acc_sum.get(fkey, 0.0) + float(thr_here)
            thr_acc_cnt[fkey] = thr_acc_cnt.get(fkey, 0) + 1

        N += 1
    # produce output: probabilities and conditional means
    out = {f"a{i}": {"p_selected": 0.0, "mean_delta": 0.0, "mean_time": 0.0} for i in range(k)}
    # N = float(N_runs)
    selection_denominator = float(max(1, N))
    for key in out.keys():
        p = select_mass[key] / selection_denominator
        out[key]["p_selected"] = p
        if select_mass[key] > 0:
            out[key]["mean_delta"] = delta_sum[key] / select_mass[key]
            out[key]["mean_time"]  = time_sum[key]  / select_mass[key]

    # ===== Post-hoc refresh (retrieve-explanation_access_mode only) =====
    now = getattr(getattr(memory, "dm", None), "time", getattr(memory, "time", 0.0))
    if feature_sel_counts:
        for ch, c in feature_sel_counts.items():
            pr = min(float(max_memory_refresh_probability), float(c) / selection_denominator)
            if pr > 0.0 and hasattr(ch, "add_prob_refresh"):
                ch.add_prob_refresh(now, pr)
    if thr_key_counts:
        for thr_key, c in thr_key_counts.items():
            pr = min(float(max_memory_refresh_probability), float(c) / selection_denominator)
            if pr > 0.0:
                build_number_profile(
                    memory, key=thr_key, requested_significant_figures=displayed_significant_figures,
                    retrieval_candidate_count=retrieval_candidate_count, memory_refresh_probability=pr
                )

    # ===== Tick memory by expected time =====
    expected_time = 0
    for v in out.values():
        expected_time += float(v["p_selected"] * v["mean_time"])
    if hasattr(memory, "tick"):
        memory.tick(expected_time)

    # === SAVE DT COMBO (direction-agnostic): type="dt_change_combo" ===
    out["expected_time"] = float(expected_time)

    if memory is not None and len(out) > 0:
        mem_core = getattr(memory, "dm", memory)
        mem_time = getattr(mem_core, "time", 0.0)

        feats = [f for f, v in out.items() if f != "expected_time" and v.get("p_selected", 0.0) > 0.0]
        if feats:
            # renormalize p over kept features
            Z = sum(out[f]["p_selected"] for f in feats) or 1.0
            p_norm = {f: out[f]["p_selected"] / Z for f in feats}
            t_map  = {f: float(out[f]["mean_time"]) for f in feats}

            # thresholds (use MC-averaged thresholds if we flipped there; else fallback to literal tree)
            thr_map = {}
            for f in feats:
                if thr_acc_cnt.get(f, 0) > 0:
                    thr_map[f] = float(thr_acc_sum[f] / thr_acc_cnt[f])
                else:
                    # fallback: first node in the tree using this feature key
                    node_thr = None
                    base_key = f  # "a{i}"
                    for n in dt_exp.tree_structure:
                        if not n["is_leaf"] and n["feature"] == base_key:
                            node_thr = float(n["threshold"])
                            break
                    thr_map[f] = float(node_thr) if node_thr is not None else 0.0

            exp_time = float(sum(p_norm[f] * t_map[f] for f in feats))
            raw_mass = float(sum(out[f]["p_selected"] for f in feats))
            refresh_p = max(0.0, min(1.0, raw_mass))

            chunk_name = "dt_change_combo"
            ch = memory.get_chunk(chunk_name) if hasattr(memory, "get_chunk") else None

            if ch is None:
                slots = {
                    "type": "dt_change_combo",
                    "features": feats,
                    "p_select": {f: float(p_norm[f]) for f in feats},
                    "threshold": {f: float(thr_map[f]) for f in feats},
                    "time": {f: float(t_map[f]) for f in feats},
                    "mass": float(raw_mass),
                    "n_updates": 1,
                    "expected_time": exp_time,
                }
                ch = memory.add_chunk(chunk_name, slots, update_retrieval=False)
            else:
                # merge with mass-weighted running means across union
                s = ch.slots
                w_old = float(s.get("mass", 0.0))
                w_new = w_old + raw_mass if (w_old + raw_mass) > 0 else raw_mass + 1e-12

                old_feats = set(s.get("features", []))
                all_feats = sorted(old_feats | set(feats))

                p_old  = dict(s.get("p_select", {}))
                t_old  = dict(s.get("time", {}))
                th_old = dict(s.get("threshold", {}))

                p_new  = {f: p_norm.get(f, 0.0) for f in all_feats}
                t_new  = {f: float(t_map.get(f, t_old.get(f, 0.0))) for f in all_feats}
                th_new = {f: float(thr_map.get(f, th_old.get(f, 0.0))) for f in all_feats}

                p_upd, t_upd, th_upd = {}, {}, {}
                for f in all_feats:
                    p_upd[f]  = (p_old.get(f, 0.0) * w_old + p_new[f] * raw_mass) / w_new
                    t_upd[f]  = (t_old.get(f, 0.0) * w_old + t_new[f] * raw_mass) / w_new
                    th_upd[f] = (th_old.get(f, 0.0) * w_old + th_new[f] * raw_mass) / w_new

                Z2 = sum(p_upd.values()) or 1.0
                for f in p_upd:
                    p_upd[f] /= Z2

                s["features"]  = list(all_feats)
                s["p_select"]  = {f: float(p_upd[f]) for f in all_feats}
                s["threshold"] = {f: float(th_upd[f]) for f in all_feats}
                s["time"]      = {f: float(t_upd[f]) for f in all_feats}
                s["mass"]      = float(w_new)
                s["n_updates"] = int(s.get("n_updates", 0)) + 1
                s["expected_time"] = float(sum(p_upd[f] * t_upd[f] for f in all_feats))

            if ch is not None and hasattr(ch, "add_prob_refresh"):
                ch.add_prob_refresh(mem_time, refresh_p)

    return out


def recall_change_dt(
    feature_vector: np.ndarray,
    memory: Any,
    bounds: Dict[str, Tuple[float, float]],
    *,
    displayed_significant_figures: int = 2,
    retrieved_combo_count: int = 2,
    memory_refresh_probability: float = 1.0,
    read_seconds_per_item: Optional[float] = READ_TIME,
) -> Dict[str, Dict[str, float]]:
    """
    Returns:
      {
        "<feature_key>": {"p_selected": float, "mean_delta": float, "mean_time": float},
        ...
        "expected_time": float
      }

    Reads combo chunks saved by cf_change_path_dt with:
      type="dt_change_combo",
      fields: features, p_select, threshold, mass, expected_time  # (time in chunk is IGNORED)

    For each numeric feature 'aN', computes the minimal signed delta needed to cross the
    stored threshold from the CURRENT feature_vector, with snapping to the slider step
    and respecting (lo, hi) bounds. Categorical tests are ignored.
    Mean time per feature = retrieval_time (of combo recall) + one read_seconds_per_item.
    """

    x = np.asarray(feature_vector, dtype=float)

    # --- helpers ---
    def _base_index_from_key(f: str) -> int:
        # f is like "a3"
        return int(f[1:])

    # --- retrieve combo chunks and capture retrieval latency ---
    def _retrieve_dt_combos():
        rt = 0.0
        if hasattr(memory, "topk_retrievals_with_prob_refresh"):
            res = memory.topk_retrievals_with_prob_refresh(
                request={"type": "dt_change_combo"},
                retrieval_candidate_count=retrieved_combo_count,
                memory_refresh_probability=memory_refresh_probability,
                add_refresh=True,
            )
            # Try common fields for retrieval latency
            rt = float(res.get("retrieval_time", res.get("rt", 0.0)))
            chunks = [ch for ch, _p in res.get("top_k", []) if ch is not None]
            return chunks, rt
        # bare DM fallback (no latency info)
        chunks = [ch for ch in getattr(memory, "chunks", [])
                  if getattr(ch, "slots", {}).get("type") == "dt_change_combo"]
        return chunks, rt

    chunks, retrieval_rt = _retrieve_dt_combos()
    if not chunks:
        return {"expected_time": float(retrieval_rt + read_seconds_per_item)}

    # --- mass-weighted mixture across chunks, then renormalize across features ---
    per_feat = {}     # feat -> accumulators over (mass * p_select)
    total_mass = 0.0

    for ch in chunks:
        s = getattr(ch, "slots", {}) or {}
        feats   = list(s.get("features", []))            # ["a0","a1",...]
        p_map   = dict(s.get("p_select", {}))            # feature -> prob within chunk
        thr_map = dict(s.get("threshold", {}))           # feature -> threshold (float, numeric only)
        mass    = float(s.get("mass", 1.0))              # chunk mass
        total_mass += mass

        for f in feats:
            p_f   = float(p_map.get(f, 0.0))
            thr_f = float(thr_map.get(f, float("nan")))  # NaN for non-numeric
            w     = mass * p_f

            acc = per_feat.setdefault(f, {
                "p_wsum": 0.0, "thr_wsum": 0.0, "thr_wsum_count": 0.0
            })
            acc["p_wsum"] += w
            if math.isfinite(thr_f):
                acc["thr_wsum"]       += w * thr_f
                acc["thr_wsum_count"] += w

    if not per_feat:
        return {"expected_time": float(retrieval_rt + read_seconds_per_item)}

    # --- renormalize p across features to form a distribution ---
    Z = sum(v["p_wsum"] for v in per_feat.values()) or 1.0

    # mean_time is identical for all features under this rule:
    mean_time_const = float(retrieval_rt + read_seconds_per_item)

    out: Dict[str, Dict[str, float]] = {}
    for f, v in per_feat.items():
        p_feat = v["p_wsum"] / Z

        # Compute mean_delta from CURRENT x against the mixed threshold (numeric only)
        mean_delta = 0.0
        if v["thr_wsum_count"] > 1e-12:
            thr_mean = v["thr_wsum"] / v["thr_wsum_count"]

            col = _base_index_from_key(f)
            cur = float(x[col])

            lo, hi = bounds.get(f, bounds.get(f"a{col}", (-float("inf"), float("inf"))))
            step = slider_step((lo, hi))

            # Minimal signed move to cross the threshold with a 1-step nudge
            if cur <= thr_mean:
                target = thr_mean + step
            else:
                target = thr_mean - step
            target = snap_to_step(target, (lo, hi), step)

            # If target collapsed to cur due to bounds/snap, push one more step.
            if target == cur:
                target = snap_to_step(target + (step if cur <= thr_mean else -step), (lo, hi), step)

            mean_delta = float(target - cur)

        out[f] = {
            "p_selected": float(p_feat),
            "mean_delta": float(mean_delta),
            "mean_time":  mean_time_const,   # <-- retrieval_rt + read_seconds_per_item
        }

    # expected time under the per-feature selection distribution
    expected_time = sum(v["p_selected"] * v["mean_time"] for v in out.values())
    out["expected_time"] = float(expected_time)

    return out


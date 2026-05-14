"""
CoXAM Forward Reasoning Strategies

Cognitive algorithm functions for memory-based reasoning using ACT-R memory.
Each function stores the trained model structure as chunks, then retrieves
probabilistically using BLL activation, slot similarity, and associative strength.

Functions:
- LRCalculation: Digit-wise LR coefficient storage + Monte Carlo reconstruction
- LRHeuristic:   Probabilistic mu/var chunk storage with Bayesian online updates
- DTTraversal:   Node-by-node DT storage with stochastic path traversal
"""

import math
import random
import numpy as np
from typing import Dict, Any, Optional, Tuple, List

from src.cognitive_models.memory.utils import (
    remember_number_to_sf,
    build_number_profile,
    digits_to_value,
)


# ============================================================
# Shared cognitive utilities
# ============================================================

def round_to_sf(value: float, sf: int = 2) -> float:
    if value == 0:
        return 0.0
    sign = -1 if value < 0 else 1
    v = abs(value)
    order = int(np.floor(np.log10(v)))
    factor = 10 ** (sf - order - 1)
    return sign * (round(v * factor) / factor)


def ddm_prob_rt(
    evidence: float,
    *,
    a: float = 1.5,
    s: float = 1.0,
    Tnd: float = 0.30,
    gain: float = 1.0,
) -> Tuple[float, float, float]:
    """Evidence-based DDM; returns (p_upper, E_RT, drift)."""
    v = gain * evidence
    if abs(v) < 1e-12:
        p_up = 0.5
        E_dec = (a * a) / (s * s)
    else:
        k = (2 * a * v) / (s ** 2)
        p_up = 1.0 / (1.0 + math.exp(-k))
        E_dec = (a / v) * math.tanh((a * v) / (s ** 2))
    return p_up, Tnd + E_dec, v


def evidence_lr_divnorm(terms, mode: str = "l2", eps: float = 1e-6) -> float:
    num = float(sum(terms))
    if mode == "l1":
        denom = eps + sum(abs(t) for t in terms)
    elif mode == "l2":
        denom = eps + math.sqrt(sum(t * t for t in terms))
    elif mode == "max":
        denom = eps + max((abs(t) for t in terms), default=0.0)
    else:
        raise ValueError("mode must be 'l1'|'l2'|'max'")
    return num / denom


def _base_index_from_key(key: str) -> int:
    return int(key.split("=")[0][1:])


def sample_number_from_profile(profile: Dict[str, Any]) -> float:
    m_vals, m_probs = zip(*profile["meta"])
    meta_choice = random.choices(m_vals, weights=m_probs, k=1)[0]
    if meta_choice is None:
        return 0.0
    sign, p10 = meta_choice
    digits = []
    for opts in profile["digits"]:
        d_vals, d_probs = zip(*opts)
        d_choice = random.choices(d_vals, weights=d_probs, k=1)[0]
        if d_choice is None:
            break
        digits.append(int(d_choice))
    if not digits:
        return 0.0
    return float(digits_to_value(sign, p10, digits, len(digits)))


def get_x_used(key: str, x: np.ndarray, compute_sf: int) -> float:
    if "=" in key:
        base, cat_idx = key.split("=")
        col_idx = int(base[1:])
        return 1.0 if int(x[col_idx]) == int(cat_idx) else 0.0
    col_idx = int(key[1:])
    return float(round_to_sf(x[col_idx], compute_sf))


# ============================================================
# LR Calculation helpers
# ============================================================

def add_lr_calculation_to_memory(
    lr_exp, memory, intercept_sf: int = 2, factor_display_sf: int = 2
) -> None:
    remember_number_to_sf(memory, key="lr:intercept", value=lr_exp.intercept, max_sf=intercept_sf)
    for feat_key, coef_val in lr_exp.coefficients.items():
        remember_number_to_sf(memory, key=f"lr:coef:{feat_key}", value=coef_val, max_sf=factor_display_sf)


def _tick(memory, dt: float, total_time_box: List[float]) -> None:
    memory.tick(dt)
    total_time_box[0] += float(dt)


def LRCalculation(
    feature_vector,
    memory,
    lr_exp,
    *,
    mode: str = "retrieve",
    compute_sf: int = 2,
    T_enc: float = 2.0,
    T_op: float = 3.0,
    ddm_a: float = 1.5,
    ddm_s: float = 1.0,
    ddm_Tnd: float = 0.30,
    ddm_norm: str = "l2",
    active_indices: Optional[List[int]] = None,
    n_mc: int = 64,
    topk_k: int = 3,
    refresh_prob: float = 1.0,
    verbose: bool = False,
    factor_display_sf: int = 2,
) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    """Returns (probs([p0,p1]), total_time, info)."""
    x = np.asarray(feature_vector, dtype=float)
    idx_set = set(active_indices) if active_indices is not None else None
    coef_items = [
        (key, coef) for key, coef in lr_exp.coefficients.items()
        if idx_set is None or _base_index_from_key(key) in idx_set
    ]
    total_time_box = [0.0]

    if mode == "read":
        terms: List[float] = []
        ops_count = 0
        _tick(memory, T_enc, total_time_box)
        c0 = float(round_to_sf(float(lr_exp.intercept), 1))
        terms.append(c0)
        for key, coef_true in coef_items:
            _tick(memory, T_enc, total_time_box)
            c = float(round_to_sf(float(coef_true), factor_display_sf))
            _tick(memory, T_enc, total_time_box)
            x_used = get_x_used(key, x, compute_sf)
            if c != 0.0 and x_used != 0.0:
                ops_count += 1
                _tick(memory, T_op, total_time_box)
            terms.append(c * x_used)
        evidence = evidence_lr_divnorm(terms, mode=ddm_norm)
        p_up, rt_dec, v_val = ddm_prob_rt(evidence, a=ddm_a, s=ddm_s, Tnd=ddm_Tnd, gain=1.0)
        _tick(memory, rt_dec, total_time_box)
        probs = np.array([1.0 - p_up, p_up], dtype=float)
        info = {
            "mode": "read", "terms": terms, "sum": float(sum(terms)),
            "ops_count": ops_count, "evidence": evidence,
            "ddm": {"a": ddm_a, "s": ddm_s, "Tnd": ddm_Tnd, "gain": 1.0,
                    "p_up": p_up, "rt_dec": rt_dec, "v": v_val},
            "T_enc": T_enc, "T_op": T_op,
        }
        return probs, total_time_box[0], info

    inter_profile = build_number_profile(
        memory, "lr:intercept", compute_sf, k=topk_k, refresh_prob=refresh_prob, verbose=verbose
    )
    coef_profiles = {
        key: build_number_profile(
            memory, f"lr:coef:{key}", factor_display_sf, k=topk_k, refresh_prob=refresh_prob, verbose=verbose
        )
        for key, _ in coef_items
    }
    mc_probs_p1: List[float] = []
    mc_times: List[float] = []
    for _ in range(int(max(1, n_mc))):
        this_time = 0.0
        terms = []
        c0 = sample_number_from_profile(inter_profile)
        this_time += inter_profile["expected_rt"]
        terms.append(c0)
        for key, _ in coef_items:
            prof = coef_profiles[key]
            c = sample_number_from_profile(prof)
            this_time += prof["expected_rt"] + T_enc
            x_used = get_x_used(key, x, compute_sf)
            if c != 0.0 and x_used != 0.0:
                this_time += T_op
            terms.append(c * x_used)
        evidence = evidence_lr_divnorm(terms, mode=ddm_norm)
        p_up, rt_dec, _ = ddm_prob_rt(evidence, a=ddm_a, s=ddm_s, Tnd=ddm_Tnd, gain=1.0)
        this_time += rt_dec
        mc_probs_p1.append(float(p_up))
        mc_times.append(float(this_time))

    p1 = float(np.mean(mc_probs_p1)) if mc_probs_p1 else 0.5
    avg_time = float(np.mean(mc_times)) if mc_times else 0.0
    probs = np.array([1.0 - p1, p1], dtype=float)
    info = {
        "mode": "retrieve", "n_mc": int(n_mc), "topk_k": int(topk_k),
        "compute_sf": int(compute_sf), "avg_p_up": p1, "avg_time": avg_time,
        "ddm": {"a": ddm_a, "s": ddm_s, "Tnd": ddm_Tnd, "norm": ddm_norm},
    }
    memory.tick(avg_time)
    return probs, avg_time, info


def refresh_lr_calculation_in_memory(
    memory, lr_exp, *, intercept_display_sf: int = 2, factor_display_sf: int = 2,
    tick_per_refresh: float = 2, active_indices: Optional[list] = None,
) -> None:
    if memory.refresh("num:lr:intercept:meta"):
        memory.tick(tick_per_refresh)
    for pos in range(1, intercept_display_sf + 1):
        if memory.refresh(f"num:lr:intercept:d{pos}"):
            memory.tick(tick_per_refresh)
    idx_set = set(active_indices) if active_indices is not None else None
    for feat_key in lr_exp.coefficients:
        if idx_set is not None and _base_index_from_key(feat_key) not in idx_set:
            continue
        if memory.refresh(f"num:lr:coef:{feat_key}:meta"):
            memory.tick(tick_per_refresh)
        for pos in range(1, factor_display_sf + 1):
            if memory.refresh(f"num:lr:coef:{feat_key}:d{pos}"):
                memory.tick(tick_per_refresh)
    memory.tick(20)


# ============================================================
# LR Heuristic helpers
# ============================================================

def add_lr_heuristic_to_memory(lr_exp, memory, initial_var: float = 1.0) -> None:
    memory.add_chunk(
        "LR_intercept_prob",
        {"type": "intercept_prob", "mu": float(0), "var": float(initial_var)},
    )
    for key, coef in lr_exp.coefficients.items():
        mu_coef = float(np.sign(coef)) if coef != 0 else 0.0
        memory.add_chunk(
            f"LR_coef_prob_{key.replace('=', '_')}",
            {
                "type": "coef_prob",
                "feature_key": key,
                "feature_name": lr_exp._format_feature(key),
                "mu": mu_coef,
                "var": float(initial_var),
            },
        )


def _draw_from_topk_heur(ret: Dict[str, Any], rng: np.random.Generator):
    top = ret.get("top_k", [])
    p_none = float(ret.get("p_none", 0.0))
    choices = [ch for ch, _ in top] + [None]
    probs = np.array([float(p) for _, p in top] + [p_none], dtype=float)
    probs[~np.isfinite(probs)] = 0.0
    probs[probs < 0] = 0.0
    s = probs.sum()
    probs = (probs / s) if s > 0.0 else np.array([1.0] + [0.0] * (len(choices) - 1), dtype=float)
    return choices[rng.choice(len(choices), p=probs)]


def _read_value_from_key(key: str, x: np.ndarray) -> Tuple[float, bool, int]:
    if "=" in key:
        base, cat = key.split("=")
        col = int(base[1:])
        return float(1.0 if int(x[col]) == int(cat) else 0.0), False, col
    col = int(key[1:])
    return float(x[col]), True, col


def LRHeuristic(
    feature_vector, memory, lr_exp,
    *, num_samples: int = 40, K_top: int = 3, T_enc: float = 2.0,
    T_INTUITIVE_OP: float = 0.5, ddm_a: float = 1.5, ddm_s: float = 1.0,
    ddm_Tnd: float = 0.30, ddm_norm: str = "l2",
    active_indices: Optional[List[int]] = None, verbose: bool = False,
) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    """Returns (probs([p0,p1]), total_time, info)."""
    rng = np.random.default_rng()
    x = np.asarray(feature_vector, dtype=float)
    active_set = set(active_indices) if active_indices is not None else None

    r_int = memory.topk_retrievals_with_prob_refresh(
        {"type": "intercept_prob"}, k=K_top, refresh_prob=1.0, add_refresh=True
    )
    retrieval_cost = float(r_int.get("expected_rt", 0.0))

    feat_meta: List[Tuple[str, float, bool, Dict[str, Any]]] = []
    for key in lr_exp.coefficients.keys():
        if active_set and _base_index_from_key(key) not in active_set:
            continue
        r_coef = memory.topk_retrievals_with_prob_refresh(
            {"type": "coef_prob", "feature_key": key}, k=K_top, refresh_prob=1.0, add_refresh=True,
        )
        retrieval_cost += float(r_coef.get("expected_rt", 0.0))
        val, is_numeric, _ = _read_value_from_key(key, x)
        feat_meta.append((key, float(val), bool(is_numeric), r_coef))

    read_cost = T_enc * len(feat_meta)
    p1_list, rt_list, v_list = [], [], []
    for _ in range(num_samples):
        terms = []
        ch_int = _draw_from_topk_heur(r_int, rng)
        mu_i = float(ch_int.slots.get("mu", 0.0)) if ch_int else 0.0
        var_i = float(ch_int.slots.get("var", 0.01)) if ch_int else 0.01
        terms.append(rng.normal(mu_i, math.sqrt(max(var_i, 1e-12))))
        for key, val, _is_num, r_coef in feat_meta:
            ch = _draw_from_topk_heur(r_coef, rng)
            mu_c = float(ch.slots.get("mu", 0.0)) if ch else 0.0
            var_c = float(ch.slots.get("var", 0.0)) if ch else 0.0
            coef = rng.normal(mu_c, math.sqrt(max(var_c, 0.0))) if var_c > 0.0 else mu_c
            terms.append(coef * val)
        evidence = evidence_lr_divnorm(terms, mode=ddm_norm)
        p_up, rt_dec, v_val = ddm_prob_rt(evidence, a=ddm_a, s=ddm_s, Tnd=ddm_Tnd, gain=1.0)
        p1_list.append(p_up)
        rt_list.append(rt_dec)
        v_list.append(v_val)

    p1 = float(np.mean(p1_list)) if p1_list else 0.5
    probs = np.array([1.0 - p1, p1], dtype=float)
    intuitive_ops = len(feat_meta)
    computation_cost = T_INTUITIVE_OP * max(0, intuitive_ops)
    ddm_rt_mean = float(np.mean(rt_list)) if rt_list else 0.0
    total_time = retrieval_cost + read_cost + computation_cost + ddm_rt_mean
    memory.tick(total_time)

    int_top = r_int.get("top_k", [])
    info = {
        "decision": {"p1": p1, "v_ratio_mean": float(np.mean(v_list)) if v_list else 0.0},
        "timing": {
            "retrieval_rt_sum": retrieval_cost, "read_time_sum": read_cost,
            "intuitive_ops": intuitive_ops, "intuitive_op_cost": T_INTUITIVE_OP,
            "ddm_rt_mean": ddm_rt_mean, "total_time": total_time,
        },
        "chunks": {
            "intercept": {"chosen_name": int_top[0][0].chunk_id if int_top else None},
            "features": [
                {
                    "key": key, "value": float(val),
                    "chosen_name": (r_coef.get("top_k", [])[0][0].chunk_id if r_coef.get("top_k") else None),
                    "is_numeric": bool(is_numeric),
                }
                for (key, val, is_numeric, r_coef) in feat_meta
            ],
        },
    }
    return probs, total_time, info


def refresh_lr_heuristic_in_memory(
    memory, lr_exp, info: Dict[str, Any], actual: int,
    *, active_indices: Optional[List[int]] = None, w_min: float = 1e-4,
) -> None:
    y = int(actual)
    p = float(info["decision"]["p1"])
    active_set = set(active_indices) if active_indices is not None else None

    def _upd(mu, var, xj, y, p, w_min):
        var = max(float(var), 1e-12)
        w = max(p * (1.0 - p), w_min)
        lam_post = (1.0 / var) + w * (xj * xj)
        var_new = 1.0 / lam_post
        mu_new = float(mu) + var_new * xj * (y - p)
        return mu_new, var_new

    cint_name = info["chunks"]["intercept"]["chosen_name"]
    if cint_name:
        ch = memory.get_chunk(cint_name)
        if ch:
            ch.slots["mu"], ch.slots["var"] = _upd(
                ch.slots.get("mu", 0.0), ch.slots.get("var", 1.0), 1.0, y, p, w_min
            )
    for f in info["chunks"]["features"]:
        cname = f.get("chosen_name")
        if not cname:
            continue
        if active_set is not None:
            key = f.get("key", "")
            try:
                base_idx = int(key.split("=")[0][1:])
            except Exception:
                continue
            if base_idx not in active_set:
                continue
        ch = memory.get_chunk(cname)
        if not ch:
            continue
        xj = float(f["value"])
        ch.slots["mu"], ch.slots["var"] = _upd(
            ch.slots.get("mu", 0.0), ch.slots.get("var", 1.0), xj, y, p, w_min
        )
    memory.tick(20)


# ============================================================
# Decision Tree helpers
# ============================================================

def evidence_firstdiff(val: float, thr: float, sf: int = 2) -> float:
    def decompose(v: float, sf: int):
        if v == 0.0:
            return (1, 0, [0] * sf)
        sign = -1 if v < 0 else 1
        v_abs = abs(v)
        p10 = int(math.floor(math.log10(v_abs)))
        scale = 10 ** (sf - 1 - p10)
        m = int(round(v_abs * scale))
        if m >= 10 ** sf:
            m //= 10
            p10 += 1
        return sign, p10, [int(d) for d in f"{m:0{sf}d}"]

    s_v, p_v, ds_v = decompose(val, sf)
    s_t, p_t, ds_t = decompose(thr, sf)
    if s_v != s_t:
        return 10.0 if s_t > s_v else -10.0
    if p_v != p_t:
        return 10.0 if p_t > p_v else -10.0
    for dv, dt in zip(ds_v, ds_t):
        if dv != dt:
            return max(-10.0, min(10.0, float(dt - dv)))
    return 0.0


def add_dt_to_memory(memory, dt_exp, *, thresh_sf: int = 2) -> None:
    nodes = {n["node"]: n for n in dt_exp.tree_structure}

    def _walk(nid: int, depth: int = 0):
        n = nodes[nid]
        memory.add_chunk(f"Node_{nid}_type",
                         {"type": "node_type", "node": nid, "is_leaf": bool(n["is_leaf"]), "depth": depth})
        if n["is_leaf"]:
            maj = int(np.argmax(n["value"]))
            label = (dt_exp.class_labels[maj]
                     if getattr(dt_exp, "class_labels", None) and maj < len(dt_exp.class_labels)
                     else f"class {maj}")
            memory.add_chunk(f"Node_{nid}_class",
                             {"type": "class_label", "node": nid, "value": label, "depth": depth})
            return
        feat_key = n["feature"]
        memory.add_chunk(f"Node_{nid}_feature",
                         {"type": "feature", "node": nid, "feat_key": feat_key, "depth": depth})
        if "=" not in feat_key:
            thr_key = f"thr:{nid}:{feat_key}"
            memory.add_chunk(f"Node_{nid}_thr_ptr",
                             {"type": "thr_ptr", "node": nid, "feat_key": feat_key,
                              "thr_key": thr_key, "depth": depth})
            remember_number_to_sf(memory, key=thr_key, value=float(n["threshold"]), max_sf=thresh_sf)
        memory.add_chunk(f"Node_{nid}_left",
                         {"type": "child_ptr", "node": nid, "which": "left", "value": n["left"], "depth": depth})
        memory.add_chunk(f"Node_{nid}_right",
                         {"type": "child_ptr", "node": nid, "which": "right", "value": n["right"], "depth": depth})
        _walk(n["left"], depth + 1)
        _walk(n["right"], depth + 1)

    _walk(0)


def DTTraversal(
    feature_vector, memory, dt_exp,
    *, mode: str = "retrieve", compute_sf: int = 2, T_enc: float = 2.0,
    ddm_a: float = 1.5, ddm_s: float = 1.0, ddm_Tnd: float = 0.30,
    ddm_norm: str = "l2", n_mc: int = 64, topk_k: int = 3,
    refresh_prob_cap: float = 1.0, verbose: bool = False,
) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    """Returns (probs(np.array[K]), expected_time, info)."""
    rng = random.Random()
    x = np.asarray(feature_vector, float)
    nodes = {n["node"]: n for n in dt_exp.tree_structure}
    num_classes = len(next(n for n in nodes.values() if n["is_leaf"])["value"])
    UNIFORM = np.full(num_classes, 1.0 / num_classes, float)

    def _round_sf(v):
        return float(round_to_sf(float(v), compute_sf))

    def _topk_no_refresh(request: dict):
        return memory.topk_retrievals_with_prob_refresh(
            request, k=topk_k, refresh_prob=0.0, add_refresh=False, verbose=False
        )

    def _draw_topk_filtered(ret, rng_, used_chunks):
        choices, probs = [], []
        for ch, p in ret["top_k"]:
            if ch is None or ch not in used_chunks:
                choices.append(ch)
                probs.append(float(p))
        choices.append(None)
        probs.append(float(ret["p_none"]))
        tot = sum(probs)
        return rng_.choices(choices, weights=probs, k=1)[0] if tot > 0 else None

    def _pick_one(options, rng_):
        if not options:
            return None, None
        weights = [float(o["prob"]) for o in options]
        tot = sum(weights)
        if tot <= 0:
            return None, None
        idx = rng_.choices(range(len(options)), weights=weights, k=1)[0]
        return idx, options[idx]

    feature_topk: Dict[int, Any] = {}
    thrptr_topk: Dict[int, Any] = {}
    num_profiles: Dict[str, Any] = {}

    if mode == "retrieve":
        for n in dt_exp.tree_structure:
            nid = n["node"]
            if n["is_leaf"]:
                continue
            feature_topk[nid] = _topk_no_refresh({"type": "feature", "node": nid})
            if "=" not in n["feature"]:
                thrptr_topk[nid] = _topk_no_refresh(
                    {"type": "thr_ptr", "node": nid, "feat_key": n["feature"]}
                )
                for ch, _ in thrptr_topk[nid]["top_k"]:
                    if ch is None:
                        continue
                    thr_key = ch.slots.get("thr_key")
                    if thr_key and thr_key not in num_profiles:
                        num_profiles[thr_key] = build_number_profile(
                            memory, key=thr_key, sf_req=compute_sf, k=topk_k, refresh_prob=0.0, verbose=False
                        )

    probs_acc = np.zeros(num_classes, float)
    time_acc = 0.0
    read_time = retrieve_time = decision_time = 0.0
    feature_sel_counts: Dict[Any, int] = {}
    thr_key_counts: Dict[str, int] = {}
    S_runs = max(1, n_mc)

    for _ in range(S_runs):
        node_id = 0
        run_time = 0.0
        used_feature_chunks: set = set()
        used_thrptr_chunks: set = set()
        used_thr_keys: set = set()
        used_num_meta_chunks: set = set()
        used_num_digit_chunks: set = set()

        while True:
            node = nodes[node_id]
            if node["is_leaf"]:
                probs_acc[int(np.argmax(node["value"]))] += 1.0
                break

            node_feat = node["feature"]
            node_is_cat = "=" in node_feat

            if mode == "read":
                run_time += T_enc
                read_time += T_enc
                feat_key = node_feat
            else:
                retF = feature_topk[node_id]
                run_time += float(retF["expected_rt"])
                retrieve_time += float(retF["expected_rt"])
                chF = _draw_topk_filtered(retF, rng, used_feature_chunks)
                if chF is None:
                    probs_acc += UNIFORM
                    break
                used_feature_chunks.add(chF)
                feature_sel_counts[chF] = feature_sel_counts.get(chF, 0) + 1
                feat_key = chF.slots.get("feat_key", node_feat)
                if node_is_cat:
                    if "=" not in feat_key or feat_key != node_feat:
                        probs_acc += UNIFORM
                        break
                else:
                    if "=" in feat_key or feat_key.split("=")[0] != node_feat:
                        probs_acc += UNIFORM
                        break

            thr_val = None
            if not node_is_cat:
                if mode == "read":
                    run_time += T_enc
                    read_time += T_enc
                    thr_val = _round_sf(nodes[node_id]["threshold"])
                else:
                    retP = thrptr_topk.get(node_id)
                    if retP is None:
                        probs_acc += UNIFORM
                        break
                    run_time += float(retP["expected_rt"])
                    retrieve_time += float(retP["expected_rt"])
                    chP = _draw_topk_filtered(retP, rng, used_thrptr_chunks)
                    if chP is None:
                        probs_acc += UNIFORM
                        break
                    used_thrptr_chunks.add(chP)
                    thr_key = chP.slots.get("thr_key")
                    if not thr_key or thr_key not in num_profiles or thr_key in used_thr_keys:
                        probs_acc += UNIFORM
                        break
                    used_thr_keys.add(thr_key)
                    thr_key_counts[thr_key] = thr_key_counts.get(thr_key, 0) + 1

                    prof = num_profiles[thr_key]
                    meta_opts = [o for o in prof["meta_with_chunks"]
                                 if o["chunk_name"] is None or o["chunk_name"] not in used_num_meta_chunks]
                    if not meta_opts:
                        probs_acc += UNIFORM
                        break
                    _, chosen_meta = _pick_one(meta_opts, rng)
                    if chosen_meta is None or chosen_meta["value"] is None:
                        probs_acc += UNIFORM
                        break
                    if chosen_meta["chunk_name"] is not None:
                        used_num_meta_chunks.add(chosen_meta["chunk_name"])
                    sign, p10 = chosen_meta["value"]
                    digits = []
                    for pos in range(1, compute_sf + 1):
                        opts = [o for o in prof["digits_with_chunks"][pos - 1]
                                if o["chunk_name"] is None or o["chunk_name"] not in used_num_digit_chunks]
                        if not opts:
                            break
                        _, pick = _pick_one(opts, rng)
                        if pick is None or pick["value"] is None:
                            break
                        if pick["chunk_name"] is not None:
                            used_num_digit_chunks.add(pick["chunk_name"])
                        digits.append(int(pick["value"]))
                    thr_val = _round_sf(digits_to_value(sign, p10, digits, len(digits)) if digits else 0.0)
                    run_time += float(prof["expected_rt"])
                    retrieve_time += float(prof["expected_rt"])

            if node_is_cat:
                base, cat_idx = node_feat.split("=")
                run_time += T_enc
                read_time += T_enc
                go_left = int(x[int(base[1:])]) == int(cat_idx)
                rt_dec = 0.0
            else:
                att = int(node_feat[1:])
                run_time += T_enc
                read_time += T_enc
                val = float(x[att])
                e = 0 if val == thr_val else (1 if val < thr_val else -1)
                p_up, E_RT, _ = ddm_prob_rt(e, a=ddm_a, s=ddm_s, Tnd=ddm_Tnd, gain=1.0)
                go_left = rng.random() < p_up
                rt_dec = E_RT

            decision_time += rt_dec
            run_time += rt_dec
            node_id = node["left"] if go_left else node["right"]

        time_acc += run_time

    probs = probs_acc / float(S_runs)
    expected_time = float(time_acc) / float(S_runs) if S_runs > 0 else 0.0
    read_time /= float(S_runs) if S_runs > 0 else 1.0
    retrieve_time /= float(S_runs) if S_runs > 0 else 1.0
    decision_time /= float(S_runs) if S_runs > 0 else 1.0

    if mode == "retrieve" and feature_sel_counts:
        now = memory.time
        for ch, c in feature_sel_counts.items():
            pr = min(refresh_prob_cap, float(c) / float(S_runs))
            if pr > 0.0:
                ch.add_prob_refresh(now, pr)
    if mode == "retrieve" and thr_key_counts:
        for thr_key, c in thr_key_counts.items():
            pr = min(refresh_prob_cap, float(c) / float(S_runs))
            if pr > 0.0:
                build_number_profile(memory, key=thr_key, sf_req=compute_sf,
                                     k=topk_k, refresh_prob=pr, verbose=False)

    memory.tick(expected_time)

    info = {
        "mode": mode, "n_mc": int(S_runs), "compute_sf": int(compute_sf), "T_enc": T_enc,
        "ddm": {"a": ddm_a, "s": ddm_s, "Tnd": ddm_Tnd, "norm": ddm_norm},
        "refresh_counts": {
            "feature": {ch.chunk_id: int(c) for ch, c in feature_sel_counts.items()},
            "thr_key": dict(thr_key_counts),
        },
    }
    return probs, expected_time, info


def refresh_dt_path_in_memory(memory, dt_exp, feature_vector, *, thresh_sf: int = 2) -> None:
    x = np.asarray(feature_vector, float)
    nodes = {n["node"]: n for n in dt_exp.tree_structure}
    nid = 0
    while True:
        n = nodes[nid]
        if n["is_leaf"]:
            memory.refresh(f"Node_{nid}_class")
            break
        feat_key = n["feature"]
        memory.refresh(f"Node_{nid}_feature")
        if "=" in feat_key:
            base, cat_idx = feat_key.split("=")
            is_member = int(x[int(base[1:])]) == int(cat_idx)
            memory.refresh(f"Node_{nid}_{'left' if is_member else 'right'}")
            nid = n["left"] if is_member else n["right"]
        else:
            memory.refresh(f"Node_{nid}_thr_ptr")
            thr_key = f"thr:{nid}:{feat_key}"
            memory.refresh(f"num:{thr_key}:meta")
            for pos in range(1, thresh_sf + 1):
                memory.refresh(f"num:{thr_key}:d{pos}")
            go_left = float(x[int(feat_key[1:])]) <= float(n["threshold"])
            memory.refresh(f"Node_{nid}_{'left' if go_left else 'right'}")
            nid = n["left"] if go_left else n["right"]


import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .memory import CombinedMemory, build_number_profile, digits_to_value, remember_number_to_sf
from .cognitive import (
    DDM_NON_DECISION_TIME,
    MENTAL_CALCULATION_TIME,
    READ_TIME,
    drift_diffusion_decision,
    lr_evidence,
    round_to_sf,
    slider_step,
    snap_to_step,
)

def _base_index_from_key(key: str) -> int:
    base = key.split('=')[0]  # "aN" from "aN" or "aN=K"
    return int(base[1:])

def add_lr_calculation_to_memory(
    lr_exp,
    memory,
    intercept_significant_figures: int = 2,
    coefficient_significant_figures: int = 2,
):
    """
    Store intercept and each coefficient of an LR explanation into memory
    as digit-wise chunks (sign, scale, digits).
    
    Keys:
      - Intercept:   "lr:intercept"
      - Coefs:       "lr:coef:{feat_key}"
    """
    # Intercept
    remember_number_to_sf(
        memory,
        key="lr:intercept",
        value=lr_exp.intercept,
        max_sf=intercept_significant_figures
    )

    # Coefficients
    for feat_key, coef_val in lr_exp.coefficients.items():
        remember_number_to_sf(
            memory,
            key=f"lr:coef:{feat_key}",
            value=coef_val,
            max_sf=coefficient_significant_figures,
        )



# ---------------------------------------------------------------------
# MAIN: lr_calculation (single entry point, LR-only)
# ---------------------------------------------------------------------

# --- Reusable helpers ---------------------------------------------------------

def _tick_memory(memory, dt: float, total_time_box: List[float]) -> None:
    """Advance internal memory time and accumulate wall time."""
    memory.tick(dt)
    total_time_box[0] += float(dt)

def _sample_number_from_profile(profile: Dict[str, Any], rng=None) -> float:
    """
    Draw a single number from a profile produced by build_number_profile.
    Expects:
      profile["meta"]   = [((sign, p10) or None, p), ...]
      profile["digits"] = [[(digit or None, p), ...], ...]
    """
    draw_rng = rng if rng is not None else random
    m_vals, m_probs = zip(*profile["meta"])
    meta_choice = draw_rng.choices(m_vals, weights=m_probs, k=1)[0]
    if meta_choice is None:
        return 0.0
    sign, p10 = meta_choice

    digits = []
    for opts in profile["digits"]:
        d_vals, d_probs = zip(*opts)
        d_choice = draw_rng.choices(d_vals, weights=d_probs, k=1)[0]
        if d_choice is None:
            break
        digits.append(int(d_choice))
    if not digits:
        return 0.0
    return float(digits_to_value(sign, p10, digits, len(digits)))

def _feature_value_for_key(
    key: str,
    x: np.ndarray,
    displayed_significant_figures: int
) -> float:
    """Value channel: 1.0 for categorical match; rounded numeric otherwise."""
    if "=" in key:
        base, cat_idx = key.split("=")
        col_idx = int(base[1:])
        return 1.0 if int(x[col_idx]) == int(cat_idx) else 0.0
    col_idx = int(key[1:])
    return float(round_to_sf(x[col_idx], displayed_significant_figures))

# --- Main function ------------------------------------------------------------

def lr_calculation(
    feature_vector,
    memory,
    lr_exp,
    *,
    explanation_access_mode: str = "retrieve",     # "retrieve" or "read"
    displayed_significant_figures: int = 2,
    read_seconds_per_item: float = READ_TIME,
    mental_calculation_seconds: float  = MENTAL_CALCULATION_TIME,
    decision_boundary: float = 1.5,
    decision_noise: float = 1.0,
    evidence_scaling: str = "l2",
    selected_feature_indices: Optional[List[int]] = None,
    # Monte Carlo controls (retrieve explanation_access_mode)
    simulation_sample_count: int = 64,
    retrieval_candidate_count: int = 3,
    memory_refresh_probability: float = 1.0,
    coefficient_significant_figures: int = 2,
    verbose: bool = False,
    verbose_sample_limit: int = 3,
    rng=None,
) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    """
    Returns: (probs(np.array([p0,p1])), total_time, info(dict))

    Assumes the following exist in your codebase:
      - build_number_profile(...)
      - round_to_sf, digits_to_value, _base_index_from_key(...)
      - lr_evidence(...), drift_diffusion_decision(...)
    """
    x = np.asarray(feature_vector, dtype=float)
    idx_set = set(selected_feature_indices) if selected_feature_indices is not None else None

    # Filter coef list by active feature indices (if provided)
    if idx_set is not None:
        coef_items = [
            (key, coef_true)
            for key, coef_true in lr_exp.coefficients.items()
            if _base_index_from_key(key) in idx_set
        ]
    else:
        coef_items = list(lr_exp.coefficients.items())

    total_time_box = [0.0]

    if verbose:
        selected_desc = "all" if selected_feature_indices is None else sorted(idx_set)
        print("[lr_calculation] starting")
        print(f"  mode={explanation_access_mode}")
        print(f"  selected_feature_indices={selected_desc}")
        print(f"  n_coefficients_used={len(coef_items)}")
        print(
            "  params: "
            f"displayed_sf={displayed_significant_figures}, "
            f"coefficient_sf={coefficient_significant_figures}, "
            f"decision_boundary={decision_boundary}, "
            f"decision_noise={decision_noise}, "
            f"evidence_scaling={evidence_scaling}"
        )

    # ------------------------ READ (deterministic) ----------------------------
    if explanation_access_mode == "read":
        terms: List[float] = []
        calculation_rows: List[Dict[str, Any]] = []
        ops_count = 0

        # Intercept factor (read once)
        _tick_memory(memory, read_seconds_per_item, total_time_box)
        c0 = float(round_to_sf(float(lr_exp.intercept), 2)) # intercept always shown with 2 sf for simplicity (can be separate if you want)
        terms.append(c0)
        calculation_rows.append({
            "feature_key": "intercept",
            "raw_feature_value": None,
            "value_used": 1.0,
            "raw_coefficient": float(lr_exp.intercept),
            "coefficient_used": c0,
            "contribution": c0,
            "cumulative_sum": float(sum(terms)),
            "read_time_charged": float(read_seconds_per_item),
            "mental_time_charged": 0.0,
        })
        if verbose:
            print("[lr_calculation:read] intercept")
            print(
                f"  read intercept time += {read_seconds_per_item:.4g}; "
                f"raw={float(lr_exp.intercept):.6g}, rounded={c0:.6g}, "
                f"cumulative_sum={sum(terms):.6g}"
            )

        # Coefficients: read factor & value; pay mental_calculation_seconds only if both nonzero
        for key, coef_true in coef_items:
            _tick_memory(memory, read_seconds_per_item, total_time_box)  # read coef
            c = float(round_to_sf(float(coef_true), coefficient_significant_figures))

            _tick_memory(memory, read_seconds_per_item, total_time_box)  # read value
            x_used = _feature_value_for_key(key, x, displayed_significant_figures)

            mental_time_charged = 0.0
            if c != 0.0 and x_used != 0.0:
                ops_count += 1
                _tick_memory(memory, mental_calculation_seconds, total_time_box)
                mental_time_charged = float(mental_calculation_seconds)
            contribution = c * x_used
            terms.append(contribution)

            base_idx = _base_index_from_key(key)
            raw_feature_value = float(x[base_idx]) if base_idx < len(x) else None
            calculation_rows.append({
                "feature_key": key,
                "raw_feature_value": raw_feature_value,
                "value_used": float(x_used),
                "raw_coefficient": float(coef_true),
                "coefficient_used": float(c),
                "contribution": float(contribution),
                "cumulative_sum": float(sum(terms)),
                "read_time_charged": float(2 * read_seconds_per_item),
                "mental_time_charged": mental_time_charged,
            })
            if verbose:
                print(f"[lr_calculation:read] feature {key}")
                print(f"  read coefficient time += {read_seconds_per_item:.4g}")
                print(
                    f"  coefficient raw={float(coef_true):.6g}, "
                    f"rounded={c:.6g}"
                )
                print(f"  read value time += {read_seconds_per_item:.4g}")
                print(
                    f"  raw_feature_value={raw_feature_value}, "
                    f"value_used={x_used:.6g}"
                )
                if mental_time_charged:
                    print(f"  mental calculation time += {mental_time_charged:.4g}")
                print(
                    f"  contribution={c:.6g} * {x_used:.6g} = {contribution:.6g}; "
                    f"cumulative_sum={sum(terms):.6g}"
                )

        # Decision (DDM)
        evidence = lr_evidence(terms, evidence_scaling=evidence_scaling)
        p_up, rt_dec, v_val = drift_diffusion_decision(evidence, decision_boundary=decision_boundary, decision_noise=decision_noise, gain=1.0)
        _tick_memory(memory, rt_dec, total_time_box)
        if verbose:
            print("[lr_calculation:read] decision")
            print(f"  terms={terms}")
            print(f"  sum={sum(terms):.6g}")
            print(f"  evidence={evidence:.6g}")
            print(
                f"  DDM: p_up={p_up:.6g}, rt_dec={rt_dec:.6g}, "
                f"v={v_val:.6g}, time += {rt_dec:.6g}"
            )
            print(
                f"  probs=[{1.0 - p_up:.6g}, {p_up:.6g}], "
                f"total_time={total_time_box[0]:.6g}"
            )

        probs = np.array([1.0 - p_up, p_up], dtype=float)
        info = {
            "explanation_access_mode": "read",
            "terms": terms,
            "calculation_rows": calculation_rows,
            "sum": float(sum(terms)),
            "ops_count": ops_count,
            "evidence": evidence,
            "ddm": {"a": decision_boundary, "s": decision_noise, "Tnd": DDM_NON_DECISION_TIME, "gain": 1.0,
                    "p_up": p_up, "rt_dec": rt_dec, "v": v_val},
            "read_seconds_per_item": read_seconds_per_item, "mental_calculation_seconds": mental_calculation_seconds,
        }
        return probs, total_time_box[0], info

    # ----------------------- RETRIEVE (Monte Carlo) --------------------------
    if verbose:
        print("[lr_calculation:retrieve] building memory retrieval profiles")
    inter_profile = build_number_profile(
        memory, "lr:intercept", displayed_significant_figures,
        retrieval_candidate_count=retrieval_candidate_count, memory_refresh_probability=memory_refresh_probability
    )
    coef_profiles = {
        key: build_number_profile(
            memory, f"lr:coef:{key}", coefficient_significant_figures,
            retrieval_candidate_count=retrieval_candidate_count, memory_refresh_probability=memory_refresh_probability
        )
        for key, _ in coef_items
    }
    if verbose:
        print(f"  intercept expected_rt={inter_profile['expected_rt']:.6g}")
        for key, prof in coef_profiles.items():
            print(f"  {key} expected_rt={prof['expected_rt']:.6g}")

    mc_probs_p1: List[float] = []
    mc_times: List[float] = []
    sample_traces: List[Dict[str, Any]] = []
    trace_limit = int(max(0, verbose_sample_limit))

    for sample_idx in range(int(max(1, simulation_sample_count))):
        this_time = 0.0
        terms: List[float] = []
        trace_rows: List[Dict[str, Any]] = []

        # Intercept retrieval
        c0 = _sample_number_from_profile(inter_profile, rng=rng)
        this_time += inter_profile["expected_rt"]
        terms.append(c0)
        if sample_idx < trace_limit:
            trace_rows.append({
                "feature_key": "intercept",
                "sampled_coefficient": float(c0),
                "value_used": 1.0,
                "contribution": float(c0),
                "time_charged": float(inter_profile["expected_rt"]),
            })

        # Coefficients retrieval
        for key, _coef_true in coef_items:
            prof = coef_profiles[key]
            c = _sample_number_from_profile(prof, rng=rng)
            this_time += prof["expected_rt"]
            this_time += read_seconds_per_item  # read value
            x_used = _feature_value_for_key(key, x, displayed_significant_figures)
            mental_time_charged = 0.0
            if c != 0.0 and x_used != 0.0:
                this_time += mental_calculation_seconds
                mental_time_charged = float(mental_calculation_seconds)
            contribution = c * x_used
            terms.append(contribution)
            if sample_idx < trace_limit:
                trace_rows.append({
                    "feature_key": key,
                    "sampled_coefficient": float(c),
                    "value_used": float(x_used),
                    "contribution": float(contribution),
                    "time_charged": float(prof["expected_rt"] + read_seconds_per_item + mental_time_charged),
                })

        # Decision (DDM)
        evidence = lr_evidence(terms, evidence_scaling=evidence_scaling)
        p_up, rt_dec, _v = drift_diffusion_decision(evidence, decision_boundary=decision_boundary, decision_noise=decision_noise, gain=1.0)
        this_time += rt_dec
        if sample_idx < trace_limit:
            sample_trace = {
                "sample_idx": int(sample_idx),
                "terms": [float(t) for t in terms],
                "rows": trace_rows,
                "sum": float(sum(terms)),
                "evidence": float(evidence),
                "p_up": float(p_up),
                "rt_dec": float(rt_dec),
                "total_time": float(this_time),
            }
            sample_traces.append(sample_trace)
            if verbose:
                print(f"[lr_calculation:retrieve] sample {sample_idx}")
                for row in trace_rows:
                    print(
                        f"  {row['feature_key']}: "
                        f"{row['sampled_coefficient']:.6g} * {row['value_used']:.6g} "
                        f"= {row['contribution']:.6g}; "
                        f"time += {row['time_charged']:.6g}"
                    )
                print(
                    f"  sum={sample_trace['sum']:.6g}, "
                    f"evidence={sample_trace['evidence']:.6g}, "
                    f"p_up={sample_trace['p_up']:.6g}, "
                    f"rt_dec={sample_trace['rt_dec']:.6g}, "
                    f"total_time={sample_trace['total_time']:.6g}"
                )

        mc_probs_p1.append(float(p_up))
        mc_times.append(float(this_time))

    # --- after the MC loop in RETRIEVE branch ---
    p1 = float(np.mean(mc_probs_p1)) if mc_probs_p1 else 0.5
    avg_time = float(np.mean(mc_times)) if mc_times else 0.0
    probs = np.array([1.0 - p1, p1], dtype=float)

    # === RESTORED ORIGINAL info dict ===
    info = {
        "explanation_access_mode": "retrieve",
        "simulation_sample_count": int(simulation_sample_count),
        "retrieval_candidate_count": int(retrieval_candidate_count),
        "displayed_significant_figures": int(displayed_significant_figures),
        "sample_traces": sample_traces,
        "avg_p_up": p1,
        "avg_time": avg_time,
        "ddm": {"a": decision_boundary, "s": decision_noise, "Tnd": DDM_NON_DECISION_TIME, "norm": evidence_scaling},
        # "explain_rows": []  # (left empty in MC retrieve; add back if you use it later)
    }
    if verbose:
        print("[lr_calculation:retrieve] summary")
        print(f"  avg_p_up={p1:.6g}")
        print(f"  probs=[{1.0 - p1:.6g}, {p1:.6g}]")
        print(f"  avg_time={avg_time:.6g}")

    memory.tick(avg_time)
    return probs, avg_time, info

# ---------------------------------------------------------------------
# Deterministic feedback refresher
# ---------------------------------------------------------------------
def refresh_lr_calculation_in_memory(
    memory,
    lr_exp,
    *,
    intercept_significant_figures: int = 2,
    coefficient_significant_figures: int = 2,
    tick_per_refresh: float = MENTAL_CALCULATION_TIME,
    selected_feature_indices: Optional[list] = None,
):
    """
    Deterministic rehearsal after feedback:
      - Refresh META + first N digits for intercept and selected coefficients.
      - Charges time per successful refresh (tick_per_refresh).
    """
    # ---- Intercept: always included ----
    if memory.refresh("num:lr:intercept:meta"):
        memory.tick(tick_per_refresh)
    for pos in range(1, intercept_significant_figures + 1):
        if memory.refresh(f"num:lr:intercept:d{pos}"):
            memory.tick(tick_per_refresh)

    # ---- Coefficients: filter by selected_feature_indices if provided ----
    idx_set = set(selected_feature_indices) if (selected_feature_indices is not None) else None

    for feat_key in lr_exp.coefficients:
        base_idx = _base_index_from_key(feat_key)
        if idx_set is not None and base_idx not in idx_set:
            continue  # skip this coefficient

        if memory.refresh(f"num:lr:coef:{feat_key}:meta"):
            memory.tick(tick_per_refresh)
        for pos in range(1, coefficient_significant_figures + 1):
            if memory.refresh(f"num:lr:coef:{feat_key}:d{pos}"):
                memory.tick(tick_per_refresh)

    # Optional baseline tick
    memory.tick(20)

# Counterfactual method 1
# w/ XAI: choose probabilistically based on range*factor (i.e., max-min * factor)
# Divide the final contribution(which is shown to 2sf) by the factor, if figure out that cannot change the first feature by required amount,
# then depending on the effort move onto the next feature.
# Amount moved is also similar to the decision tree method.
# ----------------------------------------------------------------
def cf_lr_calculation(
    feature_vector: np.ndarray,
    lr_exp: Any,
    bounds: Dict[str, Tuple[float, float]],
    *,
    # kept for interface symmetry / timing
    displayed_significant_figures: int = 2,
    read_seconds_per_item: float = READ_TIME,
    mental_calculation_seconds: float  = MENTAL_CALCULATION_TIME,
    memory: Optional[Any] = None,
    save_counterfactual_memory: bool = True,
    selected_feature_indices: Optional[List[int]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Analytic (no-MC) counterfactual distribution:
      - z_true = intercept + Σ coef_true * value_true  (no rounding)
      - z_disp = round_to_sf(z_true, 2)
      - For each feature i:
          • numeric: Δx_i = - z_disp / coef_i (feasible if within bounds)
          • categorical: feasible; Δ=1 if flip needed else 0
          • weight_i = range_i * |coef_i|  (categorical range=1)
      - p_selected(i) = weight_i / Σ weights over feasible; if none feasible -> uniform over all.

    Returns { key: {p_selected, mean_delta, mean_time} }
    """
    x = np.asarray(feature_vector, dtype=float)

    # Filter coefficients
    if selected_feature_indices is not None:
        idx_set = set(selected_feature_indices)
        coef_items = [(key, c) for key, c in lr_exp.coefficients.items()
                      if _base_index_from_key(key) in idx_set]
    else:
        coef_items = list(lr_exp.coefficients.items())

    feature_keys = [k for k, _ in coef_items]
    out = {k: dict(p_selected=0.0, mean_delta=0.0, mean_time=0.0) for k in feature_keys}

    # ---- Build z_true (no rounding) & base read/multiply time ----
    z_true = float(lr_exp.intercept)

    for key, c_true in coef_items:
        c = float(c_true)

        if "=" in key:
            base, cat_idx = key.split("=")
            col = int(base[1:])
            x_used = 1.0 if int(x[col]) == int(cat_idx) else 0.0
        else:
            col = int(key[1:])
            x_used = float(x[col])

        z_true += c * x_used
    
    base_time_const = read_seconds_per_item * 2 # read displayed sum and value

    # Only the final sum is rounded (what the user "sees")
    z_disp = float(round_to_sf(z_true, displayed_significant_figures))

    # Precompute per-feature deterministic deltas, feasibility, weights, times
    feas_weights = {}
    for key, c_true in coef_items:
        c = float(c_true)
        extra_time = 0.0
        delta = 0.0
        feasible = True

        if "=" in key:
            base, cat_idx = key.split("=")
            col = int(base[1:])
            already = (int(x[col]) == int(cat_idx))
            delta = 0.0 if already else 1.0  # 0/1 flip indicator
            if not already:
                extra_time += mental_calculation_seconds
        else:
            # numeric
            if abs(c) < 1e-12:
                feasible = False
            else:
                col = int(key[1:])
                lo, hi = bounds.get(key, (-np.inf, np.inf))
                step = slider_step((lo, hi))
                x_target = x[col] + (- z_disp / c)

                feature_range = hi - lo
                lo_relaxed = lo - feature_range / 2 if (lo - 0 < 1e8) else lo
                hi_relaxed = hi + feature_range / 2 if (hi - 1 < 1e8) else hi

                if not (np.isfinite(x_target) and lo_relaxed <= x_target <= hi_relaxed):
                    feasible = False
                else:
                    x_target = min(max(x_target, lo), hi)
                    x_landed = snap_to_step(x_target, (lo, hi), step)
                    delta = float(x_landed - x[col])
                    if delta != 0.0:
                        extra_time += mental_calculation_seconds

        weight = abs(c) if feasible else 0.0

        out[key]["mean_delta"] = delta
        out[key]["mean_time"] = base_time_const + extra_time
        feas_weights[key] = weight

    # Normalize over feasible weights; if none -> uniform over all features
    total_w = sum(feas_weights.values())
    if total_w > 0.0:
        for k in feature_keys:
            out[k]["p_selected"] = feas_weights[k] / total_w
    else:
        # fallback: uniform distribution (mirrors the MC fallback behavior)
        u = 1.0 / max(1, len(feature_keys))
        for k in feature_keys:
            out[k]["p_selected"] = u

    # === PROBABILISTIC SAVE: create NEW chunk per feature; no updates/means ===
    # === SINGLE COMBO CHUNK PER DIRECTION (update + probabilistic refresh) ===
    if save_counterfactual_memory and memory is not None and len(out) > 0:
        mem_core = getattr(memory, "dm", memory)  # CombinedMemory or DM
        mem_time = getattr(mem_core, "time", 0.0)

        # Build per-direction slices from the current analytic distribution
        feats_inc, feats_dec = {}, {}
        for feat_key, v in out.items():
            p = float(max(0.0, min(1.0, v.get("p_selected", 0.0))))
            if p <= 0.0:
                continue
            delta = float(v.get("mean_delta", 0.0))
            tcost = float(v.get("mean_time", 0.0))

            # Categorical: no sign; include in BOTH directions
            is_categorical = ("=" in feat_key)
            if is_categorical:
                feats_inc[feat_key] = dict(p=p, delta=delta, time=tcost, feature_type="categorical")
                feats_dec[feat_key] = dict(p=p, delta=delta, time=tcost, feature_type="categorical")
            else:
                # Numeric: route by sign
                if delta > 0:
                    feats_inc[feat_key] = dict(p=p, delta=delta, time=tcost, feature_type="numeric")
                elif delta < 0:
                    feats_dec[feat_key] = dict(p=p, delta=delta, time=tcost, feature_type="numeric")
                # delta == 0 -> ignore (no directional preference)

        def _update_combo(direction: str, feats: Dict[str, Dict[str, float]]):
            if not feats:
                return
            # Raw mass = sum of raw p_selected for included features (cap at 1 for refresh prob)
            raw_mass = float(sum(f["p"] for f in feats.values()))
            refresh_p = max(0.0, min(1.0, raw_mass))

            chunk_name = f"lr_change_combo:{direction}"
            ch = memory.get_chunk(chunk_name) if hasattr(memory, "get_chunk") else None

            if ch is None:
                # Create new combo with initial stats (normalized per-direction p)
                total_p = float(sum(f["p"] for f in feats.values())) or 1.0
                p_norm = {k: f["p"] / total_p for k, f in feats.items()}
                slots = {
                    "type": "lr_change_combo",
                    "direction": direction,                   # "increase" | "decrease"
                    "features": list(feats.keys()),           # canonical order is fine
                    "p_select": p_norm,                       # dict[feat] -> prob (sums to 1 over included feats)
                    "delta":   {k: float(f["delta"]) for k, f in feats.items()},
                    "time":    {k: float(f["time"])  for k, f in feats.items()},
                    "mass":    float(total_p),                # running weight accumulator (raw mass)
                    "n_updates": 1,
                    # convenience cache for expected time of this combo
                    "expected_time": float(sum(p_norm[k] * feats[k]["time"] for k in feats)),
                }
                ch = memory.add_chunk(chunk_name, slots, update_retrieval=False)
            else:
                # Merge with weighted running means (by raw_mass): update probs, deltas, times
                s = ch.slots
                w_old = float(s.get("mass", 0.0))
                w_new = w_old + raw_mass if (w_old + raw_mass) > 0 else raw_mass + 1e-12

                # Normalize current feats within direction
                total_p = float(sum(f["p"] for f in feats.values())) or 1.0
                p_norm_new = {k: f["p"] / total_p for k, f in feats.items()}

                # Union of features: carry forward old ones (decay) and add/update new ones
                old_feats = set(s.get("features", []))
                new_feats = set(feats.keys())
                all_feats = sorted(old_feats | new_feats)

                # Make sure dicts exist
                p_old = dict(s.get("p_select", {}))
                d_old = dict(s.get("delta", {}))
                t_old = dict(s.get("time", {}))

                p_upd, d_upd, t_upd = {}, {}, {}
                for k in all_feats:
                    p_k_old = float(p_old.get(k, 0.0))
                    d_k_old = float(d_old.get(k, 0.0))
                    t_k_old = float(t_old.get(k, 0.0))

                    p_k_new = float(p_norm_new.get(k, 0.0))
                    d_k_new = float(feats.get(k, {}).get("delta", d_k_old))
                    t_k_new = float(feats.get(k, {}).get("time",  t_k_old))

                    # Weighted running means by raw mass
                    p_upd[k] = (p_k_old * w_old + p_k_new * raw_mass) / w_new
                    d_upd[k] = (d_k_old * w_old + d_k_new * raw_mass) / w_new
                    t_upd[k] = (t_k_old * w_old + t_k_new * raw_mass) / w_new

                # Renormalize p_upd across the union to keep it a distribution (if you prefer)
                norm = sum(p_upd.values()) or 1.0
                for k in p_upd:
                    p_upd[k] /= norm

                s["direction"] = direction
                s["features"] = list(all_feats)
                s["p_select"] = {k: float(p_upd[k]) for k in all_feats}
                s["delta"]    = {k: float(d_upd[k]) for k in all_feats}
                s["time"]     = {k: float(t_upd[k]) for k in all_feats}
                s["mass"]     = float(w_new)
                s["n_updates"] = int(s.get("n_updates", 0)) + 1
                s["expected_time"] = float(sum(p_upd[k] * t_upd[k] for k in all_feats))

            # Probabilistic refresh for the combo
            if ch is not None and hasattr(ch, "add_prob_refresh"):
                ch.add_prob_refresh(mem_time, refresh_p)

        # Update both directions from this pass
        _update_combo("increase", feats_inc)
        _update_combo("decrease", feats_dec)
    # --- expected time under the selection distribution (no format change) ---
    expected_time = sum(v.get("p_selected", 0.0) * v.get("mean_time", 0.0) for v in out.values())
    out["expected_time"] = float(expected_time)

    return out

def recall_change_lr(
    memory: Any,
    *,
    retrieved_combo_count: int = 6,
    memory_refresh_probability: float = 1.0,
    normalize_feature_probabilities: bool = True,   # kept for fallback path
    preferred_change_direction: Optional[str] = None,  # "increase" | "decrease" | None
    read_seconds_per_item: Optional[float] = READ_TIME,
) -> Dict[str, Dict[str, float]]:
    """
    Returns a dict keyed by feature:
      {
        "<feature_key>": {"p_selected": float, "mean_delta": float, "mean_time": float},
        ...
        "expected_time": float
      }

    Timing policy:
      mean_time for every feature = retrieval_rt (for this recall) + read_seconds_per_item.
      Any time stored in chunks is ignored.
    """
    if read_seconds_per_item is None:
        read_seconds_per_item = READ_TIME

    # Helper: renormalize a dict of probabilities (safely)
    def _renorm_prob_map(p_map: Dict[str, float]) -> Dict[str, float]:
        Z = float(sum(max(0.0, v) for v in p_map.values()))
        if Z <= 1e-12:
            # fall back to uniform over keys with nonzero entries
            keys = [k for k, v in p_map.items() if v > 0.0] or list(p_map.keys())
            if not keys:
                return {}
            uni = 1.0 / len(keys)
            return {k: (uni if k in keys else 0.0) for k in p_map.keys()}
        return {k: (max(0.0, v) / Z) for k, v in p_map.items()}

    # 0) Try combo retrieval first if direction specified
    if preferred_change_direction in ("increase", "decrease"):
        combo_request = {"type": "lr_change_combo", "direction": preferred_change_direction}
        if isinstance(memory, CombinedMemory):
            res_c = memory.topk_retrievals_with_prob_refresh(
                request=combo_request, retrieval_candidate_count=1, memory_refresh_probability=memory_refresh_probability, add_refresh=True
            )
            retrieval_rt = float(res_c.get("retrieval_time", res_c.get("rt", 0.0)))
            top_c = res_c.get("top_k", [])
            if top_c:
                ch, p = top_c[0]
                s = getattr(ch, "slots", {}) or {}
                feats = list(s.get("features", []))
                p_map = _renorm_prob_map(dict(s.get("p_select", {})))
                d_map = dict(s.get("delta", {}))

                mean_time_const = float(retrieval_rt + read_seconds_per_item)

                out: Dict[str, Dict[str, float]] = {}
                for feat in feats:
                    out[feat] = {
                        "p_selected": float(p_map.get(feat, 0.0)),
                        "mean_delta": float(d_map.get(feat, 0.0)),
                        "mean_time":  mean_time_const,  # <-- retrieval_rt + read_seconds_per_item
                    }

                expected_time = sum(v["p_selected"] * v["mean_time"] for v in out.values())
                out["expected_time"] = float(expected_time)

                return out
        else:
            # bare DM fallback: exact-name lookup (no latency info available)
            ch = memory.get_chunk(f"lr_change_combo:{preferred_change_direction}") if hasattr(memory, "get_chunk") else None
            if ch:
                s = getattr(ch, "slots", {}) or {}
                feats = list(s.get("features", []))
                p_map = _renorm_prob_map(dict(s.get("p_select", {})))
                d_map = dict(s.get("delta", {}))
                retrieval_rt = 0.0
                mean_time_const = float(retrieval_rt + read_seconds_per_item)

                out: Dict[str, Dict[str, float]] = {}
                for feat in feats:
                    out[feat] = {
                        "p_selected": float(p_map.get(feat, 0.0)),
                        "mean_delta": float(d_map.get(feat, 0.0)),
                        "mean_time":  mean_time_const,
                    }
                expected_time = sum(v["p_selected"] * v["mean_time"] for v in out.values())
                out["expected_time"] = float(expected_time)
                return out

    # 1) Fallback: per-feature retrieval/averaging
    request = {"type": "lr_change"}
    if preferred_change_direction in ("increase", "decrease"):
        request["direction"] = preferred_change_direction

    if isinstance(memory, CombinedMemory):
        res = memory.topk_retrievals_with_prob_refresh(
            request=request,
            retrieval_candidate_count=retrieved_combo_count,
            memory_refresh_probability=memory_refresh_probability,
            add_refresh=True,
        )
        top_k = res.get("top_k", [])
        p_none = float(res.get("p_none", 0.0))
        retrieval_rt = float(res.get("retrieval_time", res.get("rt", 0.0)))
    else:
        ch, act, rt = memory.retrieve(request)
        top_k = [(ch, 1.0)] if ch is not None else []
        p_none = 0.0 if ch is not None else 1.0
        retrieval_rt = float(rt)

    per_feat: Dict[str, Dict[str, float]] = {}
    per_feat_kind: Dict[str, str] = {}

    for ch, p in top_k:
        if ch is None or p <= 0.0:
            continue
        s = getattr(ch, "slots", {}) or {}
        feat = str(s.get("feature", "")) or ""
        if not feat:
            continue

        feature_type = s.get("feature_type", s.get("kind", "?"))
        delta = float(s.get("delta", 0.0))

        # Respect requested direction for numeric deltas
        if feature_type == "numeric" and preferred_change_direction in ("increase", "decrease"):
            current_sign = 1 if delta > 0 else (-1 if delta < 0 else 0)
            want_sign = 1 if preferred_change_direction == "increase" else -1
            if current_sign != 0 and current_sign != want_sign:
                delta = -delta

        acc = per_feat.setdefault(feat, {"p_sum": 0.0, "delta_wsum": 0.0})
        acc["p_sum"]      += float(p)
        acc["delta_wsum"] += float(p) * delta
        per_feat_kind.setdefault(feat, feature_type)

    if not per_feat:
        return {"expected_time": float(retrieval_rt + read_seconds_per_item)}

    total_p = sum(v["p_sum"] for v in per_feat.values())
    norm_denom = total_p if (normalize_feature_probabilities and total_p > 0.0) else 1.0

    mean_time_const = float(retrieval_rt + read_seconds_per_item)

    out: Dict[str, Dict[str, float]] = {}
    for feat, v in per_feat.items():
        p_feat = v["p_sum"]
        mean_delta = (v["delta_wsum"] / p_feat) if p_feat > 1e-12 else 0.0

        out[feat] = {
            "p_selected": (p_feat / norm_denom if norm_denom > 0 else 0.0),
            "mean_delta": float(mean_delta),
            "mean_time":  mean_time_const,  # <-- retrieval_rt + read_seconds_per_item
        }

    expected_time = sum(v["p_selected"] * v["mean_time"] for v in out.values())
    out["expected_time"] = float(expected_time)

    return out




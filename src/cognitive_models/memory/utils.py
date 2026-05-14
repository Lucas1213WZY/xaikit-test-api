"""
Utility functions for memory operations shared by both backends.
"""

import math
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime


def euclidean_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute Euclidean distance, ignoring masked ``None``/NaN dimensions."""
    arr1 = np.asarray(vec1, dtype=object)
    arr2 = np.asarray(vec2, dtype=object)

    def _is_missing(value):
        if value is None:
            return True
        try:
            return bool(np.isnan(value))
        except (TypeError, ValueError):
            return False

    valid = np.array([
        not _is_missing(v1) and not _is_missing(v2)
        for v1, v2 in zip(arr1, arr2)
    ])
    if not np.any(valid):
        return float("inf")
    return float(np.linalg.norm(arr1[valid].astype(float) - arr2[valid].astype(float)))


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors (0-1, higher is more similar)."""
    denom = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    if denom < 1e-10:
        return 0.0
    return float(1.0 - (np.dot(vec1, vec2) / denom) / 2.0)  # Convert to distance


def temporal_decay(time_since_encoding: float, decay_rate: float = 0.5) -> float:
    """
    Compute temporal decay weight.
    
    CoAX uses exponential decay: activation = 1 / (1 + decay_rate * time)
    
    Args:
        time_since_encoding: Time elapsed since memory encoding
        decay_rate: Decay coefficient
        
    Returns:
        Activation weight (0-1)
    """
    if time_since_encoding < 0:
        return 1.0
    return 1.0 / (1.0 + decay_rate * time_since_encoding)


def base_level_learning(reference_count: int, decay_param: float = 0.5) -> float:
    """
    Compute Base-Level Learning (BLL) activation for ACT-R.
    
    ACT-R BLL: B = ln(sum(t_i ^ -d))
    where t_i is time since i-th retrieval, d is decay parameter
    
    Args:
        reference_count: Number of times this chunk has been retrieved
        decay_param: Decay exponent (typically 0.5)
        
    Returns:
        Base-level learning activation
    """
    if reference_count <= 0:
        return 0.0
    # Simplified: Assume uniform spacing
    # Full implementation would track actual retrieval times
    return np.log(sum((i + 1) ** (-decay_param) for i in range(reference_count)))


def compute_similarity_activation(mismatch_penalty: float = 1.5) -> float:
    """
    Compute similarity-based activation component for ACT-R.
    
    S_i_c = sum(w_k * S_{i_k,c_k})
    where w_k is slot weight, S is similarity
    
    Args:
        mismatch_penalty: Penalty factor for mismatches (typically 1.5)
        
    Returns:
        Similarity activation component (can be used in full activation)
    """
    # Placeholder: actual implementation depends on slot-by-slot comparison
    return 0.0  # This is integrated into retrieve() for actual items


def add_activation_noise(base_activation: float, noise_sd: float = 0.0) -> float:
    """
    Add stochastic noise to activation (ACT-R retrieval variability).
    
    Args:
        base_activation: Base activation score
        noise_sd: Standard deviation of noise (typically 0.0 for deterministic)
        
    Returns:
        Activation with noise applied
    """
    if noise_sd > 0:
        noise = np.random.normal(0, noise_sd)
    else:
        noise = 0.0
    return base_activation + noise


def compute_retrieval_latency(activation: float, latency_factor: float = 0.0) -> float:
    """
    Compute retrieval latency from activation (ACT-R).
    
    RT = F * exp(-activation)
    where F is latency factor, typical range: 50-100ms
    
    Args:
        activation: Chunk activation
        latency_factor: Latency scaling factor (50-100ms)
        
    Returns:
        Retrieval latency in milliseconds
    """
    if latency_factor <= 0:
        return 0.0
    return latency_factor * np.exp(-activation)


def normalize_probabilities(probs: dict) -> dict:
    """Normalize a probability dictionary to sum to 1.0."""
    total = sum(probs.values())
    if total <= 0:
        return {k: 1.0 / len(probs) for k in probs}
    return {k: v / total for k, v in probs.items()}


def compute_chunk_similarity(chunk1_slots: dict, chunk2_slots: dict, 
                             slot_weights: Optional[dict] = None) -> float:
    """
    Compute similarity between two chunks based on slot comparison.
    
    Args:
        chunk1_slots: Slots from first chunk
        chunk2_slots: Slots from second chunk
        slot_weights: Optional weight for each slot
        
    Returns:
        Similarity score (0-1, higher = more similar)
    """
    if not chunk1_slots or not chunk2_slots:
        return 0.0
    
    common_slots = set(chunk1_slots.keys()) & set(chunk2_slots.keys())
    if not common_slots:
        return 0.0
    
    if slot_weights is None:
        slot_weights = {slot: 1.0 for slot in common_slots}
    
    total_weight = 0.0
    match_score = 0.0
    
    for slot in common_slots:
        weight = slot_weights.get(slot, 1.0)
        total_weight += weight
        
        val1, val2 = chunk1_slots[slot], chunk2_slots[slot]
        
        # Handle numeric values
        if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
            # Similarity decreases with difference
            similarity = 1.0 / (1.0 + abs(val1 - val2))
        # Handle array/vector values
        elif isinstance(val1, np.ndarray) and isinstance(val2, np.ndarray):
            similarity = 1.0 - cosine_similarity(val1, val2)
        # Handle categorical/string values
        else:
            similarity = 1.0 if val1 == val2 else 0.0
        
        match_score += weight * similarity
    
    return match_score / total_weight if total_weight > 0 else 0.0


def get_timestamp_diff(ts1: datetime, ts2: Optional[datetime] = None) -> float:
    """Get time difference in seconds between two timestamps."""
    if ts2 is None:
        ts2 = datetime.now()
    delta = ts2 - ts1
    return delta.total_seconds()


# ================================================================
# Number encoding / decoding for CoXAM chunk storage
# (ported from code_for_papers/old/coxam/src/memory.py)
# ================================================================

def breakdown_number_to_sf(value: float, max_sf: int) -> Tuple[int, int, List[int]]:
    """
    Decompose a number into (sign, scale10, digits) for significant-figure storage.
    value = sign * (d1.d2d3...) * 10**scale10, d1 != 0 unless value == 0.
    """
    if value == 0 or not math.isfinite(value):
        return 1, 0, [0] * max_sf
    sign = -1 if value < 0 else 1
    ax = abs(value)
    p = math.floor(math.log10(ax))
    mant = ax / (10 ** p)
    s = f"{mant:.{max_sf - 1}f}".replace(".", "")
    if len(s) > max_sf:
        s = s[:max_sf]
        p += 1
    digits = [int(c) for c in s[:max_sf]]
    return sign, p, digits


def digits_to_value(sign: int, p: int, digits: List[int], sf_req: int) -> float:
    """Reconstruct a numeric value from the first sf_req significant-figure digits."""
    if sf_req <= 0:
        return 0.0
    s = "".join(str(d) for d in digits[:sf_req])
    mant = int(s) / (10 ** (sf_req - 1))
    return sign * mant * (10 ** p)


def remember_number_to_sf(memory: Any, key: str, value: float, max_sf: int) -> List[str]:
    """
    Store a number in ACTRMemory as:
      - META chunk  num:{key}:meta   slots: {kind, key, sign, p}
      - DIGIT chunks num:{key}:d{pos} slots: {kind, key, pos, digit}

    Args:
        memory : ACTRMemory (must expose add_chunk)
        key    : logical name (e.g. "intercept", "coef_0")
        value  : float to store
        max_sf : significant figures

    Returns list of chunk names created.
    """
    sign, scale10, digits = breakdown_number_to_sf(value, max_sf)
    created: List[str] = []

    memory.add_chunk(f"num:{key}:meta", {"kind": "nummeta", "key": key, "sign": sign, "p": scale10})
    created.append(f"num:{key}:meta")

    for pos, d in enumerate(digits, start=1):
        name = f"num:{key}:d{pos}"
        memory.add_chunk(name, {"kind": "digit", "key": key, "pos": pos, "digit": d})
        created.append(name)

    return created


def build_number_profile(
    memory: Any,
    key: str,
    sf_req: int,
    *,
    k: int = 3,
    refresh_prob: float = 1.0,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Probabilistic reconstruction of a stored number from ACTRMemory.

    Chains topk_retrievals_with_prob_refresh for the META chunk and each DIGIT chunk,
    computing expected total RT and per-digit distributions.

    Returns:
        {
            "meta":               [(value_or_None, prob), ...],
            "digits":             [[(value_or_None, prob), ...], ...],
            "expected_rt":        float,
            "meta_with_chunks":   [{"value", "prob", "chunk_name"}, ...],
            "digits_with_chunks": [[{"value", "prob", "chunk_name"}, ...], ...],
        }
    """
    F = float(memory.latency_factor)
    fexp = float(memory.latency_exponent)
    theta = float(memory.retrieval_threshold)

    if verbose:
        print(f"[build_number_profile] key={key}, sf_req={sf_req}")
        print(f"  params: F={F}, fexp={fexp}, theta={theta}")

    # ---- META retrieval ----
    meta_dist = memory.topk_retrievals_with_prob_refresh(
        {"kind": "nummeta", "key": key}, k=k,
        refresh_prob=0.0, add_refresh=False, verbose=verbose,
    )
    meta_options: List[Tuple] = []
    meta_with_chunks: List[Dict] = []

    p_none_meta = float(meta_dist["p_none"])
    p_meta = 1.0 - p_none_meta
    rt_meta = float(meta_dist["expected_rt"])

    if p_none_meta > 0:
        meta_options.append((None, p_none_meta))
        meta_with_chunks.append({"value": None, "prob": p_none_meta, "chunk_name": None})

    for ch, p in meta_dist["top_k"]:
        sign = int(ch.slots.get("sign", 1))
        p10 = int(ch.slots.get("p", 0))
        meta_options.append(((sign, p10), float(p)))
        meta_with_chunks.append({"value": (sign, p10), "prob": float(p), "chunk_name": ch.chunk_id})
        if refresh_prob > 0:
            ch.add_prob_refresh(memory.time, refresh_prob * float(p))

    # ---- DIGIT chain ----
    digit_options_all: List[List[Tuple]] = []
    digits_with_chunks: List[List[Dict]] = []

    C_prev = p_meta
    expected_rt_chain = rt_meta

    for pos in range(1, sf_req + 1):
        d_dist = memory.topk_retrievals_with_prob_refresh(
            {"kind": "digit", "key": key, "pos": pos}, k=k,
            refresh_prob=0.0, add_refresh=False, verbose=verbose,
        )
        opts_legacy: List[Tuple] = []
        opts_chunks: List[Dict] = []

        p_none = float(d_dist["p_none"])
        p_hit = 1.0 - p_none
        rt_d = float(d_dist["expected_rt"])

        if p_none > 0:
            opts_legacy.append((None, p_none))
            opts_chunks.append({"value": None, "prob": p_none, "chunk_name": None})

        for ch, p in d_dist["top_k"]:
            dval = int(ch.slots.get("digit", 0))
            opts_legacy.append((dval, float(p)))
            opts_chunks.append({"value": dval, "prob": float(p), "chunk_name": ch.chunk_id})
            if refresh_prob > 0 and C_prev > 0:
                ch.add_prob_refresh(memory.time, refresh_prob * C_prev * float(p))

        digit_options_all.append(opts_legacy)
        digits_with_chunks.append(opts_chunks)

        expected_rt_chain += C_prev * rt_d
        C_prev *= p_hit

    return {
        "meta": meta_options,
        "digits": digit_options_all,
        "expected_rt": expected_rt_chain,
        "meta_with_chunks": meta_with_chunks,
        "digits_with_chunks": digits_with_chunks,
    }


def retrieve_number_to_sf(
    memory: Any,
    key: str,
    sf_req: int,
    verbose: bool = False,
) -> Tuple[float, int, List, List, float]:
    """Convenience wrapper; returns (val, sf_got, [], [], total_rt)."""
    profile = build_number_profile(memory, key, sf_req, verbose=verbose)
    return 0.0, sf_req, [], [], profile["expected_rt"]

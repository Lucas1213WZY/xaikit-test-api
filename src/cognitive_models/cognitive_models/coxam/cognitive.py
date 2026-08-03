import math
from typing import Tuple


import numpy as np


READ_TIME = 1.0
MENTAL_CALCULATION_TIME = 0.0
DDM_NON_DECISION_TIME = 0.5


def round_to_sf(value: float, significant_figures: int = 2) -> float:
    if value == 0:
        return 0.0
    sign = -1 if value < 0 else 1
    magnitude = abs(value)
    order = int(np.floor(np.log10(magnitude)))
    factor = 10 ** (significant_figures - order - 1)
    return sign * (round(magnitude * factor) / factor)


def drift_diffusion_decision(
    evidence: float,
    *,
    decision_boundary: float = 1.5,
    decision_noise: float = 1.0,
    gain: float = 1.0,
) -> tuple[float, float, float]:
    drift = gain * evidence

    if abs(drift) < 1e-12:
        probability_upper = 0.5
        decision_time = (decision_boundary * decision_boundary) / (decision_noise * decision_noise)
    else:
        drift_ratio = (2 * decision_boundary * drift) / (decision_noise**2)
        probability_upper = 1.0 / (1.0 + math.exp(-drift_ratio))
        decision_time = (decision_boundary / drift) * math.tanh(
            (decision_boundary * drift) / (decision_noise**2)
        )

    return probability_upper, DDM_NON_DECISION_TIME + decision_time, drift


def lr_evidence(terms, evidence_scaling="l2", eps=1e-6):
    numerator = float(sum(terms))
    if evidence_scaling == "l1":
        denominator = eps + sum(abs(term) for term in terms)
    elif evidence_scaling == "l2":
        denominator = eps + math.sqrt(sum(term * term for term in terms))
    elif evidence_scaling == "max":
        denominator = eps + max((abs(term) for term in terms), default=0.0)
    else:
        raise ValueError("evidence_scaling must be 'l1'|'l2'|'max'")
    return numerator / denominator


def slider_step(bounds: Tuple[float, float]) -> float:
    low, high = bounds
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return 1.0
    span = max(high - low, 1e-12)
    return 10 ** (math.floor(math.log10(span)) - 1)


def snap_to_step(value: float, bounds: Tuple[float, float], step: float) -> float:
    low, high = bounds
    clipped = min(max(value, low), high)
    return low + round((clipped - low) / step) * step

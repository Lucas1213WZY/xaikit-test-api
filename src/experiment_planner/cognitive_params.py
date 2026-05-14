"""
CoXAM Cognitive Parameter Definitions

Defines the parameter space for CoXAM forward simulation including
default values, ranges, and the memory factory builder used by headless policies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import numpy as np


@dataclass
class CoXAMCogParams:
    """Per-episode cognitive parameters for CoXAM forward simulation."""

    # ACT-R memory
    retrieval_threshold: float = 0.0
    latency_factor: float = 3.0
    latency_exponent: float = 1.0

    # DDM decision
    ddm_a: float = 1.5
    ddm_s: float = 1.0
    ddm_Tnd: float = 0.30
    ddm_norm: str = "l2"

    # Encoding
    T_enc: float = 2.0
    compute_sf: int = 2

    # Lapse
    lapse: float = 0.05

    def to_dict(self) -> Dict[str, Any]:
        return {
            "retrieval_threshold": self.retrieval_threshold,
            "latency_factor": self.latency_factor,
            "latency_exponent": self.latency_exponent,
            "ddm_a": self.ddm_a,
            "ddm_s": self.ddm_s,
            "ddm_Tnd": self.ddm_Tnd,
            "ddm_norm": self.ddm_norm,
            "T_enc": self.T_enc,
            "compute_sf": self.compute_sf,
            "lapse": self.lapse,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> CoXAMCogParams:
        fields = CoXAMCogParams.__dataclass_fields__
        return CoXAMCogParams(**{k: v for k, v in d.items() if k in fields})

    @staticmethod
    def from_csv_row(row) -> CoXAMCogParams:
        """Load from a pandas Series (CSV parameter row)."""
        import pandas as pd

        def _get(key, default):
            val = row.get(key, default)
            return default if (isinstance(val, float) and pd.isna(val)) else val

        return CoXAMCogParams(
            retrieval_threshold=float(_get("retrieval_threshold", 0.0)),
            latency_factor=float(_get("latency_factor", 3.0)),
            latency_exponent=float(_get("latency_exponent", 1.0)),
            ddm_a=float(_get("ddm_a", 1.5)),
            ddm_s=float(_get("ddm_s", 1.0)),
            ddm_Tnd=float(_get("ddm_Tnd", 0.30)),
            ddm_norm=str(_get("ddm_norm", "l2")),
            T_enc=float(_get("T_enc", 2.0)),
            compute_sf=int(_get("compute_sf", 2)),
            lapse=float(_get("lapse", 0.05)),
        )


# Default training-time parameter ranges used for normalization inside headless policies
DEFAULT_TRAINING_COG_PARAMS: Dict[str, Any] = {
    "ddm_a": (0.5, 3.0),
    "chi": (0.0, 0.05),
    "retrieval_threshold": (-1.0, 1.0),
    "latency_factor": (0.1, 10.0),
}


def make_memory_factory(*, decay_param: float = 0.5) -> Callable:
    """
    Build a memory_factory callable for headless policies.

    Returns:
        Callable(retrieval_threshold, latency_factor) -> UnifiedMemory
    """
    from src.cognitive_models.memory import UnifiedMemory, MemoryConfig

    def _factory(retrieval_threshold: float, latency_factor: float) -> UnifiedMemory:
        config = MemoryConfig.coxam_defaults()
        config.decay_param = decay_param
        config.retrieval_threshold = retrieval_threshold
        config.latency_factor = latency_factor
        return UnifiedMemory(config)

    return _factory

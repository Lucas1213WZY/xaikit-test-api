"""
Headless Policies for Forward Trial Simulation

These are trained RL policies that execute reasoning strategies without
requiring explicit environment interactions. They are used during inference
to predict participant behavior in forward trials.

Each policy:
1. Has a trained PPO model at a specified path
2. Takes instance + trial metadata as input
3. Returns: (probabilities, prediction_time, info_dict)
4. Manages internal memory and strategy state

Strategies are loaded from the cognitive_models API layer.
"""

from typing import Any, Dict, Optional, Tuple
import numpy as np
from stable_baselines3 import PPO

from src.cognitive_models.forward.coxam_forward_rs import (
    add_dt_to_memory, DTTraversal,
    add_lr_calculation_to_memory, LRCalculation,
    add_lr_heuristic_to_memory, LRHeuristic,
)


class HeadlessDTPolicy:
    """
    Headless wrapper for trained Decision Tree forward policy.
    
    Trained policy action: [strategy_id, ddm_a_bin]
      strategy_id in {0,1,2}  -> 0=invalid, 1="read", 2="retrieve"
      ddm_a_bin in {0..B-1}
    
    Observation (matches DTForward training env):
      [chi_norm, trial_norm, with_xai,
       count_read, count_retrieve,
       succ_read, succ_retrieve]
    
    Args:
        model_path: Path to trained PPO model (.zip)
        dt_exps: Dict of decision tree explainers {dataset_id -> explainer}
        memory_factory: Callable to create memory from (retrieval_threshold, latency_factor)
        training_cog_params: Cognitive parameters from training (for normalization)
        ddm_a_bins: Number of ddm_a bins for quantization
        forbid_read_without_xai: If True, forbid "read" mode when with_xai=False
        dt_kwargs: Additional kwargs for DTTraversal function
    """
    
    def __init__(
        self,
        *,
        model_path: str,
        dt_exps: Dict[int, Any],
        memory_factory,
        training_cog_params: Dict[str, Any],
        ddm_a_bins: int = 3,
        forbid_read_without_xai: bool = True,
        dt_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.name = "dt"
        self.model = PPO.load(model_path)
        self.dt_exps = dt_exps
        self.mem_factory = memory_factory
        self.training_cog_params = dict(training_cog_params)
        self.ddm_a_bins = int(ddm_a_bins)
        self.forbid_read_without_xai = bool(forbid_read_without_xai)
        self.dt_kwargs = dict(dt_kwargs or {"n_mc": 64, "topk_k": 3, "refresh_prob_cap": 1.0})
        
        # Extract ddm_a range from training params
        ddm_spec = self.training_cog_params.get("ddm_a", (1.0, 1.0))
        if isinstance(ddm_spec, (list, tuple)) and len(ddm_spec) == 2:
            self.ddm_a_min, self.ddm_a_max = float(ddm_spec[0]), float(ddm_spec[1])
        elif isinstance(ddm_spec, (int, float)):
            self.ddm_a_min = self.ddm_a_max = float(ddm_spec)
        else:
            self.ddm_a_min = self.ddm_a_max = 1.0
        
        # Extract chi normalization reference
        chi_spec = self.training_cog_params.get("chi", [0.0, 0.03])
        self.chi_high = float(chi_spec[1]) if isinstance(chi_spec, (list, tuple)) and len(chi_spec) == 2 else 0.03
        
        # Episode runtime state
        self.episode_cogs: Dict[str, Any] = {}
        self.memory = None
        self.with_xai_schedule: Optional[np.ndarray] = None
        self.episode_len: int = 0
        self.step_idx: int = 0
        self.strategy_counts = np.zeros(2, dtype=np.int32)  # [read, retrieve]
        self.strategy_success = np.zeros(2, dtype=np.float32)
        self.dt_exp = None
    
    def _ddm_a_from_bin(self, b: int) -> float:
        """Map bin index to ddm_a value."""
        b = int(np.clip(b, 0, self.ddm_a_bins - 1))
        if self.ddm_a_bins == 1 or self.ddm_a_min == self.ddm_a_max:
            return float(self.ddm_a_min)
        frac = (b + 0.5) / self.ddm_a_bins
        return float(self.ddm_a_min + frac * (self.ddm_a_max - self.ddm_a_min))
    
    def _build_obs(self, *, chi_value: float, with_xai: bool) -> np.ndarray:
        """Build observation matching DTForward training environment."""
        obs = np.array([
            float(chi_value / max(self.chi_high, 1e-9)),
            float(self.step_idx / max(self.episode_len, 1)),
            float(with_xai),
            float(self.strategy_counts[0]),   # read count
            float(self.strategy_counts[1]),   # retrieve count
            float(self.strategy_success[0]),  # read success rate
            float(self.strategy_success[1]),  # retrieve success rate
        ], dtype=np.float32)
        return obs
    
    def _decode_action(self, action) -> Tuple[int, int]:
        """Decode multi-discrete action to [strategy_id, ddm_a_bin]."""
        a = np.asarray(action)
        if a.ndim == 0:
            a = np.array([int(a)], dtype=int)
        if a.ndim > 1:
            a = a[0]
        a = a.astype(int).ravel()
        if a.size < 2:
            out = np.zeros(2, dtype=int)
            out[:a.size] = a
            a = out
        return int(a[0]), int(a[1])
    
    def reset(
        self,
        *,
        rng,
        with_xai_schedule: np.ndarray,
        episode_cogs: Dict[str, Any],
        dataset_id: int,
        **kwargs
    ) -> None:
        """Reset policy state for new episode."""
        self.step_idx = 0
        self.with_xai_schedule = np.asarray(with_xai_schedule, dtype=bool)
        self.episode_len = int(len(self.with_xai_schedule))
        self.episode_cogs = dict(episode_cogs)
        self.strategy_counts[:] = 0
        self.strategy_success[:] = 0.0
        
        # Build memory from cognitive parameters
        rt = float(self.episode_cogs.get("retrieval_threshold", 0.5))
        lf = float(self.episode_cogs.get("latency_factor", 3.0))
        self.memory = self.mem_factory(rt, lf)
        
        # Attach DT explainer
        self.dt_exp = self.dt_exps.get(dataset_id, None)
        if self.dt_exp is not None:
            add_dt_to_memory(self.memory, self.dt_exp)
        self.memory.tick(90)
    
    def step(
        self,
        *,
        x_raw: np.ndarray,
        y_true: int,
        with_xai: bool,
        chi_value: float,
        **kwargs
    ) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        """Execute one trial with trained policy."""
        # Build observation and get action
        obs = self._build_obs(chi_value=chi_value, with_xai=with_xai)
        action, _ = self.model.predict(obs, deterministic=True)
        strategy_id, ddm_a_bin = self._decode_action(action)
        
        # Validate action
        chosen_mode = {1: "read", 2: "retrieve"}.get(strategy_id, "invalid")
        illegal = False
        if chosen_mode == "invalid":
            illegal = True
            chosen_mode = "retrieve"
        if self.forbid_read_without_xai and (not with_xai) and (chosen_mode == "read"):
            illegal = True
            chosen_mode = "retrieve"
        
        # Get parameters
        ddm_a = self._ddm_a_from_bin(ddm_a_bin)
        T_enc = float(self.episode_cogs.get("T_enc", 2.0))
        ddm_s = float(self.episode_cogs.get("ddm_s", 1.0))
        ddm_Tnd = float(self.episode_cogs.get("ddm_Tnd", 0.30))
        ddm_norm = self.episode_cogs.get("ddm_norm", "l2")
        compute_sf = int(self.episode_cogs.get("compute_sf", 2))
        lapse = float(self.episode_cogs.get("lapse", 0.05))
        
        # Run DT forward
        if self.dt_exp is not None and self.memory is not None:
            probs_out, pred_time, aux = DTTraversal(
                x_raw, self.memory, self.dt_exp,
                mode=chosen_mode,
                compute_sf=compute_sf, T_enc=T_enc, ddm_a=ddm_a,
                ddm_s=ddm_s, ddm_Tnd=ddm_Tnd, ddm_norm=ddm_norm,
                **self.dt_kwargs,
            )
            probs = np.asarray(probs_out, dtype=np.float32)
        else:
            probs = np.array([0.5, 0.5], dtype=np.float32)
            pred_time = 0.5
            aux = {}
        
        # Update statistics
        prob_correct = float(probs[int(y_true)])
        if lapse > 0.0:
            prob_correct = (1.0 - lapse) * prob_correct + 0.5 * lapse
        
        idx = 0 if (chosen_mode == "read") else 1
        self.strategy_counts[idx] += 1
        n = int(self.strategy_counts[idx])
        old = float(self.strategy_success[idx])
        self.strategy_success[idx] = (old * (n - 1) + (1.0 if prob_correct > 0.5 else 0.0)) / max(n, 1)
        
        self.step_idx += 1
        return probs, float(pred_time), {
            "illegal_action": illegal,
            "chosen_mode": chosen_mode,
            "ddm_a": float(ddm_a),
            "ddm_a_bin": int(ddm_a_bin),
            "prob_correct": prob_correct,
        }


class HeadlessLRCalcPolicy:
    """Logistic Regression Calculation Policy (trained)."""

    def __init__(
        self,
        *,
        model_path: str,
        lr_exps: Dict[int, Any],
        memory_factory,
        training_cog_params: Dict[str, Any],
        ddm_a_bins: int = 3,
        **kwargs
    ):
        self.name = "lr_calc"
        self.model = PPO.load(model_path)
        self.lr_exps = lr_exps
        self.mem_factory = memory_factory
        self.training_cog_params = dict(training_cog_params)
        self.ddm_a_bins = int(ddm_a_bins)

        ddm_spec = self.training_cog_params.get("ddm_a", (1.0, 1.0))
        if isinstance(ddm_spec, (list, tuple)) and len(ddm_spec) == 2:
            self.ddm_a_min, self.ddm_a_max = float(ddm_spec[0]), float(ddm_spec[1])
        elif isinstance(ddm_spec, (int, float)):
            self.ddm_a_min = self.ddm_a_max = float(ddm_spec)
        else:
            self.ddm_a_min = self.ddm_a_max = 1.0

        chi_spec = self.training_cog_params.get("chi", [0.0, 0.03])
        self.chi_high = float(chi_spec[1]) if isinstance(chi_spec, (list, tuple)) and len(chi_spec) >= 2 else 0.03

        self.episode_cogs: Dict[str, Any] = {}
        self.memory = None
        self.lr_exp = None
        self.step_idx: int = 0
        self.episode_len: int = 0

    def _ddm_a_from_bin(self, b: int) -> float:
        b = int(np.clip(b, 0, self.ddm_a_bins - 1))
        if self.ddm_a_bins == 1 or self.ddm_a_min == self.ddm_a_max:
            return float(self.ddm_a_min)
        frac = (b + 0.5) / self.ddm_a_bins
        return float(self.ddm_a_min + frac * (self.ddm_a_max - self.ddm_a_min))

    def _build_obs(self, *, chi_value: float, with_xai: bool) -> np.ndarray:
        return np.array([
            float(chi_value / max(self.chi_high, 1e-9)),
            float(self.step_idx / max(self.episode_len, 1)),
            float(with_xai),
        ], dtype=np.float32)

    def reset(
        self,
        *,
        rng,
        with_xai_schedule: np.ndarray,
        episode_cogs: Dict[str, Any],
        dataset_id: int,
        **kwargs
    ) -> None:
        """Reset policy state for new episode."""
        self.step_idx = 0
        self.episode_len = int(len(np.asarray(with_xai_schedule)))
        self.episode_cogs = dict(episode_cogs)
        rt = float(self.episode_cogs.get("retrieval_threshold", 0.5))
        lf = float(self.episode_cogs.get("latency_factor", 3.0))
        self.memory = self.mem_factory(rt, lf)
        self.lr_exp = self.lr_exps.get(dataset_id, None)
        if self.lr_exp is not None:
            add_lr_calculation_to_memory(self.memory, self.lr_exp)
        self.memory.tick(90)

    def step(
        self,
        *,
        x_raw: np.ndarray,
        y_true: int,
        with_xai: bool,
        chi_value: float,
        **kwargs
    ) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        """Execute one trial with trained LR calculation policy."""
        obs = self._build_obs(chi_value=chi_value, with_xai=with_xai)
        action, _ = self.model.predict(obs, deterministic=True)
        a = np.asarray(action).ravel().astype(int)
        ddm_a_bin = int(a[0]) if a.size > 0 else 0
        ddm_a = self._ddm_a_from_bin(ddm_a_bin)

        mode = "read" if with_xai else "retrieve"
        T_enc = float(self.episode_cogs.get("T_enc", 2.0))
        ddm_s = float(self.episode_cogs.get("ddm_s", 1.0))
        ddm_Tnd = float(self.episode_cogs.get("ddm_Tnd", 0.30))
        ddm_norm = self.episode_cogs.get("ddm_norm", "l2")
        compute_sf = int(self.episode_cogs.get("compute_sf", 2))

        if self.lr_exp is not None and self.memory is not None:
            probs_out, pred_time, aux = LRCalculation(
                x_raw, self.memory, self.lr_exp,
                mode=mode, compute_sf=compute_sf, T_enc=T_enc,
                ddm_a=ddm_a, ddm_s=ddm_s, ddm_Tnd=ddm_Tnd, ddm_norm=ddm_norm,
            )
            probs = np.asarray(probs_out, dtype=np.float32)
        else:
            probs = np.array([0.5, 0.5], dtype=np.float32)
            pred_time = 0.3
            aux = {}

        self.step_idx += 1
        return probs, float(pred_time), {"mode": mode, "ddm_a": float(ddm_a), **aux}


class HeadlessLRHeurPolicy:
    """Logistic Regression Heuristic Policy (trained)."""

    def __init__(
        self,
        *,
        model_path: str,
        lr_exps: Dict[int, Any],
        memory_factory,
        training_cog_params: Dict[str, Any],
        ddm_a_bins: int = 3,
        heuristic_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        self.name = "lr_heur"
        self.model = PPO.load(model_path)
        self.lr_exps = lr_exps
        self.mem_factory = memory_factory
        self.training_cog_params = dict(training_cog_params)
        self.ddm_a_bins = int(ddm_a_bins)
        self.heuristic_kwargs = dict(heuristic_kwargs or {})

        ddm_spec = self.training_cog_params.get("ddm_a", (1.0, 1.0))
        if isinstance(ddm_spec, (list, tuple)) and len(ddm_spec) == 2:
            self.ddm_a_min, self.ddm_a_max = float(ddm_spec[0]), float(ddm_spec[1])
        elif isinstance(ddm_spec, (int, float)):
            self.ddm_a_min = self.ddm_a_max = float(ddm_spec)
        else:
            self.ddm_a_min = self.ddm_a_max = 1.0

        chi_spec = self.training_cog_params.get("chi", [0.0, 0.03])
        self.chi_high = float(chi_spec[1]) if isinstance(chi_spec, (list, tuple)) and len(chi_spec) >= 2 else 0.03

        self.episode_cogs: Dict[str, Any] = {}
        self.memory = None
        self.lr_exp = None
        self.step_idx: int = 0
        self.episode_len: int = 0

    def _ddm_a_from_bin(self, b: int) -> float:
        b = int(np.clip(b, 0, self.ddm_a_bins - 1))
        if self.ddm_a_bins == 1 or self.ddm_a_min == self.ddm_a_max:
            return float(self.ddm_a_min)
        frac = (b + 0.5) / self.ddm_a_bins
        return float(self.ddm_a_min + frac * (self.ddm_a_max - self.ddm_a_min))

    def _build_obs(self, *, chi_value: float, with_xai: bool) -> np.ndarray:
        return np.array([
            float(chi_value / max(self.chi_high, 1e-9)),
            float(self.step_idx / max(self.episode_len, 1)),
            float(with_xai),
        ], dtype=np.float32)

    def reset(
        self,
        *,
        rng,
        with_xai_schedule: np.ndarray,
        episode_cogs: Dict[str, Any],
        dataset_id: int,
        **kwargs
    ) -> None:
        """Reset policy state for new episode."""
        self.step_idx = 0
        self.episode_len = int(len(np.asarray(with_xai_schedule)))
        self.episode_cogs = dict(episode_cogs)
        rt = float(self.episode_cogs.get("retrieval_threshold", 0.5))
        lf = float(self.episode_cogs.get("latency_factor", 3.0))
        self.memory = self.mem_factory(rt, lf)
        self.lr_exp = self.lr_exps.get(dataset_id, None)
        if self.lr_exp is not None:
            add_lr_heuristic_to_memory(self.lr_exp, self.memory)
        self.memory.tick(90)

    def step(
        self,
        *,
        x_raw: np.ndarray,
        y_true: int,
        with_xai: bool,
        chi_value: float,
        **kwargs
    ) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        """Execute one trial with trained LR heuristic policy."""
        obs = self._build_obs(chi_value=chi_value, with_xai=with_xai)
        action, _ = self.model.predict(obs, deterministic=True)
        a = np.asarray(action).ravel().astype(int)
        ddm_a_bin = int(a[0]) if a.size > 0 else 0
        ddm_a = self._ddm_a_from_bin(ddm_a_bin)

        T_enc = float(self.episode_cogs.get("T_enc", 2.0))
        ddm_s = float(self.episode_cogs.get("ddm_s", 1.0))
        ddm_Tnd = float(self.episode_cogs.get("ddm_Tnd", 0.30))
        ddm_norm = self.episode_cogs.get("ddm_norm", "l2")

        if self.lr_exp is not None and self.memory is not None:
            probs_out, pred_time, aux = LRHeuristic(
                x_raw, self.memory, self.lr_exp,
                T_enc=T_enc, ddm_a=ddm_a, ddm_s=ddm_s,
                ddm_Tnd=ddm_Tnd, ddm_norm=ddm_norm,
                **self.heuristic_kwargs,
            )
            probs = np.asarray(probs_out, dtype=np.float32)
        else:
            probs = np.array([0.5, 0.5], dtype=np.float32)
            pred_time = 0.4
            aux = {}

        self.step_idx += 1
        return probs, float(pred_time), {"ddm_a": float(ddm_a), **aux}

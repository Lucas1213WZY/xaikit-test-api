"""
Forward Meta Router - Strategy Selection for Forward Reasoning

Executes forward reasoning episodes where the meta PPO model:
- Observes per-strategy statistics and trial context
- Selects which forward strategy to apply for each trial
- Receives reward based on strategy correctness and execution time

Strategies are loaded from the cognitive_models API layer.
"""

from typing import Dict, Any, Optional, Sequence
import numpy as np
from stable_baselines3 import PPO

# Import from cognitive_models API
from src.cognitive_models.cr_agent.headless_policies import (
    HeadlessDTPolicy,
    HeadlessLRCalcPolicy,
    HeadlessLRHeurPolicy,
)


# Strategy/Condition Constants
STRAT_DT = "dt"
STRAT_LR_CALC = "lr_calc"
STRAT_LR_HEUR = "lr_heur"
LR_FAMILY = {STRAT_LR_CALC, STRAT_LR_HEUR}

COND_DT, COND_LR, COND_DTLR = "DT", "LR", "DT+LR"
TYPE_DT, TYPE_LR = "DT", "LR"


# ============= Schedule Building =============

def build_with_xai_schedule(
    N: int,
    ratio: float,
    rng: np.random.Generator
) -> np.ndarray:
    """Build random with_xai schedule with given ratio."""
    k = int(round(N * ratio))
    flags = np.array([1] * k + [0] * (N - k), dtype=np.int32)
    rng.shuffle(flags)
    return flags.astype(bool)


def build_trial_type_schedule(
    N: int,
    condition: str,
    rng: np.random.Generator
) -> np.ndarray:
    """Build trial type schedule based on condition."""
    if condition == COND_DT:
        return np.array([TYPE_DT] * N, dtype=object)
    if condition == COND_LR:
        return np.array([TYPE_LR] * N, dtype=object)
    # DT+LR: half/half shuffled
    m = N // 2
    arr = np.array([TYPE_DT] * m + [TYPE_LR] * (N - m), dtype=object)
    rng.shuffle(arr)
    return arr


# ============= Encoding Functions =============

def onehot_condition(condition: str) -> np.ndarray:
    """One-hot encode condition."""
    return np.array([
        1.0 if condition == COND_DT else 0.0,
        1.0 if condition == COND_LR else 0.0,
        1.0 if condition == COND_DTLR else 0.0,
    ], dtype=np.float32)


def onehot_trial_type(tt: str) -> np.ndarray:
    """One-hot encode trial type."""
    return np.array([
        1.0 if tt == TYPE_DT else 0.0,
        1.0 if tt == TYPE_LR else 0.0,
    ], dtype=np.float32)


# ============= Strategy Checking =============

def strategy_allowed_under_condition(condition: str, strat_name: str) -> bool:
    """Check if strategy is allowed under given condition."""
    if condition == COND_DT:
        return strat_name == STRAT_DT
    if condition == COND_LR:
        return strat_name in LR_FAMILY
    return True  # DT+LR allows all


def load_forward_strategies(
    *,
    dt_model_path: str,
    lr_calc_model_path: str,
    lr_heur_model_path: str,
    dt_exps: Dict[int, Any],
    lr_exps: Dict[int, Any],
    memory_factory,
    training_cog_params: Dict[str, Any],
    ddm_a_bins: int = 3,
    dt_kwargs: Optional[Dict[str, Any]] = None,
    heuristic_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Load headless forward reasoning policies.

    Args:
        dt_model_path: Path to trained PPO model for DT strategy
        lr_calc_model_path: Path to trained PPO model for LR Calculation strategy
        lr_heur_model_path: Path to trained PPO model for LR Heuristic strategy
        dt_exps: Dict {dataset_id -> DT explainer}
        lr_exps: Dict {dataset_id -> LR explainer}
        memory_factory: Callable(retrieval_threshold, latency_factor) -> UnifiedMemory
        training_cog_params: Cognitive parameters from training (ddm_a, chi ranges)
        ddm_a_bins: Number of ddm_a bins for quantization
        dt_kwargs: Additional kwargs forwarded to DTTraversal
        heuristic_kwargs: Additional kwargs forwarded to LRHeuristic

    Returns:
        Dict mapping strategy names to HeadlessPolicy instances
    """
    return {
        STRAT_DT: HeadlessDTPolicy(
            model_path=dt_model_path,
            dt_exps=dt_exps,
            memory_factory=memory_factory,
            training_cog_params=training_cog_params,
            ddm_a_bins=ddm_a_bins,
            dt_kwargs=dt_kwargs,
        ),
        STRAT_LR_CALC: HeadlessLRCalcPolicy(
            model_path=lr_calc_model_path,
            lr_exps=lr_exps,
            memory_factory=memory_factory,
            training_cog_params=training_cog_params,
            ddm_a_bins=ddm_a_bins,
        ),
        STRAT_LR_HEUR: HeadlessLRHeurPolicy(
            model_path=lr_heur_model_path,
            lr_exps=lr_exps,
            memory_factory=memory_factory,
            training_cog_params=training_cog_params,
            ddm_a_bins=ddm_a_bins,
            heuristic_kwargs=heuristic_kwargs,
        ),
    }


# ============= Main Episode Runner =============

def run_meta_on_batch(
    *,
    meta_model: PPO,
    strategies: Dict[str, Any],
    strategy_order: Optional[Sequence[str]] = None,
    
    # Episode data
    X_raw: np.ndarray,
    y_raw: np.ndarray,
    X_norm: Optional[np.ndarray] = None,
    with_xai_schedule: Optional[np.ndarray] = None,
    with_xai_ratio: Optional[float] = None,
    trial_type_schedule: Optional[np.ndarray] = None,
    condition: str = COND_DTLR,
    perm: Optional[np.ndarray] = None,
    dataset_id: Optional[int] = None,
    
    # Cognition controls
    episode_cogs: Dict[str, Any] = None,
    training_cog_params: Dict[str, Any] = None,
    chi_value: float = 0.01,
    
    # Misc
    deterministic: bool = False,
    rng: Optional[np.random.Generator] = None,
    invalid_action_penalty: float = -1.0,
) -> Dict[str, Any]:
    """
    Run ONE forward meta episode with strategy selection.
    
    Observation structure (matching training):
        [chi_norm, trial_norm, with_xai, cond_onehot(3), trial_type_onehot(2),
         per-strategy stats (4*S): for each strategy →
           [count_with/N, mean_with, count_without/N, mean_without]]
    
    The meta PPO model:
    - Observes context (chi, trial progress, xai availability, condition, stats)
    - Predicts action (strategy index)
    - Selected strategy executes without random override
    
    Args:
        meta_model: Trained PPO for strategy selection
        strategies: Dict of strategy objects {name -> strategy}
        strategy_order: Fixed ordering of strategies for action indices
        X_raw: (N, F) raw instance features
        y_raw: (N,) ground truth labels
        X_norm: (N, F) normalized features (optional)
        with_xai_schedule: (N,) bool schedule or None
        with_xai_ratio: Ratio if no schedule provided
        trial_type_schedule: (N,) trial types or None
        condition: "DT", "LR", or "DT+LR"
        episode_cogs: Per-trial cognitive parameters
        training_cog_params: Training-time parameters (for chi normalization)
        chi_value: Time cost coefficient
        deterministic: Use deterministic policy selection
        rng: Random generator
        invalid_action_penalty: Penalty for invalid strategy choices
    
    Returns:
        Dict with keys:
            - total_reward: Summed rewards across episode
            - mean_reward: Average per-trial reward
            - logs: Detailed trial-by-trial logs
            - meta: Episode metadata
    """
    assert X_raw.ndim == 2, "X_raw must be (N, F)"
    N = X_raw.shape[0]
    if X_norm is None:
        X_norm = X_raw
    assert X_norm.shape[0] == N
    assert y_raw.shape[0] == N
    assert condition in {COND_DT, COND_LR, COND_DTLR}, f"Unknown condition: {condition}"
    
    if rng is None:
        rng = np.random.default_rng()
    
    episode_cogs = dict(episode_cogs or {})
    training_cog_params = dict(training_cog_params or {})
    
    # Strategy ordering
    if strategy_order is None:
        strategy_order = list(strategies.keys())
    S = len(strategy_order)
    name_from_idx = {i: strategy_order[i] for i in range(S)}
    
    # With-XAI schedule
    if with_xai_schedule is None:
        ratio = float(with_xai_ratio if with_xai_ratio is not None else 0.5)
        with_xai_schedule = build_with_xai_schedule(N, ratio, rng)
    else:
        with_xai_schedule = np.asarray(with_xai_schedule, dtype=bool)
        assert len(with_xai_schedule) == N
    
    # Trial type schedule
    if trial_type_schedule is None:
        trial_type_schedule = build_trial_type_schedule(N, condition, rng)
    else:
        trial_type_schedule = np.asarray(trial_type_schedule, dtype=object)
        assert len(trial_type_schedule) == N
        assert set(trial_type_schedule.tolist()) <= {TYPE_DT, TYPE_LR}
    
    # Feature permutation
    F = X_raw.shape[1]
    if perm is None:
        perm = np.arange(F, dtype=np.int64)
    perm = np.asarray(perm, dtype=np.int64)
    assert perm.shape[0] == F
    inv_perm = np.empty_like(perm)
    inv_perm[perm] = np.arange(F, dtype=np.int64)
    
    # Chi normalization
    chi_spec = training_cog_params.get("chi", [0.0, 0.03])
    if isinstance(chi_spec, (list, tuple)) and len(chi_spec) == 2:
        chi_high = float(chi_spec[1])
    elif isinstance(chi_spec, (int, float)):
        chi_high = float(chi_spec)
    else:
        chi_high = 0.03
    chi_high = max(chi_high, 1e-9)
    chi_norm = float(chi_value / chi_high)
    
    # Reset strategies
    shared_reset = dict(
        rng=rng,
        with_xai_schedule=with_xai_schedule,
        perm=perm,
        inv_perm=inv_perm,
        episode_cogs=dict(episode_cogs),
        dataset_id=(dataset_id if dataset_id is not None else 1),
    )
    for sname in strategy_order:
        strategies[sname].reset(**shared_reset)
    
    # Per-episode stats
    stats = {
        name: {
            "with": {"count": 0, "sum_pr": 0.0},
            "without": {"count": 0, "sum_pr": 0.0},
        }
        for name in strategy_order
    }
    denom_N = float(max(1, N))
    
    def _stats_vector() -> np.ndarray:
        """Build per-strategy statistics vector."""
        out = []
        for name in strategy_order:
            w = stats[name]["with"]
            wo = stats[name]["without"]
            count_w = float(w["count"])
            count_wo = float(wo["count"])
            mean_w = (w["sum_pr"] / count_w) if count_w > 0 else 0.0
            mean_wo = (wo["sum_pr"] / count_wo) if count_wo > 0 else 0.0
            out.extend([
                count_w / denom_N,
                float(mean_w),
                count_wo / denom_N,
                float(mean_wo),
            ])
        return np.asarray(out, dtype=np.float32)
    
    # Run episode
    total_reward = 0.0
    logs = {
        "strategy_name": [],
        "action_idx": [],
        "with_xai_requested": [],
        "with_xai_used": [],
        "trial_type": [],
        "condition": [],
        "mismatch_applied": [],
        "invalid_under_condition": [],
        "prob_correct": [],
        "probs": [],
        "pred_time": [],
        "reward": [],
        "info": [],
    }
    
    cond_oh = onehot_condition(condition)
    
    for t in range(N):
        with_xai_req = bool(with_xai_schedule[t])
        trial_type = str(trial_type_schedule[t])
        
        # Build observation (exactly matching training env)
        obs = np.concatenate([
            np.array([chi_norm, float(t / N), float(with_xai_req)], dtype=np.float32),
            cond_oh,
            onehot_trial_type(trial_type),
            _stats_vector(),
        ]).astype(np.float32)
        
        # Get strategy from meta model (no random override)
        action, _ = meta_model.predict(obs, deterministic=deterministic)
        a = int(action) if (0 <= int(action) < S) else 0
        sname = name_from_idx[a]
        
        # Check strategy validity under condition
        if not strategy_allowed_under_condition(condition, sname):
            reward = float(invalid_action_penalty)
            total_reward += reward
            logs["strategy_name"].append(sname)
            logs["action_idx"].append(a)
            logs["with_xai_requested"].append(with_xai_req)
            logs["with_xai_used"].append(False)
            logs["trial_type"].append(trial_type)
            logs["condition"].append(condition)
            logs["mismatch_applied"].append(False)
            logs["invalid_under_condition"].append(True)
            logs["prob_correct"].append(0.0)
            logs["pred_time"].append(0.0)
            logs["probs"].append([0.0, 0.0])
            logs["reward"].append(reward)
            logs["info"].append({"invalid_under_condition": True})
            continue
        
        strat = strategies[sname]
        
        # Mismatch logic: WITH-XAI but wrong family -> run WITHOUT-XAI
        with_xai_used = with_xai_req
        mismatch = False
        if with_xai_req:
            if trial_type == TYPE_DT and sname in LR_FAMILY:
                with_xai_used = False
                mismatch = True
            elif trial_type == TYPE_LR and sname == STRAT_DT:
                with_xai_used = False
                mismatch = True
        
        # Execute strategy step from cognitive_models layer
        probs, pred_time, info = strat.step(
            x_raw=X_raw[t],
            x_norm=X_norm[t],
            y_true=int(y_raw[t]),
            with_xai=with_xai_used,
            chi_value=float(chi_value),
        )
        
        pr = float(probs[int(y_raw[t])])
        reward = pr - float(chi_value) * float(pred_time)
        total_reward += reward
        
        # Update stats
        mode_key = "with" if with_xai_used else "without"
        entry = stats[sname][mode_key]
        entry["count"] += 1
        entry["sum_pr"] += pr
        
        # Log
        logs["strategy_name"].append(sname)
        logs["action_idx"].append(a)
        logs["with_xai_requested"].append(with_xai_req)
        logs["with_xai_used"].append(with_xai_used)
        logs["trial_type"].append(trial_type)
        logs["condition"].append(condition)
        logs["mismatch_applied"].append(mismatch)
        logs["invalid_under_condition"].append(False)
        logs["prob_correct"].append(pr)
        logs["pred_time"].append(float(pred_time))
        logs["probs"].append(probs.tolist() if hasattr(probs, 'tolist') else list(probs))
        logs["reward"].append(float(reward))
        logs["info"].append(info or {})
    
    return {
        "total_reward": float(total_reward),
        "mean_reward": float(np.mean(logs["reward"])) if N > 0 else 0.0,
        "logs": logs,
        "meta": {
            "N": N,
            "chi_value": float(chi_value),
            "chi_high": float(chi_high),
            "strategy_order": list(strategy_order),
            "dataset_id": dataset_id,
            "episode_cogs": dict(episode_cogs),
            "condition": condition,
        },
    }

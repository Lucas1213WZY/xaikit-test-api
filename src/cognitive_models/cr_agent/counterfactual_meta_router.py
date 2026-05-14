"""
Counterfactual Meta Router - Strategy Selection for Counterfactual Reasoning

Gym environment that executes counterfactual reasoning episodes where:
- The meta PPO model observes per-strategy statistics and trial context
- Selects which counterfactual strategy to apply with a depth parameter
- Receives reward based on explanation success and execution time

Strategies are loaded from the cognitive_models API layer.
"""

from typing import Dict, Any, Optional, Tuple
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO

# Import from cognitive_models API
from src.cognitive_models import (
    StrategyConfig,
    StrategyType,
    ZeroOutLRHeuristic,
    ZeroOutLRDisplayed,
    ChangeDTPath,
    RecallChanges,
    MemoryBasedCF,
)


def load_counterfactual_strategies() -> Dict[str, Any]:
    """
    Load counterfactual reasoning strategies from cognitive_models API layer.
    
    Returns:
        Dict mapping strategy names to strategy instances
    """
    strategies = {
        "zero_out_lr_heuristic": ZeroOutLRHeuristic(
            StrategyConfig(
                strategy_name="zero_out_lr_heuristic",
                strategy_type=StrategyType.COXAM_COUNTERFACTUAL,
                decay_param=0.5,
                extra_params={}
            )
        ),
        "zero_out_lr_displayed": ZeroOutLRDisplayed(
            StrategyConfig(
                strategy_name="zero_out_lr_displayed",
                strategy_type=StrategyType.COXAM_COUNTERFACTUAL,
                decay_param=0.5,
                extra_params={}
            )
        ),
        "change_dt_path": ChangeDTPath(
            StrategyConfig(
                strategy_name="change_dt_path",
                strategy_type=StrategyType.COXAM_COUNTERFACTUAL,
                decay_param=0.5,
                extra_params={}
            )
        ),
        "recall_changes": RecallChanges(
            StrategyConfig(
                strategy_name="recall_changes",
                strategy_type=StrategyType.COXAM_COUNTERFACTUAL,
                decay_param=0.5,
                extra_params={}
            )
        ),
        "memory_based_cf": MemoryBasedCF(
            StrategyConfig(
                strategy_name="memory_based_cf",
                strategy_type=StrategyType.COXAM_COUNTERFACTUAL,
                decay_param=0.5,
                extra_params={}
            )
        ),
    }
    
    return strategies


class CounterfactualMetaRouter(gym.Env):
    """
    Counterfactual meta episode environment.
    
    Action space: MultiDiscrete([S, 3]) where:
      - Strategy index in [0, S)
      - Depth in {0, 1, 2}
    
    Observation: Concatenated vector of:
      - chi: Time cost coefficient
      - trial_idx: Current trial index
      - with_xai: Whether XAI is available
      - xai_type: Type of XAI (0=none, 1=dt, 2=lr, 3=both)
      - xai_type_shown: Whether XAI was shown
      - Strategy stats (3 per strategy): count, success_rate, mean_time
      - Cognitive parameters: retrieval_threshold, lapse, over_margin
    
    Reward: success - pred_time * chi
      - success: Confidence in suggested change [0, 1]
      - pred_time: Execution time of strategy
      - chi: Time cost weight
    """
    
    def __init__(
        self,
        *,
        meta_model: PPO,
        strategies: Dict[str, Any],
        X_raw: np.ndarray,
        y_raw: np.ndarray,
        instances_per_episode: int = 100,
        condition: str = "DT+LR",
        episode_cogs: Optional[Dict[str, Any]] = None,
        training_cog_params: Optional[Dict[str, Any]] = None,
        chi_spec: Tuple[float, float] = (0.0, 0.05),
        seed: Optional[int] = None,
    ):
        super().__init__()
        
        self.meta_model = meta_model
        self.strategies = dict(strategies)
        self.X_raw = np.asarray(X_raw, dtype=np.float32)
        self.y_raw = np.asarray(y_raw, dtype=np.int32)
        self.instances_per_episode = int(instances_per_episode)
        self.condition = str(condition)
        self.episode_cogs = dict(episode_cogs or {})
        self.training_cog_params = dict(training_cog_params or {})
        self.chi_spec = tuple(chi_spec)
        self.rng = np.random.default_rng(seed)
        
        # Strategy setup
        S = len(self.strategies)
        strategy_names = sorted(self.strategies.keys())
        self.strategy_idx_to_name = {i: strategy_names[i] for i in range(S)}
        
        # Action space: [strategy_idx, depth]
        self.action_space = spaces.MultiDiscrete([S, 3])
        
        # Observation space: variable-sized based on strategies
        # Base: [chi, trial_idx, with_xai, xai_type, xai_type_shown]
        # Stats: S strategies * 3 fields (count, success_rate, mean_time)
        # Cog params: 3 fields (retrieval_threshold, lapse, over_margin)
        obs_dim = 5 + (S * 3) + 3
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        # Episode state
        self.curr_chi = None
        self.episode_idx = 0
        self.counts = {name: 0 for name in strategy_names}
        self.success_rates = {name: 0.0 for name in strategy_names}
        self.mean_times = {name: 0.0 for name in strategy_names}
    
    def reset(self, seed: Optional[int] = None, **kwargs) -> Tuple[np.ndarray, Dict]:
        """Reset environment for new episode."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        
        # Reset episode state
        self.episode_idx = 0
        for name in self.counts.keys():
            self.counts[name] = 0
            self.success_rates[name] = 0.0
            self.mean_times[name] = 0.0
        
        # Sample chi value
        self.curr_chi = float(self.rng.uniform(self.chi_spec[0], self.chi_spec[1]))
        
        # Reset strategies
        with_xai_schedule = self.rng.choice([True, False], size=self.instances_per_episode)
        shared_reset = dict(
            rng=self.rng,
            with_xai_schedule=with_xai_schedule,
            perm=np.arange(self.X_raw.shape[1], dtype=np.int64),
            inv_perm=np.arange(self.X_raw.shape[1], dtype=np.int64),
            episode_cogs=dict(self.episode_cogs),
            dataset_id=1,
        )
        for strat in self.strategies.values():
            strat.reset(**shared_reset)
        
        # Return initial observation
        obs = self._build_obs()
        return obs, {}
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step (trial) in the episode."""
        a = np.asarray(action)
        strategy_idx = int(a[0]) if a.size > 0 else 0
        depth = int(a[1]) if a.size > 1 else 0
        
        # Map to strategy name
        strategy_idx = max(0, min(strategy_idx, len(self.strategy_idx_to_name) - 1))
        strategy_name = self.strategy_idx_to_name[strategy_idx]
        strategy = self.strategies[strategy_name]
        
        # Get trial data
        if self.episode_idx >= self.instances_per_episode:
            # Episode over
            obs = self._build_obs()
            return obs, 0.0, True, False, {"reached_end": True}
        
        trial_idx = self.episode_idx
        X_trial = self.X_raw[trial_idx % len(self.X_raw)]
        y_trial = self.y_raw[trial_idx % len(self.y_raw)]
        
        # Execute strategy
        try:
            change, delta, exec_time = strategy.step(
                x_raw=X_trial,
                y_true=int(y_trial),
                with_xai=True,
                chi_value=float(self.curr_chi),
            )
            
            # delta is confidence [0, 1], treat as success
            success = float(delta)
            pred_time = float(exec_time)
        except Exception:
            success = 0.0
            pred_time = 0.1
        
        # Compute reward
        reward = success - pred_time * self.curr_chi
        
        # Update statistics
        n = self.counts[strategy_name]
        old_sr = self.success_rates[strategy_name]
        self.counts[strategy_name] += 1
        self.success_rates[strategy_name] = (old_sr * n + success) / (n + 1)
        
        old_mt = self.mean_times[strategy_name]
        self.mean_times[strategy_name] = (old_mt * n + pred_time) / (n + 1)
        
        # Move to next trial
        self.episode_idx += 1
        terminated = self.episode_idx >= self.instances_per_episode
        
        # Build next observation
        obs = self._build_obs()
        
        return obs, float(reward), terminated, False, {
            "success": success,
            "pred_time": pred_time,
            "strategy": strategy_name,
            "depth": depth,
        }
    
    def _build_obs(self) -> np.ndarray:
        """Build observation vector."""
        # Normalize chi
        chi_high = max(self.chi_spec[1], 1e-9)
        chi_norm = float(self.curr_chi / chi_high)
        
        # Trial progress
        trial_norm = float(self.episode_idx / max(1, self.instances_per_episode))
        
        # XAI context (simplified)
        with_xai = 1.0  # Could vary
        xai_type = 1.0  # 0=none, 1=dt, 2=lr, 3=both
        xai_type_shown = 1.0
        
        obs = [chi_norm, trial_norm, with_xai, xai_type, xai_type_shown]
        
        # Per-strategy statistics
        for name in sorted(self.strategies.keys()):
            obs.append(float(self.counts[name]))
            obs.append(float(self.success_rates[name]))
            obs.append(float(self.mean_times[name]))
        
        # Cognitive parameters
        retrieval_threshold = float(self.episode_cogs.get("retrieval_threshold", 0.5))
        lapse = float(self.episode_cogs.get("lapse", 0.05))
        over_margin = float(self.episode_cogs.get("over_margin", 0.1))
        obs.extend([retrieval_threshold, lapse, over_margin])
        
        return np.array(obs, dtype=np.float32)

"""
CR Agent Public Interface

High-level API for running cognitive agent simulations with strategy selection.

Provides:
1. CRAgentRunner: Full simulation runner (load agents, environments, run episodes)
2. MetaRunner: Lower-level forward meta episode executor

Main entry points for users:
    runner = CRAgentRunner(meta_model_path, ...)
    results = runner.run_forward_episode(...)
    
    # For counterfactual, use CounterfactualMetaRouter directly:
    from src.cognitive_models.cr_agent import CounterfactualMetaRouter
"""

from typing import Dict, Any, List, Optional, Sequence, Tuple
import numpy as np
from pathlib import Path

from .forward_meta_router import run_meta_on_batch
from .counterfactual_meta_router import CounterfactualMetaRouter


class MetaRunner:
    """
    High-level runner for meta episodes with strategy selection.
    
    Orchestrates:
    1. Strategy initialization
    2. Episode scheduling (with_xai, trial_type)
    3. Meta model inference
    4. Per-trial strategy execution
    5. Reward computation and logging
    """
    
    def __init__(
        self,
        meta_model_path: str,
        strategies: Dict[str, Any],
        training_cog_params: Dict[str, Any],
        strategy_order: Optional[Sequence[str]] = None,
    ):
        """
        Initialize meta runner.
        
        Args:
            meta_model_path: Path to trained meta PPO model
            strategies: Dict of strategy objects {name -> strategy}
            training_cog_params: Training-time cognitive parameters
            strategy_order: Fixed strategy ordering (for action indexing)
        """
        from stable_baselines3 import PPO
        
        self.meta_model = PPO.load(meta_model_path)
        self.strategies = dict(strategies)
        self.training_cog_params = dict(training_cog_params)
        self.strategy_order = strategy_order or list(strategies.keys())
    
    def run_episode(
        self,
        *,
        X_raw: np.ndarray,
        y_raw: np.ndarray,
        X_norm: Optional[np.ndarray] = None,
        condition: str = "DT+LR",
        with_xai_ratio: float = 0.5,
        episode_cogs: Optional[Dict[str, Any]] = None,
        chi_value: float = 0.01,
        dataset_id: int = 1,
        deterministic: bool = False,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Run one meta episode.
        
        Args:
            X_raw: (N, F) raw feature matrix
            y_raw: (N,) ground truth labels
            X_norm: (N, F) normalized features (optional)
            condition: "DT", "LR", or "DT+LR"
            with_xai_ratio: Fraction of trials with XAI available
            episode_cogs: Per-episode cognitive parameters
            chi_value: Time cost coefficient
            dataset_id: Dataset identifier (for model loading)
            deterministic: Use deterministic meta strategy selection
            seed: Random seed
        
        Returns:
            Dict with episode results (total_reward, mean_reward, logs, meta)
        """
        rng = np.random.default_rng(seed)
        
        result = run_meta_on_batch(
            meta_model=self.meta_model,
            strategies=self.strategies,
            strategy_order=self.strategy_order,
            X_raw=X_raw,
            y_raw=y_raw,
            X_norm=X_norm,
            with_xai_ratio=with_xai_ratio,
            condition=condition,
            episode_cogs=episode_cogs or {},
            training_cog_params=self.training_cog_params,
            chi_value=chi_value,
            dataset_id=dataset_id,
            deterministic=deterministic,
            rng=rng,
        )
        
        return result


class CRAgentRunner:
    """
    Full CR Agent simulation runner.
    
    Loads and coordinates:
    1. Trained models and weights
    2. Cognitive strategy objects
    3. Meta routing environment
    4. Episode execution and logging
    
    Usage:
        runner = CRAgentRunner(
            meta_model_path="./weights/models_meta/best/best_model.zip",
            dt_model_path="./weights/model_dt/simple_chi_model.zip",
            ...
        )
        results = runner.run_forward_episode(X_raw, y_raw, condition='LR')
    """
    
    def __init__(
        self,
        *,
        meta_model_path: str,
        dt_model_path: Optional[str] = None,
        lr_calc_model_path: Optional[str] = None,
        lr_heur_model_path: Optional[str] = None,
        training_cog_params: Optional[Dict[str, Any]] = None,
        weights_dir: Optional[str] = None,
    ):
        """
        Initialize CR Agent runner.
        
        Args:
            meta_model_path: Path to meta strategy PPO model
            dt_model_path: Path to DT policy model (optional)
            lr_calc_model_path: Path to LR Calculation policy model (optional)
            lr_heur_model_path: Path to LR Heuristic policy model (optional)
            training_cog_params: Cognitive params from training
            weights_dir: Base directory for weight files
        """
        from stable_baselines3 import PPO
        
        self.meta_model = PPO.load(meta_model_path)
        self.training_cog_params = dict(training_cog_params or {})
        self.weights_dir = Path(weights_dir) if weights_dir else None
        
        # Store model paths for lazy loading
        self.dt_model_path = dt_model_path
        self.lr_calc_model_path = lr_calc_model_path
        self.lr_heur_model_path = lr_heur_model_path
        
        # Lazy-loaded strategies
        self._strategies = None
        self._meta_runner = None
    
    def run_forward_episode(
        self,
        *,
        X_raw: np.ndarray,
        y_raw: np.ndarray,
        X_norm: Optional[np.ndarray] = None,
        condition: str = "DT+LR",
        episode_cogs: Optional[Dict[str, Any]] = None,
        chi_value: float = 0.01,
        dataset_id: int = 1,
        deterministic: bool = False,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Run forward simulation episode.
        
        Returns per-trial predictions and strategy selections.
        """
        meta_runner = self._get_meta_runner("forward")
        
        return meta_runner.run_episode(
            X_raw=X_raw,
            y_raw=y_raw,
            X_norm=X_norm,
            condition=condition,
            with_xai_ratio=0.5,
            episode_cogs=episode_cogs,
            chi_value=chi_value,
            dataset_id=dataset_id,
            deterministic=deterministic,
            seed=seed,
        )
    
    def run_counterfactual_episode(
        self,
        *,
        X_raw: np.ndarray,
        y_raw: np.ndarray,
        instances_per_episode: int = 40,
        condition: str = "DT+LR",
        episode_cogs: Optional[Dict[str, Any]] = None,
        chi_spec: Tuple[float, float] = (0.0, 0.05),
        dataset_id: int = 1,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Run counterfactual simulation episode using gym environment interface.
        
        This runs a full Gym-based counterfactual episode with:
        1. Strategy selection via meta PPO at each trial
        2. Feature change suggestions from cognitive strategies
        3. Reward based on change effectiveness
        
        Args:
            X_raw: (N,) feature matrix for counterfactual instances
            y_raw: (N,) ground truth labels
            instances_per_episode: Number of counterfactual trials
            condition: "DT", "LR", or "DT+LR"
            episode_cogs: Per-episode cognitive parameters
            chi_spec: Tuple of (chi_low, chi_high) for time cost
            dataset_id: Dataset identifier
            seed: Random seed
            
        Returns:
            Dict with logs, rewards, and metadata from the episode
        """
        # Load counterfactual strategies
        strategies = self._load_counterfactual_strategies()
        
        # Create counterfactual router environment
        cf_router = CounterfactualMetaRouter(
            meta_model=self.meta_model,
            strategies=strategies,
            X_raw=X_raw,
            y_raw=y_raw,
            instances_per_episode=instances_per_episode,
            condition=condition,
            episode_cogs=episode_cogs or {},
            training_cog_params=self.training_cog_params,
            chi_spec=chi_spec,
            dataset_id=dataset_id,
            seed=seed,
        )
        
        # Run full episode
        obs, info = cf_router.reset(seed=seed)
        episode_log = {
            "actions": [],
            "rewards": [],
            "observations": [],
            "infos": [],
        }
        total_reward = 0.0
        
        for _ in range(instances_per_episode):
            # Get action from meta model
            action, _ = self.meta_model.predict(obs, deterministic=False)
            
            # Execute action in environment
            obs, reward, terminated, truncated, step_info = cf_router.step(action)
            
            episode_log["actions"].append(action)
            episode_log["rewards"].append(reward)
            episode_log["observations"].append(obs.copy())
            episode_log["infos"].append(step_info)
            total_reward += reward
            
            if truncated:
                break
        
        return {
            "total_reward": float(total_reward),
            "mean_reward": float(np.mean(episode_log["rewards"])) if episode_log["rewards"] else 0.0,
            "logs": episode_log,
            "meta": {
                "instances_per_episode": instances_per_episode,
                "condition": condition,
                "chi_spec": chi_spec,
                "dataset_id": dataset_id,
            },
        }
    
    def _get_meta_runner(self, sim_type: str = "forward") -> MetaRunner:
        """
        Get or create meta runner with appropriate strategies.
        
        Note: Only used for forward simulations. Counterfactual episodes use
        CounterfactualMetaRouter directly.
        """
        if sim_type == "forward":
            strategies = self._load_forward_strategies()
        else:
            raise ValueError(f"_get_meta_runner only supports 'forward' type. Use CounterfactualMetaRouter directly for counterfactual.")
        
        return MetaRunner(
            meta_model_path=self.dt_model_path or "",
            strategies=strategies,
            training_cog_params=self.training_cog_params,
        )
    
    def _load_forward_strategies(self) -> Dict[str, Any]:
        """Load forward simulation strategy policies."""
        # Placeholder - would instantiate policies from model paths
        raise NotImplementedError("Strategy loading not yet fully implemented")
    
    def _load_counterfactual_strategies(self) -> Dict[str, Any]:
        """Load counterfactual reasoning strategies from cognitive_models layer."""
        from .counterfactual_meta_router import load_counterfactual_strategies
        
        # Create wrapped counterfactual strategies from cognitive_models layer
        strategies = load_counterfactual_strategies()
        
        return strategies

from __future__ import annotations

import importlib.util
import json
import random
import time
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from .individual_policy_training_f import (
    StrategyTrainingConfig,
    _contribution_by_base_feature,
    _feature_base_index,
    _normalized_abs,
    _safe_probabilities,
    make_memory,
)
from ..dt_memory import add_dt_to_memory, dt_traverse
from ..heuristic_lr_model import add_lr_heuristic_to_memory, lr_heuristic, refresh_lr_heuristic_in_memory
from ..lr_memory import add_lr_calculation_to_memory, lr_calculation
from ..utils import AIDatasetLoader, DecisionTreeInterpreter, LogisticRegressionInterpreter, filter_by_app_and_model


META_STRATEGIES = ("lr_calculation", "lr_heuristic", "dt_traversal")
EXPLANATION_TYPES = ("none", "lr", "dt")
EXPLANATION_IDS = {name: idx for idx, name in enumerate(EXPLANATION_TYPES)}
CONDITIONS = ("linear_regression", "decision_tree", "hybrid")
CONDITION_MODES = (*CONDITIONS, "mixed")
COMPLEXITIES = ("low", "high")


@dataclass(frozen=True)
class CombinedPolicyConfig:
    data_dir: str
    output_root: str
    run_name: str | None = None
    condition_name: str = "mixed"
    total_timesteps: int = 100_000
    n_envs: int = 4
    instances_per_episode: int = 40
    max_features: int = 6
    explanation_shown_ratio: float = 0.5
    target_xai_fidelity: float | None = None
    seed: int = 123
    learning_rate: float = 3e-4
    gamma: float = 0.85
    ent_coef: float = 0.01
    n_steps: int = 512
    batch_size: int = 256
    decision_boundary_bins: int = 5
    decision_boundary_min: float = 0.6
    decision_boundary_max: float = 1.8
    decision_noise_min: float = 0.3
    decision_noise_max: float = 0.7
    memory_recall_threshold_min: float = -5.0
    memory_recall_threshold_max: float = 2.0
    opportunity_cost_min: float = 0.0
    opportunity_cost_max: float = 0.02
    memory_recall_noise: float = 0.5
    retrieval_candidate_count: int = 3
    simulation_sample_count: int = 16
    read_seconds_per_item: float = 1.0
    mental_calculation_seconds: float = 0.0
    displayed_significant_figures: int = 2
    unavailable_strategy_penalty: float = -1.0
    history_window: int = 5
    randomize_feature_order_per_episode: bool = True
    randomize_strategy_order_per_episode: bool = True
    meta_history_by_explanation: bool = True
    apps: tuple[str, ...] | None = None
    lr_calculation_model_path: str | None = None
    lr_heuristic_model_path: str | None = None
    dt_traversal_model_path: str | None = None
    allow_random_subpolicy: bool = False


@dataclass(frozen=True)
class CombinedBundle:
    app_id: str
    model_name: str
    loader: AIDatasetLoader
    lr_dense: LogisticRegressionInterpreter
    lr_sparse: LogisticRegressionInterpreter
    dt_depth2: DecisionTreeInterpreter
    dt_depth3: DecisionTreeInterpreter
    instance_ids: tuple[int, ...]


class SubStrategyState:
    def __init__(self, strategy_name: str, config: CombinedPolicyConfig):
        self.strategy_name = strategy_name
        self.memory = None
        self.current_contrib = np.zeros(config.max_features, dtype=float)
        self.contrib_history: list[np.ndarray] = []
        self.recent_prob_correct: deque[float] = deque(maxlen=config.history_window)
        self.recent_pred_time: deque[float] = deque(maxlen=config.history_window)
        self.recent_prob_correct_by_explanation = {
            explanation_type: deque(maxlen=config.history_window) for explanation_type in EXPLANATION_TYPES
        }
        self.recent_pred_time_by_explanation = {
            explanation_type: deque(maxlen=config.history_window) for explanation_type in EXPLANATION_TYPES
        }
        self.feature_order = np.arange(config.max_features, dtype=int)


def _meta_observation_dim(config: CombinedPolicyConfig) -> int:
    explanation_buckets = len(EXPLANATION_TYPES) if config.meta_history_by_explanation else 1
    history_summaries = 2 * len(META_STRATEGIES) * explanation_buckets
    selected_slot_history = len(META_STRATEGIES) * config.history_window
    return history_summaries + selected_slot_history + 10


def _normalize_scalar(value: float, low: float, high: float) -> float:
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return 0.0
    return float(np.clip((value - low) / (high - low), 0.0, 1.0))


def _base_strategy_config(config: CombinedPolicyConfig, strategy_name: str) -> StrategyTrainingConfig:
    return StrategyTrainingConfig(
        data_dir=config.data_dir,
        output_root=config.output_root,
        run_name=config.run_name,
        strategy_name=strategy_name,
        total_timesteps=config.total_timesteps,
        n_envs=config.n_envs,
        instances_per_episode=config.instances_per_episode,
        max_features=config.max_features,
        explanation_shown_ratio=config.explanation_shown_ratio,
        seed=config.seed,
        learning_rate=config.learning_rate,
        gamma=config.gamma,
        ent_coef=config.ent_coef,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        decision_boundary_bins=config.decision_boundary_bins,
        decision_boundary_min=config.decision_boundary_min,
        decision_boundary_max=config.decision_boundary_max,
        decision_noise_min=config.decision_noise_min,
        decision_noise_max=config.decision_noise_max,
        memory_recall_threshold_min=config.memory_recall_threshold_min,
        memory_recall_threshold_max=config.memory_recall_threshold_max,
        opportunity_cost_min=config.opportunity_cost_min,
        opportunity_cost_max=config.opportunity_cost_max,
        memory_recall_noise=config.memory_recall_noise,
        retrieval_candidate_count=config.retrieval_candidate_count,
        simulation_sample_count=config.simulation_sample_count,
        read_seconds_per_item=config.read_seconds_per_item,
        mental_calculation_seconds=config.mental_calculation_seconds,
        displayed_significant_figures=config.displayed_significant_figures,
        history_window=config.history_window,
        randomize_feature_order_per_episode=config.randomize_feature_order_per_episode,
        apps=config.apps,
    )


def load_combined_bundles(data_dir: Path, config: CombinedPolicyConfig) -> list[CombinedBundle]:
    values_df = pd.read_csv(data_dir / "values.csv")
    metadata_df = pd.read_csv(data_dir / "metadata.csv")
    prediction_df = pd.read_csv(data_dir / "none.csv")
    lr_df = pd.read_csv(data_dir / "logistic_regression.csv")
    dt_df = pd.read_csv(data_dir / "decision_tree.csv")

    base_loader = AIDatasetLoader(values_df, metadata_df, prediction_df)
    app_models = prediction_df[["appId", "modelName"]].drop_duplicates()
    if config.apps:
        app_models = app_models[app_models["appId"].isin(config.apps)]

    bundles: list[CombinedBundle] = []
    for row in app_models.itertuples(index=False):
        app_id = str(row.appId)
        model_name = str(row.modelName)
        try:
            loader = filter_by_app_and_model(base_loader, app_id, model_name)
            lr_dense = LogisticRegressionInterpreter(lr_df, metadata_df, app_id, model_name, variant="dense")
            lr_sparse = LogisticRegressionInterpreter(lr_df, metadata_df, app_id, model_name, variant="sparse")
            dt_depth2 = DecisionTreeInterpreter(dt_df, metadata_df, app_id, model_name, depth=2)
            dt_depth3 = DecisionTreeInterpreter(dt_df, metadata_df, app_id, model_name, depth=3)
        except Exception as exc:
            print(f"Skipping {app_id}/{model_name}: {exc}")
            continue
        feature_ids = set(loader.feature_values_df["instanceId"].dropna().astype(int).tolist())
        labeled_prediction_rows = loader.AI_predictions_df[loader.AI_predictions_df["pred"].notna()]
        prediction_ids = set(labeled_prediction_rows["instanceId"].dropna().astype(int).tolist())
        instance_ids = tuple(sorted(feature_ids & prediction_ids))
        if instance_ids:
            bundles.append(CombinedBundle(app_id, model_name, loader, lr_dense, lr_sparse, dt_depth2, dt_depth3, instance_ids))
    return bundles


def load_sub_policies(config: CombinedPolicyConfig) -> dict[str, PPO]:
    paths = {
        "lr_calculation": config.lr_calculation_model_path,
        "lr_heuristic": config.lr_heuristic_model_path,
        "dt_traversal": config.dt_traversal_model_path,
    }
    models = {}
    missing = []
    for strategy_name, path in paths.items():
        if path:
            models[strategy_name] = PPO.load(path, device="cpu")
        else:
            missing.append(strategy_name)
    if missing and not config.allow_random_subpolicy:
        raise ValueError(f"Missing pretrained sub-policy paths for: {missing}")
    return models


class CombinedStrategyPolicyEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        bundles: list[CombinedBundle],
        config: CombinedPolicyConfig,
        sub_policies: dict[str, PPO] | None = None,
        *,
        training: bool = True,
        fixed_eval_params: dict[str, float] | None = None,
    ):
        super().__init__()
        if config.condition_name not in CONDITION_MODES:
            raise ValueError(f"Unknown condition_name={config.condition_name!r}; expected one of {CONDITION_MODES}.")
        if not bundles:
            raise ValueError("CombinedStrategyPolicyEnv requires at least one bundle.")
        self.bundles = bundles
        self.config = config
        self.sub_policies = sub_policies or {}
        self.training = bool(training)
        self.fixed_eval_params = fixed_eval_params or {}
        self.action_space = spaces.Discrete(3)

        obs_dim = _meta_observation_dim(config)
        self.observation_space = spaces.Box(
            low=np.full(obs_dim, -1.0, dtype=np.float32),
            high=np.ones(obs_dim, dtype=np.float32),
            dtype=np.float32,
        )
        self.rng = np.random.default_rng(config.seed)
        self.bundle: CombinedBundle | None = None
        self.current_params: dict[str, float] = {}
        self.episode_condition = "hybrid"
        self.episode_complexity = "high"
        self.explanation_schedule: list[str] = []
        self.X_raw: np.ndarray | None = None
        self.X_norm: np.ndarray | None = None
        self.y: np.ndarray | None = None
        self.step_idx = 0
        self.states = {name: SubStrategyState(name, config) for name in META_STRATEGIES}
        self.recent_selected_strategies: deque[int] = deque(maxlen=config.history_window)
        self.strategy_slots: tuple[str, ...] = META_STRATEGIES
        self.strategy_simulation_rngs: dict[str, Any] = {}
        self.sampled_instance_ids: list[int] = []
        self.lr_explanation_matches_ai = np.array([], dtype=bool)
        self.dt_explanation_matches_ai = np.array([], dtype=bool)
        self.lr_explanation_episode_fidelity = float("nan")
        self.dt_explanation_episode_fidelity = float("nan")
        self._xai_match_pool_cache: dict[tuple[str, str, str], dict[tuple[bool, bool], list[int]]] = {}

    def _sample_param(self, key: str, low_attr: str, high_attr: str) -> float:
        if key in self.fixed_eval_params:
            return float(self.fixed_eval_params[key])
        low = float(getattr(self.config, low_attr))
        high = float(getattr(self.config, high_attr))
        if self.training:
            return float(self.rng.uniform(low, high))
        return 0.5 * (low + high)

    def _sample_episode_params(self) -> dict[str, float]:
        return {
            "decision_noise": self._sample_param("decision_noise", "decision_noise_min", "decision_noise_max"),
            "memory_recall_threshold": self._sample_param("memory_recall_threshold", "memory_recall_threshold_min", "memory_recall_threshold_max"),
            "opportunity_cost": self._sample_param("opportunity_cost", "opportunity_cost_min", "opportunity_cost_max"),
        }

    def _sample_episode_condition(self) -> str:
        if self.config.condition_name == "mixed":
            return str(self.rng.choice(CONDITIONS))
        return self.config.condition_name

    def _condition_one_hot(self) -> np.ndarray:
        out = np.zeros(len(CONDITIONS), dtype=np.float32)
        out[CONDITIONS.index(self.episode_condition)] = 1.0
        return out

    def _make_explanation_schedule(self) -> list[str]:
        n = self.config.instances_per_episode
        shown_count = int(round(n * self.config.explanation_shown_ratio))
        schedule = ["none"] * n
        shown_indices = self.rng.choice(np.arange(n), size=shown_count, replace=False)
        for idx in shown_indices:
            if self.episode_condition == "linear_regression":
                schedule[int(idx)] = "lr"
            elif self.episode_condition == "decision_tree":
                schedule[int(idx)] = "dt"
            else:
                schedule[int(idx)] = str(self.rng.choice(["lr", "dt"]))
        return schedule

    def _xai_match_pools(self, bundle: CombinedBundle) -> dict[tuple[bool, bool], list[int]]:
        key = (bundle.app_id, bundle.model_name, self.episode_complexity)
        cached = self._xai_match_pool_cache.get(key)
        if cached is not None:
            return cached

        ids = list(bundle.instance_ids)
        raw_instances, labels = bundle.loader.load_instances(ids, normalize=False)
        lr_exp = bundle.lr_sparse if self.episode_complexity == "low" else bundle.lr_dense
        dt_exp = bundle.dt_depth2 if self.episode_complexity == "low" else bundle.dt_depth3
        pools = {(lr_match, dt_match): [] for lr_match in (False, True) for dt_match in (False, True)}
        for instance_id, raw_instance, label in zip(ids, raw_instances, labels):
            ai_prediction = int(label)
            lr_prediction = int(lr_exp.apply_to_instance(raw_instance) >= 0.0)
            dt_prediction = int(dt_exp.apply_to_instance(raw_instance)["class_index"])
            pools[(lr_prediction == ai_prediction, dt_prediction == ai_prediction)].append(int(instance_id))

        self._xai_match_pool_cache[key] = pools
        return pools

    def _sample_ids_for_xai_fidelity(self, bundle: CombinedBundle, target: float) -> list[int]:
        pools = self._xai_match_pools(bundle)
        n = self.config.instances_per_episode
        target_hits = int(round(n * float(np.clip(target, 0.0, 1.0))))
        categories = ((True, True), (True, False), (False, True), (False, False))
        capacities = {category: len(pools[category]) for category in categories}
        if sum(capacities.values()) < n:
            capacities = {category: (n if pools[category] else 0) for category in categories}

        best_score: tuple[int, int] | None = None
        best_plans: list[tuple[int, int, int, int]] = []
        for both in range(min(n, capacities[(True, True)]) + 1):
            remaining_after_both = n - both
            for lr_only in range(min(remaining_after_both, capacities[(True, False)]) + 1):
                remaining = remaining_after_both - lr_only
                dt_min = max(0, remaining - capacities[(False, False)])
                dt_max = min(remaining, capacities[(False, True)])
                if dt_min > dt_max:
                    continue
                preferred_dt = target_hits - both
                dt_only = int(np.clip(preferred_dt, dt_min, dt_max))
                neither = remaining - dt_only
                lr_hits = both + lr_only
                dt_hits = both + dt_only
                score = (
                    abs(lr_hits - target_hits) + abs(dt_hits - target_hits),
                    max(abs(lr_hits - target_hits), abs(dt_hits - target_hits)),
                )
                plan = (both, lr_only, dt_only, neither)
                if best_score is None or score < best_score:
                    best_score = score
                    best_plans = [plan]
                elif score == best_score:
                    best_plans.append(plan)

        if not best_plans:
            raise RuntimeError(f"Cannot sample {n} instances from {bundle.app_id}/{bundle.model_name}.")

        counts = best_plans[int(self.rng.integers(len(best_plans)))]
        chosen: list[int] = []
        for category, count in zip(categories, counts):
            if count:
                pool = np.asarray(pools[category], dtype=int)
                chosen.extend(
                    self.rng.choice(pool, size=count, replace=count > len(pool)).astype(int).tolist()
                )
        self.rng.shuffle(chosen)
        return chosen

    def _sample_instances(self, bundle: CombinedBundle) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        ids = np.asarray(bundle.instance_ids, dtype=int)
        if self.config.target_xai_fidelity is None:
            replace = len(ids) < self.config.instances_per_episode
            chosen = self.rng.choice(ids, size=self.config.instances_per_episode, replace=replace).astype(int).tolist()
        else:
            chosen = self._sample_ids_for_xai_fidelity(bundle, self.config.target_xai_fidelity)
        raw_instances, labels = bundle.loader.load_instances(chosen, normalize=False)
        norm_instances, _ = bundle.loader.load_instances(chosen, normalize=True)
        pools = self._xai_match_pools(bundle)
        match_by_id = {
            instance_id: category
            for category, category_ids in pools.items()
            for instance_id in category_ids
        }
        self.sampled_instance_ids = chosen
        self.lr_explanation_matches_ai = np.asarray([match_by_id[instance_id][0] for instance_id in chosen], dtype=bool)
        self.dt_explanation_matches_ai = np.asarray([match_by_id[instance_id][1] for instance_id in chosen], dtype=bool)
        self.lr_explanation_episode_fidelity = float(self.lr_explanation_matches_ai.mean())
        self.dt_explanation_episode_fidelity = float(self.dt_explanation_matches_ai.mean())
        return np.asarray(raw_instances, dtype=float), np.asarray(norm_instances, dtype=float), np.asarray(labels, dtype=int)

    def _current_lr_exp(self):
        assert self.bundle is not None
        return self.bundle.lr_sparse if self.episode_complexity == "low" else self.bundle.lr_dense

    def _current_dt_exp(self):
        assert self.bundle is not None
        return self.bundle.dt_depth2 if self.episode_complexity == "low" else self.bundle.dt_depth3

    def _init_state_memory(self, state: SubStrategyState) -> None:
        state.memory = make_memory(self.current_params["memory_recall_threshold"], self.config.memory_recall_noise)
        if state.strategy_name == "lr_calculation":
            add_lr_calculation_to_memory(
                self._current_lr_exp(),
                state.memory,
                intercept_significant_figures=self.config.displayed_significant_figures,
                coefficient_significant_figures=self.config.displayed_significant_figures,
            )
        elif state.strategy_name == "lr_heuristic":
            add_lr_heuristic_to_memory(self._current_lr_exp(), state.memory)
        else:
            add_dt_to_memory(
                state.memory,
                self._current_dt_exp(),
                threshold_significant_figures=self.config.displayed_significant_figures,
            )
        state.memory.tick(1.0)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.bundle = self.rng.choice(self.bundles)
        self.current_params = self._sample_episode_params()
        self.episode_condition = self._sample_episode_condition()
        self.episode_complexity = str(self.rng.choice(COMPLEXITIES))
        self.explanation_schedule = self._make_explanation_schedule()
        self.X_raw, self.X_norm, self.y = self._sample_instances(self.bundle)
        self.strategy_slots = (
            tuple(str(name) for name in self.rng.permutation(META_STRATEGIES))
            if self.config.randomize_strategy_order_per_episode
            else META_STRATEGIES
        )
        self.strategy_simulation_rngs = {
            "lr_calculation": random.Random(int(self.rng.integers(0, 2**63 - 1))),
            "lr_heuristic": np.random.default_rng(int(self.rng.integers(0, 2**63 - 1))),
            "dt_traversal": random.Random(int(self.rng.integers(0, 2**63 - 1))),
        }
        self.step_idx = 0
        self.recent_selected_strategies.clear()

        for state in self.states.values():
            state.contrib_history = []
            state.recent_prob_correct.clear()
            state.recent_pred_time.clear()
            for explanation_type in EXPLANATION_TYPES:
                state.recent_prob_correct_by_explanation[explanation_type].clear()
                state.recent_pred_time_by_explanation[explanation_type].clear()
            state.feature_order = (
                self.rng.permutation(self.config.max_features).astype(int)
                if self.config.randomize_feature_order_per_episode
                else np.arange(self.config.max_features, dtype=int)
            )
            state.current_contrib = _contribution_by_base_feature(
                self._current_lr_exp(),
                self.X_raw[self.step_idx],
                self.config.max_features,
            )
            self._init_state_memory(state)
        return self._get_obs(), {}

    def _history_array(self, values: deque[float], *, scale_time: bool = False) -> np.ndarray:
        out = np.full(self.config.history_window, -1.0, dtype=np.float32)
        for idx, value in enumerate(list(values)[-self.config.history_window:]):
            value = float(value)
            out[idx] = float(np.clip(value / 60.0, 0.0, 1.0)) if scale_time else value
        return out

    def _history_mean(self, values: deque[float], *, scale_time: bool = False) -> float:
        if not values:
            return -1.0
        arr = np.asarray(list(values)[-self.config.history_window:], dtype=float)
        if scale_time:
            arr = np.clip(arr / 60.0, 0.0, 1.0)
        return float(np.mean(arr))

    def _recent_strategy_one_hot(self) -> np.ndarray:
        out = np.full((self.config.history_window, len(META_STRATEGIES)), -1.0, dtype=np.float32)
        recent = list(self.recent_selected_strategies)[-self.config.history_window:]
        start = self.config.history_window - len(recent)
        for offset, strategy_idx in enumerate(recent):
            out[start + offset, :] = 0.0
            out[start + offset, int(strategy_idx)] = 1.0
        return out.ravel()

    def _get_obs(self) -> np.ndarray:
        if self.step_idx >= self.config.instances_per_episode or not self.explanation_schedule:
            return np.zeros(self.observation_space.shape, dtype=np.float32)
        if self.config.meta_history_by_explanation:
            success_means = np.array(
                [
                    self._history_mean(self.states[name].recent_prob_correct_by_explanation[explanation_type])
                    for name in self.strategy_slots
                    for explanation_type in EXPLANATION_TYPES
                ],
                dtype=np.float32,
            )
            time_means = np.array(
                [
                    self._history_mean(
                        self.states[name].recent_pred_time_by_explanation[explanation_type],
                        scale_time=True,
                    )
                    for name in self.strategy_slots
                    for explanation_type in EXPLANATION_TYPES
                ],
                dtype=np.float32,
            )
        else:
            success_means = np.array(
                [self._history_mean(self.states[name].recent_prob_correct) for name in self.strategy_slots],
                dtype=np.float32,
            )
            time_means = np.array(
                [self._history_mean(self.states[name].recent_pred_time, scale_time=True) for name in self.strategy_slots],
                dtype=np.float32,
            )
        params = np.array(
            [
                _normalize_scalar(self.current_params["decision_noise"], self.config.decision_noise_min, self.config.decision_noise_max),
                _normalize_scalar(self.current_params["memory_recall_threshold"], self.config.memory_recall_threshold_min, self.config.memory_recall_threshold_max),
                _normalize_scalar(self.current_params["opportunity_cost"], self.config.opportunity_cost_min, self.config.opportunity_cost_max),
            ],
            dtype=np.float32,
        )
        explanation_id = EXPLANATION_IDS[self.explanation_schedule[self.step_idx]] if self.explanation_schedule else 0
        explanation_one_hot = np.zeros(3, dtype=np.float32)
        explanation_one_hot[explanation_id] = 1.0
        progress = np.array([float(self.step_idx / max(1, self.config.instances_per_episode))], dtype=np.float32)
        return np.concatenate([
            success_means,
            time_means,
            self._recent_strategy_one_hot(),
            params,
            self._condition_one_hot(),
            explanation_one_hot,
            progress,
        ]).astype(np.float32)

    def _sub_common_obs(self, state: SubStrategyState, own_explanation_shown: bool) -> np.ndarray:
        probs = self._history_array(state.recent_prob_correct)
        times = self._history_array(state.recent_pred_time, scale_time=True)
        return np.array(
            [
                _normalize_scalar(self.current_params["decision_noise"], self.config.decision_noise_min, self.config.decision_noise_max),
                _normalize_scalar(self.current_params["memory_recall_threshold"], self.config.memory_recall_threshold_min, self.config.memory_recall_threshold_max),
                _normalize_scalar(self.current_params["opportunity_cost"], self.config.opportunity_cost_min, self.config.opportunity_cost_max),
                float(own_explanation_shown),
                float(self.step_idx / max(1, self.config.instances_per_episode)),
                *probs.tolist(),
                *times.tolist(),
            ],
            dtype=np.float32,
        )

    def _sub_obs(self, strategy_name: str, own_explanation_shown: bool) -> np.ndarray:
        state = self.states[strategy_name]
        common = self._sub_common_obs(state, own_explanation_shown)
        if strategy_name == "lr_calculation":
            current_norm = _normalized_abs(state.current_contrib)
            if state.contrib_history:
                hist = np.vstack(state.contrib_history)
                means = _normalized_abs(hist.mean(axis=0))
                stds = _normalized_abs(hist.std(axis=0))
            else:
                means = np.zeros(self.config.max_features, dtype=float)
                stds = np.zeros(self.config.max_features, dtype=float)
            if self.config.randomize_feature_order_per_episode:
                current_norm = current_norm[state.feature_order]
                means = means[state.feature_order]
                stds = stds[state.feature_order]
            return np.concatenate([common, current_norm, means, stds]).astype(np.float32)
        if strategy_name == "dt_traversal":
            current_norm = _normalized_abs(state.current_contrib)
            if self.config.randomize_feature_order_per_episode:
                current_norm = current_norm[state.feature_order]
            return np.concatenate([common, current_norm]).astype(np.float32)

        means = np.zeros(self.config.max_features, dtype=float)
        variances = np.ones(self.config.max_features, dtype=float)
        counts = np.zeros(self.config.max_features, dtype=float)
        assert state.memory is not None
        for chunk in state.memory.chunks:
            slots = getattr(chunk, "slots", {})
            if slots.get("type") != "coef_prob":
                continue
            idx = _feature_base_index(str(slots.get("feature_key", "a0")))
            if idx >= self.config.max_features:
                continue
            means[idx] += float(slots.get("mu", 0.0))
            variances[idx] += float(slots.get("var", 1.0))
            counts[idx] += 1.0
        present = counts > 0
        means[present] /= counts[present]
        variances[present] /= counts[present]
        mean_obs = np.clip(means, -1.0, 1.0)
        std_obs = np.clip(np.sqrt(np.maximum(variances, 0.0)), 0.0, 1.0)
        if self.config.randomize_feature_order_per_episode:
            mean_obs = mean_obs[state.feature_order]
            std_obs = std_obs[state.feature_order]
        return np.concatenate([common, mean_obs, std_obs]).astype(np.float32)

    def _sub_action(self, strategy_name: str, own_explanation_shown: bool):
        if strategy_name in self.sub_policies:
            action, _ = self.sub_policies[strategy_name].predict(self._sub_obs(strategy_name, own_explanation_shown), deterministic=True)
            return action
        if not self.config.allow_random_subpolicy:
            raise RuntimeError(f"Missing sub-policy for {strategy_name}")
        if strategy_name == "dt_traversal":
            return self.rng.integers(self.config.decision_boundary_bins)
        if strategy_name == "lr_heuristic":
            return np.concatenate([[self.rng.integers(self.config.decision_boundary_bins)], self.rng.integers(2, size=self.config.max_features)])
        return np.concatenate([[self.rng.integers(2), self.rng.integers(self.config.decision_boundary_bins)], self.rng.integers(2, size=self.config.max_features)])

    def _decision_boundary_from_bin(self, bin_id: int) -> float:
        bin_id = int(np.clip(bin_id, 0, self.config.decision_boundary_bins - 1))
        frac = bin_id / max(1, self.config.decision_boundary_bins - 1)
        return float(self.config.decision_boundary_min + frac * (self.config.decision_boundary_max - self.config.decision_boundary_min))

    def _parse_feature_action(self, state: SubStrategyState, action, *, offset: int) -> tuple[list[int], list[int], np.ndarray, np.ndarray]:
        action = np.asarray(action, dtype=int).reshape(-1)
        slot_bits = action[offset : offset + self.config.max_features]
        slots = [int(i) for i, bit in enumerate(slot_bits) if bit == 1]
        actual = [int(state.feature_order[i]) for i in slots]
        slot_mask = np.zeros(self.config.max_features, dtype=int)
        feature_mask = np.zeros(self.config.max_features, dtype=int)
        for slot in slots:
            slot_mask[slot] = 1
        for idx in actual:
            feature_mask[idx] = 1
        return actual, slots, feature_mask, slot_mask

    def _run_strategy(self, strategy_name: str, explanation_type: str, y_true: int) -> dict[str, Any]:
        assert self.X_raw is not None and self.X_norm is not None
        state = self.states[strategy_name]
        assert state.memory is not None
        own_explanation = (
            (strategy_name in {"lr_calculation", "lr_heuristic"} and explanation_type == "lr")
            or (strategy_name == "dt_traversal" and explanation_type == "dt")
        )
        action = self._sub_action(strategy_name, own_explanation)

        if strategy_name == "lr_calculation":
            action_arr = np.asarray(action, dtype=int).reshape(-1)
            requested_mode = "read" if int(action_arr[0]) == 1 else "retrieve"
            mode = requested_mode if own_explanation else "retrieve"
            boundary = self._decision_boundary_from_bin(int(action_arr[1]))
            selected, selected_slots, feature_mask, slot_mask = self._parse_feature_action(state, action_arr, offset=2)
            probs, pred_time, _info = lr_calculation(
                self.X_raw[self.step_idx],
                state.memory,
                self._current_lr_exp(),
                explanation_access_mode=mode,
                displayed_significant_figures=self.config.displayed_significant_figures,
                read_seconds_per_item=self.config.read_seconds_per_item,
                mental_calculation_seconds=self.config.mental_calculation_seconds,
                decision_boundary=boundary,
                decision_noise=self.current_params["decision_noise"],
                selected_feature_indices=selected,
                simulation_sample_count=self.config.simulation_sample_count,
                retrieval_candidate_count=self.config.retrieval_candidate_count,
                rng=self.strategy_simulation_rngs["lr_calculation"],
            )
        elif strategy_name == "lr_heuristic":
            action_arr = np.asarray(action, dtype=int).reshape(-1)
            boundary = self._decision_boundary_from_bin(int(action_arr[0]))
            selected, selected_slots, feature_mask, slot_mask = self._parse_feature_action(state, action_arr, offset=1)
            probs, pred_time, info = lr_heuristic(
                self.X_norm[self.step_idx],
                state.memory,
                self._current_lr_exp(),
                simulation_sample_count=self.config.simulation_sample_count,
                retrieval_candidate_count=self.config.retrieval_candidate_count,
                read_seconds_per_item=self.config.read_seconds_per_item,
                mental_calculation_seconds=self.config.mental_calculation_seconds,
                decision_boundary=boundary,
                decision_noise=self.current_params["decision_noise"],
                selected_feature_indices=selected,
                rng=self.strategy_simulation_rngs["lr_heuristic"],
            )
            refresh_lr_heuristic_in_memory(state.memory, self._current_lr_exp(), info, y_true, selected_feature_indices=selected)
            mode = "heuristic"
        else:
            action_scalar = int(np.asarray(action).reshape(-1)[0])
            boundary = self._decision_boundary_from_bin(action_scalar)
            selected, selected_slots = [], []
            feature_mask = np.zeros(self.config.max_features, dtype=int)
            slot_mask = np.zeros(self.config.max_features, dtype=int)
            mode = "read" if own_explanation else "retrieve"
            probs, pred_time, _info = dt_traverse(
                self.X_raw[self.step_idx],
                state.memory,
                self._current_dt_exp(),
                explanation_access_mode=mode,
                displayed_significant_figures=self.config.displayed_significant_figures,
                read_seconds_per_item=self.config.read_seconds_per_item,
                decision_boundary=boundary,
                decision_noise=self.current_params["decision_noise"],
                simulation_sample_count=self.config.simulation_sample_count,
                retrieval_candidate_count=self.config.retrieval_candidate_count,
                rng=self.strategy_simulation_rngs["dt_traversal"],
            )

        probs = _safe_probabilities(probs)
        prob_correct = float(probs[y_true]) if y_true < len(probs) else 0.0
        predicted_class = int(np.argmax(probs)) if len(probs) else -1
        state.recent_prob_correct.append(prob_correct)
        state.recent_pred_time.append(float(pred_time))
        state.recent_prob_correct_by_explanation[explanation_type].append(prob_correct)
        state.recent_pred_time_by_explanation[explanation_type].append(float(pred_time))
        return {
            "strategy_name": strategy_name,
            "own_explanation_shown": own_explanation,
            "mode": mode,
            "prob_correct": prob_correct,
            "predicted_class": predicted_class,
            "predicted_correct": bool(predicted_class == y_true),
            "pred_time": float(pred_time),
            "decision_boundary": float(boundary),
            "selected_features": selected,
            "selected_feature_slots": selected_slots,
            "feature_mask": feature_mask.astype(int).tolist(),
            "feature_slot_mask": slot_mask.astype(int).tolist(),
        }

    def _strategy_available(self, strategy_name: str) -> bool:
        if self.episode_condition == "linear_regression":
            return strategy_name in {"lr_calculation", "lr_heuristic"}
        if self.episode_condition == "decision_tree":
            return strategy_name == "dt_traversal"
        return True

    def action_masks(self) -> np.ndarray:
        """Return legal anonymous slots without exposing strategy identity."""
        return np.asarray(
            [self._strategy_available(strategy_name) for strategy_name in self.strategy_slots],
            dtype=bool,
        )

    def _unavailable_outcome(self, strategy_name: str) -> dict[str, Any]:
        return {
            "strategy_name": strategy_name,
            "own_explanation_shown": False,
            "mode": "unavailable",
            "prob_correct": 0.0,
            "predicted_class": -1,
            "predicted_correct": False,
            "pred_time": 0.0,
            "decision_boundary": 0.0,
            "selected_features": [],
            "selected_feature_slots": [],
            "feature_mask": np.zeros(self.config.max_features, dtype=int).tolist(),
            "feature_slot_mask": np.zeros(self.config.max_features, dtype=int).tolist(),
        }

    def step(self, action):
        assert self.X_raw is not None and self.y is not None
        selected_slot = int(np.asarray(action).reshape(-1)[0])
        selected_strategy = self.strategy_slots[selected_slot]
        explanation_type = self.explanation_schedule[self.step_idx]
        y_true = int(self.y[self.step_idx])

        # Unavailable strategies are masked for the meta-policy and are not
        # simulated. The penalty remains defensive for manual or legacy actions.
        outcomes = {
            strategy_name: (
                self._run_strategy(strategy_name, explanation_type, y_true)
                if self._strategy_available(strategy_name)
                else self._unavailable_outcome(strategy_name)
            )
            for strategy_name in META_STRATEGIES
        }
        selected = outcomes[selected_strategy]
        unavailable = not self._strategy_available(selected_strategy)
        if unavailable:
            reward = float(self.config.unavailable_strategy_penalty)
        else:
            reward = (
                selected["prob_correct"]
                - self.current_params["opportunity_cost"] * selected["pred_time"]
            )

        for state in self.states.values():
            state.contrib_history.append(state.current_contrib.copy())
        self.recent_selected_strategies.append(selected_slot)

        info = {
            "condition_name": self.episode_condition,
            "condition_mode": self.config.condition_name,
            "episode_complexity": self.episode_complexity,
            "app_id": self.bundle.app_id if self.bundle else None,
            "model_name": self.bundle.model_name if self.bundle else None,
            "lr_explanation_matches_ai": bool(self.lr_explanation_matches_ai[self.step_idx]),
            "dt_explanation_matches_ai": bool(self.dt_explanation_matches_ai[self.step_idx]),
            "lr_explanation_episode_fidelity": self.lr_explanation_episode_fidelity,
            "dt_explanation_episode_fidelity": self.dt_explanation_episode_fidelity,
            "explanation_type": explanation_type,
            "selected_strategy_slot": selected_slot,
            "strategy_slot_order": list(self.strategy_slots),
            "selected_strategy": selected_strategy,
            "selected_strategy_available": not unavailable,
            "reward": float(reward),
            "decision_noise": self.current_params["decision_noise"],
            "memory_recall_threshold": self.current_params["memory_recall_threshold"],
            "opportunity_cost": self.current_params["opportunity_cost"],
            **{f"{name}_prob_correct": outcomes[name]["prob_correct"] for name in META_STRATEGIES},
            **{f"{name}_predicted_correct": outcomes[name]["predicted_correct"] for name in META_STRATEGIES},
            **{f"{name}_pred_time": outcomes[name]["pred_time"] for name in META_STRATEGIES},
            "selected_prob_correct": selected["prob_correct"],
            "selected_predicted_class": selected["predicted_class"],
            "selected_predicted_correct": selected["predicted_correct"],
            "selected_pred_time": selected["pred_time"],
            "selected_decision_boundary": selected["decision_boundary"],
            "selected_mode": selected["mode"],
            "selected_features": selected["selected_features"],
            "selected_feature_slots": selected["selected_feature_slots"],
            "selected_feature_mask": selected["feature_mask"],
            "selected_feature_slot_mask": selected["feature_slot_mask"],
        }

        self.step_idx += 1
        terminated = False
        truncated = self.step_idx >= self.config.instances_per_episode
        if not truncated:
            for state in self.states.values():
                state.current_contrib = _contribution_by_base_feature(
                    self._current_lr_exp(),
                    self.X_raw[self.step_idx],
                    self.config.max_features,
                )
        return self._get_obs(), float(reward), terminated, truncated, info


def make_meta_run_dir(config: CombinedPolicyConfig) -> Path:
    run_name = config.run_name or f"combined_{config.condition_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(config.output_root) / run_name / config.condition_name
    for subdir in ["models", "logs", "metrics"]:
        (run_dir / subdir).mkdir(parents=True, exist_ok=True)
    return run_dir


def make_meta_vec_env(bundles: list[CombinedBundle], config: CombinedPolicyConfig, sub_policies: dict[str, PPO], n_envs: int, training: bool):
    def factory(rank: int):
        def _init():
            env = CombinedStrategyPolicyEnv(bundles, config, sub_policies, training=training)
            env.reset(seed=config.seed + rank)
            return Monitor(env)
        return _init

    return DummyVecEnv([factory(i) for i in range(n_envs)])


def train_combined_policy(config: CombinedPolicyConfig) -> tuple[MaskablePPO, Path, list[CombinedBundle]]:
    run_dir = make_meta_run_dir(config)
    bundles = load_combined_bundles(Path(config.data_dir), config)
    if not bundles:
        raise RuntimeError(f"No combined bundles loaded for condition={config.condition_name}.")
    sub_policies = load_sub_policies(config)
    (run_dir / "config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")

    train_env = make_meta_vec_env(bundles, config, sub_policies, config.n_envs, training=True)
    eval_env = make_meta_vec_env(bundles, config, sub_policies, 1, training=False)
    model = MaskablePPO(
        "MlpPolicy",
        train_env,
        learning_rate=config.learning_rate,
        gamma=config.gamma,
        ent_coef=config.ent_coef,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        verbose=1,
        seed=config.seed,
        tensorboard_log=str(run_dir / "logs") if importlib.util.find_spec("tensorboard") else None,
        device="cpu",
    )
    callback = MaskableEvalCallback(
        eval_env,
        best_model_save_path=str(run_dir / "models"),
        log_path=str(run_dir / "metrics"),
        eval_freq=max(config.n_steps, 1),
        deterministic=True,
    )
    started = time.time()
    model.learn(total_timesteps=config.total_timesteps, callback=callback, progress_bar=False)
    elapsed = time.time() - started
    model_path = run_dir / "models" / "final_model.zip"
    model.save(model_path)
    (run_dir / "metrics" / "training_summary.json").write_text(
        json.dumps({"elapsed_seconds": elapsed, "model_path": str(model_path)}, indent=2),
        encoding="utf-8",
    )
    train_env.close()
    eval_env.close()
    return model, run_dir, bundles


def train_all_combined_conditions(base_config: CombinedPolicyConfig) -> dict[str, tuple[MaskablePPO, Path, list[CombinedBundle]]]:
    return {
        condition: train_combined_policy(CombinedPolicyConfig(**{**asdict(base_config), "condition_name": condition}))
        for condition in CONDITIONS
    }


def resolve_combined_model_path(
    config: CombinedPolicyConfig,
    *,
    model_filename: str = "final_model.zip",
) -> tuple[Path, Path]:
    if config.condition_name not in CONDITION_MODES:
        raise ValueError(f"Unknown condition_name={config.condition_name!r}; expected one of {CONDITION_MODES}.")

    output_root = Path(config.output_root)
    if config.run_name:
        run_dir = output_root / config.run_name / config.condition_name
        return run_dir / "models" / model_filename, run_dir

    candidates = sorted(
        output_root.glob(f"*/{config.condition_name}/models/{model_filename}"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No saved {model_filename} found below {output_root} for condition={config.condition_name}."
        )
    model_path = candidates[0]
    return model_path, model_path.parents[1]


def load_combined_policy(
    config: CombinedPolicyConfig,
    *,
    model_path: str | Path | None = None,
    model_filename: str = "final_model.zip",
    device: str = "cpu",
) -> tuple[MaskablePPO, Path, list[CombinedBundle]]:
    resolved_model_path, run_dir = (
        (
            Path(model_path),
            Path(model_path).parent.parent if Path(model_path).parent.name == "models" else Path(model_path).parent,
        )
        if model_path is not None
        else resolve_combined_model_path(config, model_filename=model_filename)
    )
    if not resolved_model_path.exists():
        raise FileNotFoundError(f"Saved combined meta-policy does not exist: {resolved_model_path}")

    bundles = load_combined_bundles(Path(config.data_dir), config)
    if not bundles:
        raise RuntimeError(f"No combined bundles loaded for condition={config.condition_name}.")

    (run_dir / "metrics").mkdir(parents=True, exist_ok=True)
    model = MaskablePPO.load(resolved_model_path, device=device)
    return model, run_dir, bundles


def evaluate_combined_policy(
    model: MaskablePPO,
    bundles: list[CombinedBundle],
    config: CombinedPolicyConfig,
    *,
    n_episodes: int = 50,
    deterministic: bool = True,
    fixed_eval_params: dict[str, float] | None = None,
) -> pd.DataFrame:
    env = CombinedStrategyPolicyEnv(
        bundles,
        config,
        load_sub_policies(config),
        training=False,
        fixed_eval_params=fixed_eval_params,
    )
    rows: list[dict[str, Any]] = []
    for episode in range(n_episodes):
        obs, _ = env.reset(seed=config.seed + 20_000 + episode)
        done = False
        while not done:
            action, _ = model.predict(
                obs,
                deterministic=deterministic,
                action_masks=env.action_masks(),
            )
            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            rows.append({"episode": episode, "step": env.step_idx - 1, **info})
    env.close()
    return pd.DataFrame(rows)


def summarize_combined_evaluation(evaluation_df: pd.DataFrame, *, by_dataset: bool = True) -> pd.DataFrame:
    if evaluation_df.empty:
        return pd.DataFrame()

    dataset_cols = [column for column in ("app_id", "model_name") if by_dataset and column in evaluation_df.columns]
    group_cols = [
        *dataset_cols,
        "condition_name",
        "selected_strategy",
        "selected_strategy_available",
    ]
    summary = (
        evaluation_df.groupby(group_cols, dropna=False)
        .agg(
            trials=("reward", "size"),
            mean_reward=("reward", "mean"),
            mean_prob_correct=("selected_prob_correct", "mean"),
            mean_time=("selected_pred_time", "mean"),
        )
        .reset_index()
    )

    total_cols = [*dataset_cols, "condition_name"]
    totals = evaluation_df.groupby(total_cols, dropna=False).size().reset_index(name="condition_trials")
    summary = summary.merge(totals, on=total_cols, how="left")
    summary["selection_rate"] = summary["trials"] / summary["condition_trials"].clip(lower=1)
    return summary.sort_values(group_cols).reset_index(drop=True)


def default_combined_parameter_sweep_values(config: CombinedPolicyConfig, n_points: int = 5) -> dict[str, list[float]]:
    n_points = max(2, int(n_points))
    return {
        "decision_noise": np.linspace(config.decision_noise_min, config.decision_noise_max, n_points).round(6).tolist(),
        "memory_recall_threshold": np.linspace(
            config.memory_recall_threshold_min,
            config.memory_recall_threshold_max,
            n_points,
        ).round(6).tolist(),
        "opportunity_cost": np.linspace(config.opportunity_cost_min, config.opportunity_cost_max, n_points).round(6).tolist(),
    }


def sweep_combined_parameters(
    model: MaskablePPO,
    bundles: list[CombinedBundle],
    config: CombinedPolicyConfig,
    *,
    sweep_values: dict[str, list[float]] | None = None,
    n_episodes: int = 25,
    deterministic: bool = True,
) -> pd.DataFrame:
    """
    Evaluate one trained combined policy while clamping one cognitive parameter at a time.

    The other cognitive parameters stay at their evaluation defaults, which are the
    midpoint of their configured training range.
    """
    sweep_values = sweep_values or default_combined_parameter_sweep_values(config)
    supported = {"decision_noise", "memory_recall_threshold", "opportunity_cost"}
    frames: list[pd.DataFrame] = []

    for parameter_name, values in sweep_values.items():
        if parameter_name not in supported:
            raise ValueError(
                "Supported sweep parameters are decision_noise, memory_recall_threshold, and opportunity_cost; "
                f"got {parameter_name!r}."
            )
        for parameter_value in values:
            eval_df = evaluate_combined_policy(
                model,
                bundles,
                config,
                n_episodes=n_episodes,
                deterministic=deterministic,
                fixed_eval_params={parameter_name: float(parameter_value)},
            )
            eval_df["sweep_parameter"] = parameter_name
            eval_df["sweep_value"] = float(parameter_value)
            frames.append(eval_df)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def summarize_combined_parameter_sweep(sweep_df: pd.DataFrame) -> pd.DataFrame:
    if sweep_df.empty:
        return pd.DataFrame()

    group_cols = ["sweep_parameter", "sweep_value", "condition_name", "selected_strategy"]
    grouped = sweep_df.groupby(group_cols, dropna=False).agg(
        trials=("reward", "size"),
        mean_reward=("reward", "mean"),
        mean_prob_correct=("selected_prob_correct", "mean"),
        mean_time=("selected_pred_time", "mean"),
        availability_rate=("selected_strategy_available", "mean"),
    ).reset_index()

    totals = sweep_df.groupby(["sweep_parameter", "sweep_value", "condition_name"], dropna=False).size().reset_index(name="condition_trials")
    summary = grouped.merge(totals, on=["sweep_parameter", "sweep_value", "condition_name"], how="left")
    summary["selection_rate"] = summary["trials"] / summary["condition_trials"].clip(lower=1)

    # Include explicit zero rows so tables and plots show strategies that were
    # available to the action space but never chosen at a sweep value.
    index_cols = ["sweep_parameter", "sweep_value", "condition_name"]
    all_index_rows = totals[index_cols].drop_duplicates()
    full_rows = []
    for row in all_index_rows.itertuples(index=False):
        for strategy_name in META_STRATEGIES:
            full_rows.append(
                {
                    "sweep_parameter": row.sweep_parameter,
                    "sweep_value": row.sweep_value,
                    "condition_name": row.condition_name,
                    "selected_strategy": strategy_name,
                }
            )
    full_index = pd.DataFrame(full_rows)
    summary = full_index.merge(summary, on=group_cols, how="left")
    summary = summary.merge(totals, on=index_cols, how="left", suffixes=("", "_total"))
    if "condition_trials_total" in summary:
        summary["condition_trials"] = summary["condition_trials"].fillna(summary["condition_trials_total"])
        summary = summary.drop(columns=["condition_trials_total"])

    fill_zero_cols = [
        "trials",
        "mean_reward",
        "mean_prob_correct",
        "mean_time",
        "availability_rate",
        "selection_rate",
    ]
    for column in fill_zero_cols:
        summary[column] = summary[column].fillna(0.0)
    summary["trials"] = summary["trials"].astype(int)
    summary["condition_trials"] = summary["condition_trials"].fillna(0).astype(int)
    return summary


def summarize_meta_strategy_selection_by_dataset(
    sweep_df: pd.DataFrame,
    *,
    parameter_name: str = "opportunity_cost",
) -> pd.DataFrame:
    if sweep_df.empty:
        return pd.DataFrame()
    required = {
        "sweep_parameter",
        "sweep_value",
        "condition_name",
        "app_id",
        "model_name",
        "selected_strategy",
        "reward",
        "selected_prob_correct",
        "selected_pred_time",
        "selected_strategy_available",
    }
    missing = required - set(sweep_df.columns)
    if missing:
        raise ValueError(f"sweep_df is missing required column(s): {sorted(missing)}")

    filtered = sweep_df[sweep_df["sweep_parameter"] == parameter_name].copy()
    if filtered.empty:
        return pd.DataFrame()

    index_cols = ["sweep_parameter", "sweep_value", "condition_name", "app_id", "model_name"]
    group_cols = [*index_cols, "selected_strategy"]
    grouped = (
        filtered.groupby(group_cols, dropna=False)
        .agg(
            trials=("reward", "size"),
            mean_reward=("reward", "mean"),
            mean_prob_correct=("selected_prob_correct", "mean"),
            mean_time=("selected_pred_time", "mean"),
            availability_rate=("selected_strategy_available", "mean"),
        )
        .reset_index()
    )

    totals = filtered.groupby(index_cols, dropna=False).size().reset_index(name="dataset_trials")
    all_index_rows = totals[index_cols].drop_duplicates()
    full_rows = []
    for row in all_index_rows.itertuples(index=False):
        for strategy_name in META_STRATEGIES:
            full_rows.append(
                {
                    "sweep_parameter": row.sweep_parameter,
                    "sweep_value": row.sweep_value,
                    "condition_name": row.condition_name,
                    "app_id": row.app_id,
                    "model_name": row.model_name,
                    "selected_strategy": strategy_name,
                }
            )
    full_index = pd.DataFrame(full_rows)
    summary = full_index.merge(grouped, on=group_cols, how="left")
    summary = summary.merge(totals, on=index_cols, how="left")

    for column in ["trials", "mean_reward", "mean_prob_correct", "mean_time", "availability_rate"]:
        summary[column] = summary[column].fillna(0.0)
    summary["trials"] = summary["trials"].astype(int)
    summary["dataset_trials"] = summary["dataset_trials"].fillna(0).astype(int)
    summary["selection_rate"] = summary["trials"] / summary["dataset_trials"].clip(lower=1)
    return summary.sort_values(group_cols).reset_index(drop=True)


def plot_meta_strategy_selection_by_dataset(
    sweep_df: pd.DataFrame,
    *,
    output_dir: str | Path | None = None,
    show: bool = True,
    max_columns: int = 3,
) -> dict[str, Any]:
    import matplotlib.pyplot as plt

    summary = summarize_meta_strategy_selection_by_dataset(sweep_df, parameter_name="opportunity_cost")
    if summary.empty:
        return {}

    output_path = Path(output_dir) if output_dir is not None else None
    if output_path is not None:
        output_path.mkdir(parents=True, exist_ok=True)

    plot_df = summary.copy()
    plot_df["dataset"] = plot_df["app_id"].astype(str) + " / " + plot_df["model_name"].astype(str)
    colors = dict(zip(META_STRATEGIES, ["#4c78a8", "#f58518", "#54a24b"]))
    figures: dict[str, Any] = {}

    for condition_name, condition_df in plot_df.groupby("condition_name"):
        datasets = sorted(condition_df["dataset"].dropna().unique().tolist())
        n_cols = max(1, min(int(max_columns), len(datasets)))
        n_rows = int(np.ceil(len(datasets) / n_cols))
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(5.2 * n_cols, 3.4 * n_rows),
            squeeze=False,
            sharex=True,
            sharey=True,
        )

        for ax, dataset in zip(axes.ravel(), datasets):
            dataset_df = condition_df[condition_df["dataset"] == dataset]
            for strategy_name in META_STRATEGIES:
                strategy_df = dataset_df[dataset_df["selected_strategy"] == strategy_name].sort_values("sweep_value")
                if strategy_df.empty:
                    continue
                ax.plot(
                    strategy_df["sweep_value"],
                    strategy_df["selection_rate"],
                    marker="o",
                    label=strategy_name,
                    color=colors.get(strategy_name),
                )
            ax.set_title(dataset)
            ax.set_xlabel("Opportunity cost")
            ax.set_ylim(0.0, 1.0)
            ax.grid(True, alpha=0.3)

        for ax in axes.ravel()[len(datasets):]:
            ax.axis("off")
        axes[0, 0].set_ylabel("Meta-policy selection rate")
        handles, labels = axes.ravel()[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), len(META_STRATEGIES)))
        fig.suptitle(f"Meta-policy strategy selection across opportunity cost: {condition_name}", y=1.02)
        fig.tight_layout()

        key = f"opportunity_cost_{condition_name}_strategy_selection_by_dataset"
        figures[key] = fig
        if output_path is not None:
            safe_condition = str(condition_name).replace(" ", "_")
            fig.savefig(
                output_path / f"combined_sweep_opportunity_cost_strategy_selection_by_dataset_{safe_condition}.png",
                dpi=160,
                bbox_inches="tight",
            )

    if not show:
        for fig in figures.values():
            plt.close(fig)
    return figures


def plot_combined_parameter_sweep(
    sweep_df: pd.DataFrame,
    *,
    output_dir: str | Path | None = None,
    show: bool = True,
) -> dict[str, Any]:
    import matplotlib.pyplot as plt

    if sweep_df.empty:
        return {}

    output_path = Path(output_dir) if output_dir is not None else None
    if output_path is not None:
        output_path.mkdir(parents=True, exist_ok=True)

    summary = summarize_combined_parameter_sweep(sweep_df)
    figures: dict[str, Any] = {}

    for parameter_name, param_df in summary.groupby("sweep_parameter"):
        conditions = [condition for condition in CONDITIONS if condition in set(param_df["condition_name"])]
        n_cols = max(1, len(conditions))
        fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 4), sharey=True)
        axes = np.atleast_1d(axes)

        for ax, condition in zip(axes, conditions):
            condition_df = param_df[param_df["condition_name"] == condition]
            for strategy_name in META_STRATEGIES:
                strategy_df = condition_df[condition_df["selected_strategy"] == strategy_name].sort_values("sweep_value")
                if strategy_df.empty:
                    continue
                ax.plot(strategy_df["sweep_value"], strategy_df["selection_rate"], marker="o", label=strategy_name)
            ax.set_title(condition)
            ax.set_xlabel(parameter_name)
            ax.set_ylim(0.0, 1.0)
            ax.grid(True, alpha=0.3)

        axes[0].set_ylabel("Selection rate")
        axes[-1].legend(loc="best")
        fig.suptitle(f"Meta-policy strategy selection across {parameter_name}", y=1.04)
        fig.tight_layout()

        key = f"{parameter_name}_strategy_selection"
        figures[key] = fig
        if output_path is not None:
            fig.savefig(output_path / f"combined_sweep_{parameter_name}_strategy_selection.png", dpi=160, bbox_inches="tight")

    if not show:
        for fig in figures.values():
            plt.close(fig)
    return figures


def combined_observation_action_summary(config: CombinedPolicyConfig) -> pd.DataFrame:
    obs_dim = _meta_observation_dim(config)
    if config.meta_history_by_explanation:
        history_contents = (
            f"slot-major per anonymous strategy slot, mean probability-correct and mean time over the previous "
            f"{config.history_window} occurrences of each shown explanation type (none, lr, dt)"
        )
    else:
        history_contents = (
            f"per anonymous strategy slot, mean probability-correct and mean time over the previous "
            f"{config.history_window} trials"
        )
    return pd.DataFrame(
        [
            {
                "condition_name": condition,
                "observation_space": f"Box({obs_dim}, low=-1, high=1)",
                "observation_contents": f"{history_contents} + previous {config.history_window} selected slots as one-hot slots + 3 cognitive params + condition one-hot + explanation one-hot + episode progress",
                "action_space": "Discrete(3)",
                "action_contents": "0..2 select anonymous strategy slots shuffled once per episode; unavailable slots are action-masked",
            }
            for condition in CONDITION_MODES
        ]
    )

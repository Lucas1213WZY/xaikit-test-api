from __future__ import annotations

import importlib.util
import json
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
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from ..dt_memory import add_dt_to_memory, dt_traverse
from ..heuristic_lr_model import add_lr_heuristic_to_memory, lr_heuristic, refresh_lr_heuristic_in_memory
from ..lr_memory import add_lr_calculation_to_memory, lr_calculation
from ..memory import CombinedMemory, DeclarativeMemory
from ..utils import AIDatasetLoader, DecisionTreeInterpreter, LogisticRegressionInterpreter, filter_by_app_and_model


ACCESS_MODES = {0: "retrieve", 1: "read"}
STRATEGIES = ("lr_calculation", "lr_heuristic", "dt_traversal")


@dataclass(frozen=True)
class StrategyTrainingConfig:
    data_dir: str
    output_root: str
    run_name: str | None = None
    strategy_name: str = "lr_calculation"
    total_timesteps: int = 100_000
    n_envs: int = 4
    instances_per_episode: int = 40
    max_features: int = 6
    explanation_shown_ratio: float = 0.5
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
    decision_noise_max: float = 0.5
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
    feature_cost: float = 0.002
    illegal_read_penalty: float = -1.0
    dt_depth: int = 2
    history_window: int = 5
    randomize_feature_order_per_episode: bool = True
    apps: tuple[str, ...] | None = None


@dataclass(frozen=True)
class StrategyBundle:
    app_id: str
    model_name: str
    loader: AIDatasetLoader
    lr_exp: LogisticRegressionInterpreter
    dt_exp: DecisionTreeInterpreter
    instance_ids: tuple[int, ...]


def _normalize_scalar(value: float, low: float, high: float) -> float:
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return 0.0
    return float(np.clip((value - low) / (high - low), 0.0, 1.0))


def _safe_probabilities(probs: np.ndarray) -> np.ndarray:
    p = np.asarray(probs, dtype=float).copy()
    p[~np.isfinite(p)] = 0.0
    p[p < 0.0] = 0.0
    total = float(p.sum())
    if total <= 0.0:
        return np.full(len(p), 1.0 / max(1, len(p)), dtype=float)
    return p / total


def _flatten_strategy_info(strategy_info: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(strategy_info, dict):
        return {}

    row: dict[str, Any] = {}
    if "explanation_access_mode" in strategy_info:
        row["info_access_mode"] = strategy_info["explanation_access_mode"]
    if "simulation_sample_count" in strategy_info:
        row["info_simulation_sample_count"] = int(strategy_info["simulation_sample_count"])
    if "retrieval_candidate_count" in strategy_info:
        row["info_retrieval_candidate_count"] = int(strategy_info["retrieval_candidate_count"])

    timing = strategy_info.get("timing")
    if isinstance(timing, dict):
        for key in ["retrieval_rt_sum", "read_time_sum", "ddm_rt_mean", "total_time", "read_time", "retrieve_time", "decision_time", "expected_time"]:
            if key in timing:
                row[f"info_{key}"] = float(timing[key])

    decision = strategy_info.get("decision")
    if isinstance(decision, dict):
        for key in ["p1", "v_ratio_mean"]:
            if key in decision:
                row[f"info_decision_{key}"] = float(decision[key])

    ddm = strategy_info.get("ddm")
    if isinstance(ddm, dict):
        for key in ["p_up", "rt_dec", "v", "a", "s", "Tnd"]:
            if key in ddm:
                row[f"info_ddm_{key}"] = float(ddm[key])

    path = strategy_info.get("path")
    if isinstance(path, dict):
        for key in ["categorical_decisions", "numeric_decisions"]:
            if key in path:
                row[f"info_path_{key}"] = int(path[key])
        leaf_counts = path.get("leaf_counts")
        if isinstance(leaf_counts, dict) and leaf_counts:
            top_leaf, top_count = max(leaf_counts.items(), key=lambda item: item[1])
            row["info_path_top_leaf"] = int(top_leaf)
            row["info_path_top_leaf_count"] = int(top_count)
            row["info_path_n_leaves_reached"] = int(len(leaf_counts))

    refresh_counts = strategy_info.get("refresh_counts")
    if isinstance(refresh_counts, dict):
        for key, counts in refresh_counts.items():
            if isinstance(counts, dict):
                row[f"info_refresh_{key}_count"] = int(sum(int(v) for v in counts.values()))

    chunks = strategy_info.get("chunks")
    if isinstance(chunks, dict):
        features = chunks.get("features")
        if isinstance(features, list):
            row["info_n_feature_beliefs"] = int(len(features))

    if "avg_p_up" in strategy_info:
        row["info_avg_p_up"] = float(strategy_info["avg_p_up"])
    if "avg_time" in strategy_info:
        row["info_avg_time"] = float(strategy_info["avg_time"])
    if "ops_count" in strategy_info:
        row["info_ops_count"] = int(strategy_info["ops_count"])
    return row


def _feature_base_index(feature_key: str) -> int:
    return int(feature_key.split("=")[0][1:])


def _contribution_by_base_feature(lr_exp: LogisticRegressionInterpreter, x_raw: np.ndarray, max_features: int) -> np.ndarray:
    contributions = np.zeros(max_features, dtype=float)
    for feature_key, coefficient in lr_exp.coefficients.items():
        base_idx = _feature_base_index(feature_key)
        if base_idx >= max_features or base_idx >= len(x_raw):
            continue
        if "=" in feature_key:
            _base, category = feature_key.split("=")
            value = 1.0 if int(x_raw[base_idx]) == int(category) else 0.0
        else:
            value = float(x_raw[base_idx])
        contributions[base_idx] += float(coefficient) * value
    return contributions


def _normalized_abs(values: np.ndarray) -> np.ndarray:
    arr = np.abs(np.asarray(values, dtype=float))
    return arr / (float(arr.sum()) + 1e-9)


def make_memory(memory_recall_threshold: float, memory_recall_noise: float) -> CombinedMemory:
    dm = DeclarativeMemory(
        memory_recall_threshold=memory_recall_threshold,
        cue_association_strength=2.0,
        memory_mismatch_penalty=-2.0,
        memory_recall_noise=memory_recall_noise,
    )
    return CombinedMemory(dm, working_memory_capacity=7)


def load_strategy_bundles(data_dir: Path, config: StrategyTrainingConfig) -> list[StrategyBundle]:
    values_df = pd.read_csv(data_dir / "values.csv")
    metadata_df = pd.read_csv(data_dir / "metadata.csv")
    prediction_df = pd.read_csv(data_dir / "none.csv")
    lr_df = pd.read_csv(data_dir / "logistic_regression.csv")
    dt_df = pd.read_csv(data_dir / "decision_tree.csv")

    base_loader = AIDatasetLoader(values_df, metadata_df, prediction_df)
    app_models = prediction_df[["appId", "modelName"]].drop_duplicates()
    if config.apps:
        app_models = app_models[app_models["appId"].isin(config.apps)]

    bundles: list[StrategyBundle] = []
    for row in app_models.itertuples(index=False):
        app_id = str(row.appId)
        model_name = str(row.modelName)
        try:
            loader = filter_by_app_and_model(base_loader, app_id, model_name)
            lr_exp = LogisticRegressionInterpreter(lr_df, metadata_df, app_id, model_name)
            dt_exp = DecisionTreeInterpreter(dt_df, metadata_df, app_id, model_name, depth=config.dt_depth)
        except Exception as exc:
            print(f"Skipping {app_id}/{model_name}: {exc}")
            continue
        feature_ids = set(loader.feature_values_df["instanceId"].dropna().astype(int).tolist())
        labeled_prediction_rows = loader.AI_predictions_df[loader.AI_predictions_df["pred"].notna()]
        prediction_ids = set(labeled_prediction_rows["instanceId"].dropna().astype(int).tolist())
        instance_ids = tuple(sorted(feature_ids & prediction_ids))
        if instance_ids:
            bundles.append(StrategyBundle(app_id, model_name, loader, lr_exp, dt_exp, instance_ids))
    return bundles


class StrategyPolicyEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        bundles: list[StrategyBundle],
        config: StrategyTrainingConfig,
        *,
        training: bool = True,
        fixed_eval_params: dict[str, float] | None = None,
    ):
        super().__init__()
        if config.strategy_name not in STRATEGIES:
            raise ValueError(f"Unknown strategy_name={config.strategy_name!r}; expected one of {STRATEGIES}.")
        if not bundles:
            raise ValueError("StrategyPolicyEnv requires at least one StrategyBundle.")
        self.bundles = bundles
        self.config = config
        self.training = bool(training)
        self.fixed_eval_params = fixed_eval_params or {}

        if config.strategy_name == "dt_traversal":
            self.action_space = spaces.Discrete(config.decision_boundary_bins)
        elif config.strategy_name == "lr_heuristic":
            self.action_space = spaces.MultiDiscrete([config.decision_boundary_bins] + [2] * config.max_features)
        else:
            self.action_space = spaces.MultiDiscrete([2, config.decision_boundary_bins] + [2] * config.max_features)

        obs_dim = self._observation_dim()
        self.observation_space = spaces.Box(
            low=np.full(obs_dim, -1.0, dtype=np.float32),
            high=np.ones(obs_dim, dtype=np.float32),
            dtype=np.float32,
        )

        self.rng = np.random.default_rng(config.seed)
        self.bundle: StrategyBundle | None = None
        self.memory: CombinedMemory | None = None
        self.step_idx = 0
        self.current_params: dict[str, float] = {}
        self.explanation_schedule: np.ndarray | None = None
        self.X_raw: np.ndarray | None = None
        self.X_norm: np.ndarray | None = None
        self.y: np.ndarray | None = None
        self.current_contrib = np.zeros(config.max_features, dtype=float)
        self.contrib_history: list[np.ndarray] = []
        self.recent_prob_correct: deque[float] = deque(maxlen=config.history_window)
        self.recent_pred_time: deque[float] = deque(maxlen=config.history_window)
        self.feature_order = np.arange(config.max_features, dtype=int)

    def _observation_dim(self) -> int:
        common = 5 + 2 * self.config.history_window
        if self.config.strategy_name == "lr_calculation":
            return common + 3 * self.config.max_features
        if self.config.strategy_name == "dt_traversal":
            return common + self.config.max_features
        return common + 2 * self.config.max_features

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
            "memory_recall_threshold": self._sample_param(
                "memory_recall_threshold",
                "memory_recall_threshold_min",
                "memory_recall_threshold_max",
            ),
            "opportunity_cost": self._sample_param(
                "opportunity_cost",
                "opportunity_cost_min",
                "opportunity_cost_max",
            ),
        }

    def _decision_boundary_from_bin(self, bin_id: int) -> float:
        bin_id = int(np.clip(bin_id, 0, self.config.decision_boundary_bins - 1))
        if self.config.decision_boundary_bins <= 1:
            return float(self.config.decision_boundary_min)
        frac = bin_id / max(1, self.config.decision_boundary_bins - 1)
        return float(
            self.config.decision_boundary_min
            + frac * (self.config.decision_boundary_max - self.config.decision_boundary_min)
        )

    def _build_explanation_schedule(self) -> np.ndarray:
        n = self.config.instances_per_episode
        n_readable = int(round(n * self.config.explanation_shown_ratio))
        flags = np.array([1] * n_readable + [0] * (n - n_readable), dtype=bool)
        self.rng.shuffle(flags)
        return flags

    def _sample_instances(self, bundle: StrategyBundle) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        ids = np.asarray(bundle.instance_ids, dtype=int)
        replace = len(ids) < self.config.instances_per_episode
        chosen = self.rng.choice(ids, size=self.config.instances_per_episode, replace=replace).astype(int).tolist()
        raw_instances, labels = bundle.loader.load_instances(chosen, normalize=False)
        norm_instances, _ = bundle.loader.load_instances(chosen, normalize=True)
        if any(label is None or pd.isna(label) for label in labels):
            missing = [instance_id for instance_id, label in zip(chosen, labels) if label is None or pd.isna(label)]
            raise ValueError(f"Sampled unlabeled instance(s) for {bundle.app_id}/{bundle.model_name}: {missing[:10]}")
        return np.asarray(raw_instances, dtype=float), np.asarray(norm_instances, dtype=float), np.asarray(labels, dtype=int)

    def _initialize_memory(self) -> None:
        assert self.bundle is not None
        self.memory = make_memory(
            self.current_params["memory_recall_threshold"],
            self.config.memory_recall_noise,
        )
        if self.config.strategy_name == "lr_calculation":
            add_lr_calculation_to_memory(
                self.bundle.lr_exp,
                self.memory,
                intercept_significant_figures=self.config.displayed_significant_figures,
                coefficient_significant_figures=self.config.displayed_significant_figures,
            )
        elif self.config.strategy_name == "lr_heuristic":
            add_lr_heuristic_to_memory(self.bundle.lr_exp, self.memory)
        else:
            add_dt_to_memory(
                self.memory,
                self.bundle.dt_exp,
                threshold_significant_figures=self.config.displayed_significant_figures,
            )
        self.memory.tick(1.0)

    def _common_obs(self) -> np.ndarray:
        recent_prob = np.full(self.config.history_window, -1.0, dtype=np.float32)
        recent_time = np.full(self.config.history_window, -1.0, dtype=np.float32)
        for idx, value in enumerate(list(self.recent_prob_correct)[-self.config.history_window:]):
            recent_prob[idx] = float(value)
        for idx, value in enumerate(list(self.recent_pred_time)[-self.config.history_window:]):
            recent_time[idx] = float(np.clip(float(value) / 60.0, 0.0, 1.0))
        return np.array(
            [
                _normalize_scalar(
                    self.current_params["decision_noise"],
                    self.config.decision_noise_min,
                    self.config.decision_noise_max,
                ),
                _normalize_scalar(
                    self.current_params["memory_recall_threshold"],
                    self.config.memory_recall_threshold_min,
                    self.config.memory_recall_threshold_max,
                ),
                _normalize_scalar(
                    self.current_params["opportunity_cost"],
                    self.config.opportunity_cost_min,
                    self.config.opportunity_cost_max,
                ),
                float(self.explanation_schedule[self.step_idx]) if self.explanation_schedule is not None else 0.0,
                float(self.step_idx / max(1, self.config.instances_per_episode)),
                *recent_prob.tolist(),
                *recent_time.tolist(),
            ],
            dtype=np.float32,
        )

    def _contribution_obs(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        current_norm = _normalized_abs(self.current_contrib)
        if self.contrib_history:
            hist = np.vstack(self.contrib_history)
            means = _normalized_abs(hist.mean(axis=0))
            stds = _normalized_abs(hist.std(axis=0))
        else:
            means = np.zeros(self.config.max_features, dtype=float)
            stds = np.zeros(self.config.max_features, dtype=float)
        if self.config.randomize_feature_order_per_episode:
            current_norm = current_norm[self.feature_order]
            means = means[self.feature_order]
            stds = stds[self.feature_order]
        return current_norm, means, stds

    def _heuristic_memory_obs(self) -> tuple[np.ndarray, np.ndarray]:
        means = np.zeros(self.config.max_features, dtype=float)
        variances = np.ones(self.config.max_features, dtype=float)
        counts = np.zeros(self.config.max_features, dtype=float)
        assert self.memory is not None
        for chunk in self.memory.chunks:
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
            mean_obs = mean_obs[self.feature_order]
            std_obs = std_obs[self.feature_order]
        return mean_obs, std_obs

    def _get_obs(self) -> np.ndarray:
        if self.step_idx >= self.config.instances_per_episode or self.explanation_schedule is None:
            return np.zeros(self.observation_space.shape, dtype=np.float32)
        common = self._common_obs()
        if self.config.strategy_name == "lr_calculation":
            current_norm, means, stds = self._contribution_obs()
            obs = np.concatenate([common, current_norm, means, stds])
        elif self.config.strategy_name == "dt_traversal":
            current_norm, _means, _stds = self._contribution_obs()
            obs = np.concatenate([common, current_norm])
        else:
            means, stds = self._heuristic_memory_obs()
            obs = np.concatenate([common, means, stds])
        return obs.astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.bundle = self.rng.choice(self.bundles)
        self.current_params = self._sample_episode_params()
        self._initialize_memory()
        self.explanation_schedule = self._build_explanation_schedule()
        self.X_raw, self.X_norm, self.y = self._sample_instances(self.bundle)
        if self.config.randomize_feature_order_per_episode:
            self.feature_order = self.rng.permutation(self.config.max_features).astype(int)
        else:
            self.feature_order = np.arange(self.config.max_features, dtype=int)
        self.step_idx = 0
        self.contrib_history = []
        self.recent_prob_correct.clear()
        self.recent_pred_time.clear()
        self.current_contrib = _contribution_by_base_feature(
            self.bundle.lr_exp,
            self.X_raw[self.step_idx],
            self.config.max_features,
        )
        return self._get_obs(), {}

    def _parse_action(self, action) -> tuple[str, float, list[int], list[int], np.ndarray, np.ndarray, int]:
        action = np.asarray(action, dtype=int).reshape(-1)
        if self.config.strategy_name == "dt_traversal":
            boundary_bin = int(action[0])
            chosen_mode = "retrieve"
            selected_feature_slots: list[int] = []
        elif self.config.strategy_name == "lr_heuristic":
            boundary_bin = int(action[0])
            chosen_mode = "retrieve"
            selected_feature_slots = [int(i) for i, bit in enumerate(action[1 : 1 + self.config.max_features]) if bit == 1]
        else:
            chosen_mode = ACCESS_MODES.get(int(action[0]), "retrieve")
            boundary_bin = int(action[1])
            selected_feature_slots = [int(i) for i, bit in enumerate(action[2 : 2 + self.config.max_features]) if bit == 1]
        selected_features = [int(self.feature_order[i]) for i in selected_feature_slots]
        feature_mask = np.zeros(self.config.max_features, dtype=int)
        slot_mask = np.zeros(self.config.max_features, dtype=int)
        for slot in selected_feature_slots:
            slot_mask[slot] = 1
        for actual_idx in selected_features:
            if 0 <= actual_idx < self.config.max_features:
                feature_mask[actual_idx] = 1
        return chosen_mode, self._decision_boundary_from_bin(boundary_bin), selected_features, selected_feature_slots, feature_mask, slot_mask, boundary_bin

    def step(self, action):
        assert self.bundle is not None
        assert self.memory is not None
        assert self.X_raw is not None
        assert self.X_norm is not None
        assert self.y is not None
        assert self.explanation_schedule is not None

        chosen_mode, decision_boundary, selected_features, selected_feature_slots, feature_mask, slot_mask, boundary_bin = self._parse_action(action)
        with_explanation = bool(self.explanation_schedule[self.step_idx])
        x_raw = self.X_raw[self.step_idx]
        x_norm = self.X_norm[self.step_idx]
        y_true = int(self.y[self.step_idx])

        illegal_action = False
        strategy_info: dict[str, Any] = {}
        if self.config.strategy_name == "lr_calculation" and chosen_mode == "read" and not with_explanation:
            prob_correct = 0.0
            pred_time = 0.0
            reward = float(self.config.illegal_read_penalty)
            illegal_action = True
            probs = np.array([0.5, 0.5], dtype=float)
        else:
            if self.config.strategy_name == "lr_calculation":
                probs, pred_time, _info = lr_calculation(
                    x_raw,
                    self.memory,
                    self.bundle.lr_exp,
                    explanation_access_mode=chosen_mode,
                    displayed_significant_figures=self.config.displayed_significant_figures,
                    read_seconds_per_item=self.config.read_seconds_per_item,
                    mental_calculation_seconds=self.config.mental_calculation_seconds,
                    decision_boundary=decision_boundary,
                    decision_noise=self.current_params["decision_noise"],
                    selected_feature_indices=selected_features,
                    simulation_sample_count=self.config.simulation_sample_count,
                    retrieval_candidate_count=self.config.retrieval_candidate_count,
                )
                strategy_info = _info
            elif self.config.strategy_name == "lr_heuristic":
                probs, pred_time, _info = lr_heuristic(
                    x_norm,
                    self.memory,
                    self.bundle.lr_exp,
                    simulation_sample_count=self.config.simulation_sample_count,
                    retrieval_candidate_count=self.config.retrieval_candidate_count,
                    read_seconds_per_item=self.config.read_seconds_per_item,
                    mental_calculation_seconds=self.config.mental_calculation_seconds,
                    decision_boundary=decision_boundary,
                    decision_noise=self.current_params["decision_noise"],
                    selected_feature_indices=selected_features,
                )
                refresh_lr_heuristic_in_memory(
                    self.memory,
                    self.bundle.lr_exp,
                    _info,
                    y_true,
                    selected_feature_indices=selected_features,
                )
                strategy_info = _info
            else:
                probs, pred_time, _info = dt_traverse(
                    x_raw,
                    self.memory,
                    self.bundle.dt_exp,
                    explanation_access_mode="read" if with_explanation else "retrieve",
                    displayed_significant_figures=self.config.displayed_significant_figures,
                    read_seconds_per_item=self.config.read_seconds_per_item,
                    decision_boundary=decision_boundary,
                    decision_noise=self.current_params["decision_noise"],
                    simulation_sample_count=self.config.simulation_sample_count,
                    retrieval_candidate_count=self.config.retrieval_candidate_count,
                )
                strategy_info = _info
            probs = _safe_probabilities(probs)
            prob_correct = float(probs[y_true]) if y_true < len(probs) else 0.0
            reward = (
                prob_correct
                - self.current_params["opportunity_cost"] * float(pred_time)
                - self.config.feature_cost * float(len(selected_features))
            )

        self.recent_prob_correct.append(float(prob_correct))
        self.recent_pred_time.append(float(pred_time))
        predicted_class = int(np.argmax(probs)) if len(probs) else -1
        info = {
            "app_id": self.bundle.app_id,
            "model_name": self.bundle.model_name,
            "strategy_name": self.config.strategy_name,
            "chosen_mode": chosen_mode,
            "with_explanation": with_explanation,
            "true_label": y_true,
            "predicted_class": predicted_class,
            "predicted_correct": bool(predicted_class == y_true),
            "p_predicted": float(probs[predicted_class]) if 0 <= predicted_class < len(probs) else 0.0,
            **{f"p_class_{i}": float(p) for i, p in enumerate(probs)},
            "decision_boundary": decision_boundary,
            "decision_boundary_bin": int(boundary_bin),
            "decision_noise": self.current_params["decision_noise"],
            "memory_recall_threshold": self.current_params["memory_recall_threshold"],
            "opportunity_cost": self.current_params["opportunity_cost"],
            "prob_correct": prob_correct,
            "pred_time": float(pred_time),
            "recent_prob_correct_history": ",".join(f"{v:.6g}" for v in self.recent_prob_correct),
            "recent_pred_time_history": ",".join(f"{v:.6g}" for v in self.recent_pred_time),
            "reward_without_illegal_penalty": float(
                prob_correct
                - self.current_params["opportunity_cost"] * float(pred_time)
                - self.config.feature_cost * float(len(selected_features))
            ),
            "n_selected_features": len(selected_features),
            "selected_features": selected_features,
            "selected_feature_slots": selected_feature_slots,
            "feature_mask": feature_mask.astype(int).tolist(),
            "feature_slot_mask": slot_mask.astype(int).tolist(),
            "feature_order": self.feature_order.astype(int).tolist(),
            "illegal_action": illegal_action,
            **_flatten_strategy_info(strategy_info),
        }

        self.contrib_history.append(self.current_contrib.copy())
        self.step_idx += 1
        terminated = False
        truncated = self.step_idx >= self.config.instances_per_episode
        if not truncated:
            self.current_contrib = _contribution_by_base_feature(
                self.bundle.lr_exp,
                self.X_raw[self.step_idx],
                self.config.max_features,
            )
        return self._get_obs(), float(reward), terminated, truncated, info


def make_run_dir(config: StrategyTrainingConfig) -> Path:
    run_name = config.run_name or f"{config.strategy_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(config.output_root) / run_name / config.strategy_name
    for subdir in ["models", "logs", "plots", "metrics"]:
        (run_dir / subdir).mkdir(parents=True, exist_ok=True)
    return run_dir


def make_vec_env(bundles: list[StrategyBundle], config: StrategyTrainingConfig, n_envs: int, training: bool):
    def factory(rank: int):
        def _init():
            env = StrategyPolicyEnv(bundles, config, training=training)
            env.reset(seed=config.seed + rank)
            return Monitor(env)
        return _init

    return DummyVecEnv([factory(i) for i in range(n_envs)])


def train_strategy_policy(config: StrategyTrainingConfig) -> tuple[PPO, Path, list[StrategyBundle]]:
    run_dir = make_run_dir(config)
    bundles = load_strategy_bundles(Path(config.data_dir), config)
    if not bundles:
        raise RuntimeError(f"No bundles were loaded for strategy={config.strategy_name}.")

    (run_dir / "config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
    pd.DataFrame(
        [{"app_id": b.app_id, "model_name": b.model_name, "n_instances": len(b.instance_ids)} for b in bundles]
    ).to_csv(run_dir / "metrics" / "bundles.csv", index=False)

    train_env = make_vec_env(bundles, config, config.n_envs, training=True)
    eval_env = make_vec_env(bundles, config, 1, training=False)
    model = PPO(
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
    callback = EvalCallback(
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


def train_all_strategy_policies(base_config: StrategyTrainingConfig) -> dict[str, tuple[PPO, Path, list[StrategyBundle]]]:
    results = {}
    for strategy_name in STRATEGIES:
        config = StrategyTrainingConfig(**{**asdict(base_config), "strategy_name": strategy_name})
        results[strategy_name] = train_strategy_policy(config)
    return results


def resolve_strategy_model_path(
    config: StrategyTrainingConfig,
    strategy_name: str | None = None,
    *,
    model_filename: str = "final_model.zip",
) -> tuple[Path, Path]:
    strategy_name = strategy_name or config.strategy_name
    if strategy_name not in STRATEGIES:
        raise ValueError(f"Unknown strategy_name={strategy_name!r}; expected one of {STRATEGIES}.")

    output_root = Path(config.output_root)
    if config.run_name:
        run_dir = output_root / config.run_name / strategy_name
        return run_dir / "models" / model_filename, run_dir

    candidates = sorted(
        output_root.glob(f"*/{strategy_name}/models/{model_filename}"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No saved {model_filename} found below {output_root} for strategy={strategy_name}."
        )
    model_path = candidates[0]
    return model_path, model_path.parents[1]


def load_strategy_policy(
    config: StrategyTrainingConfig,
    *,
    model_path: str | Path | None = None,
    model_filename: str = "final_model.zip",
    device: str = "cpu",
) -> tuple[PPO, Path, list[StrategyBundle]]:
    resolved_model_path, run_dir = (
        (
            Path(model_path),
            Path(model_path).parent.parent if Path(model_path).parent.name == "models" else Path(model_path).parent,
        )
        if model_path is not None
        else resolve_strategy_model_path(config, model_filename=model_filename)
    )
    if not resolved_model_path.exists():
        raise FileNotFoundError(f"Saved model does not exist: {resolved_model_path}")
    bundles = load_strategy_bundles(Path(config.data_dir), config)
    if not bundles:
        raise RuntimeError(f"No bundles were loaded for strategy={config.strategy_name}.")
    model = PPO.load(resolved_model_path, device=device)
    return model, run_dir, bundles


def load_all_strategy_policies(
    base_config: StrategyTrainingConfig,
    *,
    strategies: tuple[str, ...] = STRATEGIES,
    model_filename: str = "final_model.zip",
    device: str = "cpu",
) -> dict[str, tuple[PPO, Path, list[StrategyBundle]]]:
    results = {}
    for strategy_name in strategies:
        config = StrategyTrainingConfig(**{**asdict(base_config), "strategy_name": strategy_name})
        results[strategy_name] = load_strategy_policy(
            config,
            model_filename=model_filename,
            device=device,
        )
    return results


def evaluate_strategy_policy(
    model: PPO,
    bundles: list[StrategyBundle],
    config: StrategyTrainingConfig,
    *,
    n_episodes: int = 20,
    deterministic: bool = True,
    fixed_eval_params: dict[str, float] | None = None,
    sample_parameters: bool = True,
) -> pd.DataFrame:
    env = StrategyPolicyEnv(
        bundles,
        config,
        training=sample_parameters,
        fixed_eval_params=fixed_eval_params,
    )
    rows: list[dict[str, Any]] = []
    for episode in range(n_episodes):
        obs, _ = env.reset(seed=config.seed + 10_000 + episode)
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            rows.append(
                {
                    "episode": episode,
                    "step": env.step_idx - 1,
                    "reward": float(reward),
                    **{k: v for k, v in info.items() if k not in {"selected_features", "selected_feature_slots", "feature_mask", "feature_slot_mask", "feature_order"}},
                    "selected_features": ",".join(str(i) for i in info.get("selected_features", [])),
                    "selected_feature_slots": ",".join(str(i) for i in info.get("selected_feature_slots", [])),
                    "feature_order": ",".join(str(i) for i in info.get("feature_order", [])),
                    **{f"feature_{i}_selected": int(bit) for i, bit in enumerate(info.get("feature_mask", []))},
                    **{f"slot_{i}_selected": int(bit) for i, bit in enumerate(info.get("feature_slot_mask", []))},
                }
            )
    env.close()
    return pd.DataFrame(rows)


def summarize_strategy_evaluations(evaluation_tables: dict[str, pd.DataFrame] | pd.DataFrame) -> pd.DataFrame:
    if isinstance(evaluation_tables, pd.DataFrame):
        frames = [evaluation_tables]
    else:
        frames = []
        for strategy_name, df in evaluation_tables.items():
            if df is None or df.empty:
                continue
            frame = df.copy()
            if "strategy_name" not in frame.columns:
                frame["strategy_name"] = strategy_name
            frames.append(frame)
    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    group_cols = ["strategy_name"]
    if "chosen_mode" in combined.columns:
        group_cols.append("chosen_mode")
    if "with_explanation" in combined.columns:
        group_cols.append("with_explanation")

    agg = {
        "reward": "mean",
        "prob_correct": "mean",
        "pred_time": "mean",
        "n_selected_features": "mean",
        "decision_boundary": "mean",
    }
    if "predicted_correct" in combined.columns:
        agg["predicted_correct"] = "mean"
    if "illegal_action" in combined.columns:
        agg["illegal_action"] = "mean"
    summary = combined.groupby(group_cols, dropna=False).agg(agg).reset_index()
    summary = summary.rename(
        columns={
            "reward": "mean_reward",
            "prob_correct": "mean_prob_correct",
            "pred_time": "mean_pred_time",
            "n_selected_features": "mean_selected_features",
            "decision_boundary": "mean_decision_boundary",
            "predicted_correct": "accuracy",
            "illegal_action": "illegal_action_rate",
        }
    )
    counts = combined.groupby(group_cols, dropna=False).size().reset_index(name="n_rows")
    return counts.merge(summary, on=group_cols, how="left")


def plot_strategy_evaluations(
    evaluation_tables: dict[str, pd.DataFrame] | pd.DataFrame,
    *,
    output_dir: str | Path | None = None,
    rolling_window: int = 20,
    show: bool = True,
) -> dict[str, Any]:
    import matplotlib.pyplot as plt

    if isinstance(evaluation_tables, pd.DataFrame):
        combined = evaluation_tables.copy()
    else:
        frames = []
        for strategy_name, df in evaluation_tables.items():
            if df is None or df.empty:
                continue
            frame = df.copy()
            if "strategy_name" not in frame.columns:
                frame["strategy_name"] = strategy_name
            frames.append(frame)
        combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if combined.empty:
        return {}

    output_path = Path(output_dir) if output_dir is not None else None
    if output_path is not None:
        output_path.mkdir(parents=True, exist_ok=True)

    figures: dict[str, Any] = {}
    metrics = [
        ("prob_correct", "P(correct)"),
        ("reward", "Reward"),
        ("pred_time", "Prediction time"),
        ("n_selected_features", "Selected features"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (column, label) in zip(axes.ravel(), metrics):
        if column not in combined.columns:
            ax.axis("off")
            continue
        data = combined.groupby("strategy_name")[column].mean().reindex(STRATEGIES)
        data.dropna().plot(kind="bar", ax=ax, color=["#4c78a8", "#f58518", "#54a24b"])
        ax.set_title(label)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    figures["summary_bars"] = fig
    if output_path is not None:
        fig.savefig(output_path / "strategy_summary_bars.png", dpi=160, bbox_inches="tight")

    fig, ax = plt.subplots(figsize=(12, 5))
    for strategy_name, df in combined.groupby("strategy_name"):
        ordered = df.sort_values(["episode", "step"]).reset_index(drop=True)
        ordered["rolling_prob_correct"] = ordered["prob_correct"].rolling(rolling_window, min_periods=1).mean()
        ax.plot(ordered.index, ordered["rolling_prob_correct"], label=strategy_name)
    ax.set_title(f"Rolling P(correct), window={rolling_window}")
    ax.set_xlabel("Evaluation step")
    ax.set_ylabel("P(correct)")
    ax.legend()
    fig.tight_layout()
    figures["rolling_prob_correct"] = fig
    if output_path is not None:
        fig.savefig(output_path / "strategy_rolling_prob_correct.png", dpi=160, bbox_inches="tight")

    if "decision_boundary_bin" in combined.columns:
        fig, ax = plt.subplots(figsize=(10, 5))
        boundary_counts = pd.crosstab(combined["decision_boundary_bin"], combined["strategy_name"], normalize="columns")
        boundary_counts.reindex(columns=[s for s in STRATEGIES if s in boundary_counts.columns]).plot(kind="bar", ax=ax)
        ax.set_title("Decision-boundary choices")
        ax.set_xlabel("Boundary bin")
        ax.set_ylabel("Share of actions")
        ax.tick_params(axis="x", rotation=0)
        fig.tight_layout()
        figures["decision_boundary_bins"] = fig
        if output_path is not None:
            fig.savefig(output_path / "strategy_decision_boundary_bins.png", dpi=160, bbox_inches="tight")

    feature_cols = [c for c in combined.columns if c.startswith("feature_") and c.endswith("_selected")]
    if feature_cols:
        feature_rates = (
            combined.groupby("strategy_name")[feature_cols]
            .mean()
            .rename(columns=lambda c: c.replace("feature_", "a").replace("_selected", ""))
        )
        fig, ax = plt.subplots(figsize=(12, 4))
        feature_rates.reindex([s for s in STRATEGIES if s in feature_rates.index]).plot(kind="bar", ax=ax)
        ax.set_title("Feature selection rate")
        ax.set_xlabel("")
        ax.set_ylabel("Selection rate")
        ax.tick_params(axis="x", rotation=20)
        fig.tight_layout()
        figures["feature_selection"] = fig
        if output_path is not None:
            fig.savefig(output_path / "strategy_feature_selection.png", dpi=160, bbox_inches="tight")

    if not show:
        for fig in figures.values():
            plt.close(fig)
    return figures


def default_parameter_sweep_values(config: StrategyTrainingConfig, n_points: int = 5) -> dict[str, list[float]]:
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


def sweep_strategy_parameters(
    strategy_results: dict[str, tuple[PPO, Path, list[StrategyBundle]]],
    base_config: StrategyTrainingConfig,
    *,
    sweep_values: dict[str, list[float]] | None = None,
    n_episodes: int = 25,
    deterministic: bool = True,
) -> pd.DataFrame:
    sweep_values = sweep_values or default_parameter_sweep_values(base_config)
    frames: list[pd.DataFrame] = []
    for parameter_name, values in sweep_values.items():
        if parameter_name not in {
            "decision_noise",
            "memory_recall_threshold",
            "opportunity_cost",
        }:
            raise ValueError(
                "Supported sweep parameters are decision_noise, memory_recall_threshold, and opportunity_cost; "
                f"got {parameter_name!r}."
            )
        for parameter_value in values:
            fixed_eval_params = {parameter_name: float(parameter_value)}
            for strategy_name, (model, _run_dir, bundles) in strategy_results.items():
                config = StrategyTrainingConfig(**{**asdict(base_config), "strategy_name": strategy_name})
                eval_df = evaluate_strategy_policy(
                    model,
                    bundles,
                    config,
                    n_episodes=n_episodes,
                    deterministic=deterministic,
                    fixed_eval_params=fixed_eval_params,
                    sample_parameters=False,
                )
                eval_df["sweep_parameter"] = parameter_name
                eval_df["sweep_value"] = float(parameter_value)
                eval_df["strategy_name"] = strategy_name
                frames.append(eval_df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def summarize_parameter_sweep(sweep_df: pd.DataFrame) -> pd.DataFrame:
    if sweep_df.empty:
        return pd.DataFrame()
    group_cols = ["sweep_parameter", "sweep_value", "strategy_name"]
    agg = {
        "reward": "mean",
        "prob_correct": "mean",
        "pred_time": "mean",
        "n_selected_features": "mean",
        "decision_boundary": "mean",
    }
    if "predicted_correct" in sweep_df.columns:
        agg["predicted_correct"] = "mean"
    if "illegal_action" in sweep_df.columns:
        agg["illegal_action"] = "mean"
    summary = sweep_df.groupby(group_cols, dropna=False).agg(agg).reset_index()
    summary = summary.rename(
        columns={
            "reward": "mean_reward",
            "prob_correct": "mean_prob_correct",
            "pred_time": "mean_pred_time",
            "n_selected_features": "mean_selected_features",
            "decision_boundary": "mean_decision_boundary",
            "predicted_correct": "accuracy",
            "illegal_action": "illegal_action_rate",
        }
    )
    counts = sweep_df.groupby(group_cols, dropna=False).size().reset_index(name="n_rows")
    return counts.merge(summary, on=group_cols, how="left")


def feature_selection_sweep_summary(sweep_df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [c for c in sweep_df.columns if c.startswith("feature_") and c.endswith("_selected")]
    if sweep_df.empty or not feature_cols:
        return pd.DataFrame()
    long_df = sweep_df.melt(
        id_vars=["sweep_parameter", "sweep_value", "strategy_name"],
        value_vars=feature_cols,
        var_name="feature",
        value_name="selected",
    )
    long_df["feature"] = long_df["feature"].str.replace("feature_", "a", regex=False).str.replace("_selected", "", regex=False)
    return (
        long_df.groupby(["sweep_parameter", "sweep_value", "strategy_name", "feature"], dropna=False)["selected"]
        .mean()
        .reset_index(name="selection_rate")
    )


def plot_parameter_sweep(
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

    figures: dict[str, Any] = {}
    summary = summarize_parameter_sweep(sweep_df)
    metrics = [
        ("accuracy", "Accuracy"),
        ("mean_prob_correct", "P(correct)"),
        ("mean_reward", "Reward"),
        ("mean_pred_time", "Prediction time"),
        ("mean_selected_features", "Selected features"),
        ("mean_decision_boundary", "Decision boundary"),
    ]
    for parameter_name, param_df in summary.groupby("sweep_parameter"):
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        for ax, (column, label) in zip(axes.ravel(), metrics):
            if column not in param_df.columns:
                ax.axis("off")
                continue
            for strategy_name, strategy_df in param_df.groupby("strategy_name"):
                strategy_df = strategy_df.sort_values("sweep_value")
                ax.plot(strategy_df["sweep_value"], strategy_df[column], marker="o", label=strategy_name)
            ax.set_title(label)
            ax.set_xlabel(parameter_name)
            ax.grid(True, alpha=0.25)
        handles, labels = axes.ravel()[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 3))
        fig.suptitle(f"Strategy behavior across {parameter_name}", y=1.02)
        fig.tight_layout()
        figures[f"{parameter_name}_metrics"] = fig
        if output_path is not None:
            fig.savefig(output_path / f"sweep_{parameter_name}_metrics.png", dpi=160, bbox_inches="tight")

    feature_summary = feature_selection_sweep_summary(sweep_df)
    for (parameter_name, strategy_name), group in feature_summary.groupby(["sweep_parameter", "strategy_name"]):
        pivot = (
            group.pivot_table(index="feature", columns="sweep_value", values="selection_rate", aggfunc="mean")
            .sort_index()
        )
        if pivot.empty:
            continue
        fig, ax = plt.subplots(figsize=(max(7, 1.1 * len(pivot.columns)), max(3, 0.45 * len(pivot.index))))
        image = ax.imshow(pivot.values, aspect="auto", vmin=0.0, vmax=1.0, cmap="viridis")
        ax.set_title(f"{strategy_name}: feature selection across {parameter_name}")
        ax.set_xlabel(parameter_name)
        ax.set_ylabel("Feature")
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels([f"{v:g}" for v in pivot.columns], rotation=30, ha="right")
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        fig.colorbar(image, ax=ax, label="Selection rate")
        fig.tight_layout()
        key = f"{parameter_name}_{strategy_name}_feature_selection"
        figures[key] = fig
        if output_path is not None:
            fig.savefig(output_path / f"sweep_{parameter_name}_{strategy_name}_feature_selection.png", dpi=160, bbox_inches="tight")

    if not show:
        for fig in figures.values():
            plt.close(fig)
    return figures


def strategy_observation_action_summary(config: StrategyTrainingConfig) -> pd.DataFrame:
    rows = [
        {
            "strategy_name": "lr_calculation",
            "observation_space": f"Box({5 + 2 * config.history_window + 3 * config.max_features}, low=-1, high=1)",
            "observation_contents": f"5 context values + previous {config.history_window} probabilities + previous {config.history_window} times + current contribution norm + contribution mean norm + contribution std norm",
            "action_space": f"MultiDiscrete([2, {config.decision_boundary_bins}, " + ", ".join(["2"] * config.max_features) + "])",
            "action_contents": "read/retrieve mode, decision-boundary bin, shuffled feature-slot mask",
        },
        {
            "strategy_name": "lr_heuristic",
            "observation_space": f"Box({5 + 2 * config.history_window + 2 * config.max_features}, low=-1, high=1)",
            "observation_contents": f"5 context values + previous {config.history_window} probabilities + previous {config.history_window} times + memory coefficient means + memory coefficient stds",
            "action_space": f"MultiDiscrete([{config.decision_boundary_bins}, " + ", ".join(["2"] * config.max_features) + "])",
            "action_contents": "decision-boundary bin, shuffled feature-slot mask; no read/retrieve mode",
        },
        {
            "strategy_name": "dt_traversal",
            "observation_space": f"Box({5 + 2 * config.history_window + config.max_features}, low=-1, high=1)",
            "observation_contents": f"5 context values + previous {config.history_window} probabilities + previous {config.history_window} times + current contribution norm only",
            "action_space": f"Discrete({config.decision_boundary_bins})",
            "action_contents": "decision-boundary bin only",
        },
    ]
    return pd.DataFrame(rows)

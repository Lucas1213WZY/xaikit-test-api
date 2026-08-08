"""CoXAM's counterfactual-simulation environment, ported out of the notebook.

``counterfactual_simulation_v0.3.ipynb`` defines ``CounterfactualEnv`` and its
helpers only inside notebook cells, so nothing could import them. This module is
a faithful port of cells 9, 11, 12, 13, 16 and 18 of that notebook, with three
deliberate adaptations, each marked ADAPTED below:

1. The AI model is injected as a ``predict_fn`` callable rather than the
   notebook's ``(transform, ai)`` pair, so a study's own trained model can score
   counterfactuals.
2. ``DEFAULT_TIME_PENALTY_WEIGHT_RANGE`` is ``(0.0, 0.02)`` -- the range the
   shipped checkpoint was trained on -- not the ``(0.0, 0.05)`` in the
   notebook's current ``training_cog_params``, which post-dates the weights.
   Feeding a chi outside the trained range puts the observation outside the
   policy's ``observation_space``.
3. Instance ids come from the bundle rather than a hardcoded ``range(400)``.

Two behaviours are load-bearing and easy to break, so they are noted here:

* ``cf_change_path_dt`` only writes its ``dt_change_combo`` chunk in
  ``"retrieve"`` mode -- it returns early in ``"read"`` mode. The
  ``recall_change_*`` strategies read that chunk, so they can only recall what
  was reasoned about without the explanation on screen. This matches the
  notebook's own "can't recall if you saw it" rule.
* A single prior counterfactual trial is not enough for ``recall_change_*`` to
  retrieve anything: the chunk needs several presentations to clear its
  base-level activation, independent of the recall threshold. The forward-trial
  priming in ``initialize_memories`` is what makes recall viable, so it must not
  be skipped.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np

# stable_baselines3/gymnasium pull in a second OpenMP runtime on macOS conda
# builds, which aborts the process unless this is set before they load.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import gymnasium as gym  # noqa: E402
from gymnasium import spaces  # noqa: E402

from .memory import CombinedMemory, DeclarativeMemory  # noqa: E402
from .dt_memory import (  # noqa: E402
    add_dt_to_memory,
    cf_change_path_dt,
    dt_traverse,
    recall_change_dt,
    refresh_dt_path_in_memory,
)
from .heuristic_lr_model import (  # noqa: E402
    add_lr_heuristic_to_memory,
    cf_lr_heuristic,
    lr_heuristic,
    refresh_lr_heuristic_in_memory,
)
from .lr_memory import (  # noqa: E402
    add_lr_calculation_to_memory,
    cf_lr_calculation,
    recall_change_lr,
)


#: Action index -> strategy, exactly the notebook's ``strategies`` dict. The
#: names are the notebook's own wrappers; each calls a current ``cf_*``/
#: ``recall_change_*`` function (see the wrappers below).
STRATEGIES: dict[int, str] = {
    0: "change_path_dt",
    1: "zero_out_lr_heuristic",
    2: "zero_out_lr_displayed",
    3: "recall_change_dt",
    4: "recall_change_lr",
}

#: Condition index -> label, the notebook's ``XAI_types``. Maps onto a design's
#: ``xai_type``: decision_tree -> DT, logistic_regression -> LR, hybrid -> DT+LR.
XAI_TYPES: dict[int, str] = {0: "DT", 1: "LR", 2: "DT+LR"}

#: ADAPTED (2): the range the shipped checkpoint was trained on.
DEFAULT_TIME_PENALTY_WEIGHT_RANGE = (0.0, 0.02)

#: The three parameters the policy observes, in the order it expects them.
DEFAULT_COGNITIVE_PARAMS: dict[str, Any] = {
    "memory_recall_threshold": [-2.0, 0.5],
    "random_response_rate": [0.1, 0.5],
    "counterfactual_overshoot_fraction": [0.0, 0.5],
    "time_penalty_weight": list(DEFAULT_TIME_PENALTY_WEIGHT_RANGE),
}


def make_memory(memory_recall_threshold: float) -> CombinedMemory:
    """Notebook cell 9's ``_make_memory``."""
    return CombinedMemory(
        DeclarativeMemory(
            memory_recall_threshold=memory_recall_threshold,
            cue_association_strength=2.0,
            memory_mismatch_penalty=-2.0,
            memory_recall_noise=0.3,
        ),
        working_memory_capacity=7,
    )


def sample_from_probs(probs: Mapping[str, Any], rng: Optional[np.random.Generator] = None):
    """Notebook cell 12: pick a feature by ``p_selected``, return (feature, delta).

    Returns ``(None, 0.0)`` when nothing is feasible -- which happens whenever a
    ``recall_change_*`` strategy has no retrievable combo chunk yet.
    """
    rng = rng or np.random.default_rng()
    features = [k for k in probs if k != "expected_time"]
    weights = np.asarray(
        [float(probs[k].get("p_selected", 0.0)) for k in features], dtype=float
    )
    total = weights.sum()
    if not features or total <= 0.0:
        return None, 0.0
    chosen = str(rng.choice(features, p=weights / total))
    return chosen, float(probs[chosen]["mean_delta"])


def smooth_probs_with_random_response(
    probs: Mapping[str, Any],
    random_response_rate: float,
    num_features: int = 6,
) -> dict[str, Any]:
    """Notebook cell 12: mix the strategy's own distribution with a uniform guess."""
    smoothed = {k: dict(v) if isinstance(v, dict) else v for k, v in probs.items()}
    for index in range(num_features):
        key = f"a{index}"
        info = probs.get(key)
        if not isinstance(info, dict):
            continue
        p = float(info.get("p_selected", 0.0))
        smoothed[key]["p_selected"] = (
            (1.0 - random_response_rate) * p + random_response_rate / num_features
        )
    return smoothed


def apply_change_to_feature(
    instance: np.ndarray,
    feature_name: str,
    bounds: Mapping[str, tuple[float, float]],
    delta: float,
    counterfactual_overshoot_fraction: float = 0.1,
) -> np.ndarray:
    """Notebook cell 12: apply the chosen delta, overshoot it, clamp to bounds."""
    changed = np.asarray(instance, dtype=float).copy()
    base_key = feature_name.split("=")[0]
    index = int(base_key[1:])
    low, high = bounds.get(base_key, (-np.inf, np.inf))
    overshoot = (high - low) * counterfactual_overshoot_fraction
    overshoot = -overshoot if delta < 0 else overshoot
    changed[index] = min(max(changed[index] + delta + overshoot, low), high)
    return changed


# -- the five strategies (notebook cell 13) --------------------------------
# Each is a thin wrapper over a current cf_*/recall_change_* function: it runs
# the strategy, advances memory by the elapsed time, smooths the distribution
# with the random-response rate, then samples one (feature, delta).


def _finish(out, memory, random_response_rate, rng):
    time_taken = float(out.pop("expected_time", 0.0))
    memory.tick(time_taken)
    feature, delta = sample_from_probs(
        smooth_probs_with_random_response(out, random_response_rate), rng
    )
    return feature, delta, time_taken


def change_dt_path(instance, memory, dt_exp, bounds, actual_label, *, depth=1,
                   explanation_access_mode="retrieve", random_response_rate=0.1, rng=None):
    out = cf_change_path_dt(
        instance, dt_exp, bounds, memory=memory, counterfactual_tree_depth=depth,
        explanation_access_mode=explanation_access_mode, depth_choice_temperature=0.2,
    )
    feature, delta, time_taken = _finish(out, memory, random_response_rate, rng)
    if dt_exp.apply_to_instance(instance)["class_index"] != actual_label:
        delta = -delta
    return feature, delta, time_taken


def zero_out_lr_heuristic(norm_instance, memory, lr_exp, bounds, actual_label, *,
                          random_response_rate=0.1, rng=None):
    out = cf_lr_heuristic(
        norm_instance, memory, lr_exp, bounds, actual_label=actual_label,
        retrieval_candidate_count=6,
    )
    return _finish(out, memory, random_response_rate, rng)


def zero_out_lr_displayed(instance, memory, lr_exp, bounds, actual_label, *,
                          random_response_rate=0.1, rng=None):
    out = cf_lr_calculation(instance, lr_exp, bounds=bounds, memory=memory)
    feature, delta, time_taken = _finish(out, memory, random_response_rate, rng)
    if int(lr_exp.apply_to_instance(instance) > 0) != actual_label:
        delta = -delta
    return feature, delta, time_taken


def recall_change_dt_full(instance, memory, bounds, *, random_response_rate=0.1, rng=None):
    out = recall_change_dt(instance, memory, bounds=bounds, retrieved_combo_count=3)
    return _finish(out, memory, random_response_rate, rng)


def recall_change_lr_full(instance, memory, bounds, actual_label, *,
                          random_response_rate=0.1, rng=None):
    out = recall_change_lr(
        memory, retrieved_combo_count=6,
        preferred_change_direction="increase" if actual_label == 0 else "decrease",
    )
    return _finish(out, memory, random_response_rate, rng)


# -- forward-trial priming (notebook cell 11) ------------------------------


def prepare_memory_for_dt(memory, dt_exp, loader, forward_trials) -> None:
    """Run the forward phase so the DT memory has something to recall later."""
    add_dt_to_memory(memory, dt_exp)
    memory.tick(90)
    for trial in forward_trials:
        instances, _ = loader.load_instances([int(trial["Instance Id"])], normalize=False)
        mode = "read" if int(trial["Tested w/ XAI"]) == 1 else "retrieve"
        dt_traverse(instances[0], memory, dt_exp, explanation_access_mode=mode,
                    read_seconds_per_item=1.0, decision_boundary=1.0, decision_noise=0.8)
        if mode == "read":
            refresh_dt_path_in_memory(memory, dt_exp, instances[0])


def prepare_memory_for_lr_heuristic(memory, lr_exp, loader, forward_trials) -> None:
    """Run the forward phase so the LR memory has something to recall later."""
    add_lr_heuristic_to_memory(lr_exp, memory)
    memory.tick(90)
    for trial in forward_trials:
        instances, _ = loader.load_instances([int(trial["Instance Id"])], normalize=True)
        _probs, _time, info = lr_heuristic(
            instances[0], memory, lr_exp, read_seconds_per_item=1.0,
            decision_boundary=1.0, decision_noise=0.8,
        )
        refresh_lr_heuristic_in_memory(memory, lr_exp, info, trial["AI prediction"])


def strategy_is_legal(strategy_name: str, condition: str, with_xai: int) -> bool:
    """The notebook's own legality rules from ``step``.

    A DT condition cannot run LR strategies and vice versa, and the
    "displayed" LR strategy needs the explanation actually on screen.
    """
    if condition == "DT" and strategy_name in {
        "zero_out_lr_displayed", "zero_out_lr_heuristic", "recall_change_lr"
    }:
        return False
    if condition == "LR":
        if strategy_name in {"change_path_dt", "recall_change_dt"}:
            return False
        if strategy_name == "zero_out_lr_displayed" and int(with_xai) == 0:
            return False
    return True


class CounterfactualPolicyEnv(gym.Env):
    """Faithful port of the notebook's ``CounterfactualEnv``.

    ``predict_fn`` (ADAPTED (1)) takes a raw feature vector and returns the AI
    model's predicted label; counterfactual success is whether changing the
    chosen feature flips *that* prediction, not the surrogate's.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        loader: Any,
        lr_exp: Any,
        dt_exp: Any,
        bounds: Mapping[str, tuple[float, float]],
        predict_fn: Callable[[np.ndarray], int],
        instance_ids: Sequence[int],
        cognitive_params: Optional[Mapping[str, Any]] = None,
        instances_per_episode: int = 40,
        max_features: int = 6,
    ) -> None:
        super().__init__()
        self.loader = loader
        self.lr_exp = lr_exp
        self.dt_exp = dt_exp
        self.bounds = dict(bounds)
        self.predict_fn = predict_fn
        self.instance_ids = [int(value) for value in instance_ids]
        self.cognitive_params = dict(cognitive_params or DEFAULT_COGNITIVE_PARAMS)
        self.instances_per_episode = int(instances_per_episode)
        self.max_features = int(max_features)

        self.chi_low, self.chi_high = self.cognitive_params.get(
            "time_penalty_weight", DEFAULT_TIME_PENALTY_WEIGHT_RANGE
        )

        self.action_space = spaces.MultiDiscrete([len(STRATEGIES), 3])

        self.varied_param_names: list[str] = []
        low_params: list[float] = []
        high_params: list[float] = []
        for name, value in self.cognitive_params.items():
            if name == "time_penalty_weight":
                continue
            if isinstance(value, (list, tuple)) and len(value) == 2:
                self.varied_param_names.append(name)
                low_params.append(float(value[0]))
                high_params.append(float(value[1]))

        low = [self.chi_low, 0.0, 0.0, 0.0, 0.0] + [0.0] * 3 * len(STRATEGIES) + low_params
        high = (
            [self.chi_high, float(self.instances_per_episode - 1), 1.0,
             float(len(XAI_TYPES) - 1), 1.0]
            + [float(self.instances_per_episode), 1.0, 30.0] * len(STRATEGIES)
            + high_params
        )
        self.observation_space = spaces.Box(
            low=np.array(low, dtype=np.float32),
            high=np.array(high, dtype=np.float32),
            dtype=np.float32,
        )

        self.step_idx = 0
        self.curr_chi = 0.0
        self.current_cognitive_params: dict[str, float] = {}
        self.dt_memory: Optional[CombinedMemory] = None
        self.lr_memory: Optional[CombinedMemory] = None
        self.condition_index = 0
        self.xai_schedule: list[str] = []
        self.with_xai_schedule: list[int] = []
        self.forward_trials: list[dict[str, Any]] = []
        self.counterfactual_trials: list[dict[str, Any]] = []
        self.counts = {key: 0 for key in STRATEGIES}
        self.success_rates = {key: 0.0 for key in STRATEGIES}
        self.mean_times = {key: 0.0 for key in STRATEGIES}
        self.shown_xai_type = "DT"

    # -- episode setup ----------------------------------------------------

    def resolve_cognitive_params(self, rng: np.random.Generator,
                                 fixed: Optional[Mapping[str, float]] = None) -> None:
        self.current_cognitive_params = {}
        if fixed:
            for name, value in fixed.items():
                self.current_cognitive_params[name] = float(value)
            if "time_penalty_weight" in fixed:
                self.curr_chi = float(fixed["time_penalty_weight"])
            else:
                self.curr_chi = float(rng.uniform(self.chi_low, self.chi_high))
        else:
            for name, value in self.cognitive_params.items():
                if isinstance(value, (list, tuple)) and len(value) == 2:
                    self.current_cognitive_params[name] = float(rng.uniform(value[0], value[1]))
                elif isinstance(value, (int, float)):
                    self.current_cognitive_params[name] = float(value)
            self.curr_chi = float(rng.uniform(self.chi_low, self.chi_high))

    def initialize_memories(self) -> None:
        """Prime both memories with the forward phase.

        Not optional: ``recall_change_*`` retrieves chunks that only exist once
        enough forward/counterfactual reasoning has happened.
        """
        threshold = self.current_cognitive_params.get("memory_recall_threshold", -1.0)
        self.dt_memory = make_memory(threshold)
        self.lr_memory = make_memory(threshold)
        prepare_memory_for_dt(self.dt_memory, self.dt_exp, self.loader, self.forward_trials)
        prepare_memory_for_lr_heuristic(self.lr_memory, self.lr_exp, self.loader,
                                        self.forward_trials)

    def build_observation(self) -> np.ndarray:
        if self.step_idx >= self.instances_per_episode:
            return np.zeros(self.observation_space.shape, dtype=np.float32)
        shown_index = next(
            key for key, label in XAI_TYPES.items() if label == self.shown_xai_type
        )
        obs = [
            self.curr_chi,
            float(self.step_idx),
            float(self.with_xai_schedule[self.step_idx]),
            float(self.condition_index),
            float(shown_index),
        ]
        for key in STRATEGIES:
            obs += [float(self.counts[key]), float(self.success_rates[key]),
                    float(self.mean_times[key])]
        for name in self.varied_param_names:
            obs.append(float(self.current_cognitive_params[name]))
        return np.asarray(obs, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng = np.random.default_rng(seed)
        options = options or {}

        self.condition_index = int(options.get("condition_index", rng.integers(len(XAI_TYPES))))
        condition = XAI_TYPES[self.condition_index]
        if condition == "DT":
            self.xai_schedule = ["DT"] * self.instances_per_episode
        elif condition == "LR":
            self.xai_schedule = ["LR"] * self.instances_per_episode
        else:
            self.xai_schedule = rng.choice(
                ["DT", "LR"], size=self.instances_per_episode, p=[0.5, 0.5]
            ).tolist()
        if "xai_schedule" in options:
            self.xai_schedule = list(options["xai_schedule"])

        self.with_xai_schedule = list(
            options.get("with_xai_schedule",
                        rng.choice([0, 1], size=self.instances_per_episode).tolist())
        )

        pool = np.asarray(self.instance_ids, dtype=int)
        replace = len(pool) < self.instances_per_episode
        forward_ids = options.get(
            "forward_instance_ids",
            rng.choice(pool, size=self.instances_per_episode, replace=replace).tolist(),
        )
        instances, _ = self.loader.load_instances([int(i) for i in forward_ids], normalize=False)
        self.forward_trials = [
            {
                "Tested w/ XAI": int(self.with_xai_schedule[i]),
                "Instance Id": int(forward_ids[i]),
                "AI prediction": int(self.predict_fn(np.asarray(instances[i], dtype=float))),
            }
            for i in range(self.instances_per_episode)
        ]

        counterfactual_ids = options.get(
            "counterfactual_instance_ids",
            rng.choice(pool, size=self.instances_per_episode, replace=replace).tolist(),
        )
        self.counterfactual_trials = [
            {"Tested w/ XAI": int(self.with_xai_schedule[i]),
             "Instance Id": int(counterfactual_ids[i])}
            for i in range(self.instances_per_episode)
        ]

        self.resolve_cognitive_params(rng, options.get("cognitive_params_fixed"))
        self.initialize_memories()

        self.step_idx = 0
        self.counts = {key: 0 for key in STRATEGIES}
        self.success_rates = {key: 0.0 for key in STRATEGIES}
        self.mean_times = {key: 0.0 for key in STRATEGIES}
        self.shown_xai_type = self.xai_schedule[0]
        return self.build_observation(), {}

    # -- one trial --------------------------------------------------------

    def step(self, action):
        strategy_index, depth = int(np.asarray(action).reshape(-1)[0]), int(
            np.asarray(action).reshape(-1)[1]
        )
        strategy_name = STRATEGIES[strategy_index]
        trial = self.counterfactual_trials[self.step_idx]
        instance_id = int(trial["Instance Id"])
        with_xai = int(trial["Tested w/ XAI"])
        self.shown_xai_type = self.xai_schedule[self.step_idx]

        instances, _ = self.loader.load_instances([instance_id], normalize=False)
        norm_instances, _ = self.loader.load_instances([instance_id], normalize=True)
        instance = np.asarray(instances[0], dtype=float)
        norm_instance = np.asarray(norm_instances[0], dtype=float)
        actual_label = int(self.predict_fn(instance))

        condition = XAI_TYPES[self.condition_index]
        if not strategy_is_legal(strategy_name, condition, with_xai):
            self.step_idx += 1
            truncated = self.step_idx >= self.instances_per_episode
            info = {
                "strategy": strategy_name, "depth": None, "instance_id": instance_id,
                "with_xai": with_xai, "feature_changed": None, "delta": 0.0,
                "ai_prediction_original": actual_label, "ai_prediction_counterfactual": None,
                "success": 0, "time": 0.0, "invalid_under_condition": True,
                "condition": condition, "shown_xai_type": self.shown_xai_type,
            }
            return self.build_observation(), -1.0, False, truncated, info

        rng = np.random.default_rng()
        rate = self.current_cognitive_params.get("random_response_rate", 0.1)
        access_mode = "read" if with_xai == 1 else "retrieve"

        if strategy_name == "change_path_dt":
            feature, delta, time_taken = change_dt_path(
                instance, self.dt_memory, self.dt_exp, self.bounds, actual_label,
                depth=depth, explanation_access_mode=access_mode,
                random_response_rate=rate, rng=rng)
        elif strategy_name == "zero_out_lr_heuristic":
            feature, delta, time_taken = zero_out_lr_heuristic(
                norm_instance, self.lr_memory, self.lr_exp, self.bounds, actual_label,
                random_response_rate=rate, rng=rng)
        elif strategy_name == "zero_out_lr_displayed":
            feature, delta, time_taken = zero_out_lr_displayed(
                instance, self.lr_memory, self.lr_exp, self.bounds, actual_label,
                random_response_rate=rate, rng=rng)
        elif strategy_name == "recall_change_dt":
            feature, delta, time_taken = recall_change_dt_full(
                instance, self.dt_memory, self.bounds, random_response_rate=rate, rng=rng)
        else:
            feature, delta, time_taken = recall_change_lr_full(
                instance, self.lr_memory, self.bounds, actual_label,
                random_response_rate=rate, rng=rng)

        if feature is None:
            # Nothing retrievable yet -- the participant fails to name a change.
            changed_instance = instance
            new_label = actual_label
            success = 0
        else:
            changed_instance = apply_change_to_feature(
                instance, feature, self.bounds, delta,
                self.current_cognitive_params.get("counterfactual_overshoot_fraction", 0.1),
            )
            new_label = int(self.predict_fn(changed_instance))
            success = int(new_label != actual_label)

        self.counts[strategy_index] += 1
        count = self.counts[strategy_index]
        self.success_rates[strategy_index] = (
            self.success_rates[strategy_index] * (count - 1) + success
        ) / count
        self.mean_times[strategy_index] = (
            self.mean_times[strategy_index] * (count - 1) + time_taken
        ) / count

        self.step_idx += 1
        truncated = self.step_idx >= self.instances_per_episode
        reward = float(success) - float(time_taken) * float(self.curr_chi)
        info = {
            "strategy": strategy_name,
            "depth": depth if strategy_name == "change_path_dt" else None,
            "instance_id": instance_id,
            "with_xai": with_xai,
            "feature_changed": feature,
            "delta": float(delta),
            "ai_prediction_original": actual_label,
            "ai_prediction_counterfactual": new_label,
            "success": int(success),
            "time": float(time_taken),
            "invalid_under_condition": False,
            "condition": condition,
            "shown_xai_type": self.shown_xai_type,
        }
        return self.build_observation(), reward, False, truncated, info


__all__ = [
    "DEFAULT_COGNITIVE_PARAMS",
    "DEFAULT_TIME_PENALTY_WEIGHT_RANGE",
    "STRATEGIES",
    "XAI_TYPES",
    "CounterfactualPolicyEnv",
    "apply_change_to_feature",
    "change_dt_path",
    "make_memory",
    "prepare_memory_for_dt",
    "prepare_memory_for_lr_heuristic",
    "recall_change_dt_full",
    "recall_change_lr_full",
    "sample_from_probs",
    "smooth_probs_with_random_response",
    "strategy_is_legal",
    "zero_out_lr_displayed",
    "zero_out_lr_heuristic",
]
